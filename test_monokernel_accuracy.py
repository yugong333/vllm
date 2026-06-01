# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MoE Monokernel Accuracy Test — model-parameterized.

Compares CUDA monokernel intermediates against:
  - Python reference (dequantized fp32 matmuls)
  - Triton fused_experts (vLLM's standard FP8 path)

Debug mode: reads temp_bf16 (SiLU output) directly from the scratchpad
after the kernel runs — no printf in CUDA needed.  BS8 is always the
WGMMA path now: the up-proj epilogue fuses fp8 quantization and writes
to temp_fp8 + temp_act_scale instead of writing bf16 to temp_bf16, so
the Section 2 SiLU comparison is skipped for BS8.

Scratchpad layout (MoEGemmSpec<Dims_BSx>):
  activations[BS][K]                        fp8
  temp_bf16[TEMP_ROWS][N]                   bf16  (BS64 path)
  temp_block_max[TEMP_ROWS][UP_PROJ_BLOCKS] fp32
  temp_fp8[TEMP_ROWS][N]                    fp8   (BS8 WGMMA)
  temp_act_scale[TEMP_ROWS][N / 64]         fp32  (BS8 WGMMA)
  down_partial_out[DOWN_GROUPS][BS][K]      fp32  (BS8 WGMMA)
  act_scale[BS][K/128]                      fp32  (per-token block-wise)

Usage:
    python test_monokernel_accuracy.py                      # defaults to 'coder'
    python test_monokernel_accuracy.py --model coder        # Qwen3-Coder-30B-A3B
    python test_monokernel_accuracy.py --model qwen3.5      # (when available)
    python test_monokernel_accuracy.py --model coder --no-perf
"""

import argparse
import functools
import os
import sys

# Put the directory that *contains* the `vllm` package on sys.path so the
# local source is importable no matter where this script is run from.
# Two layouts are supported:
#   * script at repo root  (~/vllm_main_pr/test_...py)      -> pkg in ./vllm
#   * script inside vllm/  (~/vllm_main_pr/vllm/test_...py) -> pkg in .
# IMPORTANT: insert the PARENT of the `vllm` package, never the package dir
# itself. Inserting `.../vllm/vllm` would put the package's submodules
# (e.g. `vllm/vllm/tokenizers`) on the top-level path and shadow the real
# `tokenizers` wheel, breaking transformers with a circular import.
_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (_here, os.path.join(_here, "vllm")):
    if os.path.isfile(os.path.join(_cand, "vllm", "__init__.py")):
        sys.path.insert(0, _cand)
        break

import torch
import torch.nn.functional as F
import vllm._moe_C  # noqa

import vllm._custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _get_config_dtype_str,
    _get_config_quant_dtype,
    dispatch_fused_moe_kernel,
    moe_align_block_size,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.triton_utils import tl

# ── Model registry ──────────────────────────────────────────────────────────
# Each entry maps a CLI name → dims + the torch op that dispatches to the
# correct monokernel instantiation. Add a new entry here to support a new
# model shape (the matching BSx_Ey wrapper must exist in moe_wrapper.cu).
#
# An op name set to None means the kernel is not wired up in the current
# build. Running against that model will raise a clear RuntimeError
# pointing at moe_wrapper.cu / torch_bindings.cpp.
MODELS = {
    "qwen3.5": {
        "display_name": "Qwen3.5-35B block-wise FP8",
        "E": 256,
        "N_HALF": 512,  # moe_intermediate_size; fused gate+up is 2*N_HALF
        "K": 2048,
        "default_top_k": 8,
        # BS8 TMA+WGMMA path.  V2 (Pair_Layout) is the ONLY BS8 op now
        # (V1 Stripe_Layout was removed in the "full commit" cleanup), so
        # `op_bs8` points at the canonical WGMMA_TMA op.  It requires the
        # gate/up PAIR interleave (`interleave_for_tma_wgmma_up_v2`).
        "op_bs8": "moe_monokernel_topk_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA",
        # Hopper cluster + DSHM + multicast-TMA variant of the BS8 path
        # (see `.kiro/specs/monokernel-cluster-multicast`). Selected via
        # the `--cluster` CLI flag; only meaningful on sm_90a.
        "op_bs8_cluster": "moe_monokernel_topk_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA_Cluster",
        "op_bs64": "moe_monokernel_topk_BS64_E256_Qwen3_5_35B_BlockFP8",
    },
    "coder": {
        "display_name": "Qwen3-Coder-30B-A3B",
        "E": 128,
        "N_HALF": 768,  # moe_intermediate_size; fused gate+up is 2*N_HALF
        "K": 2048,
        "default_top_k": 8,
        # Not currently wired in moe_monokernel/moe_wrapper.cu — re-add the
        # Qwen3Coder wrappers + torch bindings to enable.
        "op_bs8": None,
        "op_bs64": None,
    },
}

DEV = "cuda"


def get_model_op(model_cfg, M, use_cluster=False):
    """Return the high-level monokernel dispatch op, or raise if the model's
    low-level kernel is not wired in the current build.

    For M <= 8 the BS8 path is ALWAYS TMA+WGMMA+Pair_Layout (the single
    BS8 design): a local wrapper is returned that calls the BS8
    low-level op directly with gate/up PAIR-interleaved up-projection
    weights.  `use_cluster=True` selects the Hopper cluster variant
    (`__cluster_dims__(8, 1, 1)` + multicast TMA + DSHM) instead of the
    per-block variant; both share the same Python call signature and
    the same pre-interleaved weights.

    The BS8 path uses SWIZZLE_128B on both weight descriptors; the
    up-projection weight tensor is repacked via
    `interleave_weights.interleave_for_tma_wgmma_up_v2` (gate/up PAIR
    interleave so one 128x128 TMA fetches a full WGMMA A-tile), and the
    down-projection weight tensor is passed RAW — the TMA hardware
    applies the core-matrix XOR swizzle at write time.  The up-proj
    repack caches on the weight tensor's `_tma_interleaved_up_v2`
    attribute, so repeated calls with the same weights are free.
    """
    if M <= 8 and use_cluster:
        key = "op_bs8_cluster"
    else:
        key = "op_bs8" if M <= 8 else "op_bs64"
    op_name = model_cfg.get(key)
    if op_name is None:
        raise RuntimeError(
            f"Model '{model_cfg['display_name']}' has no registered kernel "
            f"for {key}. Register the wrapper in moe_wrapper.cu and add the "
            f"binding in torch_bindings.cpp first."
        )
    if M <= 8:
        MOE_SCORING_SOFTMAX = 1
        MOE_SCORING_SIGMOID = 0

        # Lazy import so the module is only required when the BS8 path
        # actually runs.  V2 (Pair_Layout) is the only BS8 design, so
        # all BS8 ops (default + cluster) use the V2 gate/up PAIR
        # interleave.  Import from the installed `vllm` package (importable
        # from any CWD) rather than the repo-root `interleave_weights.py`
        # shim, which is only on sys.path when run from the repo root.
        from vllm.model_executor.layers.fused_moe.moe_monokernel_interleave import (
            interleave_for_tma_wgmma_up_v2,
        )

        low_level_op = getattr(torch.ops._moe_C, op_name)

        def _tma_op(
            activations_in,
            router_logits,
            expert_weights_up,
            expert_scales_up,
            expert_weights_down,
            expert_scales_down,
            scratchpad,
            top_k=1,
            scoring_func="softmax",
            renormalize=True,
        ):
            # Up-projection repack into the gate/up PAIR-interleaved
            # layout (8-row gate + 8-row up pairs per warp) so
            # silu(gate)*up is a per-lane register op after the WGMMA.
            # Cached on the tensor for repeat invocations.  This is the
            # only BS8 layout now (V1 Stripe_Layout was removed).
            cache_attr = "_tma_interleaved_up_v2"
            cached_up = getattr(expert_weights_up, cache_attr, None)
            if cached_up is None:
                cached_up = interleave_for_tma_wgmma_up_v2(
                    expert_weights_up
                ).contiguous()
                try:
                    setattr(expert_weights_up, cache_attr, cached_up)
                except (AttributeError, RuntimeError):
                    pass

            activations_out = torch.zeros_like(activations_in)
            sf = (
                MOE_SCORING_SOFTMAX
                if scoring_func == "softmax"
                else MOE_SCORING_SIGMOID
            )
            low_level_op(
                activations_in,
                router_logits,
                cached_up,
                expert_scales_up,
                expert_weights_down,
                expert_scales_down,
                activations_out,
                scratchpad,
                top_k,
                sf,
                renormalize,
            )
            return activations_out

        return _tma_op

    # The high-level op (torch.ops.vllm.moe_monokernel_topk) auto-dispatches
    # to the correct BS8/BS64 _moe_C symbol based on M. It also allocates
    # the output tensor and validates shapes.
    return torch.ops.vllm.moe_monokernel_topk


# ── Helpers ──────────────────────────────────────────────────────────────────


def quant_fp8_per_row(w):
    """Per-channel (per-row) FP8 quantization — kept for reference."""
    amax = w.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    s = amax / 448.0
    return (w.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn), s.float()


def quant_fp8_block_wise(w, block_row=128, block_col=128):
    """Block-wise (block_row × block_col) FP8 quantization.

    Args:
        w: float tensor of shape [E, rows, cols]
        block_row, block_col: block dimensions for scale granularity

    Returns:
        w_fp8: quantized tensor [E, rows, cols] in float8_e4m3fn
        scales: float tensor [E, ceil(rows/block_row), ceil(cols/block_col)]
    """
    E, rows, cols = w.shape
    rb = (rows + block_row - 1) // block_row
    cb = (cols + block_col - 1) // block_col
    wf = w.float()
    scales = torch.zeros(E, rb, cb, device=w.device, dtype=torch.float32)
    w_fp8 = torch.zeros_like(wf)

    for e in range(E):
        for ri in range(rb):
            r0 = ri * block_row
            r1 = min(r0 + block_row, rows)
            for ci in range(cb):
                c0 = ci * block_col
                c1 = min(c0 + block_col, cols)
                block = wf[e, r0:r1, c0:c1]
                amax = block.abs().max().clamp(min=1e-12)
                s = amax / 448.0
                scales[e, ri, ci] = s
                w_fp8[e, r0:r1, c0:c1] = (block / s).clamp(-448, 448)

    return w_fp8.to(torch.float8_e4m3fn), scales


def cos_sim(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    if a.norm() < 1e-8 or b.norm() < 1e-8:
        return 0.0
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def err_stats(a, b):
    d = (a.float() - b.float()).abs().reshape(-1)
    return d.max().item(), d.mean().item()


def routing_softmax_topk(logits_bf16, top_k):
    """Greedy softmax → topk → renormalize (matches CUDA kernel)."""
    scores = torch.softmax(logits_bf16.float(), dim=-1)
    M = scores.shape[0]
    ids = torch.zeros(M, top_k, dtype=torch.int64, device=DEV)
    wts = torch.zeros(M, top_k, dtype=torch.float32, device=DEV)
    s = scores.clone()
    for k in range(top_k):
        v, idx = s.max(dim=-1)
        wts[:, k] = v
        ids[:, k] = idx
        s.scatter_(1, idx.unsqueeze(1), float("-inf"))
    wts = wts / wts.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return wts, ids


def quant_fp8_activation_block_wise(x_float, group_size=128):
    """Block-wise activation quantization matching the monokernel's
    per-token-dynamic-1x128 scheme.

    Quantizes each group of `group_size` elements independently.

    Args:
        x_float: [K] float tensor (one token's activations)
        group_size: number of elements per quantization group

    Returns:
        x_fp8: [K] fp8 quantized tensor
        scales: [K // group_size] float scales, one per group
    """
    K = x_float.shape[0]
    assert K % group_size == 0
    n_groups = K // group_size
    x_fp8 = torch.zeros(K, device=x_float.device, dtype=torch.float8_e4m3fn)
    scales = torch.zeros(n_groups, device=x_float.device, dtype=torch.float32)

    for g in range(n_groups):
        g0 = g * group_size
        g1 = g0 + group_size
        block = x_float[g0:g1]
        amax = block.abs().max().clamp(min=1e-12)
        s = (amax / 448.0).item()
        scales[g] = s
        inv_s = 448.0 / amax.item()
        x_fp8[g0:g1] = (block * inv_s).clamp(-448, 448).to(torch.float8_e4m3fn)

    return x_fp8, scales


# ── Read temp_bf16 from scratchpad ───────────────────────────────────────────


def read_temp_bf16_from_scratchpad(scratchpad_bytes, M, top_k, is_bs8, N_HALF, K):
    """
    Read the SiLU output (temp_bf16) from the monokernel's scratchpad.

    Layout of MoEGemmSpec (no DEBUG_MOE):
      activations: [BS][K] fp8  → BS*K bytes
      temp_bf16:   [TEMP_ROWS * N] bf16
      temp_block_max: [TEMP_ROWS * UP_PROJ_BLOCKS] fp32
      act_scale:   [BS][K/128] fp32  (per-token block-wise, 16 scales per token)
    """
    BS = 8 if is_bs8 else 64
    TEMP_ROWS = BS * 8 + 8  # SPEC_MAX_TOPK=8
    act_bytes = BS * K  # fp8 = 1 byte each
    temp_offset = act_bytes
    temp_count = TEMP_ROWS * N_HALF
    temp_bytes = temp_count * 2  # bf16 = 2 bytes

    raw = scratchpad_bytes[temp_offset : temp_offset + temp_bytes]
    temp = raw.view(torch.bfloat16).reshape(TEMP_ROWS, N_HALF)

    # Only return the rows that were actually written: M * top_k rows
    return temp[: M * top_k].clone()


# ── Block-wise GEMM helper ───────────────────────────────────────────────────


def _block_wise_gemm(w_fp8, scales_bw, x_fp8, x_scales, block_row=128, block_col=128):
    """Python reference: block-wise dequant GEMM with block-wise activation scales.

    Computes: result[row] = sum over col-blocks of:
        (sum_{col in block} W_fp8[row][col] * x_fp8[col])
        * w_scale[row_block][col_block] * x_scale[col_block]

    Args:
        w_fp8:      [rows, cols] fp8 weight matrix
        scales_bw:  [ceil(rows/block_row), ceil(cols/block_col)] weight scales
        x_fp8:      [cols] fp8 activation vector
        x_scales:   [ceil(cols/block_col)] activation scales (one per group)

    Returns:
        [rows] float32 result
    """
    rows, cols = w_fp8.shape
    wf = w_fp8.float()
    xf = x_fp8.float()
    rb, cb = scales_bw.shape[0], scales_bw.shape[1]
    result = torch.zeros(rows, device=w_fp8.device, dtype=torch.float32)
    for ri in range(rb):
        r0 = ri * block_row
        r1 = min(r0 + block_row, rows)
        for ci in range(cb):
            c0 = ci * block_col
            c1 = min(c0 + block_col, cols)
            partial = wf[r0:r1, c0:c1] @ xf[c0:c1]
            result[r0:r1] += partial * scales_bw[ri, ci] * x_scales[ci]
    return result


# ── Python reference ─────────────────────────────────────────────────────────


def python_reference(x, w13_fp8, s13, w2_fp8, s2, topk_w, topk_ids, N_HALF, K):
    """
    Python reference using block-wise scales for both weights and activations
    — same math as the CUDA kernel.
    Returns: final_out, gate_up_all, silu_all
    """
    M, top_k = x.shape[0], topk_ids.shape[1]
    out = torch.zeros(M, K, device=DEV, dtype=torch.bfloat16)
    gate_up_all = torch.zeros(M * top_k, 2 * N_HALF, device=DEV)
    silu_all = torch.zeros(M * top_k, N_HALF, device=DEV, dtype=torch.bfloat16)

    for tok in range(M):
        xf = x[tok].float()
        # Block-wise activation quantization (groups of 128 along K)
        xq, x_scales = quant_fp8_activation_block_wise(xf, group_size=128)

        for ki in range(top_k):
            eid = topk_ids[tok, ki].item()
            rw = topk_w[tok, ki].item()
            vrow = tok * top_k + ki

            # GEMM1 — block-wise weight + activation scales
            raw = _block_wise_gemm(w13_fp8[eid], s13[eid], xq, x_scales)
            gate = raw[:N_HALF]
            up = raw[N_HALF:]
            gate_up_all[vrow, :N_HALF] = gate
            gate_up_all[vrow, N_HALF:] = up

            # SiLU * rw
            silu = rw * (up * gate) / (1.0 + torch.exp(-gate))
            silu_bf16 = silu.bfloat16()
            silu_all[vrow] = silu_bf16

            # GEMM2 — block-wise quantize intermediate, then block-wise GEMM
            sf = silu_bf16.float()
            N_half = sf.shape[0]
            if N_half % 128 == 0:
                sq, s2_act = quant_fp8_activation_block_wise(sf, group_size=128)
            else:
                # Fallback: per-token quantization for non-aligned intermediate
                am = sf.abs().max().clamp(min=1e-12)
                inv_s = 448.0 / am.item()
                sq = (sf * inv_s).clamp(-448, 448).to(torch.float8_e4m3fn)
                n_groups = (N_half + 127) // 128
                s2_act = torch.full(
                    (n_groups,),
                    (am / 448.0).item(),
                    device=sf.device,
                    dtype=torch.float32,
                )

            down = _block_wise_gemm(w2_fp8[eid], s2[eid], sq, s2_act)
            out[tok] += down.bfloat16()

    return out, gate_up_all, silu_all


# ── Triton with intermediates ────────────────────────────────────────────────


def triton_with_intermediates(
    x, topk_w, topk_ids, w13_fp8, s13, w2_fp8, s2, N_HALF, K, E
):
    """Run Triton step-by-step with block-wise scales, return intermediates."""
    M, top_k = x.shape[0], topk_ids.shape[1]
    N = 2 * N_HALF
    block_shape = [128, 128]

    cd = _get_config_dtype_str(
        use_fp8_w8a8=True,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        ocp_mx_scheme=None,
        dtype=x.dtype,
    )
    qd = _get_config_quant_dtype(
        use_fp8_w8a8=True, use_int8_w8a8=False, ocp_mx_scheme=None
    )
    cfg = functools.partial(
        try_get_optimal_moe_config,
        w13_fp8.size(),
        w2_fp8.size(),
        top_k,
        cd,
        block_shape=block_shape,
    )(M)

    c1 = torch.empty(M, top_k, N, device=DEV, dtype=x.dtype)
    ad = mk.FusedMoEExpertsModular.adjust_N_for_activation(N, MoEActivation.SILU)
    c2 = torch.empty(M * top_k, ad, device=DEV, dtype=x.dtype)
    c3 = torch.empty(M, top_k, K, device=DEV, dtype=x.dtype)

    qx, xs = moe_kernel_quantize_input(
        A=x,
        A_scale=None,
        quant_dtype=qd,
        per_act_token_quant=False,
        block_shape=block_shape,
    )
    si, ei, ntp = moe_align_block_size(
        topk_ids.int(), cfg["BLOCK_SIZE_M"], E, None, ignore_invalid_experts=True
    )

    # GEMM1
    dispatch_fused_moe_kernel(
        qx,
        w13_fp8,
        c1,
        xs,
        s13,
        None,
        topk_w,
        si,
        ei,
        ntp,
        False,
        top_k,
        cfg,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=block_shape,
    )
    gemm1 = c1.clone()

    # SiLU
    apply_moe_activation(MoEActivation.SILU, c2, c1.view(-1, N))
    silu = c2.clone()

    # Requant
    qc2, a2s = moe_kernel_quantize_input(
        A=c2,
        A_scale=None,
        quant_dtype=qd,
        per_act_token_quant=False,
        block_shape=block_shape,
    )

    # GEMM2
    dispatch_fused_moe_kernel(
        qc2,
        w2_fp8,
        c3,
        a2s,
        s2,
        None,
        topk_w,
        si,
        ei,
        ntp,
        True,
        1,
        cfg,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=block_shape,
    )
    gemm2 = c3.clone()

    final = torch.empty_like(x)
    ops.moe_sum(c3.view(M, top_k, K), final)

    return final, gemm1, silu, gemm2


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: ACCURACY TEST
# ══════════════════════════════════════════════════════════════════════════════


def accuracy_test(model_cfg, M, top_k, seed=42, use_cluster=False):
    torch.manual_seed(seed)
    E = model_cfg["E"]
    N_HALF = model_cfg["N_HALF"]
    K = model_cfg["K"]
    kernel_op = get_model_op(model_cfg, M, use_cluster=use_cluster)

    is_bs8 = M <= 8
    path = "BS8" if is_bs8 else "BS64"
    sep = "=" * 72

    print(f"\n{'#' * 72}")
    print(f"# ACCURACY [{model_cfg['display_name']}]: {path} M={M} top_k={top_k}")
    print(f"{'#' * 72}")

    # Setup — block-wise quantization for all paths
    w13_f = torch.randn(E, 2 * N_HALF, K, device=DEV) * 0.1
    w2_f = torch.randn(E, K, N_HALF, device=DEV) * 0.1
    w13_fp8, s13 = quant_fp8_block_wise(w13_f)
    w2_fp8, s2 = quant_fp8_block_wise(w2_f)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    logits = torch.randn(M, E, device=DEV, dtype=torch.bfloat16)
    topk_w, topk_ids = routing_softmax_topk(logits, top_k)

    # Allocate scratchpad as raw bytes so we can read back temp_bf16
    BS = 8 if is_bs8 else 64
    TEMP_ROWS = BS * 8 + 8
    UP_PROJ_BLOCKS = (N_HALF + 7) // 8
    ACT_SCALE_BLOCKS = (K + 127) // 128  # per-token block-wise: K/128 scales
    # WGMMA path adds three new scratchpad buffers sized to match the
    # MoEGemmSpec<Dims> layout in moe_internal.h:
    #   temp_fp8         [TEMP_ROWS][N_HALF]           fp8
    #   temp_act_scale   [TEMP_ROWS][N_HALF / 64]      fp32
    #   down_partial_out [DOWN_GROUPS][BS][K]          fp32
    # GRID_SIZE is 128 for the WGMMA BS8 kernel.
    #
    # DOWN_COL_TILE is variant-dependent after Phase 2a of the
    # software-grid-sync spec:
    #   * BS8 TMA+WGMMA:   DOWN_COL_TILE = 256 → DOWN_GRID = K/256 = 8,
    #                      DOWN_GROUPS = GRID_SIZE / 8 = 16 (== UP_GROUPS).
    #   * BS64 / non-TMA:  DOWN_COL_TILE = 128 → DOWN_GRID = K/128 = 16,
    #                      DOWN_GROUPS = GRID_SIZE / 16 = 8.
    # Mirrors MoECoreDims<Dims>::DOWN_COL_TILE / DOWN_GRID / DOWN_GROUPS
    # so the host-side scratchpad buffer matches the kernel's
    # `down_partial_out[DOWN_GROUPS][BS][HIDDEN_STATES]` footprint.
    GRID_SIZE = 128
    DOWN_COL_TILE = 256 if is_bs8 else 128
    DOWN_GRID = K // DOWN_COL_TILE
    DOWN_GROUPS = GRID_SIZE // DOWN_GRID if DOWN_GRID > 0 else 1
    DOWN_ACT_BLOCK_SIZE = 64
    TEMP_ACT_SCALE_COLS = N_HALF // DOWN_ACT_BLOCK_SIZE
    spec_size = (
        BS * K  # activations fp8
        + TEMP_ROWS * N_HALF * 2  # temp_bf16
        + TEMP_ROWS * UP_PROJ_BLOCKS * 4  # temp_block_max
        + TEMP_ROWS * N_HALF * 1  # temp_fp8 (WGMMA path)
        + TEMP_ROWS * TEMP_ACT_SCALE_COLS * 4  # temp_act_scale (WGMMA)
        + DOWN_GROUPS * BS * K * 4  # down_partial_out (WGMMA)
        + BS * ACT_SCALE_BLOCKS * 4
    )  # act_scale [BS][K/128]
    # Round up and add margin
    scratch_floats = (spec_size + 3) // 4 + 4096
    scratchpad = torch.zeros(scratch_floats, dtype=torch.float32, device=DEV)

    # ── Run CUDA monokernel ──────────────────────────────────────────────
    # High-level op: allocates activations_out internally and dispatches
    # to BS8/BS64 based on M. See vllm/_custom_ops.py moe_monokernel_topk.
    cuda_out = kernel_op(
        x,
        logits,
        w13_fp8.contiguous(),
        s13.contiguous(),
        w2_fp8.contiguous(),
        s2.contiguous(),
        scratchpad,
        top_k=top_k,
        scoring_func="softmax",
        renormalize=True,
    )
    torch.accelerator.synchronize

    # ── WGMMA reference dump ─────────────────────────────────────────────
    # The kernel's WGMMA_INPUT_DEBUG printf dumps the A (weight) and B
    # (activation) tiles for block 0, expert 0 (= shmem->experts[0], the
    # SMALLEST active expert id in this batch), K-step 0.  We compute the
    # same reference here (without the WGMMA descriptor path) so we can
    # cross-check.
    #
    # BS8 is always WGMMA now, so we always produce this dump for BS8.
    if is_bs8:
        # Active experts in this batch, sorted ascending by id (matches
        # the kernel's shmem->experts layout).
        active_eids = sorted(set(topk_ids.flatten().tolist()))
        expert_0_id = active_eids[0]
        print(f"\n[PY WGMMA REF] active_experts (ascending): {active_eids}")
        print(f"[PY WGMMA REF] shmem->experts[0] should be id={expert_0_id}")

        # Dequantize the first K=32 slab of expert_0's weights (first 64
        # rows: 32 gate rows + 32 up rows at offset N_HALF).
        w_fp8_slab = w13_fp8[expert_0_id]  # [2*N_HALF, K]
        s13_slab = s13[expert_0_id]  # [up_rows, up_cols]
        # Gate rows [0..31] × K [0..31]
        w_gate_fp8 = w_fp8_slab[0:32, 0:32].float()
        # Up rows [N_HALF..N_HALF+31] × K [0..31]
        w_up_fp8 = w_fp8_slab[N_HALF : N_HALF + 32, 0:32].float()
        # NOTE: weights are block-wise quantized.  For the pure m64n8k32
        # WGMMA (no scales applied), the fp32 output is just the fp8 dot
        # product.  Scales are applied per-128-K-block by the kernel AFTER
        # the 4 WGMMAs in that block.  So for s=0 (the first K=32), the
        # expected chunk_d is simply the fp8 dot product.  Print the
        # fp8 input slabs for comparison against the kernel's [WGMMA IN]
        # dump.
        print("[PY WGMMA REF] A (weight 64x32 fp8, row-major):")
        for r in range(4):
            print(
                f"  A row {r}: "
                + " ".join(f"{v:.2f}" for v in w_gate_fp8[r, :16].tolist())
            )
        print("  A row 32: " + " ".join(f"{v:.2f}" for v in w_up_fp8[0, :16].tolist()))

        # Activations: quantize x for tokens 0..7 to fp8 (block-wise 128).
        # Then take K=0..31 (first 32 of first 128-K block).
        # Canonical WGMMA K-major B layout: per token, print 16 consecutive
        # K-values (one core-matrix row).
        print("[PY WGMMA REF] B (act 32x8 fp8, canonical K-major = N-outer, K-inner):")
        xq_tokens = []
        for tok in range(M):
            xq, _ = quant_fp8_activation_block_wise(x[tok].float(), group_size=128)
            xq_tokens.append(xq[:32].float())  # first 32 K values
        # Print per token: K[0..15] (first core matrix of B).  This matches
        # the kernel's [WGMMA IN] B dump.
        for tok in range(min(M, 4)):
            print(
                f"  B tok={tok} K[0..15]: "
                + " ".join(f"{v:.2f}" for v in xq_tokens[tok][:16].tolist())
            )

        # Expected chunk_d for first 128-K block (s=0..3 accumulated):
        # Sum of 4 WGMMA outputs over K=0..127.  For each (row, tok), it's
        # the fp32 dot product of the fp8 slabs over K=[0,128).
        print(
            "[PY WGMMA REF] expected chunk_d after first 128-K block "
            "(WG0 m64n8k32 × 4, rows [0..3] × tokens [0..7]):"
        )
        w_gate_128 = w_fp8_slab[0:32, 0:128].float()  # 32 × 128
        w_up_128 = w_fp8_slab[N_HALF : N_HALF + 32, 0:128].float()  # 32 × 128
        x_128 = torch.stack(
            [
                quant_fp8_activation_block_wise(x[tok].float(), group_size=128)[0][
                    :128
                ].float()
                for tok in range(M)
            ],
            dim=0,
        )  # [M, 128]
        # Pad tokens to 8
        if M < 8:
            x_128 = torch.cat(
                [x_128, torch.zeros(8 - M, 128, device=x_128.device)], dim=0
            )
        # D_gate[row, tok] = sum_k W_gate[row, k] * x[tok, k]
        D_gate = w_gate_128 @ x_128.T  # [32, 8]
        D_up = w_up_128 @ x_128.T  # [32, 8]
        print(
            "  D (gate row 0, all 8 tokens): "
            + " ".join(f"{v:.4e}" for v in D_gate[0].tolist())
        )
        print(
            "  D (gate row 1, all 8 tokens): "
            + " ".join(f"{v:.4e}" for v in D_gate[1].tolist())
        )
        # Rows 8, 9 correspond to WGMMA lane d2/d3 for warp 0 (row = lane/4 + 8).
        print(
            "  D (gate row 8, all 8 tokens): "
            + " ".join(f"{v:.4e}" for v in D_gate[8].tolist())
        )
        print(
            "  D (gate row 9, all 8 tokens): "
            + " ".join(f"{v:.4e}" for v in D_gate[9].tolist())
        )
        print(
            "  D (up row 0 [= D row 32], all 8 tokens): "
            + " ".join(f"{v:.4e}" for v in D_up[0].tolist())
        )
        print(
            "  D (up row 1 [= D row 33], all 8 tokens): "
            + " ".join(f"{v:.4e}" for v in D_up[1].tolist())
        )
        print()

    # Read temp_bf16 (SiLU output) from scratchpad.  The BS8 (WGMMA) path's
    # up-proj epilogue fuses quantization: it writes fp8 to temp_fp8 +
    # per-64-col scales to temp_act_scale INSTEAD OF writing bf16 to
    # temp_bf16.  So for BS8, temp_bf16 is stale and we skip the
    # Section 2 SiLU comparison below.
    scratch_bytes = scratchpad.view(torch.uint8)
    if is_bs8:
        cuda_silu = None
    else:
        cuda_silu = read_temp_bf16_from_scratchpad(
            scratch_bytes, M, top_k, is_bs8, N_HALF, K
        )

    # ── Run Python reference ─────────────────────────────────────────────
    py_out, py_gate_up, py_silu = python_reference(
        x, w13_fp8, s13, w2_fp8, s2, topk_w, topk_ids, N_HALF, K
    )

    # ── Run Triton ───────────────────────────────────────────────────────
    tri_out, tri_gemm1, tri_silu, tri_gemm2 = triton_with_intermediates(
        x, topk_w, topk_ids, w13_fp8, s13, w2_fp8, s2, N_HALF, K, E
    )
    torch.accelerator.synchronize

    # ── 0. Routing ───────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("0. ROUTING")
    print(sep)
    for tok in range(min(M, 2)):
        print(
            f"  tok={tok}: experts={topk_ids[tok].tolist()}  "
            f"weights=[{', '.join(f'{w:.4f}' for w in topk_w[tok].tolist())}]"
        )

    # ── 1. Up projection (gate + up) ────────────────────────────────────
    print(f"\n{sep}")
    print("1. UP PROJECTION (gate + up, token 0)")
    print(sep)
    for ki in range(min(top_k, 2)):
        eid = topk_ids[0, ki].item()
        vrow = 0 * top_k + ki
        tri_g = tri_gemm1[0, ki, :N_HALF]
        tri_u = tri_gemm1[0, ki, N_HALF:]
        py_g = py_gate_up[vrow, :N_HALF].bfloat16()
        py_u = py_gate_up[vrow, N_HALF:].bfloat16()
        gc = cos_sim(tri_g, py_g)
        uc = cos_sim(tri_u, py_u)
        print(f"  Expert {eid} (k={ki}): gate cos={gc:.6f}  up cos={uc:.6f}")

    # ── 2. SiLU output ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print("2. SiLU OUTPUT (token 0)")
    if is_bs8:
        print(
            "   (skipped — BS8 WGMMA path fuses quant into the epilogue "
            "and no longer writes bf16 to temp_bf16)"
        )
        print(sep)
    else:
        print("   CUDA reads from scratchpad temp_bf16 (rw baked in)")
        print("   Triton has NO rw in SiLU")
        print(sep)
        for ki in range(min(top_k, 2)):
            eid = topk_ids[0, ki].item()
            rw = topk_w[0, ki].item()
            vrow = 0 * top_k + ki
            cs = cuda_silu[vrow]
            ps = py_silu[vrow]
            ts = tri_silu[vrow]
            cp = cos_sim(cs, ps)
            ct = cos_sim(cs, ts)
            print(f"  Expert {eid} (k={ki}, rw={rw:.4f}):")
            print(f"    CUDA vs Py-mono: cos={cp:.6f}  (both have rw)")
            print(f"    CUDA vs Triton:  cos={ct:.6f}  (Triton has no rw)")
            print(
                f"    CUDA norm={cs.float().norm():.2f}  "
                f"Py norm={ps.float().norm():.2f}  "
                f"Tri norm={ts.float().norm():.2f}"
            )

    # ── 3. Down projection / final output ────────────────────────────────
    print(f"\n{sep}")
    print("3. FINAL OUTPUT")
    print(sep)
    ct = cos_sim(cuda_out, tri_out)
    cp = cos_sim(cuda_out, py_out)
    tp = cos_sim(tri_out, py_out)
    ct_m, ct_mn = err_stats(cuda_out, tri_out)
    cp_m, cp_mn = err_stats(cuda_out, py_out)

    for tok in range(min(M, 2)):
        print(f"  Token {tok} (first 4):")
        print(f"    CUDA:   {cuda_out[tok, :4].float().tolist()}")
        print(f"    Triton: {tri_out[tok, :4].float().tolist()}")
        print(f"    Py:     {py_out[tok, :4].float().tolist()}")

    print(f"\n  All {M} tokens:")
    print(
        f"    CUDA vs Triton:  cos={ct:.6f}  max_err={ct_m:.2f}  mean_err={ct_mn:.2f}"
    )
    print(
        f"    CUDA vs Py-mono: cos={cp:.6f}  max_err={cp_m:.2f}  mean_err={cp_mn:.2f}"
    )
    print(f"    Triton vs Py:    cos={tp:.6f}")
    print(
        f"    Norms: CUDA={cuda_out.float().norm():.2f}  "
        f"Tri={tri_out.float().norm():.2f}  Py={py_out.float().norm():.2f}"
    )

    ok_route = cp > 0.99
    print(
        f"\n  Routing+computation: {'PASS' if ok_route else 'FAIL'} "
        f"(CUDA vs Py cos={cp:.6f})"
    )
    print(sep)
    return {"cuda_py": cp, "cuda_tri": ct, "tri_py": tp}


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: PERFORMANCE TEST
# ══════════════════════════════════════════════════════════════════════════════


def _triton_e2e(x, logits, w13_fp8, s13, w2_fp8, s2, top_k):
    """End-to-end Triton path with block-wise FP8: routing + fused_experts.

    Uses vLLM's ``fused_topk`` (the same fused softmax+topk+renorm CUDA op
    invoked by real serving models like MiniCPM / Arctic / BERT) so routing
    latency matches what production sees. ``fused_topk`` takes raw router
    logits and returns (topk_weights: fp32, topk_ids: int32,
    token_expert_indices: int32), which is exactly the interface
    ``fused_experts_impl`` consumes.
    """
    from vllm.model_executor.layers.fused_moe import fused_topk
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl

    topk_w, topk_ids, _ = fused_topk(
        hidden_states=x, gating_output=logits, topk=top_k, renormalize=True
    )
    return fused_experts_impl(
        hidden_states=x,
        w1=w13_fp8,
        w2=w2_fp8,
        topk_weights=topk_w,
        topk_ids=topk_ids,
        inplace=False,
        activation="silu",
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        w1_scale=s13,
        w2_scale=s2,
        a1_scale=None,
        a2_scale=None,
        block_shape=[128, 128],
    )


def _bench_cudagraph(fn, warmup=20, iters=200):
    """Benchmark with CUDA graph capture, return latency in ms."""
    # Warmup (also triggers Triton autotuning / JIT)
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    # Capture
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn()  # one more warmup in capture stream
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    torch.accelerator.synchronize()
    # Replay
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / iters


def perf_test(model_cfg, M, top_k, seed=42, use_cluster=False):
    torch.manual_seed(seed)
    E = model_cfg["E"]
    N_HALF = model_cfg["N_HALF"]
    K = model_cfg["K"]
    kernel_op = get_model_op(model_cfg, M, use_cluster=use_cluster)

    w13_f = torch.randn(E, 2 * N_HALF, K, device=DEV) * 0.1
    w2_f = torch.randn(E, K, N_HALF, device=DEV) * 0.1
    w13_fp8, s13 = quant_fp8_block_wise(w13_f)
    w2_fp8, s2 = quant_fp8_block_wise(w2_f)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    logits = torch.randn(M, E, device=DEV, dtype=torch.bfloat16)
    scratchpad = torch.zeros(1024, 4096, dtype=torch.float32, device=DEV)

    # Pre-contiguous to avoid overhead in the loop
    w13c = w13_fp8.contiguous()
    s13c = s13.contiguous()
    w2c = w2_fp8.contiguous()
    s2c = s2.contiguous()

    def cuda_fn():
        return kernel_op(
            x,
            logits,
            w13c,
            s13c,
            w2c,
            s2c,
            scratchpad,
            top_k=top_k,
            scoring_func="softmax",
            renormalize=True,
        )

    def triton_fn():
        return _triton_e2e(x, logits, w13c, s13c, w2c, s2c, top_k)

    try:
        cuda_cg_ms = _bench_cudagraph(cuda_fn)
    except Exception:
        cuda_cg_ms = float("nan")
    try:
        tri_cg_ms = _bench_cudagraph(triton_fn)
    except Exception:
        tri_cg_ms = float("nan")

    return cuda_cg_ms, tri_cg_ms


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def parse_args():
    parser = argparse.ArgumentParser(
        description="MoE Monokernel accuracy + performance test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=sorted(MODELS.keys()),
        default="qwen3.5",
        help="Model shape to test against.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Subset of M values to test (default: 1,4,8,16,32,64).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the registered model configs and exit.",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Use the Hopper cluster + DSHM + multicast-TMA variant of "
        "the BS8 TMA+WGMMA kernel (Qwen3.5-35B BlockFP8 only).  "
        "`__cluster_dims__(8, 1, 1)` co-locates one expert group's "
        "8 blocks on one GPC; activations are loaded once per "
        "cluster via TMA multicast; Phase 3 → Phase 4 hands off "
        "via per-cluster DSHM and a hardware cluster barrier.  "
        "Uses the same SWIZZLE_128B descriptors and pre-interleaved "
        "weights as the default BS8 path.  Requires sm_90a "
        "+ CUDA Toolkit 12.4+.",
    )
    return parser.parse_args()


def print_model_registry():
    print("Registered models:")
    for key, cfg in MODELS.items():
        wired = "ok" if cfg["op_bs8"] and cfg["op_bs64"] else "not wired"
        print(
            f"  {key:<10s} {cfg['display_name']:<32s} "
            f"E={cfg['E']} N_HALF={cfg['N_HALF']} K={cfg['K']}  [{wired}]"
        )


def main():
    args = parse_args()
    if args.list_models:
        print_model_registry()
        return

    model_cfg = MODELS[args.model]
    tk_default = model_cfg["default_top_k"]
    E = model_cfg["E"]
    N_HALF = model_cfg["N_HALF"]
    K = model_cfg["K"]

    if args.cluster:
        variant_tag = "  [WGMMA+TMA+CLUSTER]"
    else:
        variant_tag = "  [WGMMA+TMA]"

    banner = "#" * 72
    print(banner)
    print(
        f"# MoE Monokernel Test — {model_cfg['display_name']} "
        f"(E={E}, N={N_HALF}, K={K})"
    )
    print(f"# model={args.model}  top_k={tk_default}{variant_tag}")
    print(banner)

    # Default batch-size sweep; can be narrowed via --batch-sizes.
    all_Ms = args.batch_sizes if args.batch_sizes else [1, 2, 4, 8, 16, 32, 64]

    # ── Part 1: Accuracy ─────────────────────────────────────────────────
    acc_results = {}
    # (M, top_k, seed) — seeds vary for coverage, but top_k follows CLI.
    seeds = {1: 42, 2: 80, 4: 200, 8: 300, 16: 400, 32: 500, 64: 600}
    # Include the tk=1 smoke case only when M=1 is tested and tk_default > 1
    acc_configs = []
    for M in all_Ms:
        acc_configs.append((M, tk_default, seeds.get(M, 42 + M)))
    for M, tk, seed in acc_configs:
        label = f"M{M}_tk{tk}"
        acc_results[label] = accuracy_test(
            model_cfg, M, tk, seed, use_cluster=args.cluster
        )

    # ── Part 2: Performance (CUDA Graph) ─────────────────────────────────
    print(f"\n\n{banner}")
    print(f"# PERFORMANCE ({model_cfg['display_name']} shape, CUDA graph)")
    print("# Triton includes: routing (softmax+topk+renorm) + align + quant")
    print("#                  + GEMM1 + SiLU + requant + GEMM2 + moe_sum")
    print("# CUDA includes:   all of the above in a single cooperative kernel")
    print(banner)

    perf_configs = [(M, tk_default) for M in all_Ms]
    if 1 in all_Ms and tk_default > 1:
        perf_configs.insert(0, (1, 1))  # tk=1 smoke bench

    print(
        f"\n  {'M':>4s}  {'tk':>3s}  {'Path':>5s}  "
        f"{'CUDA(ms)':>10s}  {'Triton(ms)':>11s}  {'Speedup':>8s}"
    )
    print(f"  {'-' * 4}  {'-' * 3}  {'-' * 5}  {'-' * 10}  {'-' * 11}  {'-' * 8}")
    for M, tk in perf_configs:
        path = "BS8" if M <= 8 else "BS64"
        # Reuse the accuracy seed for this M so both runs feed the same
        # inputs to the kernel.  Keeps debug logs (same routing, same
        # quantized inputs) consistent across accuracy and perf phases.
        seed = seeds.get(M, 42 + M)
        cuda_cg, tri_cg = perf_test(
            model_cfg, M, tk, seed=seed, use_cluster=args.cluster
        )
        if cuda_cg != cuda_cg or tri_cg != tri_cg:  # NaN check
            print(
                f"  {M:>4d}  {tk:>3d}  {path:>5s}  "
                f"{'N/A':>10s}  {'N/A':>11s}  {'N/A':>8s}"
            )
        else:
            speedup = tri_cg / cuda_cg if cuda_cg > 0 else 0
            print(
                f"  {M:>4d}  {tk:>3d}  {path:>5s}  "
                f"{cuda_cg:>10.3f}  {tri_cg:>11.3f}  {speedup:>7.2f}x"
            )

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n\n{banner}")
    print(f"# ACCURACY SUMMARY — {model_cfg['display_name']}")
    print(banner)
    print(
        f"\n  {'Config':<12s}  {'CUDA-Py':>8s}  {'CUDA-Tri':>9s}  "
        f"{'Tri-Py':>7s}  {'Route':>6s}"
    )
    print(f"  {'-' * 12}  {'-' * 8}  {'-' * 9}  {'-' * 7}  {'-' * 6}")
    all_pass = True
    for label, r in acc_results.items():
        ok = r["cuda_py"] > 0.99 if not (r["cuda_py"] != r["cuda_py"]) else False
        if not ok:
            all_pass = False
        print(
            f"  {label:<12s}  {r['cuda_py']:>8.4f}  {r['cuda_tri']:>9.4f}  "
            f"{r['tri_py']:>7.4f}  {'OK' if ok else 'FAIL':>6s}"
        )
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(banner)


if __name__ == "__main__":
    main()
