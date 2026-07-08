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

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# The MoE CUDA ops moved from the full-libtorch `_moe_C` extension to the
# stable-ABI `_moe_C_stable_libtorch` extension; import that so the
# `torch.ops._moe_C.*` monokernel ops are registered.
import vllm._moe_C_stable_libtorch  # noqa: E402,F401

import vllm._custom_ops as ops  # noqa: E402
import vllm.model_executor.layers.fused_moe.modular_kernel as mk  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (  # noqa: E402
    _get_config_dtype_str,
    _get_config_quant_dtype,
    dispatch_fused_moe_kernel,
    moe_align_block_size,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.utils import (  # noqa: E402
    moe_kernel_quantize_input,
)
from vllm.triton_utils import tl  # noqa: E402

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
        "op_bs8_cluster": "moe_monokernel_topk_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA_Cluster",  # noqa: E501
        "op_bs64": "moe_monokernel_topk_BS64_E256_Qwen3_5_35B_BlockFP8",
    },
    "qwen3.5_122b": {
        "display_name": "Qwen3.5-122B block-wise FP8",
        "E": 256,
        "N_HALF": 1024,  # moe_intermediate_size; fused gate+up is 2*N_HALF
        "K": 3072,
        "default_top_k": 8,
        # BS8 TMA+WGMMA path, 122B shape (UP_COL_HALVES=2, DOWN_COL_HALVES=3).
        # TWO-TMA up-projection: gate and up are fetched with two separate
        # raw 128-row TMAs (128 gate cols + 128 up cols per block) straight
        # from the unmodified [E, 2*N, K] tensor, so NO gate/up PAIR interleave
        # is applied.  Built from csrc/moe/moe_monokernel_122B/ alongside the
        # 35B kernel.  Validated through the high-level serving op
        # (`torch.ops.vllm.moe_monokernel_topk`), which is the exact path
        # vLLM serving runs.
        "op_bs8": "moe_monokernel_topk_BS8_E256_Qwen3_5_122B_BlockFP8_WGMMA_TMA",
        # No cluster / BS64 variant for 122B.
        "op_bs8_cluster": None,
        "op_bs64": None,
    },
    # ── E-sweep shapes (35B N/K, varying expert count) ───────────────────
    # Declared in csrc/moe/moe_monokernel/shapes.json; ops are shape-keyed
    # (E{E}_N512_K2048).  Same N_half/K as the 35B shape, so the kernel path
    # is identical except for NUM_EXPERTS.  Used to validate the generic-E
    # build + occupancy at E=512.
    "e64": {
        "display_name": "E-sweep E=64 (N512 K2048) block-wise FP8",
        "E": 64, "N_HALF": 512, "K": 2048, "default_top_k": 8,
        "op_bs8": "moe_monokernel_topk_BS8_E64_N512_K2048_BlockFP8_WGMMA_TMA",
        "op_bs8_cluster": None, "op_bs64": None,
    },
    "e128": {
        "display_name": "E-sweep E=128 (N512 K2048) block-wise FP8",
        "E": 128, "N_HALF": 512, "K": 2048, "default_top_k": 8,
        "op_bs8": "moe_monokernel_topk_BS8_E128_N512_K2048_BlockFP8_WGMMA_TMA",
        "op_bs8_cluster": None, "op_bs64": None,
    },
    "e512": {
        "display_name": "E-sweep E=512 (N512 K2048) block-wise FP8",
        "E": 512, "N_HALF": 512, "K": 2048, "default_top_k": 8,
        "op_bs8": "moe_monokernel_topk_BS8_E512_N512_K2048_BlockFP8_WGMMA_TMA",
        "op_bs8_cluster": None, "op_bs64": None,
    },
    # ── Decoupled up/down carve (UP_GROUPS=64 != DOWN_GROUPS=8, R=8) ─────
    # UP_COL_HALVES=2 pinned explicitly (raw two-TMA up-proj, NO interleave);
    # site #2 uses the expert-slot producer→consumer barrier split.  Sigmoid
    # scoring + top_k=8 is the serving config for this shape.
    "glm52": {
        "display_name": "E256 N_half256 K6144 decoupled block-wise FP8",
        "E": 256, "N_HALF": 256, "K": 6144, "default_top_k": 8,
        # Serving config for this shape scores with sigmoid (renormalized),
        # not softmax; the reference and kernel calls follow this field.
        "scoring_func": "sigmoid",
        # GLM 5.2 serving routing: noaux_tc biased selection
        # (e_score_correction_bias) + routed_scaling_factor=2.5.  A non-zero
        # `use_expert_bias` flag makes accuracy_test synthesize a per-expert
        # bias and exercise the kernel's biased-select / unbiased-weight path;
        # `routed_scaling_factor` is folded into the kernel's routing weights.
        "use_expert_bias": True,
        "routed_scaling_factor": 2.5,
        # Explicit UCH (decoupled shapes only): the host-side scratchpad
        # layout math can't derive it from the DCT coupling identity.
        "up_col_halves": 2,
        "op_bs8": "moe_monokernel_topk_BS8_E256_N256_K6144_BlockFP8_WGMMA_TMA",
        "op_bs8_cluster": None, "op_bs64": None,
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


def _register_models_from_registry():
    """Auto-register every shape from the generated monokernel registry
    (shapes.json via gen_shapes.py) that isn't already hand-declared above.

    Newly onboarded shapes become testable/tunable with no edit here: the
    registry row carries dims, op names, and the optional routing metadata
    (scoring_func / use_expert_bias / routed_scaling_factor /
    up_col_halves).  Hand entries win on key collisions so local overrides
    (e.g. the 35B cluster op) are preserved.
    """
    from vllm.model_executor.layers.fused_moe import monokernel_shapes as REG

    known_dims = {(m["E"], m["N_HALF"], m["K"]) for m in MODELS.values()}
    for row in REG.SHAPES:
        dims = (row["E"], row["N_half"], row["K"])
        names = [row["key"]] + list(row.get("aliases", []))
        if dims in known_dims or any(n in MODELS for n in names):
            continue
        entry = {
            "display_name": row["display_name"],
            "E": row["E"],
            "N_HALF": row["N_half"],
            "K": row["K"],
            "default_top_k": row["default_top_k"],
            "op_bs8": row["named_op"],
            "op_bs8_cluster": None,
            "op_bs64": None,
        }
        for opt in ("scoring_func", "use_expert_bias",
                    "routed_scaling_factor", "up_col_halves"):
            if row.get(opt) is not None:
                entry[opt] = row[opt]
        for n in names:
            MODELS.setdefault(n, entry)


_register_models_from_registry()

DEV = "cuda"


def get_model_op(model_cfg, M, use_cluster=False):
    """Return the high-level monokernel dispatch op, or raise if the model's
    low-level kernel is not wired in the current build.

    For M <= 8 the BS8 path is ALWAYS TMA+WGMMA+Pair_Layout (the single
    BS8 design).  The returned wrapper calls the high-level serving op
    `torch.ops.vllm.moe_monokernel_topk`, which derives (E, N, K) from the
    weight shapes, selects the matching low-level BS8 symbol, and applies
    any per-shape gate/up interleave internally — the exact path vLLM
    serving runs.  `use_cluster=True` selects the Hopper cluster variant
    (`__cluster_dims__(8, 1, 1)` + multicast TMA + DSHM) instead of the
    per-block variant; both share the same Python call signature.
    """
    if M <= 8 and use_cluster:
        key = "op_bs8_cluster"
    elif M <= 8:
        key = "op_bs8"
    else:
        raise RuntimeError(
            f"Batch size M={M} > 8 is not supported: the BS64 monokernel "
            f"path has been removed. Use M <= 8 (the BS8 TMA+WGMMA path)."
        )
    op_name = model_cfg.get(key)
    if op_name is None:
        raise RuntimeError(
            f"Model '{model_cfg['display_name']}' has no registered kernel "
            f"for {key}. Register the wrapper in moe_wrapper.cu and add the "
            f"binding in torch_bindings.cpp first."
        )

    # Return the HIGH-LEVEL serving op (`torch.ops.vllm.moe_monokernel_topk`).
    # This is the exact dispatch vLLM serving runs: it derives (E, N, K) from
    # the weight shapes, selects the matching low-level BS8 op, and passes the
    # RAW weights (applying any per-shape interleave internally).  Testing this
    # path catches bugs in the high-level branch that a standalone low-level
    # wrapper would silently miss.
    import vllm._custom_ops  # noqa: F401  (registers torch.ops.vllm.*)

    def _high_level_op(
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
        expert_bias=None,
        routed_scaling_factor=1.0,
    ):
        return torch.ops.vllm.moe_monokernel_topk(
            activations_in,
            router_logits,
            expert_weights_up,
            expert_scales_up,
            expert_weights_down,
            expert_scales_down,
            scratchpad,
            top_k,
            scoring_func,
            renormalize,
            expert_bias,
            routed_scaling_factor,
        )

    return _high_level_op


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
    """Greedy softmax → topk → renormalize (matches CUDA kernel).

    Tie-break: LOWEST expert index wins on exactly-equal scores.  This
    matches BOTH `torch.topk`/iterative-`max` (which return the first/lowest
    index) AND vLLM's production `topk_softmax` kernel (csrc/moe/
    topk_softmax_kernels.cu: "we want lower indices to win", updates only on
    `>` not `>=`).  The monokernel routing was fixed (moe_routing.cu
    `warp_min_expert_with_max`) to break ties the same way, so the kernel and
    this reference now agree on the tie that seed=42 M=8 exposes
    (score[88]==score[115] at the rank-7/8 boundary).  Plain iterative
    `s.max()` already yields lowest-index, so no bias is needed.
    """
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


def routing_sigmoid_topk(
    logits_bf16, top_k, renormalize=True, expert_bias=None,
    routed_scaling_factor=1.0,
):
    """Sigmoid scoring → topk → optional renormalize (matches CUDA kernel).

    Without ``expert_bias``: selection is done on RAW logits (sigmoid is
    monotonic, so the top-k ids match post-activation ordering — same trick as
    the kernel's fast path in moe_routing.cu). Weights are sigmoid(logit) of
    the selected entries.

    With ``expert_bias`` (GLM noaux_tc routing): winners are ranked by the
    biased metric ``sigmoid(logit) + bias[e]``, but the routing WEIGHT stays
    the UNBIASED ``sigmoid(logit)`` of the selected experts (matching
    grouped_topk's original_scores split). This mirrors the CUDA kernel's
    biased-select / unbiased-weight path.

    Weights are divided by their sum when renormalize=True, then multiplied by
    routed_scaling_factor. Tie-break: lowest expert index wins (scatter of
    -inf + iterative max, as in routing_softmax_topk)."""
    raw = logits_bf16.float()
    M = raw.shape[0]
    ids = torch.zeros(M, top_k, dtype=torch.int64, device=DEV)
    # Rank on the biased sigmoid metric when a bias is supplied, else on the
    # raw logit (monotone-equivalent to unbiased sigmoid).
    if expert_bias is not None:
        metric = torch.sigmoid(raw) + expert_bias.float().unsqueeze(0)
    else:
        metric = raw
    s = metric.clone()
    for k in range(top_k):
        _, idx = s.max(dim=-1)
        ids[:, k] = idx
        s.scatter_(1, idx.unsqueeze(1), float("-inf"))
    # Unbiased sigmoid weight of the selected experts.
    wts = torch.sigmoid(raw.gather(1, ids))
    if renormalize:
        wts = wts / wts.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    if routed_scaling_factor != 1.0:
        wts = wts * routed_scaling_factor
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


def read_wgmma_intermediates_from_scratchpad(
    scratchpad_bytes, N_HALF, K, E, up_col_halves=None
):
    """Read the BS8 WGMMA-path intermediates from the monokernel scratchpad
    for step-by-step debugging of the 122B (and 35B) kernel.

    Mirrors the field order of MoEGemmSpec<Dims> in moe_internal.h
    (non-DEBUG build), all at the HEAD of the struct so offsets are stable:

        activations      [BS][K]                       fp8   (BS*K B)
        temp_fp8         [TEMP_ROWS][N_HALF]           fp8   (TEMP_ROWS*N_HALF B)
        temp_act_scale   [TEMP_ROWS][N_HALF/ACT_BLK]   f32
        down_partial_out [BS][K]                        f32   (single buffer)

    where ACT_BLK = DOWN_ACT_BLOCK_SIZE = UP_COL_HALVES*64
    (64 for 35B, 128 for 122B), and TEMP_ROWS = BS*8 + 8.

    Returns a dict:
        up_fp8:    [TEMP_ROWS, N_HALF]  fp8  — up-proj SiLU+quant output
        up_scale:  [TEMP_ROWS, N_HALF/ACT_BLK] f32 — per-(row, up-block) scale
        down_out:  [BS, K]   f32  — final fp32 sum (pre-bf16-cast)
        act_blk:   int       — activation-quant block size used by up output
    """
    BS = 8  # BS8 path
    TEMP_ROWS = BS * 8 + 8  # SPEC_MAX_TOPK=8
    # Shape-derived activation-quant block size (matches MoEGemmSpec).
    # DECOUPLED shapes pin UP_COL_HALVES explicitly (the DCT coupling
    # identity below doesn't hold for them); coupled shapes derive it.
    if up_col_halves is not None:
        UP_COL_HALVES = up_col_halves
    else:
        DOWN_GROUPS = 16
        DOWN_GRID = 128 // DOWN_GROUPS  # 8
        DOWN_COL_TILE = K // DOWN_GRID  # 256 (35B) / 384 (122B)
        UP_COL_HALVES = (2 * N_HALF * DOWN_COL_TILE) // (128 * K)  # 1 / 2
    ACT_BLK = max(UP_COL_HALVES * 64, 64)
    TACS = N_HALF // ACT_BLK

    off = 0
    act_bytes = BS * K  # activations fp8
    off += act_bytes

    fp8_bytes = TEMP_ROWS * N_HALF  # temp_fp8 fp8
    up_fp8 = (
        scratchpad_bytes[off : off + fp8_bytes]
        .view(torch.float8_e4m3fn)
        .reshape(TEMP_ROWS, N_HALF)
        .clone()
    )
    off += fp8_bytes

    scale_bytes = TEMP_ROWS * TACS * 4  # temp_act_scale f32
    up_scale = (
        scratchpad_bytes[off : off + scale_bytes]
        .view(torch.float32)
        .reshape(TEMP_ROWS, TACS)
        .clone()
    )
    off += scale_bytes

    down_bytes = BS * K * 4  # down_partial_out f32 (single [BS][K] buffer)
    down_out = (
        scratchpad_bytes[off : off + down_bytes]
        .view(torch.float32)
        .reshape(BS, K)
        .clone()
    )

    return {
        "up_fp8": up_fp8,
        "up_scale": up_scale,
        "down_out": down_out,
        "act_blk": ACT_BLK,
    }


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
    qd = _get_config_quant_dtype(use_fp8_w8a8=True, use_int8_w8a8=False)
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
    scoring_func = model_cfg.get("scoring_func", "softmax")
    kernel_op = get_model_op(model_cfg, M, use_cluster=use_cluster)

    is_bs8 = M <= 8
    path = "BS8" if is_bs8 else "BS64"
    sep = "=" * 72

    print(f"\n{'#' * 72}")
    print(f"# ACCURACY [{model_cfg['display_name']}]: {path} M={M} top_k={top_k}"
          f" scoring={scoring_func}")
    print(f"{'#' * 72}")

    # Setup — block-wise quantization for all paths
    w13_f = torch.randn(E, 2 * N_HALF, K, device=DEV) * 0.1
    w2_f = torch.randn(E, K, N_HALF, device=DEV) * 0.1
    w13_fp8, s13 = quant_fp8_block_wise(w13_f)
    w2_fp8, s2 = quant_fp8_block_wise(w2_f)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    logits = torch.randn(M, E, device=DEV, dtype=torch.bfloat16)
    # GLM-style routing knobs (default: no bias, no scaling => Qwen path).
    routed_scaling_factor = model_cfg.get("routed_scaling_factor", 1.0)
    expert_bias = None
    if model_cfg.get("use_expert_bias"):
        # Synthesize a per-expert bias with the same magnitude regime as GLM
        # 5.2's e_score_correction_bias (~7.0, small spread), so selection is
        # dominated by the bias just like production. float32 [E] on device.
        expert_bias = (
            7.0 + 0.05 * torch.randn(E, device=DEV, dtype=torch.float32)
        )
    if scoring_func == "sigmoid":
        topk_w, topk_ids = routing_sigmoid_topk(
            logits, top_k, expert_bias=expert_bias,
            routed_scaling_factor=routed_scaling_factor,
        )
    else:
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
    # Shape-derived layout constants.  These mirror MoECoreDims<Dims> /
    # MoEGemmSpec<Dims> in moe_internal.h and MUST track the per-shape
    # geometry, not 35B literals:
    #   * UP_COL_HALVES   = (2*N_HALF * DOWN_COL_TILE) / (128 * K)
    #                       — 1 for 35B, 2 for 122B.
    #   * DOWN_COL_TILE    = K / DOWN_GRID; for the BS8 TMA path it is the
    #                       value that makes DOWN_GROUPS == UP_GROUPS == 16
    #                       (256 for 35B's K=2048, 384 for 122B's K=3072).
    #   * DOWN_ACT_BLOCK_SIZE = UP_COL_HALVES * 64 (64 for 35B, 128 for 122B)
    #     ⇒ TEMP_ACT_SCALE_COLS = N_HALF / DOWN_ACT_BLOCK_SIZE.
    # `down_partial_out` is a SINGLE [BS][K] buffer in the current kernel
    # (no per-DOWN_GROUPS dimension); Phase 5 reads each cell once.
    GRID_SIZE = 128
    if is_bs8 and model_cfg.get("up_col_halves") is not None:
        # DECOUPLED shape: UCH is pinned explicitly (the coupling identity
        # below has no integer solution).  Only UP_COL_HALVES feeds the
        # scratchpad layout (via DOWN_ACT_BLOCK_SIZE); the down-side carve
        # doesn't change the MoEGemmSpec field sizes.
        UP_COL_HALVES = model_cfg["up_col_halves"]
    elif is_bs8:
        # BS8 TMA+WGMMA: DOWN_COL_TILE chosen so DOWN_GROUPS == UP_GROUPS.
        # UP_GROUPS = GRID_SIZE / (2*N_HALF / 128) for UP_COL_HALVES=1, but
        # the invariant the kernel enforces is DOWN_GRID = K / DOWN_COL_TILE
        # with DOWN_GROUPS = 16, i.e. DOWN_COL_TILE = K / (GRID_SIZE / 16).
        DOWN_GROUPS = 16
        DOWN_GRID = GRID_SIZE // DOWN_GROUPS  # 8
        DOWN_COL_TILE = K // DOWN_GRID  # 256 (35B) / 384 (122B)
        UP_COL_HALVES = (2 * N_HALF * DOWN_COL_TILE) // (128 * K)  # 1 / 2
    else:
        DOWN_COL_TILE = 128
        DOWN_GRID = K // DOWN_COL_TILE
        DOWN_GROUPS = GRID_SIZE // DOWN_GRID if DOWN_GRID > 0 else 1
        UP_COL_HALVES = 1
    DOWN_ACT_BLOCK_SIZE = max(UP_COL_HALVES * 64, 64)
    TEMP_ACT_SCALE_COLS = N_HALF // DOWN_ACT_BLOCK_SIZE
    spec_size = (
        BS * K  # activations fp8
        + TEMP_ROWS * N_HALF * 2  # temp_bf16 (legacy BS64; kept for margin)
        + TEMP_ROWS * UP_PROJ_BLOCKS * 4  # temp_block_max (legacy BS64)
        + TEMP_ROWS * N_HALF * 1  # temp_fp8 (WGMMA path)
        + TEMP_ROWS * TEMP_ACT_SCALE_COLS * 4  # temp_act_scale (WGMMA)
        + BS * K * 4  # down_partial_out [BS][K] (single buffer, WGMMA)
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
        scoring_func=scoring_func,
        renormalize=True,
        expert_bias=expert_bias,
        routed_scaling_factor=routed_scaling_factor,
    )
    torch.accelerator.synchronize()

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
    torch.accelerator.synchronize()

    # ── STEP-BY-STEP scratchpad readback (BS8 WGMMA path) ────────────────
    # Localize accuracy divergence to a specific phase by comparing the
    # GM-persisted intermediates against the Python reference:
    #   * up output  → temp_fp8 * temp_act_scale   vs  py_silu (rw baked in)
    #   * down output→ down_partial_out[tok]        vs  py_out[tok] (pre-cast)
    # The up-proj rows are stored expert-sorted (row = sorted_slot[...]),
    # a permutation the host can't see, so we match each reference SiLU
    # row to its best-cosine kernel row and report the WORST such match —
    # if the up phase is correct every reference row finds a ~1.0 partner.
    if is_bs8:
        print(f"\n{sep}")
        print("STEP-BY-STEP INTERMEDIATES (CUDA scratchpad vs Py reference)")
        print(sep)
        interm = read_wgmma_intermediates_from_scratchpad(
            scratch_bytes, N_HALF, K, E,
            up_col_halves=model_cfg.get("up_col_halves"),
        )
        act_blk = interm["act_blk"]
        print(f"  up-proj activation-quant block size = {act_blk} "
              f"(expect 64 for 35B, 128 for 122B/decoupled)")

        # Dequantize all written up rows: row r, feature f uses scale
        # column f // act_blk.
        n_rows = M * top_k
        up_fp8 = interm["up_fp8"][:n_rows].float()  # [n_rows, N_HALF]
        up_scale = interm["up_scale"][:n_rows]  # [n_rows, N_HALF/act_blk]
        up_deq = up_fp8 * up_scale.repeat_interleave(act_blk, dim=1)

        # Best-cosine match each reference SiLU row → kernel up row.
        worst_up = 1.0
        worst_vrow = -1
        for vrow in range(n_rows):
            ref = py_silu[vrow].float()
            if ref.norm() < 1e-8:
                continue
            best = max(cos_sim(up_deq[r], ref) for r in range(n_rows))
            if best < worst_up:
                worst_up, worst_vrow = best, vrow
        print(f"  UP   : worst ref-row best-match cos={worst_up:.6f} "
              f"(vrow={worst_vrow})  [1.0 = up phase correct]")

        # Down output is natural [tok][hidden]; compare directly to py_out
        # (pre-bf16-cast fp32 sum over top_k).
        down_out = interm["down_out"][:M].float()  # [M, K]
        down_cos = cos_sim(down_out, py_out[:M].float())
        dm, dmn = err_stats(down_out, py_out[:M].float())
        print(f"  DOWN : down_partial_out vs py_out  cos={down_cos:.6f}  "
              f"max_err={dm:.4f}  mean_err={dmn:.4f}")
        # Per-token down cosine to spot a single bad token / col-stripe.
        for tok in range(min(M, 4)):
            tc = cos_sim(down_out[tok], py_out[tok].float())
            print(f"          tok{tok} down cos={tc:.6f}")

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
    scoring_func = model_cfg.get("scoring_func", "softmax")
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
            scoring_func=scoring_func,
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
# CAPTURED-ROUTING REPLAY
# ══════════════════════════════════════════════════════════════════════════════
#
# The synthetic perf sweep above routes from torch.randn(M, E) logits, which
# spreads ~M*top_k assignments almost uniformly across the experts (at M=8,
# top_k=8 that's ~64 assignments over ~64 distinct experts — one token each).
# Real decode routing is imbalanced: a few "hot" experts get most tokens. That
# changes the per-expert GEMM batching and is the likely reason the monokernel
# wins ~1.5x end-to-end but only ~1.04x on this synthetic kernel benchmark.
#
# This block replays REAL router logits captured from a live `vllm serve` run
# (see route_capture.py / MONOKERNEL_ROUTE_CAPTURE) into the exact same perf
# harness. Both the CUDA monokernel and the Triton baseline route internally
# from the provided logits, so feeding captured logits reproduces production
# routing for both paths. Weights and x stay random — kernel timing is
# data-independent, only the routing (which experts, how many tokens) matters.


def load_route_capture(path):
    """Load captured routing snapshots written by route_capture.py."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    snaps = blob.get("snapshots", [])
    meta = blob.get("meta", {})
    return snaps, meta


def _expert_concentration(logits, top_k, E):
    """Return (n_distinct_experts, max_tokens_on_one_expert, total_assignments)
    for one routing snapshot — a measure of how imbalanced the routing is."""
    _, ids = routing_softmax_topk(logits, top_k)
    flat = ids.reshape(-1).to(torch.int64)
    counts = torch.bincount(flat, minlength=E)
    n_distinct = int((counts > 0).sum().item())
    max_load = int(counts.max().item())
    return n_distinct, max_load, int(flat.numel())


def route_capture_perf(model_cfg, snapshots, seed=42, use_cluster=False):
    """Benchmark CUDA monokernel vs Triton on REAL captured routing.

    Weights are built once and reused across snapshots (they are shape-fixed
    per model); only the per-snapshot logits / M / top_k vary.
    """
    torch.manual_seed(seed)
    E = model_cfg["E"]
    N_HALF = model_cfg["N_HALF"]
    K = model_cfg["K"]

    # Shape-fixed inputs built once.
    w13_f = torch.randn(E, 2 * N_HALF, K, device=DEV) * 0.1
    w2_f = torch.randn(E, K, N_HALF, device=DEV) * 0.1
    w13c, s13c = quant_fp8_block_wise(w13_f)
    w13c, s13c = w13c.contiguous(), s13c.contiguous()
    w2c, s2c = quant_fp8_block_wise(w2_f)
    w2c, s2c = w2c.contiguous(), s2c.contiguous()
    scratchpad = torch.zeros(1024, 4096, dtype=torch.float32, device=DEV)
    # Free the fp32 weight masters; the fp8 copies are all the kernels need.
    del w13_f, w2_f
    torch.cuda.empty_cache()

    banner = "#" * 72
    print(f"\n\n{banner}")
    print(f"# CAPTURED-ROUTING PERF ({model_cfg['display_name']} shape, CUDA graph)")
    print(f"# {len(snapshots)} snapshots replayed from a live serve run")
    print(banner)
    print(
        f"\n  {'#':>4s}  {'layer':>5s}  {'M':>3s}  {'tk':>3s}  "
        f"{'experts':>7s}  {'maxld':>5s}  "
        f"{'CUDA(ms)':>9s}  {'Triton(ms)':>10s}  {'Speedup':>8s}"
    )
    print(
        f"  {'-' * 4}  {'-' * 5}  {'-' * 3}  {'-' * 3}  "
        f"{'-' * 7}  {'-' * 5}  {'-' * 9}  {'-' * 10}  {'-' * 8}"
    )

    speedups = []
    for i, snap in enumerate(snapshots):
        M = snap["M"]
        top_k = snap["top_k"]
        layer_id = snap.get("layer_id", -1)
        logits = snap["router_logits"].to(DEV, dtype=torch.bfloat16)
        if logits.shape != (M, E):
            # Skip snapshots whose shape doesn't match this model.
            continue
        kernel_op = get_model_op(model_cfg, M, use_cluster=use_cluster)
        n_distinct, max_load, _ = _expert_concentration(logits, top_k, E)
        x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)

        def cuda_fn():
            return kernel_op(
                x, logits, w13c, s13c, w2c, s2c, scratchpad,
                top_k=top_k, scoring_func="softmax", renormalize=True,
            )

        def triton_fn():
            return _triton_e2e(x, logits, w13c, s13c, w2c, s2c, top_k)

        try:
            cuda_cg = _bench_cudagraph(cuda_fn)
        except Exception:
            cuda_cg = float("nan")
        try:
            tri_cg = _bench_cudagraph(triton_fn)
        except Exception:
            tri_cg = float("nan")

        if cuda_cg != cuda_cg or tri_cg != tri_cg or cuda_cg <= 0:
            print(
                f"  {i:>4d}  {layer_id:>5d}  {M:>3d}  {top_k:>3d}  "
                f"{n_distinct:>7d}  {max_load:>5d}  "
                f"{'N/A':>9s}  {'N/A':>10s}  {'N/A':>8s}"
            )
            continue
        speedup = tri_cg / cuda_cg
        speedups.append(speedup)
        print(
            f"  {i:>4d}  {layer_id:>5d}  {M:>3d}  {top_k:>3d}  "
            f"{n_distinct:>7d}  {max_load:>5d}  "
            f"{cuda_cg:>9.3f}  {tri_cg:>10.3f}  {speedup:>7.2f}x"
        )

    if speedups:
        import statistics

        geomean = statistics.geometric_mean(speedups)
        mean = sum(speedups) / len(speedups)
        print(f"\n  Snapshots benched : {len(speedups)}")
        print(f"  Mean speedup      : {mean:.3f}x")
        print(f"  Geomean speedup   : {geomean:.3f}x")
        print(f"  Min / Max speedup : {min(speedups):.3f}x / {max(speedups):.3f}x")
    else:
        print("\n  No snapshots benchmarked (shape mismatch or all failed).")
    print(banner)


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
        default=[1, 2, 4, 8],
        help="Subset of M values to test.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the registered model configs and exit.",
    )
    parser.add_argument(
        "--route-capture",
        type=str,
        default=None,
        metavar="DUMP.pt",
        help="Replay REAL router logits captured from a live `vllm serve` run "
        "(via MONOKERNEL_ROUTE_CAPTURE) into the perf benchmark instead of the "
        "synthetic random sweep. Reproduces production's imbalanced expert "
        "routing for both the CUDA and Triton paths.",
    )
    parser.add_argument(
        "--route-limit",
        type=int,
        default=None,
        help="With --route-capture, benchmark at most this many snapshots.",
    )
    parser.add_argument(
        "--route-layer",
        type=int,
        default=None,
        help="With --route-capture, only benchmark snapshots from this "
        "MoE layer_id.",
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

    variant_tag = "  [WGMMA+TMA+CLUSTER]" if args.cluster else "  [WGMMA+TMA]"

    banner = "#" * 72
    print(banner)
    print(
        f"# MoE Monokernel Test — {model_cfg['display_name']} "
        f"(E={E}, N={N_HALF}, K={K})"
    )
    print(f"# model={args.model}  top_k={tk_default}{variant_tag}")
    print(banner)

    # ── Captured-routing replay mode ─────────────────────────────────────
    # When --route-capture is given, skip the synthetic accuracy+perf sweep
    # and instead replay real production routing through the perf harness.
    if args.route_capture:
        snaps, meta = load_route_capture(args.route_capture)
        print(f"# Loaded {len(snaps)} routing snapshots (meta={meta})")
        if args.route_layer is not None:
            snaps = [s for s in snaps if s.get("layer_id") == args.route_layer]
            print(f"# Filtered to layer_id={args.route_layer}: {len(snaps)} snapshots")
        if args.route_limit is not None:
            snaps = snaps[: args.route_limit]
            print(f"# Limited to first {len(snaps)} snapshots")
        if not snaps:
            print("# No snapshots to benchmark — exiting.")
            return
        route_capture_perf(model_cfg, snaps, use_cluster=args.cluster)
        return

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
        ok = r["cuda_py"] > 0.99 if r["cuda_py"] == r["cuda_py"] else False
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
