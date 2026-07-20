# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Profiling driver for the MoE monokernel and the Triton FP8 baseline.

Invokes either:
  * `torch.ops.vllm.moe_monokernel_topk` (the in-tree CUDA monokernel), or
  * `fused_topk` + `fused_experts_impl` (vLLM's Triton FP8 path), the
    same code path real serving stacks hit on the FP8 + block-scale
    config (matches `_triton_e2e` in `test_monokernel_accuracy.py`).

Designed to be run under `ncu` (per-kernel SM metrics) or `nsys`
(end-to-end timeline + NVTX ranges) — running bare is only useful for
sanity-checking the driver itself; the Python dispatch path dominates a
single launch.

Each (path, batch size) pair is wrapped in its own NVTX range so:
  * `ncu --nvtx --nvtx-include "monokernel_<model>_bs<BS>/"` (or the
    matching `triton_<model>_bs<BS>` range) limits capture to exactly
    that path's iterations
  * `nsys profile` writes one timeline that you can filter by NVTX
    range in the Nsight Systems UI.

Models (--model):
    qwen3.5       Qwen3.5-35B  block-wise FP8 (E=256, N_HALF=512,  K=2048, top_k=8) [default]
    qwen3.5_122b  Qwen3.5-122B block-wise FP8 (E=256, N_HALF=1024, K=3072, top_k=8)

Usage:
    # CUDA monokernel only, single BS
    python profile_monokernel.py --model qwen3.5 --bs 8

    # CUDA monokernel only, predefined sweep BS in {1, 2, 4, 8}
    python profile_monokernel.py --model qwen3.5 --bs8

    # Triton FP8 path only, same sweep
    python profile_monokernel.py --model qwen3.5 --bs8 --path triton

    # Both paths back-to-back (separate NVTX ranges per path × BS)
    python profile_monokernel.py --model qwen3.5 --bs8 --path both

    # Profile the 122B shape instead
    python profile_monokernel.py --model qwen3.5_122b --bs8 --path both --graph

    # Replay each call under a CUDA graph (matches production inference-
    # engine dispatch: no Python / allocator / torch dispatch overhead
    # between launches).  Both monokernel and Triton paths support this.
    python profile_monokernel.py --model qwen3.5 --bs8 --path both --graph
"""

import argparse
import os
import sys
import time

# Put the directory that *contains* the `vllm` package on sys.path so the
# local source is importable no matter where this script is run from.
# Two layouts are supported:
#   * script at repo root  (~/vllm/profile_monokernel.py)       -> pkg in ./vllm
#   * script inside vllm/  (~/vllm/vllm/profile_monokernel.py)  -> pkg in .
# IMPORTANT: insert the PARENT of the `vllm` package, never the package dir
# itself. Inserting `.../vllm/vllm` would put the package's submodules
# (e.g. `vllm/vllm/tokenizers`) on the top-level path and shadow the real
# `tokenizers` wheel, breaking transformers with a circular import.
# (Mirrors the sys.path setup in test_monokernel_accuracy.py.)
_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (_here, os.path.join(_here, "vllm")):
    if os.path.isfile(os.path.join(_cand, "vllm", "__init__.py")):
        sys.path.insert(0, _cand)
        break

import torch

# The MoE CUDA ops moved from the full-libtorch `_moe_C` extension to the
# stable-ABI `_moe_C_stable_libtorch` extension; import that so the
# `torch.ops._moe_C.*` monokernel ops are registered.
import vllm._moe_C_stable_libtorch  # noqa: F401  registers the _moe_C symbols

import vllm._custom_ops  # noqa: F401  registers torch.ops.vllm.moe_monokernel_topk

# ── Model registry ──────────────────────────────────────────────────────────
# Each entry provides the dims needed by the monokernel. Only shapes that
# have a matching BSx_Ey wrapper registered in moe_wrapper.cu will work.
# The high-level op (torch.ops.vllm.moe_monokernel_topk) dispatches by
# (E, N_fused, K) — see vllm/_custom_ops.py:
#   * qwen3.5      → E=256, N_fused=1024, K=2048 (35B  low-level op)
#   * qwen3.5_122b → E=256, N_fused=2048, K=3072 (122B low-level op)
MODELS = {
    "qwen3.5": {
        "display_name": "Qwen3.5-35B block-wise FP8",
        "E": 256,
        "N_HALF": 512,  # moe_intermediate_size; fused gate+up matrix has 2*N_HALF rows
        "K": 2048,  # hidden_states
        "top_k": 8,
    },
    "qwen3.5_122b": {
        "display_name": "Qwen3.5-122B block-wise FP8",
        "E": 256,
        "N_HALF": 1024,  # moe_intermediate_size; fused gate+up matrix has 2*N_HALF rows
        "K": 3072,  # hidden_states
        "top_k": 8,
    },
    "down2048": {
        # Synthetic down-boundary benchmark: N_half = K = 2048 gives the
        # down projection an 8-step K-loop per expert (vs 2 on the 35B
        # BS16 base op) with the SAME DCT=128/KDN=256/DOWN_GROUPS=8 down
        # geometry — isolates the expert-boundary amortization question.
        "display_name": "Synthetic down-K2048 (E256 N_half2048 K2048)",
        "E": 256,
        "N_HALF": 2048,
        "K": 2048,
        "top_k": 8,
    },
}
DEV = "cuda"

# Predefined sweep for --bs8: "the small batch sizes we care about first"
BS8_SWEEP = [1, 2, 4, 8]


def quant_fp8_block_wise(w, block_row=128, block_col=128):
    """Minimal block-wise FP8 quantizer (128x128 blocks).

    Slower than the monokernel's on-device path, but we only call this once
    per profiling run to prepare weights.
    """
    E_, rows, cols = w.shape
    rb = (rows + block_row - 1) // block_row
    cb = (cols + block_col - 1) // block_col
    wf = w.float()
    scales = torch.zeros(E_, rb, cb, device=w.device, dtype=torch.float32)
    w_fp8 = torch.zeros_like(wf)

    for e in range(E_):
        for ri in range(rb):
            r0, r1 = ri * block_row, min((ri + 1) * block_row, rows)
            for ci in range(cb):
                c0, c1 = ci * block_col, min((ci + 1) * block_col, cols)
                block = wf[e, r0:r1, c0:c1]
                amax = block.abs().max().clamp(min=1e-12)
                s = amax / 448.0
                scales[e, ri, ci] = s
                w_fp8[e, r0:r1, c0:c1] = (block / s).clamp(-448, 448)
    return w_fp8.to(torch.float8_e4m3fn), scales


def build_scratchpad(bs, n_half, k, num_experts=256, grid_size=128):
    """Allocate the monokernel scratchpad matching the current `MoEGemmSpec<Dims>`
    layout in `csrc/moe/moe_monokernel/src/moe_internal.h`.

    Mirrors the (shape-parameterized) sizing in `test_monokernel_accuracy.py`.
    Fields covered (in struct order):
        activations      [BS][K]                         fp8
        temp_bf16        [TEMP_ROWS][N_half]             bf16
        temp_fp8         [TEMP_ROWS][N_half]             fp8     (WGMMA path)
        temp_act_scale   [TEMP_ROWS][N_half / ACT_BLK]   fp32    (WGMMA path)
        down_partial_out [DOWN_GROUPS][BS][K]            fp32    (WGMMA path)
        act_scale        [BS][ceil(K/128)]               fp32
        grid_barrier     [2]                             uint32
        partial_barrier  [NUM_EXPERTS+DOWN_GRID][2]      uint32

    `TEMP_ROWS` uses the compile-time `Dims::BS` (= 8 for the BS8 variant,
    64 for BS64) — NOT the runtime batch size — because that's what the
    kernel's struct layout uses.  Same for `down_partial_out`.

    Shape-derived layout constants (mirror MoECoreDims<Dims>) — these MUST
    track per-shape geometry, not 35B literals:
        DOWN_COL_TILE = K / (GRID_SIZE / DOWN_GROUPS)   (256 for 35B, 384 for 122B)
        UP_COL_HALVES = (2*N_half * DOWN_COL_TILE) / (128 * K)   (1 for 35B, 2 for 122B)
        DOWN_ACT_BLOCK_SIZE = max(UP_COL_HALVES * 64, 64)        (64 for 35B, 128 for 122B)
        TEMP_ACT_SCALE_COLS = N_half / DOWN_ACT_BLOCK_SIZE
    """
    is_bs8 = bs <= 8
    dims_bs = 8 if is_bs8 else 64  # compile-time Dims::BS

    SPEC_MAX_TOPK = 8
    TEMP_ROWS = dims_bs * SPEC_MAX_TOPK + 8  # 72 on BS8, 520 on BS64
    ACT_SCALE_BLOCKS = (k + 127) // 128

    # Variant-dependent DOWN_* / UP_* (matches MoECoreDims<Dims>).  The BS8
    # TMA+WGMMA path fixes DOWN_GROUPS == 16 so DOWN_COL_TILE = K/(GRID/16)
    # (256 for 35B's K=2048, 384 for 122B's K=3072); BS64/non-TMA uses 128.
    if is_bs8:
        down_groups = 16
        down_grid = grid_size // down_groups  # 8
        down_col_tile = k // down_grid  # 256 (35B) / 384 (122B)
        up_col_halves = (2 * n_half * down_col_tile) // (128 * k)  # 1 / 2
    else:
        down_col_tile = 128
        down_grid = k // down_col_tile
        down_groups = grid_size // down_grid if down_grid > 0 else 1
        up_col_halves = 1
    down_act_block_size = max(up_col_halves * 64, 64)  # 64 (35B) / 128 (122B)
    temp_act_scale_cols = n_half // down_act_block_size

    spec_bytes = (
        dims_bs * k * 1  # activations fp8
        + TEMP_ROWS * n_half * 2  # temp_bf16
        + TEMP_ROWS * n_half * 1  # temp_fp8
        + TEMP_ROWS * temp_act_scale_cols * 4  # temp_act_scale
        + down_groups * dims_bs * k * 4  # down_partial_out
        + dims_bs * ACT_SCALE_BLOCKS * 4  # act_scale
        + 2 * 4  # grid_barrier
        + (num_experts + down_grid) * 2 * 4  # partial_barrier
    )
    # Round up to float32 and add a 16 KB safety margin.
    floats = (spec_bytes + 3) // 4 + 4096
    return torch.zeros(floats, dtype=torch.float32, device=DEV)


def resolve_batch_sizes(args):
    """Combine --bs8 (predefined) and --bs (explicit) into a de-duplicated,
    ordered list preserving first-seen order.
    """
    bss = []
    if args.bs8:
        bss.extend(BS8_SWEEP)
    if args.bs:
        bss.extend(args.bs)
    if not bss:
        raise SystemExit(
            "No batch sizes selected. Pass --bs8 and/or --bs <N> "
            "(e.g. --bs 8, or --bs8 for the {1,2,4,8} sweep)."
        )
    seen, out = set(), []
    for b in bss:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def _make_monokernel_call(
    bs, n_half, k, num_experts, top_k, x_full, logits_full, w13_fp8, s13, w2_fp8, s2
):
    """Build a 0-arg closure that runs one monokernel iteration."""
    op = torch.ops.vllm.moe_monokernel_topk
    x = x_full[:bs].contiguous()
    logits = logits_full[:bs].contiguous()
    scratchpad = build_scratchpad(bs, n_half, k, num_experts=num_experts)

    def _call():
        return op(
            x,
            logits,
            w13_fp8,
            s13,
            w2_fp8,
            s2,
            scratchpad,
            top_k=top_k,
            scoring_func="softmax",
            renormalize=True,
        )

    return _call


def _make_triton_call(bs, top_k, x_full, logits_full, w13_fp8, s13, w2_fp8, s2):
    """Build a 0-arg closure that runs one Triton FP8 iteration.

    Mirrors `_triton_e2e` in `test_monokernel_accuracy.py`: vLLM's
    `fused_topk` produces (topk_w, topk_ids) from raw router logits and
    `fused_experts_impl` runs the Triton block-wise FP8 GEMM1 + SiLU +
    requant + GEMM2 + moe_sum pipeline.  Imports are lazy so this file
    still works in CUDA-only builds without the Triton fused-MoE path.
    """
    from vllm.model_executor.layers.fused_moe import fused_topk
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts_impl,
    )

    x = x_full[:bs].contiguous()
    logits = logits_full[:bs].contiguous()

    def _call():
        topk_w, topk_ids, _ = fused_topk(
            hidden_states=x,
            gating_output=logits,
            topk=top_k,
            renormalize=True,
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

    return _call


def _profile_call(call_fn, range_name, warmup, iters, use_graph):
    """Generic warmup + (optional graph capture) + NVTX-wrapped iter loop.

    Identical structure for both the monokernel and the Triton path so
    the NVTX timeline is directly comparable across paths.

    The monokernel launches via standard `cudaLaunchKernel` (spec R1
    replaces cooperative-groups grid sync with the software
    `grid_barrier`), and Triton's `fused_experts_impl` is graph-safe by
    construction, so CUDA graph capture is legal for both paths.

    NOTE on nsys + CUDA graphs: nsys's `--capture-range=cudaProfilerApi`
    only records events between `cudaProfilerStart` and `cudaProfilerStop`.
    If the graph is captured BEFORE `cudaProfilerStart`, nsys never sees
    the graph creation and therefore cannot resolve individual node
    kernels — it only shows opaque `cudaGraphLaunch` events.  To fix
    this, we perform graph capture INSIDE the profiler window (after
    `cudaProfilerStart`).  The capture overhead is a one-time cost that
    is negligible relative to `iters` replays.
    """
    # ── Warmup (outside the NVTX range) ─────────────────────────────────
    # Stabilises allocator caches, triggers Triton autotuning / JIT
    # compilation, and primes the tma-interleave cache on the
    # monokernel's up-projection weight tensor.
    for _ in range(warmup):
        _ = call_fn()
    torch.cuda.synchronize()

    # ── Start profiler window ───────────────────────────────────────────
    # Everything from here is visible to nsys / ncu.
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(range_name)

    graph = None
    if use_graph:
        # Capture the graph INSIDE the profiler window so nsys can
        # observe graph creation and resolve node-level kernel events.
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            # One warmup iteration on the capture stream to prime it.
            _ = call_fn()
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            _ = call_fn()
        torch.cuda.synchronize()

    for _ in range(iters):
        if use_graph:
            graph.replay()
        else:
            _ = call_fn()
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()

    mode = "graph-replay" if use_graph else "eager"
    print(f"[profile]     {iters} iters ({mode}) inside '{range_name}'")


def profile_one_bs(
    path,
    x_full,
    logits_full,
    w13_fp8,
    s13,
    w2_fp8,
    s2,
    bs,
    n_half,
    k,
    top_k,
    num_experts,
    model_name,
    warmup,
    iters,
    use_graph=False,
):
    """Run warmup + profiled iterations for one (path, batch size) pair.

    `path` is `"monokernel"` or `"triton"`; the NVTX range name is
    `<path>_<model>_bs<BS>` (`+_graph` if `use_graph=True`).
    """
    if path == "monokernel":
        call_fn = _make_monokernel_call(
            bs,
            n_half,
            k,
            num_experts,
            top_k,
            x_full,
            logits_full,
            w13_fp8,
            s13,
            w2_fp8,
            s2,
        )
    elif path == "triton":
        call_fn = _make_triton_call(
            bs,
            top_k,
            x_full,
            logits_full,
            w13_fp8,
            s13,
            w2_fp8,
            s2,
        )
    else:
        raise ValueError(f"unknown path {path!r}")

    range_name = f"{path}_{model_name}_bs{bs}"
    if use_graph:
        range_name += "_graph"

    print(f"[profile]   bs={bs:>4d}  path={path}")
    _profile_call(call_fn, range_name, warmup, iters, use_graph)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        choices=sorted(MODELS.keys()),
        default="qwen3.5",
        help="Model shape to profile (default: qwen3.5). "
        "Determines E, N, K, and top_k.",
    )
    ap.add_argument(
        "--bs",
        type=int,
        action="append",
        default=None,
        help="Batch size (tokens). Repeat to add more, e.g. "
        "`--bs 1 --bs 4`. Dispatcher uses BS8 for M<=8, BS64 otherwise.",
    )
    ap.add_argument(
        "--bs8",
        action="store_true",
        help="Sweep BS in {1, 2, 4, 8} (all on the BS8 code path). "
        "Can be combined with --bs to add more sizes.",
    )
    ap.add_argument(
        "--path",
        choices=("monokernel", "triton", "both"),
        default="monokernel",
        help="Which implementation to profile: the in-tree CUDA "
        "monokernel, vLLM's Triton FP8 fused-experts path, or "
        "both back-to-back (separate NVTX ranges per path).",
    )
    ap.add_argument(
        "--warmup", type=int, default=3, help="Warmup iterations per BS (not profiled)."
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Profiled iterations per BS, inside the NVTX range.",
    )
    ap.add_argument(
        "--graph",
        action="store_true",
        help="Capture the op into a CUDA graph and replay "
        "`--iters` times inside the NVTX range.  Matches "
        "production inference-engine dispatch (no Python / "
        "torch-dispatch overhead between launches).",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = MODELS[args.model]
    E = cfg["E"]
    N_HALF = cfg["N_HALF"]
    N_FUSED = 2 * N_HALF
    K = cfg["K"]
    top_k = cfg["top_k"]

    batch_sizes = resolve_batch_sizes(args)

    torch.manual_seed(args.seed)
    assert torch.cuda.is_available(), "CUDA required"
    print(
        f"[profile] model={args.model} ({cfg['display_name']}) "
        f"E={E} N_fused={N_FUSED} K={K} top_k={top_k}"
    )
    print(
        f"[profile] device={torch.cuda.get_device_name(0)} "
        f"batch_sizes={batch_sizes} warmup={args.warmup} iters={args.iters} "
        f"path={args.path} graph={args.graph}"
    )

    # ── Build inputs once (largest BS), slice per iteration ─────────────
    max_bs = max(batch_sizes)
    w13_f = torch.randn(E, N_FUSED, K, device=DEV) * 0.1
    w2_f = torch.randn(E, K, N_HALF, device=DEV) * 0.1
    print("[profile] quantizing weights (one-time setup)...")
    t0 = time.time()
    w13_fp8, s13 = quant_fp8_block_wise(w13_f)
    w2_fp8, s2 = quant_fp8_block_wise(w2_f)
    print(f"[profile] weight quantization done in {time.time() - t0:.1f}s")

    x_full = torch.randn(max_bs, K, device=DEV, dtype=torch.bfloat16)
    logits_full = torch.randn(max_bs, E, device=DEV, dtype=torch.bfloat16)

    op = torch.ops.vllm.moe_monokernel_topk

    # Ensure contiguous — the op asserts this.
    w13_fp8 = w13_fp8.contiguous()
    s13 = s13.contiguous()
    w2_fp8 = w2_fp8.contiguous()
    s2 = s2.contiguous()

    paths = ("monokernel", "triton") if args.path == "both" else (args.path,)
    # Force-resolve the monokernel op so any registration error surfaces
    # before the first NVTX range starts.
    if "monokernel" in paths:
        _ = op
    for bs in batch_sizes:
        for path in paths:
            profile_one_bs(
                path,
                x_full,
                logits_full,
                w13_fp8,
                s13,
                w2_fp8,
                s2,
                bs,
                N_HALF,
                K,
                top_k,
                E,
                args.model,
                args.warmup,
                args.iters,
                use_graph=args.graph,
            )


if __name__ == "__main__":
    main()
