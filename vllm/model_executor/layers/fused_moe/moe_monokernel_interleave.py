# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight pre-interleave for the TMA+WGMMA MoE kernel (SWIZZLE_128B path).

The BS8 TMA+WGMMA kernel uses `CU_TENSOR_MAP_SWIZZLE_128B` on both weight
TMA descriptors, paired with WGMMA A descriptors in CUTLASS Major::K B128
form (`swizzle=1`, `A_LBO=16`, `A_SBO=1024`).  Under SWZ128 the TMA
hardware applies the 8-row × 128-byte core-matrix XOR swizzle at write
time, so byte-level pre-interleave is NOT needed — the raw row-major
tensor bytes are what the hardware expects.

The one thing still needed is a GATE/UP ROW repack for the up-projection
weights.  The kernel's WGMMA tile composites gate and up stripes into a
single 128-row A operand; packing those stripes contiguously in GM lets
one `boxDim=(128, 128)` TMA fetch the full tile in a single issue.

`interleave_for_tma_wgmma_up` implements that repack.  The down-projection
weights need no Python-side preparation — the raw `[E, K, N]` row-major
fp8 tensor is passed straight through to the kernel.

NOTE: This module is the canonical home for the interleave helpers so they
are importable from anywhere on `sys.path` (e.g. the vLLM server process,
which does not run from the repo root).  The repo-root `interleave_weights.py`
re-exports these functions for backwards compatibility with the standalone
benchmark / accuracy scripts.
"""

import torch


def interleave_for_tma_wgmma_up(w_fp8: torch.Tensor) -> torch.Tensor:
    """Repack fp8 up-projection weights so a single
    `boxDim=(128, 128)` SWIZZLE_128B TMA issue fetches one full
    128-row × 128-K WGMMA A-tile into SHM.

    Input layout: `[E, 2*N, K]` row-major fp8, with the first `N` rows
    per expert being the gate weights and the last `N` rows being the
    up weights.

    Output layout (still `[E, 2*N, K]`, same total byte footprint):
      For every expert `e` and every 64-gate-row block `k` in `[0, N/64)`:
        new[e, 128k +   0 : 128k +  32, :] = gate[e, 64k     : 64k + 32, :]
        new[e, 128k +  32 : 128k +  64, :] =   up[e, 64k     : 64k + 32, :]
        new[e, 128k +  64 : 128k +  96, :] = gate[e, 64k + 32 : 64k + 64, :]
        new[e, 128k +  96 : 128k + 128, :] =   up[e, 64k + 32 : 64k + 64, :]

    With this packing, for any `base_row_up` that is a multiple of 64,
    rows `[base_row_up, base_row_up + 64)` of gate and rows
    `[base_row_up, base_row_up + 64)` of up are laid out as 4 × 32-row
    gate/up/gate/up stripes in a contiguous 128-row block of the
    interleaved tensor.  A single `boxDim=(128, 128)` TMA at outer
    coordinate `expert_id * 2 * N + 2 * base_row_up` fetches exactly
    the composite 128×128 WGMMA A-tile the kernel expects: gate-WG0 at
    SHM rows [0..31], up-WG0 at [32..63], gate-WG1 at [64..95], up-WG1
    at [96..127].

    Under SWZ128 the TMA applies the 8-row × 128-byte core-matrix XOR
    swizzle automatically, so this helper only rearranges GM rows —
    no byte-level permutation inside each sub-block.

    Caching: the result is stashed on the input tensor as
    `_tma_interleaved_up`; repeat calls return the cached tensor.

    Args:
        w_fp8: [E, 2*N, K] fp8 tensor (row-major).  `N` must be a
               multiple of 64.

    Returns:
        [E, 2*N, K] fp8 tensor with the gate/up repack applied.
    """
    cached = getattr(w_fp8, "_tma_interleaved_up", None)
    if cached is not None:
        return cached

    E, rows, K = w_fp8.shape
    assert rows % 2 == 0, f"expected rows = 2*N, got rows={rows}"
    N_half = rows // 2
    assert N_half % 64 == 0, (
        f"N (half of rows) must be a multiple of 64; got N={N_half}"
    )

    # Split into gate / up (first N rows / last N rows).
    gate = w_fp8[:, :N_half, :]
    up = w_fp8[:, N_half:, :]

    # Each 64-row block of gate pairs with the same 64-row block of up.
    # Split both along the row axis into 32-row stripes.
    blocks = N_half // 64  # number of 64-row gate/up blocks per expert
    gate_r = gate.reshape(E, blocks, 64, K)
    up_r = up.reshape(E, blocks, 64, K)
    gate_lo = gate_r[:, :, :32, :]  # [E, blocks, 32, K]
    gate_hi = gate_r[:, :, 32:, :]  # [E, blocks, 32, K]
    up_lo = up_r[:, :, :32, :]  # [E, blocks, 32, K]
    up_hi = up_r[:, :, 32:, :]  # [E, blocks, 32, K]

    # Stack as gate_lo, up_lo, gate_hi, up_hi along a new stripe axis,
    # then flatten into a contiguous 128-row per-block slab.
    # [E, blocks, 4, 32, K] -> [E, blocks * 128, K]
    stripes = torch.stack([gate_lo, up_lo, gate_hi, up_hi], dim=2)
    result = stripes.reshape(E, blocks * 128, K).contiguous()

    try:
        w_fp8._tma_interleaved_up = result
    except AttributeError:
        pass

    return result


def interleave_for_tma_wgmma_up_v2(w_fp8: torch.Tensor) -> torch.Tensor:
    """Repack fp8 up-projection weights for the Pair_Layout (V2) variant.

    Under Pair_Layout, each warp's 16-row SHM stripe holds 8 gate rows
    in the d[0..1] half AND 8 up rows in the d[2..3] half, so
    silu(gate)*up becomes a per-lane register operation after the WGMMA.

    Input layout: `[E, 2*N, K]` row-major fp8, with the first `N` rows
    per expert being the gate weights and the last `N` rows being the
    up weights.

    Output layout (still `[E, 2*N, K]`, same total byte footprint):
      For every expert `e` and every 64-gate-row block `b` in `[0, N/64)`:
        Within the 128-row slab starting at row 128*b:
          For each WG `wg` in [0, 2) and each warp `w` in [0, 4):
            SHM rows [(wg*64 + w*16) .. (wg*64 + w*16 + 7)]
              = gate[e, 64*b + wg*32 + w*8 : 64*b + wg*32 + w*8 + 8, :]
            SHM rows [(wg*64 + w*16 + 8) .. (wg*64 + w*16 + 15)]
              = up[e, 64*b + wg*32 + w*8 : 64*b + wg*32 + w*8 + 8, :]

    The TMA descriptor is unchanged (same boxDim, globalStride, swizzle);
    only the byte arrangement in GM differs from V1.

    Caching: the result is stashed on the input tensor as
    `_tma_interleaved_up_v2`; repeat calls return the cached tensor.

    Args:
        w_fp8: [E, 2*N, K] fp8 tensor (row-major).  `N` must be a
               multiple of 64.

    Returns:
        [E, 2*N, K] fp8 tensor with the Pair_Layout gate/up repack.
    """
    cached = getattr(w_fp8, "_tma_interleaved_up_v2", None)
    if cached is not None:
        return cached

    E, rows, K = w_fp8.shape
    assert rows % 2 == 0, f"expected rows = 2*N, got rows={rows}"
    N_half = rows // 2
    assert N_half % 64 == 0, (
        f"N (half of rows) must be a multiple of 64; got N={N_half}"
    )

    # Split into gate / up (first N rows / last N rows).
    gate = w_fp8[:, :N_half, :]  # [E, N, K]
    up = w_fp8[:, N_half:, :]  # [E, N, K]

    blocks = N_half // 64  # number of 64-gate-row blocks per expert

    # Reshape gate and up into [E, blocks, 64, K]
    gate_r = gate.reshape(E, blocks, 64, K)
    up_r = up.reshape(E, blocks, 64, K)

    # For each block, split into WG0 (rows 0..31) and WG1 (rows 32..63)
    # Then for each WG, split into 4 warps of 8 rows each.
    # gate_r[:, :, :32, :] = WG0 gate rows, gate_r[:, :, 32:, :] = WG1 gate
    gate_wg0 = gate_r[:, :, :32, :].reshape(
        E, blocks, 4, 8, K
    )  # [E, b, 4warps, 8rows, K]
    gate_wg1 = gate_r[:, :, 32:, :].reshape(E, blocks, 4, 8, K)
    up_wg0 = up_r[:, :, :32, :].reshape(E, blocks, 4, 8, K)
    up_wg1 = up_r[:, :, 32:, :].reshape(E, blocks, 4, 8, K)

    # For each warp, interleave gate(8 rows) then up(8 rows) = 16 rows
    # WG0: [gate_w0(8), up_w0(8), gate_w1(8), up_w1(8), gate_w2(8), up_w2(8), gate_w3(8), up_w3(8)]
    # = 64 rows for WG0, then same for WG1 = 128 rows total per block.
    # Stack gate and up along a new axis: [E, blocks, 4warps, 2(gate/up), 8rows, K]
    wg0_interleaved = torch.stack([gate_wg0, up_wg0], dim=3)  # [E, b, 4, 2, 8, K]
    wg1_interleaved = torch.stack([gate_wg1, up_wg1], dim=3)  # [E, b, 4, 2, 8, K]

    # Flatten warp and gate/up dims: [E, b, 4*2*8, K] = [E, b, 64, K]
    wg0_flat = wg0_interleaved.reshape(E, blocks, 64, K)
    wg1_flat = wg1_interleaved.reshape(E, blocks, 64, K)

    # Concatenate WG0 and WG1: [E, b, 128, K]
    result = torch.cat([wg0_flat, wg1_flat], dim=2)
    result = result.reshape(E, blocks * 128, K).contiguous()

    try:
        w_fp8._tma_interleaved_up_v2 = result
    except (AttributeError, RuntimeError):
        pass

    return result
