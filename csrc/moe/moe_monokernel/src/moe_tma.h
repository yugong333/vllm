#ifndef MOE_TMA_H
#define MOE_TMA_H

#pragma once

// Host-side factory declarations for TMA (Tensor Memory Accelerator)
// `CUtensorMap` descriptors used by the BS8 WGMMA up-projection path.
//
// These descriptors are built once per kernel launch from the torch-binding
// wrapper and passed to the device as `__grid_constant__ CUtensorMap const`
// kernel parameters.  They describe a tiled 2D view of the full
// `[E, 2*N, K]` fp8 up-projection weight tensor and the full
// `[BS, K_hidden]` bf16 activation tensor respectively.
//
// The definitions live in `moe_tma.cu` / `moe_tma.cpp` (host-side translation
// unit) and call `cuTensorMapEncodeTiled` from the CUDA Driver API, which
// requires CUDA 12.0 or later.  See the spec requirements R5.x and R12.x.

#include <cstdint>

// CUDA Driver API: provides `CUtensorMap` and `cuTensorMapEncodeTiled`.
// `CUtensorMap` itself was introduced in CUDA 12.0.
#include <cuda.h>

// Device-side PTX wrappers (`tma_load_2d`, `mbarrier_*`, ...) used by the
// device-side TMA load helpers below.  Safe to include from any
// nvcc-compiled TU (pulls in `<cuda/pipeline>`, `<cuda_bf16.h>`, etc.);
// not safe from pure host `.cpp` TUs.  All current includers of this
// header (`moe_tma.cu`, `tma_descriptor_factory_test.cu`, and the device
// TUs that will consume the loaders) are CUDA-compiled.
#include "ptx_utils.h"

// Build-time guard: TMA descriptor encoding requires CUDA toolkit 12.0+
// (see requirement R12.1).  The CUDA Driver API exposes `CUtensorMap` only
// on 12.0+, so fail fast with a clear message on older toolchains.
#if defined(CUDA_VERSION) && (CUDA_VERSION < 12000)
  #error \
      "moe_tma.h requires CUDA 12.0 or later for CUtensorMap / cuTensorMapEncodeTiled"
#endif

namespace moe_monokernel {

/**
 * @brief Build a `CUtensorMap` describing the fp8 up-projection weight
 *        tensor `[E, 2*N, K]` for TMA tile loads in Phase 3.
 *
 * The returned descriptor targets `CU_TENSOR_MAP_DATA_TYPE_UINT8` with
 * `rank = 2`, `globalDim = [K, num_experts * 2 * N]`, `boxDim = [128, 32]`,
 * `elementStrides = [1, 1]`, `CU_TENSOR_MAP_INTERLEAVE_NONE`,
 * `CU_TENSOR_MAP_SWIZZLE_128B`, `CU_TENSOR_MAP_L2_PROMOTION_L2_128B`, and
 * `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`.  See requirements R5.1, R5.2, R1.4.
 *
 * NOTE: `SWIZZLE_128B` permutes bytes within each 128-byte row during the
 * TMA write.  The WGMMA A-descriptor must use `swizzle=3` to de-swizzle
 * on read.  This is the standard CUTLASS SM90 fp8 GEMM pattern.
 *
 * @param weights_ptr   Device pointer to the base of `expert_weights_up`
 *                      (fp8 e4m3 values, row-major `[E, 2*N, K]`).
 * @param num_experts   Number of experts (`E`).
 * @param N             Half of the fused gate+up intermediate size (so the
 *                      flattened row axis has length `num_experts * 2 * N`).
 * @param K             Hidden size / reduction dimension.
 * @return A 128-byte POD `CUtensorMap`, safe to pass by value as a
 *         `__grid_constant__ CUtensorMap const` kernel parameter.
 *
 * On `cuTensorMapEncodeTiled` failure, raises `TORCH_CHECK` identifying
 * "up-projection weights" as the failing tensor (R5.5).
 */
CUtensorMap create_up_weight_tma_desc(const void* weights_ptr,
                                      uint32_t num_experts, uint32_t N,
                                      uint32_t K);

/**
 * @brief Build a `CUtensorMap` describing the bf16 activation tensor
 *        `[BS, K_hidden]` for TMA tile loads in Phase 3.
 *
 * The returned descriptor targets `CU_TENSOR_MAP_DATA_TYPE_BFLOAT16` with
 * `rank = 2`, `globalDim = [K_hidden, batch_size_cap]`,
 * `boxDim = [128, 8]`, `elementStrides = [1, 1]`,
 * `CU_TENSOR_MAP_INTERLEAVE_NONE`, `CU_TENSOR_MAP_SWIZZLE_NONE`,
 * `CU_TENSOR_MAP_L2_PROMOTION_L2_128B`, and
 * `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`.  See requirements R5.3, R5.4, R2.4,
 * R12.4.
 *
 * @param activations_ptr Device pointer to the base of `activations_in`
 *                        (bf16 values, row-major `[BS, K_hidden]`).
 * @param batch_size_cap  Maximum batch size the kernel will process
 *                        (used as the outer axis of `globalDim`).
 * @param K_hidden        Hidden size of the activation tensor.
 * @return A 128-byte POD `CUtensorMap`, safe to pass by value as a
 *         `__grid_constant__ CUtensorMap const` kernel parameter.
 *
 * On `cuTensorMapEncodeTiled` failure, raises `TORCH_CHECK` identifying
 * "activations" as the failing tensor (R5.5).
 */
CUtensorMap create_activations_tma_desc(const void* activations_ptr,
                                        uint32_t batch_size_cap,
                                        uint32_t K_hidden);

/**
 * @brief Build a `CUtensorMap` describing the fp8 down-projection weight
 *        tensor `[E, K, N]` for TMA tile loads in Phase 4.
 *
 * The returned descriptor targets `CU_TENSOR_MAP_DATA_TYPE_UINT8` with
 * `rank = 2`, `globalDim = [N, num_experts * K]`, `boxDim = [128, 128]`,
 * `elementStrides = [1, 1]`, `CU_TENSOR_MAP_INTERLEAVE_NONE`,
 * `CU_TENSOR_MAP_SWIZZLE_NONE`, `CU_TENSOR_MAP_L2_PROMOTION_L2_128B`, and
 * `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`.  See requirements R1.4, R1.7, R6.1,
 * R6.2.
 *
 * NOTE on axis ordering: unlike the up-projection weight descriptor (where
 * `K` is the innermost reduction axis and the outer axis is
 * `num_experts * 2 * N` row-major rows), the down-projection descriptor
 * flips this — here `N` is the innermost reduction axis (the TMA tile
 * covers 128 N-elements per row) and the outer axis is the flattened
 * `expert_id * K + k_row` row index.  This matches the down-proj's GM
 * layout `[E, K, N]` where each expert's `K` output rows are each a
 * contiguous `N`-element fp8 vector, and the down-proj's WGMMA consumes
 * 128 N-elements at a time as the reduction dimension.
 *
 * Down-proj uses `SWIZZLE_NONE` (not `SWIZZLE_128B` like up-proj) per the
 * Stage-1 design choice (see requirements.md scope item 6).  Matching the
 * SHM layout for the WGMMA A-descriptor (`LBO=128, SBO=1024, swizzle=0`)
 * is the responsibility of the Python pre-interleave helper
 * `interleave_for_tma_wgmma_down` (R8.x).
 *
 * @param weights_ptr   Device pointer to the base of the pre-interleaved
 *                      `expert_weights_down` (fp8 e4m3 values, row-major
 *                      `[E, K, N]`).
 * @param num_experts   Number of experts (`E`).
 * @param K             Down-projection output dimension (= hidden size).
 * @param N             Down-projection reduction dimension (= up-proj
 *                      intermediate output size).
 * @return A 128-byte POD `CUtensorMap`, safe to pass by value as a
 *         `__grid_constant__ CUtensorMap const` kernel parameter.
 *
 * On `cuTensorMapEncodeTiled` failure, raises `TORCH_CHECK` identifying
 * "down-projection weights" as the failing tensor (R6.5).
 */
CUtensorMap create_down_weight_tma_desc(const void* weights_ptr,
                                        uint32_t num_experts, uint32_t K,
                                        uint32_t N);

/**
 * @brief Build a `CUtensorMap` describing the fp8 intermediate-activation
 *        tensor `spec->temp_fp8[TEMP_ROWS, N]` for TMA tile loads in
 *        Phase 4.
 *
 * The returned descriptor targets `CU_TENSOR_MAP_DATA_TYPE_UINT8` with
 * `rank = 2`, `globalDim = [N, temp_rows]`, `boxDim = [128, 8]`,
 * `elementStrides = [1, 1]`, `CU_TENSOR_MAP_INTERLEAVE_NONE`,
 * `CU_TENSOR_MAP_SWIZZLE_NONE`, `CU_TENSOR_MAP_L2_PROMOTION_L2_128B`, and
 * `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`.  See requirements R2.4, R2.8, R6.3,
 * R6.4.
 *
 * Axis ordering: innermost = `N` (128 fp8 K-values per token row),
 * outer = `sorted_slot` row index into `spec->temp_fp8`.  `boxDim[1] = 8`
 * sets the maximum rows per bulk TMA issue; the actual per-issue row
 * count is controlled at runtime by the instruction's dynamic row-count
 * operand and equals `routed_token_count ∈ [1, 8]` for the current
 * expert (R2.2, R2.5).
 *
 * The `temp_fp8` tensor is written by the Phase-3 up-projection epilogue
 * into a reorganized `[expert, token]` layout (R11.x), so each expert's
 * routed tokens occupy a contiguous slab of rows
 * `[expert_slot_start[id], expert_slot_start[id] + routed_token_count[id])`
 * that this descriptor can fetch in a single bulk TMA.
 *
 * @param activations_ptr Device pointer to the base of `spec->temp_fp8`
 *                        (fp8 e4m3 values, row-major `[TEMP_ROWS, N]`).
 * @param temp_rows       Outer-axis length `Dims::BS * MAX_TOPK` (e.g. 64
 *                        for `BS=8, MAX_TOPK=8`).
 * @param N               Down-projection reduction dimension.
 * @return A 128-byte POD `CUtensorMap`, safe to pass by value as a
 *         `__grid_constant__ CUtensorMap const` kernel parameter.
 *
 * On `cuTensorMapEncodeTiled` failure, raises `TORCH_CHECK` identifying
 * "down-projection activations" as the failing tensor (R6.5).
 */
CUtensorMap create_down_activation_tma_desc(const void* activations_ptr,
                                            uint32_t temp_rows, uint32_t N);

// ─── Device-side TMA load helpers ────────────────────────────────────────
//
// Thin wrappers around the low-level PTX TMA primitive `tma_load_2d`
// (in `ptx_utils.h`) that bake in the MoE up-projection path's coordinate
// convention and per-stripe semantics.  All helpers below are `__device__
// __forceinline__` and are expected to inline fully into the kernel's TMA
// launcher thread.
//
// ── Caller contract (critical) ──
// Every helper in this group MUST be invoked by exactly one thread in the
// block — the "TMA launcher thread" (warp 8, lane 0 in the BS8 WGMMA path,
// selected via `is_tma_launcher_thread<Dims>()`).  These helpers do NOT
// internally gate on `threadIdx`; calling them from multiple threads will
// issue duplicate `cp.async.bulk.tensor.2d` instructions and corrupt the
// barrier's transaction-bytes accounting.  The caller is also responsible
// for pre-arming the target mbarrier exactly once with the total expected
// byte count before issuing the TMA(s) that target it — see R3.3, R3.4.

/**
 * @brief Issue one TMA sub-tile load for a single 32-row stripe of the
 *        fp8 up-projection weight tile.
 *
 * This is a direct device-side specialization of `tma_load_2d` for the
 * weight descriptor built by `create_up_weight_tma_desc`.  A single call
 * fetches one 128-K × 32-row rectangular box from the flattened
 * `[num_experts * 2 * N, K]` weight tensor into a 16-B aligned SHM region
 * (4096 B per stripe).  Four back-to-back calls — one per stripe
 * (gate-WG0, up-WG0, gate-WG1, up-WG1) — cover the full 16 KB WGMMA
 * weight tile; they all share a single mbarrier that the caller must
 * pre-arm once with `expect_tx = 16384` (see R1.2, R1.3).
 *
 * NOTE: The weight descriptor uses `CU_TENSOR_MAP_SWIZZLE_128B`, which
 * permutes bytes within each 128-byte row during the TMA write to SHM.
 * The WGMMA A-descriptor must use `swizzle=3` to apply the inverse
 * permutation on read, so the matmul sees correct logical values.  This
 * is the standard CUTLASS SM90 fp8 GEMM pattern for 128-byte-wide rows.
 *
 * Coordinate convention for the weight descriptor: the innermost axis is
 * K and the outer axis is the flattened row index
 * `expert_id * 2 * N + row_in_expert` (R1.4, R1.5).  Per
 * `cuTensorMapEncodeTiled`, the coordinate operand order matches that of
 * `globalDim`, so `coord0 = k_start` and `coord1 = global_row`.  This
 * wrapper enforces that mapping at the PTX boundary — the caller passes
 * `global_row` and `k_start` in the natural "rows, then K" order.
 *
 * Caller contract (R1.1, R4.2):
 *   - Must be called by exactly ONE thread per block (the TMA launcher).
 *   - `bar_smem_ptr` must have been pre-armed once with
 *     `mbarrier_arrive_expect_tx(bar_smem_ptr, 16384)` before the first
 *     of the four sub-tile issues targeting this same barrier.  This
 *     function does NOT call `mbarrier_arrive_expect_tx`.
 *   - `dest_smem_ptr` must be 16-B aligned and point at the 4096-B stripe
 *     region (SHM offset 0, 4096, 8192, or 12288 inside the slot).
 *   - `desc` must be the descriptor produced by
 *     `create_up_weight_tma_desc`, typically passed to the kernel as a
 *     `__grid_constant__ CUtensorMap const` parameter.
 *
 * @param desc          Host-built weight TMA descriptor
 *                      (`__grid_constant__`).
 * @param global_row    Outer-axis global row coordinate, flattened as
 *                      `expert_id * 2 * N + row_in_expert` (R1.5).
 * @param k_start       Innermost-axis starting K column (multiple of 128).
 * @param dest_smem_ptr 16-B aligned SHM destination pointer for this
 *                      32-row stripe (4096 B).
 * @param bar_smem_ptr  16-B aligned SHM mbarrier pre-armed by the caller.
 */
__device__ __forceinline__ void tma_load_up_wgmma_subtile(
    CUtensorMap const& desc, std::uint32_t global_row, std::uint32_t k_start,
    void* dest_smem_ptr, std::uint64_t* bar_smem_ptr) {
  // Descriptor axis order (innermost first): coord0 = K, coord1 = row.
  tma_load_2d(desc, /*coord0=*/k_start, /*coord1=*/global_row, dest_smem_ptr,
              bar_smem_ptr);
}

/**
 * @brief Issue all four TMA sub-tile loads that make up one full 128×128
 *        fp8 up-projection WGMMA weight tile.
 *
 * This is the composite helper used inside the Phase-3 K-loop: a single
 * call populates the complete 16 KB weight tile at SHM slot
 * `shm->w_wgmma[slot]` for one `(expert_id, base_row_up, k_start)` tuple
 * by emitting four back-to-back `tma_load_up_wgmma_subtile` calls, one
 * per 32-row stripe.  All four sub-tiles target the SAME mbarrier
 * (`bar_smem_ptr`), whose transaction-bytes counter is decremented
 * independently by each sub-tile as its bytes land.
 *
 * NOTE: The weight descriptor uses `CU_TENSOR_MAP_SWIZZLE_128B` (not
 * `SWIZZLE_NONE` as originally designed in Stage-1).  The swizzle
 * permutes bytes within each 128-byte row during the TMA write.  Since
 * each fp8 weight row is exactly 128 bytes (128 K × 1 B/fp8), the
 * swizzle operates within a single row.  The WGMMA A-descriptor must
 * use `swizzle=3` to apply the inverse permutation on read, so the
 * matmul sees correct logical values.  The mathematical result is
 * identical to the cp.async path — only the SHM byte arrangement differs.
 *
 * SHM stripe layout produced (R1.5, R1.6, matches the hand-written
 * `moe_load_up_wgmma_tile_128x128` `cp.async` loader byte-for-byte):
 *
 *   | SHM rows   | SHM offset | Stripe   | Global row |
 *   |------------|-----------:|----------|--------------------------------------|
 *   | [ 0 ..  31]|        0 B | gate-WG0 | expert_id*2*N + base_row_up | | [32
 * ..  63]|     4096 B | up-WG0   | expert_id*2*N + base_row_up + N      | | [64
 * ..  95]|     8192 B | gate-WG1 | expert_id*2*N + base_row_up + 32     | | [96
 * .. 127]|    12288 B | up-WG1   | expert_id*2*N + base_row_up + 32 + N |
 *
 * The four issues share one barrier per spec design (Option B, strategy
 * 2): the TMA engine accumulates per-sub-tile `complete_tx(4096)` into
 * the single barrier, which flips parity exactly once after all 16 384
 * bytes have landed.
 *
 * Caller contract (critical — R1.2, R1.3, R3.3, R4.2):
 *   - Must be called by exactly ONE thread per block (the TMA launcher,
 *     typically warp 8 lane 0).  This function does NOT gate on
 *     `threadIdx`.
 *   - `bar_smem_ptr` MUST have been pre-armed EXACTLY ONCE with
 *     `mbarrier_arrive_expect_tx(bar_smem_ptr, 16384)` BEFORE calling
 *     this function.  This function itself does NOT call
 *     `mbarrier_arrive_expect_tx`; it only issues the four TMA loads.
 *   - `dest_slot_smem_ptr` MUST be 16 KB aligned and point at the base
 *     of `shm->w_wgmma[slot][0][0]` (the 16 384-B weight-tile slot).
 *     Treated internally as `uint8_t*` for per-stripe offset arithmetic
 *     (+4096, +8192, +12288).
 *   - `desc` MUST be the descriptor produced by
 *     `create_up_weight_tma_desc`, typically passed to the kernel as a
 *     `__grid_constant__ CUtensorMap const` parameter.
 *   - Valid inputs (R1.6): `0 ≤ expert_id < num_experts`,
 *     `0 ≤ base_row_up + 64 ≤ N`, `0 ≤ k_start` with
 *     `k_start + 128 ≤ K`.
 *
 * @param desc               Host-built weight TMA descriptor
 *                           (`__grid_constant__`).
 * @param expert_id          Expert index (`0 ≤ expert_id < num_experts`).
 * @param N                  Half of the fused gate+up intermediate size;
 *                           gate rows occupy `[0, N)` and up rows occupy
 *                           `[N, 2N)` inside each expert's row block.
 * @param base_row_up        First WG0-gate row inside the expert (R1.5).
 * @param k_start            Innermost-axis starting K column (multiple
 *                           of 128).
 * @param dest_slot_smem_ptr 16 KB aligned SHM base of
 *                           `shm->w_wgmma[slot][0][0]`.  Internally
 *                           treated as `uint8_t*` for the +4096, +8192,
 *                           +12288 per-stripe offsets.
 * @param bar_smem_ptr       16-B aligned SHM mbarrier pre-armed by the
 *                           caller with `expect_tx = 16384`.
 */
__device__ __forceinline__ void tma_load_up_wgmma_tile_full(
    CUtensorMap const& desc, std::uint32_t expert_id, std::uint32_t N,
    std::uint32_t base_row_up, std::uint32_t k_start, void* dest_slot_smem_ptr,
    std::uint64_t* bar_smem_ptr) {
  // Flattened outer-axis base for this expert in the
  // `[num_experts * 2 * N, K]` descriptor view (R1.5).
  const std::uint32_t expert_row_base = expert_id * 2u * N + base_row_up;

  // Byte-addressable view of the 16 KB weight-tile slot for the four
  // per-stripe offsets (+0, +4096, +8192, +12288).  `dest_slot_smem_ptr`
  // is assumed 16 KB aligned (base of `shm->w_wgmma[slot][0][0]`); the
  // per-stripe pointers it yields are each 16-B aligned as required by
  // `tma_load_2d`.
  auto* dst_u8 = reinterpret_cast<std::uint8_t*>(dest_slot_smem_ptr);

  // Stripe 0 — gate-WG0: SHM rows [0..31], SHM offset 0.
  tma_load_up_wgmma_subtile(desc,
                            /*global_row=*/expert_row_base,
                            /*k_start=*/k_start,
                            /*dest_smem_ptr=*/dst_u8 + 0,
                            /*bar_smem_ptr=*/bar_smem_ptr);

  // Stripe 1 — up-WG0: SHM rows [32..63], SHM offset 4096 B.
  tma_load_up_wgmma_subtile(desc,
                            /*global_row=*/expert_row_base + N,
                            /*k_start=*/k_start,
                            /*dest_smem_ptr=*/dst_u8 + 4096,
                            /*bar_smem_ptr=*/bar_smem_ptr);

  // Stripe 2 — gate-WG1: SHM rows [64..95], SHM offset 8192 B.
  tma_load_up_wgmma_subtile(desc,
                            /*global_row=*/expert_row_base + 32u,
                            /*k_start=*/k_start,
                            /*dest_smem_ptr=*/dst_u8 + 8192,
                            /*bar_smem_ptr=*/bar_smem_ptr);

  // Stripe 3 — up-WG1: SHM rows [96..127], SHM offset 12288 B.
  tma_load_up_wgmma_subtile(desc,
                            /*global_row=*/expert_row_base + 32u + N,
                            /*k_start=*/k_start,
                            /*dest_smem_ptr=*/dst_u8 + 12288,
                            /*bar_smem_ptr=*/bar_smem_ptr);
}

/**
 * @brief Issue one TMA load for the bf16 activation tile used by Phase-3
 *        streaming quantize.
 *
 * This is the TMA-path counterpart of the existing `cp.async`-based
 * `moe_load_bf16_input_tile` in `moe_up_projection.cu`.  A single call
 * fetches one 128-K × 8-token rectangular box from the full
 * `[batch_size_cap, K_hidden]` activation tensor into a 16-B aligned SHM
 * region (2048 B total: 8 tokens × 128 K × 2 B per bf16) using exactly
 * one `cp.async.bulk.tensor.2d` instruction (R2.1, R2.2).
 *
 * Coordinate convention for the activation descriptor (built by
 * `create_activations_tma_desc`): the innermost axis is K and the outer
 * axis is the token index.  Per spec R2.5, every K-step `s` fetches all
 * 8 tokens starting at token 0, so the tile coordinates are
 * `(k_start, 0) = (s * K_STEP, 0)`.  Per `cuTensorMapEncodeTiled`, the
 * coordinate operand order matches that of `globalDim`, so this wrapper
 * passes `coord0 = k_start` and `coord1 = 0` at the PTX boundary.
 *
 * Activations are expert-invariant (R15.2): the coordinates depend only
 * on `k_start` and are the same across the outer expert loop.
 *
 * Caller contract (critical — R2.1, R2.3, R3.4, R4.2):
 *   - Must be called by exactly ONE thread per block (the TMA launcher,
 *     typically warp 8 lane 0).  This function does NOT gate on
 *     `threadIdx`; calling from multiple threads issues duplicate TMAs
 *     and corrupts the barrier's transaction-bytes accounting.
 *   - `bar_smem_ptr` MUST have been pre-armed EXACTLY ONCE with
 *     `mbarrier_arrive_expect_tx(bar_smem_ptr, 2048)` BEFORE calling
 *     this function.  This function itself does NOT call
 *     `mbarrier_arrive_expect_tx`; it only issues the TMA load.
 *   - `dest_smem_ptr` MUST be 16-B aligned and point at
 *     `shm->bf16_in[slot][0][0]` (the 2048-B activation-tile slot).
 *   - `desc` MUST be the descriptor produced by
 *     `create_activations_tma_desc`, typically passed to the kernel as
 *     a `__grid_constant__ CUtensorMap const` parameter.
 *   - Valid inputs (R15.2): `0 ≤ k_start` with `k_start + 128 ≤ K_hidden`.
 *
 * @param desc          Host-built activation TMA descriptor
 *                      (`__grid_constant__`).
 * @param k_start       Innermost-axis starting K column (multiple of 128).
 * @param dest_smem_ptr 16-B aligned SHM destination pointer to
 *                      `shm->bf16_in[slot][0][0]` (2048 B).
 * @param bar_smem_ptr  16-B aligned SHM mbarrier pre-armed by the caller
 *                      with `expect_tx = 2048`.
 */
__device__ __forceinline__ void tma_load_bf16_input_tile(
    CUtensorMap const& desc, std::uint32_t k_start, void* dest_smem_ptr,
    std::uint64_t* bar_smem_ptr) {
  // Descriptor axis order (innermost first): coord0 = K, coord1 = token.
  // We always fetch all 8 tokens starting at token 0 (R2.5, R15.2).
  tma_load_2d(desc, /*coord0=*/k_start, /*coord1=*/0u, dest_smem_ptr,
              bar_smem_ptr);
}

// ─── Down-projection (Phase 4) TMA load helpers ──────────────────────────
//
// Mirror of the up-projection helpers above, specialized for the fp8
// down-projection weight and intermediate-activation descriptors built by
// `create_down_weight_tma_desc` and `create_down_activation_tma_desc`.
//
// Key differences vs the up-proj helpers:
//   * Weight descriptor axis order is (innermost=N, outer=expert*K + row)
//     instead of (innermost=K, outer=expert*2*N + row).  That is, the
//     down-proj's reduction dimension N is the innermost (K-for-WGMMA)
//     axis, and its output dimension K is the outer row axis.
//   * Weight descriptor uses `SWIZZLE_NONE`, not `SWIZZLE_128B`.  The
//     canonical WGMMA Major::K core-matrix permutation is applied
//     host-side (model load time) by `interleave_for_tma_wgmma_down`,
//     so the device-side TMA write lands in the exact byte pattern the
//     WGMMA A-descriptor (`A_LBO=128, A_SBO=1024, swizzle=0`) expects.
//   * The 128×128 weight tile is fetched in ONE TMA issue (16 384 B =
//     `boxDim = (128, 128)`), not four 32-row sub-tiles: no need to
//     split because `SWIZZLE_NONE` + pre-interleave already produces
//     the correct layout and the box size fits in one issue.
//   * The activation descriptor uses `boxDim = (128, 8)`; each issue
//     fetches up to 8 contiguous rows starting at `expert_slot_start[id]`
//     in the Phase-3-reorganized `temp_fp8` layout (R11.x).
//
// Caller contract (same as up-proj helpers): every helper MUST be invoked
// by exactly ONE thread per block — the TMA launcher thread (warp 8,
// lane 0 on the BS8 WGMMA_TMA path, selected via
// `is_tma_launcher_thread<Dims>()`).  They do NOT internally gate on
// `threadIdx` and the caller is responsible for pre-arming the target
// mbarrier with the total expected byte count before issuing the TMA(s).

/**
 * @brief Issue one TMA load for the full 128×128 fp8 down-projection
 *        weight tile.
 *
 * A single `cp.async.bulk.tensor.2d` issue fetches one `(128 N, 128 K)`
 * box (16 384 bytes) from the descriptor built by
 * `create_down_weight_tma_desc` into the caller-supplied 16 KB SHM slot.
 * The weight tensor must have been pre-interleaved by
 * `interleave_for_tma_wgmma_down` in Python (R8.x) so that the TMA's
 * row-major byte write lands in canonical WGMMA Major::K form
 * (`A_LBO = 128`, `A_SBO = 1024`, `swizzle = 0`).
 *
 * Coordinate convention for the down-weight descriptor: innermost axis
 * is N and outer axis is the flattened row index
 * `expert_id * K + output_row` (R1.5).  Per `cuTensorMapEncodeTiled` the
 * coordinate operand order matches `globalDim`, so this wrapper emits
 * `coord0 = k_start` (innermost, the starting N column of the tile) and
 * `coord1 = expert_id * K + base_col` (outer, the starting output row).
 *
 * NOTE on parameter names: the outer axis is here called `base_col`
 * because it represents a column in the down-projection's output
 * (= hidden-size K of the layer).  Likewise `k_start` is a column in
 * the reduction dimension N.  The kernel-level spec uses these names
 * consistently with the mathematical roles of the dimensions.
 *
 * Caller contract (R1.1, R1.2, R1.3, R4.2, R7.1, R7.4):
 *   - Must be called by exactly ONE thread per block (the TMA launcher,
 *     typically warp 8 lane 0).  This function does NOT gate on
 *     `threadIdx`.
 *   - `bar_smem_ptr` MUST have been pre-armed EXACTLY ONCE with
 *     `mbarrier_arrive_expect_tx(bar_smem_ptr, 16384)` BEFORE calling
 *     this function.  This function itself does NOT call
 *     `mbarrier_arrive_expect_tx`; it only issues the single TMA load.
 *   - `dest_smem_ptr` MUST be 16 KB aligned and point at the base of
 *     `shm->w_down_wgmma[slot][0][0]` (the 16 384-B weight-tile slot).
 *   - `desc` MUST be the descriptor produced by
 *     `create_down_weight_tma_desc`, typically passed to the kernel as
 *     a `__grid_constant__ CUtensorMap const` parameter.
 *   - Valid inputs (R1.6): `0 ≤ expert_id < num_experts`,
 *     `0 ≤ base_col` with `base_col + 128 ≤ K`, and `0 ≤ k_start` with
 *     `k_start + 128 ≤ N`.
 *
 * SHM byte layout produced (R7.5): canonical WGMMA Major::K
 *   `byte_off(m, k) = (m/8)*1024 + (k/16)*128 + (m%8)*16 + (k%16)`
 * byte-for-byte equal to the SHM produced by the existing
 * `moe_load_down_wgmma_weight_tile` `cp.async` loader on the
 * un-interleaved weight tensor (R1.6 / R12.2).
 *
 * @param desc          Host-built down-weight TMA descriptor
 *                      (`__grid_constant__`).
 * @param expert_id     Expert index (`0 ≤ expert_id < num_experts`).
 * @param K             Down-projection output dimension (= hidden size).
 *                      Used to flatten the outer-axis coordinate as
 *                      `expert_id * K + base_col`.
 * @param base_col      First output row of the 128-row tile (multiple
 *                      of 128).
 * @param k_start       First reduction-dim column of the tile (multiple
 *                      of 128).
 * @param dest_smem_ptr 16 KB aligned SHM destination pointer to
 *                      `shm->w_down_wgmma[slot][0][0]` (16 384 B).
 * @param bar_smem_ptr  16-B aligned SHM mbarrier pre-armed by the caller
 *                      with `expect_tx = 16384`.
 */
__device__ __forceinline__ void tma_load_down_wgmma_tile(
    CUtensorMap const& desc, std::uint32_t expert_id, std::uint32_t K,
    std::uint32_t base_col, std::uint32_t k_start, void* dest_smem_ptr,
    std::uint64_t* bar_smem_ptr) {
  // Descriptor axis order (innermost first): coord0 = N, coord1 = row.
  // Flattened outer-axis row for this expert: `expert_id * K + base_col`
  // (R1.5).
  const std::uint32_t global_row = expert_id * K + base_col;
  tma_load_2d(desc, /*coord0=*/k_start, /*coord1=*/global_row, dest_smem_ptr,
              bar_smem_ptr);
}

/**
 * @brief Issue one bulk TMA load for up to 8 contiguous rows of the fp8
 *        intermediate-activation tile consumed by the Phase-4 WGMMA
 *        down-projection.
 *
 * A single `cp.async.bulk.tensor.2d` issue fetches one
 * `(128 N, boxDim[1] = 8)` box (1 KB maximum) from the descriptor built
 * by `create_down_activation_tma_desc` into the caller-supplied 1 KB
 * SHM slot.  The fetched rows start at `expert_slot_start` in the
 * Phase-3-reorganized `spec->temp_fp8` layout (R11.x), where each
 * expert's routed tokens occupy a contiguous slab
 * `[expert_slot_start[id], expert_slot_start[id] + routed_token_count[id])`.
 *
 * Since `boxDim[1] = 8` is baked into the descriptor, this issue always
 * fetches the full 8-row × 128-N box (1024 bytes).  When the expert's
 * `routed_token_count < 8`, the "unused" rows
 * `[routed_token_count, 8)` in the SHM slot must be overwritten with
 * zero-valued fp8 bytes before the WGMMA consumes the tile; that
 * responsibility lives in `zero_fill_unused_down_act_slots` below and
 * is ordered by the `__syncthreads()` at the end of each Phase-4 K-step.
 *
 * Coordinate convention for the down-activation descriptor: innermost
 * axis is N and outer axis is the `sorted_slot` row index into
 * `spec->temp_fp8`.  Per `cuTensorMapEncodeTiled` the coordinate operand
 * order matches `globalDim`, so this wrapper emits
 * `coord0 = k_start` (innermost, starting N column) and
 * `coord1 = expert_slot_start` (outer, first reorganized row for this
 * expert).
 *
 * Caller contract (R2.1, R2.2, R2.3, R2.5, R4.3, R7.3, R7.4, R7.6):
 *   - Must be called by exactly ONE thread per block (the TMA launcher).
 *     This function does NOT gate on `threadIdx`.
 *   - MUST only be invoked when `routed_token_count > 0` (R2.3).  When
 *     no tokens route to this expert, the caller SHALL neither arm
 *     `bar_smem_ptr` nor call this helper; the SHM slot is instead
 *     zero-filled entirely by `zero_fill_unused_down_act_slots`.
 *   - `bar_smem_ptr` MUST have been pre-armed EXACTLY ONCE with
 *     `mbarrier_arrive_expect_tx(bar_smem_ptr, 128 * routed_token_count)`
 *     BEFORE calling this function.  Each of the 8 K-chunks per row is
 *     16 fp8 bytes, so one row contributes `8 * 16 = 128` transaction
 *     bytes.  For the maximum 8-token issue this comes to 1024 B.
 *   - `dest_smem_ptr` MUST be 16-B aligned and point at
 *     `&shm->a_down_wgmma[slot][0][0][0]` (the 1024-B activation-tile
 *     slot).
 *   - `desc` MUST be the descriptor produced by
 *     `create_down_activation_tma_desc`, typically passed to the kernel
 *     as a `__grid_constant__ CUtensorMap const` parameter.
 *   - Valid inputs: `0 ≤ k_start` with `k_start + 128 ≤ N`;
 *     `expert_slot_start + 8 ≤ TEMP_ROWS` (`boxDim[1]` always fetches
 *     8 rows, so the outer coordinate must leave room for 8 rows of
 *     the descriptor view).
 *
 * SHM byte layout produced (R7.6, R12.8): the helper issues 8 TMAs
 * (one per K-chunk kc in [0, 8)) each writing a 16-col x 8-row block
 * into the disjoint slab at &a_down_wgmma[slot][kc][0][0]. This
 * matches the WGMMA B-operand's expected layout kc*128 + tok*16 + ki
 * byte-for-byte with the cp.async loader's output on the un-reorganized
 * path (R12.8).
 *
 * @param desc               Host-built down-activation TMA descriptor
 *                           (`__grid_constant__`, boxDim=(16, 8)).
 * @param k_start            First reduction-dim column of the K-step
 *                           (multiple of 128).
 * @param expert_slot_start  First reorganized row of the current
 *                           expert's contiguous slab in `temp_fp8`
 *                           (R2.5).
 * @param dest_smem_ptr      16-B aligned SHM destination pointer to
 *                           `&shm->a_down_wgmma[slot][0][0][0]` (1024 B).
 * @param bar_smem_ptr       16-B aligned SHM mbarrier pre-armed by the
 *                           caller with `expect_tx = 1024` (full slot,
 *                           the 8 TMAs collectively deliver 1024 B).
 */
__device__ __forceinline__ void tma_load_down_wgmma_activation_bulk(
    CUtensorMap const& desc, std::uint32_t k_start,
    std::uint32_t expert_slot_start, void* dest_smem_ptr,
    std::uint64_t* bar_smem_ptr) {
  // K-chunk-major descriptor axes (innermost first):
  //   coord0 = per-kc-slab byte offset of the starting row
  //   coord1 = starting K-chunk index (k_start / 16)
  //
  // One TMA issue (boxDim=(128, 8)) fetches the full 8-K-chunk x 8-token
  // 1024-byte slab and writes it row-major into SHM at dest_smem_ptr.
  // The row-major write produces byte offset row*128 + col, which maps
  // directly to a_down_wgmma[kc=row][tok=col/16][ki=col%16] — the
  // WGMMA B-operand's canonical K-major layout for m64n8k32 with N=8.
  tma_load_2d(desc,
              /*coord0=*/expert_slot_start * 16u,
              /*coord1=*/k_start / 16u, dest_smem_ptr, bar_smem_ptr);
}

/**
 * @brief Cooperatively zero-fill the unused token rows of a down-proj
 *        activation SHM slot.
 *
 * On the Phase-4 TMA path the down-activation loader
 * (`tma_load_down_wgmma_activation_bulk`) fills the first
 * `routed_count` rows of an 8-row × 8-K-chunk × 16-byte SHM slot
 * (`shm->a_down_wgmma[slot][kc][tok][0..15]`) with the routed tokens'
 * fp8 K-chunks.  The remaining rows `[routed_count, 8)` are left with
 * TMA-fetched bytes beyond the expert's own slab (or undefined data if
 * the expert sits at the tail of `temp_fp8`).  This helper overwrites
 * those tail rows with zero bytes so that the subsequent WGMMA's reads
 * contribute zero to the accumulator — exactly matching the `cp.async`
 * loader's zero-fill behavior for the not-routed case (R2.6, R7.3,
 * R12.8).
 *
 * Thread mapping (R5.4): cooperative across warp 8, lanes 1..31
 * (lane 0 is the TMA launcher and is busy issuing the bulk TMA).  The
 * 31 participating lanes each stride through the unused region with
 * 4-byte (uint32) stores, covering at most
 *   `(8 - routed_count) * 8 kc * 16 B = (8 - routed_count) * 128 B`
 * bytes = at most 896 B when `routed_count == 1`, i.e. at most
 * `896 / 4 = 224` 4-byte stores distributed over 31 lanes
 * (~8 iterations per lane).
 *
 * Ordering / visibility (R2.6, R5.4): the zero-fill is NOT synchronized
 * against the TMA completion here — the caller is expected to issue a
 * `__syncthreads()` at the end of the Phase-4 K-step iteration before
 * the next iteration's WGMMA consumers wait on `bar_a[slot]`.  That
 * syncthreads() both makes the zero-fill visible across the block and
 * orders it before the next use of the slot.  This helper therefore
 * does NOT itself emit any fence or sync — it is a pure set of
 * per-lane SHM stores.
 *
 * SHM layout assumed: `shm->a_down_wgmma[slot]` has shape
 *   `[FP8_ACT_NUM_CHUNKS = 8][T_TILE = 8][FP8_ACT_K_CHUNK = 16]`
 * with row-major byte stride `(kc, tok, kb) → kc*128 + tok*16 + kb`.
 * The unused region is the sub-cube `(kc, tok, kb)` with
 * `kc ∈ [0, 8)`, `tok ∈ [routed_count, 8)`, `kb ∈ [0, 16)` — which is
 * a non-contiguous region along `tok`, so the loop body computes the
 * per-element address from `(kc, tok, kb_dword)` triplets rather than
 * iterating a flat byte index.
 *
 * Caller contract:
 *   - Must be called by every thread in warp 8 (lanes 0..31).  Lane 0
 *     performs no stores and is expected to be issuing the bulk TMA
 *     in parallel; lanes 1..31 perform the actual zero-fill stores.
 *     Inside the function, the gate is `lane_id != 0`, so it is safe
 *     to invoke uniformly from warp 8.
 *   - `0 ≤ routed_count ≤ 8`.  When `routed_count == 8` the helper is
 *     a no-op and may also be skipped by the caller.  When
 *     `routed_count == 0` the helper zero-fills the full 1 KB slot
 *     (8 tokens × 8 kc × 16 B).
 *   - `a_down_wgmma_slot_base_ptr` MUST be 4-byte aligned and point at
 *     `&shm->a_down_wgmma[slot][0][0][0]` (the base of one 1 KB slot).
 *
 * Template parameter `Dims` is unused by the body but retained so the
 * helper reads symmetrically with other `Dims`-templated helpers in
 * this file and so future Dims-dependent constants (e.g. alternative
 * `T_TILE` / `FP8_ACT_NUM_CHUNKS` values) can be wired in without
 * changing call sites.
 *
 * @tparam Dims                        The MoE `Dims` variant (unused
 *                                     today, see note above).
 * @param  routed_count                Number of routed tokens for the
 *                                     current expert at this slot,
 *                                     in `[0, 8]`.
 * @param  a_down_wgmma_slot_base_ptr  4-B aligned SHM base pointer of
 *                                     one `a_down_wgmma[slot]` slice,
 *                                     i.e. `&shm->a_down_wgmma[slot][0][0][0]`.
 */
template <typename Dims>
__device__ __forceinline__ void zero_fill_unused_down_act_slots(
    std::uint32_t routed_count, void* a_down_wgmma_slot_base_ptr) {
  // Fixed SHM layout of `a_down_wgmma[slot]` (see moe_internal.h):
  //   [FP8_ACT_NUM_CHUNKS = 8 kc][T_TILE = 8 tok][FP8_ACT_K_CHUNK = 16 kb]
  // with row-major bytes `(kc, tok, kb) → kc*128 + tok*16 + kb`.
  constexpr std::uint32_t FP8_ACT_NUM_CHUNKS = 8u;  // K / 16
  constexpr std::uint32_t T_TILE_TOK = 8u;          // tokens per slot
  constexpr std::uint32_t FP8_ACT_K_CHUNK = 16u;    // fp8 bytes per (kc, tok)

  // Early-out when no tail rows need zeroing.  This is the common case
  // for fully-packed experts (routed_count == 8).  Guarded here so the
  // caller can invoke uniformly without its own check.
  if (routed_count >= T_TILE_TOK) {
    return;
  }

  // Lane within warp 8.  Lane 0 is the TMA launcher; skip it so the
  // store traffic does not block TMA issue (and so only 31 lanes
  // contribute).  This uses `% 32` rather than `- 8*32` to stay safe if
  // the helper is called from a different warp role in the future: the
  // store arithmetic is lane-local and independent of absolute warp id.
  const std::uint32_t lane_id = threadIdx.x & 31u;
  if (lane_id == 0u) {
    return;
  }

  // 4-byte view of the slot.  Each `(kc, tok)` pair is 16 B = 4 dwords;
  // we iterate a flat index over the unused region's dwords and map
  // back to `(kc, tok, d)` so the offset math is cheap.
  std::uint32_t* dst_u32 =
      reinterpret_cast<std::uint32_t*>(a_down_wgmma_slot_base_ptr);

  // Total 4-byte stores in the unused region:
  //   (8 - routed_count) tokens × 8 kc × 4 dwords per 16-B chunk
  const std::uint32_t unused_tok = T_TILE_TOK - routed_count;
  constexpr std::uint32_t DWORDS_PER_CHUNK = FP8_ACT_K_CHUNK / 4u;  // 4
  const std::uint32_t total_dwords =
      unused_tok * FP8_ACT_NUM_CHUNKS * DWORDS_PER_CHUNK;
// Bound: at most 7 * 8 * 4 = 224 dwords = 896 B, distributed across
// 31 worker lanes (~8 dwords per lane worst case).

// Strided loop: lanes 1..31 step by 31 through the flat dword index.
// Shift the starting index by `lane_id - 1` so lanes 1..31 map to
// flat indices 0..30 on the first iteration.
#pragma unroll 1
  for (std::uint32_t i = lane_id - 1u; i < total_dwords; i += 31u) {
    // Unpack `i = (t_in_unused, kc, d)` with layout
    //   d   ∈ [0, 4)              fastest
    //   kc  ∈ [0, 8)              next
    //   t_in_unused ∈ [0, unused_tok)   outermost
    // The SHM linear offset in uint32 units is
    //   dst_u32[kc * (T_TILE_TOK * DWORDS_PER_CHUNK)
    //         + tok * DWORDS_PER_CHUNK + d]
    // where `tok = routed_count + t_in_unused` (never out of bounds).
    const std::uint32_t d = i & (DWORDS_PER_CHUNK - 1u);        // i % 4
    const std::uint32_t kc_t = i >> 2;                          // i / 4
    const std::uint32_t kc = kc_t & (FP8_ACT_NUM_CHUNKS - 1u);  // kc % 8
    const std::uint32_t t_in_unused = kc_t >> 3;                // kc / 8
    const std::uint32_t tok = routed_count + t_in_unused;

    const std::uint32_t slot_u32_off =
        kc * (T_TILE_TOK * DWORDS_PER_CHUNK) + tok * DWORDS_PER_CHUNK + d;
    dst_u32[slot_u32_off] = 0u;
  }
}

}  // namespace moe_monokernel

#endif  // MOE_TMA_H
