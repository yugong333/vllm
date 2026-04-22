
#pragma once
#ifndef MOE_TMA_UTILS_CU
  #define MOE_TMA_UTILS_CU

  #ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
    #error Do not include this file directly.
  #endif

  #include <cuda.h>
  #include <cuda_fp8.h>

  #include "moe_interface.h"
  #include "moe_internal.h"
  #include "ptx_utils.h"

///////////////////////////////////////////////////////////////////////////////
//
// TMA (Tensor Memory Accelerator) copy functions for the MoE monokernel.
//
// These replace the per-thread copy128() loops in the prefetch warps with
// single-thread TMA bulk copies.  The hardware handles address generation,
// swizzling, and cache coherence.
//
// Key differences from the copy128 path:
//   - Only ONE thread (typically thread 0 of a prefetch warp) issues the
//     TMA copy.  All other threads just participate in the barrier.
//   - Completion is tracked via mbarrier instead of cuda::pipeline.
//   - The shared memory destination must be 128-byte aligned for swizzled
//     modes (the existing layout already satisfies this for weight tiles).
//   - TMA handles the rotate_col_32 swizzle in hardware when configured
//     with CU_TENSOR_MAP_SWIZZLE_32B.
//
// Activation tiles with indirect (gathered) token addressing still use
// the copy128 path since TMA cannot do scatter/gather.
//
///////////////////////////////////////////////////////////////////////////////

namespace moe_monokernel {

// ── TMA barrier management helpers ──────────────────────────────────────

/**
 * @brief Initialize all TMA barriers in shared memory.
 *
 * Must be called once at kernel start by a single thread (typically
 * threadIdx.x == 0) before any TMA copies are issued.
 */
template <typename Dims>
__device__ inline void tma_init_barriers(MoE_SHM<Dims>* shmem) {
  for (uint32_t i = 0; i < MoE_SHM<Dims>::TMA_BAR_COUNT; ++i) {
    // Initialize with 0 expected bytes; will be set before each use
    tma_barrier_init(&shmem->tma_bar[i], 0);
  }
}

// ── TMA weight copy functions ───────────────────────────────────────────

/**
 * @brief TMA copy of up-projection expert weights to shared memory.
 *
 * Replaces moe_request_up_expert() for the BS64 path (half-K tiles).
 * Copies W_UP_TILE/2 rows from the lower N and upper N weight blocks
 * for the given expert, using two TMA 2D copies.
 *
 * The tensor map w_up describes the full [E*2*N, K] weight matrix.
 * We select the tile by computing the row coordinate from expert id
 * and blockIdx.
 *
 * @param tma_descs  TMA descriptor bundle.
 * @param id         Expert index.
 * @param dest       Shared memory destination (w[index] in Gemm1Data).
 * @param bar        Pointer to the mbarrier to track completion.
 * @param col_offset Column offset (0 or K/2) for half-K tiling.
 */
template <typename Dims, std::size_t Rows, std::size_t Cols>
__device__ inline void tma_request_up_expert_half(
    const MoETmaDescriptors& tma_descs, std::uint32_t id,
    W_element (&dest)[Rows][Cols], uint64_t* bar, std::uint32_t col_offset) {
  using CoreDims = MoECoreDims<Dims>;
  const unsigned base_row = blockIdx.x * CoreDims::W_UP_TILE / 2;

  // Total bytes for this tile: W_UP_TILE/2 rows from lower N + W_UP_TILE/2
  // rows from upper N, each row is K/2 fp8 elements = K/2 bytes
  const uint32_t half_tile_bytes =
      (CoreDims::W_UP_TILE / 2) * (Dims::HIDDEN_STATES / 2);
  const uint32_t total_bytes = 2 * half_tile_bytes;

  // Only thread 0 of the first prefetch warp issues the TMA copies
  if (threadIdx.x == CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP) {
    tma_barrier_expect_tx(bar, total_bytes);

    // Lower N rows: expert row = id * 2*N + base_row
    int32_t row_lower = id * 2 * Dims::N + base_row;
    tma_copy_2d(&tma_descs.w_up, bar, &dest[0][0],
                /*coord_x=*/col_offset, /*coord_y=*/row_lower);

    // Upper N rows: expert row = id * 2*N + N + base_row
    int32_t row_upper = id * 2 * Dims::N + Dims::N + base_row;
    tma_copy_2d(&tma_descs.w_up, bar, &dest[CoreDims::W_UP_TILE / 2][0],
                /*coord_x=*/col_offset, /*coord_y=*/row_upper);
  }
}

/**
 * @brief TMA copy of up-projection expert weights to shared memory (full K).
 *
 * Replaces moe_request_up_expert() for the BS8 path (full-K tiles).
 * Copies W_UP_TILE/2 rows from lower N and upper N weight blocks.
 *
 * @param tma_descs  TMA descriptor bundle.
 * @param id         Expert index.
 * @param dest       Shared memory destination.
 * @param bar        Pointer to the mbarrier to track completion.
 */
template <typename Dims, std::size_t Rows, std::size_t Cols>
__device__ inline void tma_request_up_expert_full(
    const MoETmaDescriptors& tma_descs, std::uint32_t id,
    W_element (&dest)[Rows][Cols], uint64_t* bar) {
  using CoreDims = MoECoreDims<Dims>;
  const unsigned base_row = blockIdx.x * CoreDims::W_UP_TILE / 2;

  // Full K: W_UP_TILE/2 rows × K bytes per row, times 2 (lower + upper N)
  const uint32_t half_tile_bytes =
      (CoreDims::W_UP_TILE / 2) * Dims::HIDDEN_STATES;
  const uint32_t total_bytes = 2 * half_tile_bytes;

  if (threadIdx.x == CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP) {
    tma_barrier_expect_tx(bar, total_bytes);

    int32_t row_lower = id * 2 * Dims::N + base_row;
    tma_copy_2d(&tma_descs.w_up, bar, &dest[0][0],
                /*coord_x=*/0, /*coord_y=*/row_lower);

    int32_t row_upper = id * 2 * Dims::N + Dims::N + base_row;
    tma_copy_2d(&tma_descs.w_up, bar, &dest[CoreDims::W_UP_TILE / 2][0],
                /*coord_x=*/0, /*coord_y=*/row_upper);
  }
}

/**
 * @brief TMA copy of down-projection expert weights and scales to shared
 * memory.
 *
 * Replaces moe_request_down_expert() for the BS64 path.
 * Copies W_DOWN_TILE rows of weights and the corresponding scale vector.
 *
 * @param tma_descs  TMA descriptor bundle.
 * @param id         Expert index.
 * @param shm        Shared memory Gemm2Data struct.
 * @param w_index    Double-buffer index (0 or 1).
 * @param bar        Pointer to the mbarrier to track completion.
 */
template <typename Dims>
__device__ inline void tma_request_down_expert(
    const MoETmaDescriptors& tma_descs, std::uint32_t id,
    typename MoE_SHM<Dims>::U::Gemm2Data* shm, std::uint32_t w_index,
    uint64_t* bar) {
  using CoreDims = MoECoreDims<Dims>;
  const unsigned base_row = blockIdx.x * CoreDims::W_DOWN_TILE;

  // Weight tile: W_DOWN_TILE rows × N fp8 elements = W_DOWN_TILE * N bytes
  // Scale tile: W_DOWN_TILE × sizeof(float) bytes
  const uint32_t w_bytes = CoreDims::W_DOWN_TILE * Dims::N;
  const uint32_t s_bytes = CoreDims::W_DOWN_TILE * sizeof(S_element);
  const uint32_t total_bytes = w_bytes + s_bytes;

  if (threadIdx.x == CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP) {
    tma_barrier_expect_tx(bar, total_bytes);

    // Weight tile: row in global = id * HIDDEN_STATES + base_row
    int32_t w_row = id * Dims::HIDDEN_STATES + base_row;
    tma_copy_2d(&tma_descs.w_down, bar, &shm->w[w_index][0][0],
                /*coord_x=*/0, /*coord_y=*/w_row);

    // Scale tile: 1D copy, coordinate = id * HIDDEN_STATES + base_row
    int32_t s_coord = id * Dims::HIDDEN_STATES + base_row;
    tma_copy_1d(&tma_descs.s_down, bar, &shm->scale[w_index][0], s_coord);
  }
}

/**
 * @brief TMA copy of down-projection expert weights and scales for BS8 path.
 *
 * Replaces moe_request_down_expert_tiny().
 */
template <typename Dims>
__device__ inline void tma_request_down_expert_tiny(
    const MoETmaDescriptors& tma_descs, std::uint32_t id,
    typename MoE_SHM<Dims>::U::TinyData* shm, std::uint32_t w_index,
    uint64_t* bar) {
  using CoreDims = MoECoreDims<Dims>;
  const unsigned base_row = blockIdx.x * CoreDims::W_DOWN_TILE;

  const uint32_t w_bytes = CoreDims::W_DOWN_TILE * Dims::N;
  const uint32_t s_bytes = CoreDims::W_DOWN_TILE * sizeof(S_element);
  const uint32_t total_bytes = w_bytes + s_bytes;

  if (threadIdx.x == CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP) {
    tma_barrier_expect_tx(bar, total_bytes);

    int32_t w_row = id * Dims::HIDDEN_STATES + base_row;
    tma_copy_2d(&tma_descs.w_down, bar, &shm->w[w_index].down[0][0],
                /*coord_x=*/0, /*coord_y=*/w_row);

    int32_t s_coord = id * Dims::HIDDEN_STATES + base_row;
    tma_copy_1d(&tma_descs.s_down, bar, &shm->scale[w_index][0], s_coord);
  }
}

/**
 * @brief TMA copy of temp fp32 tokens for down-projection (BS64 path).
 *
 * Replaces moe_request_temp_token().  Copies T_TILE rows of N fp32 elements
 * from spec->temp_fp32.
 *
 * @param tma_descs  TMA descriptor bundle.
 * @param expert     Expert reference with first_token/last_token.
 * @param a_row      Row offset within the expert's token range.
 * @param dest       Shared memory destination tile.
 * @param bar        Pointer to the mbarrier.
 */
template <typename Dims, std::size_t Rows, std::size_t Cols>
__device__ inline void tma_request_temp_token(
    const MoETmaDescriptors& tma_descs, const ExpertRef& expert, unsigned a_row,
    T_element (&dest)[Rows][Cols], uint64_t* bar) {
  using CoreDims = MoECoreDims<Dims>;

  // Rows to copy: min(Rows, remaining tokens)
  unsigned a_rows = expert.last_token - expert.first_token;
  unsigned copy_rows = (a_row + Rows <= a_rows) ? Rows : (a_rows - a_row);
  const uint32_t total_bytes = copy_rows * Dims::N * sizeof(T_element);

  if (threadIdx.x == CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP) {
    tma_barrier_expect_tx(bar, total_bytes);

    // Row coordinate in the global temp buffer
    int32_t row = expert.first_token + a_row;
    tma_copy_2d(&tma_descs.t_down, bar, &dest[0][0],
                /*coord_x=*/0, /*coord_y=*/row);
  }
}

/**
 * @brief TMA copy of bf16 activations for BS8 path (Phase 1).
 *
 * Replaces moe_fetch_activation_async() for a single token.
 * Copies one row of K bf16 elements from global to shared memory.
 *
 * @param tma_descs  TMA descriptor bundle.
 * @param tok        Token index in the global activation array.
 * @param dest       Shared memory destination row.
 * @param bar        Pointer to the mbarrier.
 */
template <typename Dims>
__device__ inline void tma_fetch_activation(const MoETmaDescriptors& tma_descs,
                                            std::uint32_t tok, A_element* dest,
                                            uint64_t* bar) {
  using CoreDims = MoECoreDims<Dims>;

  const uint32_t total_bytes = Dims::HIDDEN_STATES * sizeof(A_element);

  if (threadIdx.x == CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP) {
    tma_barrier_expect_tx(bar, total_bytes);
    tma_copy_2d(&tma_descs.a_orig, bar, dest,
                /*coord_x=*/0, /*coord_y=*/tok);
  }
}

/**
 * @brief TMA copy of bf16 intermediate results for BS8 down-projection.
 *
 * Copies one token's N bf16 values from spec->temp_bf16 to shared memory.
 *
 * @param tma_descs  TMA descriptor bundle.
 * @param vrow       Virtual row index in temp_bf16.
 * @param dest       Shared memory destination row.
 * @param bar        Pointer to the mbarrier.
 */
template <typename Dims>
__device__ inline void tma_fetch_temp_bf16_row(
    const MoETmaDescriptors& tma_descs, std::uint32_t vrow, A_element* dest,
    uint64_t* bar) {
  using CoreDims = MoECoreDims<Dims>;

  const uint32_t total_bytes = Dims::N * sizeof(A_element);

  if (threadIdx.x == CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP) {
    tma_barrier_expect_tx(bar, total_bytes);
    tma_copy_2d(&tma_descs.t_bf16, bar, dest,
                /*coord_x=*/0, /*coord_y=*/vrow);
  }
}

}  // namespace moe_monokernel

#endif
