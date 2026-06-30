
#pragma once
#ifndef MOE_UP_PROJECTION_CU
  #define MOE_UP_PROJECTION_CU

  #ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
    #error Do not include this file directly.
  #endif

  #include <cuda.h>
  #include <cuda/pipeline>
  #include <cuda_fp8.h>
  #include <stdio.h>

  #include "ptx_utils.h"
  #include "moe_interface.h"
  #include "moe_internal.h"
  #include "moe_down_projection.cu"

///////////////////////////////////////////////////////////////////////////////
//
// Design Considerations
//
// * The smallest matrix dimensions that the Tensor Cores support is 16 x 8 x K
//   - 8 is a reasonable amount of input tokens we can process at once per
//   expert,
//     if there are <= 64 tokens and >= 16 experts.
//   - 16 rows of the weight matrix per SM matches the 2048 = 2*N rows we have
//   with TP=8
//   - with K=5120, input token activation tile is 40kB, a weight tile is 80kB
//
// * Max. 220kB of the 224kB Shared Memory are available for matrix tiles
//   - 'tiny' kernel holds 1 fixed token tile and 2 weight tiles (200kB total),
//     so we use double-buffering for 'tiny': fetching the next expert weights
//     while processing the current one.
//   - 'normal' kernel needs to also prefetch input tokens, hence we use tiles
//     that only span HALF of K, but have now enough space for 3x20kB input
//     token tiles plus 3x40kB of weight tiles. We use triple-buffering to
//     ensure that data is likely to arrive early enough: 1 tile in MMA
//     processing, 1 coming in, 1 being currently requested from Global Memory.
//
// * Reading matrix data from Shared Memory accesses the same columns in
// different
//   rows simultaneously. We prevent bank conflicts in different ways:
//   - 'normal' kernel uses tile rows lengths with an extra 16 byte padding,
//     mapping the same column for all 8 consecutive rows onto to different
//     banks.
//   - downside: prevents Global->Shared Memory transfers from being fully
//   coalesced.
//   - 'tiny' kernel has larger tile sizes that make it more easy to be memory
//   bound.
//     We don't pad the row 'swizzle' the data inside them. Experiments found
//     that 32-byte swizzles be the sweet spot, eliminating non-coalesced
//     transfers at the expense of 50% Shared Memory bank conflicts.
//
///////////////////////////////////////////////////////////////////////////////

namespace moe_monokernel {

/**
 * @brief Initiate the copy of token activations from Global to Shared Memory
 *
 * This device function issues the asynchronous data copy requests for a tile of
 * token activation vectors. The copy operations will be queued in the given @a
 * pipe, which the caller must use to wait for their completion.
 *
 * Token addressing is indirect, using @a token_indexes to specify which
 * activation vectors to copy from @a source. If @a max_count is less than the
 * number of tokens that fit into the given Shared Memory tile at @a dest, only
 * the content for the first @a max_count tokens will be replaced.
 *
 * @note Like all prefetching functions, this function must only be called by
 *       threads in prefetch warps.
 *
 * @param source Pointer token activation array of shape [BS, HIDDEN_STATES] in
 * row-major order. Individual elements are quantized in __nv_fp8_e4m3 format.
 *               Tile offsets within the source rows are expressed by shifting
 * the pointer by the respective number of columns. Stored in Global Memory.
 * @param token_indexes Row indexes within @a source of the tokens to fetch.
 * @param dest Shared Memory struct (tile of token activations) to store the
 * result to.
 * @param max_count Maximum number of tokens to copy. Implicitly limited to @a
 * dest capacity.
 * @param pipe Asynchronous completion pipe to use.
 */
template <typename Dims>
__device__ inline void moe_request_input_tokens(
    const AQ_element* __restrict__ source,
    const std::uint16_t* __restrict__ token_indexes,
    AQ_element (&dest)[MoECoreDims<Dims>::A_TILE]
                      [MoECoreDims<Dims>::K_DIM_HALF_PADDED_A],
    std::uint32_t max_count, cuda::pipeline<cuda::thread_scope_thread>& pipe) {
  // position within the block
  using CoreDims = MoECoreDims<Dims>;
  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_prefetch_warp<Dims>();

  // async transfers are 16 bytes / thread
  const unsigned chunk_size = 16 / sizeof(*source);

  for (unsigned row = warp; row < CoreDims::A_TILE;
       row += CoreDims::PREFETCH_WARP_COUNT) {
    if (row < max_count) {
      std::uint32_t row_idx = token_indexes[row];
      const AQ_element* a = source + row_idx * Dims::HIDDEN_STATES;
      for (unsigned col = thread * chunk_size; col < Dims::HIDDEN_STATES / 2;
           col += CoreDims::THREADS_PER_WARP * chunk_size) {
        unsigned dest_col = rotate_col_32(col, row);
        copy128(dest[row][dest_col], a[col], pipe);
      }
    }
  }
}

/**
 * @brief Initiate the copy of expert weights from Global to Shared Memory
 *
 * This device function issues the asynchronous data copy requests for a tile of
 * expert weights. The copy operations will be queued in the given @a pipe,
 * which the caller must use to wait for their completion.
 *
 * While the expert is selected by @a id, the tile to copy is implicitly
 * selected by the @c blockIdx. Two ranges of rows will be copied: @a Rows/2
 * rows from the first @c N rows of weights, followed by @a Rows/2 rows of the
 * second @c N rows of weights.
 *
 * To allow for differences in Shared Memory data layout, @a CopyCols template
 * parameter specifies the number of elements to copy per row.
 *
 * @note Like all prefetching functions, this function must only be called by
 *       threads in prefetch warps.
 *
 * @tparam CopyCols Number of elements to copy in each row.
 * @param source Pointer token weights array of shape [NUM_EXPERTS, 2*N,
 * HIDDEN_STATES] in expert, row-major order. Tile offsets within the source
 * rows are expressed by shifting the pointer by the respective number of
 * columns. Individual elements are in __nv_fp8_e4m3 format. Stored in Global
 * Memory.
 * @param id Expert index within @a source.
 * @param dest Shared Memory struct (tile of token activations) to store the
 * result to.
 * @param pipe Asynchronous completion pipe to use.
 */
template <typename Dims, std::size_t CopyCols, std::size_t Rows,
          std::size_t Cols>
__device__ inline void moe_request_up_expert(
    const W_element* __restrict__ source, std::uint32_t id,
    W_element (&dest)[Rows][Cols],
    cuda::pipeline<cuda::thread_scope_thread>& pipe) {
  static_assert(CopyCols <= Cols);

  // position within the block
  using CoreDims = MoECoreDims<Dims>;
  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_prefetch_warp<Dims>();
  const unsigned chunk_size = 16 / sizeof(*source);

  // starting row to process
  const unsigned base_row = blockIdx.x * CoreDims::W_UP_TILE / 2;
  const unsigned item_cols_per_iteration =
      CoreDims::THREADS_PER_WARP * chunk_size;

  // Each row within the weights matrix needs to be multiple of the copy size
  static_assert(Dims::HIDDEN_STATES % item_cols_per_iteration == 0);
  const OpaqueElement* weights =
      (const OpaqueElement*)(source + id * 2 * Dims::N * Dims::HIDDEN_STATES);

  // bring W tile
  // Even warps fetch 8 rows from lower N weight rows, odd warps fetch 8 rows
  // from upper N Curb unrolling for smaller and slightly faster kernel code
  #pragma unroll 1
  for (unsigned row = warp / 2; row < CoreDims::W_UP_TILE / 2;
       row += CoreDims::PREFETCH_WARP_COUNT / 2) {
    for (unsigned col = 0; col < CopyCols; col += item_cols_per_iteration) {
      unsigned source_col = thread;
      unsigned is_upper = warp & 1;

      copy128(
          dest[row + is_upper * CoreDims::W_UP_TILE / 2]
              [rotate_col_32(col + source_col * chunk_size, row)],
          weights[((row + base_row + is_upper * Dims::N) * Dims::HIDDEN_STATES +
                   col + source_col * chunk_size) /
                  sizeof(OpaqueElement)],
          pipe);
    }
  }
}

/**
 * @brief Performs the MMA result reduction and sigmoid step of up-projection.
 *
 * This device function sums up the partial scalar products created by all
 * warps, applied the respective weight and token activation scales, calculates
 * the sigmoid, and finally stores the results in Global Memory.  The tile to be
 * written is implicitly determined by the @c blockIdx.
 *
 * The output can be filtered, i.e. @a store_row0 and @a store_row1 control
 * whether the results for the respective rows @a row0 and @a row1 shall be
 * written.  This allows the called to always process data at the full tile size
 * and simply suppress superfluous results in the output.
 *
 * @note This function is supposed to be called by warp 0 only.
 *
 * @param partial_result Array of MMA results from all warps of shape [WARPS, 4,
 * THREADS] in row-major order. Individual elements are in FP32 format.
 * @param d0 First element of the MMA result of warp 0.
 * @param d1 Second element of the MMA result of warp 0.
 * @param d2 Third element of the MMA result of warp 0.
 * @param d3 Forth element of the MMA result of warp 0.
 * @param ws0 First weight scale for the respective expert.
 * @param ws1 Second weight scale for the respective expert.
 * @param ts0 First token activation scale.
 * @param ts1 Second token activation scale.
 * @param store_row0 Specifies if result in @a row0 shall be stored.
 * @param store_row1 Specifies if result in @a row1 shall be stored.
 * @param row0 Row to store the scalar products for the first token.
 * @param row1 Row to store the scalar products for the second token.
 * @param result Pointer to the output array of shape [BS, N] in row-major
 * order. Individual elements are in FP32 format.
 */
template <typename Dims, std::size_t Rows, std::size_t Cols>
__device__ inline void moe_up_reduction(
    const float (&partial_result)[Rows][Cols], float d0, float d1, float d2,
    float d3, float ws0, float ws1, float ts0, float ts1, bool store_row0,
    bool store_row1, unsigned row0, unsigned row1,
  #ifdef DEBUG_MOE
    float* __restrict__ gemm1,
  #endif
    T_element* __restrict__ result) {
  // position within the block
  using CoreDims = MoECoreDims<Dims>;
  const unsigned thread = get_thread<Dims>();

  // starting row to process
  const unsigned base_row = blockIdx.x * CoreDims::W_UP_TILE / 2;

  // combine results
  for (unsigned i = 1; i < CoreDims::CALC_WARP_COUNT; ++i) {
    d0 += partial_result[i][thread + 0];
    d1 += partial_result[i][thread + 32];
    d2 += partial_result[i][thread + 64];
    d3 += partial_result[i][thread + 96];
  }

  // for debugging purposes
  #ifdef DEBUG_MOE
  if (store_row0) {
    gemm1[row0 * 2 * Dims::N + (thread / 4) + base_row + 0] = d0 * ts0 * ws0;
    gemm1[row0 * 2 * Dims::N + (thread / 4) + base_row + Dims::N] =
        d2 * ts0 * ws1;
  }
  if (store_row1) {
    gemm1[row1 * 2 * Dims::N + (thread / 4) + base_row + 0] = d1 * ts1 * ws0;
    gemm1[row1 * 2 * Dims::N + (thread / 4) + base_row + Dims::N] =
        d3 * ts1 * ws1;
  }
  #endif

  // apply weights and store as temp
  // x: columns 0 ..  7
  // w: columns 8 .. 15
  float x0 = d0 * ts0 * ws0;
  float x1 = d1 * ts1 * ws0;
  float w0 = d2 * ts0 * ws1;
  float w1 = d3 * ts1 * ws1;

  float sig0 = (w0 * x0) / (1 + expf(-x0));
  float sig1 = (w1 * x1) / (1 + expf(-x1));

  // write to temporary buffer
  // Guard: blocks beyond N have no valid up-proj columns to write.
  // GRID_SIZE is sized for the down-proj (K columns) but the up-proj
  // only has N columns.  Skip the write for out-of-range blocks.
  if (store_row0 && (thread / 4) + base_row < Dims::N) {
    result[row0 * Dims::N + (thread / 4) + base_row] = sig0;
  }
  if (store_row1 && (thread / 4) + base_row < Dims::N) {
    result[row1 * Dims::N + (thread / 4) + base_row] = sig1;
  }
}

/**
 * @brief Performs the MMA reduction and sigmoid step of up-projection for
 * 'Tiny' kernels.
 *
 * This device function sums up the partial scalar products created by all
 * warps, applied the respective weight and token activation scales, calculates
 * the sigmoid, and finally stores the results in Global Memory.  The tile to be
 * written is implicitly determined by the @c blockIdx.
 *
 * @note This function is supposed to be called by warps 0 and 1 only.
 *
 * @param partial_result Array of MMA results from all warps of shape [WARPS, 4,
 * THREADS] in row-major order. Individual elements are in FP32 format.
 * @param ws0 First weight scale for the respective expert.
 * @param ws1 Second weight scale for the respective expert.
 * @param ts Token activation scale.
 * @param row Row to store the scalar products for the token.
 * @param result Pointer to the output array of shape [BS, N] in row-major
 * order. Individual elements are in FP32 format.
 */
template <typename Dims, std::size_t Rows, std::size_t Cols>
__device__ inline void moe_up_reduction_tiny(
    const float (&partial_result)[Rows][Cols], float ws0, float ws1, float ts,
    unsigned row,
  #ifdef DEBUG_MOE
    float* __restrict__ gemm1,
  #endif
    T_element* __restrict__ result) {
  // position within the block
  using CoreDims = MoECoreDims<Dims>;
  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_calc_warp<Dims>();

  // starting row to process
  const unsigned base_row = blockIdx.x * CoreDims::W_UP_TILE / 2;

  // combine results, reduce dependency chain on dX
  float d0 = partial_result[0][thread + warp * 32 + 0] +
             partial_result[1][thread + warp * 32 + 0];
  float d2 = partial_result[0][thread + warp * 32 + 64] +
             partial_result[1][thread + warp * 32 + 64];

  for (unsigned i = 2; i < CoreDims::CALC_WARP_COUNT; i += 2) {
    d0 += partial_result[i][thread + warp * 32 + 0] +
          partial_result[i + 1][thread + warp * 32 + 0];
    d2 += partial_result[i][thread + warp * 32 + 64] +
          partial_result[i + 1][thread + warp * 32 + 64];
  }

  // for debugging purposes
  #ifdef DEBUG_MOE
  gemm1[row * 2 * Dims::N + (thread / 4) + base_row + 0] = d0 * ts * ws0;
  gemm1[row * 2 * Dims::N + (thread / 4) + base_row + Dims::N] = d2 * ts * ws1;
  #endif

  // write to temporary buffer
  float x0 = d0 * ts * ws0;
  float w0 = d2 * ts * ws1;
  float sig0 = (w0 * x0) / (1 + expf(-x0));
  result[row * Dims::N + (thread / 4) + base_row] = sig0;
}

/**
 * @brief Standard kernel for the first GEMM ("up projection"), combined with a
 * sigmoid reduction.
 *
 * This device function processes @c BS tokens, grouped by expert in internal
 * batches of 8 tokens. The experts to use and the respective list of tokes for
 * each of them is given by @a shmem.
 *
 * All non-expert data is taken from our temporary storage in either @a spec or
 * @a shmem and results will be written to @a spec. Outputs are grouped by
 * expert.
 *
 * @param expert_weights_up Pointer token weights array of shape [NUM_EXPERTS,
 * 2*N, HIDDEN_STATES] in expert, row-major order. Individual elements are in
 * __nv_fp8_e4m3 format. Stored in Global Memory.
 * @param expert_scales_up Pointer weights scales array of shape [NUM_EXPERTS,
 * 2*N] in row-major order. Individual elements are in FP32 format. Stored in
 * Global Memory.
 * @param spec Global Memory struct containing the scaled input token
 * activations. It will also receive the output of this function.
 * @param shmem Shared Memory struct containing the expert<=>token mapping,
 * activation weights and will be uses as local scratch pad store for faster
 * operation.
 */

template <typename Dims>
__device__ inline void moe_up_projection_topk(
    const W_element* __restrict__ expert_weights_up,
    const S_element* __restrict__ expert_scales_up,
    MoEGemmSpec<Dims>* __restrict__ spec, MoE_SHM<Dims>* __restrict__ shmem) {
  static_assert(Dims::BS > 8,
                "Tiny is handled by its own kernel. Do not use "
                "moe_up_projection_topk for BS<=8");
  using CoreDims = MoECoreDims<Dims>;
  using MoE_SHM_t = MoE_SHM<Dims>;

  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_any_warp<Dims>();

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  const unsigned base_row = blockIdx.x * CoreDims::W_UP_TILE / 2;

  typename MoE_SHM_t::U::Gemm1Data* shm = &shmem->u.gemm1;
  std::uint32_t expert_count = shmem->expert_count;
  // activations are indexed by original token; token_indexes_topk maps
  // sorted_pos -> original token
  const AQ_element* activations = spec->activations[0];

  // triple-buffering: queue first 2 tiles
  if (is_prefetch_warp<Dims>()) {
    const ExpertRef& expert = shmem->experts[0];
    pipe.producer_acquire();
    moe_request_input_tokens<Dims>(
        activations, &shmem->path.bs64.token_indexes_topk[expert.first_token],
        shm->a[0], expert.last_token, pipe);
    moe_request_up_expert<Dims, Dims::HIDDEN_STATES / 2>(
        expert_weights_up, expert.id, shm->w[0], pipe);
    pipe.producer_commit();

    pipe.producer_acquire();
    moe_request_input_tokens<Dims>(
        activations + Dims::HIDDEN_STATES / 2,
        &shmem->path.bs64.token_indexes_topk[expert.first_token], shm->a[1],
        expert.last_token, pipe);
    moe_request_up_expert<Dims, Dims::HIDDEN_STATES / 2>(
        expert_weights_up + Dims::HIDDEN_STATES / 2, expert.id, shm->w[1],
        pipe);
    pipe.producer_commit();
  }

  std::uint32_t t_index_read = 0;
  std::uint32_t t_index_write = 2;
  std::uint32_t w_index_read = 0;
  std::uint32_t w_index_write = 2;

  for (std::uint32_t e = 0; e < expert_count; ++e) {
    const ExpertRef& expert = shmem->experts[e];
    std::uint32_t id = expert.id;
    const S_element* scales = expert_scales_up + id * 2 * Dims::N;
    unsigned int a_rows = expert.last_token - expert.first_token;
    // temp is indexed by sorted position (first_token..last_token)
    T_element* temp = &spec->temp[expert.first_token * Dims::N];

    float ws0 = scales[base_row + thread / 4];
    float ws1 = scales[base_row + thread / 4 + Dims::N];

    for (unsigned a_row = 0; a_row < a_rows; a_row += CoreDims::A_TILE) {
      float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

      cuda::pipeline_consumer_wait_prior<1>(pipe);
      __syncthreads();

      if (is_prefetch_warp<Dims>()) {
        pipe.producer_acquire();
        if (e + 1 < expert_count && a_row == 0) {
          moe_request_up_expert<Dims, Dims::HIDDEN_STATES / 2>(
              expert_weights_up, shmem->experts[e + 1].id,
              shm->w[w_index_write], pipe);
          w_index_write = w_index_write == 2 ? 0 : w_index_write + 1;
        }
        if (a_row + CoreDims::A_TILE < a_rows) {
          moe_request_input_tokens<Dims>(
              activations,
              &shmem->path.bs64.token_indexes_topk[expert.first_token +
                                                   CoreDims::A_TILE + a_row],
              shm->a[t_index_write], a_rows - CoreDims::A_TILE - a_row, pipe);
          t_index_write = t_index_write == 2 ? 0 : t_index_write + 1;
        } else if (e + 1 < expert_count) {
          const ExpertRef& next = shmem->experts[e + 1];
          moe_request_input_tokens<Dims>(
              activations,
              &shmem->path.bs64.token_indexes_topk[next.first_token],
              shm->a[t_index_write], next.last_token - next.first_token, pipe);
          t_index_write = t_index_write == 2 ? 0 : t_index_write + 1;
        }
        pipe.producer_commit();
      } else {
        for (unsigned base_col = warp * CoreDims::K_TILE;
             base_col < Dims::HIDDEN_STATES / 2;
             base_col += CoreDims::BLOCK_STRIDE) {
          unsigned row = thread / 4;
          unsigned col = 4 * (thread % 4);
          __nv_fp8x4_e4m3 w0 =
              *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 0][rotate_col_32(
                  base_col + col + 0, row)];
          __nv_fp8x4_e4m3 w1 =
              *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 8][rotate_col_32(
                  base_col + col + 0, row)];
          __nv_fp8x4_e4m3 w2 =
              *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 0][rotate_col_32(
                  base_col + col + 16, row)];
          __nv_fp8x4_e4m3 w3 =
              *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 8][rotate_col_32(
                  base_col + col + 16, row)];
          __nv_fp8x4_e4m3 a02 =
              *(__nv_fp8x4_e4m3*)(&shm->a[t_index_read][row][rotate_col_32(
                  base_col + col + 0, row)]);
          __nv_fp8x4_e4m3 a13 =
              *(__nv_fp8x4_e4m3*)(&shm->a[t_index_read][row][rotate_col_32(
                  base_col + col + 16, row)]);
          mma_fp8_fp8(d0, d1, d2, d3, w0, w1, w2, w3, a02, a13, d0, d1, d2, d3);
        }
      }

      __syncthreads();
      w_index_read = w_index_read == 2 ? 0 : w_index_read + 1;
      t_index_read = t_index_read == 2 ? 0 : t_index_read + 1;

      cuda::pipeline_consumer_wait_prior<1>(pipe);
      __syncthreads();

      if (is_prefetch_warp<Dims>()) {
        pipe.producer_acquire();
        if (a_row + CoreDims::A_TILE < a_rows) {
          moe_request_input_tokens<Dims>(
              activations + Dims::HIDDEN_STATES / 2,
              &shmem->path.bs64.token_indexes_topk[expert.first_token +
                                                   CoreDims::A_TILE + a_row],
              shm->a[t_index_write], a_rows - CoreDims::A_TILE - a_row, pipe);
          t_index_write = t_index_write == 2 ? 0 : t_index_write + 1;
        } else if (e + 1 < expert_count) {
          const ExpertRef& next = shmem->experts[e + 1];
          moe_request_input_tokens<Dims>(
              activations + Dims::HIDDEN_STATES / 2,
              &shmem->path.bs64.token_indexes_topk[next.first_token],
              shm->a[t_index_write], next.last_token - next.first_token, pipe);
          t_index_write = t_index_write == 2 ? 0 : t_index_write + 1;
          moe_request_up_expert<Dims, Dims::HIDDEN_STATES / 2>(
              expert_weights_up + Dims::HIDDEN_STATES / 2, next.id,
              shm->w[w_index_write], pipe);
          w_index_write = w_index_write == 2 ? 0 : w_index_write + 1;
        }
        pipe.producer_commit();
      } else {
        for (unsigned base_col = warp * CoreDims::K_TILE;
             base_col < Dims::HIDDEN_STATES / 2;
             base_col += CoreDims::BLOCK_STRIDE) {
          unsigned row = thread / 4;
          unsigned col = 4 * (thread % 4);
          __nv_fp8x4_e4m3 w0 =
              *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 0][rotate_col_32(
                  base_col + col + 0, row)];
          __nv_fp8x4_e4m3 w1 =
              *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 8][rotate_col_32(
                  base_col + col + 0, row)];
          __nv_fp8x4_e4m3 w2 =
              *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 0][rotate_col_32(
                  base_col + col + 16, row)];
          __nv_fp8x4_e4m3 w3 =
              *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 8][rotate_col_32(
                  base_col + col + 16, row)];
          __nv_fp8x4_e4m3 a02 =
              *(__nv_fp8x4_e4m3*)(&shm->a[t_index_read][row][rotate_col_32(
                  base_col + col + 0, row)]);
          __nv_fp8x4_e4m3 a13 =
              *(__nv_fp8x4_e4m3*)(&shm->a[t_index_read][row][rotate_col_32(
                  base_col + col + 16, row)]);
          mma_fp8_fp8(d0, d1, d2, d3, w0, w1, w2, w3, a02, a13, d0, d1, d2, d3);
        }
        shm->partial_result[warp][thread + 0] = d0;
        shm->partial_result[warp][thread + 32] = d1;
        shm->partial_result[warp][thread + 64] = d2;
        shm->partial_result[warp][thread + 96] = d3;
      }

      __syncthreads();
      w_index_read = w_index_read == 2 ? 0 : w_index_read + 1;
      t_index_read = t_index_read == 2 ? 0 : t_index_read + 1;
      if (a_row + CoreDims::A_TILE < a_rows)
        w_index_read = w_index_read == 2 ? 0 : w_index_read + 1;

      if (warp == 0) {
        std::uint32_t row0 = a_row + (thread % 4) * 2 + 0;
        std::uint32_t row1 = a_row + (thread % 4) * 2 + 1;
        // Per-slot routing weight: indexed by sorted position
        float ts0 =
            (row0 < a_rows)
                ? shmem->path.bs64.token_weights[expert.first_token + row0]
                : 0.f;
        float ts1 =
            (row1 < a_rows)
                ? shmem->path.bs64.token_weights[expert.first_token + row1]
                : 0.f;

        moe_up_reduction<Dims>(shm->partial_result, d0, d1, d2, d3, ws0, ws1,
                               ts0, ts1, row0 < a_rows, row1 < a_rows, row0,
                               row1,
  #ifdef DEBUG_MOE
                               &spec->gemm1[expert.first_token * 2 * Dims::N],
  #endif
                               temp);
      }
    }
  }
}

}  // namespace moe_monokernel

#endif
