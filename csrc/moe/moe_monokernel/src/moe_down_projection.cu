
#pragma once
#ifndef MOE_DOWN_PROJECTION_CU
  #define MOE_DOWN_PROJECTION_CU

  #ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
    #error Do not include this file directly.
  #endif

  #include <cuda.h>
  #include <cuda/pipeline>
  #include <cuda_fp8.h>
  #include <stdio.h>

  #include "moe_interface.h"
  #include "moe_internal.h"
  #include "ptx_utils.h"

///////////////////////////////////////////////////////////////////////////////
//
// Design Considerations
//
// * The smallest matrix dimensions that the Tensor Cores support is 16 x 8 x K
//   - 8 is a reasonable amount of input tokens we can process at once per
//   expert,
//     if there are <= 64 tokens and >= 16 experts.
//   - Weight matrix is 5120 rows, distributed over 128 SMs, gives 40 rows/SM
//   - run 2 MMA iterations with 16 rows each
//   - duplicate the remaining 8 rows when running the 3rd iteration and filter
//     the output (don't overwrite results of other SM!)
//
// * Max. 220kB of the 224kB Shared Memory are available for matrix tiles
//   - token/'temp' tiles are 32kB each, weight tiles are 40kB each
//   - plenty to space for double buffering both: fetching the next tile of
//   expert
//     weights and / or token activations while processing the current one.
//
// * Reading matrix data from Shared Memory accesses the same columns in
// different
//   rows simultaneously.
//   - kernel uses tile rows lengths with an extra 16 byte padding,
//     mapping the same column for all 8 consecutive rows onto to different
//     banks.
//   - downside: prevents Global->Shared Memory transfers from being fully
//   coalesced.
//
///////////////////////////////////////////////////////////////////////////////

namespace moe_monokernel {

/**
 * @brief Initiate the copy of expert weights and scales from Global to Shared
 * Memory
 *
 * This device function issues the asynchronous data copy requests for a tile of
 * expert weights and corresponding weights. The copy operations will be queued
 * in the given @a pipe, which the caller must use to wait for their completion.
 *
 * While the expert is selected by @a id, the tile to copy is implicitly
 * selected by the @c blockIdx. The result is stored in the tile @a w_index
 * within @a shm.
 *
 * @note Like all prefetching functions, this function must only be called by
 *       threads in prefetch warps.
 *
 * @param expert_weights_down Pointer weights array of shape [NUM_EXPERTS,
 * HIDDEN_STATES, N] in expert, row-major order. Individual elements are in
 * __nv_fp8_e4m3 format. Stored in Global Memory.
 * @param expert_scales_down Pointer scales array of shape [NUM_EXPERTS,
 * HIDDEN_STATES] in row-major order. Individual elements are in __nv_fp8_e4m3
 * format. Stored in Global Memory.
 * @param id Expert index within @a expert_weights_down and @a
 * expert_scales_down.
 * @param shm Shared Memory struct to store the result to.
 * @param w_index Index of tile to use within @a shm.
 * @param pipe Asynchronous completion pipe to use.
 */
template <typename Dims>
__device__ inline void moe_request_down_expert(
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down, std::uint32_t id,
    typename MoE_SHM<Dims>::U::Gemm2Data* shm, std::uint32_t w_index,
    cuda::pipeline<cuda::thread_scope_thread>& pipe) {
  const unsigned base_row = blockIdx.x * MoECoreDims<Dims>::W_DOWN_TILE;

  // position within the block
  using CoreDims = MoECoreDims<Dims>;
  const unsigned d_thread = threadIdx.x % (2 * CoreDims::THREADS_PER_WARP);
  const unsigned d_warp = get_prefetch_warp<Dims>() / 2;

  // request W tile
  {
    const unsigned chunk_size = 16;
    static_assert(Dims::N <= 2 * CoreDims::THREADS_PER_WARP * chunk_size);

    // Compiler on H200 barfs out on plain FP8 transfers as it fails to
    // propagate alignment guarantees: just case to a 32-bit value; we know all
    // data to be 16-byte aligned.
    const OpaqueElement* weights =
        (const OpaqueElement*)(expert_weights_down +
                               id * Dims::N * Dims::HIDDEN_STATES +
                               base_row * Dims::N);
    for (unsigned row = d_warp, i = 0;
         i < CoreDims::W_DOWN_TILE / (CoreDims::PREFETCH_WARP_COUNT / 2);
         row += CoreDims::PREFETCH_WARP_COUNT / 2, i++) {
      unsigned col = d_thread * chunk_size;
      // "clever" condition to allow for compile-time optimization (becomes
      // no-op on H200)
      if (Dims::N == 2 * CoreDims::THREADS_PER_WARP * chunk_size ||
          col < Dims::N) {
        copy128(shm->w[w_index][row][col],
                weights[(row * Dims::N + col) / sizeof(OpaqueElement)], pipe);
      }
    }
  }

  // request Scale tile
  if (d_warp == 0) {
    const unsigned chunk_size = 16 / sizeof(*expert_scales_down);
    if (d_thread < CoreDims::W_DOWN_TILE / chunk_size) {
      copy128(shm->scale[w_index][chunk_size * d_thread],
              expert_scales_down[id * Dims::HIDDEN_STATES + base_row +
                                 chunk_size * d_thread],
              pipe);
    }
  }
}

/**
 * @brief Initiate the copy of expert weights and scales from Global to Shared
 * Memory for 'Tiny' kernel.
 *
 * This device function issues the asynchronous data copy requests for a tile of
 * expert weights and corresponding weights. The copy operations will be queued
 * in the given @a pipe, which the caller must use to wait for their completion.
 *
 * While the expert is selected by @a id, the tile to copy is implicitly
 * selected by the @c blockIdx. The result is stored in the tile @a w_index
 * within @a shm.
 *
 * @note Like all prefetching functions, this function must only be called by
 *       threads in prefetch warps.
 *
 * @param expert_weights_down Pointer weights array of shape [NUM_EXPERTS,
 * HIDDEN_STATES, N] in expert, row-major order. Individual elements are in
 * __nv_fp8_e4m3 format. Stored in Global Memory.
 * @param expert_scales_down Pointer scales array of shape [NUM_EXPERTS,
 * HIDDEN_STATES] in row-major order. Individual elements are in __nv_fp8_e4m3
 * format. Stored in Global Memory.
 * @param id Expert index within @a expert_weights_down and @a
 * expert_scales_down.
 * @param shm Shared Memory struct to store the result to.
 * @param w_index Index of tile to use within @a shm.
 * @param pipe Asynchronous completion pipe to use.
 */
template <typename Dims>
__device__ inline void moe_request_down_expert_tiny(
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down, std::uint32_t id,
    typename MoE_SHM<Dims>::U::TinyData* shm, std::uint32_t w_index,
    cuda::pipeline<cuda::thread_scope_thread>& pipe) {
  const unsigned base_row = blockIdx.x * MoECoreDims<Dims>::W_DOWN_TILE;

  // position within the block
  using CoreDims = MoECoreDims<Dims>;
  const unsigned d_thread = threadIdx.x % (2 * CoreDims::THREADS_PER_WARP);
  const unsigned d_warp = get_prefetch_warp<Dims>() / 2;

  // request W tile
  {
    const unsigned chunk_size = 16;
    static_assert(Dims::N <= 2 * CoreDims::THREADS_PER_WARP * chunk_size);

    // Compiler on H200 barfs out on plain FP8 transfers as it fails to
    // propagate alignment guarantees: just case to a 32-bit value; we know all
    // data to be 16-byte aligned.
    const OpaqueElement* weights =
        (const OpaqueElement*)(expert_weights_down +
                               id * Dims::N * Dims::HIDDEN_STATES +
                               base_row * Dims::N);
    for (unsigned row = d_warp, i = 0;
         i < CoreDims::W_DOWN_TILE / (CoreDims::PREFETCH_WARP_COUNT / 2);
         row += CoreDims::PREFETCH_WARP_COUNT / 2, i++) {
      unsigned col = d_thread * chunk_size;
      // "clever" condition to allow for compile-time optimization (becomes
      // no-op on H200)
      if (Dims::N == 2 * CoreDims::THREADS_PER_WARP * chunk_size ||
          col < Dims::N) {
        copy128(shm->w[w_index].down[row][col],
                weights[(row * Dims::N + col) / sizeof(OpaqueElement)], pipe);
      }
    }
  }

  // request Scale tile
  if (d_warp == 0) {
    const unsigned chunk_size = 16 / sizeof(*expert_scales_down);
    if (d_thread < CoreDims::W_DOWN_TILE / chunk_size) {
      copy128(shm->scale[w_index][chunk_size * d_thread],
              expert_scales_down[id * Dims::HIDDEN_STATES + base_row +
                                 chunk_size * d_thread],
              pipe);
    }
  }
}

/**
 * @brief Initiate the copy of up-projection output ('temp tokens') from Global
 * to Shared Memory
 *
 * This device function issues the asynchronous data copy requests for a tile of
 * token activations. The copy operations will be queued in the given @a pipe,
 * which the caller must use to wait for their completion.
 *
 * The number of tokens copied is limited by the capacity of @a dest as well as
 * the remaining range of tokens specified by @a expert and the starting offset
 * @a a_row.
 *
 * @note Like all prefetching functions, this function must only be called by
 *       threads in prefetch warps.
 *
 * @param source Pointer activation array of shape [BS, N] in row-major order.
 *               Individual elements are in FP32 format.
 *               Stored in Global Memory.
 * @param expert Specified the range of rows relevant for the current expert.
 * @param a_row Offset of the first row within the range of @a expert to copy.
 * @param dest Shared Memory tile to store the copy.
 * @param pipe Asynchronous completion pipe to use.
 */
template <typename Dims, std::size_t Rows, std::size_t Cols>
__device__ inline void moe_request_temp_token(
    const T_element* __restrict__ source, const ExpertRef& expert,
    unsigned a_row, T_element (&dest)[Rows][Cols],
    cuda::pipeline<cuda::thread_scope_thread>& pipe) {
  // position within the block
  using CoreDims = MoECoreDims<Dims>;
  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_prefetch_warp<Dims>();

  // async transfers are 16 bytes / thread
  const unsigned chunk_size = 16 / sizeof(*source);

  const T_element* t = &source[expert.first_token * Dims::N];
  unsigned int a_rows = expert.last_token - expert.first_token;
  for (unsigned i = warp; i < Rows; i += CoreDims::PREFETCH_WARP_COUNT) {
    if (i + a_row < a_rows) {
      for (unsigned col = thread * chunk_size; col < Cols;
           col += CoreDims::THREADS_PER_WARP * chunk_size) {
        copy128(dest[i][col], t[(i + a_row) * Dims::N + col], pipe);
      }
    }
  }
}

/**
 * @brief Run partial the MMAs on all warps in parallel.
 *
 * This device function multiplies the given @a temps tile of activations with
 * the tile of @a weights, multiplies the result with the respective @a scale
 * and stores it in @a partial_result.
 *
 * The output can be filtered, i.e. @a store_row0 and @a store_row1 control
 * whether the results for the respective tokens shall be written.  This allows
 * the called to optionally interleave the result of multiple experts by
 * selectively suppressing results in the output.
 *
 * @note Like all calculation functions, this function must only be called by
 *       threads in calculation warps.
 *
 * @param weights Weights tile. Elements are in __nv_fp8_e4m3 format.
 * @param scale Scaling factor for each row of @a weights.
 * @param temps Activations tile to multiply with @a weights.
 * @param store_row0 If @c false , suppress the output of the first token.
 * @param store_row1 If @c false , suppress the output of the second token.
 * @param partial_result .
 */
template <typename Dims, std::size_t Rows, std::size_t Cols,
          std::size_t OutCols>
__device__ inline void moe_down_mult(
    const W_element __restrict__ (&weights)[Rows][Cols],
    const float* __restrict__ scale,
    const T_element __restrict__ (&temps)[Dims::N], bool store_row0,
    bool store_row1,
    float (&partial_result)[Rows / 2 + MoECoreDims<Dims>::CALC_WARP_COUNT / 2]
                           [OutCols]) {
  // position within the block
  using CoreDims = MoECoreDims<Dims>;
  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_calc_warp<Dims>();

  for (unsigned w_row = 0; w_row < CoreDims::W_DOWN_TILE;
       w_row += CoreDims::W_DOWN_MMA_TILE) {
    // init accumulators
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    // run partial scalar products
    for (unsigned base_col = warp * CoreDims::K_TILE, i = 0;
         i < Dims::N / CoreDims::BLOCK_STRIDE;
         base_col += CoreDims::BLOCK_STRIDE, i++) {
      unsigned far_row =
          (Rows % 16 == 8 && w_row + 8 == Rows) ? w_row : w_row + 8;
      __nv_fp8x4_e4m3 w0 =
          *(__nv_fp8x4_e4m3*)&weights[w_row + thread / 4]
                                     [base_col + 4 * (thread % 4) + 0];
      __nv_fp8x4_e4m3 w1 =
          *(__nv_fp8x4_e4m3*)&weights[far_row + thread / 4]
                                     [base_col + 4 * (thread % 4) + 0];
      __nv_fp8x4_e4m3 w2 =
          *(__nv_fp8x4_e4m3*)&weights[w_row + thread / 4]
                                     [base_col + 4 * (thread % 4) + 16];
      __nv_fp8x4_e4m3 w3 =
          *(__nv_fp8x4_e4m3*)&weights[far_row + thread / 4]
                                     [base_col + 4 * (thread % 4) + 16];

      float4 b0 = *(float4*)&temps[base_col + 4 * (thread % 4) + 0];
      float4 b1 = *(float4*)&temps[base_col + 4 * (thread % 4) + 16];

      mma_fp8_tf32(d0, d1, d2, d3, w0, w1, w2, w3, b0, b1, d0, d1, d2, d3);
    }

    // load scales (quick-ish as they are stored in SHM)
    float ws0 = scale[(thread / 4) + w_row + 0];
    float ws1 = scale[(thread / 4) + w_row + 8];

    d0 *= ws0;
    d1 *= ws0;
    d2 *= ws1;
    d3 *= ws1;

    if (store_row0) {
      partial_result[w_row / 2 + warp][thread + 0] = d0;
      partial_result[w_row / 2 + warp][thread + 64] = d2;
    }
    if (store_row1) {
      partial_result[w_row / 2 + warp][thread + 32] = d1;
      partial_result[w_row / 2 + warp][thread + 96] = d3;
    }
  }
}

/**
 * @brief Performs the MMA result reduction.
 *
 * This device function sums up the partial scalar products created by all warps
 * and stores the results in Global Memory.  The tile to be written is
 * implicitly determined by the @c blockIdx.
 *
 * The output can be filtered, i.e. @a store_row0 and @a store_row1 control
 * whether the results for the respective rows @a row0 and @a row1 shall be
 * written.  This allows the called to always process data at the full tile size
 * and simply suppress superfluous results in the output.
 *
 * @param partial_result Array of MMA results from all warps of shape [20, 4,
 * THREADS] in row-major order. Individual elements are in FP32 format.
 * @param store_row0 Specifies if result in @a row0 shall be stored.
 * @param store_row1 Specifies if result in @a row1 shall be stored.
 * @param row0 Row to store the scalar products for the first token.
 * @param row1 Row to store the scalar products for the second token.
 * @param result Pointer to the output array of shape [BS, N] in row-major
 * order. Individual elements are in FP32 format.
 */
template <typename Dims, std::size_t Rows, std::size_t Cols>
__device__ inline void moe_down_reduction(
    const float (&partial_result)[Rows][Cols], bool store_row0, bool store_row1,
    unsigned row0, unsigned row1, R_element* __restrict__ result) {
  // position within the block
  using CoreDims = MoECoreDims<Dims>;
  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_any_warp<Dims>();

  // starting row to process
  const unsigned base_row = blockIdx.x * CoreDims::W_DOWN_TILE;

  // reduction and output
  for (unsigned w_row = warp * CoreDims::W_DOWN_MMA_TILE;
       w_row < CoreDims::W_DOWN_TILE;
       w_row += CoreDims::W_DOWN_MMA_TILE * CoreDims::TOTAL_WARP_COUNT) {
    float d0 = partial_result[w_row / 2][thread + 0];
    float d1 = partial_result[w_row / 2][thread + 32];
    float d2 = partial_result[w_row / 2][thread + 64];
    float d3 = partial_result[w_row / 2][thread + 96];

    // combine results
    for (unsigned i = 1; i < CoreDims::CALC_WARP_COUNT; ++i) {
      d0 += partial_result[w_row / 2 + i][thread + 0];
      d1 += partial_result[w_row / 2 + i][thread + 32];
      d2 += partial_result[w_row / 2 + i][thread + 64];
      d3 += partial_result[w_row / 2 + i][thread + 96];
    }

    // write final result. Only write valid lines
    if (store_row0) {
      result[row0 * Dims::HIDDEN_STATES + (thread / 4) + base_row + w_row + 0] =
          d0;
      if (CoreDims::W_DOWN_TILE % 16 == 0 ||
          w_row + 8 < CoreDims::W_DOWN_TILE) {
        result[row0 * Dims::HIDDEN_STATES + (thread / 4) + base_row + w_row +
               8] = d2;
      }
    }
    if (store_row1) {
      result[row1 * Dims::HIDDEN_STATES + (thread / 4) + base_row + w_row + 0] =
          d1;
      if (CoreDims::W_DOWN_TILE % 16 == 0 ||
          w_row + 8 < CoreDims::W_DOWN_TILE) {
        result[row1 * Dims::HIDDEN_STATES + (thread / 4) + base_row + w_row +
               8] = d3;
      }
    }
  }
}

/**
 * @brief Standard kernel for the second GEMM ("down projection").
 *
 * This device function processes @c BS tokens, grouped by expert in internal
 * batches of 8 tokens. The experts to use and the respective list of tokes for
 * each of them is given by @a spec.
 *
 * Activations are taken from temporary storage in @a spec and all non-expert
 * data is taken from
 * @a shmem. Outputs are in the same order as the input to "up projection".
 *
 * @param expert_weights_up Pointer token weights array of shape [NUM_EXPERTS,
 * HIDDEN_STATES, N] in expert, row-major order. Individual elements are in
 * __nv_fp8_e4m3 format. Stored in Global Memory.
 * @param expert_scales_up Pointer weights scales array of shape [NUM_EXPERTS,
 * HIDDEN_STATES] in row-major order. Individual elements are in FP32 format.
 *                         Stored in Global Memory.
 * @param result Global Memory array of shape [BS, HIDDEN_STATES] in row-major
 * order, receiving the output.  Individual elements are in __nv_bfloat16
 * format.
 * @param spec Global Memory struct containing the scaled input token
 * activations.
 * @param shmem Shared Memory struct containing the expert<=>token mapping,
 * activation weights and will be uses as local scratch pad store for faster
 * operation.
 */
template <typename Dims>
__device__ inline void moe_down_projection_normal(
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down,
    R_element* __restrict__ result, MoEGemmSpec<Dims>* __restrict__ spec,
    MoE_SHM<Dims>* __restrict__ shmem) {
  using CoreDims = MoECoreDims<Dims>;
  using MoE_SHM = MoE_SHM<Dims>;

  // position within the block
  const unsigned thread = get_thread<Dims>();

  // required for async copies
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape4 = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));

  // commonly used values
  typename MoE_SHM::U::Gemm2Data* shm = &shmem->u.gemm2;
  const unsigned base_row = blockIdx.x * CoreDims::W_DOWN_TILE;
  std::uint32_t expert_count = shmem->expert_count;

  const ExpertRef& expert = shmem->experts[0];
  std::uint32_t id = expert.id;

  assert(expert_count > 0);

  // request first W tile
  if (is_prefetch_warp<Dims>()) {
    pipe.producer_acquire();
    moe_request_down_expert<Dims>(expert_weights_down, expert_scales_down, id,
                                  shm, 0, pipe);

    // request first T tile
    moe_request_temp_token<Dims>(spec->temp, expert, 0, shm->t[0], pipe);
    pipe.producer_commit();
  }

  // current tile indexes
  // will be toggled to 0 at the start of the first loop iterations
  std::uint32_t t_index = 1;
  std::uint32_t w_index = 1;

  // loop over all experts
  for (std::uint32_t e = 0; e < expert_count; ++e) {
    const ExpertRef& expert = shmem->experts[e];
    unsigned int a_rows = expert.last_token - expert.first_token;
    w_index ^= 1;

    // process all activations for this set of weights
    for (unsigned a_row = 0; a_row < a_rows; a_row += CoreDims::T_TILE) {
      t_index ^= 1;

      // wait for currently needed data to come in
      cuda::pipeline_consumer_wait_prior<0>(pipe);
      __syncthreads();

      // request next T, S and W tiles as needed
      if (is_prefetch_warp<Dims>()) {
        pipe.producer_acquire();
        if (e + 1 < expert_count && a_row == 0) {
          // request first W tile
          moe_request_down_expert<Dims>(expert_weights_down, expert_scales_down,
                                        shmem->experts[e + 1].id, shm,
                                        w_index ^ 1, pipe);
        }
        if (a_row + CoreDims::T_TILE < a_rows) {
          // request the next T tile for same expert
          moe_request_temp_token<Dims>(spec->temp, expert,
                                       a_row + CoreDims::T_TILE,
                                       shm->t[t_index ^ 1], pipe);
        } else if (e + 1 < expert_count) {
          // request the first T tile for next expert
          moe_request_temp_token<Dims>(spec->temp, shmem->experts[e + 1], 0,
                                       shm->t[t_index ^ 1], pipe);
        }
        pipe.producer_commit();
      } else {
        // matrix multiplication step
        static_assert(CoreDims::W_DOWN_TILE % 8 == 0);
        moe_down_mult<Dims>(shm->w[w_index], shm->scale[w_index],
                            shm->t[t_index][thread / 4], true, true,
                            shm->partial_result);
      }

      __syncthreads();

      // reduction and output
      moe_down_reduction<Dims>(
          shm->partial_result, a_row + (thread % 4) * 2 < a_rows,
          a_row + (thread % 4) * 2 + 1 < a_rows,
          shmem->token_indexes[expert.first_token + a_row + (thread % 4) * 2],
          shmem->token_indexes[expert.first_token + a_row + (thread % 4) * 2 +
                               1],
          result);
    }

    __syncthreads();
  }
}

/**
 * @brief 'Tiny' kernel for the second GEMM ("down projection").
 *
 * This device function processes up to 8 tokens, store in @a spec.
 * The experts to use with each token is given by @a shm, the function will
 * apply them to all and only filter the output accordingly. The weights and
 * scales for the first expert have already been prefetched into @a shmem as
 * well.
 *
 * All non-expert data is taken from our temporary storage in @a shmem
 * and results will be written to @a result. Output order is the input token
 * order.
 *
 * @param token_count Number of input tokens.
 * @param expert_weights_up Pointer token weights array of shape [NUM_EXPERTS,
 * HIDDEN_STATES, N] in expert, row-major order. Individual elements are in
 * __nv_fp8_e4m3 format. Stored in Global Memory.
 * @param expert_scales_up Pointer weights scales array of shape [NUM_EXPERTS,
 * HIDDEN_STATES] in row-major order. Individual elements are in FP32 format.
 *                         Stored in Global Memory.
 * @param w_index Index of the tile in @a shmem that contains the prefetched
 * weight data for the first expert.
 * @param result Global Memory array of shape [BS, HIDDEN_STATES] in row-major
 * order, receiving the output.  Individual elements are in __nv_bfloat16
 * format.
 * @param spec Global Memory struct containing the token activations.
 * @param shmem Shared Memory struct containing the expert<=>token mapping,
 * activation weights and will be uses as local scratch pad store for faster
 * operation.
 * @param pipe Asynchronous completion pipe to use with prefetching.
 */
template <typename Dims>
__device__ inline void moe_down_projection_tiny(
    std::uint32_t token_count,
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down, std::uint32_t w_index,
    R_element* __restrict__ result, MoEGemmSpec<Dims>* __restrict__ spec,
    MoE_SHM<Dims>* __restrict__ shmem,
    cuda::pipeline<cuda::thread_scope_thread>& pipe) {
  using CoreDims = MoECoreDims<Dims>;
  using MoE_SHM = MoE_SHM<Dims>;

  assert(w_index == 0 || w_index == 1);

  // position within the block
  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_any_warp<Dims>();

  // commonly used values
  typename MoE_SHM::U::TinyData* shm = &shmem->u.tiny;
  const unsigned base_row = blockIdx.x * CoreDims::W_DOWN_TILE;
  std::uint32_t expert_count = shmem->expert_count;
  std::uint64_t expert_mask = shmem->expert_mask;
  std::uint64_t expert_ids = shmem->expert_ids;

  assert(expert_count > 0);
  std::uint32_t id = expert_ids & 0xff;

  // request data for the first expert
  pipe.producer_acquire();

  // request T (W has already been requested by up-projection)
  if (warp < token_count) {
    for (unsigned col = thread * 4, i = 0;
         i < Dims::N / (CoreDims::THREADS_PER_WARP * 4);
         i++, col += CoreDims::THREADS_PER_WARP * 4) {
      copy128(shm->a.down[warp][col], spec->temp[warp * Dims::N + col], pipe);
    }
  }

  pipe.producer_commit();

  // loop over all experts
  for (std::uint32_t e = 0; e < expert_count; ++e) {
    id = expert_ids & 0xff;
    expert_ids >>= 8;

    // wait for currently needed data to come in
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    __syncthreads();

    // request next S and W tiles as needed
    if (is_prefetch_warp<Dims>()) {
      pipe.producer_acquire();
      if (e + 1 < expert_count) {
        moe_request_down_expert_tiny<Dims>(
            expert_weights_down, expert_scales_down, expert_ids & 0xff, shm,
            w_index ^ 1, pipe);
      }
      pipe.producer_commit();
    } else {
      // matrix multiplication step
      static_assert(CoreDims::W_DOWN_TILE % 8 == 0);
      std::uint32_t row0 = (thread % 4) * 2 + 0;
      std::uint32_t row1 = (thread % 4) * 2 + 1;
      bool store_row0 = (expert_mask >> (row0 * 8) & 0xff) == id;
      bool store_row1 = (expert_mask >> (row1 * 8) & 0xff) == id;

      moe_down_mult<Dims>(shm->w[w_index].down, shm->scale[w_index],
                          shm->a.down[thread / 4], store_row0, store_row1,
                          shm->partial_result.down);
    }

    __syncthreads();
    w_index ^= 1;
  }

  // reduction and output
  std::uint32_t row0 = (thread % 4) * 2 + 0;
  std::uint32_t row1 = (thread % 4) * 2 + 1;
  moe_down_reduction<Dims>(shm->partial_result.down, row0 < token_count,
                           row1 < token_count, row0, row1, result);
}

/**
 * @brief Forwards to @c moe_down_projection_normal.
 *
 * @see moe_down_projection_normal
 */
template <typename Dims>
__device__ void moe_down_projection(
    std::uint32_t token_count,
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down,
    R_element* __restrict__ result, MoEGemmSpec<Dims>* __restrict__ spec,
    MoE_SHM<Dims>* __restrict__ shmem) {
  static_assert(Dims::BS > 8,
                "Tiny is handled by its own kernel. Do not use "
                "moe_down_projection for BS<=8");
  moe_down_projection_normal<Dims>(expert_weights_down, expert_scales_down,
                                   result, spec, shmem);
}

}  // namespace moe_monokernel

#endif
