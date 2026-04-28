
#pragma once
#ifndef MOE_DOWN_PROJECTION_CU
  #define MOE_DOWN_PROJECTION_CU

  #ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
    #error Do not include this file directly.
  #endif

  #include <cuda.h>
  #include <cuda/pipeline>
  #include <cuda_fp8.h>

  #include "moe_interface.h"
  #include "moe_internal.h"
  #include "ptx_utils.h"
  #include "moe_debug.h"

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

  // request Scale tile — block-wise 2D scale
  if (d_warp == 0) {
    constexpr uint32_t SCALE_TILE_SIZE =
        MoE_SHM<Dims>::U::Gemm2Data::DOWN_SCALE_TILE_SIZE;
    constexpr uint32_t COL_BLOCKS = (Dims::N + 127) / 128;
    if (d_thread < SCALE_TILE_SIZE) {
      uint32_t rb = d_thread / COL_BLOCKS;
      uint32_t cb = d_thread % COL_BLOCKS;
      uint32_t global_rb = (base_row / 128) + rb;
      shm->scale[w_index][d_thread] =
          expert_scales_down[id * Dims::DOWN_SCALE_ROWS *
                                 Dims::DOWN_SCALE_COLS +
                             global_rb * Dims::DOWN_SCALE_COLS + cb];
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
 *               Individual elements are in BF16 format.
 *               Stored in Global Memory.
 * @param expert Specified the range of rows relevant for the current expert.
 * @param a_row Offset of the first row within the range of @a expert to copy.
 * @param dest Shared Memory tile to store the copy.
 * @param pipe Asynchronous completion pipe to use.
 */
template <typename Dims, std::size_t Rows, std::size_t Cols>
__device__ inline void moe_request_temp_token(
    const A_element* __restrict__ source, const ExpertRef& expert,
    unsigned a_row, A_element (&dest)[Rows][Cols],
    cuda::pipeline<cuda::thread_scope_thread>& pipe) {
  // position within the block
  using CoreDims = MoECoreDims<Dims>;
  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_prefetch_warp<Dims>();

  // async transfers are 16 bytes / thread
  const unsigned chunk_size = 16 / sizeof(*source);

  const A_element* t = &source[expert.first_token * Dims::N];
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
 * @brief Top-K single-pass down-projection reduction with weighted
 * accumulation.
 *
 *  - @p row0 / @p row1 are sorted positions; the original token index is
 *    looked up via @c shmem->path.bs64.token_indexes_topk[sorted_pos].
 *  - The output is accumulated with @c += (not @c =) because each original
 *    token may receive contributions from multiple experts.
 */
template <typename Dims, std::size_t Rows, std::size_t Cols>
__device__ inline void moe_down_reduction_topk(
    const float (&partial_result)[Rows][Cols], bool store_row0, bool store_row1,
    unsigned sorted_row0, unsigned sorted_row1,
    const MoE_SHM<Dims>* __restrict__ shmem, R_element* __restrict__ result) {
  using CoreDims = MoECoreDims<Dims>;
  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_any_warp<Dims>();
  const unsigned base_row = blockIdx.x * CoreDims::W_DOWN_TILE;

  // Map sorted positions to original token indexes
  unsigned orig_row0 =
      store_row0 ? shmem->path.bs64.token_indexes_topk[sorted_row0] : 0;
  unsigned orig_row1 =
      store_row1 ? shmem->path.bs64.token_indexes_topk[sorted_row1] : 0;

  for (unsigned w_row = warp * CoreDims::W_DOWN_MMA_TILE;
       w_row < CoreDims::W_DOWN_TILE;
       w_row += CoreDims::W_DOWN_MMA_TILE * CoreDims::TOTAL_WARP_COUNT) {
    float d0 = partial_result[w_row / 2][thread + 0];
    float d1 = partial_result[w_row / 2][thread + 32];
    float d2 = partial_result[w_row / 2][thread + 64];
    float d3 = partial_result[w_row / 2][thread + 96];

    for (unsigned i = 1; i < CoreDims::CALC_WARP_COUNT; ++i) {
      d0 += partial_result[w_row / 2 + i][thread + 0];
      d1 += partial_result[w_row / 2 + i][thread + 32];
      d2 += partial_result[w_row / 2 + i][thread + 64];
      d3 += partial_result[w_row / 2 + i][thread + 96];
    }

    // Accumulate into original token positions (+=) weighted by routing_weight
    // path.bs64.token_weights[sorted_pos] = routing_weight (populated in
    // prepare_moe_topk_BSx_Ey — we read it directly to avoid a separate
    // copy-back pass).
    if (store_row0) {
      float rw0 = shmem->path.bs64.token_weights[sorted_row0];
      unsigned col0 =
          orig_row0 * Dims::HIDDEN_STATES + (thread / 4) + base_row + w_row;
      result[col0 + 0] = (R_element)((float)result[col0 + 0] + rw0 * d0);
      if (CoreDims::W_DOWN_TILE % 16 == 0 || w_row + 8 < CoreDims::W_DOWN_TILE)
        result[col0 + 8] = (R_element)((float)result[col0 + 8] + rw0 * d2);
    }
    if (store_row1) {
      float rw1 = shmem->path.bs64.token_weights[sorted_row1];
      unsigned col1 =
          orig_row1 * Dims::HIDDEN_STATES + (thread / 4) + base_row + w_row;
      result[col1 + 0] = (R_element)((float)result[col1 + 0] + rw1 * d1);
      if (CoreDims::W_DOWN_TILE % 16 == 0 || w_row + 8 < CoreDims::W_DOWN_TILE)
        result[col1 + 8] = (R_element)((float)result[col1 + 8] + rw1 * d3);
    }
  }
}

/**
 * @brief Down-projection MMA with fp8 activations (shared by BS8 and BS64).
 *
 * Uses mma_fp8_fp8 (m16n8k32) — both weights and activations are fp8.
 *
 * D-output mapping (from PTX spec):
 *   d0 → D[t/4,     (t%4)*2]     row=t/4,     col=(t%4)*2
 *   d1 → D[t/4,     (t%4)*2 + 1] row=t/4,     col=(t%4)*2 + 1
 *   d2 → D[t/4 + 8, (t%4)*2]     row=t/4 + 8, col=(t%4)*2
 *   d3 → D[t/4 + 8, (t%4)*2 + 1] row=t/4 + 8, col=(t%4)*2 + 1
 *
 * D-columns are tokens. d0/d2 use token=(t%4)*2, d1/d3 use token=(t%4)*2+1.
 * Each token has its own per-block activation scale.
 */
template <typename Dims, std::size_t Rows, std::size_t Cols,
          std::size_t OutRows, std::size_t OutCols, std::size_t ScaleRows,
          std::size_t ScaleCols>
__device__ inline void moe_down_mult_fp8(
    const W_element __restrict__ (&weights)[Rows][Cols],
    const float* __restrict__ scale, const AQ_element* __restrict__ act_row,
    const float (&act_block_scales_all)[ScaleRows][ScaleCols],
    std::uint32_t tok_02, std::uint32_t tok_13, bool store_row0,
    bool store_row1, float (&partial_result)[OutRows][OutCols]) {
  using CoreDims = MoECoreDims<Dims>;
  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_calc_warp<Dims>();

  for (unsigned w_row = 0; w_row < CoreDims::W_DOWN_TILE;
       w_row += CoreDims::W_DOWN_MMA_TILE) {
    constexpr uint32_t COL_BLOCKS = (Dims::N + 127) / 128;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    for (unsigned base_col = warp * CoreDims::K_TILE, i = 0;
         i < Dims::N / CoreDims::BLOCK_STRIDE;
         base_col += CoreDims::BLOCK_STRIDE, i++) {
      float md0 = 0.f, md1 = 0.f, md2 = 0.f, md3 = 0.f;
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

      __nv_fp8x4_e4m3 b0 =
          *(__nv_fp8x4_e4m3*)&act_row[base_col + 4 * (thread % 4) + 0];
      __nv_fp8x4_e4m3 b1 =
          *(__nv_fp8x4_e4m3*)&act_row[base_col + 4 * (thread % 4) + 16];

      mma_fp8_fp8(md0, md1, md2, md3, w0, w1, w2, w3, b0, b1, 0.f, 0.f, 0.f,
                  0.f);

      unsigned n_block = base_col / 128;
      float ws = scale[0 * COL_BLOCKS + n_block];
      // d0/d2 use tok_02's scale; d1/d3 use tok_13's scale
      float as_02 = store_row0 ? act_block_scales_all[tok_02][n_block] : 0.f;
      float as_13 = store_row1 ? act_block_scales_all[tok_13][n_block] : 0.f;

  #if 0  // disabled — too verbose
      if (blockIdx.x == 0 && threadIdx.x == 0 && w_row == 0 && warp == 0 && tok_02 == 0) {
        printf("[DBG MMA t0 bc=%u] md0=%.4f md1=%.4f md2=%.4f md3=%.4f ws=%.6f as_02=%.6f as_13=%.6f\n",
               base_col, md0, md1, md2, md3, ws, as_02, as_13);
      }
  #endif

      d0 += md0 * ws * as_02;
      d1 += md1 * ws * as_13;
      d2 += md2 * ws * as_02;
      d3 += md3 * ws * as_13;
    }

    if (store_row0) {
      partial_result[w_row / 2 + warp][thread + 0] = d0;
      partial_result[w_row / 2 + warp][thread + 64] = d2;
    }
    if (store_row1) {
      partial_result[w_row / 2 + warp][thread + 32] = d1;
      partial_result[w_row / 2 + warp][thread + 96] = d3;
    }

  #if 0  // disabled — too verbose
    if (blockIdx.x == 0 && threadIdx.x == 0 && w_row == 0 && tok_02 == 0) {
      printf("[DBG MMA_FINAL t0 w_row=0 warp=%u] d0=%.4f d1=%.4f d2=%.4f d3=%.4f (stored? s0=%d s1=%d)\n",
             warp, d0, d1, d2, d3, (int)store_row0, (int)store_row1);
    }
  #endif
  }
}

/**
 * @brief Top-K single-pass down-projection for BS > 8.
 *
 * Uses the same fp8 MMA approach as BS8:
 *   1. Fetch bf16 SiLU output from global memory → t_bf16 staging buffer
 *   2. Quantize bf16 → fp8 with per-token block-wise scales → t_fp8
 *   3. MMA: fp8 weights × fp8 activations via mma_fp8_fp8 (m16n8k32)
 *   4. Reduce with correct D-mapping token assignment
 *
 * Uses token_indexes_topk (sorted positions → original tokens).
 * Calls moe_down_reduction_topk which accumulates with +=.
 * The output buffer must be zeroed before calling this function.
 */
template <typename Dims>
__device__ inline void moe_down_projection_topk(
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down,
    R_element* __restrict__ result, MoEGemmSpec<Dims>* __restrict__ spec,
    MoE_SHM<Dims>* __restrict__ shmem) {
  static_assert(Dims::BS > 8,
                "Tiny is handled by its own kernel. Do not use "
                "moe_down_projection_topk for BS<=8");
  using CoreDims = MoECoreDims<Dims>;
  using MoE_SHM_t = MoE_SHM<Dims>;

  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_any_warp<Dims>();

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  typename MoE_SHM_t::U::Gemm2Data* shm = &shmem->u.gemm2;
  std::uint32_t expert_count = shmem->expert_count;

  const ExpertRef& first_expert = shmem->experts[0];

  assert(expert_count > 0);

  // Quantization constants
  constexpr float FP8_MAX = 448.f;
  constexpr float FP8_MAX_INV = 1.0f / 448.f;
  constexpr std::uint32_t FLOATS_PER_LOAD = 4;
  constexpr std::uint32_t ACT_DOWN_BLOCK = 128;
  constexpr std::uint32_t NUM_DOWN_BLOCKS = Dims::N / ACT_DOWN_BLOCK;
  static_assert(Dims::N % ACT_DOWN_BLOCK == 0,
                "N must be divisible by activation block size");

  // Prime: prefetch first expert's weights + first token tile (bf16)
  if (is_prefetch_warp<Dims>()) {
    pipe.producer_acquire();
    moe_request_down_expert<Dims>(expert_weights_down, expert_scales_down,
                                  first_expert.id, shm, 0, pipe);
    moe_request_temp_token<Dims>(spec->temp_bf16, first_expert, 0,
                                 shm->t_bf16[0], pipe);
    pipe.producer_commit();
  }

  std::uint32_t t_index = 1;  // ping-pong for t_bf16 fetch
  std::uint32_t w_index = 1;  // ping-pong for weights
  std::uint32_t q_index = 0;  // ping-pong for t_fp8 quantized buffer

  for (std::uint32_t e = 0; e < expert_count; ++e) {
    const ExpertRef& expert = shmem->experts[e];
    unsigned int a_rows = expert.last_token - expert.first_token;
    w_index ^= 1;

    for (unsigned a_row = 0; a_row < a_rows; a_row += CoreDims::T_TILE) {
      t_index ^= 1;

      // Wait for bf16 temp tokens + weights to arrive
      cuda::pipeline_consumer_wait_prior<0>(pipe);
      __syncthreads();

  #ifdef DEBUG_MOE_PRINT
      if (blockIdx.x == 0 && threadIdx.x == 0 && e == 0 && a_row == 0) {
        printf("[DBG64 DOWN_BF16 e=0] t_bf16[%u][0][0..7]:", t_index);
        for (int i = 0; i < 8; i++)
          printf(" %.4f", (float)shm->t_bf16[t_index][0][i]);
        printf("\n");
        printf("[DBG64 DOWN_BF16 e=0] t_bf16[%u][0][508..511]:", t_index);
        for (int i = 508; i < 512; i++)
          printf(" %.4f", (float)shm->t_bf16[t_index][0][i]);
        printf("\n");
      }
      // Also print for expert 1 to check later-expert data
      if (blockIdx.x == 0 && threadIdx.x == 0 && e == 1 && a_row == 0) {
        printf(
            "[DBG64 DOWN_BF16 e=1] expert_id=%u first=%u last=%u a_rows=%u\n",
            shmem->experts[1].id, shmem->experts[1].first_token,
            shmem->experts[1].last_token,
            shmem->experts[1].last_token - shmem->experts[1].first_token);
        printf("[DBG64 DOWN_BF16 e=1] t_bf16[%u][0][0..7]:", t_index);
        for (int i = 0; i < 8; i++)
          printf(" %.4f", (float)shm->t_bf16[t_index][0][i]);
        printf("\n");
        // Also print what's in global memory at that address
        unsigned ft = shmem->experts[1].first_token;
        printf("[DBG64 DOWN_BF16 e=1] spec->temp_bf16[ft=%u][0..7]:", ft);
        for (int i = 0; i < 8; i++)
          printf(" %.4f", (float)spec->temp_bf16[ft * Dims::N + i]);
        printf("\n");
      }
  #endif

      if (is_prefetch_warp<Dims>()) {
        // Prefetch next tile while calc warps quantize + MMA
        pipe.producer_acquire();
        if (e + 1 < expert_count && a_row == 0) {
          moe_request_down_expert<Dims>(expert_weights_down, expert_scales_down,
                                        shmem->experts[e + 1].id, shm,
                                        w_index ^ 1, pipe);
        }
        if (a_row + CoreDims::T_TILE < a_rows) {
          moe_request_temp_token<Dims>(spec->temp_bf16, expert,
                                       a_row + CoreDims::T_TILE,
                                       shm->t_bf16[t_index ^ 1], pipe);
        } else if (e + 1 < expert_count) {
          moe_request_temp_token<Dims>(spec->temp_bf16, shmem->experts[e + 1],
                                       0, shm->t_bf16[t_index ^ 1], pipe);
        }
        pipe.producer_commit();
      } else {
        // ── Calc warps: quantize bf16 → fp8 ──────────────────────────────
        // Each calc warp handles one or more tokens.
        // Per-block (1, 128) quantization: each 128-element block gets its
        // own scale. With N=512, that's 4 blocks per row.
        const std::uint32_t cw = get_calc_warp<Dims>();
        unsigned valid_tokens = min((unsigned)CoreDims::T_TILE, a_rows - a_row);
        for (std::uint32_t tok = cw; tok < valid_tokens;
             tok += CoreDims::CALC_WARP_COUNT) {
          float regs[NUM_DOWN_BLOCKS * 4];

  #pragma unroll
          for (std::uint32_t blk = 0; blk < NUM_DOWN_BLOCKS; ++blk) {
            std::uint32_t blk_start = blk * ACT_DOWN_BLOCK;
            std::uint32_t col = blk_start + thread * FLOATS_PER_LOAD;

            // Read bf16 from staging buffer and convert to fp32 in registers.
            // 4 bf16 values per thread per block iteration, read as 2×bf162.
            __nv_bfloat162 bf_01 = *reinterpret_cast<const __nv_bfloat162*>(
                &shm->t_bf16[t_index][tok][col + 0]);
            __nv_bfloat162 bf_23 = *reinterpret_cast<const __nv_bfloat162*>(
                &shm->t_bf16[t_index][tok][col + 2]);
            float2 f01 = __bfloat1622float2(bf_01);
            float2 f23 = __bfloat1622float2(bf_23);
            regs[blk * 4 + 0] = f01.x;
            regs[blk * 4 + 1] = f01.y;
            regs[blk * 4 + 2] = f23.x;
            regs[blk * 4 + 3] = f23.y;

            float local_max = fmaxf(fmaxf(fabsf(f01.x), fabsf(f01.y)),
                                    fmaxf(fabsf(f23.x), fabsf(f23.y)));
            float blk_max = warp_reduce_max_float(local_max);
            if (blk_max < __FLT_MIN__) blk_max = 1.f;

            float blk_scale = blk_max * FP8_MAX_INV;
            float blk_inv_scale = FP8_MAX / blk_max;

            // Quantize → write fp8 to SHM
            __nv_fp8x4_e4m3 q{float4{regs[blk * 4 + 0] * blk_inv_scale,
                                     regs[blk * 4 + 1] * blk_inv_scale,
                                     regs[blk * 4 + 2] * blk_inv_scale,
                                     regs[blk * 4 + 3] * blk_inv_scale}};
            *reinterpret_cast<__nv_fp8x4_e4m3*>(
                &shm->t_fp8[q_index][tok][col]) = q;

            if (thread == 0) shm->t_scale[q_index][tok][blk] = blk_scale;
          }
        }
      }

      __syncthreads();

  #ifdef DEBUG_MOE_PRINT
      if (blockIdx.x == 0 && threadIdx.x == 0 && e == 0 && a_row == 0) {
        printf("[DBG64 DOWN_QUANT e=0] t_fp8[%u][0][0..7]:", q_index);
        for (int i = 0; i < 8; i++)
          printf(" %.4f", (float)shm->t_fp8[q_index][0][i]);
        printf("\n");
        printf("[DBG64 DOWN_QUANT e=0] t_scale[%u][0][0..3]:", q_index);
        for (int i = 0; i < 4; i++)
          printf(" %.6f", shm->t_scale[q_index][0][i]);
        printf("\n");
        printf("[DBG64 DOWN_QUANT e=0] w[%u] row0[0..7]:", w_index);
        for (int i = 0; i < 8; i++)
          printf(" %.4f", (float)shm->w[w_index][0][i]);
        printf("\n");
        printf("[DBG64 DOWN_QUANT e=0] w[%u] row1[0..7]:", w_index);
        for (int i = 0; i < 8; i++)
          printf(" %.4f", (float)shm->w[w_index][1][i]);
        printf("\n");
        printf("[DBG64 DOWN_QUANT e=0] w_scale[%u][0..3]:", w_index);
        for (int i = 0; i < 4; i++) printf(" %.6f", shm->scale[w_index][i]);
        printf("\n");
      }
  #endif

      // ── MMA: fp8 weights × fp8 activations ────────────────────────────
      if (!is_prefetch_warp<Dims>()) {
        static_assert(CoreDims::W_DOWN_TILE % 8 == 0);

        // D-output mapping (m16n8k32):
        //   d0/d2 → token=(t%4)*2,  d1/d3 → token=(t%4)*2+1
        const std::uint32_t tok_02 = (thread % 4) * 2;
        const std::uint32_t tok_13 = (thread % 4) * 2 + 1;
        bool s0 = tok_02 < min((unsigned)CoreDims::T_TILE, a_rows - a_row);
        bool s1 = tok_13 < min((unsigned)CoreDims::T_TILE, a_rows - a_row);

        moe_down_mult_fp8<Dims>(shm->w[w_index], shm->scale[w_index],
                                shm->t_fp8[q_index][thread / 4],
                                shm->t_scale[q_index], tok_02, tok_13, s0, s1,
                                shm->partial_result);
      }

      __syncthreads();

  #ifdef DEBUG_MOE_PRINT
      if (blockIdx.x == 0 && threadIdx.x == 0 && e == 0 && a_row == 0) {
        printf("[DBG64 DOWN_MMA e=0] partial_result[0][0..7]:");
        for (int i = 0; i < 8; i++) printf(" %.4f", shm->partial_result[0][i]);
        printf("\n");
        printf("[DBG64 DOWN_MMA e=0] partial_result[1][0..7]:");
        for (int i = 0; i < 8; i++) printf(" %.4f", shm->partial_result[1][i]);
        printf("\n");
      }
  #endif

      // ── Reduce: accumulate into original token positions ───────────────
      // D-mapping: d0/d2 → sorted_row0 = (t%4)*2, d1/d3 → sorted_row1
      unsigned sorted_row0 = expert.first_token + a_row + (thread % 4) * 2;
      unsigned sorted_row1 = sorted_row0 + 1;
      moe_down_reduction_topk<Dims>(shm->partial_result,
                                    sorted_row0 < expert.last_token,
                                    sorted_row1 < expert.last_token,
                                    sorted_row0, sorted_row1, shmem, result);

  #ifdef DEBUG_MOE_PRINT
      if (blockIdx.x == 0 && threadIdx.x == 0 && e == 0 && a_row == 0) {
        unsigned orig_tok = shmem->path.bs64.token_indexes_topk[sorted_row0];
        float rw = shmem->path.bs64.token_weights[sorted_row0];
        printf(
            "[DBG64 DOWN_REDUCE e=0] sorted_row0=%u orig_tok=%u rw=%.4f "
            "store=%d\n",
            sorted_row0, orig_tok, rw, (int)(sorted_row0 < expert.last_token));
        printf("[DBG64 DOWN_REDUCE e=0] result[tok=%u][0..7]:", orig_tok);
        for (int i = 0; i < 8; i++)
          printf(" %.4f", (float)result[orig_tok * Dims::HIDDEN_STATES + i]);
        printf("\n");
      }
  #endif

      q_index ^= 1;
    }

    __syncthreads();
  }

  #ifdef DEBUG_MOE_PRINT
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("[DBG64 DOWN_FINAL] result[tok=0][0..7] after all experts:");
    for (int i = 0; i < 8; i++)
      printf(" %.4f", (float)result[0 * Dims::HIDDEN_STATES + i]);
    printf("\n");
  }
  #endif
}

}  // namespace moe_monokernel

namespace moe_monokernel {

/**
 * @brief Split-phase down-projection for BS8: pipelined 4-stage design.
 *
 * After grid.sync(), all blocks have written their SiLU output to
 * spec->temp_bf16.  This function uses a pipelined design with ping-pong
 * w[2] buffers to overlap bf16 fetch, quantization, weight fetch, and MMA:
 *
 *   w[2] union slots have fixed roles (no flipping):
 *     w[0].bf16_buf — holds bf16 intermediate results
 *     w[1].down     — holds fp8 down-projection weights
 *
 *   Stage 0 (prime): All warps fetch expert 0's bf16 intermediate →
 * w[0].bf16_buf Steady-state loop for expert e: Stage A: prefetch warps fetch
 * e's w_down → w[1].down calc warps quantize w[0].bf16_buf → a.down[buf_fp8]
 *     Stage B: prefetch warps fetch (e+1)'s bf16 → w[0].bf16_buf
 *              calc warps MMA a.down[buf_fp8] × w[1].down, reduce, accumulate
 *     Flip: buf_fp8 ^= 1 (only the fp8 activation double-buffer flips)
 *
 * This eliminates the separate bf16_buf[2] array — it now lives inside
 * the w[2] union, saving T_TILE × N × sizeof(bf16) × 2 bytes of SHM.
 *
 * @param expert_weights_down  [E, K, N] fp8 weights in global memory.
 * @param expert_scales_down   [E, K] fp32 scales in global memory.
 * @param top_k                Number of experts per token.
 * @param batch_size           Number of active tokens.
 * @param spec                 Global scratchpad (reads spec->temp_bf16).
 * @param shmem                Shared memory.
 */
template <typename Dims>
__device__ inline void moe_down_projection_BS8_allexperts(
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down, std::uint32_t top_k,
    std::uint32_t batch_size, const MoEGemmSpec<Dims>* __restrict__ spec,
    MoE_SHM<Dims>* __restrict__ shmem) {
  static_assert(Dims::BS <= 8);
  using CoreDims = MoECoreDims<Dims>;
  constexpr uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;

  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_any_warp<Dims>();
  const unsigned base_row_dn = blockIdx.x * CoreDims::W_DOWN_TILE;

  auto* shm = &shmem->u.tiny;
  const std::uint32_t expert_count = shmem->expert_count;

  constexpr float FP8_MAX = 448.f;
  constexpr float FP8_MAX_INV = 1.0f / 448.f;
  constexpr std::uint32_t FLOATS_PER_LOAD = 4;  // 4 bf16 = 8 bytes
  constexpr std::uint32_t COLS_PER_WARP_ITER =
      CoreDims::THREADS_PER_WARP * FLOATS_PER_LOAD;

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  // Fixed slot assignment — no flipping needed:
  //   w[0]: bf16_buf (intermediate results from global memory)
  //   w[1]: down weights (fp8 expert weights)
  // They never alias within the same slot.
  constexpr std::uint32_t BUF_BF16 = 0;  // w[] slot for bf16 intermediates
  constexpr std::uint32_t BUF_W = 1;     // w[] slot for down weights
  std::uint32_t buf_fp8 = 0;             // current a.down[] double-buffer index

  // ── Stage 0: Prime — all warps fetch expert 0's bf16 intermediate ───────
  {
    const std::uint32_t id0 = shmem->experts[0].id;
    pipe.producer_acquire();
    for (std::uint32_t tok = 0; tok < batch_size; ++tok) {
      for (uint32_t k = 0; k < top_k; k++) {
        if (shmem->topk_ids_flat[tok * MAX_TOPK + k] == (uint16_t)id0) {
          std::uint32_t vrow = tok * top_k + k;
          const A_element* src = &spec->temp_bf16[vrow * Dims::N];
          // All warps cooperate: global thread index across all warps
          for (std::uint32_t col =
                   (warp * CoreDims::THREADS_PER_WARP + thread) *
                   FLOATS_PER_LOAD;
               col < Dims::N;
               col += CoreDims::TOTAL_WARP_COUNT * CoreDims::THREADS_PER_WARP *
                      FLOATS_PER_LOAD) {
            const auto shape8 = cuda::aligned_size_t<8>(8);
            cuda::memcpy_async(&shm->w[BUF_BF16].bf16_buf[tok][col], &src[col],
                               shape8, pipe);
          }
          break;
        }
      }
    }
    pipe.producer_commit();
  }
  cuda::pipeline_consumer_wait_prior<0>(pipe);
  __syncthreads();

  // ── Main expert loop ────────────────────────────────────────────────────
  for (std::uint32_t e = 0; e < expert_count; ++e) {
    const std::uint32_t id = shmem->experts[e].id;

    // ── Stage A: prefetch w_down → w[BUF_W] || quantize w[BUF_BF16].bf16_buf →
    // a.down[buf_fp8]
    if (is_prefetch_warp<Dims>()) {
      // Fetch this expert's w_down + scales into w[BUF_W].down
      pipe.producer_acquire();
      {
        const unsigned d_thread_l =
            threadIdx.x % (2 * CoreDims::THREADS_PER_WARP);
        const unsigned d_warp_l = get_prefetch_warp<Dims>() / 2;
        const unsigned chunk_size = 16;
        const OpaqueElement* weights =
            (const OpaqueElement*)(expert_weights_down +
                                   id * Dims::N * Dims::HIDDEN_STATES +
                                   base_row_dn * Dims::N);
        for (unsigned row = d_warp_l, i = 0;
             i < CoreDims::W_DOWN_TILE / (CoreDims::PREFETCH_WARP_COUNT / 2);
             row += CoreDims::PREFETCH_WARP_COUNT / 2, i++) {
          unsigned col = d_thread_l * chunk_size;
          if (Dims::N == 2 * CoreDims::THREADS_PER_WARP * chunk_size ||
              col < Dims::N) {
            copy128(shm->w[BUF_W].down[row][col],
                    weights[(row * Dims::N + col) / sizeof(OpaqueElement)],
                    pipe);
          }
        }
        if (d_warp_l == 0) {
          // Block-wise: fetch 2D scale tile
          constexpr uint32_t SCALE_TILE_SIZE =
              MoE_SHM<Dims>::U::TinyData::DOWN_SCALE_TILE_SIZE;
          constexpr uint32_t COL_BLOCKS = (Dims::N + 127) / 128;
          if (d_thread_l < SCALE_TILE_SIZE) {
            uint32_t rb = d_thread_l / COL_BLOCKS;
            uint32_t cb = d_thread_l % COL_BLOCKS;
            uint32_t global_rb = (base_row_dn / 128) + rb;
            shm->scale[BUF_W][d_thread_l] =
                expert_scales_down[id * Dims::DOWN_SCALE_ROWS *
                                       Dims::DOWN_SCALE_COLS +
                                   global_rb * Dims::DOWN_SCALE_COLS + cb];
          }
        }
      }
      pipe.producer_commit();
    } else {
      // Calc warps: quantize w[BUF_BF16].bf16_buf → a.down[buf_fp8]
      // Per-block (1, 128) quantization: each 128-element block gets its own
      // scale. With N=512, that's 4 blocks per row.
      // Single-pass per block: read bf16 into registers, compute block_max,
      // then quantize from registers.
      constexpr uint32_t ACT_DOWN_BLOCK = 128;
      constexpr uint32_t NUM_DOWN_BLOCKS = Dims::N / ACT_DOWN_BLOCK;
      static_assert(Dims::N % ACT_DOWN_BLOCK == 0,
                    "N must be divisible by activation block size");
      // Elements per warp iteration = 32 threads × 4 floats = 128
      // So each iteration covers exactly one 128-element block.
      constexpr std::uint32_t ELEMS_PER_ITER = COLS_PER_WARP_ITER;
      static_assert(ELEMS_PER_ITER == ACT_DOWN_BLOCK,
                    "Warp iteration size must equal activation block size");
      constexpr std::uint32_t ITERS = Dims::N / ELEMS_PER_ITER;
      static_assert(ITERS == NUM_DOWN_BLOCKS);

      const std::uint32_t cw = get_calc_warp<Dims>();
      for (std::uint32_t tok = cw; tok < batch_size;
           tok += CoreDims::CALC_WARP_COUNT) {
        bool assigned = false;
        for (uint32_t k = 0; k < top_k; k++) {
          if (shmem->topk_ids_flat[tok * MAX_TOPK + k] == (uint16_t)id) {
            assigned = true;
            break;
          }
        }
        if (!assigned) continue;

        // Process each 128-element block independently
        float regs[ITERS * 4];  // hold all converted floats

  #pragma unroll
        for (std::uint32_t blk = 0; blk < NUM_DOWN_BLOCKS; ++blk) {
          std::uint32_t blk_start = blk * ACT_DOWN_BLOCK;
          std::uint32_t col = blk_start + thread * FLOATS_PER_LOAD;

          // Read bf16 from SHM into registers and find block max
          __nv_bfloat162 bf_01 = *reinterpret_cast<const __nv_bfloat162*>(
              &shm->w[BUF_BF16].bf16_buf[tok][col + 0]);
          __nv_bfloat162 bf_23 = *reinterpret_cast<const __nv_bfloat162*>(
              &shm->w[BUF_BF16].bf16_buf[tok][col + 2]);
          float2 f01 = __bfloat1622float2(bf_01);
          float2 f23 = __bfloat1622float2(bf_23);
          regs[blk * 4 + 0] = f01.x;
          regs[blk * 4 + 1] = f01.y;
          regs[blk * 4 + 2] = f23.x;
          regs[blk * 4 + 3] = f23.y;
          float local_max = fmaxf(fmaxf(fabsf(f01.x), fabsf(f01.y)),
                                  fmaxf(fabsf(f23.x), fabsf(f23.y)));

          float blk_max = warp_reduce_max_float(local_max);
          if (blk_max < __FLT_MIN__) blk_max = 1.f;

          float blk_scale = blk_max * FP8_MAX_INV;
          float blk_inv_scale = FP8_MAX / blk_max;

          // Quantize this block from registers → write fp8 to SHM
          __nv_fp8x4_e4m3 q{float4{regs[blk * 4 + 0] * blk_inv_scale,
                                   regs[blk * 4 + 1] * blk_inv_scale,
                                   regs[blk * 4 + 2] * blk_inv_scale,
                                   regs[blk * 4 + 3] * blk_inv_scale}};
          *reinterpret_cast<__nv_fp8x4_e4m3*>(&shm->a.down[buf_fp8][tok][col]) =
              q;

          if (thread == 0) shm->a_down_scale[buf_fp8][tok][blk] = blk_scale;

  #ifdef DEBUG_MOE_PRINT
          // Print ALL 4 blocks of bf16 input + quantization for expert 0, tok 0
          if (blockIdx.x == 0 && thread == 0 && tok == 0 && e == 0) {
            printf(
                "[DBG QUANT e=0 blk=%u] bf16_in[col=%u..]: %.6f %.6f %.6f "
                "%.6f\n",
                blk, col, regs[blk * 4 + 0], regs[blk * 4 + 1],
                regs[blk * 4 + 2], regs[blk * 4 + 3]);
            printf(
                "[DBG QUANT e=0 blk=%u] blk_max=%.6f scale=%.6f "
                "inv_scale=%.6f\n",
                blk, blk_max, blk_scale, blk_inv_scale);
            printf(
                "[DBG QUANT e=0 blk=%u] fp8_out[col=%u..]: %.4f %.4f %.4f "
                "%.4f\n",
                blk, col, (float)shm->a.down[buf_fp8][tok][col + 0],
                (float)shm->a.down[buf_fp8][tok][col + 1],
                (float)shm->a.down[buf_fp8][tok][col + 2],
                (float)shm->a.down[buf_fp8][tok][col + 3]);
          }
  #endif
        }
      }
    }

    // Wait for w_down fetch to complete
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    __syncthreads();

    // ── Stage B: prefetch next bf16 → w[BUF_BF16] || MMA + reduce ────────
    // No conflict: bf16 goes to w[0], MMA reads weights from w[1].
    if (is_prefetch_warp<Dims>()) {
      // Fetch next expert's bf16 intermediate into w[BUF_BF16].bf16_buf
      if (e + 1 < expert_count) {
        const std::uint32_t next_id = shmem->experts[e + 1].id;
        const unsigned pw = get_prefetch_warp<Dims>();
        pipe.producer_acquire();
        for (std::uint32_t tok = 0; tok < batch_size; ++tok) {
          for (uint32_t k = 0; k < top_k; k++) {
            if (shmem->topk_ids_flat[tok * MAX_TOPK + k] == (uint16_t)next_id) {
              std::uint32_t vrow = tok * top_k + k;
              const A_element* src = &spec->temp_bf16[vrow * Dims::N];
              for (std::uint32_t col =
                       (pw * CoreDims::THREADS_PER_WARP + thread) *
                       FLOATS_PER_LOAD;
                   col < Dims::N;
                   col += CoreDims::PREFETCH_WARP_COUNT *
                          CoreDims::THREADS_PER_WARP * FLOATS_PER_LOAD) {
                const auto shape8 = cuda::aligned_size_t<8>(8);
                cuda::memcpy_async(&shm->w[BUF_BF16].bf16_buf[tok][col],
                                   &src[col], shape8, pipe);
              }
              break;
            }
          }
        }
        pipe.producer_commit();
      }
    } else {
      // Calc warps: MMA a.down[buf_fp8] × w[BUF_W].down
      // Zero partial results
      for (unsigned wr = warp * CoreDims::W_DOWN_MMA_TILE;
           wr < CoreDims::W_DOWN_TILE;
           wr += CoreDims::W_DOWN_MMA_TILE * CoreDims::CALC_WARP_COUNT) {
        shm->partial_result.down[wr / 2 + warp][thread + 0] = 0.f;
        shm->partial_result.down[wr / 2 + warp][thread + 32] = 0.f;
        shm->partial_result.down[wr / 2 + warp][thread + 64] = 0.f;
        shm->partial_result.down[wr / 2 + warp][thread + 96] = 0.f;
      }

      // MMA m16n8k32 D-output mapping (from PTX spec):
      //   d0 → D[t/4,     (t%4)*2]     row=t/4,     token=(t%4)*2
      //   d1 → D[t/4,     (t%4)*2 + 1] row=t/4,     token=(t%4)*2 + 1
      //   d2 → D[t/4 + 8, (t%4)*2]     row=t/4 + 8, token=(t%4)*2
      //   d3 → D[t/4 + 8, (t%4)*2 + 1] row=t/4 + 8, token=(t%4)*2 + 1
      //
      // store_row0 controls d0/d2 (token=(t%4)*2).
      // store_row1 controls d1/d3 (token=(t%4)*2+1).
      const std::uint32_t tok_02 = (thread % 4) * 2;
      const std::uint32_t tok_13 = (thread % 4) * 2 + 1;
      bool s0 = false, s1 = false;
      if (tok_02 < batch_size)
        for (uint32_t k = 0; k < top_k; k++)
          if (shmem->topk_ids_flat[tok_02 * MAX_TOPK + k] == (uint16_t)id) {
            s0 = true;
            break;
          }
      if (tok_13 < batch_size)
        for (uint32_t k = 0; k < top_k; k++)
          if (shmem->topk_ids_flat[tok_13 * MAX_TOPK + k] == (uint16_t)id) {
            s1 = true;
            break;
          }

      // MMA: d0/d2 use tok_02's act scale; d1/d3 use tok_13's
      moe_down_mult_fp8<Dims>(shm->w[BUF_W].down, shm->scale[BUF_W],
                              shm->a.down[buf_fp8][thread / 4],
                              shm->a_down_scale[buf_fp8], tok_02, tok_13, s0,
                              s1, shm->partial_result.down);

  #ifdef DEBUG_MOE_PRINT
      if (blockIdx.x == 0 && threadIdx.x == 0 && e == 0) {
        printf("\n[DBG MMA_INPUTS e=0] w_scales (row-block 0, 4 col-blocks):");
        for (uint32_t si = 0; si < 4; si++)
          printf(" %.6f", shm->scale[BUF_W][si]);
        printf("\n");
        printf("[DBG MMA_INPUTS e=0] act_scales tok=0 (4 blocks):");
        for (uint32_t si = 0; si < 4; si++)
          printf(" %.6f", shm->a_down_scale[buf_fp8][0][si]);
        printf("\n");
        printf("[DBG MMA_INPUTS e=0] a.down[0][0..7]:");
        for (int ai = 0; ai < 8; ai++)
          printf(" %.4f", (float)shm->a.down[buf_fp8][0][ai]);
        printf("\n");
        printf("[DBG MMA_INPUTS e=0] w.down row 0 [0..7]:");
        for (int wi = 0; wi < 8; wi++)
          printf(" %.4f", (float)shm->w[BUF_W].down[0][wi]);
        printf("\n");
        printf("[DBG MMA_INPUTS e=0] w.down row 1 [0..7]:");
        for (int wi = 0; wi < 8; wi++)
          printf(" %.4f", (float)shm->w[BUF_W].down[1][wi]);
        printf("\n");
        printf("[DBG MMA_INPUTS e=0] w.down row 8 [0..7]:");
        for (int wi = 0; wi < 8; wi++)
          printf(" %.4f", (float)shm->w[BUF_W].down[8][wi]);
        printf("\n");
      }
  #endif
    }
    __syncthreads();

    // Reduce and accumulate into out_accum
    //
    // MMA m16n8k32 D-output mapping (CONFIRMED from PTX spec):
    //   d0 → D[t/4,     (t%4)*2]     row=t/4,     col=(t%4)*2
    //   d1 → D[t/4,     (t%4)*2 + 1] row=t/4,     col=(t%4)*2 + 1
    //   d2 → D[t/4 + 8, (t%4)*2]     row=t/4 + 8, col=(t%4)*2
    //   d3 → D[t/4 + 8, (t%4)*2 + 1] row=t/4 + 8, col=(t%4)*2 + 1
    //
    // D-columns are tokens. For BS<=8, only valid tokens have valid B-data.
    // So d0/d2 use token=(t%4)*2, d1/d3 use token=(t%4)*2+1.
    //
    // For the reduction, each thread reads partial_result at the position
    // where the MMA wrote its d0..d3.
    //
    //   partial_result[warp][thread+0]  = d0 (row=t/4,     token=(t%4)*2)
    //   partial_result[warp][thread+32] = d1 (row=t/4,     token=(t%4)*2+1)
    //   partial_result[warp][thread+64] = d2 (row=t/4 + 8, token=(t%4)*2)
    //   partial_result[warp][thread+96] = d3 (row=t/4 + 8, token=(t%4)*2+1)
    //
    // Reduction thread mapping:
    //   tok0 = (t%4)*2       (token for d0/d2)
    //   tok1 = (t%4)*2 + 1   (token for d1/d3)
    //   row_even = t/4       (weight row for d0)
    //   row_far  = t/4 + 8   (weight row for d2)
    //
    // Store to out_accum[token][weight_row + wr].
    if (!is_prefetch_warp<Dims>()) {
      const std::uint32_t tok0 = (thread % 4) * 2;
      const std::uint32_t tok1 = (thread % 4) * 2 + 1;
      const std::uint32_t row_col = thread / 4;  // weight row for d0 (d2 = +8)

      bool s0 = false, s1 = false;
      if (tok0 < batch_size)
        for (uint32_t k = 0; k < top_k; k++)
          if (shmem->topk_ids_flat[tok0 * MAX_TOPK + k] == (uint16_t)id) {
            s0 = true;
            break;
          }
      if (tok1 < batch_size)
        for (uint32_t k = 0; k < top_k; k++)
          if (shmem->topk_ids_flat[tok1 * MAX_TOPK + k] == (uint16_t)id) {
            s1 = true;
            break;
          }

      for (unsigned wr = warp * CoreDims::W_DOWN_MMA_TILE;
           wr < CoreDims::W_DOWN_TILE;
           wr += CoreDims::W_DOWN_MMA_TILE * CoreDims::CALC_WARP_COUNT) {
        float d0 = shm->partial_result.down[wr / 2][thread + 0];
        float d1 = shm->partial_result.down[wr / 2][thread + 32];
        float d2 = shm->partial_result.down[wr / 2][thread + 64];
        float d3 = shm->partial_result.down[wr / 2][thread + 96];
        for (unsigned i = 1; i < CoreDims::CALC_WARP_COUNT; ++i) {
          d0 += shm->partial_result.down[wr / 2 + i][thread + 0];
          d1 += shm->partial_result.down[wr / 2 + i][thread + 32];
          d2 += shm->partial_result.down[wr / 2 + i][thread + 64];
          d3 += shm->partial_result.down[wr / 2 + i][thread + 96];
        }

        if (s0) {
          shm->out_accum[tok0][row_col + wr + 0] += d0;
          if (CoreDims::W_DOWN_TILE % 16 == 0 || wr + 8 < CoreDims::W_DOWN_TILE)
            shm->out_accum[tok0][row_col + wr + 8] += d2;
        }
        if (s1) {
          shm->out_accum[tok1][row_col + wr + 0] += d1;
          if (CoreDims::W_DOWN_TILE % 16 == 0 || wr + 8 < CoreDims::W_DOWN_TILE)
            shm->out_accum[tok1][row_col + wr + 8] += d3;
        }

  #ifdef DEBUG_MOE_PRINT
        if (blockIdx.x == 0 && warp == 0 && thread < 8 && e == 0) {
          printf(
              "[DBG REDUCE e=0 t=%u tok0=%u tok1=%u row_col=%u wr=%u s0=%d "
              "s1=%d d0=%.4f d1=%.4f d2=%.4f d3=%.4f\n",
              thread, tok0, tok1, row_col, wr, (int)s0, (int)s1, d0, d1, d2,
              d3);
        }
  #endif
      }
    }

    // Flip a.down double-buffer index only (w[] slots are fixed)
    buf_fp8 ^= 1;

    // Wait for next expert's bf16 fetch before next iteration's quantize
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    __syncthreads();

  #ifdef DEBUG_MOE_PRINT
    if (blockIdx.x == 0 && threadIdx.x == 0 && e == 0) {
      printf("[DBG AFTER_EXPERT_0 out_accum[0][0..15]:");
      for (unsigned c = 0; c < 16; c++) printf(" %.4f", shm->out_accum[0][c]);
      printf("\n");
    }
  #endif
  }
}

}  // namespace moe_monokernel

#endif
