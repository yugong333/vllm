
#pragma once
#ifndef MOE_UP_PROJECTION_CU
  #define MOE_UP_PROJECTION_CU

  #ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
    #error Do not include this file directly.
  #endif

  #include <cuda.h>
  #include <cuda/pipeline>
  #include <cuda_fp8.h>

  #include "ptx_utils.h"
  #include "moe_interface.h"
  #include "moe_internal.h"
  #include "moe_down_projection.cu"
  #include "moe_debug.h"

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
template <typename Dims, std::size_t CopyCols = Dims::HIDDEN_STATES / 2,
          std::size_t DestCols = MoECoreDims<Dims>::K_DIM_HALF_PADDED_A>
__device__ inline void moe_request_input_tokens(
    const AQ_element* __restrict__ source,
    const std::uint16_t* __restrict__ token_indexes,
    AQ_element (&dest)[MoECoreDims<Dims>::A_TILE][DestCols],
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
      for (unsigned col = thread * chunk_size; col < CopyCols;
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
__device__ inline void moe_request_up_expert_for_row(
    const W_element* __restrict__ source, std::uint32_t id, unsigned base_row,
    W_element (&dest)[Rows][Cols],
    cuda::pipeline<cuda::thread_scope_thread>& pipe) {
  static_assert(CopyCols <= Cols);

  // position within the block
  using CoreDims = MoECoreDims<Dims>;
  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_prefetch_warp<Dims>();
  const unsigned chunk_size = 16 / sizeof(*source);

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
 * @brief Legacy wrapper: compute base_row from blockIdx.x.
 *
 * Use this for code paths (e.g. BS64) that map blockIdx.x 1:1 to up-proj
 * row tiles. For the BS8 two-expert-group design, call the
 * `moe_request_up_expert_for_row` variant directly with an explicit
 * `base_row` derived from the logical up-block index.
 */
template <typename Dims, std::size_t CopyCols, std::size_t Rows,
          std::size_t Cols>
__device__ inline void moe_request_up_expert(
    const W_element* __restrict__ source, std::uint32_t id,
    W_element (&dest)[Rows][Cols],
    cuda::pipeline<cuda::thread_scope_thread>& pipe) {
  using CoreDims = MoECoreDims<Dims>;
  const unsigned base_row = blockIdx.x * CoreDims::W_UP_TILE / 2;
  moe_request_up_expert_for_row<Dims, CopyCols>(source, id, base_row, dest,
                                                pipe);
}

/**
 * @brief Load this block's slice of block-wise up-projection scales into SHM.
 *
 * `base_row_up` is the first weight row (in the lower half of the 2*N rows)
 * owned by this block, i.e. `base_row_up ∈ [0, N)` with 8-row granularity.
 * This lets callers decouple the scale fetch from `blockIdx.x` — needed
 * by the two-expert-group BS8 design where both groups reuse the same
 * row-tile layout but with different blockIdx ranges.
 */
template <typename Dims>
__device__ inline void moe_request_up_scale_for_row(
    const S_element* __restrict__ expert_scales_up, std::uint32_t id,
    unsigned base_row_up, S_element* __restrict__ dest) {
  constexpr uint32_t COLS = Dims::UP_SCALE_COLS;  // e.g. 16 for K=2048
  constexpr uint32_t TILE = 2 * COLS;
  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_prefetch_warp<Dims>();

  // Only the first prefetch warp loads the scales — 32 scalars total for
  // Qwen3.5 (2×16). Synchronous shared-memory writes are fine; we don't
  // need async copy for 128 bytes.
  if (warp == 0 && thread < TILE) {
    uint32_t rb_local = thread / COLS;  // 0 → low half, 1 → upper half
    uint32_t kb = thread % COLS;
    uint32_t row = base_row_up + rb_local * Dims::N;
    uint32_t rb_global = row / Dims::BLOCK_SCALE_ROW;
    dest[thread] =
        expert_scales_up[id * Dims::UP_SCALE_ROWS * Dims::UP_SCALE_COLS +
                         rb_global * Dims::UP_SCALE_COLS + kb];
  }
}

/**
 * @brief Legacy wrapper — computes base_row_up from blockIdx.x.
 *
 * Used by code paths that map blockIdx.x 1:1 to up-proj row tiles
 * (the single-group design).
 */
template <typename Dims>
__device__ inline void moe_request_up_scale(
    const S_element* __restrict__ expert_scales_up, std::uint32_t id,
    S_element* __restrict__ dest) {
  using CoreDims = MoECoreDims<Dims>;
  const unsigned base_row_up = blockIdx.x * CoreDims::W_UP_TILE / 2;
  moe_request_up_scale_for_row<Dims>(expert_scales_up, id, base_row_up, dest);
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
 * order. Individual elements are in BF16 format.
 */
template <typename Dims, std::size_t Rows, std::size_t Cols>
__device__ inline void moe_up_reduction(
    const float (&partial_result)[Rows][Cols], float d0, float d1, float d2,
    float d3, float ws0, float ws1, float ts0, float ts1, bool store_row0,
    bool store_row1, unsigned row0, unsigned row1,
  #ifdef DEBUG_MOE
    float* __restrict__ gemm1,
  #endif
    A_element* __restrict__ result) {
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

  float sig0 = (w0 * x0) / (1 + __expf(-x0));
  float sig1 = (w1 * x1) / (1 + __expf(-x1));

  // write to temporary buffer (fp32 → bf16; saturation-free round-to-nearest)
  // Guard: blocks beyond N have no valid up-proj columns to write.
  // GRID_SIZE is sized for the down-proj (K columns) but the up-proj
  // only has N columns.  Skip the write for out-of-range blocks.
  if (store_row0 && (thread / 4) + base_row < Dims::N) {
    result[row0 * Dims::N + (thread / 4) + base_row] = (A_element)sig0;
  }
  if (store_row1 && (thread / 4) + base_row < Dims::N) {
    result[row1 * Dims::N + (thread / 4) + base_row] = (A_element)sig1;
  }
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
  const AQ_element* activations = spec->activations[0];

  // ── Full-K double-buffer design (requires K small enough — e.g. K=2048) ──
  //
  //   a[2][A_TILE][K]  — activation tile (fp8), ping-pong
  //   w[2][W_UP_TILE][K] — expert weights (fp8), ping-pong
  //
  // Pipeline, per expert e, per activation tile a_row:
  //   calc warps  : MMA using (w[w_read], a[t_read])
  //   prefetch    : if this is NOT the last a_row of e  →  fetch next a_row
  //                 into a[t_read ^ 1]
  //                 if this IS the last a_row of e AND e+1 < expert_count
  //                 →  fetch next expert's weights into w[w_read ^ 1] AND
  //                    its first activation tile into a[t_read ^ 1]
  //
  // Two outstanding pipe stages at most; double-buffering is sufficient.

  // ── Prime: fetch expert[0].w + expert[0].first_tile ──────────────────────
  if (is_prefetch_warp<Dims>()) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH
    const ExpertRef& expert = shmem->experts[0];
    pipe.producer_acquire();
    moe_request_up_expert<Dims, Dims::HIDDEN_STATES>(
        expert_weights_up, expert.id, shm->w[0], pipe);
    moe_request_input_tokens<Dims, Dims::HIDDEN_STATES,
                             CoreDims::K_DIM_PADDED_A>(
        activations, &shmem->path.bs64.token_indexes_topk[expert.first_token],
        shm->a[0], expert.last_token, pipe);
    pipe.producer_commit();
  #endif
  }

  std::uint32_t t_read = 0;  // current read slot for activations
  std::uint32_t w_read = 0;  // current read slot for weights

  for (std::uint32_t e = 0; e < expert_count; ++e) {
    const ExpertRef& expert = shmem->experts[e];
    std::uint32_t id = expert.id;
    unsigned int a_rows = expert.last_token - expert.first_token;
    A_element* temp = &spec->temp_bf16[expert.first_token * Dims::N];

    for (unsigned a_row = 0; a_row < a_rows; a_row += CoreDims::A_TILE) {
      const bool last_tile_of_expert = (a_row + CoreDims::A_TILE >= a_rows);
      const bool has_next_expert = (e + 1 < expert_count);

      // Wait for (weights + activation) for this iteration.
      cuda::pipeline_consumer_wait_prior<0>(pipe);
      __syncthreads();

      // d0..d3 declared here so they're visible to both the MMA (calc warps)
      // and the reduction (warp 0) after the syncthreads.
      float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

      if (is_prefetch_warp<Dims>()) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH
        // ── Prefetch ahead for the NEXT iteration ───────────────────────
        // Case A: still have more activation tiles for this expert
        //         → fetch next tile into a[t_read ^ 1]; weights stay put.
        // Case B: this was the last tile of this expert and there's a next
        //         expert  → fetch next expert's weights into w[w_read ^ 1]
        //         AND its first activation tile into a[t_read ^ 1].
        // Case C: last tile of last expert  → nothing to do.
        if (!last_tile_of_expert) {
          pipe.producer_acquire();
          moe_request_input_tokens<Dims, Dims::HIDDEN_STATES,
                                   CoreDims::K_DIM_PADDED_A>(
              activations,
              &shmem->path.bs64.token_indexes_topk[expert.first_token + a_row +
                                                   CoreDims::A_TILE],
              shm->a[t_read ^ 1], a_rows - (a_row + CoreDims::A_TILE), pipe);
          pipe.producer_commit();
        } else if (has_next_expert) {
          const ExpertRef& next = shmem->experts[e + 1];
          pipe.producer_acquire();
          moe_request_up_expert<Dims, Dims::HIDDEN_STATES>(
              expert_weights_up, next.id, shm->w[w_read ^ 1], pipe);
          moe_request_input_tokens<Dims, Dims::HIDDEN_STATES,
                                   CoreDims::K_DIM_PADDED_A>(
              activations,
              &shmem->path.bs64.token_indexes_topk[next.first_token],
              shm->a[t_read ^ 1], next.last_token - next.first_token, pipe);
          pipe.producer_commit();
        }
  #endif
      } else {
  #ifndef MONO_PROFILE_SKIP_CALC
        // ── Calc warps: MMA current tile × current expert weights ────────
        // Block-wise: apply per-iteration weight + activation scales.
        constexpr uint32_t ACT_BLOCK = 128;
        uint32_t sorted_pos0 = expert.first_token + a_row + (thread % 4) * 2;
        uint32_t sorted_pos1 = sorted_pos0 + 1;
        uint32_t tok0 = (sorted_pos0 < expert.last_token)
                            ? shmem->path.bs64.token_indexes_topk[sorted_pos0]
                            : 0;
        uint32_t tok1 = (sorted_pos1 < expert.last_token)
                            ? shmem->path.bs64.token_indexes_topk[sorted_pos1]
                            : 0;

    #ifdef DEBUG_MOE_PRINT
        if (blockIdx.x == 0 && threadIdx.x == 0 && e == 0 && a_row == 0) {
          printf(
              "[DBG64 UP e=0] sorted_pos0=%u sorted_pos1=%u tok0=%u tok1=%u "
              "first=%u last=%u\n",
              sorted_pos0, sorted_pos1, tok0, tok1, expert.first_token,
              expert.last_token);
          printf("[DBG64 UP e=0] expert_id=%u a_rows=%u\n", id, a_rows);
        }
    #endif

        // d0..d3 are declared before the if/else — just reset them here.
        d0 = 0.f;
        d1 = 0.f;
        d2 = 0.f;
        d3 = 0.f;
        for (unsigned base_col = warp * CoreDims::K_TILE;
             base_col < Dims::HIDDEN_STATES;
             base_col += CoreDims::BLOCK_STRIDE) {
          float md0 = 0.f, md1 = 0.f, md2 = 0.f, md3 = 0.f;
          unsigned row = thread / 4;
          unsigned col = 4 * (thread % 4);
          __nv_fp8x4_e4m3 w0 =
              *(__nv_fp8x4_e4m3*)&shm
                   ->w[w_read][row + 0][rotate_col_32(base_col + col + 0, row)];
          __nv_fp8x4_e4m3 w1 =
              *(__nv_fp8x4_e4m3*)&shm
                   ->w[w_read][row + 8][rotate_col_32(base_col + col + 0, row)];
          __nv_fp8x4_e4m3 w2 =
              *(__nv_fp8x4_e4m3*)&shm->w[w_read][row + 0][rotate_col_32(
                  base_col + col + 16, row)];
          __nv_fp8x4_e4m3 w3 =
              *(__nv_fp8x4_e4m3*)&shm->w[w_read][row + 8][rotate_col_32(
                  base_col + col + 16, row)];
          __nv_fp8x4_e4m3 a02 =
              *(__nv_fp8x4_e4m3*)(&shm->a[t_read][row][rotate_col_32(
                  base_col + col + 0, row)]);
          __nv_fp8x4_e4m3 a13 =
              *(__nv_fp8x4_e4m3*)(&shm->a[t_read][row][rotate_col_32(
                  base_col + col + 16, row)]);
          mma_fp8_fp8(md0, md1, md2, md3, w0, w1, w2, w3, a02, a13, 0.f, 0.f,
                      0.f, 0.f);

          unsigned full_k_col = base_col;
          float bws0 = get_up_block_scale<Dims>(
              expert_scales_up, id, base_row + thread / 4, full_k_col);
          float bws1 = get_up_block_scale<Dims>(expert_scales_up, id,
                                                base_row + thread / 4 + Dims::N,
                                                full_k_col);
          float as0 = shmem->act_scale[tok0][full_k_col / ACT_BLOCK];
          float as1 = shmem->act_scale[tok1][full_k_col / ACT_BLOCK];
          d0 += md0 * bws0 * as0;
          d1 += md1 * bws0 * as1;
          d2 += md2 * bws1 * as0;
          d3 += md3 * bws1 * as1;
        }

        shm->partial_result[warp][thread + 0] = d0;
        shm->partial_result[warp][thread + 32] = d1;
        shm->partial_result[warp][thread + 64] = d2;
        shm->partial_result[warp][thread + 96] = d3;

    #ifdef DEBUG_MOE_PRINT
        if (blockIdx.x == 0 && threadIdx.x == 0 && e == 0 && a_row == 0 &&
            warp == 0) {
          printf(
              "[DBG64 UP_FINAL e=0 t0 warp0] d0=%.6f d1=%.6f d2=%.6f d3=%.6f "
              "(gate0,gate1,up0,up1)\n",
              d0, d1, d2, d3);
        }
    #endif
  #endif
      }

      // All warps must participate in __syncthreads() — moved outside
      // the if/else to avoid deadlock.
      __syncthreads();

      // ── Reduce + SiLU + write (only warp 0) ────────────────────────
      if (!is_prefetch_warp<Dims>() && warp == 0) {
  #ifndef MONO_PROFILE_SKIP_CALC
        std::uint32_t row0 = a_row + (thread % 4) * 2 + 0;
        std::uint32_t row1 = a_row + (thread % 4) * 2 + 1;
        // Routing weight is NOT applied here — it's applied once in the
        // down-projection reduction (moe_down_reduction_topk) to avoid
        // double-counting.
        float ts0 = (row0 < a_rows) ? 1.0f : 0.f;
        float ts1 = (row1 < a_rows) ? 1.0f : 0.f;
        moe_up_reduction<Dims>(shm->partial_result, d0, d1, d2, d3, 1.0f, 1.0f,
                               ts0, ts1, row0 < a_rows, row1 < a_rows, row0,
                               row1,
    #ifdef DEBUG_MOE
                               &spec->gemm1[expert.first_token * 2 * Dims::N],
    #endif
                               temp);

    #ifdef DEBUG_MOE_PRINT
        if (blockIdx.x == 0 && threadIdx.x == 0 && e == 0 && a_row == 0) {
          printf("[DBG64 UP_REDUCE e=0] row0=%u row1=%u ts0=%.6f ts1=%.6f\n",
                 row0, row1, ts0, ts1);
          printf("[DBG64 UP_REDUCE e=0] SiLU temp[row0=0][0..7]:");
          for (int i = 0; i < 8; i++)
            printf(" %.6f", (float)temp[row0 * Dims::N + i]);
          printf("\n");
          if (row1 < a_rows) {
            printf("[DBG64 UP_REDUCE e=0] SiLU temp[row1=%u][0..7]:", row1);
            for (int i = 0; i < 8; i++)
              printf(" %.6f", (float)temp[row1 * Dims::N + i]);
            printf("\n");
          }
        }
    #endif
  #endif
      }

      __syncthreads();

      // ── Advance read buffers ───────────────────────────────────────────
      // Activation buffer always flips (a new tile is loaded every iteration
      // unless we're on the last tile of the last expert).
      t_read ^= 1;
      // Weight buffer flips only at expert boundary (when we transition from
      // the last tile of expert e to the first tile of expert e+1).
      if (last_tile_of_expert && has_next_expert) {
        w_read ^= 1;
      }
    }
  }
}

}  // namespace moe_monokernel

/**
 * @brief Split-phase up-projection for BS8: iterates over ALL experts,
 *        writing bf16 SiLU output to spec->temp_bf16 and block-local absmax
 *        to spec->temp_block_max.
 *
 * Uses double-buffered w_up[2] for pipelining: while computing expert e
 * with w_up[cur], prefetch warps load expert e+1 into w_up[next].
 *
 * a_up (quantized fp8 activations) is persistent in SHM across all experts.
 *
 * For each expert, each block computes W_UP_TILE/2 = 8 columns of the
 * N-wide SiLU output for all tokens routed to that expert.  The result
 * is written to spec->temp_bf16[row * N + col] where row is a virtual-batch
 * index assigned per (token, expert) pair.
 *
 * @param expert_weights_up  [E, 2*N, K] fp8 weights in global memory.
 * @param expert_scales_up   [E, 2*N] fp32 scales in global memory.
 * @param top_k              Number of experts per token.
 * @param batch_size         Number of active tokens.
 * @param spec               Global scratchpad (receives output in
 * spec->temp_bf16).
 * @param shmem              Shared memory with routing info and a_up.
 */
namespace moe_monokernel {

template <typename Dims>
__device__ inline void moe_up_projection_BS8_allexperts(
    const W_element* __restrict__ expert_weights_up,
    const S_element* __restrict__ expert_scales_up, std::uint32_t top_k,
    std::uint32_t batch_size, MoEGemmSpec<Dims>* __restrict__ spec,
    MoE_SHM<Dims>* __restrict__ shmem, std::uint32_t up_block_idx = 0xffffffffu,
    std::uint32_t expert_start = 0, std::uint32_t expert_stride = 1) {
  static_assert(Dims::BS <= 8);
  using CoreDims = MoECoreDims<Dims>;
  constexpr uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;

  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_any_warp<Dims>();

  // If the caller didn't pass an explicit up_block_idx, default to
  // blockIdx.x (single-group behaviour, identical to the pre-refactor code).
  // Otherwise use the provided index, which decouples the row-tile
  // assignment from blockIdx and lets multiple block groups share the same
  // row-tile layout (two-expert-parallel design).
  const unsigned effective_bid =
      (up_block_idx == 0xffffffffu) ? blockIdx.x : up_block_idx;
  const unsigned base_row_up = effective_bid * CoreDims::W_UP_TILE / 2;

  auto* shm = &shmem->u.tiny;
  const std::uint32_t expert_count = shmem->expert_count;

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  // w_up[expert_start] was already prefetched into w[1].up by phase 2 in
  // moe.cu for this group. We start with w_cur=1 (the buffer holding the
  // first expert's weights for this group).
  std::uint32_t w_cur = 1;

  for (std::uint32_t e = expert_start; e < expert_count; e += expert_stride) {
    const std::uint32_t id = shmem->experts[e].id;
    const std::uint32_t e_next = e + expert_stride;
    const bool has_next = e_next < expert_count;

    // Wait for current expert's weights
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    __syncthreads();

    if (is_prefetch_warp<Dims>()) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH
      // Prefetch NEXT expert (for this group) into the other buffer
      if (has_next) {
        pipe.producer_acquire();
        moe_request_up_expert_for_row<Dims, Dims::HIDDEN_STATES>(
            expert_weights_up, shmem->experts[e_next].id, base_row_up,
            shm->w[w_cur ^ 1].up, pipe);
        pipe.producer_commit();
        // Load next expert's scale slice into the other slot.
        moe_request_up_scale_for_row<Dims>(
            expert_scales_up, shmem->experts[e_next].id, base_row_up,
            shm->up_scale[w_cur ^ 1]);
      }
  #endif
    } else {
  #ifndef MONO_PROFILE_SKIP_CALC
      // ── MMA: a.up × w[w_cur].up ──────────────────────────────────────
      // Block-wise quantization (Qwen3.5): per-iteration weight + activation
      // scale application. Weight scales are pre-loaded into
      // shm->up_scale[w_cur] by the prefetch path; activation scales are in
      // shmem->act_scale[tok][k_col / 128].
      constexpr uint32_t ACT_BLOCK = 128;
      uint32_t tok_02 = (thread % 4) * 2;      // token for d0/d1
      uint32_t tok_13 = (thread % 4) * 2 + 1;  // token for d2/d3

      // Loop-invariant: whether this block's 8 weight rows fall inside
      // the valid up-projection row range [0, N). Out-of-range blocks
      // still do the MMA but multiply by zero scales so they don't
      // contribute.
      const bool in_range = (base_row_up + thread / 4 < Dims::N);

      // Weight-scale pointers for this expert:
      //   slot 0 = lower-half row-block (rows [base_row_up .. +7])
      //   slot 1 = upper-half row-block (rows [base_row_up + N .. +N+7])
      const S_element* up_scale_lo = &shm->up_scale[w_cur][0];
      const S_element* up_scale_hi = &shm->up_scale[w_cur][Dims::UP_SCALE_COLS];

      float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
      for (unsigned bc = warp * CoreDims::K_TILE, i = 0;
           i < Dims::HIDDEN_STATES / CoreDims::BLOCK_STRIDE;
           ++i, bc += CoreDims::BLOCK_STRIDE) {
        float md0 = 0.f, md1 = 0.f, md2 = 0.f, md3 = 0.f;
        unsigned r = thread / 4, c = 4 * (thread % 4);
        __nv_fp8x4_e4m3 w0 = *(__nv_fp8x4_e4m3*)&shm->w[w_cur]
                                  .up[r + 0][rotate_col_32(bc + c + 0, r)];
        __nv_fp8x4_e4m3 w1 = *(__nv_fp8x4_e4m3*)&shm->w[w_cur]
                                  .up[r + 8][rotate_col_32(bc + c + 0, r)];
        __nv_fp8x4_e4m3 w2 = *(__nv_fp8x4_e4m3*)&shm->w[w_cur]
                                  .up[r + 0][rotate_col_32(bc + c + 16, r)];
        __nv_fp8x4_e4m3 w3 = *(__nv_fp8x4_e4m3*)&shm->w[w_cur]
                                  .up[r + 8][rotate_col_32(bc + c + 16, r)];
        __nv_fp8x4_e4m3 a02 =
            *(__nv_fp8x4_e4m3*)&shm->a.up[r][rotate_col_32(bc + c + 0, r)];
        __nv_fp8x4_e4m3 a13 =
            *(__nv_fp8x4_e4m3*)&shm->a.up[r][rotate_col_32(bc + c + 16, r)];

        mma_fp8_fp8(md0, md1, md2, md3, w0, w1, w2, w3, a02, a13, 0.f, 0.f, 0.f,
                    0.f);

        unsigned full_k_col = bc;
        uint32_t kb = full_k_col / Dims::BLOCK_SCALE_COL;
        float bws0 = in_range ? up_scale_lo[kb] : 0.f;
        float bws1 = in_range ? up_scale_hi[kb] : 0.f;
        // Per-token block-wise activation scale
        float as_02 = shmem->act_scale[tok_02][full_k_col / ACT_BLOCK];
        float as_13 = shmem->act_scale[tok_13][full_k_col / ACT_BLOCK];

        d0 += md0 * bws0 * as_02;
        d1 += md1 * bws0 * as_13;
        d2 += md2 * bws1 * as_02;
        d3 += md3 * bws1 * as_13;
      }

      // Swap d1↔d2 for gate/up layout
      {
        float tmp = d1;
        d1 = d2;
        d2 = tmp;
      }
      shm->partial_result.up[warp][thread + 0] = d0;
      shm->partial_result.up[warp][thread + 32] = d1;
      shm->partial_result.up[warp][thread + 64] = d2;
      shm->partial_result.up[warp][thread + 96] = d3;
  #endif
    }
    __syncthreads();

    // ── Reduction → SiLU → write bf16 to spec->temp + block-local max ────
    //
    // MMA m16n8k32 D-output mapping (CONFIRMED by debug):
    //   d0 → D[t/4,     (t%4)*2]     gate for weight_row=t/4,   token=(t%4)*2
    //   d1 → D[t/4 + 8, (t%4)*2]     up   for weight_row=t/4,   token=(t%4)*2
    //   d2 → D[t/4,     (t%4)*2 + 1] gate for weight_row=t/4,   token=(t%4)*2+1
    //   d3 → D[t/4 + 8, (t%4)*2 + 1] up   for weight_row=t/4,   token=(t%4)*2+1
    //
    // After d1↔d2 swap, partial_result layout:
    //   [warp][thread + 0]  = d0 = gate, weight_row=t/4, token=(t%4)*2
    //   [warp][thread + 32] = d1 = gate, weight_row=t/4, token=(t%4)*2+1  (was
    //   d2) [warp][thread + 64] = d2 = up,   weight_row=t/4, token=(t%4)*2 (was
    //   d1) [warp][thread + 96] = d3 = up,   weight_row=t/4, token=(t%4)*2+1
    //
    // For SiLU we need (gate, up) for the same (token, weight_row).
    //   token=(t%4)*2:   gate at offset +0,  up at offset +64
    //   token=(t%4)*2+1: gate at offset +32, up at offset +96
    //
    // Warp 0 handles token=(t%4)*2,   warp 1 handles token=(t%4)*2+1.
    // Output column = weight_row = t/4 + base_row_up.
  #ifndef MONO_PROFILE_SKIP_CALC
    if (warp < 2) {
      const std::uint32_t tok = (thread % 4) * 2 + warp;  // token index
      bool store = false;
      float rw = 0.f;
      std::uint32_t virtual_row = 0;

      if (tok < batch_size) {
        for (uint32_t k = 0; k < top_k; k++) {
          if (shmem->topk_ids_flat[tok * MAX_TOPK + k] == (uint16_t)id) {
            store = true;
            rw = shmem->topk_weights_flat[tok * MAX_TOPK + k];
            virtual_row = tok * top_k + k;
            break;
          }
        }
      }

      if (store) {
        // Gate at offset warp*64, up at offset warp*64 + 32
        const std::uint32_t gate_off = warp * 64;
        const std::uint32_t up_off = warp * 64 + 32;

        float x0 = shm->partial_result.up[0][thread + gate_off] +
                   shm->partial_result.up[1][thread + gate_off];
        float w0v = shm->partial_result.up[0][thread + up_off] +
                    shm->partial_result.up[1][thread + up_off];
        for (unsigned i = 2; i < CoreDims::CALC_WARP_COUNT; i += 2) {
          x0 += shm->partial_result.up[i][thread + gate_off] +
                shm->partial_result.up[i + 1][thread + gate_off];
          w0v += shm->partial_result.up[i][thread + up_off] +
                 shm->partial_result.up[i + 1][thread + up_off];
        }

        // Output column = weight_row = thread/4 + base_row_up
        if ((thread / 4) + base_row_up < Dims::N) {
          float val = rw * (w0v * x0) / (1.f + __expf(-x0));
          spec->temp_bf16[virtual_row * Dims::N + (thread / 4) + base_row_up] =
              (__nv_bfloat16)val;
        }
      }
    }
  #endif

    w_cur ^= 1;
    __syncthreads();
  }
}

}  // namespace moe_monokernel

#endif
