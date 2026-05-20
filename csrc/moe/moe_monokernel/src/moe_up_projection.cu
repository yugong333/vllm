
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
  #include "moe_tma.h"

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
 * @brief v1 streaming-pipeline 128 × 128 weight tile loader.
 *
 * Loads a 128-row × 128-K tile of expert weights into SHM in WGMMA
 * canonical Major::K layout.  Called by prefetch warps (warps 8..11,
 * 128 threads).
 *
 * Row mapping (dst_row in [0, 128)):
 *   [  0..31] = gate rows [base_row      .. base_row+31]   (WG0 gate)
 *   [ 32..63] =   up rows [base_row + N  .. base_row+N+31] (WG0 up)
 *   [ 64..95] = gate rows [base_row + 32 .. base_row+63]   (WG1 gate)
 *   [ 96..127] =   up rows [base_row+N+32 .. base_row+N+63] (WG1 up)
 *
 * Canonical byte offset in `dest` for element (dst_row, k) where
 *   m_outer = dst_row / 8
 *   m_inner = dst_row % 8
 *   k_outer = k / 16
 *   k_inner = k % 16
 * is
 *   byte_off = m_outer * 1024 + k_outer * 128 + m_inner * 16 + k_inner
 *
 * Strides in the A descriptor:
 *   LBO = 128 B  (one 8×16-byte K-core-matrix)
 *   SBO = 1024 B (one 8-row M-block = 8 K-core-matrices × 128 B)
 *
 * Total tile size: 16 (m_outer) × 8 (k_outer) = 128 core matrices
 *                  × 128 B = 16 384 B = 16 KB.
 *
 * Thread distribution: 128 prefetch threads, 128 core matrices
 *                      × 8 rows per core matrix × 16 B = 16384 B total,
 *                      each thread issues 8 × 16-byte cp.async.  Each
 *                      thread handles one `(m_outer, m_inner)` pair
 *                      (1024 such pairs = 128 m_outer × 8 m_inner, i.e.
 *                      one full ROW of the logical [128][128] tile)
 *                      by striding over the 8 k_outer values.
 *
 * Concretely, `tin = m_outer * 8 + m_inner = dst_row`, so each thread
 * owns one destination row and issues 8 chunks along K.
 *
 * @tparam Dims        MoE dims.
 * @tparam DestRows    Must be 128.
 * @tparam DestCols    Must be K_STEP_WGMMA (128).
 * @param  source      GM pointer [E, 2*N, K] fp8 weights.
 * @param  id          Expert index.
 * @param  base_row    First gate-row of this block's M stripe (multiple
 *                     of 64).
 * @param  k_start     Starting K column of this tile (multiple of 128).
 * @param  dest        SHM tile [128][128] fp8.
 * @param  pipe        Async-copy pipeline.
 *
 * @note Prefetch-warp only.  Must be followed by `pipe.producer_commit()`
 *       and a consumer wait before WGMMA reads `dest`.
 */
template <typename Dims, std::size_t DestRows, std::size_t DestCols>
__device__ inline void moe_load_up_wgmma_tile_128x128(
    const W_element* __restrict__ source, std::uint32_t id,
    std::uint32_t base_row, std::uint32_t k_start,
    W_element (&dest)[DestRows][DestCols],
    cuda::pipeline<cuda::thread_scope_thread>& pipe) {
  using CoreDims = MoECoreDims<Dims>;
  static_assert(DestRows == 128,
                "v1 streaming WGMMA weight tile must be 128 rows");
  static_assert(DestCols == 128,
                "v1 streaming WGMMA weight tile must be 128 K-values");

  constexpr unsigned CHUNK_B = 16;
  constexpr unsigned CHUNK_ELEMS = CHUNK_B / sizeof(W_element);  // 16
  constexpr unsigned K_CHUNKS_PER_ROW = DestCols / CHUNK_ELEMS;  // 8

  // Only prefetch warps issue the cp.async. 128 threads = 4 warps × 32.
  const unsigned thread = get_thread<Dims>();
  const unsigned pw = get_prefetch_warp<Dims>();  // 0..3
  const unsigned pflat = pw * 32 + thread;        // 0..127 — one row each

  const OpaqueElement* weights =
      (const OpaqueElement*)(source + id * 2 * Dims::N * Dims::HIDDEN_STATES);

  pipe.producer_acquire();
  if (pflat < 128) {
    const unsigned dst_row = pflat;
    const unsigned m_outer = dst_row / 8;  // 0..15
    const unsigned m_inner = dst_row % 8;  // 0..7

    // Map dst_row to global weight row.  The 128-row tile is structured
    // as [gate(WG0)|up(WG0)|gate(WG1)|up(WG1)] with 32 rows each.
    unsigned global_row;
    {
      const unsigned stripe = dst_row / 32;   // 0..3
      const unsigned r_in_32 = dst_row % 32;  // 0..31
      const bool is_up = (stripe == 1) || (stripe == 3);
      const bool is_wg1 = (stripe >= 2);
      const unsigned wg_row_base = is_wg1 ? 32u : 0u;
      global_row = base_row + wg_row_base + r_in_32 + (is_up ? Dims::N : 0);
    }

    OpaqueElement* dest_oe = (OpaqueElement*)&dest[0][0];

  #pragma unroll
    for (unsigned k_outer = 0; k_outer < K_CHUNKS_PER_ROW; ++k_outer) {
      const unsigned k_col = k_outer * CHUNK_ELEMS;
      // Canonical byte offset:
      //   m_outer * 1024 + k_outer * 128 + m_inner * 16
      const unsigned byte_off =
          m_outer * 1024 + k_outer * 128 + m_inner * CHUNK_B;
      const unsigned dest_chunk_idx = byte_off / sizeof(OpaqueElement);
      copy128(dest_oe[dest_chunk_idx],
              weights[(global_row * Dims::HIDDEN_STATES + k_start + k_col) /
                      sizeof(OpaqueElement)],
              pipe);
    }
  }
  pipe.producer_commit();
}

/**
 * @brief v1 streaming-pipeline bf16 input tile loader.
 *
 * Loads one K=128 tile of bf16 input activations into a SHM slot.
 * Shape: 8 tokens × 128 K-values = 2048 bytes = one 128-byte copy
 * per token, or 32 × 16-byte copies per token, or in aggregate
 * 128 × 16-byte copies (= 2 KB) spread across 128 prefetch threads.
 *
 * SHM layout written: dest[tok][k] row-major contiguous.
 *
 * Thread distribution: 128 prefetch threads; 8 tokens × 16 chunks
 * per token = 128 chunks, exactly one 16-byte chunk per thread.
 *   tok = pflat / 16   (0..7)
 *   k   = (pflat % 16) * CHUNK_ELEMS  (0, 8, 16, ..., 120 for bf16)
 *
 * @tparam Dims              MoE dims.
 * @tparam DestTok           Must be T_TILE (8).
 * @tparam DestK             Must be K_STEP_WGMMA (128).
 * @param  activations_in    GM pointer to activations [BS][K] bf16.
 * @param  k_start           Starting K column of this tile (multiple of 128).
 * @param  batch_size        Number of real tokens (remaining are skipped).
 * @param  dest              SHM tile [T_TILE][K_STEP_WGMMA] bf16.
 * @param  pipe              Async-copy pipeline.
 *
 * @note Prefetch-warp only. Unused token slots (`tok >= batch_size`)
 *       are NOT loaded; the streaming quantize step zero-fills the
 *       corresponding fp8_act slots regardless of bf16_in contents,
 *       so leaving bf16_in[tok>=batch_size] stale is safe.
 */
template <typename Dims, std::size_t DestTok, std::size_t DestK>
__device__ inline void moe_load_bf16_input_tile(
    const A_element* __restrict__ activations_in, std::uint32_t k_start,
    std::uint32_t batch_size, A_element (&dest)[DestTok][DestK],
    cuda::pipeline<cuda::thread_scope_thread>& pipe) {
  using CoreDims = MoECoreDims<Dims>;
  static_assert(DestTok == CoreDims::T_TILE,
                "bf16 input tile must have T_TILE=8 token rows");
  static_assert(DestK == CoreDims::K_STEP_WGMMA,
                "bf16 input tile must have K_STEP_WGMMA=128 K-values");
  static_assert(sizeof(A_element) == 2, "A_element (bf16) must be 2 bytes");

  constexpr unsigned CHUNK_B = 16;
  constexpr unsigned CHUNK_ELEMS = CHUNK_B / sizeof(A_element);  // 8
  constexpr unsigned CHUNKS_PER_ROW = DestK / CHUNK_ELEMS;       // 16

  const unsigned thread = get_thread<Dims>();
  const unsigned pw = get_prefetch_warp<Dims>();  // 0..3
  const unsigned pflat = pw * 32 + thread;        // 0..127

  pipe.producer_acquire();
  if (pflat < DestTok * CHUNKS_PER_ROW) {
    const unsigned tok = pflat / CHUNKS_PER_ROW;    // 0..7
    const unsigned chunk = pflat % CHUNKS_PER_ROW;  // 0..15
    const unsigned k_col = chunk * CHUNK_ELEMS;     // 0, 8, ..., 120

    if (tok < batch_size) {
      copy128(dest[tok][k_col],
              activations_in[tok * Dims::HIDDEN_STATES + k_start + k_col],
              pipe);
    }
  }
  pipe.producer_commit();
}

/**
 * @brief WGMMA weight K-tile prefetch.
 *
 * Loads a 64-row × K_TILE_WGMMA-K tile of expert weights into a
 * double-buffered SHM slot.  K_TILE_WGMMA is 64 (= 2 × m64n8k32 K)
 * by default.
 *
 * SHM layout written: dest[row][k_off] with:
 *   rows [0  ..31]: weight rows [base_row     .. base_row+31]      (gate)
 *   rows [32 ..63]: weight rows [base_row + N .. base_row+N+31]    (up)
 *   k_off ∈ [0, K_TILE_WGMMA), row-major contiguous
 *
 * Each row is K_TILE_WGMMA bytes (fp8 = 1 B/elem).  For K_TILE_WGMMA = 64,
 * that's exactly 4 consecutive 16-byte async-copy transactions per row
 * when using all 32 threads of a warp (or 1 transaction per 4 threads).
 * We use 16 B / thread so 4 threads cover one row, letting one warp
 * process 8 rows per iteration.
 *
 * @tparam Dims              MoE dims.
 * @tparam K_TILE_WGMMA      K-width of the tile (64).
 * @param  source            GM pointer [E, 2*N, K] fp8 weights.
 * @param  id                Expert index.
 * @param  base_row          First gate-row this block owns (in [0, N)).
 * @param  k_start           Starting K column of this tile.
 * @param  dest              SHM tile [64][K_TILE_WGMMA] fp8.
 * @param  pipe              Async-copy pipeline.
 *
 * @note Prefetch-warp only.
 */
template <typename Dims, std::size_t K_TILE_WGMMA, std::size_t DestRows,
          std::size_t DestCols>
__device__ inline void moe_request_up_wgmma_tile(
    const W_element* __restrict__ source, std::uint32_t id,
    std::uint32_t base_row, std::uint32_t k_start,
    W_element (&dest)[DestRows][DestCols],
    cuda::pipeline<cuda::thread_scope_thread>& pipe) {
  static_assert(DestRows == 64, "WGMMA weight tile must be 64 rows");
  static_assert(DestCols == K_TILE_WGMMA, "dest K-dim must match K_TILE_WGMMA");
  static_assert(K_TILE_WGMMA % 16 == 0,
                "K_TILE_WGMMA must allow 16-byte copies");

  using CoreDims = MoECoreDims<Dims>;
  constexpr unsigned CHUNK_B = 16;  // cp.async granularity
  constexpr unsigned CHUNK_ELEMS = CHUNK_B / sizeof(W_element);
  constexpr unsigned CHUNKS_PER_ROW = K_TILE_WGMMA / CHUNK_ELEMS;

  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_prefetch_warp<Dims>();

  // Global-memory base for this expert's weight matrix.
  const OpaqueElement* weights =
      (const OpaqueElement*)(source + id * 2 * Dims::N * Dims::HIDDEN_STATES);

  // Each warp handles a stride of rows.  With 4 prefetch warps and 64 rows,
  // 16 rows per warp (in 4 outer iterations since each warp stripes rows).
  // The 32 threads of a warp cover the K-tile in CHUNKS_PER_ROW transactions.
  //
  // Row layout in dest: [0..31] = gate[base_row + r], [32..63] = up[base_row +
  // r]. Both map to the same (thread, chunk) pattern, differing only by the
  // global-row offset.
  #pragma unroll 1
  for (unsigned dst_row = warp; dst_row < 64;
       dst_row += CoreDims::PREFETCH_WARP_COUNT) {
    const bool is_up = (dst_row >= 32);
    const unsigned r_in_half = is_up ? (dst_row - 32) : dst_row;
    const unsigned global_row = base_row + r_in_half + (is_up ? Dims::N : 0);

    // Each thread fetches CHUNKS_PER_ROW/32 chunks — but for CHUNK_ELEMS=16
    // and K_TILE_WGMMA=64 we have CHUNKS_PER_ROW=4, so 4 threads cover the
    // row.  Let thread i handle chunk i % CHUNKS_PER_ROW.
    if (thread < CHUNKS_PER_ROW) {
      const unsigned k_col = thread * CHUNK_ELEMS;
      copy128(dest[dst_row][k_col],
              weights[(global_row * Dims::HIDDEN_STATES + k_start + k_col) /
                      sizeof(OpaqueElement)],
              pipe);
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
  #ifndef MONO_PROFILE_SKIP_PREFETCH_UP
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
  #ifndef MONO_PROFILE_SKIP_PREFETCH_UP
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
  #ifndef MONO_PROFILE_SKIP_CALC_UP
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
  #ifndef MONO_PROFILE_SKIP_CALC_UP
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

namespace moe_monokernel {

///////////////////////////////////////////////////////////////////////////////
//
// moe_up_projection_BS8_allexperts_wgmma_tma
//
// TMA+WGMMA up-projection for BS<=8. Only variant of the BS8 up-proj
// kernel: the cp.async reference path has been removed. Replaces the
// prefetch-warp `cp.async` loaders for BOTH the bf16 activation tile and
// the fp8 expert-weight tile with `cp.async.bulk.tensor.2d` issued by a
// single TMA launcher thread (warp 8, lane 0). Completion of each tile
// is signalled via SHM mbarriers (`bar_w[2]`, `bar_a[2]`); consumer
// warps wait with `mbarrier.try_wait.parity` instead of
// `cuda::pipeline_consumer_wait_prior`.
//
// Descriptors are built host-side in the torch binding wrapper and passed
// to the top-level kernel as `__grid_constant__ CUtensorMap const`
// parameters. At the device-side inlined helper boundary (this function),
// they appear as `CUtensorMap const&` — the `__grid_constant__` qualifier
// only applies at the kernel-function boundary.
//
// See `.kiro/specs/tma-wgmma-weight-load/design.md` for the full dataflow
// (SHM layout, barrier arming, warp-role allocation) and requirements
// R1.1, R2.1, R6.1.
//
template <typename Dims>
__device__ inline void moe_up_projection_BS8_allexperts_wgmma_tma(
    const A_element* __restrict__ activations_in,
    const W_element* __restrict__ expert_weights_up,
    const S_element* __restrict__ expert_scales_up, std::uint32_t top_k,
    std::uint32_t batch_size, MoEGemmSpec<Dims>* __restrict__ spec,
    MoE_SHM<Dims>* __restrict__ shmem, CUtensorMap const& up_weights_desc,
    CUtensorMap const& activations_desc,
    std::uint32_t up_block_idx = 0xffffffffu, std::uint32_t expert_start = 0,
    std::uint32_t expert_stride = 1, bool external_priming = false) {
  static_assert(Dims::BS <= 8);
  using CoreDims = MoECoreDims<Dims>;
  constexpr uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;

  // `activations_in` / `expert_weights_up` are retained on the parameter
  // list for signature parity with the `cp.async` reference but are not
  // dereferenced on the TMA path — all GM reads go through the two TMA
  // descriptors.
  (void)activations_in;
  (void)expert_weights_up;

  // `external_priming = true` means the caller has already performed:
  //   * `mbarrier_init` on all 4 barriers + release-fence
  //   * the first expert's Step A (arm bar_a[0] + bf16_in[0] TMA at k=0)
  //   * a block-wide `__syncthreads()` to publish both.
  //
  // Stage A pipeline (always-external-priming path):
  //   * Pre-loop: helper arms bar_w[0] + TMAs w[0] of expert_start at k=0.
  //   * K-loop (QUANT-first): iter s QUANT waits bar_a[s%2], quantizes;
  //                            iter s COMPUTE waits bar_w[s%2], WGMMA +
  //                            scale-apply, and launcher arms the
  //                            NEXT slot (intra-expert s+1 or next
  //                            expert's k=0 stitch).
  //   * One __syncthreads() per iteration (between QUANT and COMPUTE).
  //   * No priming block, no per-expert Step A, no tail prefetch — the
  //     cross-expert mbarrier chain carries tiles forward automatically.

  // ── Compile-time constants (v1) ─────────────────────────────────────
  // Byte-for-byte mirror of the `cp.async` reference variant's constants
  // so that SHM layouts, WGMMA descriptors, and K-step sizing remain
  // identical across the two paths (design P1/P2).
  constexpr uint32_t W_UP_M = CoreDims::W_UP_TILE_WGMMA;     // 128
  constexpr uint32_t K_STEP_WGMMA = CoreDims::K_STEP_WGMMA;  // 128
  constexpr uint32_t K_STEP = CoreDims::K_STEP_UP;           // 128 / 256
  constexpr uint32_t K_SUBSTEPS = CoreDims::K_SUBSTEPS_UP;   // 1 / 2
  constexpr uint32_t K_TILES = CoreDims::K_TILES_UP;         // K/K_STEP
  constexpr uint32_t WGMMAS_PER_SUBSTEP = CoreDims::WGMMAS_PER_STEP;  // 4
  constexpr uint32_t UP_SCALE_COLS = Dims::UP_SCALE_COLS;             // 16

  // Descriptor strides for 128×128 Major::K B128-swizzled A operand.
  //
  // The TMA hardware applies the 8-row × 128-byte core-matrix XOR
  // swizzle at write time, so each 1024-B atom holds one 8-row M-block.
  // CUTLASS Major::K B128 layout:
  //   LBO = 16 B   (one K-core-matrix within the 1024-B atom)
  //   SBO = 1024 B (next M-block atom)
  //   swizzle_mode = 1
  // The Python pre-interleave repacks gate/up row stripes so that a
  // single 128x128 TMA fetches the full WGMMA A-tile; it does NOT
  // apply the canonical core-matrix byte permutation (TMA does that).
  constexpr uint64_t A_LBO = 16ULL;
  constexpr uint64_t A_SBO = 1024ULL;
  constexpr uint32_t A_SWIZZLE = 1u;
  // B operand (K-major, N=8): 1 N-block, LBO=128 between K-core-matrices.
  // Always SWIZZLE_NONE — the activation tile is 8-token × 128-K bf16
  // and small enough that bank-conflict cost is bounded.
  constexpr uint64_t B_LBO = 128;
  constexpr uint64_t B_SBO = 128;  // unused (only 1 N-block for N=8)

  const unsigned thread_in_block = threadIdx.x;
  const unsigned warp = thread_in_block / 32;  // 0..11
  const unsigned lane = thread_in_block & 31;
  const bool is_wg0 = (warp < 4);
  const bool is_wg1 = (warp >= 4 && warp < 8);
  const bool is_calc = (warp < 8);
  const unsigned warp_in_wg = warp & 3;  // 0..3 within each WG
  // Gate/up split within each WG: warps 0,1 (in-WG) → gate rows [0..31];
  // warps 2,3 (in-WG) → up rows [32..63] (within the WG's 64-row stripe).
  const bool is_gate_half = (warp_in_wg < 2);
  (void)thread_in_block;
  (void)is_wg0;

  // TMA path uses the `tiny_wgmma_tma` union variant (byte-identical to
  // `tiny_wgmma` plus 32 B of mbarriers at the tail).
  auto* shm = &shmem->u.tiny_wgmma_tma;

  const unsigned effective_bid =
      (up_block_idx == 0xffffffffu) ? blockIdx.x : up_block_idx;
  // Each block owns 128 M rows = 2 WG stripes × 64 rows.  WG0's gate
  // rows start at base_row_up; WG1's gate rows start at base_row_up + 32.
  const unsigned base_row_up = effective_bid * (W_UP_M / 2);
  const std::uint32_t expert_count = shmem->expert_count;

  // Per-thread fp32 accumulators for WGMMA m64n8k32.
  // WG0 and WG1 threads each hold their own 4-register accumulator for
  // their respective M stripe.
  float chunk_d0 = 0.f, chunk_d1 = 0.f, chunk_d2 = 0.f, chunk_d3 = 0.f;
  float final_d0 = 0.f, final_d1 = 0.f, final_d2 = 0.f, final_d3 = 0.f;

  // ── Phase-3 preamble ──────────────────────────────────────────────────
  //
  // Entry contract (Stage A — external priming is mandatory):
  //   * All four mbarriers (bar_w[0..1], bar_a[0..1]) are initialized
  //     with arrival_count=1 and release-fenced.
  //   * bar_a[0] is armed + a TMA for bf16_in[0] at k=0 is in flight
  //     (from Phase-1 greedy prefetch in moe.cu).
  //   * A block-wide __syncthreads() has already published the init to
  //     every consumer warp.
  //
  // The helper never re-initializes barriers.  If `external_priming`
  // is false (legacy call sites), the helper falls back to the old
  // behaviour; we keep the branch for signature parity but on the only
  // live call site `external_priming` is always true.
  if (!external_priming) {
    if (is_tma_launcher_thread<Dims>()) {
      mbarrier_init(&shm->bar_w[0], 1u);
      mbarrier_init(&shm->bar_w[1], 1u);
      mbarrier_init(&shm->bar_a[0], 1u);
      mbarrier_init(&shm->bar_a[1], 1u);
      fence_mbarrier_init_release_cluster();
    }
    __syncthreads();
  }

  // Stage A requires an even K_TILES so the end-of-K-loop launcher arm
  // lands on `next_slot = K_TILES % 2 = 0`.  That slot-0 stitch is what
  // the next expert's iter-0 QUANT/COMPUTE waits on; an odd K_TILES
  // would land the stitch on slot 1, breaking the cross-expert
  // mbarrier chain.  At K_STEP_UP=128 (default) HIDDEN_STATES=2048 →
  // K_TILES=16; at K_STEP_UP=256 → K_TILES=8; both satisfy the
  // invariant.
  static_assert(K_TILES % 2 == 0,
                "Stage-A pipeline requires K_TILES to be even so the "
                "end-of-loop stitch arms the same slot that the next "
                "expert's iter-0 QUANT waits on.");

  // ── Pre-loop: arm bar_w[0] + TMA w[0] of expert_start at k=0 ──────────
  //
  // bar_w[0] is not pre-armed by the caller (only bar_a[0] is).  The
  // helper fires the first expert's weight TMA here, in parallel with
  // the still-in-flight Phase-1 bf16_in[0] TMA, so the iter-0 COMPUTE
  // can start the moment both barriers flip.
  //
  // For K_STEP > K_STEP_WGMMA the slot holds K_SUBSTEPS_UP back-to-back
  // 128×128 SWZ128 atoms stacked along the K axis (one TMA per atom);
  // bar_w[0] is armed once with the TOTAL tx_bytes so a single
  // `mbarrier.try_wait.parity` on the calc side drains all atoms.
  //
  // For subsequent experts inside the same helper invocation, the
  // previous expert's K-loop stitch (at s=K_TILES-1 COMPUTE) arms
  // bar_w[0] + TMAs w[0] of the next expert.  No pre-loop work there.
  //
  // Compiled out under MONO_PROFILE_SKIP_PREFETCH_UP; the matching calc-
  // warp wait on bar_w[0] inside the K-loop is also compiled out so
  // there is no spin-forever deadlock.
  constexpr uint32_t UP_W_TX_BYTES_PER_SUBSTEP = 16384u;  // 128×128 fp8 atom
  constexpr uint32_t UP_W_TX_BYTES_TOTAL =
      UP_W_TX_BYTES_PER_SUBSTEP * K_SUBSTEPS;            // 16 KB / 32 KB
  constexpr uint32_t UP_A_TX_BYTES_PER_SUBSTEP = 2048u;  // 8 tok × 128 K bf16
  constexpr uint32_t UP_A_TX_BYTES_TOTAL =
      UP_A_TX_BYTES_PER_SUBSTEP * K_SUBSTEPS;  // 2 KB / 4 KB
  if (is_tma_launcher_thread<Dims>() && expert_start < expert_count) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH_UP
    const uint32_t first_id = shmem->experts[expert_start].id;
    mbarrier_arrive_expect_tx(&shm->bar_w[0],
                              /*tx_bytes=*/UP_W_TX_BYTES_TOTAL);
    #pragma unroll
    for (uint32_t kk = 0; kk < K_SUBSTEPS; ++kk) {
      tma_load_up_wgmma_tile(up_weights_desc, /*expert_id=*/first_id,
                             /*N=*/Dims::N,
                             /*base_row_up=*/base_row_up,
                             /*k_start=*/kk * K_STEP_WGMMA,
                             /*dest_slot=*/&shm->w_wgmma[0][kk * W_UP_M][0],
                             /*bar=*/&shm->bar_w[0]);
    }
  #endif
  }

  // ── Phase-3 expert loop ───────────────────────────────────────────────
  MONO_PHASE_TIMESTAMP(t_up_after_preloop);
  for (uint32_t e = expert_start; e < expert_count; e += expert_stride) {
    const uint32_t id = shmem->experts[e].id;
    const bool has_next_e = (e + expert_stride < expert_count);
    const uint32_t next_id =
        has_next_e ? shmem->experts[e + expert_stride].id : 0u;

    // Per-expert parity state.  bar_{w,a}[0] are always pre-armed at the
    // start of each expert (by Phase-1 greedy / helper pre-loop / prior
    // expert's stitch), so register 0 correctly expects physical 1 on
    // the first try_wait.parity.  bar_{w,a}[1] are first armed inside
    // this expert's iter 0 COMPUTE, so register 0 expects physical 1
    // on iter 1's first wait.
    uint32_t parity_w[2] = {0, 0};
    uint32_t parity_a[2] = {0, 0};

    // Reset per-expert accumulators.
    final_d0 = final_d1 = final_d2 = final_d3 = 0.f;

    // Load this expert's block-wise weight scales.  Scales cover 2
    // row-blocks (gate + up) × UP_SCALE_COLS col-blocks.  Both WGs share
    // the same scales (see moe_internal.h comment on UP_SCALE_TILE_SIZE).
    // Synchronous SHM write (32 elements) by prefetch warp 0; the
    // iter-0 QUANT→COMPUTE sync below publishes it to calc warps.
    if (is_prefetch_warp<Dims>()) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH_UP
      moe_request_up_scale_for_row<Dims>(expert_scales_up, id, base_row_up,
                                         shm->up_scale[0]);
  #endif
    }

    // ── Main K-loop (Stage A: QUANT-first, mbarrier-only sync) ─────────
    //
    // Pipeline per iteration:
    //   QUANT half (calc):  wait bar_a[s%2]; bf16_in[s%2] → fp8_act[s%2].
    //   __syncthreads()  ← publishes fp8_act + act_scale to calc warps,
    //                       and (on iter 0) publishes up_scale from the
    //                       prefetch warp's synchronous load above.
    //   COMPUTE half:
    //     calc:      wait bar_w[s%2]; K_SUBSTEPS × (4× WGMMA + scale-apply).
    //     launcher:  arm + TMA the NEXT slot's K_SUBSTEPS weight + bf16
    //                atoms (UP_W_TX_BYTES_TOTAL + UP_A_TX_BYTES_TOTAL
    //                bytes total).  Both are issued here so the launcher
    //                is guaranteed to run AFTER the calc warp's wait on
    //                bar_w[cur_slot] has completed (the __syncthreads()
    //                between QUANT and COMPUTE ensures this).  Issuing
    //                the weight TMA in QUANT instead would create a race:
    //                the launcher could arm bar_w[next_slot] before the
    //                calc warp's wait on bar_w[next_slot] from the
    //                previous iteration has returned, causing an illegal
    //                double-arm (arrival counter goes negative).
    //                target = (s+1, current expert) for intra-expert
    //                         steps, or (0, next expert) on the last
    //                         step when a next expert is scheduled.
    //                When no next step and no next expert, skip — the
    //                trailing barriers are left idle; Phase 4 reinits.
    //   (no trailing sync — the next iter's QUANT re-establishes order
    //    via try_wait.parity on bar_a, and COMPUTE via bar_w.)
    for (uint32_t s = 0; s < K_TILES; ++s) {
      const uint32_t cur_slot = s & 1;
      const uint32_t next_slot = (s + 1) & 1;
      const bool has_next_s = (s + 1 < K_TILES);

      // ───── QUANT half ───────────────────────────────────────────────
      if (is_calc) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH_UP
        // Wait on activation tile arrival. Gated by SKIP_PREFETCH (not
        // SKIP_CALC) because the launcher below also skips the matching
        // `arrive_expect_tx` when SKIP_PREFETCH is defined; skipping
        // the wait on the calc side avoids a spin-forever deadlock.
        // Under SKIP_CALC (launcher still issues TMAs) the wait stays
        // so traces capture the full barrier-stall cost.
        while (!mbarrier_try_wait_parity(&shm->bar_a[cur_slot],
                                         parity_a[cur_slot])) {
        }
        parity_a[cur_slot] ^= 1;
  #endif

  #ifndef MONO_PROFILE_SKIP_CALC_UP
        // Each calc warp `w ∈ {0..7}` quantizes token `w`; warps
        // whose `tok >= batch_size` get zero-filled `fp8_act` and
        // `act_scale = 1.0` inside `moe_streaming_quantize_k128`
        // (ragged-batch isolation matches the cp.async reference
        // byte-for-byte on tokens `[0, batch_size)`).
        //
        // For K_STEP > K_STEP_WGMMA the slot holds K_SUBSTEPS_UP
        // back-to-back 128-K bf16 / fp8 atoms; the quantize helper
        // takes one [T_TILE][128] bf16 atom plus its [8][T_TILE][16]
        // fp8 atom per call, with the per-128-K block scale written
        // to `act_scale[tok][s * K_SUBSTEPS + kk]`.
        const uint32_t tok = warp;
    #pragma unroll
        for (uint32_t kk = 0; kk < K_SUBSTEPS; ++kk) {
          moe_streaming_quantize_k128<Dims>(
              shm->bf16_in[cur_slot][kk], shm->fp8_act[cur_slot][kk], tok,
              batch_size, &shmem->act_scale[tok][s * K_SUBSTEPS + kk]);
        }
  #endif
      }

      __syncthreads();

      // ───── COMPUTE half ─────────────────────────────────────────────
      if (is_calc) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH_UP
        // Wait on weight tile arrival.  Same gating rationale as the
        // bar_a wait above — compiled in/out together with the
        // launcher's arm.
        while (!mbarrier_try_wait_parity(&shm->bar_w[cur_slot],
                                         parity_w[cur_slot])) {
        }
        parity_w[cur_slot] ^= 1;
  #endif

  #ifndef MONO_PROFILE_SKIP_CALC_UP
        // WGMMA descriptor bases per WG.  In the M-stacked SHM layout,
        // substep `kk` occupies SHM rows `[kk*128 .. kk*128 + 128)`,
        // so the WG row offset (0 for WG0, 64 for WG1) is added on top
        // of `kk * 128` to pick the per-WG 64-row half within each
        // 128-row substep atom.
        // WG0 substep 0: rows [0..63]    → &w_wgmma[slot][0][0]
        // WG1 substep 0: rows [64..127]  → &w_wgmma[slot][64][0]
        // WG0 substep 1: rows [128..191] → &w_wgmma[slot][128][0]
        // WG1 substep 1: rows [192..255] → &w_wgmma[slot][192][0]
        const void* a_slot_base = (const void*)&shm->w_wgmma[cur_slot][0][0];
        const uint32_t wg_offset_bytes = is_wg1 ? 8192u : 0u;
        // Bytes between consecutive 128-row substep atoms in
        // `w_wgmma[slot]`: 128 rows × 128 K-bytes = 16 KB.
        constexpr uint32_t K_SUBSTEP_W_BYTES = 16384u;

        // Per-substep activation base: kk-th 1024-B SWZ128 atom inside
        // the activation slot (addressed via `fp8_act[slot][kk]`).

        // Chain 4 WGMMAs per K-substep, each consuming K=32 (= 2
        // consecutive K-chunks of 16 from the fp8 activation tile).
        // Scales are applied at every K=128 boundary (matching the
        // block-wise FP8 scale granularity).
        constexpr uint32_t A_K_STRIDE = 2u * static_cast<uint32_t>(A_LBO);
    #pragma unroll
        for (uint32_t kk = 0; kk < K_SUBSTEPS; ++kk) {
          // A new `wgmma.fence` is required at the start of every group
          // of dependent WGMMAs (one fence ↔ one commit-group/wait-group
          // pair below).  Folding the fence outside the kk loop relies
          // on a single commit+wait pair covering all substeps, which
          // would force the compiler to emit a single "supergroup" of
          // WGMMAs without intermediate scale-apply ordering — incorrect.
          wgmma_fence();

          // Per-substep weight base: kk-th 128-row substep atom + this
          // WG's 64-row half within the atom.
          const void* a_kk_base =
              (const void*)((const char*)a_slot_base + kk * K_SUBSTEP_W_BYTES +
                            wg_offset_bytes);

    #pragma unroll
          for (uint32_t j = 0; j < WGMMAS_PER_SUBSTEP; ++j) {
            const void* a_ptr =
                (const void*)((const char*)a_kk_base + j * A_K_STRIDE);
            const void* b_ptr =
                (const void*)&shm->fp8_act[cur_slot][kk][j * 2][0][0];
            uint64_t desc_a = make_wgmma_desc(a_ptr, A_LBO, A_SBO, A_SWIZZLE);
            uint64_t desc_b = make_wgmma_desc(b_ptr, B_LBO, B_SBO, 0);
            wgmma_m64n8k32_e4m3_e4m3_f32(desc_a, desc_b, chunk_d0, chunk_d1,
                                         chunk_d2, chunk_d3);
          }

          wgmma_commit_group();
          wgmma_wait_group<0>();

          // ── Scale-apply at the K=128 boundary (per-substep) ──────────
          //
          // HOT-PATH BRANCH HYGIENE (ptxas C7520 fix).  The chain of 4
          // `wgmma.mma_async` above, and the one that will fire on the
          // next K-substep, must not be separated by any *within-warp*
          // divergent control flow.  When they are, ptxas inserts a
          // WG.AR (warp-group arrive-release) fence into the divergent
          // path and emits a C7520 perf warning.  Both former offenders
          // (lane-divergent batch_size predicate and within-warp
          // gate/up branch) are folded into unconditional reads — the
          // quantize path zero-fills out-of-range tokens and writes
          // `act_scale[tok][...] = 1.0f`, so the WGMMA sees a zero
          // B-operand and the scale-apply is numerically irrelevant
          // for those lanes.
          //
          // Scale indices for outer step `s`, substep `kk`:
          //   * activation: `act_scale[tok][s * K_SUBSTEPS + kk]`
          //   * weight:     `up_scale[0][s * K_SUBSTEPS + kk + ws_off]`
          // For K_STEP_UP=K_STEP_WGMMA (legacy 128-K step) K_SUBSTEPS=1
          // and the index collapses to the original `s` form.
          const uint32_t kblk = s * K_SUBSTEPS + kk;
          const uint32_t ws_off = is_gate_half ? 0u : UP_SCALE_COLS;
          const float ws = shm->up_scale[0][kblk + ws_off];
          const uint32_t tok_02 = (lane % 4) * 2;
          const uint32_t tok_13 = tok_02 + 1;
          const float as_02 = shmem->act_scale[tok_02][kblk];
          const float as_13 = shmem->act_scale[tok_13][kblk];
          final_d0 += chunk_d0 * ws * as_02;
          final_d1 += chunk_d1 * ws * as_13;
          final_d2 += chunk_d2 * ws * as_02;
          final_d3 += chunk_d3 * ws * as_13;
          chunk_d0 = chunk_d1 = chunk_d2 = chunk_d3 = 0.f;
        }
  #endif
      }

      // Launcher runs IN PARALLEL with the WGMMA above.  Both weight
      // and bf16 TMAs are issued here (not in QUANT) to avoid a
      // double-arm race: the __syncthreads() between QUANT and COMPUTE
      // guarantees the calc warp's wait on bar_w[cur_slot] from the
      // previous iteration has completed before the launcher arms
      // bar_w[next_slot] for the next iteration.
      //
      // For K_STEP > K_STEP_WGMMA the launcher issues K_SUBSTEPS_UP
      // back-to-back TMAs per slot (one per 128-K substep, stacked
      // along the K axis in SHM).  Both barriers are armed once with
      // the TOTAL tx_bytes so a single `mbarrier.try_wait.parity` on
      // the calc side drains all atoms.
      //
      // Compiled out under MONO_PROFILE_SKIP_PREFETCH_UP; the matching
      // calc-warp waits on bar_{w,a}[next_slot] in the next iteration
      // are also compiled out so there is no spin-forever deadlock.
      if (is_tma_launcher_thread<Dims>()) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH_UP
        if (has_next_s) {
          // Intra-expert: fetch (s+1) tiles of the CURRENT expert.
          const uint32_t next_k_start = (s + 1) * K_STEP;
          mbarrier_arrive_expect_tx(&shm->bar_w[next_slot],
                                    /*tx_bytes=*/UP_W_TX_BYTES_TOTAL);
    #pragma unroll
          for (uint32_t kk = 0; kk < K_SUBSTEPS; ++kk) {
            tma_load_up_wgmma_tile(
                up_weights_desc, /*expert_id=*/id,
                /*N=*/Dims::N,
                /*base_row_up=*/base_row_up,
                /*k_start=*/next_k_start + kk * K_STEP_WGMMA,
                /*dest_slot=*/&shm->w_wgmma[next_slot][kk * W_UP_M][0],
                /*bar=*/&shm->bar_w[next_slot]);
          }
          mbarrier_arrive_expect_tx(&shm->bar_a[next_slot],
                                    /*tx_bytes=*/UP_A_TX_BYTES_TOTAL);
    #pragma unroll
          for (uint32_t kk = 0; kk < K_SUBSTEPS; ++kk) {
            tma_load_bf16_input_tile(
                activations_desc,
                /*k_start=*/next_k_start + kk * K_STEP_WGMMA,
                &shm->bf16_in[next_slot][kk][0][0], &shm->bar_a[next_slot]);
          }
        } else if (has_next_e) {
          // End-of-expert stitch: fetch iter-0 tiles of the NEXT expert.
          // For K_TILES even, `next_slot == 0` — matches the next
          // expert's iter-0 cur_slot.
          mbarrier_arrive_expect_tx(&shm->bar_w[next_slot],
                                    /*tx_bytes=*/UP_W_TX_BYTES_TOTAL);
    #pragma unroll
          for (uint32_t kk = 0; kk < K_SUBSTEPS; ++kk) {
            tma_load_up_wgmma_tile(
                up_weights_desc, /*expert_id=*/next_id,
                /*N=*/Dims::N,
                /*base_row_up=*/base_row_up,
                /*k_start=*/kk * K_STEP_WGMMA,
                /*dest_slot=*/&shm->w_wgmma[next_slot][kk * W_UP_M][0],
                /*bar=*/&shm->bar_w[next_slot]);
          }
          mbarrier_arrive_expect_tx(&shm->bar_a[next_slot],
                                    /*tx_bytes=*/UP_A_TX_BYTES_TOTAL);
    #pragma unroll
          for (uint32_t kk = 0; kk < K_SUBSTEPS; ++kk) {
            tma_load_bf16_input_tile(activations_desc,
                                     /*k_start=*/kk * K_STEP_WGMMA,
                                     &shm->bf16_in[next_slot][kk][0][0],
                                     &shm->bar_a[next_slot]);
          }
        }
          // Else: last expert's last iteration — leave barriers idle.
  #endif
      }

      // NO trailing __syncthreads() — the next iter's QUANT/COMPUTE
      // waits on mbarriers re-establish acquire ordering for any async
      // writes, and the next iter's QUANT→COMPUTE sync re-establishes
      // thread visibility for generic writes (up_scale, act_scale,
      // fp8_act).
    }  // end K-loop

    MONO_PHASE_TIMESTAMP_IF(t_up_after_expert0_kloop, e == expert_start);

    // ── End-of-expert: write final_d to partial_result.wgmma_out[128][8] ──
    // Canonical WGMMA D-matrix layout per thread (m64n8k32):
    //   d[0]: row = warp_in_wg*16 + lane/4 + 0,  col = (lane%4)*2 + 0
    //   d[1]: row = warp_in_wg*16 + lane/4 + 0,  col = (lane%4)*2 + 1
    //   d[2]: row = warp_in_wg*16 + lane/4 + 8,  col = (lane%4)*2 + 0
    //   d[3]: row = warp_in_wg*16 + lane/4 + 8,  col = (lane%4)*2 + 1
    // For WG1, rows shift by +64 in the full 128-row output tile.
    if (is_calc) {
  #ifndef MONO_PROFILE_SKIP_CALC_UP
      const uint32_t wg_row_offset = is_wg1 ? 64u : 0u;
      const uint32_t row_base = wg_row_offset + warp_in_wg * 16 + lane / 4;
      const uint32_t col_base = (lane % 4) * 2;
      shm->partial_result.wgmma_out[row_base + 0][col_base + 0] = final_d0;
      shm->partial_result.wgmma_out[row_base + 0][col_base + 1] = final_d1;
      shm->partial_result.wgmma_out[row_base + 8][col_base + 0] = final_d2;
      shm->partial_result.wgmma_out[row_base + 8][col_base + 1] = final_d3;
  #endif
    }
    __syncthreads();

    // ── SiLU + fused fp8 quantization write-back ──
    //
    // The 128 M rows form 64 output columns of this up-block:
    //   WG0 half: gate rows [0..31]  + up rows [32..63]  → out_cols [0..31]
    //   WG1 half: gate rows [64..95] + up rows [96..127] → out_cols [32..63]
    //
    // Thread mapping (warps 0..7, 256 calc threads):
    //   tok          = warp     (0..7) — one token per warp
    //   col_in_half  = lane     (0..31)
    //
    // Each lane computes BOTH halves:
    //   val1 = SiLU(WG0 gate, WG0 up) at out_col_1 = base_row_up + col_in_half
    //   val2 = SiLU(WG1 gate, WG1 up) at out_col_2 = base_row_up + 32 +
    //   col_in_half
    //
    // The warp-reduce of max(|val1|, |val2|) across the 32 lanes yields
    // the per-token block max over all 64 output cols of this up-block.
    //
    // block_scale = block_max / 448
    // inv_scale   = 448 / block_max
    // q1 = (AQ_element)(val1 * inv_scale)   (saturating round-to-nearest-even)
    // q2 = (AQ_element)(val2 * inv_scale)
    //
    // Writes:
    //   spec->temp_fp8[dest_row * N + out_col_1] = q1
    //   spec->temp_fp8[dest_row * N + out_col_2] = q2
    //   (lane 0 only) spec->temp_act_scale[dest_row * (N/64) + up_block_idx]
    //                 = block_scale
    //
    // Destination row: `dest_row = shm->sorted_slot[pair]` —
    // expert-sorted row in the reorganized temp_fp8, consumed by the
    // Phase-4 bulk-per-expert TMA (spec R11.3, R11.4).
    //
    // Guards: store && tok < batch_size && out_col < Dims::N.
    // (The warp-reduce requires all 32 lanes to participate, so we
    //  compute val1/val2 on every lane but zero out-of-range lanes'
    //  contributions to the max and skip their writes.)
    //
    // NOTE: no write to spec->temp_bf16 on the WGMMA path — the scalar
    // path retains that behavior unchanged elsewhere.
    if (is_calc) {
  #ifndef MONO_PROFILE_SKIP_CALC_UP
      // `tok` from warp id, `col_in_half` from lane id.
      const uint32_t tok = warp;          // 0..7, one per calc warp
      const uint32_t col_in_half = lane;  // 0..31

      // Uniform-across-warp: find this token's top-K match for the
      // current expert.  All 32 lanes of the warp agree on these.
      bool store = false;
      float rw = 0.f;
      uint32_t dest_row = 0;
      if (tok < batch_size) {
        for (uint32_t k = 0; k < top_k; ++k) {
          if (shmem->topk_ids_flat[tok * MAX_TOPK + k] == (uint16_t)id) {
            store = true;
            rw = shmem->topk_weights_flat[tok * MAX_TOPK + k];
            const uint32_t pair = tok * top_k + k;
            // Expert-sorted reorganization (spec R11.3): Phase 4's
            // bulk-per-expert TMA expects each expert's routed tokens
            // to occupy a contiguous run of rows in spec->temp_fp8.
            dest_row = shm->sorted_slot[pair];
            break;
          }
        }
      }

      // Compute val1 / val2 on every lane of the warp (needed so that
      // the subsequent warp-reduce sees all 64 cols of this up-block).
      const float gate1 = shm->partial_result.wgmma_out[col_in_half][tok];
      const float up1 = shm->partial_result.wgmma_out[col_in_half + 32][tok];
      const float gate2 = shm->partial_result.wgmma_out[col_in_half + 64][tok];
      const float up2 = shm->partial_result.wgmma_out[col_in_half + 96][tok];

      float val1 = rw * up1 * gate1 / (1.0f + __expf(-gate1));
      float val2 = rw * up2 * gate2 / (1.0f + __expf(-gate2));

      const uint32_t out_col_1 = base_row_up + col_in_half;
      const uint32_t out_col_2 = base_row_up + 32 + col_in_half;
      const bool write1 = store && (out_col_1 < Dims::N);
      const bool write2 = store && (out_col_2 < Dims::N);

      // Out-of-range lanes must not influence block_max; zero their
      // contributions.  (Also zero everything when !store or
      // tok >= batch_size so block_max is meaningful on skipped warps
      // — although we won't write either way.)
      if (!write1) val1 = 0.f;
      if (!write2) val2 = 0.f;

      // Warp-reduce max(|val1|, |val2|) across the 32 lanes → max over
      // all 64 output cols of this up-block for this token.
      float local_max = fmaxf(fabsf(val1), fabsf(val2));
      float block_max = warp_reduce_max_float(local_max);
      if (block_max < __FLT_MIN__) block_max = 1.0f;

      constexpr float FP8_MAX = 448.0f;
      constexpr float FP8_MAX_INV = 1.0f / 448.0f;
      const float block_scale = block_max * FP8_MAX_INV;
      const float inv_scale = FP8_MAX / block_max;

      // Saturating round-to-nearest-even fp32 → fp8 e4m3 conversion
      // (matches __nv_fp8x4_e4m3 with __NV_SATFINITE).
      const AQ_element q1 = (AQ_element)(val1 * inv_scale);
      const AQ_element q2 = (AQ_element)(val2 * inv_scale);

      if (store && tok < batch_size) {
        // Token-major layout for `temp_fp8`:
        //   byte_off(row, col) = row * N + col
        // Matches the SWZ128 Major::K B-operand canonical form
        // (`tok * 128 + kc * 16 + ki`) once the down-proj TMA applies
        // the 8-row × 128-byte XOR swizzle at write time into SHM
        // `a_down_wgmma[tok][kc][ki]`.
        if (write1) {
          spec->temp_fp8[dest_row * Dims::N + out_col_1] = q1;
        }
        if (write2) {
          spec->temp_fp8[dest_row * Dims::N + out_col_2] = q2;
        }
        // Lane 0 of each warp writes the per-(dest_row, up_block_idx)
        // scale once.  All lanes in the warp hold the same block_scale
        // after the warp-reduce, so picking lane 0 is arbitrary.
        if (lane == 0) {
          constexpr uint32_t SCALE_COLS =
              MoEGemmSpec<Dims>::TEMP_ACT_SCALE_COLS;  // = Dims::N / 64
          spec->temp_act_scale[dest_row * SCALE_COLS + effective_bid] =
              block_scale;
        }
      }
  #endif
    }

    // ── Tail of expert loop ──
    //
    // The next expert's iter-0 TMAs (weight + bf16) were already issued
    // during this expert's s=K_TILES-1 COMPUTE half via the
    // cross-expert stitch (see the launcher branch in the K-loop).
    // No tail prefetch is needed here; the mbarrier chain carries
    // across the expert boundary without any __syncthreads().
    //
    // A single trailing __syncthreads() publishes the SiLU writeback's
    // spec->temp_fp8 / spec->temp_act_scale writes to every thread in
    // the block before the next iteration's prefetch warp overwrites
    // `shm->up_scale[0]`.  This sync also aligns the launcher thread
    // with calc warps so the launcher cannot race ahead and issue the
    // stitch-triggered mbarrier arm for expert e+2 before expert e+1's
    // iter-0 QUANT wait has consumed the stitch arrival for expert e+1.
    __syncthreads();

    MONO_PHASE_TIMESTAMP_IF(t_up_after_expert0_writeback, e == expert_start);
  }  // end expert loop
}

}  // namespace moe_monokernel

#endif