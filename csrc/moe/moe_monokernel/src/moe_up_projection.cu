
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

    // Map dst_row to global weight row (V2 Pair_Layout): each warp's
    // 16-row stripe holds 8 gate rows in the d[0..1] half AND 8 up rows
    // in the d[2..3] half, so silu(gate)*up becomes a per-lane register
    // op after the WGMMA.  See `up-proj-gate-up-pair-layout` design,
    // Component 2.
    unsigned global_row;
    {
      const unsigned wg_half = dst_row / 64;      // 0 (WG0) or 1 (WG1)
      const unsigned r_in_wg = dst_row % 64;      // 0..63
      const unsigned warp_in_wg = r_in_wg / 16;   // 0..3
      const unsigned r_in_warp = r_in_wg % 16;    // 0..15
      const bool is_up = r_in_warp >= 8;          // d[2..3] half
      const unsigned local_r = r_in_warp & 7;     // 0..7
      const unsigned wg_row_base = wg_half * 32;  // 0 or 32
      global_row = base_row + wg_row_base + warp_in_wg * 8 + local_r +
                   (is_up ? Dims::N : 0);
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
 *       corresponding fp8 atoms regardless of source tile contents,
 *       so leaving stale rows for `tok >= batch_size` is safe.
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
 * @brief Async (cp.async) variant of `moe_request_up_scale_for_row`.
 *
 * Issues the per-expert block-scale tile load as non-blocking 4-byte
 * `cp.async.ca.shared.global` copies instead of a synchronous GM read.
 * Used by the V2 (Pair_Layout) up-projection to PREFETCH the NEXT
 * expert's scales during the CURRENT expert's K-loop, overlapping the
 * ~0.5 µs cold-miss DRAM latency with compute instead of paying it
 * exposed at the cross-expert boundary (see the GAP-a analysis in the
 * `up-proj-gate-up-pair-layout` spec).
 *
 * The caller must `cp_async_commit_group()` after issuing and drain
 * with `cp_async_wait_group<N>()` before the scale values are read.
 * Prefetch-warp 0 only (32 lanes, one scalar each for the 32-element
 * Qwen3.5 tile).
 */
template <typename Dims>
__device__ inline void moe_prefetch_up_scale_for_row_async(
    const S_element* __restrict__ expert_scales_up, std::uint32_t id,
    unsigned base_row_up, S_element* __restrict__ dest) {
  constexpr uint32_t COLS = Dims::UP_SCALE_COLS;
  constexpr uint32_t TILE = 2 * COLS;
  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_prefetch_warp<Dims>();

  if (warp == 0 && thread < TILE) {
    uint32_t rb_local = thread / COLS;
    uint32_t kb = thread % COLS;
    uint32_t row = base_row_up + rb_local * Dims::N;
    uint32_t rb_global = row / Dims::BLOCK_SCALE_ROW;
    const S_element* src =
        &expert_scales_up[id * Dims::UP_SCALE_ROWS * Dims::UP_SCALE_COLS +
                          rb_global * Dims::UP_SCALE_COLS + kb];
    cp_async_cg_4(&dest[thread], src);
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
          float as0 = shmem->act_scale[full_k_col / ACT_BLOCK][tok0];
          float as1 = shmem->act_scale[full_k_col / ACT_BLOCK][tok1];
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
// prefetch-warp `cp.async` loaders for the fp8 expert-weight tile with
// `cp.async.bulk.tensor.2d` issued by a single TMA launcher thread
// (warp 8, lane 0). Completion is signalled via SHM mbarrier `bar_w[2]`;
// consumer warps wait with `mbarrier.try_wait.parity` instead of
// `cuda::pipeline_consumer_wait_prior`.  The activation operand is
// sourced from `fp8_act_full` (populated by Phase 1 + Phase 2 of
// `moe_kernel_topk_BS8`); no per-K-step bf16-input TMA or `bar_a` arm
// fires from this helper.
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
    std::uint32_t expert_stride = 1) {
  static_assert(Dims::BS <= 8);
  using CoreDims = MoECoreDims<Dims>;

  // `activations_in` / `expert_weights_up` are retained on the parameter
  // list for signature parity with the `cp.async` reference but are not
  // dereferenced on the TMA path — all GM reads go through the two TMA
  // descriptors.
  (void)activations_in;
  (void)expert_weights_up;

  // Caller contract (post topk-bs8-tma-prefetch-quant-fusion):
  //   * The Phase-1 routing-window TMA (in `moe.cu`) has already
  //     fetched the full BF16 input tile into `bf16_in_full`.
  //   * Phase 2 has run `routing_phase_quantize` and a block-wide
  //     `__syncthreads()`, so `fp8_act_full[k_block]` and
  //     `act_scale[token][k_block]` are visible to every calc warp
  //     for `k_block in [0, K_BLOCKS_TOTAL)`.
  //   * `bar_w[0..1]` have been initialized and release-fenced.
  //
  // Stage A pipeline:
  //   * Pre-loop: helper arms bar_w[0] + TMAs w[0] of expert_start at k=0.
  //   * K-loop: iter s waits bar_w[s%2], runs WGMMA +
  //             scale-apply (B operand from `fp8_act_full`), and the
  //             launcher arms the NEXT slot's bar_w + weight TMAs
  //             (intra-expert s+1 or next expert's k=0 stitch).
  //   * No bar_a, no bf16 TMAs, no QUANT half, no QUANT/COMPUTE
  //     __syncthreads() — the activation operand is already in
  //     `fp8_act_full` for the entire K range (Req 3.1, 3.2, 3.3,
  //     3.6, 3.7, 3.12).

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
  // B operand (K-major, N=8): 1 N-block, LBO between K-core-matrices.
  // Always SWIZZLE_NONE — the activation tile is 8-token × 128-K bf16
  // and small enough that bank-conflict cost is bounded.
  //
  // `B_LBO` = bytes between successive 8-row × 16-byte WGMMA core
  // matrices along K = the byte stride between successive kc atoms in
  // `fp8_act_full`'s `[FP8_NUM_CHUNKS][T_TILE_PADDED][FP8_K_CHUNK]`
  // per-kblk layout (see comment on `fp8_act_full` in `moe_internal.h`
  // for the kc-padding design).  The pad widens each kc atom from 128
  // B (T_TILE=8) to 144 B (T_TILE_PADDED=9), and the 9th token row of
  // every kc atom is unused — the WGMMA core matrix is still rows
  // [0..7] × bytes [0..15] (= 128 B contiguous) at the head of each
  // atom, and `B_LBO = 144` steps over the unused 9th row to land on
  // the next kc atom's core matrix.
  constexpr uint64_t B_LBO =
      static_cast<uint64_t>(
          MoE_SHM<Dims>::U::TinyDataWGMMA_TMA::FP8_ACT_T_TILE_PADDED) *
      static_cast<uint64_t>(
          MoE_SHM<Dims>::U::TinyDataWGMMA_TMA::FP8_ACT_K_CHUNK);  // 9 × 16 =
                                                                  // 144 B
  constexpr uint64_t B_SBO = B_LBO;  // unused (only 1 N-block for N=8)

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
  (void)is_gate_half;

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
  // Entry contract (post-fusion):
  //   * `bar_w[0..1]` are initialized (arrival_count=1) and
  //     release-fenced by the kernel prologue in `moe.cu`.
  //   * Phase 1 + Phase 2 have populated `bf16_in_full` and then
  //     `fp8_act_full` + `act_scale[token][k_block]` for every
  //     `k_block ∈ [0, K_BLOCKS_TOTAL)`; the Phase-2 trailing
  //     `__syncthreads()` published those writes to all warps.
  //   * `bar_a[0..1]` are NOT initialized by `moe.cu`'s prologue —
  //     they are reused by the down-projection in Phase 4 and
  //     re-initialized in the down-proj prologue.
  //
  // This helper never re-initializes barriers and never issues
  // bf16-input TMAs; the K-loop reads FP8 activations directly from
  // `fp8_act_full` (Req 3.4).

  // Stage A requires an even K_TILES so the end-of-K-loop launcher arm
  // lands on `next_slot = K_TILES % 2 = 0`.  That slot-0 stitch is what
  // the next expert's iter-0 COMPUTE waits on; an odd K_TILES would
  // land the stitch on slot 1, breaking the cross-expert mbarrier
  // chain.  At K_STEP_UP=128 (default) HIDDEN_STATES=2048 → K_TILES=16;
  // at K_STEP_UP=256 → K_TILES=8; both satisfy the invariant.
  static_assert(K_TILES % 2 == 0,
                "Stage-A pipeline requires K_TILES to be even so the "
                "end-of-loop stitch arms the same slot that the next "
                "expert's iter-0 COMPUTE waits on.");

  // ── Pre-loop: arm bar_w[0] + TMA w[0] of expert_start at k=0 ──────────
  //
  // bar_w[0] is not pre-armed by the caller; this helper fires the
  // first expert's weight TMA here.  iter-0 COMPUTE waits on
  // bar_w[0] before issuing WGMMAs.
  // For subsequent experts inside the same helper invocation, the
  // previous expert's K-loop stitch (at s=K_TILES-1 COMPUTE) arms
  // bar_w[0] + TMAs w[0] of the next expert.  No pre-loop work there.
  //
  // Compiled out under MONO_PROFILE_SKIP_PREFETCH_UP; the matching calc-
  // warp wait on bar_w[0] inside the K-loop is also compiled out so
  // there is no spin-forever deadlock.
  constexpr uint32_t UP_W_TX_BYTES_PER_SUBSTEP = 16384u;  // 128×128 fp8 atom
  constexpr uint32_t UP_W_TX_BYTES_TOTAL =
      UP_W_TX_BYTES_PER_SUBSTEP * K_SUBSTEPS;  // 16 KB / 32 KB
  // The bf16-input TMA + `bar_a` arm have been removed from this
  // helper (Req 3.7).  Phase-1 + Phase-2 in `moe.cu` populate
  // `fp8_act_full` once per kernel invocation; the K-loop reads it
  // directly.  The legacy `UP_A_TX_BYTES_*` constants live in the
  // kernel prologue for now (until task 12 removes the legacy
  // hoisted Step A entirely).
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
    // 4-deep lookahead: also pre-arm bar_w[1] with expert_start's
    // iter-1 weight tile, so iter-1's calc-warp wait doesn't have to
    // wait for the launcher to issue the TMA from cold.  This stitches
    // the lookahead 1 iter earlier than the default pipeline: the
    // launcher starts at iter 0 arming bar_w[2] for iter 2, then the
    // wraparound is 4-deep.
    static_assert(K_TILES >= 4,
                  "4-deep weight lookahead requires K_TILES >= 4 to fit a "
                  "2-iter-ahead arm without wrap-around collisions on "
                  "bar_w[4].");
    mbarrier_arrive_expect_tx(&shm->bar_w[1],
                              /*tx_bytes=*/UP_W_TX_BYTES_TOTAL);
    #pragma unroll
    for (uint32_t kk = 0; kk < K_SUBSTEPS; ++kk) {
      tma_load_up_wgmma_tile(up_weights_desc, /*expert_id=*/first_id,
                             /*N=*/Dims::N,
                             /*base_row_up=*/base_row_up,
                             /*k_start=*/K_STEP + kk * K_STEP_WGMMA,
                             /*dest_slot=*/&shm->w_wgmma[1][kk * W_UP_M][0],
                             /*bar=*/&shm->bar_w[1]);
    }
  #endif
  }

  // ── Deferred-writeback bookkeeping ────────────────────────────────────
  //
  // Phase-3's per-expert SiLU+fp8 quant writeback is DEFERRED: the
  // writeback is moved to iters `s == 0` / `s == 1` of the next
  // expert, handled by prefetch warps (8..11), so calc warps' iter-0
  // WGMMAs run concurrently with the previous expert's SiLU+quant.
  // The LAST expert in the per-block range has no next iter-0 to
  // defer to and runs its writeback inline on calc warps after the
  // expert loop ends (the post-loop drain).
  //
  // `prev_id_for_writeback` and `has_pending_writeback` are uniform
  // across threads (the loop runs in lockstep).
  uint32_t prev_id_for_writeback = 0;
  bool has_pending_writeback = false;
  (void)prev_id_for_writeback;
  (void)has_pending_writeback;

  // ── Scale double-buffer state ─────────────────────────────────────────
  // We prefetch the NEXT expert's block-scale tile (async cp.async)
  // during the CURRENT expert's K-loop, so the cold GM-miss latency
  // overlaps compute instead of sitting exposed at the cross-expert
  // boundary (the GAP-a cost in the spec's phase timing).
  // `cur_scale_slot` ping-pongs between the two `up_scale[2]` buffers:
  // expert `e` CONSUMES `up_scale[cur_scale_slot]` and PREFETCHES
  // expert `e+1` into `up_scale[cur_scale_slot ^ 1]`.
  uint32_t cur_scale_slot = 0;
  (void)cur_scale_slot;
  {
  #ifndef MONO_PROFILE_SKIP_PREFETCH_UP
    // Prime expert_start's scale into slot 0 (async).  Drained at the
    // K-loop top below before the publish __syncthreads.
    if (is_prefetch_warp<Dims>() && expert_start < expert_count) {
      moe_prefetch_up_scale_for_row_async<Dims>(expert_scales_up,
                                                shmem->experts[expert_start].id,
                                                base_row_up, shm->up_scale[0]);
      cp_async_commit_group();
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

    // Per-expert parity state.  bar_w[0] is always pre-armed at the
    // start of each expert (by the helper pre-loop above for
    // expert_start, or by the prior expert's stitch for subsequent
    // experts), so register 0 correctly expects physical 1 on
    // the first try_wait.parity.  bar_w[1] is first armed inside
    // this expert's iter 0 COMPUTE, so register 0 expects physical 1
    // on iter 1's first wait.
    uint32_t parity_w[4] = {0, 0, 0, 0};

    // Reset per-expert accumulators.
    final_d0 = final_d1 = final_d2 = final_d3 = 0.f;

    // Load this expert's block-wise weight scales.  Scales cover 2
    // row-blocks (gate + up) × UP_SCALE_COLS col-blocks.  Both WGs share
    // the same scales (see moe_internal.h comment on UP_SCALE_TILE_SIZE).
    //
    // V2 (Pair_Layout): the scale for THIS expert was prefetched (async
    // cp.async) one expert earlier into up_scale[cur_scale_slot] — by
    // the pre-loop for expert_start, or by the previous expert's K-loop
    // top for all others.  Here we (a) drain that prefetch with
    // cp_async_wait_group so the values are guaranteed landed before the
    // publish __syncthreads, and (b) issue the prefetch for the NEXT
    // expert into the other buffer so its cold-miss DRAM latency
    // overlaps THIS expert's K-loop instead of sitting exposed at the
    // boundary (the GAP-a cost).
    {
  #ifndef MONO_PROFILE_SKIP_PREFETCH_UP
      if (is_prefetch_warp<Dims>()) {
        // (a) Ensure THIS expert's prefetched scale has landed.  It was
        // issued a full K-loop ago, so this wait is essentially free.
        cp_async_wait_group<0>();
        // (b) Prefetch the NEXT expert's scale into the other buffer.
        if (has_next_e) {
          moe_prefetch_up_scale_for_row_async<Dims>(
              expert_scales_up, next_id, base_row_up,
              shm->up_scale[cur_scale_slot ^ 1u]);
          cp_async_commit_group();
        }
      }
  #endif
    }

    // Phase-timing: after the scale block (scale handling only),
    // before the rank-cache populate.  Splits GAP-a so we can tell
    // scale-handling cost from rank-populate cost.
    MONO_PHASE_TIMESTAMP_IF(t_up_e1_after_scale_block,
                            e == expert_start + expert_stride);

    // ── Per-expert `up_rank_for_tok` cache populate ──
    //
    // The per-expert epilogue runs entirely in registers on
    // calc warps and applies the routing weight `rw` per (lane, tok)
    // pair.  Looking up `rw = topk_weights_flat[tok*MAX_TOPK + k]`
    // requires the topk INDEX `k` for which
    // `topk_ids_flat[tok*MAX_TOPK + k] == id`.  Hoisting that scan
    // out of the per-lane combine to a once-per-expert cache populate
    // by 8 calc threads (one per token) replaces O(top_k) shared-mem
    // reads per (lane, tok) with a single `[tok]` byte broadcast in
    // the epilogue.
    //
    // Placed BEFORE the `up_scale` publish `__syncthreads()` below so
    // the SAME sync publishes both writes — no extra barrier needed.
    // Calc warps are otherwise idle here (the prefetch warp is
    // loading `up_scale[0]` synchronously), so the 8 thread-serial
    // scans cost ~0.05 µs amortized across the expert (Req R1.2,
    // design Component 4 "Routing Weight Cache").
    //
    // ── Vectorized populate ──
    // The naive version had 8 threads (one per token) each do an
    // up-to-`top_k`-iteration *dependent* SHM scan over
    // `topk_ids_flat` — ~0.46 µs on the critical path because each
    // thread serially chases `top_k` SHM reads.  Instead, each token's
    // 8 topk ids are 8×uint16 = 16 contiguous bytes (16-byte aligned
    // for MAX_TOPK=8), so one thread reads them all in a SINGLE 16-byte
    // vector load and compares the 8 ids in registers — no dependent
    // SHM chain.  Still 8 threads (one per token, each fully owns its
    // token), so it's race-free with no extra sync.
    {
  #ifndef MONO_PROFILE_SKIP_CALC_UP
      constexpr std::uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;
      if (thread_in_block < batch_size && MAX_TOPK == 8u) {
        const std::uint32_t tok = thread_in_block;
        // Vector-load this token's 8 topk ids (16 bytes) in one shot.
        // `topk_ids_flat` is `alignas(uint64_t)` (8-byte) and each
        // token slab is `tok*16` bytes in, so 8-byte `uint2` loads are
        // safely aligned (a 16-byte `uint4` would need 16-byte align).
        const uint16_t* idb = &shmem->topk_ids_flat[tok * MAX_TOPK];
        uint16_t ids[8];
        *reinterpret_cast<uint2*>(&ids[0]) =
            *reinterpret_cast<const uint2*>(idb);
        *reinterpret_cast<uint2*>(&ids[4]) =
            *reinterpret_cast<const uint2*>(idb + 4);

        // Also load all 8 routing weights up front (8 independent SHM
        // reads, issued together — they pipeline).  This avoids the
        // DEPENDENT read `topk_weights_flat[tok*MAX_TOPK + k_found]`
        // whose address can't be formed until the id-compare resolves
        // `k_found`; that dependent read was the dominant cost of this
        // populate (~0.45 µs on the cross-expert critical path, since
        // all 384 threads wait on it at the publish __syncthreads).
        const S_element* wb = &shmem->topk_weights_flat[tok * MAX_TOPK];
        float w[8];
    #pragma unroll
        for (std::uint32_t k = 0; k < 8u; ++k) w[k] = wb[k];

        // Branchless select: exactly one k matches per token (topk ids
        // are distinct), so `rw` ends up holding that match's weight
        // and `k_found` its rank — all from registers, no dependent
        // SHM read.
        const uint16_t target = (uint16_t)id;
        uint8_t k_found = 0xFFu;
        float rw_found = 0.f;
    #pragma unroll
        for (std::uint32_t k = 0; k < 8u; ++k) {
          const bool m = (k < top_k) && (ids[k] == target);
          if (m) {
            k_found = static_cast<uint8_t>(k);
            rw_found = w[k];
          }
        }
        shm->up_rank_for_tok[tok] = k_found;
        shm->up_rw_for_tok[tok] = rw_found;
      }
  #endif
    }

    // Phase-timing: cross-expert GAP sub-phase 1 — after the scale
    // load (PF warp) + cache populate (calc warps), before the
    // publish sync.  Recorded on e1's K-loop top (warp 0 lane 0).
    MONO_PHASE_TIMESTAMP_IF(t_up_e1_after_scale_cache,
                            e == expert_start + expert_stride);

    // Publish the per-expert `up_scale` write from the prefetch
    // warp (above) to the calc warps that consume it inside the
    // K-loop scale-apply.  The legacy QUANT/COMPUTE sync used to
    // serve this purpose; it is removed by Req 3.6 / 3.12.  This
    // sync sits OUTSIDE the K-loop, so R3.12 ("at most one
    // __syncthreads() per outer K-step iteration") is preserved.
    //
    // Under V2 (Pair_Layout) this same sync also publishes the
    // `up_rank_for_tok[]` cache populated by 8 calc threads above.
    __syncthreads();

    // Phase-timing: cross-expert GAP sub-phase 2 — after the publish
    // __syncthreads.  Δ from t_up_e1_after_scale_cache = barrier cost
    // (includes the wait for the PF warp's up_scale load to land).
    // The remaining GAP (this → t_up_e1_iter0_after_wait) is the
    // iter-0 bar_w wait for the cross-expert weight TMA.
    MONO_PHASE_TIMESTAMP_IF(t_up_e1_after_publish_sync,
                            e == expert_start + expert_stride);

    // ── Main K-loop (FP8-direct: COMPUTE-only, mbarrier-only sync) ─────
    //
    // Pipeline per iteration:
    //   COMPUTE half:
    //     calc:      wait bar_w[s%2]; K_SUBSTEPS × (4× WGMMA + scale-apply).
    //                B operand reads `fp8_act_full[s * K_SUBSTEPS + kk]`
    //                — single-buffer FP8 produced once per kernel
    //                invocation by Phase 2's `routing_phase_quantize`
    //                and published by the Phase-2 trailing
    //                `__syncthreads()` in `moe.cu` (Req 3.4, 3.5).
    //     launcher:  arm + TMA the NEXT slot's K_SUBSTEPS weight atoms
    //                (UP_W_TX_BYTES_TOTAL bytes total).  No bar_a arm
    //                and no bf16-input TMA (Req 3.7).
    //                target = (s+1, current expert) for intra-expert
    //                         steps, or (0, next expert) on the last
    //                         step when a next expert is scheduled.
    //                When no next step and no next expert, skip — the
    //                trailing barriers are left idle; Phase 4 reinits.
    //   No QUANT half, no QUANT/COMPUTE __syncthreads() (Req 3.1, 3.2,
    //   3.3, 3.6, 3.12).  The next iter's COMPUTE wait on
    //   bar_w[next_slot] re-establishes acquire ordering for the
    //   weight TMA's async writes.
    for (uint32_t s = 0; s < K_TILES; ++s) {
      // 4-deep lookahead: launcher arms `bar_w[(s+2) & 3]` and
      // calc waits on `bar_w[s & 3]`.  Wraparound is 4 iters; the
      // launcher's arm at iter `s+2` lands 2 iters before the
      // matching consumer wait, giving DRAM extra time to drain
      // the cross-expert stitch and the iter-1 weight TMA.
      const uint32_t cur_slot = s & 3u;
      const uint32_t next_slot = (s + 2u) & 3u;
      const bool has_next_s = (s + 2u < K_TILES);

      // ───── COMPUTE half ─────────────────────────────────────────────
      if (is_calc) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH_UP
        // Wait on weight tile arrival.  Compiled in/out together with
        // the launcher's arm; under SKIP_PREFETCH the launcher elides
        // the arm so skipping the wait avoids a spin-forever deadlock.
        while (!mbarrier_try_wait_parity(&shm->bar_w[cur_slot],
                                         parity_w[cur_slot])) {
        }
        parity_w[cur_slot] ^= 1;
  #endif

        // Phase-timing: after-wait on iter 0 / iter 1 of expert 0 and
        // expert 1 (calc warp 0 lane 0 = threadIdx.x == 0).
        MONO_PHASE_TIMESTAMP_IF(t_up_e0_iter0_after_wait,
                                e == expert_start && s == 0u);
        MONO_PHASE_TIMESTAMP_IF(t_up_e0_iter1_after_wait,
                                e == expert_start && s == 1u);
        MONO_PHASE_TIMESTAMP_IF(t_up_e1_iter0_after_wait,
                                e == expert_start + expert_stride && s == 0u);
        MONO_PHASE_TIMESTAMP_IF(t_up_e1_iter1_after_wait,
                                e == expert_start + expert_stride && s == 1u);

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

        // Per-substep activation base: B operand reads
        // `fp8_act_full[s * K_SUBSTEPS + kk]` (single buffer covering
        // all K substeps; produced once per kernel invocation by
        // Phase 2's `routing_phase_quantize`).

        // Chain 4 WGMMAs per K-substep, each consuming K=32 (= 2
        // consecutive K-chunks of 16 from the fp8 activation tile).
        // Scales are applied at every K=128 boundary (matching the
        // block-wise FP8 scale granularity).
        constexpr uint32_t A_K_STRIDE = 2u * static_cast<uint32_t>(A_LBO);
    #pragma unroll
        for (uint32_t kk = 0; kk < K_SUBSTEPS; ++kk) {
          // A new `wgmma.fence` is required at the start of every group
          // of dependent WGMMAs (one fence ↔ one commit-group/wait-group
          // pair below).
          wgmma_fence();

          // Per-substep weight base: kk-th 128-row substep atom + this
          // WG's 64-row half within the atom.
          const void* a_kk_base =
              (const void*)((const char*)a_slot_base + kk * K_SUBSTEP_W_BYTES +
                            wg_offset_bytes);

          // Single-buffer activation atom for this (s, kk):
          //   fp8_act_full[s * K_SUBSTEPS + kk][...]
          // Replaces the legacy `fp8_act[cur_slot][kk]` indexing
          // (Req 3.4).  `cur_slot` is unused for the activation
          // operand and remains in scope only for the weight tile.
          const uint32_t kblk = s * K_SUBSTEPS + kk;

    #pragma unroll
          for (uint32_t j = 0; j < WGMMAS_PER_SUBSTEP; ++j) {
            const void* a_ptr =
                (const void*)((const char*)a_kk_base + j * A_K_STRIDE);
            const void* b_ptr =
                (const void*)&shm->fp8_act_full[kblk][j * 2][0][0];
            uint64_t desc_a = make_wgmma_desc(a_ptr, A_LBO, A_SBO, A_SWIZZLE);
            uint64_t desc_b = make_wgmma_desc(b_ptr, B_LBO, B_SBO, 0);
            wgmma_m64n8k32_e4m3_e4m3_f32(desc_a, desc_b, chunk_d0, chunk_d1,
                                         chunk_d2, chunk_d3);
          }

          wgmma_commit_group();
          wgmma_wait_group<0>();

          // ── Scale-apply at the K=128 boundary (per-substep) ──────────
          //
          // Scale indices for outer step `s`, substep `kk`:
          //   * activation: `act_scale[tok][s * K_SUBSTEPS + kk]`
          //   * weight:     `up_scale[0][s * K_SUBSTEPS + kk + ws_off]`
          // Indexing matches the legacy form (Req 3.5); the values
          // are now produced by `routing_phase_quantize` instead of
          // by the per-K-step `moe_streaming_quantize_k128` call.
          const uint32_t tok_02 = (lane % 4) * 2;
          const uint32_t tok_13 = tok_02 + 1;
          // SHM `act_scale` is laid out as `[blk][tok]` (see comment on
          // its declaration in `MoE_SHM`); the index swap from the
          // legacy `[tok][blk]` form is cosmetic at the source level
          // but eliminates the 4-way bank conflict NCU flagged on
          // these LDS sites.
          const float as_02 = shmem->act_scale[kblk][tok_02];
          const float as_13 = shmem->act_scale[kblk][tok_13];
          {
            // Under Pair_Layout, d[0..1] are gate rows and d[2..3] are
            // up rows within the SAME warp.  Each needs its own weight
            // scale: gate scale at offset 0, up scale at UP_SCALE_COLS.
            // V2 consumes the ping-pong buffer `up_scale[cur_scale_slot]`
            // (prefetched one expert ago).
            const S_element* ws_buf = shm->up_scale[cur_scale_slot];
            const float ws_gate = ws_buf[kblk + 0u];
            const float ws_up = ws_buf[kblk + UP_SCALE_COLS];
            final_d0 += chunk_d0 * ws_gate * as_02;
            final_d1 += chunk_d1 * ws_gate * as_13;
            final_d2 += chunk_d2 * ws_up * as_02;
            final_d3 += chunk_d3 * ws_up * as_13;
          }
          chunk_d0 = chunk_d1 = chunk_d2 = chunk_d3 = 0.f;
        }
  #endif
      }

      // Phase-timing: after-compute on iter 0 / iter 1 of expert 0
      // and expert 1.  Outside the calc-only block so the macro's
      // `is_calc` gating doesn't suppress threadIdx.x == 0 — but
      // threadIdx.x == 0 is itself in calc, so the capture lands
      // at the same point either way.
      MONO_PHASE_TIMESTAMP_IF(t_up_e0_iter0_after_compute,
                              e == expert_start && s == 0u);
      MONO_PHASE_TIMESTAMP_IF(t_up_e0_iter1_after_compute,
                              e == expert_start && s == 1u);
      MONO_PHASE_TIMESTAMP_IF(t_up_e1_iter0_after_compute,
                              e == expert_start + expert_stride && s == 0u);
      MONO_PHASE_TIMESTAMP_IF(t_up_e1_iter1_after_compute,
                              e == expert_start + expert_stride && s == 1u);

      // Launcher runs IN PARALLEL with the WGMMA above.  Only the
      // weight TMA + bar_w arm remain; the bf16-input TMA + bar_a
      // arm have been removed (Req 3.7).  The activation operand is
      // sourced from `fp8_act_full`, which is produced once per
      // kernel invocation by Phase 2 — there is nothing to fetch
      // per K-step on the activation side.
      //
      // For K_STEP > K_STEP_WGMMA the launcher issues K_SUBSTEPS_UP
      // back-to-back weight TMAs per slot (one per 128-K substep,
      // stacked along the K axis in SHM).  bar_w is armed once with
      // the TOTAL tx_bytes so a single `mbarrier.try_wait.parity` on
      // the calc side drains all atoms.
      //
      // Compiled out under MONO_PROFILE_SKIP_PREFETCH_UP; the matching
      // calc-warp wait on bar_w[next_slot] in the next iteration is
      // also compiled out so there is no spin-forever deadlock.
      if (is_tma_launcher_thread<Dims>()) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH_UP
        // 4-deep lookahead: at iter `s` arm `bar_w[(s+2)&3]` for
        // the iter-(s+2) weight tile.  Three cases by source:
        //   (A) Intra-expert: s+2 < K_TILES → fetch CURRENT expert's
        //                     iter-(s+2) tile.
        //   (B) Cross-expert iter-0: s+2 == K_TILES (i.e. s ==
        //                     K_TILES-2) AND has_next_e → fetch
        //                     NEXT expert's iter-0 tile.
        //   (C) Cross-expert iter-1: s+2 == K_TILES+1 (i.e. s ==
        //                     K_TILES-1) AND has_next_e → fetch
        //                     NEXT expert's iter-1 tile.
        //   Else: idle.
        //
        // Cases (B) and (C) together replace the single-deep
        // "stitch" from the original pipeline; they pre-load both
        // iter-0 AND iter-1 of the next expert during the current
        // expert's last two K-iters.  The matching pre-loop in the
        // helper does the same for the first expert.
        if (has_next_s) {
          // Case (A): intra-expert fetch of (s+2)-th tile.
          const uint32_t next_k_start = (s + 2u) * K_STEP;
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
        } else if (has_next_e) {
          // Cases (B)/(C): cross-expert stitch.
          //   At s == K_TILES-2: fetch next expert's iter-0.
          //   At s == K_TILES-1: fetch next expert's iter-1.
          const uint32_t next_e_iter = (s == K_TILES - 2u) ? 0u : 1u;
          const uint32_t next_e_k_start = next_e_iter * K_STEP;
          mbarrier_arrive_expect_tx(&shm->bar_w[next_slot],
                                    /*tx_bytes=*/UP_W_TX_BYTES_TOTAL);
    #pragma unroll
          for (uint32_t kk = 0; kk < K_SUBSTEPS; ++kk) {
            tma_load_up_wgmma_tile(
                up_weights_desc, /*expert_id=*/next_id,
                /*N=*/Dims::N,
                /*base_row_up=*/base_row_up,
                /*k_start=*/next_e_k_start + kk * K_STEP_WGMMA,
                /*dest_slot=*/&shm->w_wgmma[next_slot][kk * W_UP_M][0],
                /*bar=*/&shm->bar_w[next_slot]);
          }
        }
          // Else: last expert, last two iters — leave barriers idle.
  #endif
      }

      // ── Deferred SiLU + fp8 quant writeback for the PREVIOUS expert ──
      //
      // Runs on prefetch warps (8..11, 128 threads) at iters `s == 0`
      // AND `s == 1` of every expert AFTER the first.  The 8 tokens
      // of the previous expert's `wgmma_out` are split across two
      // K-loop iterations:
      //   iter 0: tokens [0..3]   (4 warps × 1 token each)
      //   iter 1: tokens [4..7]   (4 warps × 1 token each)
      //
      // Calc warps run their iter-0..iter-1 WGMMAs in parallel,
      // hiding the SiLU/SFU + GM-store latency behind compute.  See
      // the inline-vs-deferred A/B comment block at the top of the
      // expert loop.
      static_assert(K_TILES >= 2,
                    "Deferred up-proj writeback requires K_TILES >= 2.");
      // Phase-timing: bracket the deferred SiLU body so we can
      // measure its per-iter wall-clock and compare against the
      // calc-warp iter compute window.  Recorded on warp 8 lane 0
      // (= threadIdx.x == 256, the first prefetch lane).
      MONO_PHASE_TIMESTAMP_IF_TID(t_up_e1_pf_iter0_before_silu,
                                  e == expert_start + expert_stride && s == 0u,
                                  8u * 32u);
      MONO_PHASE_TIMESTAMP_IF_TID(t_up_e1_pf_iter1_before_silu,
                                  e == expert_start + expert_stride && s == 1u,
                                  8u * 32u);
      if (s == 0u && has_pending_writeback && is_prefetch_warp<Dims>()) {
  #ifndef MONO_PROFILE_SKIP_CALC_UP
        {
          // ── DEFER iter-0: tokens 0..3, one warp per token ────
          //
          // Calc warps of the PREVIOUS expert have already done the
          // per-lane silu(gate)*up*rw combine and stored val0/val1 to
          // `shm->partial_result.post_silu_scratch[row_in_tile][tok]`
          // (see the V2+DEFER calc-warp epilogue above and task 6.3).
          // Row layout in post_silu_scratch:
          //   WG0 rows: [0..31]   = warp_in_wg*8 + lane_in_warp/4
          //   WG1 rows: [64..95]  = 64 + warp_in_wg*8 + lane_in_warp/4
          // (rows 32..63 and 96..127 are unused under V2 — they only
          //  exist because post_silu_scratch aliases wgmma_out's
          //  128-row extent for SHM-union reuse.)
          //
          // This PF warp's lane `col_in_half` (0..31) reads:
          //   val1_l = post_silu_scratch[col_in_half][tok]
          //            → WG0 row col_in_half → output feat
          //              base_row_up + col_in_half
          //   val2_l = post_silu_scratch[col_in_half + 64][tok]
          //            → WG1 row col_in_half → output feat
          //              base_row_up + 32 + col_in_half
          // The 32 lanes of the warp cover all 64 output features of
          // the up-block for `tok`.  rw is already baked into the
          // stored values, so there's no runtime rw multiply here.
          constexpr std::uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;
          const unsigned pf_warp = warp - CoreDims::CALC_WARP_COUNT;  // 0..3
          const uint32_t tok = pf_warp + 0u;                          // 0..3
          const uint32_t col_in_half = lane;                          // 0..31

          // ── Section A: topk lookup via cache ──────────────────────
          // The store guard depends on whether this token routed to
          // the previous expert; if not, val0/val1 in post_silu_scratch
          // were zeroed by calc warps but we still must not emit a GM
          // store (that would clobber another expert's row).
          //
          // Read from `up_rank_for_tok_prev` — the snapshot taken at
          // the calc-warp epilogue of the previous expert.  The
          // current expert's K-loop top has already overwritten
          // `up_rank_for_tok` with the current expert's ranks, so
          // reading that would give the wrong answer.
          bool store_local = false;
          std::uint32_t dest_row_local = 0;
          if (tok < batch_size) {
            const uint8_t k = shm->up_rank_for_tok_prev[tok];
            if (k != 0xFFu) {
              store_local = true;
              dest_row_local = shm->sorted_slot[tok * top_k + k];
            }
          }

          // ── Section B: 2 post_silu_scratch SHM reads ──────────────
          // (Section C — SiLU compute — is empty: calc warps already
          //  did silu(gate)*up*rw before storing.)
          const float val1_l =
              shm->partial_result.post_silu_scratch[col_in_half][tok];
          const float val2_l =
              shm->partial_result.post_silu_scratch[col_in_half + 64][tok];

          const std::uint32_t out_col_1_l = base_row_up + col_in_half;
          const std::uint32_t out_col_2_l = base_row_up + 32 + col_in_half;
          const bool write1_l = store_local && (out_col_1_l < Dims::N);
          const bool write2_l = store_local && (out_col_2_l < Dims::N);
          float v1 = write1_l ? val1_l : 0.f;
          float v2 = write2_l ? val2_l : 0.f;

          // ── Section D: warp-reduce-max + fp8 quantize ─────────────
          float local_max_l = fmaxf(fabsf(v1), fabsf(v2));
          float block_max_l = warp_reduce_max_float(local_max_l);
          if (block_max_l < __FLT_MIN__) block_max_l = 1.0f;
          constexpr float FP8_MAX = 448.0f;
          constexpr float FP8_MAX_INV = 1.0f / 448.0f;
          const float block_scale_l = block_max_l * FP8_MAX_INV;
          const float inv_scale_l = FP8_MAX / block_max_l;
          const AQ_element q1_l = (AQ_element)(v1 * inv_scale_l);
          const AQ_element q2_l = (AQ_element)(v2 * inv_scale_l);

          // ── Section E: GM stores ──────────────────────────────────
          if (store_local && tok < batch_size) {
            if (write1_l) {
              spec->temp_fp8[dest_row_local * Dims::N + out_col_1_l] = q1_l;
            }
            if (write2_l) {
              spec->temp_fp8[dest_row_local * Dims::N + out_col_2_l] = q2_l;
            }
            if (lane == 0) {
              constexpr std::uint32_t SCALE_COLS =
                  MoEGemmSpec<Dims>::TEMP_ACT_SCALE_COLS;
              spec->temp_act_scale[dest_row_local * SCALE_COLS +
                                   effective_bid] = block_scale_l;
            }
          }
          (void)MAX_TOPK;  // unused under DEFER (cache replaces topk scan).
        }
  #endif
      } else if (s == 1u && has_pending_writeback && is_prefetch_warp<Dims>()) {
  #ifndef MONO_PROFILE_SKIP_CALC_UP
        {
          // ── DEFER iter-1: tokens 4..7, one warp per token ────
          //
          // Mirror of the iter-0 body above with phase-timing
          // instrumentation.  See the iter-0 comment for the row
          // layout discussion.  The MONO_PHASE_TIMESTAMP_IF_TID macros
          // are no-ops unless PHASE_TIMING is enabled and (e, warp,
          // lane) match the gate, so SASS is identical to a plain
          // V2+DEFER body when PHASE_TIMING is off.
          constexpr std::uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;
          const unsigned pf_warp = warp - CoreDims::CALC_WARP_COUNT;  // 0..3
          const uint32_t tok = pf_warp + 4u;                          // 4..7
          const uint32_t col_in_half = lane;                          // 0..31

          // ── Section A: topk cache lookup ──────────────────────────
          // Read from `up_rank_for_tok_prev` (snapshot of the
          // previous expert's cache).  See iter-0 PF body for the
          // rationale — the current expert's K-loop top has already
          // overwritten `up_rank_for_tok` with this expert's ranks.
          bool store_local = false;
          std::uint32_t dest_row_local = 0;
          if (tok < batch_size) {
            const uint8_t k = shm->up_rank_for_tok_prev[tok];
            if (k != 0xFFu) {
              store_local = true;
              dest_row_local = shm->sorted_slot[tok * top_k + k];
            }
          }
          MONO_PHASE_TIMESTAMP_IF_TID(t_up_e1_pf_iter1_after_topk_lookup,
                                      e == expert_start + expert_stride,
                                      8u * 32u);

          // ── Section B: 2 post_silu_scratch SHM reads ──────────────
          // (replaces V1's 4 wgmma_out reads — calc warps have
          //  already combined silu(gate)*up*rw before storing.)
          const float val1_l =
              shm->partial_result.post_silu_scratch[col_in_half][tok];
          const float val2_l =
              shm->partial_result.post_silu_scratch[col_in_half + 64][tok];
          MONO_PHASE_TIMESTAMP_IF_TID(t_up_e1_pf_iter1_after_wgmma_read,
                                      e == expert_start + expert_stride,
                                      8u * 32u);

          // ── Section C: SiLU compute (empty under V2+DEFER) ────────
          // Under V1+DEFER this section did rw * up * gate * sigmoid
          // (~0.37 µs per iter — the dominant cost).  Under V2+DEFER
          // calc warps have already done it in registers, so the PF
          // body just routes the cached values to quant.  The
          // timestamp is kept (as a near-zero delta) so the section
          // table layout matches V1+DEFER for direct comparison.
          const std::uint32_t out_col_1_l = base_row_up + col_in_half;
          const std::uint32_t out_col_2_l = base_row_up + 32 + col_in_half;
          const bool write1_l = store_local && (out_col_1_l < Dims::N);
          const bool write2_l = store_local && (out_col_2_l < Dims::N);
          float v1 = write1_l ? val1_l : 0.f;
          float v2 = write2_l ? val2_l : 0.f;
          MONO_PHASE_TIMESTAMP_IF_TID(t_up_e1_pf_iter1_after_silu_compute,
                                      e == expert_start + expert_stride,
                                      8u * 32u);

          // ── Section D: warp-reduce-max + fp8 quantize ─────────────
          float local_max_l = fmaxf(fabsf(v1), fabsf(v2));
          float block_max_l = warp_reduce_max_float(local_max_l);
          if (block_max_l < __FLT_MIN__) block_max_l = 1.0f;
          constexpr float FP8_MAX = 448.0f;
          constexpr float FP8_MAX_INV = 1.0f / 448.0f;
          const float block_scale_l = block_max_l * FP8_MAX_INV;
          const float inv_scale_l = FP8_MAX / block_max_l;
          const AQ_element q1_l = (AQ_element)(v1 * inv_scale_l);
          const AQ_element q2_l = (AQ_element)(v2 * inv_scale_l);
          MONO_PHASE_TIMESTAMP_IF_TID(t_up_e1_pf_iter1_after_warp_reduce,
                                      e == expert_start + expert_stride,
                                      8u * 32u);

          // ── Section E: GM stores ──────────────────────────────────
          if (store_local && tok < batch_size) {
            if (write1_l) {
              spec->temp_fp8[dest_row_local * Dims::N + out_col_1_l] = q1_l;
            }
            if (write2_l) {
              spec->temp_fp8[dest_row_local * Dims::N + out_col_2_l] = q2_l;
            }
            if (lane == 0) {
              constexpr std::uint32_t SCALE_COLS =
                  MoEGemmSpec<Dims>::TEMP_ACT_SCALE_COLS;  // = Dims::N / 64
              spec->temp_act_scale[dest_row_local * SCALE_COLS +
                                   effective_bid] = block_scale_l;
            }
          }
          (void)MAX_TOPK;
        }
  #endif
      }
      MONO_PHASE_TIMESTAMP_IF_TID(t_up_e1_pf_iter0_after_silu,
                                  e == expert_start + expert_stride && s == 0u,
                                  8u * 32u);
      MONO_PHASE_TIMESTAMP_IF_TID(t_up_e1_pf_iter1_after_silu,
                                  e == expert_start + expert_stride && s == 1u,
                                  8u * 32u);

      // ── Inter-iteration sync ──
      //
      // The launcher arms `bar_w[next_slot]` at iter s.  Two
      // iterations later (iter s+2), the launcher arms the SAME
      // `bar_w[next_slot]` again (because the slot index repeats every
      // 2 iters in the ping-pong).  For the second arm not to
      // double-arm an mbarrier whose current phase is still pending,
      // the calc-warp consume of that slot at iter s+1 must complete
      // BEFORE iter s+2's launcher arm runs.
      //
      // The legacy QUANT/COMPUTE __syncthreads() served this role.
      // Without it, the launcher (a single warp-8-lane-0 thread that
      // never waits) can race ahead through all K_TILES launcher
      // arms before any calc warp consumes its bar_w wait.  This
      // single end-of-iter sync re-establishes ordering: every iter,
      // all warps (including launcher and calc) rendezvous at the
      // sync, so the launcher cannot arm the next-slot mbarrier until
      // the calc warps' wait on the same slot has completed.
      //
      // R3.12 allows one __syncthreads() per outer K-step iteration;
      // this sync publishes the launcher's `mbarrier_arrive_expect_tx`
      // (a producer-side state mutation on bar_w) to the calc warps
      // that will issue `try_wait_parity` against it next iter.
      __syncthreads();
    }  // end K-loop

    MONO_PHASE_TIMESTAMP_IF(t_up_after_expert0_kloop, e == expert_start);

    // ── End-of-expert: write final_d to partial_result.wgmma_out[128][8] ──
    // Canonical WGMMA D-matrix layout per thread (m64n8k32):
    //   d[0]: row = warp_in_wg*16 + lane/4 + 0,  col = (lane%4)*2 + 0
    //   d[1]: row = warp_in_wg*16 + lane/4 + 0,  col = (lane%4)*2 + 1
    //   d[2]: row = warp_in_wg*16 + lane/4 + 8,  col = (lane%4)*2 + 0
    //   d[3]: row = warp_in_wg*16 + lane/4 + 8,  col = (lane%4)*2 + 1
    // For WG1, rows shift by +64 in the full 128-row output tile.
    //
    // ── Profile-only escape: MONO_PROFILE_SKIP_UP_EPILOGUE ────────────
    // Wrapping (a) the wgmma_out SHM store, (b) the inter-warp sync
    // that publishes wgmma_out, and (c) the SiLU+fp8-quant writeback
    // helper.  When this flag is defined, the WGMMA accumulator's
    // final_d{0..3} are dropped on the floor (no SHM publish, no
    // GM writeback) — accuracy WILL fail, by design.  The flag
    // exists so an NCU run with these three operations elided can
    // tell us whether the cross-expert "yellow + blue both idle"
    // visible in PM-sampling profiles is attributable to the
    // epilogue or to something else (cross-expert TMA wait, scale
    // load, etc.).
    //
    // The trailing `__syncthreads()` at the bottom of the per-expert
    // loop body is left in place even under this flag because it
    // also serves the cross-expert launcher/calc ordering for the
    // bar_w stitch arm — eliding it would change the visible
    // pipeline structure in NCU and confound the comparison.
  #ifndef MONO_PROFILE_SKIP_UP_EPILOGUE
    {
      // ── V2 Pair_Layout + DEFER per-expert epilogue (calc-warp side) ──
      //
      // Under Pair_Layout, each WGMMA lane holds after the K-loop:
      //   final_d0 = gate(r), tok_even = (lane%4)*2
      //   final_d1 = gate(r), tok_odd  = (lane%4)*2 + 1
      //   final_d2 = up(r),   tok_even = (lane%4)*2
      //   final_d3 = up(r),   tok_odd  = (lane%4)*2 + 1
      // where r = warp_in_wg*8 + lane/4 within the WG's 32-row half.
      //
      // Calc warps perform ONLY the per-lane register combine
      // silu(gate)*up*rw and store val0/val1 to
      // `shm->partial_result.post_silu_scratch[row_in_tile][tok]`.
      // The cross-warp reduce-max, fp8 quantize, and GM stores are
      // deferred to PF warps in the next expert's iters 0/1 (task
      // 6.4 wires up the PF reader).  See design.md "Component 5"
      // and tasks 6.3 / 6.4.
      //
      // The existing inter-expert `__syncthreads()` at the bottom of
      // the expert loop body publishes post_silu_scratch to PF warps;
      // no extra barrier inside this branch.
      if (is_calc) {
    #ifndef MONO_PROFILE_SKIP_CALC_UP
        // Token indices owned by this lane (N-dimension cols = tokens).
        const std::uint32_t tok_even = (lane % 4) * 2;
        const std::uint32_t tok_odd = tok_even + 1;

        // ── Look up routing weight from the precomputed cache ───────
        // Single independent SHM read per token.  `up_rw_for_tok` is
        // exactly 0.0f for tokens that don't route to this expert (set
        // at the populate site), so the rw value doubles as the route
        // predicate — no separate `up_rank_for_tok` read needed here
        // (the V2+DEFER calc epilogue doesn't need the rank `k`; the
        // PF body that needs `dest_row` reads the rank snapshot).
        float rw_even = 0.f, rw_odd = 0.f;
        bool store_even = false, store_odd = false;

        if (tok_even < batch_size) {
          rw_even = shm->up_rw_for_tok[tok_even];
          store_even = (rw_even != 0.f);
        }
        if (tok_odd < batch_size) {
          rw_odd = shm->up_rw_for_tok[tok_odd];
          store_odd = (rw_odd != 0.f);
        }
        MONO_PHASE_TIMESTAMP_IF(t_up_e0_defer_after_rwlookup,
                                e == expert_start);

        // ── Per-lane combine: rw * up * gate / (1+exp(-gate)) ────────
        // silu(gate)*up*rw.  Uses `__fdividef` (fast approximate
        // reciprocal, ~2 SFU instrs) instead of the IEEE `/` (which
        // expands to MUFU.RCP + Newton-Raphson refinement, ~8 instrs).
        // The output is fp8-quantized downstream, so the extra divide
        // precision is wasted — `__fdividef` is bit-adequate and
        // shortens the SFU dependency chain (the dominant cost of
        // this section).  Multiplication order otherwise matches V1.
        float val0 =
            __fdividef(rw_even * final_d2 * final_d0, 1.0f + __expf(-final_d0));
        float val1 =
            __fdividef(rw_odd * final_d3 * final_d1, 1.0f + __expf(-final_d1));

        // Zero out values for unrouted tokens (out-of-bounds in
        // batch_size or topk-miss).  Token out-of-bounds in N is
        // handled by the PF-side store guard in task 6.4.
        if (!store_even) val0 = 0.f;
        if (!store_odd) val1 = 0.f;
        MONO_PHASE_TIMESTAMP_IF(t_up_e0_defer_after_combine, e == expert_start);

        // ── Store to post_silu_scratch ───────────────────────────────
        // Row index = global row in the 128-row up-block tile.
        const uint32_t row_in_tile =
            (is_wg1 ? 64u : 0u) + warp_in_wg * 8 + lane / 4;
        shm->partial_result.post_silu_scratch[row_in_tile][tok_even] = val0;
        shm->partial_result.post_silu_scratch[row_in_tile][tok_odd] = val1;
        MONO_PHASE_TIMESTAMP_IF(t_up_e0_defer_after_store, e == expert_start);

        // Snapshot the current expert's `up_rank_for_tok` cache to
        // `up_rank_for_tok_prev` for the PF body of the NEXT expert
        // to consume.  The K-loop top of the next expert overwrites
        // `up_rank_for_tok` with that expert's ranks, so without
        // this snapshot the PF body would read the wrong ranks and
        // either skip valid writes (store_local=false) or write
        // them to the wrong dest_row.  8 calc threads (one per
        // token), one byte each.  The existing inter-expert
        // `__syncthreads()` at the bottom of the expert loop body
        // publishes the snapshot to PF warps.
        if (warp == 0 && lane < Dims::BS) {
          shm->up_rank_for_tok_prev[lane] = shm->up_rank_for_tok[lane];
        }
        MONO_PHASE_TIMESTAMP_IF(t_up_e0_defer_after_snapshot,
                                e == expert_start);
    #endif
      }

      // Mark this expert's writeback as pending; the PF body in iters
      // 0/1 of the next expert's K-loop (task 6.4) will read
      // post_silu_scratch, warp-reduce-max, fp8 quantize, and write
      // the GM stores.  These two flags already exist in function
      // scope (declared above the expert loop for V1+DEFER) and are
      // reused unchanged.
      prev_id_for_writeback = id;
      has_pending_writeback = true;

      // Phase-timing parity: the V1 / V2-inline epilogues emit
      // `t_up_after_expert0_wgmma_out` between the wgmma_out store and
      // the SiLU writeback.  The V2+DEFER calc epilogue has no such
      // mid-point, but the profiler's legacy "wgmma_out store + sync"
      // / "SiLU + quant + writeback" rows subtract this field — if it
      // stays 0 (unwritten) those rows show ±garbage.  Emit it here so
      // the two legacy rows degrade gracefully: row 1 = calc-epilogue
      // cost (combine+store+snapshot), row 2 = trailing-sync cost.
      MONO_PHASE_TIMESTAMP_IF(t_up_after_expert0_wgmma_out, e == expert_start);
    }
  #else
    // Profile-only: kill-use of the WGMMA accumulators so the
    // compiler does not optimize the K-loop into a no-op when the
    // epilogue is elided.  `volatile` on a register-resident value
    // forces nvcc to keep the WGMMA dependency chain alive, which is
    // what we want for an apples-to-apples NCU comparison of the
    // K-loop pipeline.
    {
      volatile float sink = final_d0 + final_d1 + final_d2 + final_d3;
      (void)sink;
    }
  #endif  // MONO_PROFILE_SKIP_UP_EPILOGUE

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

    // Advance the scale ping-pong.  The next expert consumes the
    // buffer we just prefetched into (cur_scale_slot ^ 1), and will
    // prefetch e+2 into the buffer this expert just finished reading.
    cur_scale_slot ^= 1u;
  }  // end expert loop

  // ── Post-loop drain for the LAST expert (deferred path) ──
  //
  // The deferred path inside the K-loop processes expert e's
  // wgmma_out at iter 0/1 of expert e+1.  The LAST expert visited
  // (expert_count - expert_stride for stride==1) has nothing to
  // defer to, so we run its writeback inline now, on calc warps,
  // with the original (warp → tok) mapping (warps 0..7, 1 token
  // each).  All warps are free at this point — the K-loop is done.
  if (has_pending_writeback && is_calc) {
  #ifndef MONO_PROFILE_SKIP_CALC_UP
    const uint32_t tok = warp;          // 0..7
    const uint32_t col_in_half = lane;  // 0..31
    {
      // ── DEFER post-loop drain: last expert, on calc warps ──
      //
      // Under V2+DEFER, calc warps inside the K-loop wrote
      // silu(gate)*up*rw to `shm->partial_result.post_silu_scratch`
      // for the previous expert, and PF warps drained that scratch in
      // the next expert's iter-0/iter-1 (see the in-K-loop V2 PF
      // bodies around the iter-0 / iter-1 sites).  The LAST expert
      // has no successor expert to defer to, so we drain it here
      // post-K-loop on calc warps with the same row/col mapping the
      // PF bodies use.  Reading from `wgmma_out` here would be
      // incorrect under V2 — calc warps stored to post_silu_scratch,
      // not wgmma_out.
      constexpr std::uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;

      // ── Section A: topk lookup via cache ─────────────────────────
      // Read from `up_rank_for_tok_prev` — the snapshot taken at the
      // calc-warp epilogue of the LAST expert.  See the in-K-loop
      // V2+DEFER PF body comment for the rationale.  Note: the
      // post-loop drain runs AFTER the expert loop exits, so even
      // though no "next expert" exists to overwrite the cache, the
      // last calc-warp epilogue is what populated _prev — we read
      // _prev here for consistency with the in-K-loop bodies.
      bool store_local = false;
      std::uint32_t dest_row_local = 0;
      if (tok < batch_size) {
        const uint8_t k = shm->up_rank_for_tok_prev[tok];
        if (k != 0xFFu) {
          store_local = true;
          dest_row_local = shm->sorted_slot[tok * top_k + k];
        }
      }

      // ── Section B: 2 post_silu_scratch SHM reads ─────────────────
      // (rw * silu(gate) * up was already baked in by calc warps.)
      const float val1_l =
          shm->partial_result.post_silu_scratch[col_in_half][tok];
      const float val2_l =
          shm->partial_result.post_silu_scratch[col_in_half + 64][tok];

      const std::uint32_t out_col_1_l = base_row_up + col_in_half;
      const std::uint32_t out_col_2_l = base_row_up + 32 + col_in_half;
      const bool write1_l = store_local && (out_col_1_l < Dims::N);
      const bool write2_l = store_local && (out_col_2_l < Dims::N);
      float v1 = write1_l ? val1_l : 0.f;
      float v2 = write2_l ? val2_l : 0.f;

      // ── Section D: warp-reduce-max + fp8 quantize ────────────────
      float local_max_l = fmaxf(fabsf(v1), fabsf(v2));
      float block_max_l = warp_reduce_max_float(local_max_l);
      if (block_max_l < __FLT_MIN__) block_max_l = 1.0f;
      constexpr float FP8_MAX = 448.0f;
      constexpr float FP8_MAX_INV = 1.0f / 448.0f;
      const float block_scale_l = block_max_l * FP8_MAX_INV;
      const float inv_scale_l = FP8_MAX / block_max_l;
      const AQ_element q1_l = (AQ_element)(v1 * inv_scale_l);
      const AQ_element q2_l = (AQ_element)(v2 * inv_scale_l);

      // ── Section E: GM stores ─────────────────────────────────────
      if (store_local && tok < batch_size) {
        if (write1_l) {
          spec->temp_fp8[dest_row_local * Dims::N + out_col_1_l] = q1_l;
        }
        if (write2_l) {
          spec->temp_fp8[dest_row_local * Dims::N + out_col_2_l] = q2_l;
        }
        if (lane == 0) {
          constexpr std::uint32_t SCALE_COLS =
              MoEGemmSpec<Dims>::TEMP_ACT_SCALE_COLS;
          spec->temp_act_scale[dest_row_local * SCALE_COLS + effective_bid] =
              block_scale_l;
        }
      }
      (void)MAX_TOPK;  // unused under DEFER (cache replaces topk scan).
    }
  #endif
  }
}

}  // namespace moe_monokernel

#endif