
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
  #include "moe_tma.h"

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
 * @brief v1 streaming-pipeline WGMMA down-projection weight tile loader.
 *
 * Loads one 128-row × 128-K fp8 weight tile (= one K-step) of the
 * down-projection weight matrix into a SHM slot, laid out in canonical
 * WGMMA Major::K core-matrix format so the tile can be consumed without
 * swizzle.
 *
 * Unlike the up-projection (which interleaves gate/up halves in its
 * 128-row tile), the down-projection maps rows [0..127] of the tile
 * DIRECTLY to output cols `[base_col, base_col + 128)`.  There is no
 * gate/up split because the down-projection has no SiLU gate.
 *
 * GM layout assumed: `expert_weights_down[E][HIDDEN_STATES][N]` fp8,
 * where:
 *   - axis 0 = expert id,
 *   - axis 1 = down-proj output col (= the WGMMA A-operand M row),
 *   - axis 2 = K (down-proj inner dim, size `Dims::N`).
 *
 * The M row `m` in `[0, 128)` of the SHM tile corresponds to global
 * weight row `base_col + m`.  The K index `k` in `[0, 128)` corresponds
 * to global K column `k_start + k`.
 *
 * SHM layout written (canonical Major::K, zero swizzle):
 *   byte_off(m, k) = m_outer * 1024 + k_outer * 128
 *                  + m_inner *   16 + k_inner
 * with `m_outer = m / 8`, `m_inner = m % 8`,
 *      `k_outer = k / 16`, `k_inner = k % 16`.
 *
 * Total tile size: 16 (m_outer) × 8 (k_outer) = 128 core matrices
 *                  × 128 B = 16 384 B = 16 KB — exactly one slot of
 *                  `w_down_wgmma[2][128][128]`.
 *
 * Thread distribution: 128 prefetch threads, each thread owns one
 * destination row (`dst_row = pflat = pw * 32 + thread`) and issues
 * 8 × 16-byte cp.async transfers striding over `k_outer in [0, 8)`.
 *
 * @tparam Dims        MoE dims.
 * @tparam DestRows    Must be 128.
 * @tparam DestCols    Must be K_STEP_WGMMA (128).
 * @param  source      GM pointer [E, HIDDEN_STATES, N] fp8 down weights.
 * @param  id          Expert index.
 * @param  base_col    First output col of this block's M stripe (multiple
 *                     of 128).
 * @param  k_start     Starting K column of this tile (multiple of 128).
 * @param  dest        SHM tile [128][128] fp8.
 * @param  pipe        Async-copy pipeline.
 *
 * @note Prefetch-warp only.  Must be followed by `pipe.producer_commit()`
 *       and a consumer wait before WGMMA reads `dest`.
 */
template <typename Dims, std::size_t DestRows, std::size_t DestCols>
__device__ inline void moe_load_down_wgmma_weight_tile(
    const W_element* __restrict__ source, std::uint32_t id,
    std::uint32_t base_col, std::uint32_t k_start,
    W_element (&dest)[DestRows][DestCols],
    cuda::pipeline<cuda::thread_scope_thread>& pipe) {
  using CoreDims = MoECoreDims<Dims>;
  static_assert(DestRows == 128,
                "down-proj streaming WGMMA weight tile must be 128 rows");
  static_assert(DestCols == 128,
                "down-proj streaming WGMMA weight tile must be 128 K-values");

  constexpr unsigned CHUNK_B = 16;
  constexpr unsigned CHUNK_ELEMS = CHUNK_B / sizeof(W_element);  // 16
  constexpr unsigned K_CHUNKS_PER_ROW = DestCols / CHUNK_ELEMS;  // 8

  // Only prefetch warps issue the cp.async. 128 threads = 4 warps × 32.
  const unsigned thread = get_thread<Dims>();
  const unsigned pw = get_prefetch_warp<Dims>();  // 0..3
  const unsigned pflat = pw * 32 + thread;        // 0..127 — one row each

  // GM base: axis-0 stride is `HIDDEN_STATES * N`, axis-1 stride is `N`
  // (K is innermost).  Cast to OpaqueElement (uint32_t) for alignment-
  // guaranteed 16-byte copies, same trick as the up-proj loader.
  const OpaqueElement* weights =
      (const OpaqueElement*)(source + id * Dims::HIDDEN_STATES * Dims::N);

  pipe.producer_acquire();
  if (pflat < 128) {
    const unsigned dst_row = pflat;
    const unsigned m_outer = dst_row / 8;  // 0..15
    const unsigned m_inner = dst_row % 8;  // 0..7

    // Down-proj: rows [0..127] of the tile map directly to output cols
    // [base_col, base_col + 128).  No gate/up split — this is the key
    // difference from the up-projection.
    const unsigned global_row = base_col + dst_row;

    OpaqueElement* dest_oe = (OpaqueElement*)&dest[0][0];

  #pragma unroll
    for (unsigned k_outer = 0; k_outer < K_CHUNKS_PER_ROW; ++k_outer) {
      const unsigned k_col = k_outer * CHUNK_ELEMS;
      // Canonical byte offset into the SHM tile:
      //   m_outer * 1024 + k_outer * 128 + m_inner * 16
      const unsigned byte_off =
          m_outer * 1024 + k_outer * 128 + m_inner * CHUNK_B;
      const unsigned dest_chunk_idx = byte_off / sizeof(OpaqueElement);
      // GM byte offset: global_row * N + (k_start + k_col), with K stride
      // of `N` (not `HIDDEN_STATES`, since down-proj weight axis 2 is K).
      copy128(dest_oe[dest_chunk_idx],
              weights[(global_row * Dims::N + k_start + k_col) /
                      sizeof(OpaqueElement)],
              pipe);
    }
  }
  pipe.producer_commit();
}

/**
 * @brief Hoisted per-expert activation scale loader.
 *
 * Loads ALL `routed_count × 8` per-(token, K-block) activation scales
 * for one expert into SHM in a single coalesced cp.async burst, before
 * the K-loop starts.  Replaces the previous per-K-step loader that put
 * a cp.async wait on the critical path of every K-step.
 *
 * Layout: `dest_scale[T_TILE=8][SCALE_COLS=8]` indexed by
 * `[tok][k_block_idx]` where `k_block_idx = s * 2 + half`.
 *
 * @note Prefetch-warp only.  Must be followed by `pipe.producer_commit()`
 *       and a consumer wait before the K-loop reads `dest_scale`.
 */
template <typename Dims, std::size_t ScaleTok, std::size_t ScaleCols>
__device__ inline void moe_load_down_wgmma_activation_scales_for_expert(
    const MoEGemmSpec<Dims>* __restrict__ spec,
    const MoE_SHM<Dims>* __restrict__ shmem, std::uint32_t id,
    S_element (&dest_scale)[ScaleTok][ScaleCols],
    cuda::pipeline<cuda::thread_scope_thread>& pipe) {
  using CoreDims = MoECoreDims<Dims>;
  static_assert(ScaleTok == CoreDims::T_TILE,
                "down-proj activation scale tile must have T_TILE=8 tokens");
  static_assert(ScaleCols == Dims::N / 64u,
                "down-proj activation scale tile must have N/64 cols per "
                "token (one scale per 64-col activation block)");
  constexpr unsigned ACT_BLOCK = 64;
  constexpr unsigned SCALE_COLS = Dims::N / ACT_BLOCK;  // = 8 for N=512

  const unsigned thread = get_thread<Dims>();
  const unsigned pw = get_prefetch_warp<Dims>();  // 0..3
  const unsigned pflat = pw * 32 + thread;        // 0..127

  pipe.producer_acquire();

  // 64 fp32 scales total (8 tokens × 8 K-blocks) — one per (rank, k_block).
  // 64 prefetch threads (pflat 0..63) each load exactly one scale.
  //
  // Rank-indexing invariant (R11.3, R12.9):
  //   `a_down_scale[rank][k_block]` must match the scale for the token
  //   whose fp8 payload sits at `a_down_wgmma[slot][rank][kc][ki]`.  The
  //   fp8 payload is bulk-loaded from GM rows
  //   `[expert_slot_start, expert_slot_start + routed_count)` of
  //   `temp_fp8`, and `temp_act_scale` rows are written by the up-proj
  //   epilogue at the SAME row indices.  So reading
  //   `temp_act_scale[expert_start + rank][k_block]` keeps fp8 ↔ scale
  //   alignment.
  if (pflat < ScaleTok * ScaleCols) {
    const unsigned slot_row = pflat / ScaleCols;  // 0..7 (tok rank)
    const unsigned k_block = pflat % ScaleCols;   // 0..7

    const auto* tma_shm = &shmem->u.tiny_wgmma_tma;
    const uint32_t routed_count =
        static_cast<uint32_t>(tma_shm->expert_routed_count[id]);
    const uint32_t expert_start =
        static_cast<uint32_t>(tma_shm->expert_slot_start[id]);

    if (slot_row < routed_count) {
      const uint32_t source_row = expert_start + slot_row;
      dest_scale[slot_row][k_block] =
          spec->temp_act_scale[source_row * SCALE_COLS + k_block];
    } else {
      // Unused SHM rank — fp8 payload is zero-filled by
      // `zero_fill_unused_down_act_slots`, so the scale is irrelevant;
      // write 0 to avoid any chance of NaN propagation through
      // scale-apply (defensive — those ranks are never read by the
      // accumulate loop).
      dest_scale[slot_row][k_block] = 0.f;
    }
  }

  pipe.producer_commit();
}

/**
 * @brief Streaming-pipeline WGMMA down-projection per-token activation
 *        scale loader.
 *
 * Loads the 2 per-token per-64-col fp32 activation scales that apply to
 * one K-step of the down-projection, for all 8 SHM slot rows.  The fp8
 * activation payload itself is loaded separately by
 * `tma_load_down_wgmma_activation_bulk` (TMA + SWZ128) — this helper is
 * scales-only.
 *
 * The down-proj B operand is the fp8 quantized SiLU output produced by
 * the up-proj epilogue.  Each (token, expert) pair that routes through
 * this expert has a dedicated row in
 * `spec->temp_fp8[TEMP_ROWS_TMA][N]` at index `sorted_slot[pair] =
 * expert_slot_start[id] + rank`.  Scales for those same rows live in
 * `spec->temp_act_scale[row][col]` with `col = s * 2 + half`.
 *
 * Thread distribution (128 prefetch threads):
 *   threads 0..15  : each load one scale (8 tok × 2 halves) from GM.
 *   threads 16..127: idle.
 *
 * @tparam Dims       MoE dims.
 * @param  spec       Global scratchpad (reads `spec->temp_act_scale`).
 * @param  shmem      Shared-memory struct (for `expert_routed_count` /
 *                    `expert_slot_start`).
 * @param  id         Expert id currently being processed.
 * @param  top_k      Number of experts per token (unused; retained for
 *                    signature parity).
 * @param  batch_size Number of real tokens (unused; retained for
 *                    signature parity).
 * @param  k_start    Starting K column of this tile (multiple of 128;
 *                    unused; retained for signature parity).
 * @param  s          K-step index within the expert's K-loop
 *                    (in [0, K_TILES_DOWN=4) ).  Selects which pair
 *                    of 64-col scales to load for each token.
 * @param  dest_act   SHM fp8 activation tile — unused here (populated
 *                    by the TMA path); retained for signature parity
 *                    with earlier cp.async loaders.
 * @param  dest_scale SHM per-token scales [8 tok][2 halves] fp32.
 * @param  pipe       Async-copy pipeline.
 *
 * @note Prefetch-warp only.  Must be followed by `pipe.producer_commit()`
 *       and a consumer wait before WGMMA reads the activation slab (the
 *       WGMMA barrier is the TMA's `bar_a[slot]`, not this pipe).
 *
 * @note The 2 scale values per token cover K[0..63] and K[64..127] of
 *       the current K-step.  Up-proj epilogue writes one scale per
 *       (virtual_row, up_block_idx) with up_block_idx == 64-col index
 *       along N; so `spec->temp_act_scale[vr * (N/64) + (s*2 + half)]`
 *       addresses the scale for K[half*64 .. half*64+63] of this step.
 */
template <typename Dims, std::size_t DestTok, std::size_t DestChunks,
          std::size_t DestKInner, std::size_t ScaleTok, std::size_t ScaleHalf>
__device__ inline void moe_load_down_wgmma_activation_tile(
    const MoEGemmSpec<Dims>* __restrict__ spec,
    const MoE_SHM<Dims>* __restrict__ shmem, std::uint32_t id,
    std::uint32_t top_k, std::uint32_t batch_size, std::uint32_t k_start,
    std::uint32_t s, AQ_element (&dest_act)[DestTok][DestChunks][DestKInner],
    S_element (&dest_scale)[ScaleTok][ScaleHalf],
    cuda::pipeline<cuda::thread_scope_thread>& pipe) {
  using CoreDims = MoECoreDims<Dims>;
  static_assert(DestTok == CoreDims::T_TILE,
                "down-proj activation tile must have T_TILE=8 tokens");
  static_assert(DestChunks == 8,
                "down-proj activation tile must have 8 K-chunks per token");
  static_assert(DestKInner == 16,
                "down-proj activation tile must have 16 fp8 K-bytes per chunk");
  static_assert(ScaleTok == CoreDims::T_TILE,
                "down-proj activation scale tile must have T_TILE=8 tokens");
  static_assert(ScaleHalf == 2,
                "down-proj activation scale tile must have 2 halves per token");
  static_assert(sizeof(AQ_element) == 1, "AQ_element (fp8) must be 1 byte");

  constexpr uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;
  constexpr unsigned ACT_BLOCK = 64;  // per-64-col up-block size
  constexpr unsigned SCALE_COLS = Dims::N / ACT_BLOCK;  // N/64
  (void)MAX_TOPK;
  (void)top_k;
  (void)batch_size;
  (void)k_start;
  (void)dest_act;

  const unsigned thread = get_thread<Dims>();
  const unsigned pw = get_prefetch_warp<Dims>();  // 0..3
  const unsigned pflat = pw * 32 + thread;        // 0..127

  pipe.producer_acquire();

  // ── fp8 payload: loaded by the Phase-4 TMA launcher ───────────────────
  //
  // The fp8 payload is loaded by `tma_load_down_wgmma_activation_bulk`
  // (plus the cooperative zero-fill for unused slots); this helper only
  // handles the scale loader below.

  // ── Per-token activation scales: 16 fp32 values total (8 tok × 2) ─────
  //
  // Destination index is `rank` — the intra-expert SHM slot position
  // produced by the Phase-3→4 expert-sorted reorganization (spec R11).
  // Source row in `spec->temp_act_scale` is
  // `expert_slot_start[id] + rank`.
  //
  // Rank-indexing invariant (R11.3, R12.9):
  //   `a_down_scale[slot][rank][half]` must match the scale for the
  //   token whose fp8 payload sits at
  //   `a_down_wgmma[slot][rank][kc][ki]`.  That fp8 payload is loaded
  //   by the bulk-per-expert TMA (R2) from GM rows
  //   `[expert_slot_start[id], expert_slot_start[id] + routed_count)`
  //   of `spec->temp_fp8`.  Those same rows were written by the
  //   Phase-3 up-proj epilogue at
  //   `dest_row = sorted_slot[pair] = expert_slot_start[id] + rank`.
  //   The epilogue also writes `temp_act_scale[dest_row][...]` at the
  //   same row (R11.4), so reading scales from row
  //   `expert_slot_start[id] + rank` guarantees byte-level alignment
  //   between the fp8 payload and its per-token scale.
  //
  // Threads 0..15 each load one scale (8 tok × 2 halves).
  if (pflat < DestTok * ScaleHalf) {
    const unsigned slot_row = pflat / ScaleHalf;  // 0..7
    const unsigned half = pflat % ScaleHalf;  // 0..1 (K[0..63] or K[64..127])

    // `tiny_wgmma_tma` is the TMA union variant (byte-identical to
    // `tiny_wgmma` plus 32 B of mbarriers at the tail);
    // `expert_routed_count` and `expert_slot_start` were populated by
    // the Phase-1/2 routing-prep extension.
    const auto* tma_shm = &shmem->u.tiny_wgmma_tma;
    const uint32_t routed_count =
        static_cast<uint32_t>(tma_shm->expert_routed_count[id]);
    const uint32_t expert_start =
        static_cast<uint32_t>(tma_shm->expert_slot_start[id]);

    if (slot_row < routed_count) {
      // Source row in temp_act_scale matches the reorganized layout
      // (see invariant above).
      const uint32_t source_row = expert_start + slot_row;
      const uint32_t scale_col = s * 2 + half;
      dest_scale[slot_row][half] =
          spec->temp_act_scale[source_row * SCALE_COLS + scale_col];
    } else {
      // Unused SHM slot in the rank-indexed layout.  Its fp8 payload
      // is zero-filled by `zero_fill_unused_down_act_slots`, so the
      // scale value is irrelevant; write 0 to avoid NaN propagation
      // through the scale-apply multiply.
      dest_scale[slot_row][half] = 0.f;
    }
  }

  pipe.producer_commit();
}

/**
 * @brief v1 streaming-pipeline WGMMA down-projection weight-scale tile loader.
 *
 * Loads one K-step's worth of the down-projection's fp32 weight scales
 * into a SHM slot — one scale per (warpgroup, col-block) pair.  There
 * are 2 warpgroups (WG0 covers weight rows `[base_col..base_col+63]`,
 * WG1 covers rows `[base_col+64..base_col+127]`) and `W_DOWN_SCALE_COLS`
 * col-blocks along the K = Dims::N dimension.  (For block-wise quant,
 * `W_DOWN_SCALE_COLS == Dims::DOWN_SCALE_COLS == N/128`; for per-channel,
 * it is a 1-wide placeholder and these bytes are never consumed.)
 *
 * The down-proj weight matrix is `[NUM_EXPERTS, HIDDEN_STATES, N]`, so
 * the 128 output cols that this down-block owns map to 128 weight rows
 * `[base_col, base_col + 128)`.  Because `base_col` is a multiple of
 * 128, WG0's 64 weight rows and WG1's 64 weight rows always fall within
 * the SAME 128-row scale-block (so `row_block_wg0 == row_block_wg1`),
 * but we still compute and store both independently for generality.
 *
 * Scale tensor layout (block-wise):
 *   `expert_scales_down[NUM_EXPERTS][DOWN_SCALE_ROWS][DOWN_SCALE_COLS]` fp32,
 *   where `DOWN_SCALE_ROWS = HIDDEN_STATES / 128`
 *   and   `DOWN_SCALE_COLS = N             / 128`.
 *
 * SHM layout written:
 *   `dest[wg][col_block]` for `wg in [0, 2)`, `col_block in [0,
 * W_DOWN_SCALE_COLS)`.
 *
 * Thread distribution: the first prefetch warp loads all
 * `2 * W_DOWN_SCALE_COLS` scalars synchronously.  For Qwen3.5-35B
 * (W_DOWN_SCALE_COLS=4) this is just 8 fp32 values — well under a warp.
 * No async copy is used; the synchronous SHM stores complete as soon
 * as the warp retires them.
 *
 * @tparam Dims             MoE dims.
 * @tparam DestRows         Must be 2 (WG0, WG1).
 * @tparam DestCols         Must be `W_DOWN_SCALE_COLS` (= N/128 for
 * block-wise).
 * @param  expert_scales_down  GM pointer to the full scale tensor.
 * @param  id                  Expert index.
 * @param  base_col            First output col of this block's M stripe
 *                             (multiple of 128).
 * @param  dest                SHM slot `[2 row-blocks][W_DOWN_SCALE_COLS]`
 *                             fp32.
 *
 * @note Prefetch-warp only.  Unlike the fp8 tile loaders, this function
 *       uses synchronous SHM stores (no `pipe`) because the payload is
 *       tiny (≤ 32 bytes for Qwen3.5-35B) and fits comfortably in a
 *       single warp-sized burst.
 */
template <typename Dims, std::size_t DestRows, std::size_t DestCols>
__device__ inline void moe_load_down_wgmma_weight_scale_tile(
    const S_element* __restrict__ expert_scales_down, std::uint32_t id,
    std::uint32_t base_col, S_element (&dest)[DestRows][DestCols]) {
  using CoreDims = MoECoreDims<Dims>;
  static_assert(
      DestRows == 2,
      "down-proj weight-scale tile must have 2 row-blocks (WG0, WG1)");

  constexpr uint32_t COLS = DestCols;  // W_DOWN_SCALE_COLS = N/128 (block-wise)

  const unsigned thread = get_thread<Dims>();
  const unsigned pw = get_prefetch_warp<Dims>();  // 0..3

  // WG0 covers weight rows [base_col    .. base_col + 63]  → row-block
  // (base_col      ) / 128 WG1 covers weight rows [base_col+64 .. base_col +
  // 127] → row-block (base_col + 64 ) / 128
  //
  // For the canonical layout (`base_col` multiple of 128) these two
  // indices are equal.  Computing them independently keeps the loader
  // correct if a future config picks a non-multiple-of-128 base_col.
  if (pw == 0 && thread < DestRows * COLS) {
    const uint32_t wg = thread / COLS;  // 0..1
    const uint32_t cb = thread % COLS;  // col-block index in [0, COLS)
    const uint32_t weight_row = base_col + (wg == 0 ? 0u : 64u);
    const uint32_t rb = weight_row / 128;
    dest[wg][cb] =
        expert_scales_down[id * Dims::DOWN_SCALE_ROWS * Dims::DOWN_SCALE_COLS +
                           rb * Dims::DOWN_SCALE_COLS + cb];
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
  #ifndef MONO_PROFILE_SKIP_PREFETCH
    pipe.producer_acquire();
    moe_request_down_expert<Dims>(expert_weights_down, expert_scales_down,
                                  first_expert.id, shm, 0, pipe);
    moe_request_temp_token<Dims>(spec->temp_bf16, first_expert, 0,
                                 shm->t_bf16[0], pipe);
    pipe.producer_commit();
  #endif
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
  #ifndef MONO_PROFILE_SKIP_PREFETCH
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
  #endif
      } else {
  #ifndef MONO_PROFILE_SKIP_CALC
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
  #endif
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
  #ifndef MONO_PROFILE_SKIP_CALC
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
  #endif
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
  #ifndef MONO_PROFILE_SKIP_CALC
      moe_down_reduction_topk<Dims>(shm->partial_result,
                                    sorted_row0 < expert.last_token,
                                    sorted_row1 < expert.last_token,
                                    sorted_row0, sorted_row1, shmem, result);
  #endif

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

///////////////////////////////////////////////////////////////////////////////
//
// moe_down_projection_BS8_allexperts_wgmma_tma
//
// TMA + WGMMA down-projection for BS <= 8.  Both the fp8 expert-weight
// tile and the fp8 intermediate-activation tile are loaded via
// `cp.async.bulk.tensor.2d` with `CU_TENSOR_MAP_SWIZZLE_128B`, issued by
// a single TMA launcher thread (warp 8, lane 0).  Completion of each
// tile is signalled via SHM mbarriers (`bar_w[2]`, `bar_a[2]` reused
// from Phase 3); consumer warps wait via `mbarrier.try_wait.parity`.
//
// Descriptors are built host-side in the torch binding wrapper and
// passed to the top-level kernel as `__grid_constant__ CUtensorMap
// const` parameters.  At this device-side helper boundary they appear
// as `CUtensorMap const&` — the `__grid_constant__` qualifier only
// applies at the kernel-function boundary.
//
template <typename Dims>
__device__ inline void moe_down_projection_BS8_allexperts_wgmma_tma(
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down, std::uint32_t top_k,
    std::uint32_t batch_size, MoEGemmSpec<Dims>* __restrict__ spec,
    MoE_SHM<Dims>* __restrict__ shmem, CUtensorMap const& down_weights_desc,
    CUtensorMap const& down_activations_desc,
    // ── Per-expert-fused mode (cluster path, design §2 / §7.2, R5.5) ──
    // When `single_expert == false` (default, byte-identical legacy
    // path) the helper runs prologue → all-experts loop → epilogue
    // exactly as before.  When `single_expert == true` the helper:
    //   * skips the prologue (`out_accum` zero-fill + mbarrier init);
    //     the caller MUST emit it once before the per-expert-fused
    //     outer loop in `moe_kernel_topk_BS8`.
    //   * iterates ONLY the single expert at index `single_expert_e`
    //     (no `expert_stride` walk).
    //   * skips the epilogue (`out_accum → spec->down_partial_out`
    //     GMEM writeback); the caller MUST emit it once after the
    //     last per-expert iteration.
    // This split lets the cluster path interleave Phase 3 and Phase 4
    // on a per-expert cadence with a `cluster_sync()` between them
    // (site #2 fires once per assigned expert, R6.1, R6.2).  The
    // `out_accum` SHM slab is the per-expert accumulator carried
    // across iterations so the post-loop writeback merges all of the
    // cluster's assigned-expert contributions into one HBM partial.
    bool single_expert = false, std::uint32_t single_expert_e = 0u,
    bool emit_prologue = true, bool emit_epilogue = true) {
  static_assert(Dims::BS <= 8,
                "moe_down_projection_BS8_allexperts_wgmma_tma is BS<=8 only");
  using CoreDims = MoECoreDims<Dims>;
  constexpr uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;

  // `expert_weights_down` is retained on the parameter list for signature
  // parity with the `cp.async` reference but is not dereferenced on the
  // TMA path — all weight GM reads go through `down_weights_desc`.
  (void)expert_weights_down;

  // ── Compile-time constants ────────────────────────────────────────────
  constexpr std::uint32_t K_TILE_W = CoreDims::K_TILE_WGMMA;              // 32
  constexpr std::uint32_t K_STEP_DOWN = CoreDims::K_STEP_WGMMA;           // 128
  constexpr std::uint32_t WGMMAS_PER_STEP_DOWN = K_STEP_DOWN / K_TILE_W;  // 4
  constexpr std::uint32_t K_TILES_DOWN = Dims::N / K_STEP_DOWN;           // 4
  constexpr std::uint32_t DOWN_COL_TILE =
      CoreDims::DOWN_COL_TILE;                                  // 128 or 256
  constexpr std::uint32_t DOWN_GRID = CoreDims::DOWN_GRID;      // 16 or 8
  constexpr std::uint32_t DOWN_GROUPS = CoreDims::DOWN_GROUPS;  // 8 or 16
  // Phase-2a layout alignment: one block owns `DOWN_COL_TILE` output
  // cols.  The two WGs (64 output cols each per WGMMA pass) together
  // cover 128 cols per pass, so we need `HALVES = DOWN_COL_TILE / 128`
  // sequential passes per K-step:
  //   * Pre Phase 2a (DOWN_COL_TILE=128): HALVES=1, pre-alignment 128-col
  //     behaviour.  Weight tile in SHM is 128×128.
  //   * Post Phase 2a (DOWN_COL_TILE=256, BS8 TMA+WGMMA only): HALVES=2,
  //     weight tile in SHM is 256×128.  Pass 0 covers output rows
  //     [base_col+0..base_col+127], pass 1 covers [+128..+255].
  constexpr std::uint32_t DOWN_COL_HALVES = DOWN_COL_TILE / 128u;  // 1 or 2
  static_assert(DOWN_COL_TILE % 128u == 0,
                "DOWN_COL_TILE must be a multiple of 128 for the TMA+WGMMA "
                "down-projection (the weight tile is laid out as 128-row "
                "SWZ128 atoms on the M axis).");
  static_assert(DOWN_COL_HALVES <= 2u,
                "DOWN_COL_TILE > 256 not supported by the Phase-2a "
                "two-halves WGMMA structure.");
  // Per-K-step weight TMA transfer size.  Each half is a 128×128 fp8
  // tile = 16384 B; HALVES halves stack along the M axis into
  // `w_down_wgmma[slot][0..DOWN_COL_TILE-1]`.
  constexpr std::uint32_t DOWN_W_TX_BYTES_PER_HALF = 16384u;
  constexpr std::uint32_t DOWN_W_TX_BYTES_TOTAL =
      DOWN_W_TX_BYTES_PER_HALF * DOWN_COL_HALVES;  // 16384 or 32768
  constexpr std::uint32_t W_DOWN_SCALE_COLS =
      MoE_SHM<Dims>::U::TinyDataWGMMA_TMA::W_DOWN_SCALE_COLS;

  static_assert(Dims::N % K_STEP_DOWN == 0,
                "Dims::N must be a multiple of K_STEP_DOWN=128");

  // A descriptor strides for the 128×128 fp8 weight tile under
  // SWIZZLE_128B.  The TMA hardware applies the 8-row × 128-byte
  // core-matrix XOR swizzle at write time, so each 1024-B atom holds
  // one 8-row M-block.  CUTLASS Major::K B128 layout:
  //   LBO = 16 B   (one K-core-matrix within the 1024-B atom)
  //   SBO = 1024 B (next M-block atom)
  //   swizzle_mode = 1
  // The Python pre-interleave is NOT applied — the raw `[E, K, N]`
  // row-major fp8 weight tensor is fed to the TMA, and the swizzle
  // hardware produces the canonical layout in SHM.
  constexpr std::uint64_t A_LBO = 16ULL;
  constexpr std::uint64_t A_SBO = 1024ULL;
  constexpr std::uint32_t A_SWIZZLE = 1u;
  // B descriptor strides for the 8-token × 128-K fp8 activation tile
  // under SWIZZLE_128B (token-major Major::K B128 canonical layout).
  // The TMA hardware applies the 8-row × 128-byte XOR at write time:
  //   logical byte(tok, kc, ki) = tok * 128 + kc * 16 + ki
  //   SHM byte = logical byte  XOR  swizzle(tok bits)
  // CUTLASS strides for this layout:
  //   LBO = 16 B   (stride between 16-B K-chunks along K inside a row)
  //   SBO = 128 B  (stride between 8-row atoms; unused here, N=8 → 1 atom)
  //   swizzle = 1  (SWZ128)
  constexpr std::uint64_t B_LBO = 16ULL;
  constexpr std::uint64_t B_SBO = 128ULL;
  constexpr std::uint32_t B_SWIZZLE = 1u;

  // ── Warp / lane identity ──────────────────────────────────────────────
  const unsigned thread_in_block = threadIdx.x;
  const unsigned warp = thread_in_block / 32;  // 0..11
  const unsigned lane = thread_in_block & 31;  // 0..31
  const bool is_wg1 = (warp >= 4 && warp < 8);
  const bool is_calc = (warp < 8);
  const unsigned warp_in_wg = warp & 3;  // 0..3 within each WG
  const unsigned my_wg = is_wg1 ? 1u : 0u;

  // TMA path uses the `tiny_wgmma_tma` union variant (byte-identical to
  // `tiny_wgmma` plus 32 B of mbarriers + the reorg tables at the tail).
  auto* shm = &shmem->u.tiny_wgmma_tma;

  // Cluster-only sub-struct accessor.  When `use_cluster<Dims>::value`
  // is false this is an empty (sizeof == 1 B) sub-struct and `cl` is
  // never dereferenced — the legacy path's `if constexpr` branches
  // skip every `cl->...` access at compile time so the byte layout
  // and observable behaviour stay byte-identical (R11.1, R11.3).
  // On the cluster path `cl` exposes `cluster_temp_fp8`,
  // `cluster_temp_act_scale`, `cluster_down_b_staging`, and
  // `cluster_down_b_staging_scales` per design §5.3.
  auto* cl = &shm->cluster_ext;

  // ── Grid-to-(expert-group, output-col-tile) mapping ───────────────────
  const std::uint32_t down_group = blockIdx.x / DOWN_GRID;
  const std::uint32_t down_block_idx = blockIdx.x % DOWN_GRID;
  const std::uint32_t base_col = down_block_idx * DOWN_COL_TILE;

  // `pipe` is retained for the activation-scale cp.async path: the fp8
  // weight / activation tiles are loaded via TMA + mbarrier, not via
  // `pipe`.
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  // ── Per-thread fp32 accumulators ──────────────────────────────────────
  // One m64n8k32 WGMMA holds 4 fp32 accumulators per thread (d0/d1/d2/d3).
  // The down-projection's inner K chain stacks 4 WGMMAs per K-step into a
  // "lo" chunk (K[0..63], 2 WGMMAs) and "hi" chunk (K[64..127], 2 WGMMAs).
  // Phase-2a generalizes this to `DOWN_COL_HALVES` parallel halves along
  // the M axis, so we hold `4 * HALVES` final accumulators per thread:
  //   * chunk_d_lo[h][0..3] / chunk_d_hi[h][0..3] — per-half per-K-chunk
  //                                                  scratch, reset each
  //                                                  K-step.
  //   * final_d[h][0..3]                           — per-half per-expert
  //                                                  K-loop accumulator
  //                                                  (scaled by ws·as).
  float chunk_d_lo[DOWN_COL_HALVES][4] = {{0.f}};
  float chunk_d_hi[DOWN_COL_HALVES][4] = {{0.f}};
  float final_d[DOWN_COL_HALVES][4] = {{0.f}};

  // ── Zero out_accum + Phase-4 mbarrier (re-)initialization ────────────
  //
  // Two independent SHM publishes happen here, merged behind a single
  // block-wide sync:
  //
  //   (1) Zero per-block SHM `out_accum[BS][DOWN_COL_TILE]`.  `out_accum`
  //       is not read until the per-expert accumulate loop below, which
  //       follows several additional `__syncthreads()` (priming drain +
  //       per-K-iter syncs).  A single trailing sync here is sufficient.
  //
  //   (2) Re-initialize the 4 TMA mbarriers (R4.1, R4.9).  The Phase-3→4
  //       `grid.sync()` guarantees the barriers are idle at entry; we
  //       reset them to `arrival_count = 1` on the launcher thread and
  //       publish with `fence.mbarrier_init.release.cluster`.  They are
  //       not armed or waited on until inside the expert loop, strictly
  //       after this sync.
  //
  // Both targets (`out_accum`, `bar_w/bar_a`) are disjoint SHM regions
  // so the two operations race-freely run in parallel; only one sync is
  // needed to publish both.
  //
  // The `out_accum` zero-fill is unconditional — its cost is negligible
  // (1 KB per block with 384 threads) and keeping it well-defined
  // simplifies reasoning under either profile flag.  The mbarrier inits
  // are also unconditional: they cost only 4 SHM stores + 1 fence and
  // leave the barriers in a known-idle state.  The K-loop
  // waits/arms are themselves gated on SKIP_PREFETCH, so uninit'd
  // barriers are never a concern.
  //
  // Per-expert-fused cluster path: the prologue is emitted once by the
  // caller before the outer per-expert loop in `moe_kernel_topk_BS8`,
  // so we gate both halves on `emit_prologue` (default true on the
  // legacy all-experts call site, false when the cluster fused loop
  // calls us per-expert — see the parameter-list comment block above).
  if (emit_prologue) {
    for (unsigned idx = thread_in_block; idx < Dims::BS * DOWN_COL_TILE;
         idx += blockDim.x) {
      const unsigned tok = idx / DOWN_COL_TILE;
      const unsigned col = idx % DOWN_COL_TILE;
      shm->out_accum[tok][col] = 0.f;
    }
    if (is_tma_launcher_thread<Dims>()) {
      mbarrier_init(&shm->bar_w[0], 1u);
      mbarrier_init(&shm->bar_w[1], 1u);
      mbarrier_init(&shm->bar_a[0], 1u);
      mbarrier_init(&shm->bar_a[1], 1u);
      fence_mbarrier_init_release_cluster();
    }
    __syncthreads();
  }

  MONO_PHASE_TIMESTAMP(t_down_after_prologue);

  const std::uint32_t expert_count = shmem->expert_count;

  // ── Per-expert loop (expert_start = down_group, stride = DOWN_GROUPS) ─
  //
  // Per-expert-fused cluster path (`single_expert == true`): the loop
  // collapses to ONE iteration over `e = single_expert_e`.  The caller
  // walks the per-cluster expert sequence externally and fires a
  // `cluster_sync()` between each (Phase-3, Phase-4) pair (R6.1, R5.5).
  // Bypassing the `expert_stride` walk on this path is correct because
  // the cluster path has no cross-expert mbarrier stitching to
  // preserve: each per-expert call re-primes weight TMAs from scratch
  // (the down helper has no end-of-K-loop next-expert weight stitch).
  const std::uint32_t loop_e_start =
      single_expert ? single_expert_e : down_group;
  const std::uint32_t loop_e_end =
      single_expert ? (single_expert_e + 1u) : expert_count;
  const std::uint32_t loop_e_stride = single_expert ? 1u : DOWN_GROUPS;
  for (std::uint32_t e = loop_e_start; e < loop_e_end; e += loop_e_stride) {
    const std::uint32_t id = shmem->experts[e].id;

    // Per-expert (expert, token) reorganization state (R11).  These
    // values are populated by `prepare_moe_topk_BS8` in Phase 1/2 and
    // drive the bulk activation TMA's outer coordinate and row count.
    const std::uint32_t routed_count =
        static_cast<std::uint32_t>(shm->expert_routed_count[id]);
    const std::uint32_t expert_start =
        static_cast<std::uint32_t>(shm->expert_slot_start[id]);

    // Per-slot parity state.  MUST be reset every expert because the
    // slot re-load on a new expert invalidates the previous expert's
    // parity state (R4.7).
    std::uint32_t parity_w[2] = {0u, 0u};
    std::uint32_t parity_a[2] = {0u, 0u};

    // Reset per-expert `final_d` accumulator.
  #pragma unroll
    for (std::uint32_t h = 0; h < DOWN_COL_HALVES; ++h) {
  #pragma unroll
      for (std::uint32_t r = 0; r < 4u; ++r) {
        final_d[h][r] = 0.f;
      }
    }

    // ── Load weight scales ─────────────────────────────────────────────
    // Weights scales are fixed across K-steps, so we only load them
    // once per expert.  Loaded via cp.async from the prefetch warps —
    // not TMA.
    //
    // Phase-2a layout alignment: with DOWN_COL_HALVES halves per block,
    // the two halves occupy different 128-row scale blocks (half 0 at
    // row-block `base_col/128`, half 1 at `base_col/128 + 1`).  We load
    // each half's scales into `w_down_scale[h][wg][cb]` — reusing the
    // existing double-buffer dimension (never double-used for scales;
    // they're loaded once per expert) as the half index.
    if (is_prefetch_warp<Dims>()) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH
    #pragma unroll
      for (std::uint32_t h = 0; h < DOWN_COL_HALVES; ++h) {
        moe_load_down_wgmma_weight_scale_tile<Dims>(
            expert_scales_down, id, base_col + h * 128u, shm->w_down_scale[h]);
      }
  #endif
    }

    // ── Priming: prefetch slot 0 (w + a + a_scale for K-step 0) ────────
    //
    // Non-cluster path (USE_CLUSTER == false): the fp8 weight and
    // activation tiles land via `cp.async.bulk.tensor.2d` issued by
    // the launcher thread, with completion signalled on `bar_w[0]` /
    // `bar_a[0]`.  The activation scale tile loads via cp.async
    // through `pipe`.
    //
    // Cluster path (USE_CLUSTER == true): the WEIGHT TMA priming is
    // unchanged (R4.6 — weights stay per-block).  The ACTIVATION TMA
    // priming is replaced by per-K-step DSHM staging from the two
    // peer up-blocks of cluster ranks `(2*kk, 2*kk + 1)` for
    // `kk = 0` (peers `(0, 1)`).  The activation scale cp.async load
    // is also replaced by DSHM scale staging (R5.4).  See the K-loop
    // body below for the per-K-step DSHM staging implementation.
    //
    // Compiled out under MONO_PROFILE_SKIP_PREFETCH; the matching
    // compute-side `mbarrier_try_wait_parity` on bar_{w,a}[0] inside
    // the K-loop below is also compiled out so there is no
    // spin-forever deadlock.
    if (is_tma_launcher_thread<Dims>()) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH
      // Weight tile for K-step 0: arm bar_w[0] with the TOTAL tx_bytes
      // (= 16384 · HALVES = 16 KB for pre Phase 2a, 32 KB for post)
      // and issue `HALVES` back-to-back 128×128 TMAs.  Half 0 lands at
      // `&w_down_wgmma[0][0][0]`, half 1 at
      // `&w_down_wgmma[0][128][0]` (= base + 16384 B).  All TMAs in
      // this block retire into the same bar_w[0], so a single
      // `mbarrier.try_wait.parity` on the compute side drains both.
      mbarrier_arrive_expect_tx(&shm->bar_w[0],
                                /*tx_bytes=*/DOWN_W_TX_BYTES_TOTAL);
    #pragma unroll
      for (std::uint32_t h = 0; h < DOWN_COL_HALVES; ++h) {
        W_element* dest_base = &shm->w_down_wgmma[0][h * 128u][0];
        tma_load_down_wgmma_tile(down_weights_desc, /*expert_id=*/id,
                                 /*K=*/Dims::HIDDEN_STATES,
                                 /*base_col=*/base_col + h * 128u,
                                 /*k_start=*/0u,
                                 /*dest_smem_ptr=*/(void*)dest_base,
                                 /*bar_smem_ptr=*/&shm->bar_w[0]);
      }

      if constexpr (!use_cluster<Dims>::value) {
        // Activation tile for K-step 0: only issue when routed_count > 0.
        // The helper issues 8 TMAs (one per K-chunk); the descriptor's
        // boxDim=(16, 8) means each TMA delivers 16*8=128 B, collectively
        // 8*128=1024 B = full 1 KB slot regardless of routed_count.
        if (routed_count > 0u) {
          mbarrier_arrive_expect_tx(&shm->bar_a[0], /*tx_bytes=*/1024u);
          tma_load_down_wgmma_activation_bulk(
              down_activations_desc, /*k_start=*/0u,
              /*expert_slot_start=*/expert_start,
              /*dest_smem_ptr=*/&shm->a_down_wgmma[0][0][0][0],
              /*bar_smem_ptr=*/&shm->bar_a[0]);
        }
      }
  #endif
    }
    if constexpr (!use_cluster<Dims>::value) {
      // Zero-fill the unused tail of slot 0. Warp 8 lanes 1..31 do the
      // work; lane 0 is gated out internally so it can issue the TMA in
      // parallel.  Also gated by SKIP_PREFETCH because the slot only
      // exists (and only matters) when prefetches are live.
      //
      // Cluster path: the slot is unused; the zero-fill happens on the
      // local `cluster_down_b_staging` slab inside the K-loop instead
      // (R5.8, R10.4).
      if (warp == 8u && routed_count < 8u) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH
        zero_fill_unused_down_act_slots<Dims>(routed_count,
                                              &shm->a_down_wgmma[0][0][0][0]);
  #endif
      }
    }

    if constexpr (!use_cluster<Dims>::value) {
      // Prefetch warps load all per-(token, K-block) activation scales
      // for THIS EXPERT in a single cp.async burst, before the K-loop
      // starts.  Hoisting all 64 fp32 (8 tok × 8 K-blocks) out of the
      // K-loop eliminates the per-K-step cp.async wait that previously
      // sat on the K-loop's critical path.
      //
      // Cluster path: the activation scales are sourced from peer
      // `cluster_temp_act_scale` slabs via DSHM at the head of every
      // K-step (R5.4); no per-expert cp.async burst is needed.
      if (is_prefetch_warp<Dims>()) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH
        moe_load_down_wgmma_activation_scales_for_expert<Dims>(
            spec, shmem, id, shm->a_down_scale, pipe);
  #endif
      }
    }
    // Drain the scale cp.asyncs and publish the zero-fill across the
    // block before the WGMMA consumers wait on bar_w[0] / bar_a[0].
    //
    // The pipe drain is a no-op when no `producer_commit` has happened
    // (SKIP_PREFETCH elides the only producers above), so it's safe to
    // keep it unconditional.  The `__syncthreads()` must stay
    // regardless to publish the unconditional out_accum zero-fill and
    // mbarrier init from the prologue.
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    __syncthreads();

    // ── Main K-loop ─────────────────────────────────────────────────────
    //
    // Main K-loop. Both WEIGHT and ACTIVATION tiles are loaded via
    // TMA (mbarrier wait); the activation-scale tile stays on cp.async
    // (pipeline drain) because it is small and the layout is already
    // convenient on that path.
    for (std::uint32_t s = 0; s < K_TILES_DOWN; ++s) {
      const std::uint32_t read_slot = s & 1;

      // ── Cluster path: per-K-step DSHM staging into local SHM ────────
      //
      // For the cluster variant the down-projection's fp8 B-operand
      // and per-token scales come from peer up-blocks via DSHM, not
      // from `spec->temp_fp8` via TMA (R5.3, R5.4, R5.6, R5.9; design
      // §7.5 path-(b)).  At K-step `kk = s ∈ [0, K_TILES_DOWN = 4)`:
      //   peer_lo = 2 * kk      → supplies cols `[ 0, 64)` of the
      //                            `[T_TILE = 8] × [128]` fp8 B-tile.
      //   peer_hi = 2 * kk + 1  → supplies cols `[64,128)`.
      // The pairing pattern is fixed across all 8 down-blocks of the
      // cluster (R5.3, design §7.8): every consumer down-block walks
      // the same 4-step (peer_lo, peer_hi) sequence.
      //
      // Staging byte layout ────────────────────────────────────────────
      //
      // The legacy WGMMA B descriptor for this kernel uses SWZ128 with
      // `B_LBO = 16`, `B_SBO = 128`, which expects the activation slab
      // to have the SWZ128 hardware-applied byte permutation produced
      // by a TMA write.  On the cluster path the staging is filled by
      // software (no TMA), so we cannot reproduce the SWZ128
      // permutation cheaply.  Instead we lay the staging out in
      // canonical CUTLASS Major::K B layout (SWIZZLE_NONE), exactly
      // matching the up-projection's `fp8_act[kc][tok][ki]` byte
      // pattern: `byte(kc, tok, ki) = kc * 128 + tok * 16 + ki`.  The
      // matching WGMMA B descriptor then uses `swizzle = 0`,
      // `B_LBO_CL = 128`, `B_SBO_CL = 128` — the same encoding the
      // up-projection's WGMMA already uses (`src/moe_up_projection.cu`
      // — `make_wgmma_desc(b_ptr, B_LBO=128, B_SBO=128, 0)`).  Only
      // the B-operand source pointer and B-side descriptor encoding
      // change; A-side / accumulators / k-stride accounting stay
      // byte-identical to the legacy SWZ128 path (design §7.5 path-(b)).
      //
      // The declared field shape is
      // `cluster_down_b_staging[T_TILE = 8][K_STEP_WGMMA = 128]`
      // (1024 B), `alignas(16)`.  We `reinterpret_cast` the same 1024
      // bytes into the K-major view `[FP8_ACT_NUM_CHUNKS = 8][T_TILE =
      // 8][FP8_ACT_K_CHUNK = 16]` for the staging stores, and the
      // WGMMA descriptor reads from that K-major view directly.  The
      // declared field shape and the K-major byte layout addresses
      // disagree on byte ordering — that disagreement is internal to
      // the cluster path and never observed by the legacy variant
      // (which simply does not use the field).  The 1024 B size is
      // identical either way.
      //
      // Mapping from peer-`cluster_temp_fp8` byte to staging byte:
      //   * peer_lo's slab is `[T_TILE = 8 tok] × [W_UP_COLS_WGMMA =
      //     64 c]` token-major; covers K-cols `[0, 64)` of the K-step.
      //     For dword `(t, c4 ∈ [0, 16))`:
      //       src = peer_lo_fp8[t * 64 + c4 * 4 .. + 3]
      //       dst kc_off = c4 / 4               (kc in [0, 4))
      //       dst ki_off = (c4 & 3) * 4         (0, 4, 8, 12)
      //       dst = staging[kc_off * 128 + t * 16 + ki_off .. + 3]
      //   * peer_hi's slab covers K-cols `[64, 128)` of the K-step.
      //     For dword `(t, c4 ∈ [0, 16))`:
      //       src = peer_hi_fp8[t * 64 + c4 * 4 .. + 3]
      //       dst kc_off = c4 / 4 + 4           (kc in [4, 8))
      //       dst ki_off = (c4 & 3) * 4
      //       dst = staging[kc_off * 128 + t * 16 + ki_off .. + 3]
      //
      // Per-thread work: 256 calc threads handle the 256 dwords (128
      // dwords per peer × 2 peers).  Threads `[0, 128)` cover peer_lo,
      // threads `[128, 256)` cover peer_hi.  Threads `[256, 256 + 16)`
      // load the 16 fp32 scale values (8 tokens × 2 peers).
      //
      // Under-T_TILE zero-fill (R5.8, R10.4): when the producing
      // up-block has fewer than `T_TILE = 8` routed tokens for the
      // current expert, the consumer overwrites the unused token rows
      // (`tok ∈ [routed_count, T_TILE)`) in the local staging slab
      // with zero fp8 bytes so the WGMMA contributes zero to the
      // accumulator for those rows.  Same semantics as the legacy
      // `zero_fill_unused_down_act_slots`, applied to
      // `cluster_down_b_staging` instead of `a_down_wgmma`.
  #ifndef MONO_PROFILE_SKIP_PREFETCH
      if constexpr (use_cluster<Dims>::value) {
        const std::uint32_t kk = s;
        const std::uint32_t peer_lo = 2u * kk;
        const std::uint32_t peer_hi = 2u * kk + 1u;

        // Translate peer ranks to cluster-shared address-space pointers
        // into the peer up-blocks' `cluster_temp_fp8` /
        // `cluster_temp_act_scale` slabs (R5.6, design §7.3).  Both
        // peers' SHM live at the same offset within their own block,
        // so `map_shared_rank` is correct for each.
        AQ_element* peer_lo_fp8 =
            map_shared_rank<AQ_element>(&cl->cluster_temp_fp8[0][0], peer_lo);
        AQ_element* peer_hi_fp8 =
            map_shared_rank<AQ_element>(&cl->cluster_temp_fp8[0][0], peer_hi);
        S_element* peer_lo_sc =
            map_shared_rank<S_element>(&cl->cluster_temp_act_scale[0], peer_lo);
        S_element* peer_hi_sc =
            map_shared_rank<S_element>(&cl->cluster_temp_act_scale[0], peer_hi);

        // Raw byte view of the local staging slab: 1024 B sized exactly
        // for the K-major `[FP8_ACT_NUM_CHUNKS = 8][T_TILE = 8]
        // [FP8_ACT_K_CHUNK = 16]` view we write here.
        std::uint8_t* staging_bytes =
            reinterpret_cast<std::uint8_t*>(&cl->cluster_down_b_staging[0][0]);

        // ── Copy fp8 [T_TILE = 8][W_UP_COLS_WGMMA = 64] tiles ─────────
        // 256 dwords total = 8 tokens × 16 dwords per row × 2 peers.
        // Each thread loads/stores ONE 4-byte dword.
        constexpr std::uint32_t LO_DWORDS_PER_PEER = 128u;  // 64 / 4 * 8
        constexpr std::uint32_t LO_DWORDS_TOTAL = 256u;     // both peers
        if (thread_in_block < LO_DWORDS_TOTAL) {
          const std::uint32_t which = thread_in_block / LO_DWORDS_PER_PEER;
          const std::uint32_t i = thread_in_block % LO_DWORDS_PER_PEER;
          // Token row: 16 dwords per token row (= 64 B / 4).
          const std::uint32_t tok = i >> 4;      // i / 16, 0..7
          const std::uint32_t c4 = (i & 15u);    // i % 16, 0..15
          const std::uint32_t c_byte = c4 * 4u;  // 0..60

          AQ_element* peer_src = (which == 0u) ? peer_lo_fp8 : peer_hi_fp8;
          // K-major destination kc offset:
          //   peer_lo: kc in [0, 4)
          //   peer_hi: kc in [4, 8)
          const std::uint32_t kc_off =
              (c4 >> 2) + (which == 0u ? 0u : 4u);       // 0..7
          const std::uint32_t ki_byte = (c4 & 3u) * 4u;  // 0, 4, 8, 12

          // Source: peer's `cluster_temp_fp8[tok][c_byte]` —
          // contiguous `[T_TILE][W_UP_COLS_WGMMA]` row-major.
          // Destination: K-major `staging[kc_off][tok][ki_byte]`.
          const std::uint32_t dst_off = kc_off * 128u + tok * 16u + ki_byte;

          if (tok < routed_count) {
            const std::uint32_t src_off = tok * 64u + c_byte;
            std::uint32_t dword =
                *reinterpret_cast<std::uint32_t*>(peer_src + src_off);
            *reinterpret_cast<std::uint32_t*>(staging_bytes + dst_off) = dword;
          } else {
            // Under-T_TILE zero-fill (R5.8, R10.4).  Peer's slab for
            // this expert produced no token at row `tok`; zero the
            // staging area so the WGMMA reads zero and contributes
            // zero to the accumulator.
            *reinterpret_cast<std::uint32_t*>(staging_bytes + dst_off) = 0u;
          }
        }

        // ── Copy fp32 per-token scales [T_TILE = 8] from each peer ────
        // 8 fp32 scales per peer; 16 total.  Threads
        // `[LO_DWORDS_TOTAL, LO_DWORDS_TOTAL + 16)` handle them.
        if (thread_in_block >= LO_DWORDS_TOTAL &&
            thread_in_block < LO_DWORDS_TOTAL + 2u * CoreDims::T_TILE) {
          const std::uint32_t i = thread_in_block - LO_DWORDS_TOTAL;
          const std::uint32_t which = i / CoreDims::T_TILE;
          const std::uint32_t tok = i % CoreDims::T_TILE;
          S_element* peer_src = (which == 0u) ? peer_lo_sc : peer_hi_sc;
          // `cluster_down_b_staging_scales[half][tok]`:
          //   half = 0 ← peer_lo (lower-64-cols of K-step)
          //   half = 1 ← peer_hi (upper-64-cols of K-step)
          S_element* local_dst = &cl->cluster_down_b_staging_scales[which][0];
          if (tok < routed_count) {
            local_dst[tok] = peer_src[tok];
          } else {
            // Defensive zero-fill so any stray multiply through the
            // scale-apply doesn't propagate NaN.  The fp8 bytes for
            // this token row were already zeroed above, so the
            // accumulator contribution is zero either way.
            local_dst[tok] = 0.f;
          }
        }
      }
      // Make the cluster-staging writes visible to the WGMMA-issuing
      // calc warps (R6.2; design §7.5 — "a single `__syncthreads()`
      // after the cooperative copy makes the local staging slab
      // visible to the WGMMA-issuing warpgroup").  No-op when
      // `use_cluster<Dims>::value` is false (the legacy path's
      // `__syncthreads()` at the end of each iteration covers slot
      // visibility).
      if constexpr (use_cluster<Dims>::value) {
        __syncthreads();
      }
  #endif

      // ── COMPUTE half: WGMMA + scale-apply || TMA prefetch step s+1 ──
      if (is_calc) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH
        // Wait for this step's weight tile to be fully in SHM.  The
        // launcher pre-armed bar_w[read_slot] with tx=16384 before
        // issuing the 128x128 weight TMA (priming for s=0, previous
        // compute-half for s>0).
        //
        // Tied to SKIP_PREFETCH so the wait and the launcher's arm are
        // compiled in/out together — skipping only one of them would
        // either spin forever (skip arm) or fire without a consumer
        // (skip wait).
        while (!mbarrier_try_wait_parity(&shm->bar_w[read_slot],
                                         parity_w[read_slot])) {
        }
        parity_w[read_slot] ^= 1;

        if constexpr (!use_cluster<Dims>::value) {
          // Wait for this step's activation tile if any tokens route to
          // this expert.  When routed_count == 0 the launcher did not arm
          // bar_a[read_slot] and did not issue a TMA; the SHM slot is
          // entirely zero-filled by `zero_fill_unused_down_act_slots`,
          // and the WGMMA reads zero — no wait needed.  (R4.5, R12.5.)
          //
          // Cluster path: bar_a is unused (the activation tile lives in
          // `cluster_down_b_staging` and is published by the
          // `__syncthreads()` after the per-K-step DSHM staging above).
          if (routed_count > 0u) {
            while (!mbarrier_try_wait_parity(&shm->bar_a[read_slot],
                                             parity_a[read_slot])) {
            }
            parity_a[read_slot] ^= 1;
          }
        }
  #endif

  #ifndef MONO_PROFILE_SKIP_CALC
        // A descriptor bases per WG per half — the weight tile is
        // laid out as `HALVES` contiguous 128-row M-slabs (half 0 at
        // rows [0..127], half 1 at rows [128..255]).  Within each
        // slab, WG0 owns rows [0..63] and WG1 owns rows [64..127].
        //
        //   A-desc base for (wg, h):
        //     &w_down_wgmma[slot][h * 128 + (wg==1 ? 64 : 0)][0]
        //     = slot_base + h * 16384 + (wg==1 ? 8192 : 0)
        const void* a_slot_base =
            (const void*)&shm->w_down_wgmma[read_slot][0][0];
        const std::uint32_t wg_offset_bytes = is_wg1 ? 8192u : 0u;

        wgmma_fence();

        // Per-WGMMA K-advancement (per j inside the inner pass loop):
        //   A: 2 * A_LBO = 32 B inside the 1024-B A atom.
        //   B: depends on path (legacy SWZ128 vs. cluster SWZ_NONE).
        //     Legacy:  2 * B_LBO   = 32 B inside the 1024-B B atom
        //              (next 2 K-chunks within a swizzle atom).
        //     Cluster: 2 * 128 B   = 256 B advances 2 kc-blocks in
        //              the K-major SWIZZLE_NONE layout
        //              `[kc=8][tok=8][ki=16]`; matches the
        //              up-projection K-loop's b-pointer stride
        //              (`&shm->fp8_act[slot][j * 2][0][0]`).
        constexpr std::uint32_t A_K_STRIDE = 2u * A_LBO;
        // B-operand source pointer + descriptor encoding.
        //
        // Legacy (USE_CLUSTER == false): the activation tile lives in
        // `a_down_wgmma[read_slot]` (1024 B SWZ128 atom).  Descriptor
        // uses `swizzle = 1`, `B_LBO = 16`, `B_SBO = 128`.
        //
        // Cluster (USE_CLUSTER == true): the activation tile lives in
        // local `cluster_down_b_staging` (1024 B `alignas(16)`
        // SWIZZLE_NONE K-major view, written by the staging step
        // above).  Descriptor uses `swizzle = 0`, `B_LBO_CL = 128`,
        // `B_SBO_CL = 128` — same encoding the up-projection's WGMMA
        // already uses for its `fp8_act` SWIZZLE_NONE B-operand.  Per
        // design §7.5 path-(b) this keeps the rest of the WGMMA
        // encoding (A-side, accumulator chain) byte-identical to the
        // legacy path; only the B-source pointer + B-side
        // swizzle/LBO/SBO and per-j K-stride change.
        constexpr std::uint64_t B_LBO_CL = 128ULL;
        constexpr std::uint64_t B_SBO_CL = 128ULL;
        constexpr std::uint32_t B_SWIZZLE_CL = 0u;
        [[maybe_unused]] constexpr std::uint32_t B_K_STRIDE = 2u * B_LBO;
        [[maybe_unused]] constexpr std::uint32_t B_K_STRIDE_CL = 2u * 128u;
        const void* b_slot_base;
        if constexpr (use_cluster<Dims>::value) {
          b_slot_base = (const void*)&cl->cluster_down_b_staging[0][0];
        } else {
          b_slot_base = (const void*)&shm->a_down_wgmma[read_slot][0][0][0];
        }

          // Per-half WGMMA passes.  Each pass runs the same
          // 4-chained-m64n8k32 structure as the pre Phase-2a kernel and
          // accumulates into `chunk_d_lo[h]` / `chunk_d_hi[h]`, then
          // applies the (ws, as) scales at the K=128 boundary and folds
          // into `final_d[h]`.
    #pragma unroll
        for (std::uint32_t h = 0; h < DOWN_COL_HALVES; ++h) {
          const std::uint32_t half_offset_bytes = h * 16384u;
          const void* a_base =
              (const void*)((const char*)a_slot_base + half_offset_bytes +
                            wg_offset_bytes);

            // 4 chained WGMMAs into chunk_d_lo[h]  (K[0..63], j = 0, 1).
    #pragma unroll
          for (std::uint32_t j = 0; j < 2; ++j) {
            const void* a_ptr =
                (const void*)((const char*)a_base + j * A_K_STRIDE);
            const void* b_ptr =
                use_cluster<Dims>::value
                    ? (const void*)((const char*)b_slot_base +
                                    j * B_K_STRIDE_CL)
                    : (const void*)((const char*)b_slot_base + j * B_K_STRIDE);
            std::uint64_t desc_a =
                make_wgmma_desc(a_ptr, A_LBO, A_SBO, A_SWIZZLE);
            std::uint64_t desc_b =
                use_cluster<Dims>::value
                    ? make_wgmma_desc(b_ptr, B_LBO_CL, B_SBO_CL, B_SWIZZLE_CL)
                    : make_wgmma_desc(b_ptr, B_LBO, B_SBO, B_SWIZZLE);
            wgmma_m64n8k32_e4m3_e4m3_f32(desc_a, desc_b, chunk_d_lo[h][0],
                                         chunk_d_lo[h][1], chunk_d_lo[h][2],
                                         chunk_d_lo[h][3]);
          }

            // 4 chained WGMMAs into chunk_d_hi[h]  (K[64..127], j = 2, 3).
    #pragma unroll
          for (std::uint32_t j = 2; j < WGMMAS_PER_STEP_DOWN; ++j) {
            const void* a_ptr =
                (const void*)((const char*)a_base + j * A_K_STRIDE);
            const void* b_ptr =
                use_cluster<Dims>::value
                    ? (const void*)((const char*)b_slot_base +
                                    j * B_K_STRIDE_CL)
                    : (const void*)((const char*)b_slot_base + j * B_K_STRIDE);
            std::uint64_t desc_a =
                make_wgmma_desc(a_ptr, A_LBO, A_SBO, A_SWIZZLE);
            std::uint64_t desc_b =
                use_cluster<Dims>::value
                    ? make_wgmma_desc(b_ptr, B_LBO_CL, B_SBO_CL, B_SWIZZLE_CL)
                    : make_wgmma_desc(b_ptr, B_LBO, B_SBO, B_SWIZZLE);
            wgmma_m64n8k32_e4m3_e4m3_f32(desc_a, desc_b, chunk_d_hi[h][0],
                                         chunk_d_hi[h][1], chunk_d_hi[h][2],
                                         chunk_d_hi[h][3]);
          }
        }

        wgmma_commit_group();
        wgmma_wait_group<0>();

        // ── Scale-apply at the K=128 boundary (per-half) ──────────────
        // Each half uses its own weight-scale row-block entry
        // `w_down_scale[h][my_wg][ws_col]` loaded once per expert.
        // The activation scales are shared across halves (one set per
        // slot, indexed by token) because both halves process the same
        // 128-K K-step against the same activation tile.
        //
        // Pre-multiply `ws * as_*` once per K-step / half so each
        // accumulator update is a single fused-multiply-add chain
        // (2 dependent FMAs per output) instead of `(c * as) * ws +
        // (c * as) * ws + final_d` which serialises into ~5 dependent
        // ops per output.  Saves ~30-50% on the scale-apply portion of
        // the K-step at zero numerical risk (math is associative for
        // fp32 accumulators that are themselves fma chains).
        //
        // Activation scales are read from the hoisted per-expert table
        // `a_down_scale[tok][k_block_idx]` (loaded once at the top of
        // the expert loop), where `k_block_idx = s * 2 + half`.
        //
        // Cluster path: scales come from the per-K-step DSHM-staged
        // `cluster_down_b_staging_scales[half][tok]`, populated above
        // (R5.4).  Half 0 holds the lower-64-cols' scale (from
        // peer_lo), half 1 the upper-64-cols' (from peer_hi).
        const std::uint32_t tok_02 = (lane % 4) * 2;
        const std::uint32_t tok_13 = (lane % 4) * 2 + 1;
        const std::uint32_t k_block_lo = s * 2u + 0u;
        const std::uint32_t k_block_hi = s * 2u + 1u;

        float as_lo_02, as_hi_02, as_lo_13, as_hi_13;
        if constexpr (use_cluster<Dims>::value) {
          as_lo_02 = cl->cluster_down_b_staging_scales[0][tok_02];
          as_hi_02 = cl->cluster_down_b_staging_scales[1][tok_02];
          as_lo_13 = cl->cluster_down_b_staging_scales[0][tok_13];
          as_hi_13 = cl->cluster_down_b_staging_scales[1][tok_13];
          (void)k_block_lo;
          (void)k_block_hi;
        } else {
          as_lo_02 = shm->a_down_scale[tok_02][k_block_lo];
          as_hi_02 = shm->a_down_scale[tok_02][k_block_hi];
          as_lo_13 = shm->a_down_scale[tok_13][k_block_lo];
          as_hi_13 = shm->a_down_scale[tok_13][k_block_hi];
        }

        const std::uint32_t ws_col = (W_DOWN_SCALE_COLS > 1) ? s : 0u;

    #pragma unroll
        for (std::uint32_t h = 0; h < DOWN_COL_HALVES; ++h) {
          const float ws = shm->w_down_scale[h][my_wg][ws_col];
          const float wsa_lo_02 = ws * as_lo_02;
          const float wsa_lo_13 = ws * as_lo_13;
          const float wsa_hi_02 = ws * as_hi_02;
          const float wsa_hi_13 = ws * as_hi_13;
          final_d[h][0] +=
              chunk_d_lo[h][0] * wsa_lo_02 + chunk_d_hi[h][0] * wsa_hi_02;
          final_d[h][1] +=
              chunk_d_lo[h][1] * wsa_lo_13 + chunk_d_hi[h][1] * wsa_hi_13;
          final_d[h][2] +=
              chunk_d_lo[h][2] * wsa_lo_02 + chunk_d_hi[h][2] * wsa_hi_02;
          final_d[h][3] +=
              chunk_d_lo[h][3] * wsa_lo_13 + chunk_d_hi[h][3] * wsa_hi_13;
          chunk_d_lo[h][0] = chunk_d_lo[h][1] = chunk_d_lo[h][2] =
              chunk_d_lo[h][3] = 0.f;
          chunk_d_hi[h][0] = chunk_d_hi[h][1] = chunk_d_hi[h][2] =
              chunk_d_hi[h][3] = 0.f;
        }
  #endif
      }

      // Launcher runs IN PARALLEL with the WGMMA above. It arms and
      // issues TMA for step s+1 into the OTHER slot ((s+1)%2),
      // following the same pattern as the priming block.
      //
      // Compiled out under MONO_PROFILE_SKIP_PREFETCH; the matching
      // compute-side waits on bar_{w,a}[next_slot] in the next
      // iteration are also compiled out so there is no spin-forever
      // deadlock.
      if (is_tma_launcher_thread<Dims>()) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH
        if (s + 1 < K_TILES_DOWN) {
          const std::uint32_t next_slot = (s + 1) & 1;
          const std::uint32_t next_k_start = (s + 1) * K_STEP_DOWN;

          // Next weight tile — HALVES back-to-back 128×128 TMAs,
          // same structure as the priming block.  Single bar_w arm
          // with the TOTAL tx_bytes drains every half in one wait on
          // the compute side.
          mbarrier_arrive_expect_tx(&shm->bar_w[next_slot],
                                    /*tx_bytes=*/DOWN_W_TX_BYTES_TOTAL);
    #pragma unroll
          for (std::uint32_t h = 0; h < DOWN_COL_HALVES; ++h) {
            W_element* dest_base = &shm->w_down_wgmma[next_slot][h * 128u][0];
            tma_load_down_wgmma_tile(down_weights_desc, /*expert_id=*/id,
                                     /*K=*/Dims::HIDDEN_STATES,
                                     /*base_col=*/base_col + h * 128u,
                                     /*k_start=*/next_k_start,
                                     /*dest_smem_ptr=*/(void*)dest_base,
                                     /*bar_smem_ptr=*/&shm->bar_w[next_slot]);
          }

          if constexpr (!use_cluster<Dims>::value) {
            // Next activation tile — only if any tokens route to this
            // expert.  The 8-TMA bulk helper delivers the full 1024 B
            // regardless of routed_count.
            //
            // Cluster path: no per-slot activation TMA — the next
            // K-step's activation tile is assembled cooperatively
            // from peer DSHM at the head of that iteration's loop
            // body (see "Cluster path: per-K-step DSHM staging" above).
            if (routed_count > 0u) {
              mbarrier_arrive_expect_tx(&shm->bar_a[next_slot],
                                        /*tx_bytes=*/1024u);
              tma_load_down_wgmma_activation_bulk(
                  down_activations_desc, /*k_start=*/next_k_start,
                  /*expert_slot_start=*/expert_start,
                  /*dest_smem_ptr=*/&shm->a_down_wgmma[next_slot][0][0][0],
                  /*bar_smem_ptr=*/&shm->bar_a[next_slot]);
            }
          }
        }
  #endif
      }
      if constexpr (!use_cluster<Dims>::value) {
        // Warp 8 (all lanes including the launcher): zero-fill the unused
        // tail of the NEXT slot when routed_count < 8. Lane 0 is gated
        // out inside the helper so TMA issue on lane 0 is not delayed.
        //
        // Cluster path: the under-T_TILE zero-fill is fused into the
        // per-K-step DSHM staging step at the head of each iteration
        // (R5.8, R10.4); no separate next-slot zero-fill is needed.
        if (warp == 8u && s + 1 < K_TILES_DOWN && routed_count < 8u) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH
          const std::uint32_t next_slot = (s + 1) & 1;
          zero_fill_unused_down_act_slots<Dims>(
              routed_count, &shm->a_down_wgmma[next_slot][0][0][0]);
  #endif
        }
      }

      // Activation scales are now loaded once per expert at the top of
      // the expert loop (see `moe_load_down_wgmma_activation_scales_for_expert`
      // call above), so no per-K-step scale prefetch is needed here.
      // The pipe drain below is kept to publish the (now-empty) scale
      // pipeline + the zero-fill stores from the launcher above.

      // Drain the scale cp.asyncs and make the zero-fill visible before
      // the next iteration's WGMMA reads the new slot.  Safe to keep
      // unconditional: the pipe drain is a no-op when SKIP_PREFETCH
      // has elided every `producer_commit`, and the __syncthreads() is
      // still needed to keep all warps aligned per iteration.
      cuda::pipeline_consumer_wait_prior<0>(pipe);
      __syncthreads();
    }  // end K-loop

    MONO_PHASE_TIMESTAMP_IF(t_down_after_expert0_kloop, e == down_group);

    // ── End-of-expert: write final_d → partial_result.down_out[DCT][8] ─
    //
    // Each half `h` contributes to output cols
    // `[base_col + h*128, base_col + h*128 + 127]`, which map to
    // `down_out` rows `[h*128 + wg_row_offset + warp_in_wg*16 + lane/4]`
    // (row_base) / `[row_base + 8]` for the d0..d3 halves.
    //
    // WG1 adds +64 to the row offset within its 128-row half because
    // WG1 owns output cols [h*128+64 .. h*128+127] within that half.
    if (is_calc) {
  #ifndef MONO_PROFILE_SKIP_CALC
      const std::uint32_t wg_row_offset = is_wg1 ? 64u : 0u;
      const std::uint32_t col_base = (lane % 4) * 2;
    #pragma unroll
      for (std::uint32_t h = 0; h < DOWN_COL_HALVES; ++h) {
        const std::uint32_t row_base =
            h * 128u + wg_row_offset + warp_in_wg * 16 + lane / 4;
        shm->partial_result.down_out[row_base + 0][col_base + 0] =
            final_d[h][0];
        shm->partial_result.down_out[row_base + 0][col_base + 1] =
            final_d[h][1];
        shm->partial_result.down_out[row_base + 8][col_base + 0] =
            final_d[h][2];
        shm->partial_result.down_out[row_base + 8][col_base + 1] =
            final_d[h][3];
      }
  #endif
    }
    __syncthreads();

    // ── Accumulate per-token contributions into out_accum ──────────────
    //
    // `down_out[col][rank]` holds the partial contribution for SHM
    // slot `rank` (intra-expert row inside the expert's contiguous
    // slab of `temp_fp8`), NOT for the natural logical token id.  We
    // therefore walk the (tok, k_in_topk) grid in `topk_ids_flat`,
    // filter by the current expert id, and derive the intra-expert
    // rank as `rank = sorted_slot[pair] - expert_start`.  This rank
    // is THE SAME across all DOWN_COL_TILE columns of a given token,
    // so we hoist the 8-iter top-K scan out of the (tok, col) inner
    // loop and cache `rank_for_tok[tok]` once per expert in SHM.
    //
    // 8 threads (one per token) compute the rank in parallel; the
    // remaining 376 threads idle for the same number of cycles they
    // were already spending on the redundant inner scan, so the
    // hoist is pure win.
    //
    // Iterating up to `batch_size * DOWN_COL_TILE` (not `Dims::BS *
    // DOWN_COL_TILE`) is safe: tokens >= batch_size are never written
    // in the final GM writeback below.
    //
    // Compiled out under MONO_PROFILE_SKIP_CALC: out_accum stays at
    // its prologue zero-fill, so Phase 5 reads zeros and produces
    // garbage output — matches the SKIP_CALC contract.
  #ifndef MONO_PROFILE_SKIP_CALC
    // Pre-compute rank for each token once.  Sentinel 0xFF means this
    // token does not route through the current expert (no contribution).
    if (thread_in_block < batch_size) {
      const unsigned tok = thread_in_block;
      uint8_t rank_val = 0xFFu;
      for (std::uint32_t k = 0; k < top_k; ++k) {
        if (shmem->topk_ids_flat[tok * MAX_TOPK + k] == (uint16_t)id) {
          const std::uint32_t pair = tok * top_k + k;
          rank_val = static_cast<uint8_t>(
              static_cast<std::uint32_t>(shm->sorted_slot[pair]) -
              expert_start);
          break;
        }
      }
      shm->rank_for_tok[tok] = rank_val;
    }
    __syncthreads();

    for (unsigned tok_col = thread_in_block;
         tok_col < batch_size * DOWN_COL_TILE; tok_col += blockDim.x) {
      const unsigned tok = tok_col / DOWN_COL_TILE;
      const unsigned col = tok_col % DOWN_COL_TILE;
      const uint8_t rank_u8 = shm->rank_for_tok[tok];
      if (rank_u8 != 0xFFu) {
        shm->out_accum[tok][col] += shm->partial_result.down_out[col][rank_u8];
      }
    }
  #endif
    __syncthreads();

    MONO_PHASE_TIMESTAMP_IF(t_down_after_expert0_accum, e == down_group);
  }  // end expert loop

  MONO_PHASE_TIMESTAMP(t_down_after_all_experts);

  // ── After all experts in this group: write out_accum → GM partial_out ─
  //
  // Per-expert-fused cluster path (R5.5 / R6.3, design §2 / §7.2):
  // cluster-path callers emit the writeback only after the LAST
  // per-expert iteration so all per-expert contributions accumulate in
  // the SHM `out_accum` slab first and are merged into a single HBM
  // partial.  On the byte-identical legacy non-cluster call site
  // `emit_epilogue == true` is the default so the writeback fires once
  // at the end of the all-experts loop, exactly as before (R11.4).
  if (emit_epilogue) {
    const std::uint32_t group_stride = Dims::BS * Dims::HIDDEN_STATES;
    float* gm_partial = spec->down_partial_out + down_group * group_stride;
    for (unsigned idx = thread_in_block; idx < batch_size * DOWN_COL_TILE;
         idx += blockDim.x) {
      const unsigned tok = idx / DOWN_COL_TILE;
      const unsigned col = idx % DOWN_COL_TILE;
      gm_partial[tok * Dims::HIDDEN_STATES + base_col + col] =
          shm->out_accum[tok][col];
    }
  }
}

}  // namespace moe_monokernel

#endif
