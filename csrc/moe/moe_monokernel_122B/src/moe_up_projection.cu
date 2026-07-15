
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
  #include "moe_debug.h"
  #include "moe_tma.h"

namespace moe_monokernel {

// ── Shared helper functions for the BS8 TMA+WGMMA up-projection ────────────
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

  // Only the first prefetch warp loads the scales — TILE = 2 * UP_SCALE_COLS
  // scalars total (32 for 35B's K=2048, 48 for 122B's K=3072).  A warp has
  // only 32 lanes, so when TILE > 32 (122B) a single `thread < TILE` guard
  // would silently drop elements [32, TILE).  Use a 32-strided loop so each
  // lane handles ceil(TILE/32) elements and the whole tile is covered.
  // Synchronous shared-memory writes are fine; we don't need async copy.
  if (warp == 0) {
    for (uint32_t i = thread; i < TILE; i += 32u) {
      uint32_t rb_local = i / COLS;  // 0 → gate half, 1 → up half
      uint32_t kb = i % COLS;
      uint32_t row = base_row_up + rb_local * Dims::N;
      uint32_t rb_global = row / Dims::BLOCK_SCALE_ROW;
      dest[i] =
          expert_scales_up[id * Dims::UP_SCALE_ROWS * Dims::UP_SCALE_COLS +
                           rb_global * Dims::UP_SCALE_COLS + kb];
    }
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
  constexpr uint32_t TILE = 2 * COLS;  // 32 for 35B (K=2048), 48 for 122B
  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_prefetch_warp<Dims>();

  // 32-strided loop so the whole TILE loads even when TILE > 32 (122B's
  // K=3072 → UP_SCALE_COLS=24 → TILE=48).  A single `thread < TILE` guard
  // would silently drop elements [32, TILE) — the upper half of the up
  // (d[2..3]) weight scales — leaving them garbage and making the up
  // WGMMA accumulators blow up to ±inf.
  if (warp == 0) {
    for (uint32_t i = thread; i < TILE; i += 32u) {
      uint32_t rb_local = i / COLS;  // 0 → gate half, 1 → up half
      uint32_t kb = i % COLS;
      uint32_t row = base_row_up + rb_local * Dims::N;
      uint32_t rb_global = row / Dims::BLOCK_SCALE_ROW;
      const S_element* src =
          &expert_scales_up[id * Dims::UP_SCALE_ROWS * Dims::UP_SCALE_COLS +
                            rb_global * Dims::UP_SCALE_COLS + kb];
      cp_async_cg_4(&dest[i], src);
    }
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

// ════════════════════════════════════════════════════════════════════════
// 122B FORK: moe_up_projection_BS8_122B_wgmma_tma
// ════════════════════════════════════════════════════════════════════════
//
// Specialized up-projection for the Qwen3.5-122B shape (N=1024, K=3072),
// where `MoECoreDims<Dims>::UP_COL_HALVES == 2`: each block owns TWO stacked
// 128-row WGMMA M-atoms instead of one, and the fp8 activation-quant block
// along the intermediate axis is 128 features (both atoms) instead of 64.
//
// TWO-TMA (non-interleaved) up-projection: each block owns 128 output
// features and fetches gate and up with TWO SEPARATE raw 128-row TMAs (no
// Python gate/up PAIR interleave).  Atom h=0 is the pure-gate weight tile,
// atom h=1 the pure-up tile; both cover the same feature block.  After the
// WGMMAs, register i of atom0 (gate) pairs with register i of atom1 (up)
// for the SAME output feature, so silu(gate)*up is a per-register combine
// across the two atoms.
//
// Differences from the 35B `moe_up_projection_BS8_allexperts_wgmma_tma`:
//   * `base_row_up = effective_bid * 128`; the block produces 128 gate +
//     128 up output features (vs 64 for the 35B single-atom block).
//   * Each K-substep runs the 4-WGMMA chain for BOTH atoms `h ∈ {0,1}`,
//     accumulating into per-atom `final_d[2][4]`.  Atom `h`'s weight tile
//     lives at SHM rows `[(kk*2 + h)*128 .. +128)` (kk-major, h-minor —
//     matches `W_WGMMA_M_TOTAL = 128 * K_SUBSTEPS_UP * UP_COL_HALVES`).  Its
//     TMA reads the RAW `[E, 2*N, K]` tensor: atom0 at gate rows
//     `[base_row_up, +128)`, atom1 at up rows `[N + base_row_up, +128)`
//     (`tma_load_up_wgmma_tile_raw`, `is_up = (h == 1)`).
//   * Weight TMA lookahead is 2-deep (`UP_W_SLOTS == 2`): calc waits on
//     `bar_w[s & 1]`, launcher arms `bar_w[(s+1) & 1]`.
//   * Scale-apply uses ONE weight scale per atom (gate-scale for h=0,
//     up-scale for h=1) since each atom is a pure half.
//   * The epilogue combines silu(gate)*up across atoms and stores by output
//     feature into `post_silu_scratch[0,128)`; the deferred SiLU+fp8
//     writeback reduce-max spans those 128 features for the block-128 quant.
//   * Both atoms share the same gate/up weight-scale blocks (their gate
//     rows fall in the same BLOCK_SCALE_ROW=128 block, rb_global == bid),
//     so the `up_scale` buffer layout is unchanged.
//
// This fork targets the DEFER epilogue only — the 122B Dims sets
// `USE_PAIR_LAYOUT = true` (the SHM byte layout is shared; only the GM-row
// fetch and register combine differ from the interleaved Pair_Layout).
template <typename Dims>
__device__ inline void moe_up_projection_BS8_122B_wgmma_tma(
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
  static_assert(CoreDims::UP_COL_HALVES == 2u,
                "moe_up_projection_BS8_122B_wgmma_tma is specialized for "
                "UP_COL_HALVES == 2 (the Qwen3.5-122B shape).");

  (void)activations_in;
  (void)expert_weights_up;

  constexpr uint32_t W_UP_M = CoreDims::W_UP_TILE_WGMMA;     // 128 (per atom)
  constexpr uint32_t K_STEP_WGMMA = CoreDims::K_STEP_WGMMA;  // 128
  constexpr uint32_t K_STEP = CoreDims::K_STEP_UP;           // 128
  constexpr uint32_t K_SUBSTEPS = CoreDims::K_SUBSTEPS_UP;   // 1
  constexpr uint32_t K_TILES = CoreDims::K_TILES_UP;         // 3072/128 = 24
  constexpr uint32_t WGMMAS_PER_SUBSTEP = CoreDims::WGMMAS_PER_STEP;  // 4
  constexpr uint32_t UP_SCALE_COLS = Dims::UP_SCALE_COLS;             // 24
  constexpr uint32_t HALVES = CoreDims::UP_COL_HALVES;                // 2

  // A operand descriptor strides (128×128 Major::K B128 swizzle).
  constexpr uint64_t A_LBO = 16ULL;
  constexpr uint64_t A_SBO = 1024ULL;
  constexpr uint32_t A_SWIZZLE = 1u;
  // B operand stride: steps over the padded 9th token-row of each kc atom.
  constexpr uint64_t B_LBO =
      static_cast<uint64_t>(
          MoE_SHM<Dims>::U::TinyDataWGMMA_TMA::FP8_ACT_T_TILE_PADDED) *
      static_cast<uint64_t>(
          MoE_SHM<Dims>::U::TinyDataWGMMA_TMA::FP8_ACT_K_CHUNK);  // 144 B
  constexpr uint64_t B_SBO = B_LBO;

  const unsigned thread_in_block = threadIdx.x;
  const unsigned warp = thread_in_block / 32;  // 0..11
  const unsigned lane = thread_in_block & 31;
  const bool is_wg1 = (warp >= 4 && warp < 8);
  const bool is_calc = (warp < 8);
  const unsigned warp_in_wg = warp & 3;  // 0..3 within each WG

  auto* shm = &shmem->u.tiny_wgmma_tma;

  const unsigned effective_bid =
      (up_block_idx == 0xffffffffu) ? blockIdx.x : up_block_idx;
  // Each block owns 256 M rows = 2 atoms × (2 WG stripes × 64 rows).
  const unsigned base_row_up = effective_bid * (HALVES * W_UP_M / 2);  // *128
  const std::uint32_t expert_count = shmem->expert_count;

  // Per-atom WGMMA accumulators: final_d[h][0..3] for atoms h ∈ {0,1}.
  float chunk_d0 = 0.f, chunk_d1 = 0.f, chunk_d2 = 0.f, chunk_d3 = 0.f;
  float final_d[HALVES][4];
  #pragma unroll
  for (uint32_t h = 0; h < HALVES; ++h) {
    final_d[h][0] = final_d[h][1] = final_d[h][2] = final_d[h][3] = 0.f;
  }

  // Per-arm weight TMA tx_bytes: one 128×128 fp8 atom (16 KB) × K_SUBSTEPS ×
  // HALVES (both atoms loaded per slot).
  constexpr uint32_t UP_W_TX_BYTES_PER_ATOM = 16384u;
  constexpr uint32_t UP_W_TX_BYTES_TOTAL =
      UP_W_TX_BYTES_PER_ATOM * K_SUBSTEPS * HALVES;  // 32 KB

  // Helper: issue the (K_SUBSTEPS × HALVES) weight TMAs for one slot of a
  // given expert / outer-K iter into `bar`.  Caller pre-arms `bar` with
  // UP_W_TX_BYTES_TOTAL.  `k_iter` is the outer K-step index.
  //
  // TWO-TMA (non-interleaved) layout: atom h=0 fetches the RAW 128-row gate
  // tile, atom h=1 fetches the RAW 128-row up tile, both for the SAME output
  // feature block `[base_row_up, base_row_up + 128)`.  No Python gate/up
  // pre-interleave is applied — the raw `[E, 2*N, K]` tensor is read directly
  // (gate at rows [0,N), up at rows [N,2N) per expert).  atom0 lands in SHM
  // rows [0,128) (consumed as the gate WGMMA operand), atom1 in rows
  // [128,256) (the up operand); the SWIZZLE_128B TMA applies the core-matrix
  // XOR swizzle at write time, so the WGMMA A-descriptor is unchanged.
  auto tma_slot = [&](uint32_t slot, uint32_t eid, uint32_t k_iter) {
  #pragma unroll
    for (uint32_t kk = 0; kk < K_SUBSTEPS; ++kk) {
  #pragma unroll
      for (uint32_t h = 0; h < HALVES; ++h) {
        const uint32_t atom = kk * HALVES + h;  // SHM row-atom index
        tma_load_up_wgmma_tile_raw(
            up_weights_desc, /*expert_id=*/eid, /*N=*/Dims::N,
            /*row_in_half=*/base_row_up, /*is_up=*/(h == 1u),
            /*k_start=*/k_iter * K_STEP + kk * K_STEP_WGMMA,
            /*dest_slot=*/&shm->w_wgmma[slot][atom * W_UP_M][0],
            /*bar=*/&shm->bar_w[slot]);
      }
    }
  };

  // ── Pre-loop: arm ONLY bar_w[0] (iter 0) for expert_start ───────────────
  //
  // With 2 buffers the lookahead is 1-deep: at iter s the launcher arms
  // slot `(s+1)&1` for iter s+1.  The pre-loop primes only slot 0 (iter 0);
  // slot 1 (iter 1) is armed by the s==0 launcher below.  Priming slot 1
  // here too would double-arm it against that s==0 arm.
  if (is_tma_launcher_thread<Dims>() && expert_start < expert_count) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH_UP
    const uint32_t first_id = shmem->experts[expert_start].id;
    static_assert(K_TILES >= 2 && K_TILES % 2 == 0,
                  "122B up-proj 2-slot ping-pong requires K_TILES even and "
                  ">= 2 so per-slot arm/wait counts balance across experts "
                  "and the cross-expert stitch lands on slot 0.");
    mbarrier_arrive_expect_tx(&shm->bar_w[0], UP_W_TX_BYTES_TOTAL);
    tma_slot(0u, first_id, 0u);
  #endif
  }

  uint32_t prev_id_for_writeback = 0;
  bool has_pending_writeback = false;
  (void)prev_id_for_writeback;
  (void)has_pending_writeback;

  uint32_t cur_scale_slot = 0;
  {
  #ifndef MONO_PROFILE_SKIP_PREFETCH_UP
    if (is_prefetch_warp<Dims>() && expert_start < expert_count) {
      moe_prefetch_up_scale_for_row_async<Dims>(expert_scales_up,
                                                shmem->experts[expert_start].id,
                                                base_row_up, shm->up_scale[0]);
      cp_async_commit_group();
    }
  #endif
  }

  // ── Expert loop ──────────────────────────────────────────────────────────
  for (uint32_t e = expert_start; e < expert_count; e += expert_stride) {
    const uint32_t id = shmem->experts[e].id;
    const bool has_next_e = (e + expert_stride < expert_count);
    const uint32_t next_id =
        has_next_e ? shmem->experts[e + expert_stride].id : 0u;

    // 2-slot parity state.
    uint32_t parity_w[2] = {0, 0};

  #pragma unroll
    for (uint32_t h = 0; h < HALVES; ++h) {
      final_d[h][0] = final_d[h][1] = final_d[h][2] = final_d[h][3] = 0.f;
    }

    // Drain this expert's prefetched scale; prefetch the next expert's.
    {
  #ifndef MONO_PROFILE_SKIP_PREFETCH_UP
      if (is_prefetch_warp<Dims>()) {
        cp_async_wait_group<0>();
        if (has_next_e) {
          moe_prefetch_up_scale_for_row_async<Dims>(
              expert_scales_up, next_id, base_row_up,
              shm->up_scale[cur_scale_slot ^ 1u]);
          cp_async_commit_group();
        }
      }
  #endif
    }

    // Per-expert routing-weight cache populate (8 calc threads, vectorized).
    {
  #ifndef MONO_PROFILE_SKIP_CALC_UP
      constexpr std::uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;
      if (thread_in_block < batch_size && MAX_TOPK == 8u) {
        const std::uint32_t tok = thread_in_block;
        const uint16_t* idb = &shmem->topk_ids_flat[tok * MAX_TOPK];
        uint16_t ids[8];
        *reinterpret_cast<uint2*>(&ids[0]) =
            *reinterpret_cast<const uint2*>(idb);
        *reinterpret_cast<uint2*>(&ids[4]) =
            *reinterpret_cast<const uint2*>(idb + 4);
        const S_element* wb = &shmem->topk_weights_flat[tok * MAX_TOPK];
        float w[8];
    #pragma unroll
        for (std::uint32_t k = 0; k < 8u; ++k) w[k] = wb[k];
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

    // Publish up_scale + rank cache.
    __syncthreads();

    // ── Main K-loop (2-slot ping-pong) ─────────────────────────────────────
    for (uint32_t s = 0; s < K_TILES; ++s) {
      const uint32_t cur_slot = s & 1u;
      const uint32_t next_slot = (s + 1u) & 1u;

      // ── COMPUTE half: 2 atoms × K_SUBSTEPS × 4 WGMMAs ────────────────────
      if (is_calc) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH_UP
        while (!mbarrier_try_wait_parity(&shm->bar_w[cur_slot],
                                         parity_w[cur_slot])) {
        }
        parity_w[cur_slot] ^= 1;
  #endif

  #ifndef MONO_PROFILE_SKIP_CALC_UP
        const void* a_slot_base = (const void*)&shm->w_wgmma[cur_slot][0][0];
        const uint32_t wg_offset_bytes = is_wg1 ? 8192u : 0u;
        constexpr uint32_t ATOM_W_BYTES = 16384u;  // 128 rows × 128 K-bytes
        constexpr uint32_t A_K_STRIDE = 2u * static_cast<uint32_t>(A_LBO);

    #pragma unroll
        for (uint32_t kk = 0; kk < K_SUBSTEPS; ++kk) {
          const uint32_t kblk = s * K_SUBSTEPS + kk;
    #pragma unroll
          for (uint32_t h = 0; h < HALVES; ++h) {
            wgmma_fence();
            const uint32_t atom = kk * HALVES + h;
            const void* a_kk_base =
                (const void*)((const char*)a_slot_base + atom * ATOM_W_BYTES +
                              wg_offset_bytes);
            chunk_d0 = chunk_d1 = chunk_d2 = chunk_d3 = 0.f;
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

            // Scale-apply at the K=128 boundary.  TWO-TMA layout: atom h=0
            // is the pure-gate tile, atom h=1 the pure-up tile, so each
            // atom's 4 accumulators share a SINGLE weight-scale (gate-scale
            // for h=0, up-scale for h=1).  The 4 regs span 2 features
            // (rows R and R+8) × 2 tokens; the activation scale still picks
            // d0/d2 → tok_02 and d1/d3 → tok_13.  Both atoms read the same
            // gate/up weight-scale block (rb_global == effective_bid).
            const uint32_t tok_02 = (lane % 4) * 2;
            const uint32_t tok_13 = tok_02 + 1;
            const float as_02 = shmem->act_scale[kblk][tok_02];
            const float as_13 = shmem->act_scale[kblk][tok_13];
            const S_element* ws_buf = shm->up_scale[cur_scale_slot];
            const float ws =
                (h == 0u) ? ws_buf[kblk + 0u] : ws_buf[kblk + UP_SCALE_COLS];
            final_d[h][0] += chunk_d0 * ws * as_02;
            final_d[h][1] += chunk_d1 * ws * as_13;
            final_d[h][2] += chunk_d2 * ws * as_02;
            final_d[h][3] += chunk_d3 * ws * as_13;
          }
        }
  #endif
      }

      // ── Launcher: arm + TMA bar_w[next_slot] (2-deep) ────────────────────
      if (is_tma_launcher_thread<Dims>()) {
  #ifndef MONO_PROFILE_SKIP_PREFETCH_UP
        const bool has_next_s = (s + 1u < K_TILES);
        if (has_next_s) {
          // Intra-expert: fetch iter (s+1).
          mbarrier_arrive_expect_tx(&shm->bar_w[next_slot],
                                    UP_W_TX_BYTES_TOTAL);
          tma_slot(next_slot, id, s + 1u);
        } else if (has_next_e) {
          // Cross-expert stitch: fetch next expert's iter 0.
          mbarrier_arrive_expect_tx(&shm->bar_w[next_slot],
                                    UP_W_TX_BYTES_TOTAL);
          tma_slot(next_slot, next_id, 0u);
        }
  #endif
      }

      // ── Deferred SiLU + fp8 writeback for the PREVIOUS expert ────────────
      // 8 tokens split across iters 0 (tok 0..3) and 1 (tok 4..7), 4 PF
      // warps (8..11), one token each.  Reduce-max spans 128 features
      // (4 values/lane: atom0 WG0/WG1 + atom1 WG0/WG1).
      static_assert(K_TILES >= 2,
                    "Deferred up-proj writeback requires K_TILES >= 2.");
      const bool defer_iter = (s == 0u || s == 1u);
      if (defer_iter && has_pending_writeback && is_prefetch_warp<Dims>()) {
  #ifndef MONO_PROFILE_SKIP_CALC_UP
        const unsigned pf_warp = warp - CoreDims::CALC_WARP_COUNT;  // 0..3
        const uint32_t tok = pf_warp + (s == 0u ? 0u : 4u);         // 0..7
        const uint32_t col = lane;                                  // 0..31

        bool store_local = false;
        std::uint32_t dest_row_local = 0;
        if (tok < batch_size) {
          const uint8_t k = shm->up_rank_for_tok_prev[tok];
          if (k != 0xFFu) {
            store_local = true;
            dest_row_local = shm->sorted_slot[tok * top_k + k];
          }
        }

        // 4 post_silu_scratch reads.  TWO-TMA layout: post_silu_scratch is
        // indexed DIRECTLY by base-relative output feature [0,128) (the
        // epilogue's silu(gate)*up combine stores to feature f = f_lo/f_hi),
        // so this lane reads features {col, col+32, col+64, col+96} — exactly
        // oc[i] - base_row_up.
        const float pv[4] = {
            shm->partial_result.post_silu_scratch[col][tok],
            shm->partial_result.post_silu_scratch[col + 32][tok],
            shm->partial_result.post_silu_scratch[col + 64][tok],
            shm->partial_result.post_silu_scratch[col + 96][tok]};
        // Output feature columns for each value.
        const std::uint32_t oc[4] = {base_row_up + col, base_row_up + 32 + col,
                                     base_row_up + 64 + col,
                                     base_row_up + 96 + col};
        bool wr[4];
        float v[4];
    #pragma unroll
        for (int i = 0; i < 4; ++i) {
          wr[i] = store_local && (oc[i] < Dims::N);
          v[i] = wr[i] ? pv[i] : 0.f;
        }

        // Block-128 reduce-max: max over this lane's 4 values, then across
        // the warp's 32 lanes = 128 features.
        float local_max = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])),
                                fmaxf(fabsf(v[2]), fabsf(v[3])));
        float block_max = warp_reduce_max_float(local_max);
        if (block_max < __FLT_MIN__) block_max = 1.0f;
        constexpr float FP8_MAX = 448.0f;
        constexpr float FP8_MAX_INV = 1.0f / 448.0f;
        const float block_scale = block_max * FP8_MAX_INV;
        const float inv_scale = FP8_MAX / block_max;

        if (store_local && tok < batch_size) {
    #pragma unroll
          for (int i = 0; i < 4; ++i) {
            if (wr[i]) {
              spec->temp_fp8[dest_row_local * Dims::N + oc[i]] =
                  (AQ_element)(v[i] * inv_scale);
            }
          }
          if (lane == 0) {
            constexpr std::uint32_t SCALE_COLS =
                MoEGemmSpec<Dims>::TEMP_ACT_SCALE_COLS;
            spec->temp_act_scale[dest_row_local * SCALE_COLS + effective_bid] =
                block_scale;
          }
        }
  #endif
      }

      // Inter-iteration sync (ordering for the launcher's next-slot arm).
      __syncthreads();
    }  // end K-loop

    // ── End-of-expert calc epilogue: combine + store BOTH atoms ────────────
    {
      if (is_calc) {
  #ifndef MONO_PROFILE_SKIP_CALC_UP
        const std::uint32_t tok_even = (lane % 4) * 2;
        const std::uint32_t tok_odd = tok_even + 1;

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

    #ifdef DEBUG_MOE_PRINT
        // Raw WGMMA accumulators for the FIRST expert, block 0, warp 0
        // lane 0 (tok_even=0).  TWO-TMA layout: atom0 = pure GATE, atom1 =
        // pure UP, both for features {fR, fR+8}; register i pairs gate
        // (final_d[0][i]) with up (final_d[1][i]) for the same feature.
        // This is the rawest pre-SiLU signal — compare against the Python
        // reference GEMM1 for (tok0, experts[expert_start].id) BEFORE the
        // SiLU / rw / quant steps to localize a register↔feature bug.
        if (blockIdx.x == 0 && threadIdx.x == 0 && e == expert_start) {
          printf(
              "[MONO_DBG] UP-WGMMA blk0 e0 id=%u tok0: "
              "fR{gate=%.5f up=%.5f} fR+8{gate=%.5f up=%.5f} "
              "base_row_up=%u\n",
              id, final_d[0][0], final_d[1][0], final_d[0][2], final_d[1][2],
              base_row_up);
        }
    #endif
        // TWO-TMA combine: silu(gate)*up across atoms, register-by-register.
        // final_d[0][i] = gate, final_d[1][i] = up for the same (feature,
        // token).  Each thread owns two output features fR and fR+8 (× 2
        // tokens), where fR = (is_wg1?64:0) + warp_in_wg*16 + lane/4 spans
        // the block's full [0,128) feature tile across the 8 calc warps.
        const float* gd = final_d[0];  // gate (atom0)
        const float* ud = final_d[1];  // up   (atom1)
        float v_lo_even =
            __fdividef(rw_even * ud[0] * gd[0], 1.0f + __expf(-gd[0]));
        float v_lo_odd =
            __fdividef(rw_odd * ud[1] * gd[1], 1.0f + __expf(-gd[1]));
        float v_hi_even =
            __fdividef(rw_even * ud[2] * gd[2], 1.0f + __expf(-gd[2]));
        float v_hi_odd =
            __fdividef(rw_odd * ud[3] * gd[3], 1.0f + __expf(-gd[3]));
        if (!store_even) {
          v_lo_even = 0.f;
          v_hi_even = 0.f;
        }
        if (!store_odd) {
          v_lo_odd = 0.f;
          v_hi_odd = 0.f;
        }
        const uint32_t f_lo = (is_wg1 ? 64u : 0u) + warp_in_wg * 16u + lane / 4;
        const uint32_t f_hi = f_lo + 8u;
        shm->partial_result.post_silu_scratch[f_lo][tok_even] = v_lo_even;
        shm->partial_result.post_silu_scratch[f_lo][tok_odd] = v_lo_odd;
        shm->partial_result.post_silu_scratch[f_hi][tok_even] = v_hi_even;
        shm->partial_result.post_silu_scratch[f_hi][tok_odd] = v_hi_odd;

        // Snapshot rank cache for the next expert's PF body.
        if (warp == 0 && lane < Dims::BS) {
          shm->up_rank_for_tok_prev[lane] = shm->up_rank_for_tok[lane];
        }
  #endif
      }
      prev_id_for_writeback = id;
      has_pending_writeback = true;
    }

    __syncthreads();
    cur_scale_slot ^= 1u;
  }  // end expert loop

  // ── Post-loop drain for the LAST expert (calc warps, 128-feat reduce) ────
  if (has_pending_writeback && is_calc) {
  #ifndef MONO_PROFILE_SKIP_CALC_UP
    const uint32_t tok = warp;  // 0..7
    const uint32_t col = lane;  // 0..31

    bool store_local = false;
    std::uint32_t dest_row_local = 0;
    if (tok < batch_size) {
      const uint8_t k = shm->up_rank_for_tok_prev[tok];
      if (k != 0xFFu) {
        store_local = true;
        dest_row_local = shm->sorted_slot[tok * top_k + k];
      }
    }

    // TWO-TMA layout: post_silu_scratch indexed by base-relative feature
    // [0,128); read features {col, col+32, col+64, col+96} (= oc[i]-base).
    const float pv[4] = {shm->partial_result.post_silu_scratch[col][tok],
                         shm->partial_result.post_silu_scratch[col + 32][tok],
                         shm->partial_result.post_silu_scratch[col + 64][tok],
                         shm->partial_result.post_silu_scratch[col + 96][tok]};
    const std::uint32_t oc[4] = {base_row_up + col, base_row_up + 32 + col,
                                 base_row_up + 64 + col,
                                 base_row_up + 96 + col};
    bool wr[4];
    float v[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      wr[i] = store_local && (oc[i] < Dims::N);
      v[i] = wr[i] ? pv[i] : 0.f;
    }

    float local_max =
        fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    float block_max = warp_reduce_max_float(local_max);
    if (block_max < __FLT_MIN__) block_max = 1.0f;
    constexpr float FP8_MAX = 448.0f;
    constexpr float FP8_MAX_INV = 1.0f / 448.0f;
    const float block_scale = block_max * FP8_MAX_INV;
    const float inv_scale = FP8_MAX / block_max;

    if (store_local && tok < batch_size) {
    #pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (wr[i]) {
          spec->temp_fp8[dest_row_local * Dims::N + oc[i]] =
              (AQ_element)(v[i] * inv_scale);
        }
      }
      if (lane == 0) {
        constexpr std::uint32_t SCALE_COLS =
            MoEGemmSpec<Dims>::TEMP_ACT_SCALE_COLS;
        spec->temp_act_scale[dest_row_local * SCALE_COLS + effective_bid] =
            block_scale;
      }
    }
  #endif
  }
}

}  // namespace moe_monokernel

#endif
