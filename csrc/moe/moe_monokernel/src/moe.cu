/**
 * This is the main file of the MoE monokernel for Qwen3-Coder (top-K path).
 * It is designed so that you just need to build this file. It includes all
 * relevant implementations. For documentation of the main entry function
 * moe_kernel_topk, see moe_interface.h
 */

#include <cstdint>

#include "moe_interface.h"

#define INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
#include "moe_debug.h"
#include "moe_down_projection.cu"
#include "moe_grid_barrier.h"
#include "moe_internal.h"
#include "moe_prepare.cu"
#include "moe_scale_inputs.cu"
#include "moe_tma.h"
#include "moe_up_projection.cu"
#include "moe_routing.cu"
#undef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION

namespace moe_monokernel {

/**
 * @brief Top-K MoE kernel — split-phase WGMMA path for BS <= 8.
 *
 * Uses the v1 dual-warpgroup K=128 streaming WGMMA pipeline for both
 * up- and down-projections.
 *
 * Pipeline:
 *   Phase 1: routing + topK (running in parallel with a greedy TMA
 *            prefetch of the k_start=0 bf16 activation tile on every
 *            block — activations are expert-independent)
 *   Phase 2: no-op (streaming WGMMA up-proj does its own priming)
 *   Phase 3: up-proj — streaming WGMMA with on-the-fly bf16→fp8
 *            quantize → SiLU → write bf16 to spec->temp_bf16
 *   grid.sync()
 *   Phase 4: down-proj — streaming WGMMA; each block writes a
 *            per-expert-group fp32 partial sum to
 *            spec->down_partial_out[group][tok][col]
 *   grid.sync()
 *   Phase 5: reduce across groups, write bf16 activations_out.
 *
 * 2 grid syncs total.
 */
template <typename Dims>
__device__ void moe_kernel_topk_BS8(
    const A_element* __restrict__ activations_in, std::uint32_t batch_size,
    const __nv_bfloat16* __restrict__ router_logits,
    const W_element* __restrict__ expert_weights_up,
    const S_element* __restrict__ expert_scales_up,
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down,
    R_element* __restrict__ activations_out, uint32_t top_k,
    ScoringFunc scoring_func, bool renormalize,
    MoEGemmSpec<Dims>* __restrict__ spec, MoE_SHM<Dims>* __restrict__ shmem,
    CUtensorMap const& up_weights_desc, CUtensorMap const& activations_desc,
    CUtensorMap const& down_weights_desc,
    CUtensorMap const& down_activations_desc,
    uint32_t* __restrict__ grid_counters, uint32_t& grid_phase,
    uint32_t* __restrict__ expert_counters, uint32_t& expert_phase,
    uint32_t* __restrict__ colstripe_counters, uint32_t& colstripe_phase) {
  static_assert(Dims::BS <= 8);
  static_assert(use_wgmma<Dims>::value,
                "BS8 path requires the WGMMA configuration (use_wgmma).");
  static_assert(use_tma<Dims>::value, "BS8 path requires USE_TMA");
  using CoreDims = MoECoreDims<Dims>;

  MONO_PHASE_TIMESTAMP(t_start);

  // ── Phase 1: routing (topK) + greedy activation prefetch ───────────────
  // Phase 1 runs routing (topK / prepare_moe_topk) which writes
  // shmem->experts and shmem->topk_ids_flat that later phases depend on.
  //
  // In parallel with routing, the launcher thread greedily fires the
  // bf16 activation TMA for k_start=0.  Activations are expert-independent
  // (the descriptor covers all tokens × all K), so every block can start
  // fetching immediately — without waiting for routing to determine
  // expert_count.  Blocks that later turn out to be outside the feasible
  // range (up_group >= shmem->expert_count) simply leave their 2 KB tile
  // unused and bar_a[0] sits at parity 1; no hang, no corruption, just
  // one wasted 2 KB fetch that L2 coalesces across SMs.
  //
  // Correctness requirements:
  //   * mbarriers must be initialized before any `arrive_expect_tx`.
  //     The init and the arm run on the same launcher thread, so
  //     program order guarantees local visibility.  The
  //     `fence_mbarrier_init_release_cluster()` between them (and the
  //     block-wide `__syncthreads()` below) publishes the init to every
  //     consumer warp before it waits on `bar_a[0]`.
  //   * The hoisted Step A handles the e==expert_start iteration only;
  //     the up-proj helper is called with `external_priming = true` and
  //     never issues its own bf16_in[0] TMA.  For experts 1..N inside
  //     a group, Step A is issued at the tail of the previous expert's
  //     iteration (in parallel with SiLU writeback) — see the end of
  //     `moe_up_projection_BS8_allexperts_wgmma_tma`'s expert loop.
  auto* u_tma = &shmem->u.tiny_wgmma_tma;
  if (is_tma_launcher_thread<Dims>()) {
    // mbarrier inits are kept regardless of the profile flags: they are
    // cheap SHM writes and make the SHM state well-defined even when
    // SKIP_PREFETCH elides every arrive/TMA below.  The K-loop waits
    // inside the up-proj helper are themselves gated on SKIP_PREFETCH,
    // so a consumer will never block on an uninitialized parity.
    mbarrier_init(&u_tma->bar_w[0], 1u);
    mbarrier_init(&u_tma->bar_w[1], 1u);
    mbarrier_init(&u_tma->bar_a[0], 1u);
    mbarrier_init(&u_tma->bar_a[1], 1u);
    fence_mbarrier_init_release_cluster();

#ifndef MONO_PROFILE_SKIP_PREFETCH
    // Greedy Step A: activations are expert-independent, so fire the
    // k_start=0 TMA now — in parallel with routing — instead of waiting
    // for Phase 2.  The arm happens on the same thread that just did the
    // init, so no fence is required between them.
    //
    // Compiled out under MONO_PROFILE_SKIP_PREFETCH; the matching calc-
    // warp wait on bar_a[0] inside the up-proj helper is also compiled
    // out so there is no spin-forever deadlock.  The calc warps read
    // garbage from the still-uninitialized `bf16_in[0]` slot and the
    // kernel produces junk output — useful only for timing.
    mbarrier_arrive_expect_tx(&u_tma->bar_a[0], /*tx_bytes=*/2048u);
    tma_load_bf16_input_tile(activations_desc, /*k_start=*/0u,
                             &u_tma->bf16_in[0][0][0], &u_tma->bar_a[0]);
#endif
  }
  if (is_prefetch_warp<Dims>()) {
    // WGMMA path: prefetch warps are idle here.  The streaming pipeline's
    // first bf16 tile is already in flight (greedy TMA above) and the
    // rest of priming runs inside the up-proj helper.
  } else {
    // Routing is intentionally NOT guarded by MONO_PROFILE_SKIP_CALC:
    // `shmem->expert_count` / `shmem->experts[e].id` drive the helper's
    // expert loop bounds and an uninitialized expert_count could be
    // anything from 0 to 2^32 (runaway loop).  The BS64 path handles
    // MONO_PROFILE_SKIP_CALC the same way — `topK_BS64` and
    // `prepare_moe_topk_BSx_Ey` run regardless; only the per-expert
    // QUANT / WGMMA / writeback work is compiled out.
    topK_BS8<Dims>(top_k, scoring_func, renormalize, router_logits, batch_size,
                   shmem);
    MONO_PHASE_TIMESTAMP(t_after_topk);
    sync_calc_threads<Dims>();
    MONO_PHASE_TIMESTAMP(t_after_sync_calc);
    prepare_moe_topk_BS8<Dims>(batch_size, top_k, shmem, spec);
  }
  __syncthreads();

  MONO_PHASE_TIMESTAMP(t_after_routing);

  // ── Phase 2: setup up-projection group mapping ──────────────────────────
  // GRID=128 design, expert-group parallelism (WGMMA path):
  //   UP_GRID = 2*N / W_UP_TILE_EFFECTIVE blocks cover the full 2*N weight
  //   rows for one expert.  With GRID_SIZE=128, we run
  //   UP_GROUPS = GRID_SIZE / UP_GRID groups processing DIFFERENT experts
  //   in parallel, with each group's blocks indexed by
  //   blockIdx.x % UP_GRID.
  //
  //   WGMMA v1 path (W_UP_TILE_EFFECTIVE=128): UP_GRID = 2*N/128,
  //   UP_GROUPS = 128 / UP_GRID.  For N=512: UP_GRID=8, UP_GROUPS=16
  //   (sixteen experts processed in parallel per grid).
  constexpr std::uint32_t UP_GRID = 2 * Dims::N / CoreDims::W_UP_TILE_EFFECTIVE;
  constexpr std::uint32_t UP_GROUPS = Dims::KernelConfig::GRID_SIZE / UP_GRID;
  static_assert(Dims::KernelConfig::GRID_SIZE % UP_GRID == 0,
                "GRID_SIZE must be a multiple of UP_GRID.");
  // UP_GROUPS = number of expert groups processed in parallel per grid.
  // Each token contributes at most `top_k` virtual_row slots in
  // spec->temp_bf16, so at most `top_k` blocks write to any given token
  // (one per expert in the token's top-K list).  Blocks processing
  // experts NOT in a token's top-K silently skip the write.  Therefore
  // UP_GROUPS has no upper bound from a correctness standpoint — only
  // a wasted-work concern (higher UP_GROUPS ⇒ more WGMMAs whose
  // experts aren't in any active token's top-K list).
  //
  // We cap at UP_GROUPS <= NUM_EXPERTS (trivially always true) and
  // leave perf tuning to the caller's choice of GRID_SIZE / UP_GRID.
  static_assert(UP_GROUPS <= Dims::NUM_EXPERTS,
                "UP_GROUPS cannot exceed the total number of experts.");
  const std::uint32_t up_group = blockIdx.x / UP_GRID;
  const std::uint32_t up_block_idx = blockIdx.x % UP_GRID;
  const bool in_up = (up_group < UP_GROUPS);

  // Phase 2 is a no-op for the v1 streaming WGMMA pipeline.  Phase 3's
  // moe_up_projection_BS8_allexperts_wgmma_tma does its own priming:
  //   (1) bf16_in[0] from global (HOISTED to Phase 1 above — fired
  //       greedily on every block in parallel with routing; the helper
  //       skips its internal first-expert Step A because we pass
  //       external_priming=true).
  //   (2) prefetch w[0] and bf16_in[1] || quantize bf16_in[0] → fp8[0]
  // No __syncthreads() here: the Phase-1 sync above already published
  // both the barrier init and shmem->expert_count, and computing
  // up_group / up_block_idx / in_up is pure register work.

  // ── Phase 3: Up-projection — expert groups in parallel ────────────────
  // Group `g` (blocks [g*UP_GRID, (g+1)*UP_GRID)) iterates experts starting
  // at index `g`, stepping by UP_GROUPS. Each group writes to DIFFERENT
  // virtual_row slots of spec->temp_bf16 (because each expert has its own
  // k index within a token's top-K list), so the groups never have a
  // write conflict.
  //
  // The BS8 path is TMA+WGMMA only; the kernel asserts
  // `use_wgmma<Dims>::value` and `use_tma<Dims>::value` at the top of
  // this function, so dispatch is unconditional.
  if (in_up && up_group < shmem->expert_count) {
    moe_up_projection_BS8_allexperts_wgmma_tma<Dims>(
        activations_in, expert_weights_up, expert_scales_up, top_k, batch_size,
        spec, shmem, up_weights_desc, activations_desc, up_block_idx,
        /*expert_start=*/up_group,
        /*expert_stride=*/UP_GROUPS,
        /*external_priming=*/true);
  }

  MONO_PHASE_TIMESTAMP(t_after_up);

  // ── Site #2 — Expert-local barrier (Phase 2b) ────────────────────────
  //
  // Phase 2a aligned `DOWN_GROUPS == UP_GROUPS` so the producer-set
  // (8 blocks with `up_group == g` writing `spec->temp_fp8` rows for
  // expert group `g`) is identical to the consumer-set (same 8 blocks,
  // now reading those rows in Phase 4 as `down_group == g`).  An
  // `expert_barrier` with `arrival_count = UP_GRID = 8` and `id = up_group`
  // is therefore sufficient: the 8 blocks rendezvous on one of
  // `UP_GROUPS = 16` independent expert-keyed Counter_Pairs, reducing
  // per-barrier atomic contention from 128 → 8 and allowing 16 expert
  // groups to sync concurrently (Design "Site #2 Phase 2b change
  // summary", Requirements 9.6, 9.8).
  //
  // `in_up` is always true in the GRID_SIZE=128, UP_GRID=8, UP_GROUPS=16
  // configuration (every block maps to a valid up_group), but the gate
  // is kept defensively so a future config with UP_GROUPS < GRID_SIZE /
  // UP_GRID won't silently deadlock.
  if (in_up) {
    moe_monokernel::expert_barrier(expert_counters,
                                   /*expert_id=*/up_group,
                                   /*arrival_count=*/UP_GRID,
                                   /*seed_blockidx=*/up_group * UP_GRID,
                                   expert_phase);
  }

  MONO_PHASE_TIMESTAMP(t_after_barrier2);

  // ── Phase 4 (WGMMA): dual-WG streaming down-projection ────────────────
  // Each block owns DOWN_COL_TILE=128 output cols; blocks partition
  // into DOWN_GROUPS expert groups × DOWN_GRID col-blocks.  Each group
  // writes a partial sum into spec->down_partial_out[group][tok][col];
  // Phase 5 reduces across groups into activations_out (bf16).
  //
  // The WGMMA down-projection function zeroes its own per-block
  // out_accum in SHM internally, so no pre-zero is needed here.
  //
  // The BS8 path is TMA+WGMMA only; the kernel asserts
  // `use_wgmma<Dims>::value` and `use_tma<Dims>::value` at the top of
  // this function, so dispatch is unconditional.
  moe_down_projection_BS8_allexperts_wgmma_tma<Dims>(
      expert_weights_down, expert_scales_down, top_k, batch_size, spec, shmem,
      down_weights_desc, down_activations_desc);

  MONO_PHASE_TIMESTAMP(t_after_down);

  // ── Site #3 — Col-stripe-local barrier (Phase 2b) ────────────────────
  //
  // Phase 4 wrote `spec->down_partial_out[down_group_r][tok][col_stripe
  // * DOWN_COL_TILE .. +DOWN_COL_TILE-1]`.  Phase 5 on block `b` reads
  // every `down_group_r`'s partial at its own col stripe `b`, so its
  // producer-set is exactly the `DOWN_GROUPS` blocks with
  // `blockIdx.x % DOWN_GRID == b`.  That sub-grid is also the arrival
  // set of `colstripe_barrier(col_stripe = b, arrival_count =
  // DOWN_GROUPS)`.  Every block (including those with
  // `down_group_r > 0` that don't enter Phase 5) calls the barrier to
  // publish its Phase-4 write; the block with `blockIdx.x = b` is the
  // Phase-5 writer and also the seed block (its ID == its col stripe).
  //
  // Per-barrier atomic contention drops from 128 → 16; DOWN_GRID = 8
  // independent col-stripe barriers run concurrently (Design "Site #3
  // correctness argument", Requirements 9.7, 9.8).
  {
    const uint32_t col_stripe_id = blockIdx.x % MoECoreDims<Dims>::DOWN_GRID;
    moe_monokernel::colstripe_barrier(
        colstripe_counters,
        /*col_stripe=*/col_stripe_id,
        /*arrival_count=*/MoECoreDims<Dims>::DOWN_GROUPS,
        /*seed_blockidx=*/col_stripe_id, colstripe_phase);
  }

  MONO_PHASE_TIMESTAMP(t_after_barrier3);

  // ── Phase 5 (WGMMA): reduction + writeback ─────────────────────────
  // Each block reads its own DOWN_COL_TILE output cols ×
  // DOWN_GROUPS groups × Dims::BS tokens of fp32 partials from GM and
  // sums across the DOWN_GROUPS dim into bf16 activations_out.
  //
  // Block-to-col mapping mirrors Phase 4a: only blocks with
  // `blockIdx.x < DOWN_GRID` are responsible for writing (the first
  // DOWN_GRID blocks cover the full HIDDEN_STATES output).  Blocks
  // beyond DOWN_GRID would map to duplicate cols via
  // `blockIdx.x % DOWN_GRID`, so we gate on the primary group
  // (down_group == 0) to avoid redundant writes.
  //
  // Phase-2a layout alignment (software-grid-sync spec):
  //   * Pre Phase 2a (BS8 TMA+WGMMA baseline): DOWN_COL_TILE=128,
  //     DOWN_GRID=16, DOWN_GROUPS=8.  The inner `g` loop runs 8 times;
  //     the primary-block col stripe covers 128 cols.
  //   * Post Phase 2a (BS8 TMA+WGMMA only): DOWN_COL_TILE=256,
  //     DOWN_GRID=8, DOWN_GROUPS=16.  The inner `g` loop now runs 16
  //     times; the primary-block col stripe covers 256 cols.  Total
  //     Phase-5 data read across all blocks is unchanged
  //     (HIDDEN_STATES × BS tokens), but per-block runtime doubles —
  //     compensated by halving the number of Phase-5 blocks.
  //
  //   Expressing the bounds via `CoreDims` (not hard-coded literals)
  //   keeps BS64 / non-TMA variants on the 128-col layout.  The
  //   `DOWN_GROUPS == UP_GROUPS` alignment is the prerequisite for the
  //   Phase-2b Expert_Barrier at site #2 and the ColStripe_Barrier at
  //   site #3 (replacing the two `grid_barrier` calls above).
  constexpr std::uint32_t DOWN_GRID_LOCAL = CoreDims::DOWN_GRID;
  constexpr std::uint32_t DOWN_GROUPS_LOCAL = CoreDims::DOWN_GROUPS;
  constexpr std::uint32_t DOWN_COL_TILE_LOCAL = CoreDims::DOWN_COL_TILE;
  const std::uint32_t down_group_r = blockIdx.x / DOWN_GRID_LOCAL;
  const std::uint32_t down_block_idx_r = blockIdx.x % DOWN_GRID_LOCAL;
  const std::uint32_t base_col_r = down_block_idx_r * DOWN_COL_TILE_LOCAL;

  if (down_group_r == 0) {
    const std::uint32_t group_stride_r = Dims::BS * Dims::HIDDEN_STATES;

    // Sum the DOWN_GROUPS partials for tokens in [0, batch_size) and
    // write bf16 to activations_out.
    //
    // Loading into a register array first (then summing) gives the
    // compiler license to issue all `DOWN_GROUPS` loads as
    // independent instructions, exposing parallelism the hardware can
    // exploit even though each load has high HBM latency.  Without
    // this, naive `sum += a[...]` introduces a sum-dependency chain
    // that serialises the loads at the back-end.
    for (std::uint32_t flat = threadIdx.x;
         flat < batch_size * DOWN_COL_TILE_LOCAL; flat += blockDim.x) {
      const std::uint32_t tok = flat / DOWN_COL_TILE_LOCAL;
      const std::uint32_t col_in_block = flat % DOWN_COL_TILE_LOCAL;
      const std::uint32_t col = base_col_r + col_in_block;
      const float* base_ptr =
          spec->down_partial_out + tok * Dims::HIDDEN_STATES + col;

      float vals[DOWN_GROUPS_LOCAL];
#pragma unroll
      for (std::uint32_t g = 0; g < DOWN_GROUPS_LOCAL; ++g) {
        vals[g] = base_ptr[g * group_stride_r];
      }

      float sum = 0.f;
#pragma unroll
      for (std::uint32_t g = 0; g < DOWN_GROUPS_LOCAL; ++g) {
        sum += vals[g];
      }
      activations_out[tok * Dims::HIDDEN_STATES + col] = (R_element)sum;
    }

    // Zero out activations_out[tok] for tok in [batch_size, Dims::BS)
    // for this block's DOWN_COL_TILE col stripe.
    for (std::uint32_t flat = threadIdx.x;
         flat < (Dims::BS - batch_size) * DOWN_COL_TILE_LOCAL;
         flat += blockDim.x) {
      const std::uint32_t tok = batch_size + flat / DOWN_COL_TILE_LOCAL;
      const std::uint32_t col_in_block = flat % DOWN_COL_TILE_LOCAL;
      const std::uint32_t col = base_col_r + col_in_block;
      activations_out[tok * Dims::HIDDEN_STATES + col] = (R_element)0.0f;
    }
  }

  MONO_PHASE_TIMESTAMP(t_after_phase5);
}

/**
 * @brief Top-K MoE kernel — single-pass path for BS > 8.
 *
 * Two-phase pipeline:
 *
 *  Phase 1: Calc warps compute all K expert selections into shmem flat arrays,
 *           then sort the virtual batch (num_tokens * top_k) by expert.
 *
 *  Phase 2: Quantize activations once per original token. Separate act_scale
 *           (for up-proj inside silu) from routing_weight (for down-proj).
 *
 *  Then: up-projection over sorted virtual batch → grid.sync →
 *        down-projection accumulating += into original token positions.
 *
 * The output buffer must be zeroed before calling this function.
 */
template <typename Dims>
__device__ void moe_kernel_topk_BS64(
    const A_element* __restrict__ activations_in, std::uint32_t token_count,
    const __nv_bfloat16* __restrict__ router_logits,
    const W_element* __restrict__ expert_weights_up,
    const S_element* __restrict__ expert_scales_up,
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down,
    R_element* __restrict__ activations_out, uint32_t top_k,
    ScoringFunc scoring_func, bool renormalize,
    MoEGemmSpec<Dims>* __restrict__ spec, MoE_SHM<Dims>* __restrict__ shmem,
    uint32_t* __restrict__ grid_counters, uint32_t& grid_phase) {
  static_assert(Dims::BS > 8);

  // Step 1: compute all K selections into shmem flat arrays
  if (is_calc_warp<Dims>()) {
    topK_BS64<Dims>(top_k, scoring_func, renormalize, router_logits,
                    token_count, shmem);
  }
  __syncthreads();

  // Step 2: sort BS*top_k virtual rows by expert, build token_indexes_topk
  //         and token_weights
  prepare_moe_topk_BSx_Ey<Dims>(token_count, top_k, shmem);
  __syncthreads();

  // Step 3: quantize activations once per original token.
  // Writes spec->activations[tok] (fp8) and shmem->act_scale[tok].
  //
  // Note: Stage 3b (copy routing_weight → topk_weights_flat) was removed.
  // The down-projection now reads path.bs64.token_weights[sorted_pos]
  // directly — the copy-back was a redundant pass.
  moe_scale_activation_BSx<Dims>(activations_in, token_count, spec, shmem,
                                 grid_counters, grid_phase);

  // Step 4: up-projection (reads token_weights per sorted slot)
  moe_up_projection_topk<Dims>(expert_weights_up, expert_scales_up, spec,
                               shmem);
  moe_monokernel::grid_barrier<Dims::KernelConfig::GRID_SIZE>(grid_counters,
                                                              grid_phase);

  // Step 5: down-projection (accumulates += into original token positions)
  moe_down_projection_topk<Dims>(expert_weights_down, expert_scales_down,
                                 activations_out, spec, shmem);
}

/**
 * @brief Top-K MoE kernel with configurable scoring and renormalization.
 *
 * Dispatches to moe_kernel_topk_BS8 (BS <= 8) or moe_kernel_topk_BS64 (BS > 8).
 *
 * `up_weights_desc` and `activations_desc` are the host-built TMA
 * descriptors consumed by `moe_up_projection_BS8_allexperts_wgmma_tma` when
 * `use_tma<Dims>::value` is true.  For non-TMA variants the torch-binding
 * wrapper passes zero-initialized `CUtensorMap` values and the descriptors
 * are never read (spec R6.1, R6.3).  The `__grid_constant__` qualifier
 * places them in constant memory coherent with all threads without SMEM
 * cost.
 */
// Requirement 4.4: pin the kernel to 1 block per SM at compile time.
// The software grid / partial barriers rely on the co-residency invariant
// (GRID_SIZE <= SM_count and max_active_blocks_per_SM == 1) so every
// launched block is guaranteed to be running when any other block spins
// on its arrival counter. `__launch_bounds__(BLOCK_SIZE, 1)` is the
// compile-time half of that invariant; the launcher enforces the runtime
// half via `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`.
template <typename Dims>
__global__
__launch_bounds__(Dims::KernelConfig::BLOCK_SIZE, 1) void moe_kernel_topk(
    const A_element* __restrict__ activations_in, std::uint32_t token_count,
    const __nv_bfloat16* __restrict__ router_logits,
    const W_element* __restrict__ expert_weights_up,
    const S_element* __restrict__ expert_scales_up,
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down,
    R_element* __restrict__ activations_out, void* __restrict__ scratchpad,
    size_t scratchpad_size, size_t shmem_size, std::uint32_t top_k,
    ScoringFunc scoring_func, bool renormalize,
    __grid_constant__ CUtensorMap const up_weights_desc,
    __grid_constant__ CUtensorMap const activations_desc,
    __grid_constant__ CUtensorMap const down_weights_desc,
    __grid_constant__ CUtensorMap const down_activations_desc) {
  // ── Compile-time preconditions on `Dims` (spec R7.3, R11.3) ─────────────
  // These fire at the first point where `Dims` is instantiated, so any
  // misconfigured variant is caught at compile time before any TMA /
  // WGMMA code is instantiated below.
  //
  //  * R7.3: `USE_TMA` requires `USE_WGMMA`. There is no TMA support for
  //    the scalar up-projection path.
  //  * R11.3: For every TMA-enabled variant, `MoE_SHM<Dims>` must fit in
  //    the H100 opt-in 228 KB per-block SHM budget. The existing 224 KB
  //    check inside `get_moe_shmem_size<Dims>()` is tighter, but this
  //    assertion documents the per-variant TMA budget and catches future
  //    SHM-layout regressions that loosen the opt-in cap.
  static_assert(!use_tma<Dims>::value || use_wgmma<Dims>::value,
                "USE_TMA requires USE_WGMMA; no TMA support for the scalar "
                "path.");
  static_assert(!use_tma<Dims>::value || sizeof(MoE_SHM<Dims>) <= 228 * 1024,
                "MoE_SHM<Dims> exceeds the 228 KB per-block SHM budget "
                "for TMA variants.");
  // Phase-2a layout alignment (software-grid-sync spec, Req 9.4):
  // For the BS8 TMA+WGMMA variant, `DOWN_COL_TILE` is bumped to 256,
  // which doubles the down-proj weight tile in SHM from 16 KB to 32 KB
  // per double-buffer slot (the `w_wgmma` / `w_down_wgmma` union grows
  // from 32 KB to 64 KB total).  The re-assertion below makes this
  // explicit at the BS8 TMA+WGMMA instantiation site so that any
  // future layout regression that overflows the 228 KB opt-in budget
  // after the Phase-2a alignment is flagged with a pointed error.
  static_assert(!(use_tma<Dims>::value && Dims::BS <= 8) ||
                    sizeof(MoE_SHM<Dims>) <= 228 * 1024,
                "Exceeds 228 KB opt-in SHM budget for BS8 TMA+WGMMA after "
                "Phase 2a layout alignment (DOWN_COL_TILE = 256 doubles "
                "the per-block down-proj weight tile).");

  assert(MoECoreDims<Dims>::THREADS_PER_WARP == 32);
  assert(blockDim.x == Dims::KernelConfig::BLOCK_SIZE);
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  assert(gridDim.x == Dims::KernelConfig::GRID_SIZE);
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);

  assert(token_count <= Dims::BS);
  assert(token_count > 0);
  assert(top_k >= 1 && top_k <= MoE_SHM<Dims>::MAX_TOPK);

  MoEGemmSpec<Dims>* spec = reinterpret_cast<MoEGemmSpec<Dims>*>(scratchpad);

  extern __shared__ char shmem_buffer[];
  MoE_SHM<Dims>* shmem = reinterpret_cast<MoE_SHM<Dims>*>(shmem_buffer);

  // ── Software barrier pointers and per-region phase state ─────────────
  //
  // Design Component A ("Device-side function signature") + Component B
  // ("Per-sub-grid phase state").  The barrier primitives
  // (`grid_barrier`, `expert_barrier`, `colstripe_barrier`) are pure
  // device functions that take a counter-region pointer + an in/out
  // register-resident phase counter; we materialize the pointers once in
  // the kernel prologue from the scratchpad base and keep the phase
  // counters in the block-local register file.
  //
  //   * `grid_counters` — used by every site that stays at
  //     `Grid_Barrier` (BS64 sites #1, #4, #5 in Phase 1 and BS8 sites
  //     #2, #3 in Phase 1 before Phase-2b downgrades them to
  //     Expert_Barrier / ColStripe_Barrier).
  //   * `expert_counters` — reserved for site #2 in Phase 2b
  //     (task 12.1).  Threaded into `moe_kernel_topk_BS8` so the BS8
  //     Phase-2b migration can wire it into the Expert_Barrier call
  //     without re-plumbing the call chain.
  //   * `colstripe_counters` — reserved for site #3 in Phase 2b
  //     (task 12.2).  Same plumbing treatment as `expert_counters`.
  //   * `grid_phase` / `expert_phase` / `colstripe_phase` — per-region
  //     block-local phase counters; initialized to 0 at kernel entry,
  //     bumped by each barrier call on that region.
  //   * `GRID_SIZE_STATIC` — template non-type arg to
  //     `grid_barrier<>`; the compile-time `Dims::KernelConfig::GRID_SIZE`
  //     value lets the primitive fold its seed value and degenerate-case
  //     gate.
  //
  // Validates: Requirements 2.8, 3.1.
  uint32_t* grid_counters = spec->grid_barrier.slot;
  uint32_t* expert_counters = &spec->partial_barrier.expert_slot[0][0];
  uint32_t* colstripe_counters = &spec->partial_barrier.colstripe_slot[0][0];
  uint32_t grid_phase = 0;
  uint32_t expert_phase = 0;
  uint32_t colstripe_phase = 0;
  constexpr uint32_t GRID_SIZE_STATIC = Dims::KernelConfig::GRID_SIZE;

  // Site #1 — top-of-kernel output zero-out + sync.
  //
  // For BS8 (TMA+WGMMA): ELIMINATED. The Phase 5 reduction in
  // moe_kernel_topk_BS8 `=`-writes every element of activations_out
  // (assigns reduced sum for tokens [0, batch_size) and explicitly
  // zeros [batch_size, Dims::BS) per block col stripe), so the
  // pre-zero + sync is dead work. See Requirements 3.4, 3.5 and
  // Design Migration Plan Site #1.
  //
  // For BS64: PRESERVED. The BS64 down-projection uses `+=` into
  // activations_out and needs the buffer to start zeroed.
  // cooperative_groups::this_grid().sync() → grid_barrier<>.
  if constexpr (Dims::BS > 8) {
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
         i < token_count * Dims::HIDDEN_STATES; i += blockDim.x * gridDim.x) {
      activations_out[i] = (__nv_bfloat16)0.0f;
    }
    moe_monokernel::grid_barrier<GRID_SIZE_STATIC>(grid_counters, grid_phase);
  }

  if constexpr (Dims::BS <= 8) {
    moe_kernel_topk_BS8<Dims>(
        activations_in, token_count, router_logits, expert_weights_up,
        expert_scales_up, expert_weights_down, expert_scales_down,
        activations_out, top_k, scoring_func, renormalize, spec, shmem,
        up_weights_desc, activations_desc, down_weights_desc,
        down_activations_desc, grid_counters, grid_phase, expert_counters,
        expert_phase, colstripe_counters, colstripe_phase);
  } else {
    moe_kernel_topk_BS64<Dims>(
        activations_in, token_count, router_logits, expert_weights_up,
        expert_scales_up, expert_weights_down, expert_scales_down,
        activations_out, top_k, scoring_func, renormalize, spec, shmem,
        grid_counters, grid_phase);
  }
}

}  // namespace moe_monokernel
