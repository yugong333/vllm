/**
 * Main file of the MoE monokernel (top-K path).  Building this file builds
 * the whole kernel — it includes all other implementation files.  See
 * DESIGN.md for the architecture and moe_interface.h for the entry-point
 * documentation.
 */

#include <cstdint>

#include "moe_interface.h"

#define INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
#include "moe_down_projection.cu"
#include "moe_grid_barrier.h"
#include "moe_internal.h"
#include "moe_scale_inputs.cu"
#include "moe_tma.h"
#include "moe_up_projection.cu"
#include "moe_routing.cu"
#undef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION

namespace moe_monokernel {

/**
 * @brief Top-K MoE kernel — split-phase TMA+WGMMA path for BS <= 8.
 *
 * Pipeline (see DESIGN.md "Phase pipeline"):
 *   Phase 1: routing (topK) ∥ routing-window TMA (full BF16 input tile
 *            into bf16_in_full, completion on bar_rwin)
 *   Phase 2: warp 0 runs prepare_moe_topk_BS8; warps 1..11 wait on
 *            bar_rwin and quantize the tile into fp8_act_full + act_scale
 *   Phase 3: up-proj — streaming WGMMA reading FP8 from fp8_act_full →
 *            SiLU → fp8 writeback to spec->temp_fp8
 *   site #2: expert barrier
 *   Phase 4: down-proj — streaming WGMMA; each block atomicAdds its fp32
 *            partial sum into spec->down_partial_out
 *   site #3: col-stripe barrier
 *   Phase 5: fp32 → bf16 cast, write activations_out
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
    const float* __restrict__ expert_bias, float routed_scaling_factor,
    MoEGemmSpec<Dims>* __restrict__ spec, MoE_SHM<Dims>* __restrict__ shmem,
    CUtensorMap const& up_weights_desc, CUtensorMap const& activations_desc,
    CUtensorMap const& down_weights_desc,
    CUtensorMap const& down_activations_desc,
    uint32_t* __restrict__ expert_counters, uint32_t& expert_phase,
    uint32_t* __restrict__ colstripe_counters, uint32_t& colstripe_phase) {
  static_assert(Dims::BS <= 8);
  static_assert(use_wgmma<Dims>::value,
                "BS8 path requires the WGMMA configuration (use_wgmma).");
  static_assert(use_tma<Dims>::value, "BS8 path requires USE_TMA");
  using CoreDims = MoECoreDims<Dims>;

  MONO_PHASE_TIMESTAMP(t_start);

  // Zero the Phase-4 accumulator (all blocks cooperate); the site-#2
  // expert barrier publishes the zero across blocks before any Phase-4
  // atomicAdd fires.
  {
    const uint32_t partial_n = Dims::BS * Dims::HIDDEN_STATES;
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < partial_n;
         i += blockDim.x * gridDim.x) {
      spec->down_partial_out[i] = 0.f;
    }
  }

  // ── mbarrier init ───────────────────────────────────────────────────
  // Inits run unconditionally (cheap SHM writes, keeps state well-defined
  // even when SKIP_PREFETCH elides the arms below; the matching waits are
  // gated on the same flag, so nothing blocks on an uninitialized parity).
  auto* u_tma = &shmem->u.tiny_wgmma_tma;
  if (is_tma_launcher_thread<Dims>()) {
  #pragma unroll
    for (uint32_t i = 0; i < MoECoreDims<Dims>::UP_W_SLOTS; ++i) {
      mbarrier_init(&u_tma->bar_w[i], 1u);
    }
    mbarrier_init(&u_tma->bar_rwin, 1u);
    fence_mbarrier_init_release_cluster();
  }
  // Publish the launcher-thread inits to every warp before any
  // try_wait.parity.  fence_mbarrier_init alone is not sufficient: it pairs
  // with a matching acquire, but try_wait.parity is not an acquire of the
  // init — without this sync a prefetch warp can hit the Phase-2 bar_rwin
  // wait while the barrier is still uninitialized (mbarrier state
  // corruption under compute-sanitizer).
  __syncthreads();

  // ── Phase 1: routing ∥ routing-window BF16 prefetch ───────────────────
  const unsigned warp_id = get_any_warp<Dims>();
  if (warp_id >= CoreDims::CALC_WARP_COUNT) {
    // Prefetch warps: the launcher arms bar_rwin once for the whole tile
    // and issues the K_BLOCKS_TOTAL bulk loads; other lanes idle.
    if (is_tma_launcher_thread<Dims>()) {
#ifndef MONO_PROFILE_SKIP_PREFETCH_UP
      constexpr std::uint32_t RWIN_TX_BYTES =
          Dims::BS * MoE_SHM<Dims>::U::TinyDataWGMMA_TMA::K_BLOCKS_TOTAL *
          CoreDims::K_STEP_WGMMA *
          static_cast<std::uint32_t>(sizeof(A_element));
      mbarrier_arrive_expect_tx(&u_tma->bar_rwin,
                                /*tx_bytes=*/RWIN_TX_BYTES);
      moe_load_full_bf16_input<Dims>(activations_desc, u_tma->bf16_in_full,
                                     &u_tma->bar_rwin);
#endif
    }
  } else {
    // Calc warps.  Routing is never gated on the SKIP_CALC profile flags:
    // shmem->expert_count / experts[] drive downstream loop bounds and
    // must always be valid.
    topK_BS8<Dims>(top_k, scoring_func, renormalize, router_logits, batch_size,
                   shmem, expert_bias, routed_scaling_factor);
    MONO_PHASE_TIMESTAMP(t_after_topk);
    sync_calc_threads<Dims>();
    MONO_PHASE_TIMESTAMP(t_after_sync_calc);
  }

  // ── Phase 2: prepare (warp 0) ∥ quantize (warps 1..11) ────────────────
  // Warp 0 builds the routing tables and does not read bf16_in_full, so it
  // skips the bar_rwin wait.  Warps 1..11 wait for the Phase-1 load, then
  // quantize bf16 → fp8_act_full + act_scale.  The two sides touch
  // disjoint SHM; the single trailing __syncthreads() publishes both to
  // all warps before Phase 3.
  if (warp_id == 0) {
    prepare_moe_topk_BS8<Dims>(batch_size, top_k, shmem, spec);
  } else {
#ifndef MONO_PROFILE_SKIP_PREFETCH_UP
    uint32_t parity_rwin = 0;
    while (!mbarrier_try_wait_parity(&u_tma->bar_rwin, parity_rwin)) {
    }
#endif
#ifndef MONO_PROFILE_SKIP_CALC_UP
    routing_phase_quantize<Dims>(u_tma->bf16_in_full, u_tma->fp8_act_full,
                                 shmem->act_scale, batch_size);
#endif
  }
  __syncthreads();

  MONO_PHASE_TIMESTAMP(t_after_routing);

  // ── Phase 3: up-projection — expert groups in parallel ────────────────
  // Group g (blocks [g*UP_GRID, (g+1)*UP_GRID)) iterates experts starting
  // at index g, stepping by UP_GROUPS.  Groups write disjoint temp_fp8
  // rows (each routed (tok, expert) pair has its own sorted_slot row).
  constexpr std::uint32_t UP_GRID = 2 * Dims::N / CoreDims::W_UP_TILE_EFFECTIVE;
  constexpr std::uint32_t UP_GROUPS = Dims::KernelConfig::GRID_SIZE / UP_GRID;
  static_assert(Dims::KernelConfig::GRID_SIZE % UP_GRID == 0,
                "GRID_SIZE must be a multiple of UP_GRID.");
  static_assert(UP_GROUPS <= Dims::NUM_EXPERTS,
                "UP_GROUPS cannot exceed the total number of experts.");
  const std::uint32_t up_group = blockIdx.x / UP_GRID;
  const std::uint32_t up_block_idx = blockIdx.x % UP_GRID;
  const bool in_up = (up_group < UP_GROUPS);

  if (in_up && up_group < shmem->expert_count) {
    // Compile-time dispatch: two stacked M-atoms per block (raw two-TMA
    // layout, 122B) vs the single-atom interleaved layout (35B).
    if constexpr (CoreDims::UP_COL_HALVES == 2u) {
      moe_up_projection_BS8_122B_wgmma_tma<Dims>(
          activations_in, expert_weights_up, expert_scales_up, top_k,
          batch_size, spec, shmem, up_weights_desc, activations_desc,
          up_block_idx,
          /*expert_start=*/up_group,
          /*expert_stride=*/UP_GROUPS);
    } else {
      moe_up_projection_BS8_allexperts_wgmma_tma<Dims>(
          activations_in, expert_weights_up, expert_scales_up, top_k,
          batch_size, spec, shmem, up_weights_desc, activations_desc,
          up_block_idx,
          /*expert_start=*/up_group,
          /*expert_stride=*/UP_GROUPS);
    }
  }

  MONO_PHASE_TIMESTAMP(t_after_up);

  // ── Site #2: Phase 3 → Phase 4 barrier ────────────────────────────────
  //
  // COUPLED carve (UP_GROUPS == DOWN_GROUPS, all shipped coupled shapes):
  // the UP_GRID blocks that produced an expert group's temp_fp8 rows are
  // exactly the blocks that consume them in Phase 4, so a symmetric
  // per-up_group expert barrier suffices (arrival contention UP_GRID
  // instead of GRID_SIZE; UP_GROUPS barriers run concurrently).
  //
  // DECOUPLED carve (UP_GROUPS != DOWN_GROUPS): producer set != consumer
  // set, so the rendezvous splits into a producer-arrive on this block's
  // up_group plus consumer-waits on every up_group whose temp_fp8 rows
  // this block reads in Phase 4.  The wait loop mirrors the Phase-4 expert
  // loop exactly (same bound/stride), so the wait set is precisely the
  // produced set.  expert_count is block-uniform, so the __syncthreads
  // inside expert_consume_wait stays collective.
  if constexpr (UP_GROUPS == MoECoreDims<Dims>::DOWN_GROUPS) {
    if (in_up) {
      moe_monokernel::expert_barrier(expert_counters,
                                     /*expert_id=*/up_group,
                                     /*arrival_count=*/UP_GRID,
                                     /*seed_blockidx=*/up_group * UP_GRID,
                                     expert_phase);
    }
  } else {
    if (in_up) {
      moe_monokernel::expert_produce_arrive(
          expert_counters, /*up_group=*/up_group,
          /*arrival_count=*/UP_GRID, /*seed_blockidx=*/up_group * UP_GRID);
    }
    const std::uint32_t down_group_c =
        blockIdx.x / MoECoreDims<Dims>::DOWN_GRID;
    for (std::uint32_t e = down_group_c; e < shmem->expert_count;
         e += MoECoreDims<Dims>::DOWN_GROUPS) {
      moe_monokernel::expert_consume_wait(expert_counters,
                                          /*up_group=*/e % UP_GROUPS,
                                          /*arrival_count=*/UP_GRID);
    }
  }

  MONO_PHASE_TIMESTAMP(t_after_barrier2);

  // ── Phase 4: down-projection ──────────────────────────────────────────
  moe_down_projection_BS8_allexperts_wgmma_tma<Dims>(
      expert_weights_down, expert_scales_down, top_k, batch_size, spec, shmem,
      down_weights_desc, down_activations_desc);

  MONO_PHASE_TIMESTAMP(t_after_down);

  // ── Site #3: Phase 4 → Phase 5 barrier ────────────────────────────────
  // Phase 5 on block b reads down_partial_out cells at col stripe b, whose
  // producers are the DOWN_GROUPS blocks with blockIdx.x % DOWN_GRID == b.
  // Every block calls the barrier to publish its Phase-4 atomicAdds; the
  // seed block (blockIdx.x == stripe id) is also the Phase-5 reader.
  {
    const uint32_t col_stripe_id = blockIdx.x % MoECoreDims<Dims>::DOWN_GRID;
    moe_monokernel::colstripe_barrier(
        colstripe_counters,
        /*col_stripe=*/col_stripe_id,
        /*arrival_count=*/MoECoreDims<Dims>::DOWN_GROUPS,
        /*seed_blockidx=*/col_stripe_id, colstripe_phase);
  }

  MONO_PHASE_TIMESTAMP(t_after_barrier3);

  // ── Phase 5: fp32 → bf16 cast + writeback ─────────────────────────────
  // Each cell of down_partial_out already holds the full sum (Phase-4
  // atomicAdds); the first DOWN_GRID blocks stream-cast their own col
  // stripe.  Every output element is `=`-written (real tokens get the sum,
  // padding tokens get zero), so no output pre-zero pass exists anywhere.
  constexpr std::uint32_t DOWN_GRID_LOCAL = CoreDims::DOWN_GRID;
  constexpr std::uint32_t DOWN_COL_TILE_LOCAL = CoreDims::DOWN_COL_TILE;
  const std::uint32_t down_group_r = blockIdx.x / DOWN_GRID_LOCAL;
  const std::uint32_t down_block_idx_r = blockIdx.x % DOWN_GRID_LOCAL;
  const std::uint32_t base_col_r = down_block_idx_r * DOWN_COL_TILE_LOCAL;

  if (down_group_r == 0) {
    for (std::uint32_t flat = threadIdx.x;
         flat < batch_size * DOWN_COL_TILE_LOCAL; flat += blockDim.x) {
      const std::uint32_t tok = flat / DOWN_COL_TILE_LOCAL;
      const std::uint32_t col_in_block = flat % DOWN_COL_TILE_LOCAL;
      const std::uint32_t col = base_col_r + col_in_block;
      const float v = spec->down_partial_out[tok * Dims::HIDDEN_STATES + col];
      activations_out[tok * Dims::HIDDEN_STATES + col] = (R_element)v;
    }

    // Zero the padding tokens [batch_size, Dims::BS) in this col stripe.
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
 * @brief Kernel entry point.  Dispatches to moe_kernel_topk_BS8.
 *
 * The TMA descriptors are built host-side (moe_wrapper.cu) and passed as
 * `__grid_constant__` parameters.
 *
 * `__launch_bounds__(BLOCK_SIZE, 1)` is the compile-time half of the
 * one-block-per-SM co-residency invariant the software barriers rely on
 * (every participating block must be scheduled from launch); the host
 * launcher checks the runtime half (GRID_SIZE <= SM count).
 */
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
    const float* __restrict__ expert_bias, float routed_scaling_factor,
    __grid_constant__ CUtensorMap const up_weights_desc,
    __grid_constant__ CUtensorMap const activations_desc,
    __grid_constant__ CUtensorMap const down_weights_desc,
    __grid_constant__ CUtensorMap const down_activations_desc) {
  static_assert(!use_tma<Dims>::value || use_wgmma<Dims>::value,
                "USE_TMA requires USE_WGMMA; no TMA support for the scalar "
                "path.");
  static_assert(!use_tma<Dims>::value || sizeof(MoE_SHM<Dims>) <= 228 * 1024,
                "MoE_SHM<Dims> exceeds the 228 KB per-block SHM budget "
                "for TMA variants.");

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

  // Software barrier counter regions (in the scratchpad) + block-local
  // register-resident phase counters.
  uint32_t* expert_counters = &spec->partial_barrier.expert_slot[0][0];
  uint32_t* colstripe_counters = &spec->partial_barrier.colstripe_slot[0][0];
  uint32_t expert_phase = 0;
  uint32_t colstripe_phase = 0;

  moe_kernel_topk_BS8<Dims>(
      activations_in, token_count, router_logits, expert_weights_up,
      expert_scales_up, expert_weights_down, expert_scales_down,
      activations_out, top_k, scoring_func, renormalize, expert_bias,
      routed_scaling_factor, spec, shmem, up_weights_desc, activations_desc,
      down_weights_desc, down_activations_desc, expert_counters, expert_phase,
      colstripe_counters, colstripe_phase);
}

}  // namespace moe_monokernel
