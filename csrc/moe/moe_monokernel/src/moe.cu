/**
 * This is the main file of the MoE monokernel for Qwen3-Coder (top-K path).
 * It is designed so that you just need to build this file. It includes all
 * relevant implementations. For documentation of the main entry function
 * moe_kernel_topk, see moe_interface.h
 */

#include <cooperative_groups.h>
#include <cstdint>

#include "moe_interface.h"

#define INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
#include "moe_debug.h"
#include "moe_down_projection.cu"
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
 *   Phase 1: routing + topK (prefetch warps idle)
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
    CUtensorMap const& down_activations_desc) {
  static_assert(Dims::BS <= 8);
  static_assert(use_wgmma<Dims>::value,
                "BS8 path requires the WGMMA configuration (use_wgmma).");
  static_assert(use_tma<Dims>::value, "BS8 path requires USE_TMA");
  using CoreDims = MoECoreDims<Dims>;

  // ── Phase 1: routing (topK) — no prefetch needed ────────────────────────
  // Phase 1 runs routing (topK / prepare_moe_topk) which writes
  // shmem->experts and shmem->topk_ids_flat that later phases depend on.
  //
  // The WGMMA streaming pipeline reads bf16 activations directly from
  // global memory one K=128 tile at a time (see Phase 3), so no
  // prefetch into SHM is needed here. Prefetch warps idle during Phase 1.
  if (is_prefetch_warp<Dims>()) {
    // WGMMA path: prefetch warps have nothing to do here. The streaming
    // pipeline's first bf16 prefetch is issued inside Phase 3 priming.
  } else {
    topK_BS8<Dims>(top_k, scoring_func, renormalize, router_logits, batch_size,
                   shmem);
    sync_calc_threads<Dims>();
    prepare_moe_topk_BS8<Dims>(batch_size, top_k, shmem);
  }
  __syncthreads();

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
  //   (1) prefetch bf16_in[0] from global
  //   (2) prefetch w[0] and bf16_in[1] || quantize bf16_in[0] → fp8[0]
  // and then the streaming K-loop alternates WGMMA + bf16 prefetch with
  // quantize + weight prefetch.
  __syncthreads();

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
        /*expert_stride=*/UP_GROUPS);
  }

  // ── Single grid.sync — all blocks finish writing spec->temp_bf16 ──────
  cooperative_groups::this_grid().sync();

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

  // ── grid.sync — all blocks finish writing spec->down_partial_out ───
  cooperative_groups::this_grid().sync();

  // ── Phase 5 (WGMMA): reduction + writeback ─────────────────────────
  // Each block reads its own DOWN_COL_TILE=128 output cols ×
  // DOWN_GROUPS groups × Dims::BS tokens of fp32 partials from GM and
  // sums across the DOWN_GROUPS dim into bf16 activations_out.
  //
  // Block-to-col mapping mirrors Phase 4a: only blocks with
  // `blockIdx.x < DOWN_GRID` are responsible for writing (the first
  // DOWN_GRID blocks cover the full HIDDEN_STATES output).  Blocks
  // beyond DOWN_GRID would map to duplicate cols via
  // `blockIdx.x % DOWN_GRID`, so we gate on the primary group
  // (down_group == 0) to avoid redundant writes.
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
    for (std::uint32_t flat = threadIdx.x;
         flat < batch_size * DOWN_COL_TILE_LOCAL; flat += blockDim.x) {
      const std::uint32_t tok = flat / DOWN_COL_TILE_LOCAL;
      const std::uint32_t col_in_block = flat % DOWN_COL_TILE_LOCAL;
      const std::uint32_t col = base_col_r + col_in_block;

      float sum = 0.f;
#pragma unroll
      for (std::uint32_t g = 0; g < DOWN_GROUPS_LOCAL; ++g) {
        sum += spec->down_partial_out[g * group_stride_r +
                                      tok * Dims::HIDDEN_STATES + col];
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
    MoEGemmSpec<Dims>* __restrict__ spec, MoE_SHM<Dims>* __restrict__ shmem) {
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
  moe_scale_activation_BSx<Dims>(activations_in, token_count, spec, shmem);

  // Step 4: up-projection (reads token_weights per sorted slot)
  moe_up_projection_topk<Dims>(expert_weights_up, expert_scales_up, spec,
                               shmem);
  cooperative_groups::this_grid().sync();

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
template <typename Dims>
__global__ void moe_kernel_topk(
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

  // Zero output before accumulation
  for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
       i < token_count * Dims::HIDDEN_STATES; i += blockDim.x * gridDim.x) {
    activations_out[i] = (__nv_bfloat16)0.0f;
  }

  cooperative_groups::this_grid().sync();

  if constexpr (Dims::BS <= 8) {
    moe_kernel_topk_BS8<Dims>(activations_in, token_count, router_logits,
                              expert_weights_up, expert_scales_up,
                              expert_weights_down, expert_scales_down,
                              activations_out, top_k, scoring_func, renormalize,
                              spec, shmem, up_weights_desc, activations_desc,
                              down_weights_desc, down_activations_desc);
  } else {
    moe_kernel_topk_BS64<Dims>(
        activations_in, token_count, router_logits, expert_weights_up,
        expert_scales_up, expert_weights_down, expert_scales_down,
        activations_out, top_k, scoring_func, renormalize, spec, shmem);
  }
}

}  // namespace moe_monokernel
