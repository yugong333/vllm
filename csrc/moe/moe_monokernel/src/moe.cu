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
#include "moe_up_projection.cu"
#include "moe_routing.cu"
#undef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION

namespace moe_monokernel {

/**
 * @brief Top-K MoE kernel — split-phase path for BS <= 8.
 *
 * Compact SHM layout with unions:
 *   a: fp8 up-activations / double-buffered fp8 down-activations
 *   w[2]: double-buffered orig(bf16) / w_up(fp8) / w_down(fp8)
 *   partial_result: up / down scratch
 *
 * Pipeline:
 *   Phase 1: fetch orig into w[0].orig || routing + topK
 *   Phase 2: quantize w[0].orig → a.up || prefetch w_up into w[1].up
 *   Phase 3: up-proj loop (double-buffered w[].up)
 *            → SiLU → write bf16 to spec->temp_bf16
 *   grid.sync()
 *   Phase 4: down-proj pipelined design with fixed w[2] slot roles:
 *            w[0] = bf16 intermediates, w[1] = fp8 weights
 *            Stage 0: all warps fetch expert 0's bf16 → w[0].bf16_buf
 *            Per expert:
 *              Stage A: prefetch w_down → w[1] || quantize w[0].bf16_buf →
 * a.down Stage B: prefetch next bf16 → w[0] || MMA a.down × w[1].down → accum
 *   Phase 5: writeback out_accum → global bf16
 *
 * 1 grid sync total.
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
    MoEGemmSpec<Dims>* __restrict__ spec, MoE_SHM<Dims>* __restrict__ shmem) {
  static_assert(Dims::BS <= 8);
  using CoreDims = MoECoreDims<Dims>;

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  auto* shm = &shmem->u.tiny;

  // ── Phase 1: prefetch activations into w[0].orig || routing ─────────────
  // Phase 1 is always executed — routing (topK / prepare_moe_topk) writes
  // shmem->experts and shmem->topk_ids_flat which later phases depend on.
  // Skipping it would leave those fields uninitialized and make profiling
  // garbage.
  if (is_prefetch_warp<Dims>()) {
    const std::uint32_t pw = get_prefetch_warp<Dims>();
    for (std::uint32_t tok = pw; tok < batch_size;
         tok += CoreDims::PREFETCH_WARP_COUNT)
      moe_fetch_activation_async<Dims>(
          activations_in + tok * Dims::HIDDEN_STATES, shm->w[0].orig[tok],
          pipe);
  } else {
    topK_BS8<Dims>(top_k, scoring_func, renormalize, router_logits, batch_size,
                   shmem);
    sync_calc_threads<Dims>();
    prepare_moe_topk_BS8<Dims>(batch_size, top_k, shmem);
  }
  cuda::pipeline_consumer_wait_prior<0>(pipe);
  __syncthreads();

  // ── Phase 2: quantize w[0].orig → a.up || prefetch w_up → w[1] ────────
  // GRID=128 design, two-expert-group parallelism:
  //   UP_GRID = 2*N / W_UP_TILE = 64 row-tiles cover the full 2*N weight
  //   rows for one expert. With GRID_SIZE=128, we run TWO groups of
  //   UP_GRID blocks each, processing DIFFERENT experts in parallel:
  //     group 0 (blockIdx.x in [0,  UP_GRID))  → experts at indices 0,2,4,...
  //     group 1 (blockIdx.x in [UP_GRID, 2*UP_GRID)) → experts 1,3,5,...
  //   Within each group, blockIdx.x % UP_GRID indexes the row-tile.
  //
  // Each group prefetches its OWN starting expert's weights + scales.
  constexpr std::uint32_t UP_GRID = 2 * Dims::N / CoreDims::W_UP_TILE;
  constexpr std::uint32_t UP_GROUPS =
      Dims::KernelConfig::GRID_SIZE / UP_GRID;  // 1 or 2
  static_assert(Dims::KernelConfig::GRID_SIZE % UP_GRID == 0,
                "GRID_SIZE must be a multiple of UP_GRID.");
  static_assert(UP_GROUPS <= 2,
                "Two-expert parallelism supports up to 2 groups.");
  const std::uint32_t up_group = blockIdx.x / UP_GRID;
  const std::uint32_t up_block_idx = blockIdx.x % UP_GRID;
  const bool in_up = (up_group < UP_GROUPS);

  // Skip Phase 2 entirely if this block is not in any up-proj group (only
  // relevant if GRID_SIZE > UP_GROUPS*UP_GRID, which currently never
  // happens — kept for safety).
  if (in_up) {
    // Each group starts from expert index `up_group` (0 or 1), stepping
    // by UP_GROUPS. If there are fewer experts than groups, the trailing
    // group's starting expert may not exist; skip the prefetch in that
    // case.
    const std::uint32_t my_expert_start = up_group;
    const bool my_group_has_work = my_expert_start < shmem->expert_count;

    if (is_prefetch_warp<Dims>()) {
#ifndef MONO_PROFILE_SKIP_PREFETCH
      if (my_group_has_work) {
        pipe.producer_acquire();
        // Fetch this group's starting expert's weights using this block's
        // row-tile (not blockIdx directly) so both groups reuse the same
        // 64-row-tile layout.
        const unsigned base_row_up = up_block_idx * CoreDims::W_UP_TILE / 2;
        moe_request_up_expert_for_row<Dims, Dims::HIDDEN_STATES>(
            expert_weights_up, shmem->experts[my_expert_start].id, base_row_up,
            shm->w[1].up, pipe);
        pipe.producer_commit();
        moe_request_up_scale_for_row<Dims>(expert_scales_up,
                                           shmem->experts[my_expert_start].id,
                                           base_row_up, shm->up_scale[1]);
      }
#endif
    } else {
#ifndef MONO_PROFILE_SKIP_CALC
      // Both groups independently quantize the input activations into
      // their own SHM. (Redundant across blocks but fully parallel.)
      const std::uint32_t cw = get_calc_warp<Dims>();
      if (cw < batch_size) {
        moe_scale_activation_BS8<Dims>(shm->w[0].orig[cw], shm->a.up[cw], cw,
                                       shmem->act_scale[cw]);
      }
#endif
    }
  }
  __syncthreads();

  // ── Phase 3: Up-projection — two expert groups in parallel ──────────────
  // Group `g` (blocks [g*UP_GRID, (g+1)*UP_GRID)) iterates experts starting
  // at index `g`, stepping by UP_GROUPS. Each group writes to DIFFERENT
  // virtual_row slots of spec->temp_bf16 (because each expert has its own
  // k index within a token's top-K list), so the two groups never have a
  // write conflict.
  if (in_up && up_group < shmem->expert_count) {
    moe_up_projection_BS8_allexperts<Dims>(
        expert_weights_up, expert_scales_up, top_k, batch_size, spec, shmem,
        up_block_idx, /*expert_start=*/up_group,
        /*expert_stride=*/UP_GROUPS);
  }

  // ── Single grid.sync — all blocks finish writing spec->temp_bf16 ──────
  cooperative_groups::this_grid().sync();

  // Zero the per-block fp32 output accumulator in SHM.
  // Zero exactly the `[BS][W_DOWN_TILE]` logical shape, not including row
  // padding, to preserve the 2D indexing `out_accum[tok][col]` used below.
  const unsigned base_row_dn = blockIdx.x * CoreDims::W_DOWN_TILE;
  for (unsigned idx = threadIdx.x; idx < Dims::BS * CoreDims::W_DOWN_TILE;
       idx += blockDim.x) {
    unsigned tok = idx / CoreDims::W_DOWN_TILE;
    unsigned col = idx % CoreDims::W_DOWN_TILE;
    shm->out_accum[tok][col] = 0.f;
  }
  __syncthreads();

  // ── Phase 4: Down-projection — pipelined 4-stage design ──────────────
  // For each expert: fetch bf16 intermediate → SHM, quantize bf16→fp8,
  // MMA fp8×fp8 with w_down, all pipelined with double-buffering.
  moe_down_projection_BS8_allexperts<Dims>(
      expert_weights_down, expert_scales_down, top_k, batch_size, spec, shmem);

  // ── Phase 5: Writeback SHM fp32 accumulator → global bf16 output ────────
#ifdef DEBUG_MOE_PRINT
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("[DBG FINAL out_accum[0][0..15]:");
    for (unsigned c = 0; c < 16 && c < CoreDims::W_DOWN_TILE; c++)
      printf(" %.4f", shm->out_accum[0][c]);
    printf("\n");
  }
#endif
  for (unsigned tok = 0; tok < batch_size; ++tok) {
    for (unsigned col = threadIdx.x; col < CoreDims::W_DOWN_TILE;
         col += blockDim.x) {
      activations_out[tok * Dims::HIDDEN_STATES + base_row_dn + col] =
          (R_element)shm->out_accum[tok][col];
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
    ScoringFunc scoring_func, bool renormalize) {
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
    moe_kernel_topk_BS8<Dims>(
        activations_in, token_count, router_logits, expert_weights_up,
        expert_scales_up, expert_weights_down, expert_scales_down,
        activations_out, top_k, scoring_func, renormalize, spec, shmem);
  } else {
    moe_kernel_topk_BS64<Dims>(
        activations_in, token_count, router_logits, expert_weights_up,
        expert_scales_up, expert_weights_down, expert_scales_down,
        activations_out, top_k, scoring_func, renormalize, spec, shmem);
  }
}

}  // namespace moe_monokernel
