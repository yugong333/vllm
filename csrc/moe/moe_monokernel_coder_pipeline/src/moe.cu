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

  // ── Phase 2: quantize w[0].orig → a.up || prefetch w_up[0] into w[1] ───
  if (is_prefetch_warp<Dims>()) {
    pipe.producer_acquire();
    moe_request_up_expert<Dims, Dims::HIDDEN_STATES>(
        expert_weights_up, shmem->experts[0].id, shm->w[1].up, pipe);
    pipe.producer_commit();
  } else {
    const std::uint32_t cw = get_calc_warp<Dims>();
    if (cw < batch_size) {
      float act_scale =
          moe_scale_activation_BS8<Dims>(shm->w[0].orig[cw], shm->a.up[cw], cw);
      if (get_thread<Dims>() == 0) {
        shmem->act_scale[cw] = act_scale;
      }
    }
  }
  // // Wait for Phase 2 weight prefetch to complete before Phase 3 reads
  // w[1].up cuda::pipeline_consumer_wait_prior<0>(pipe);
  __syncthreads();

  // ── Phase 3: Up-projection — all experts, double-buffered w[].up ────────
  moe_up_projection_BS8_allexperts<Dims>(expert_weights_up, expert_scales_up,
                                         top_k, batch_size, spec, shmem);

  // ── Single grid.sync — all blocks finish writing spec->temp_bf16 ──────
  cooperative_groups::this_grid().sync();

  // Zero the per-block fp32 output accumulator in SHM.
  const unsigned base_row_dn = blockIdx.x * CoreDims::W_DOWN_TILE;
  for (unsigned idx = threadIdx.x; idx < Dims::BS * CoreDims::W_DOWN_TILE;
       idx += blockDim.x)
    ((T_element*)shm->out_accum)[idx] = 0.f;
  __syncthreads();

  // ── Phase 4: Down-projection — pipelined 4-stage design ──────────────
  // For each expert: fetch bf16 intermediate → SHM, quantize bf16→fp8,
  // MMA fp8×fp8 with w_down, all pipelined with double-buffering.
  moe_down_projection_BS8_allexperts<Dims>(
      expert_weights_down, expert_scales_down, top_k, batch_size, spec, shmem);

  // ── Phase 5: Writeback SHM fp32 accumulator → global bf16 output ────────
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
  moe_scale_activation_BSx<Dims>(activations_in, token_count, spec, shmem);

  // Step 3b: for each sorted slot, store act_scale in token_weights (for
  // up-proj inside silu) and routing_weight in topk_weights_flat (for
  // down-proj). token_weights[sorted_pos] was set to routing_weight in prepare
  // step.
  {
    const uint32_t virtual_batch = token_count * top_k;
    for (uint32_t sp = threadIdx.x; sp < virtual_batch; sp += blockDim.x) {
      uint32_t tok = shmem->path.bs64.token_indexes_topk[sp];
      float rw =
          shmem->path.bs64.token_weights[sp];   // routing_weight (from prepare)
      float as = shmem->act_scale[tok];         // act_scale (from step 3)
      shmem->path.bs64.token_weights[sp] = as;  // act_scale for up-proj
      shmem->topk_weights_flat[sp] = rw;        // routing_weight for down-proj
    }
  }
  __syncthreads();

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
