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
 * @brief Top-K MoE kernel — single-pass tiny path for BS <= 8.
 *
 * Uses separate w_up / w_down SMEM buffers (no aliasing union) so that
 * memory transfers and compute can be fully overlapped:
 *
 *  Phase 1: Prefetch warps fetch activations || calc warps do routing + topK.
 *
 *  Phase 2: Prefetch warps fetch expert[0] up-weights into w_up ||
 *           calc warps quantize activations into a_up (persistent in SHM).
 *
 *  Per-expert loop (fully pipelined):
 *    - wait for w_up ready
 *    - prefetch warps: start loading w_down for THIS expert  (overlaps
 * up-compute)
 *    - calc warps:     up-projection MMA using w_up → spec->temp
 *    - grid.sync()  (all blocks finish spec->temp)
 *    - prefetch warps: start loading w_up for NEXT expert    (overlaps
 * down-compute)
 *    - calc warps:     down-projection MMA using w_down → accumulate
 * activations_out
 *    - grid.sync()
 *    - a_up is persistent in SHM — no restore needed
 *
 * SMEM saving vs. old design: one w_up buffer instead of w[2] saves ~80 KB.
 * The output buffer must be zeroed before calling this function.
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
  constexpr uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  // ── Phase 1: prefetch activations || routing ────────────────────────────
  if (is_prefetch_warp<Dims>()) {
    const std::uint32_t pw = get_prefetch_warp<Dims>();
    for (std::uint32_t tok = pw; tok < batch_size;
         tok += CoreDims::PREFETCH_WARP_COUNT)
      moe_fetch_activation_async<Dims>(
          activations_in + tok * Dims::HIDDEN_STATES, shmem->u.tiny.orig[tok],
          pipe);
  } else {
    topK_BS8<Dims>(top_k, scoring_func, renormalize, router_logits, batch_size,
                   shmem);
    sync_calc_threads<Dims>();
    prepare_moe_topk_BS8<Dims>(batch_size, top_k, shmem);
  }
  cuda::pipeline_consumer_wait_prior<0>(pipe);
  __syncthreads();

  // ── Phase 2: prefetch expert[0] up-weights into w_up || quantize activations
  if (is_prefetch_warp<Dims>()) {
    pipe.producer_acquire();
    moe_request_up_expert<Dims, Dims::HIDDEN_STATES>(
        expert_weights_up, shmem->experts[0].id, shmem->u.tiny.w_up, pipe);
    pipe.producer_commit();
  } else {
    const std::uint32_t cw = get_calc_warp<Dims>();
    if (cw < batch_size) {
      float act_scale = moe_scale_activation_BS8<Dims>(shmem->u.tiny.orig[cw],
                                                       shmem->u.tiny.a_up[cw]);
      if (get_thread<Dims>() == 0) shmem->act_scale[cw] = act_scale;
    }
  }
  __syncthreads();

  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_any_warp<Dims>();
  const unsigned base_row_up = blockIdx.x * CoreDims::W_UP_TILE / 2;
  const unsigned base_row_dn = blockIdx.x * CoreDims::W_DOWN_TILE;

  typename MoE_SHM<Dims>::U::TinyData* shm = &shmem->u.tiny;
  const std::uint32_t expert_count = shmem->expert_count;

  // ── Per-expert loop ──────────────────────────────────────────────────────
  // Pipeline: w_up is ready at loop entry (prefetched in phase 2 or at end of
  // previous iteration's down-phase).  During up-compute, prefetch w_down.
  // During down-compute, prefetch next expert's w_up.

  // Zero the per-block fp32 output accumulator in SHM.
  for (unsigned idx = threadIdx.x; idx < Dims::BS * CoreDims::W_DOWN_TILE;
       idx += blockDim.x)
    ((T_element*)shm->out_accum)[idx] = 0.f;
  __syncthreads();

  for (std::uint32_t e = 0; e < expert_count; ++e) {
    const std::uint32_t id = shmem->experts[e].id;

    // ── Up-projection: w_up is ready; prefetch w_down in parallel ───────────
    const S_element* scales_up = expert_scales_up + id * 2 * Dims::N;
    float ws0 = scales_up[base_row_up + thread / 4];
    float ws1 = scales_up[base_row_up + thread / 4 + Dims::N];

    // Wait for w_up (issued in phase 2 or at end of previous down-phase).
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    __syncthreads();

    if (is_prefetch_warp<Dims>()) {
      // Prefetch THIS expert's down-weights into w_down while calc warps
      // compute the up-projection.  w_down is a separate buffer so no aliasing.
      pipe.producer_acquire();
      moe_request_down_expert_tiny<Dims>(expert_weights_down,
                                         expert_scales_down, id, shm, 0, pipe);
      pipe.producer_commit();
    } else {
      float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
      for (unsigned bc = warp * CoreDims::K_TILE, i = 0;
           i < Dims::HIDDEN_STATES / CoreDims::BLOCK_STRIDE;
           ++i, bc += CoreDims::BLOCK_STRIDE) {
        unsigned r = thread / 4, c = 4 * (thread % 4);
        __nv_fp8x4_e4m3 w0 =
            *(__nv_fp8x4_e4m3*)&shm->w_up[r + 0][rotate_col_32(bc + c + 0, r)];
        __nv_fp8x4_e4m3 w1 =
            *(__nv_fp8x4_e4m3*)&shm->w_up[r + 8][rotate_col_32(bc + c + 0, r)];
        __nv_fp8x4_e4m3 w2 =
            *(__nv_fp8x4_e4m3*)&shm->w_up[r + 0][rotate_col_32(bc + c + 16, r)];
        __nv_fp8x4_e4m3 w3 =
            *(__nv_fp8x4_e4m3*)&shm->w_up[r + 8][rotate_col_32(bc + c + 16, r)];
        __nv_fp8x4_e4m3 a02 =
            *(__nv_fp8x4_e4m3*)&shm->a_up[r][rotate_col_32(bc + c + 0, r)];
        __nv_fp8x4_e4m3 a13 =
            *(__nv_fp8x4_e4m3*)&shm->a_up[r][rotate_col_32(bc + c + 16, r)];
        mma_fp8_fp8(d0, d1, d2, d3, w0, w1, w2, w3, a02, a13, d0, d1, d2, d3);
      }
      // MMA fragment layout (m16n8k16, row.col):
      //   d0 = C[t/4,   2*(t%4)]     gate · tok_even
      //   d1 = C[t/4,   2*(t%4)+1]   gate · tok_odd
      //   d2 = C[t/4+8, 2*(t%4)]     up   · tok_even
      //   d3 = C[t/4+8, 2*(t%4)+1]   up   · tok_odd
      // Swap d1↔d2 so layout becomes:
      //   d0 = gate · tok_even   (slot +0)
      //   d1 = up   · tok_even   (slot +32)
      //   d2 = gate · tok_odd    (slot +64)
      //   d3 = up   · tok_odd    (slot +96)
      {
        float tmp = d1;
        d1 = d2;
        d2 = tmp;
      }
      shm->partial_result.up[warp][thread + 0] = d0;
      shm->partial_result.up[warp][thread + 32] = d1;
      shm->partial_result.up[warp][thread + 64] = d2;
      shm->partial_result.up[warp][thread + 96] = d3;
    }
    __syncthreads();

    // Reduce → write spec->temp (=) for tokens assigned to this expert
    if (warp < 2) {
      const std::uint32_t row = (thread % 4) * 2 + warp;
      bool store = false;
      float rw = 0.f;
      if (row < batch_size) {
        for (uint32_t k = 0; k < top_k; k++) {
          if (shmem->topk_ids_flat[row * MAX_TOPK + k] == (uint8_t)id) {
            store = true;
            rw = shmem->topk_weights_flat[row * MAX_TOPK + k];
            break;
          }
        }
      }
      if (store) {
        float as = shmem->act_scale[row];  // act_scale for this token
        // After d1↔d2 swap, partial_result layout per calc warp is:
        //   [thread +  0] = gate · tok_even
        //   [thread + 32] = up   · tok_even
        //   [thread + 64] = gate · tok_odd
        //   [thread + 96] = up   · tok_odd
        // Reduction warp 0 handles even tokens (row = (t%4)*2),
        //           warp 1 handles odd  tokens (row = (t%4)*2+1).
        // gate offset = warp * 64, up offset = warp * 64 + 32
        float d0 = shm->partial_result.up[0][thread + warp * 64 + 0] +
                   shm->partial_result.up[1][thread + warp * 64 + 0];
        float d2 = shm->partial_result.up[0][thread + warp * 64 + 32] +
                   shm->partial_result.up[1][thread + warp * 64 + 32];
        for (unsigned i = 2; i < CoreDims::CALC_WARP_COUNT; i += 2) {
          d0 += shm->partial_result.up[i][thread + warp * 64 + 0] +
                shm->partial_result.up[i + 1][thread + warp * 64 + 0];
          d2 += shm->partial_result.up[i][thread + warp * 64 + 32] +
                shm->partial_result.up[i + 1][thread + warp * 64 + 32];
        }
        // act_scale inside silu (non-linear), routing_weight outside
        float x0 = d0 * as * ws0, w0v = d2 * as * ws1;
        // Guard: blocks beyond N have no valid up-proj columns to write.
        // GRID_SIZE is sized for the down-proj (K columns) but the up-proj
        // only has N columns.  Skip the write for out-of-range blocks.
        if ((thread / 4) + base_row_up < Dims::N) {
          spec->temp[row * Dims::N + (thread / 4) + base_row_up] =
              rw * (w0v * x0) / (1.f + expf(-x0));
        }
      }
    }

    // All blocks finish writing spec->temp
    cooperative_groups::this_grid().sync();

    // ── Down-projection: w_down is ready; prefetch next w_up in parallel ────
    // Wait for w_down (issued by prefetch warps during up-compute above).
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    __syncthreads();

    if (is_prefetch_warp<Dims>()) {
      // Prefetch NEXT expert's up-weights into w_up while calc warps compute
      // the down-projection.  w_up is now free (up-compute finished).
      pipe.producer_acquire();
      if (e + 1 < expert_count) {
        const std::uint32_t nid = shmem->experts[e + 1].id;
        moe_request_up_expert<Dims, Dims::HIDDEN_STATES>(expert_weights_up, nid,
                                                         shm->w_up, pipe);
      }
      // Copy spec->temp for all tokens into shm->a_down (always needed)
      const uint32_t pw = get_prefetch_warp<Dims>();
      for (uint32_t tok = pw; tok < batch_size;
           tok += CoreDims::PREFETCH_WARP_COUNT) {
        for (unsigned col = thread * 4, i = 0;
             i < Dims::N / (CoreDims::THREADS_PER_WARP * 4);
             i++, col += CoreDims::THREADS_PER_WARP * 4) {
          copy128(shm->a_down[tok][col], spec->temp[tok * Dims::N + col], pipe);
        }
      }
      pipe.producer_commit();
    }
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    __syncthreads();

    // Zero partial_result.down (stale from up-projection union)
    if (!is_prefetch_warp<Dims>()) {
      for (unsigned wr = warp * CoreDims::W_DOWN_MMA_TILE;
           wr < CoreDims::W_DOWN_TILE;
           wr += CoreDims::W_DOWN_MMA_TILE * CoreDims::TOTAL_WARP_COUNT) {
        shm->partial_result.down[wr / 2][thread + 0] = 0.f;
        shm->partial_result.down[wr / 2][thread + 32] = 0.f;
        shm->partial_result.down[wr / 2][thread + 64] = 0.f;
        shm->partial_result.down[wr / 2][thread + 96] = 0.f;
      }
    }
    __syncthreads();

    if (!is_prefetch_warp<Dims>()) {
      const std::uint32_t row0 = (thread % 4) * 2 + 0;
      const std::uint32_t row1 = (thread % 4) * 2 + 1;
      bool s0 = false, s1 = false;
      if (row0 < batch_size)
        for (uint32_t k = 0; k < top_k; k++)
          if (shmem->topk_ids_flat[row0 * MAX_TOPK + k] == (uint8_t)id) {
            s0 = true;
            break;
          }
      if (row1 < batch_size)
        for (uint32_t k = 0; k < top_k; k++)
          if (shmem->topk_ids_flat[row1 * MAX_TOPK + k] == (uint8_t)id) {
            s1 = true;
            break;
          }
      moe_down_mult<Dims>(shm->w_down, shm->scale_down, shm->a_down[thread / 4],
                          s0, s1, shm->partial_result.down);
    }
    __syncthreads();

    // a_up is now a separate persistent buffer — no restore needed.

    // Reduce and accumulate += into SHM out_accum (fp32, no precision loss)
    {
      const std::uint32_t row0 = (thread % 4) * 2 + 0;
      const std::uint32_t row1 = (thread % 4) * 2 + 1;
      bool s0 = false, s1 = false;
      if (row0 < batch_size)
        for (uint32_t k = 0; k < top_k; k++)
          if (shmem->topk_ids_flat[row0 * MAX_TOPK + k] == (uint8_t)id) {
            s0 = true;
            break;
          }
      if (row1 < batch_size)
        for (uint32_t k = 0; k < top_k; k++)
          if (shmem->topk_ids_flat[row1 * MAX_TOPK + k] == (uint8_t)id) {
            s1 = true;
            break;
          }
      for (unsigned wr = warp * CoreDims::W_DOWN_MMA_TILE;
           wr < CoreDims::W_DOWN_TILE;
           wr += CoreDims::W_DOWN_MMA_TILE * CoreDims::TOTAL_WARP_COUNT) {
        float d0 = shm->partial_result.down[wr / 2][thread + 0];
        float d1 = shm->partial_result.down[wr / 2][thread + 32];
        float d2 = shm->partial_result.down[wr / 2][thread + 64];
        float d3 = shm->partial_result.down[wr / 2][thread + 96];
        for (unsigned i = 1; i < CoreDims::CALC_WARP_COUNT; ++i) {
          d0 += shm->partial_result.down[wr / 2 + i][thread + 0];
          d1 += shm->partial_result.down[wr / 2 + i][thread + 32];
          d2 += shm->partial_result.down[wr / 2 + i][thread + 64];
          d3 += shm->partial_result.down[wr / 2 + i][thread + 96];
        }
        if (s0) {
          shm->out_accum[row0][(thread / 4) + wr + 0] += d0;
          if (CoreDims::W_DOWN_TILE % 16 == 0 || wr + 8 < CoreDims::W_DOWN_TILE)
            shm->out_accum[row0][(thread / 4) + wr + 8] += d2;
        }
        if (s1) {
          shm->out_accum[row1][(thread / 4) + wr + 0] += d1;
          if (CoreDims::W_DOWN_TILE % 16 == 0 || wr + 8 < CoreDims::W_DOWN_TILE)
            shm->out_accum[row1][(thread / 4) + wr + 8] += d3;
        }
      }
    }

    // Grid sync before next expert's up-projection
    cooperative_groups::this_grid().sync();
  }

  // ── Final writeback: SHM fp32 accumulator → global bf16 output ──────────
  // Each block writes its W_DOWN_TILE columns for all active tokens.
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
