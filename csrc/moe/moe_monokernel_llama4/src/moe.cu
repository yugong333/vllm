/**
 * This is the main file of the MoE monokernel.
 * It is designed so that you just need to build this file. It includes all
 * relevant implementations. For documentation of the main entry function
 * moe_kernel, see moe_interface.h
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

template <typename Dims>
__device__ void moe_kernel_BS64(const A_element* __restrict__ activations_in,
                                std::uint32_t batch_size,
                                const __nv_bfloat16* __restrict__ router_logits,
                                const W_element* __restrict expert_weights_up,
                                const S_element* __restrict expert_scales_up,
                                const W_element* __restrict expert_weights_down,
                                const S_element* __restrict expert_scales_down,
                                R_element* __restrict activations_out,
                                MoEGemmSpec<Dims>* __restrict__ spec,
                                MoE_SHM<Dims>* __restrict__ shmem) {
  if (is_calc_warp<Dims>()) {
    top1_BS64<Dims>(router_logits, batch_size, shmem);
  }
  __syncthreads();
  prepare_moe_BSx_Ey<Dims>(batch_size, shmem);
  __syncthreads();
  assert(shmem->experts[shmem->expert_count - 1].last_token == batch_size);

  moe_scale_activation_BSx<Dims>(activations_in, batch_size, spec, shmem);

#ifdef DEBUG_MOE
  if (blockIdx.x == 0) {
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
      spec->token_indexes[i] = shmem->token_indexes[i];
    }
  }
#endif

  moe_up_projection<Dims>(expert_weights_up, expert_scales_up, spec, shmem);
  cooperative_groups::this_grid().sync();
  moe_down_projection<Dims>(batch_size, expert_weights_down, expert_scales_down,
                            activations_out, spec, shmem);
}

template <typename Dims>
__device__ void moe_kernel_BS8(const A_element* __restrict__ activations_in,
                               std::uint32_t batch_size,
                               const __nv_bfloat16* __restrict__ router_logits,
                               const W_element* __restrict expert_weights_up,
                               const S_element* __restrict expert_scales_up,
                               const W_element* __restrict expert_weights_down,
                               const S_element* __restrict expert_scales_down,
                               R_element* __restrict activations_out,
                               MoEGemmSpec<Dims>* __restrict__ spec,
                               MoE_SHM<Dims>* __restrict__ shmem) {
  static_assert(Dims::BS <= 8);

  using CoreDims = MoECoreDims<Dims>;

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  if (is_prefetch_warp<Dims>()) {
    // Prefetch activations for rescaling
    const std::uint32_t warp = get_prefetch_warp<Dims>();
    for (std::uint32_t token = warp; token < batch_size;
         token += CoreDims::PREFETCH_WARP_COUNT) {
      moe_fetch_activation_async<Dims>(
          activations_in + token * Dims::HIDDEN_STATES,
          shmem->u.tiny.w[0].orig[token], pipe);
    }
  } else {
    top1_BS8<Dims>(router_logits, batch_size, shmem);
    sync_calc_threads<Dims>();
    prepare_moe_BS8<Dims>(batch_size, shmem);
  }

  cuda::pipeline_consumer_wait_prior<0>(pipe);
  __syncthreads();
  if (is_prefetch_warp<Dims>()) {
    //
    // Prefetch first expert weights
    //
    pipe.producer_acquire();

    // bring in first W tile
    moe_request_up_expert<Dims, Dims::HIDDEN_STATES>(
        expert_weights_up, shmem->expert_ids & 0xff, shmem->u.tiny.w[1].up,
        pipe);

    pipe.producer_commit();
  } else {
    //
    // Rescale activations
    //
    const std::uint32_t warp = get_calc_warp<Dims>();
    if (warp < batch_size) {
      moe_scale_activation_BS8<Dims>(shmem->u.tiny.w[0].orig[warp],
                                     (AQ_element*)shmem->u.tiny.a.up[warp],
                                     shmem->topk_weights[warp]);
    }
  }

  __syncthreads();

#ifdef DEBUG_MOE
  if (blockIdx.x == 0) {
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
      spec->token_indexes[i] = shmem->token_indexes[i];
    }
  }
#endif

  std::uint32_t w_index = moe_up_projection_tiny<Dims>(
      expert_weights_up, expert_scales_up, expert_weights_down,
      expert_scales_down, 1, spec, shmem, pipe);
  cooperative_groups::this_grid().sync();
  moe_down_projection_tiny<Dims>(batch_size, expert_weights_down,
                                 expert_scales_down, w_index, activations_out,
                                 spec, shmem, pipe);
}

template <typename Dims>
__global__ void moe_kernel(const A_element* __restrict__ activations_in,
                           std::uint32_t token_count,
                           const __nv_bfloat16* __restrict__ router_logits,
                           const W_element* __restrict__ expert_weights_up,
                           const S_element* __restrict__ expert_scales_up,
                           const W_element* __restrict__ expert_weights_down,
                           const S_element* __restrict__ expert_scales_down,
                           R_element* __restrict__ activations_out,
                           void* __restrict__ scratchpad,
                           size_t scratchpad_size, size_t shmem_size) {
  // we require 8 warps per SM and assume X to be the only relevant dimension
  assert(MoECoreDims<Dims>::THREADS_PER_WARP == 32);
  assert(blockDim.x == Dims::KernelConfig::BLOCK_SIZE);
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);

  assert(gridDim.x == Dims::KernelConfig::GRID_SIZE);
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);

  static_assert(Dims::M <= Dims_Max::M,
                "Dimension larger than the maximum supported dimension.");
  static_assert(Dims::N <= Dims_Max::N,
                "Dimension larger than the maximum supported dimension.");
  static_assert(Dims::K <= Dims_Max::K,
                "Dimension larger than the maximum supported dimension.");
  static_assert(Dims::NUM_EXPERTS <= Dims_Max::NUM_EXPERTS,
                "Dimension larger than the maximum supported dimension.");

  assert(token_count <= Dims::BS);
  assert(token_count > 0);

  assert((uintptr_t)scratchpad % alignof(MoEGemmSpec<Dims>) == 0);
  assert(scratchpad_size >= get_moe_scratchpad_size<Dims>());
  MoEGemmSpec<Dims>* spec = reinterpret_cast<MoEGemmSpec<Dims>*>(scratchpad);

  assert(shmem_size >= get_moe_shmem_size<Dims>());

  extern __shared__ char shmem_buffer[];
  MoE_SHM<Dims>* shmem = reinterpret_cast<MoE_SHM<Dims>*>(shmem_buffer);

  if constexpr (Dims::BS <= 8) {
    moe_kernel_BS8(activations_in, token_count, router_logits,
                   expert_weights_up, expert_scales_up, expert_weights_down,
                   expert_scales_down, activations_out, spec, shmem);
  } else {
    moe_kernel_BS64(activations_in, token_count, router_logits,
                    expert_weights_up, expert_scales_up, expert_weights_down,
                    expert_scales_down, activations_out, spec, shmem);
  }
}

/**
 * @brief Top-K MoE kernel — single-pass tiny path for BS <= 8.
 *
 * Mirrors moe_kernel_BS8 but handles top-K routing in a single pass:
 *
 *  1. Prefetch warps fetch activations while calc warps compute all K
 *     expert selections (stored in shmem->topk_ids_flat /
 *     shmem->topk_weights_flat) and build the unique-expert list.
 *  2. Activations are quantized once per token.
 *  3. The kernel iterates over unique experts (same as top-1 tiny path).
 *     For each expert, the up-projection runs on ALL tokens (same as top-1),
 *     but the output filter uses a K-slot scan instead of the single-byte
 *     expert_mask lookup:
 *       for k in 0..top_k: if topk_ids_flat[token*MAX_TOPK+k] == expert_id
 *     The matching routing weight is used to scale the result.
 *  4. Down-projection ADDS (+=) the weighted result into the output buffer
 *     instead of overwriting, since each token may receive contributions
 *     from multiple experts.
 *
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
          activations_in + tok * Dims::HIDDEN_STATES,
          shmem->u.tiny.w[0].orig[tok], pipe);
  } else {
    topK_BS8<Dims>(top_k, scoring_func, renormalize, router_logits, batch_size,
                   shmem);
    sync_calc_threads<Dims>();
    prepare_moe_topk_BS8<Dims>(batch_size, top_k, shmem);
  }
  cuda::pipeline_consumer_wait_prior<0>(pipe);
  __syncthreads();

  // ── Phase 2: prefetch first up-weights || quantize activations ──────────
  if (is_prefetch_warp<Dims>()) {
    pipe.producer_acquire();
    moe_request_up_expert<Dims, Dims::HIDDEN_STATES>(
        expert_weights_up, shmem->expert_ids & 0xff, shmem->u.tiny.w[1].up,
        pipe);
    pipe.producer_commit();
  } else {
    const std::uint32_t cw = get_calc_warp<Dims>();
    if (cw < batch_size) {
      // Quantize activations into shm->a.up. act_scale = max(|x|)/448.
      // Store act_scale in topk_weights[cw] — used inside silu (non-linear).
      float act_scale = 1.0f;  // TODO: we can avoid this computation
      moe_scale_activation_BS8<Dims>(shmem->u.tiny.w[0].orig[cw],
                                     (AQ_element*)shmem->u.tiny.a.up[cw],
                                     act_scale);
      if (get_thread<Dims>() == 0) shmem->topk_weights[cw] = act_scale;
      // Save quantized activations to spec->activations so we can restore
      // shm->a.up after each expert's down-projection overwrites it.
      const unsigned t = get_thread<Dims>();
      const unsigned chunk = CoreDims::THREADS_PER_WARP;
      for (unsigned col = t; col < Dims::HIDDEN_STATES; col += chunk)
        spec->activations[cw][col] = ((AQ_element*)shmem->u.tiny.a.up[cw])[col];
    }
  }
  __syncthreads();

  const unsigned thread = get_thread<Dims>();
  const unsigned warp = get_any_warp<Dims>();
  const unsigned base_row_up = blockIdx.x * CoreDims::W_UP_TILE / 2;
  const unsigned base_row_dn = blockIdx.x * CoreDims::W_DOWN_TILE;

  typename MoE_SHM<Dims>::U::TinyData* shm = &shmem->u.tiny;
  const std::uint32_t expert_count = shmem->expert_count;
  const std::uint64_t expert_ids_all = shmem->expert_ids;
  std::uint32_t w_index = 1;  // expert 0 up-weights in slot [1]

  // ── Per-expert loop: up → grid.sync → down → grid.sync ──────────────────
  // Flow mirrors the original top-1 BS8 path but repeated per expert:
  //   moe_up_projection_tiny  → spec->temp  (= per expert)
  //   grid.sync()
  //   moe_down_projection_tiny → activations_out (+=)
  //   grid.sync()
  //
  // shm->a.up (quantized activations) is preserved across all up-projections
  // because the down-projection reads spec->temp from global memory directly
  // (not via shm->a.down), avoiding the union conflict.
  for (std::uint32_t e = 0; e < expert_count; ++e) {
    const std::uint32_t id = (expert_ids_all >> (e * 8)) & 0xff;

    // ── Up-projection ───────────────────────────────────────────────────────
    const S_element* scales_up = expert_scales_up + id * 2 * Dims::N;
    float ws0 = scales_up[base_row_up + thread / 4];
    float ws1 = scales_up[base_row_up + thread / 4 + Dims::N];

    cuda::pipeline_consumer_wait_prior<0>(pipe);
    __syncthreads();

    if (is_prefetch_warp<Dims>()) {
      pipe.producer_acquire();
      if (e + 1 < expert_count) {
        const std::uint32_t nid = (expert_ids_all >> ((e + 1) * 8)) & 0xff;
        moe_request_up_expert<Dims, Dims::HIDDEN_STATES>(
            expert_weights_up, nid, shm->w[w_index ^ 1].up, pipe);
      }
      pipe.producer_commit();
    } else {
      float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
      for (unsigned bc = warp * CoreDims::K_TILE, i = 0;
           i < Dims::HIDDEN_STATES / CoreDims::BLOCK_STRIDE;
           ++i, bc += CoreDims::BLOCK_STRIDE) {
        unsigned r = thread / 4, c = 4 * (thread % 4);
        __nv_fp8x4_e4m3 w0 = *(__nv_fp8x4_e4m3*)&shm->w[w_index]
                                  .up[r + 0][rotate_col_32(bc + c + 0, r)];
        __nv_fp8x4_e4m3 w1 = *(__nv_fp8x4_e4m3*)&shm->w[w_index]
                                  .up[r + 8][rotate_col_32(bc + c + 0, r)];
        __nv_fp8x4_e4m3 w2 = *(__nv_fp8x4_e4m3*)&shm->w[w_index]
                                  .up[r + 0][rotate_col_32(bc + c + 16, r)];
        __nv_fp8x4_e4m3 w3 = *(__nv_fp8x4_e4m3*)&shm->w[w_index]
                                  .up[r + 8][rotate_col_32(bc + c + 16, r)];
        __nv_fp8x4_e4m3 a02 =
            *(__nv_fp8x4_e4m3*)&shm->a.up[r][rotate_col_32(bc + c + 0, r)];
        __nv_fp8x4_e4m3 a13 =
            *(__nv_fp8x4_e4m3*)&shm->a.up[r][rotate_col_32(bc + c + 16, r)];
        mma_fp8_fp8(d0, d1, d2, d3, w0, w1, w2, w3, a02, a13, d0, d1, d2, d3);
      }
      shm->partial_result.up[warp][thread + 0] = d0;
      shm->partial_result.up[warp][thread + 32] = d1;
      shm->partial_result.up[warp][thread + 64] = d2;
      shm->partial_result.up[warp][thread + 96] = d3;
    }
    __syncthreads();
    w_index ^= 1;

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
        float as = shmem->topk_weights[row];  // act_scale (stored in phase 2)
        float d0 = shm->partial_result.up[0][thread + warp * 32 + 0] +
                   shm->partial_result.up[1][thread + warp * 32 + 0];
        float d2 = shm->partial_result.up[0][thread + warp * 32 + 64] +
                   shm->partial_result.up[1][thread + warp * 32 + 64];
        for (unsigned i = 2; i < CoreDims::CALC_WARP_COUNT; i += 2) {
          d0 += shm->partial_result.up[i][thread + warp * 32 + 0] +
                shm->partial_result.up[i + 1][thread + warp * 32 + 0];
          d2 += shm->partial_result.up[i][thread + warp * 32 + 64] +
                shm->partial_result.up[i + 1][thread + warp * 32 + 64];
        }
        // act_scale inside silu (non-linear), routing_weight outside
        float x0 = d0 * as * ws0, w0v = d2 * as * ws1;
        spec->temp[row * Dims::N + (thread / 4) + base_row_up] =
            rw * (w0v * x0) / (1.f + expf(-x0));
      }
    }

    // All blocks finish writing spec->temp
    cooperative_groups::this_grid().sync();

    // ── Down-projection ─────────────────────────────────────────────────────
    // Prefetch: down weights + spec->temp (into shm->a.down, overwriting
    // shm->a.up)
    if (is_prefetch_warp<Dims>()) {
      pipe.producer_acquire();
      moe_request_down_expert_tiny<Dims>(
          expert_weights_down, expert_scales_down, id, shm, w_index, pipe);
      // Copy spec->temp for all tokens into shm->a.down
      const uint32_t pw = get_prefetch_warp<Dims>();
      for (uint32_t tok = pw; tok < batch_size;
           tok += CoreDims::PREFETCH_WARP_COUNT) {
        for (unsigned col = thread * 4, i = 0;
             i < Dims::N / (CoreDims::THREADS_PER_WARP * 4);
             i++, col += CoreDims::THREADS_PER_WARP * 4) {
          copy128(shm->a.down[tok][col], spec->temp[tok * Dims::N + col], pipe);
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
      moe_down_mult<Dims>(shm->w[w_index].down, shm->scale[w_index],
                          shm->a.down[thread / 4], s0, s1,
                          shm->partial_result.down);
    }
    __syncthreads();

    // Restore shm->a.up from spec->activations (overwritten by shm->a.down
    // above) Needed for the next expert's up-projection.
    if (e + 1 < expert_count) {
      if (!is_prefetch_warp<Dims>()) {
        const std::uint32_t cw = get_calc_warp<Dims>();
        if (cw < batch_size) {
          for (unsigned col = thread; col < Dims::HIDDEN_STATES;
               col += CoreDims::THREADS_PER_WARP)
            ((AQ_element*)shmem->u.tiny.a.up[cw])[col] =
                spec->activations[cw][col];
        }
      }
      __syncthreads();
    }

    // Reduce and accumulate += into activations_out
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
          unsigned c0 =
              row0 * Dims::HIDDEN_STATES + (thread / 4) + base_row_dn + wr;
          activations_out[c0 + 0] =
              (R_element)((float)activations_out[c0 + 0] + d0);
          if (CoreDims::W_DOWN_TILE % 16 == 0 || wr + 8 < CoreDims::W_DOWN_TILE)
            activations_out[c0 + 8] =
                (R_element)((float)activations_out[c0 + 8] + d2);
        }
        if (s1) {
          unsigned c1 =
              row1 * Dims::HIDDEN_STATES + (thread / 4) + base_row_dn + wr;
          activations_out[c1 + 0] =
              (R_element)((float)activations_out[c1 + 0] + d1);
          if (CoreDims::W_DOWN_TILE % 16 == 0 || wr + 8 < CoreDims::W_DOWN_TILE)
            activations_out[c1 + 8] =
                (R_element)((float)activations_out[c1 + 8] + d3);
        }
      }
    }

    // Grid sync before next expert's up-projection
    cooperative_groups::this_grid().sync();
  }
}

/**
 * @brief Top-K MoE kernel with configurable scoring and renormalization.
 *
 * For BS <= 8: single-pass tiny path — routing computed once, all unique
 * experts processed in one loop, weighted results accumulated in output.
 *
 * For BS > 8: K-iteration path — routing computed once (k==0), then the
 * existing BS64 pipeline runs K times with the k-th expert selection,
 * accumulating results.
 *
 * In both cases all routing results live in shared memory
 * (shmem->topk_ids_flat / shmem->topk_weights_flat) — no global memory
 * MoETopKSpec needed.
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

  using CoreDims = MoECoreDims<Dims>;

  // Scratchpad holds only MoEGemmSpec — routing results live in shmem
  MoEGemmSpec<Dims>* spec = reinterpret_cast<MoEGemmSpec<Dims>*>(scratchpad);

  extern __shared__ char shmem_buffer[];
  MoE_SHM<Dims>* shmem = reinterpret_cast<MoE_SHM<Dims>*>(shmem_buffer);

  if constexpr (Dims::BS <= 8) {
    // ===== Single-pass tiny path ============================================
    // Zero output before accumulation
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
         i < token_count * Dims::HIDDEN_STATES; i += blockDim.x * gridDim.x) {
      activations_out[i] = (__nv_bfloat16)0.0f;
    }
    cooperative_groups::this_grid().sync();

    moe_kernel_topk_BS8<Dims>(
        activations_in, token_count, router_logits, expert_weights_up,
        expert_scales_up, expert_weights_down, expert_scales_down,
        activations_out, top_k, scoring_func, renormalize, spec, shmem);

  } else {
    // ===== BS64 single-pass path =============================================
    // Zero output before accumulation (each token receives contributions from
    // multiple experts via +=)
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
         i < token_count * Dims::HIDDEN_STATES; i += blockDim.x * gridDim.x) {
      activations_out[i] = (__nv_bfloat16)0.0f;
    }
    cooperative_groups::this_grid().sync();

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

    // Step 3: quantize activations once per original token
    // moe_scale_activation_BSx folds act_scale into shmem->topk_weights[i],
    // giving topk_weights[i] = routing_weight_i * act_scale_i.
    moe_scale_activation_BSx<Dims>(activations_in, token_count, spec, shmem);

    // Step 3b: update token_weights[sorted_pos] to contain act_scale only
    // (not routing_weight), so that moe_up_projection_topk applies act_scale
    // inside silu correctly. routing_weight stays in topk_weights_flat and
    // is applied after down-projection in moe_down_reduction_topk.
    //
    // After moe_scale_activation_BSx: shmem->topk_weights[tok] = rw * as
    // token_weights[sorted_pos] was set to topk_weights_flat[virtual_row]
    //   = raw routing_weight (no act_scale)
    //
    // We want token_weights[sorted_pos] = act_scale_for_original_token
    // act_scale = topk_weights[tok] / routing_weight
    //           = (rw * as) / rw = as
    // But division is fragile. Instead: overwrite token_weights with
    // shmem->topk_weights[original_token] / topk_weights_flat[virtual_row].
    // Simpler: just store act_scale per token in a scratch location.
    //
    // We use spec->topk_weights_scaled[tok] which already = rw * as.
    // And topk_weights_flat[tok*MAX_TOPK+k] = rw.
    // So act_scale = spec->topk_weights_scaled[tok] /
    // topk_weights_flat[tok*MAX_TOPK+k]. But this requires knowing which k-slot
    // maps to which sorted_pos.
    //
    // Simplest: iterate over all sorted positions and compute act_scale.
    if (threadIdx.x < MoE_SHM<Dims>::MAX_TOPK * token_count) {
      // virtual_row i: original_token = i / top_k, k_slot = i % top_k
      // token_weights[sorted_pos] was set from topk_weights_flat[i]
      // We need to find sorted_pos for virtual_row i.
      // But we don't have that mapping here.
      // Alternative: just update token_weights after the sort by iterating
      // over sorted positions and looking up the original token.
    }
    // NOTE: The above approach is complex. Instead, we update token_weights
    // directly in a separate pass using token_indexes_topk (which maps
    // sorted_pos -> original_token) and topk_weights_flat (routing weights).
    // act_scale[tok] = spec->topk_weights_scaled[tok] /
    // topk_weights[tok_original] But topk_weights[tok] was overwritten by
    // moe_scale_activation_BSx.
    //
    // FINAL APPROACH: store act_scale per token in spec->activations scratch,
    // then update token_weights in a grid-cooperative pass.
    // Actually the simplest: just divide spec->topk_weights_scaled[tok] by
    // the original routing weight stored in topk_weights_flat.
    // We need to iterate over sorted positions.
    {
      const uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;
      const uint32_t virtual_batch = token_count * top_k;
      // For each sorted position, compute act_scale and store in token_weights.
      // token_indexes_topk[sorted_pos] = original_token
      // We need the routing weight for this (token, k) pair.
      // topk_weights_flat[tok * MAX_TOPK + k] = routing_weight, but we don't
      // know k from sorted_pos alone.
      // HOWEVER: spec->topk_weights_scaled[tok] = routing_weight * act_scale
      // and token_weights[sorted_pos] = routing_weight (set in prepare step).
      // So: act_scale = spec->topk_weights_scaled[tok] /
      // token_weights[sorted_pos] This is safe as long as routing_weight != 0.
      for (uint32_t sp = threadIdx.x; sp < virtual_batch; sp += blockDim.x) {
        uint32_t tok = shmem->token_indexes_topk[sp];
        float rw = shmem->token_weights[sp];  // routing_weight
        float rw_as =
            spec->topk_weights_scaled[tok];  // routing_weight * act_scale
        // act_scale = rw_as / rw  (safe: rw > 0 for selected experts)
        shmem->token_weights[sp] = (rw > 0.f) ? (rw_as / rw) : 0.f;
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
}

}  // namespace moe_monokernel
