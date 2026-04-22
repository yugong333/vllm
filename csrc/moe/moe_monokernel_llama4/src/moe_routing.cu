
#pragma once
#ifndef MOE_GATING_CU
  #define MOE_GATING_CU

  #ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
    #error Do not include this file directly.
  #endif

  #include <cstdint>
  #include <cfloat>
  #include <cuda_bf16.h>

  #include "moe_internal.h"

namespace moe_monokernel {

/**
 * @brief Determines the thread with the lowest index that has a specific value.
 *
 * This device function checks, within a warp, which threads have their @p
 * haystack value equal to the specified @p needle value. If there are multiple
 * such threads, returns the one with the lowest thread index.
 *
 * @param haystack The value to compare against the needle for the calling
 * thread.
 * @param needle The value to search for among the threads in the warp.
 * @return uint32_t Thread index
 */
__device__ static inline uint32_t warp_who_has(float haystack, float needle) {
  uint32_t mask = __ballot_sync(0xFFFFFFFFU, haystack == needle);
  assert(mask > 0);
  return __ffs(mask) - 1;  // Nvidia starts counting bits at 1
}

/**
 * @brief Computes the sigmoid activation function for a given input.
 *
 * @param x The input value.
 * @return The sigmoid of the input value.
 */
__device__ static inline float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

/**
 * @brief Selects the top-1 expert for each of up to 64 tokens based on router
 * logits.
 *
 * This device function processes a batch of up to 64 tokens, each with up to 16
 * experts. It takes as input a pointer to the router logits (in bfloat16
 * format) and determines, for each token, the expert with the highest routing
 * score.
 *
 * The selection is done on each CUDA block redundantly such that the result can
 * be placed in shared memory.
 *
 * @param router_logits Pointer to the input router logits array of shape
 * [num_tokens, experts] in row-major order. Individual elements are in
 * __nv_bfloat16 format.
 * @param num_tokens Number of tokens
 * @param shmem Shared Memory struct to store the result to.
 */
template <typename Dims>
__device__ static void top1_BS64_E16(
    const __nv_bfloat16* __restrict__ router_logits, uint32_t num_tokens,
    MoE_SHM<Dims>* shmem) {
  static_assert(Dims::NUM_EXPERTS <= 16,
                "Dispatch to incorrect implementation");
  static_assert(Dims::BS <= 64, "Dispatch to incorrect implementation");
  static_assert(Dims::BS * Dims::NUM_EXPERTS < UINT32_MAX,
                "Batch size or number of experts too high for uint32 indices.");

  uint32_t thread_idx = threadIdx.x;

  // On all SMs in parallel: Every thread does one token
  for (uint32_t tokidx = thread_idx; tokidx < num_tokens; tokidx += 256) {
    float max_value = -FLT_MAX;
    uint32_t max_index = 0;
    for (uint32_t idx = 0; idx < Dims::NUM_EXPERTS; idx++) {
      uint32_t index = tokidx * Dims::NUM_EXPERTS +
                       idx;  // Make NVCC produce simpler array indexing code
      float value = (float)router_logits[index];
      // Branchless version of:
      // if (max_value < value) {
      //     max_index = idx;
      //     max_value = value;
      // }
      max_value = fmaxf(max_value, value);
      int is_new = max_value == value;
      max_index = max_index * (1 - is_new) + idx * is_new;
    }

    shmem->topk_ids[tokidx] = (uint8_t)max_index;
    shmem->topk_weights[tokidx] = sigmoid(max_value);
  }
}

/**
 * @brief Selects the top-1 expert for each of up to 64 tokens based on router
 * logits.
 *
 * This device function processes a batch of up to 64 tokens, each with up to
 * 128 experts. It takes as input a pointer to the router logits (in bfloat16
 * format) and determines, for each token, the expert with the highest routing
 * score.
 *
 * It should work for any number of experts, but is tested for performance for
 * up to 128 experts only.
 *
 * The selection is done on each CUDA block redundantly such that the result can
 * be placed in shared memory.
 *
 * @param router_logits Pointer to the input router logits array of shape
 * [num_tokens, experts] in row-major order. Individual elements are in
 * __nv_bfloat16 format.
 * @param num_tokens Number of tokens
 * @param shmem Shared Memory struct to store the result to.
 */
template <typename Dims>
__device__ static void top1_BS64_E128(
    const __nv_bfloat16* __restrict__ router_logits, uint32_t num_tokens,
    MoE_SHM<Dims>* shmem) {
  static_assert(Dims::BS <= 64, "Dispatch to incorrect implementation");
  // This function does one token with multiple threads.
  // The following constants define the per-token parallelism:
  constexpr uint32_t NUM_EXPERTS_PER_THREAD = 16;  // Best performance for BS=64
  constexpr uint32_t NUM_THREADS_PER_TOKEN =
      Dims::NUM_EXPERTS / NUM_EXPERTS_PER_THREAD;
  constexpr uint32_t NUM_TOKENS_PER_WARP = 32 / NUM_THREADS_PER_TOKEN;

  static_assert(Dims::BS * Dims::NUM_EXPERTS < UINT32_MAX,
                "Batch size or number of experts too high for uint32 indices.");
  static_assert(Dims::NUM_EXPERTS % NUM_THREADS_PER_TOKEN == 0,
                "Number of experts must be divisible.");

  uint32_t thread_idx = threadIdx.x;

  // Ensure participation of whole warps in the loop
  int padded_num_tokens = (num_tokens + NUM_TOKENS_PER_WARP - 1) /
                          NUM_TOKENS_PER_WARP * NUM_TOKENS_PER_WARP;

  // On all SMs in parallel: NUM_THREADS_PER_TOKEN threads together do one token
  for (uint32_t tokidx = thread_idx / NUM_THREADS_PER_TOKEN;
       tokidx < padded_num_tokens; tokidx += 256 / NUM_THREADS_PER_TOKEN) {
    // Check really full warps participate
    assert(__activemask() == 0xFFFFFFFFU);

    float max_value = -FLT_MAX;
    uint32_t max_index = 0;

    if (tokidx < num_tokens) {
      const uint32_t part_idx = thread_idx % NUM_THREADS_PER_TOKEN;
      for (uint32_t idx = part_idx * NUM_EXPERTS_PER_THREAD;
           idx < (part_idx + 1) * NUM_EXPERTS_PER_THREAD; idx++) {
        uint32_t index = tokidx * Dims::NUM_EXPERTS +
                         idx;  // Make NVCC produce simpler array indexing code
        float value = (float)router_logits[index];
        // Branchless version of:
        // if (max_value < value) {
        //     max_index = idx;
        //     max_value = value;
        // }
        max_value = fmaxf(max_value, value);
        int is_new = max_value == value;
        max_index = max_index * (1 - is_new) + idx * is_new;
      }
    }

    // In each warp: synchronize between those threads that do the same token
    float quad_max_value = max_value;
    for (int i = 1; i < NUM_THREADS_PER_TOKEN; i *= 2)
      quad_max_value = fmaxf(
          quad_max_value, __shfl_xor_sync(0xFFFFFFFFU, quad_max_value, i, 32));

    // ATTENTION: Here, potentially multiple threads can write to the same shmem
    // values. But the writes themselves are atomic and if multiple write, they
    // have the same max_value. In this case, one of the maximum topk_values is
    // chosen indeterministically.
    if (quad_max_value == max_value) {
      shmem->topk_ids[tokidx] = (uint8_t)max_index;
      shmem->topk_weights[tokidx] = 1.0f / (1.0f + std::exp(-max_value));
    }
  }
}

/**
 * @brief Selects the top-1 expert for each of up to 64 tokens based on router
 * logits.
 *
 * It takes as input a pointer to the router logits (in bfloat16 format) and
 * determines, for each token, the expert with the highest routing score.
 *
 * The selection is done on each CUDA block redundantly such that the result can
 * be placed in shared memory.
 *
 * @param router_logits Pointer to the input router logits array of shape
 * [num_tokens, experts] in row-major order. Individual elements are in
 * __nv_bfloat16 format.
 * @param num_tokens Number of tokens
 * @param shmem Shared Memory struct to store the result to.
 */
template <typename Dims>
__device__ __forceinline__ void top1_BS64(
    const __nv_bfloat16* __restrict__ router_logits, uint32_t num_tokens,
    MoE_SHM<Dims>* shmem) {
  static_assert(Dims::BS <= 64, "Dispatch to incorrect implementation");
  if constexpr (Dims::NUM_EXPERTS <= 16) {
    top1_BS64_E16(router_logits, num_tokens, shmem);
  } else {
    top1_BS64_E128(router_logits, num_tokens, shmem);
  }
}

/**
 * @brief Selects the top-1 expert for each of up to 8 tokens based on router
 * logits.
 *
 * This device function processes a batch of up to 8 tokens, each with up to 128
 * experts. It takes as input a pointer to the router logits (in bfloat16
 * format) and determines, for each token, the expert with the highest routing
 * score.
 *
 * It should work for any number of experts, but is tested for performance for
 * up to 128 experts only.
 *
 * The selection is done on each CUDA block redundantly such that the result can
 * be placed in shared memory. In case, num_tokens is smaller than 8, pads
 * topk_ids to 8 by setting the remaining elements to 0xFF.
 *
 * @param router_logits Pointer to the input router logits array of shape
 * [num_tokens, experts] in row-major order. Individual elements are in
 * __nv_bfloat16 format.
 * @param num_tokens Number of tokens
 * @param shmem Shared Memory struct to store the result to.
 */
template <typename Dims>
__device__ void top1_BS8(const __nv_bfloat16* __restrict__ router_logits,
                         uint32_t num_tokens, MoE_SHM<Dims>* shmem) {
  static_assert(Dims::BS <= 8, "Dispatch to incorrect implementation");
  static_assert(Dims::BS * Dims::NUM_EXPERTS < UINT32_MAX,
                "Batch size or number of experts too high for uint32 indices.");

  uint32_t warp_idx = get_calc_warp<Dims>();
  uint32_t thread_idx_within_warp = get_thread<Dims>();

  // On all SMs in parallel: Every warp does one token
  if (warp_idx < num_tokens) {
    float max_value = -FLT_MAX;
    uint32_t max_index = 0;
    for (uint32_t idx = thread_idx_within_warp; idx < Dims::NUM_EXPERTS;
         idx += 32) {
      uint32_t index = warp_idx * Dims::NUM_EXPERTS +
                       idx;  // Make NVCC produce simpler array indexing code
      float value = (float)router_logits[index];
      // if (std::isfinite(value)) {
      if constexpr (Dims::NUM_EXPERTS > 32) {
        if (max_value < value) {
          max_index = idx;
          max_value = value;
        }
      } else {
        max_index = idx;
        max_value = value;
      }
      // }
    }

    // Warp reduction
    float warpmax_value = warp_reduce_max_float(max_value);
    assert(max_value <= warpmax_value);
    // We need this in case several threads have the exact same float.
    // In this case, we cannot simply use "warpmax_value == max_value" as
    // condition for the writing below.
    uint32_t max_thread = warp_who_has(max_value, warpmax_value);

    // Thread that has max value writes
    if (thread_idx_within_warp == max_thread) {
      assert(warpmax_value == max_value);
      assert(max_index < Dims::NUM_EXPERTS);
      shmem->topk_ids[warp_idx] = (uint8_t)max_index;
      shmem->topk_weights[warp_idx] = 1.0f / (1.0f + std::exp(-max_value));
    }
  } else if (warp_idx < 8) {
    // Padding elements
    if (thread_idx_within_warp == 0) {
      shmem->topk_ids[warp_idx] = (uint8_t)0xFF;
    }
  }
}

// =============================================================================
// Top-K routing with configurable scoring function and renormalization.
//
// All routing results are stored directly in shared memory
// (shmem->topk_ids_flat / shmem->topk_weights_flat) — no global memory
// needed.  Overhead: ~320 bytes for BS=8 top-K=8, ~2.5KB for BS=64 top-K=8,
// both negligible relative to the ~220KB MoE_SHM.
//
// Two-phase approach mirroring top1_BS8 / top1_BS64:
//
//  k_idx == 0: full scoring + top-K selection → shmem flat arrays
//              + copy 0th expert into topk_ids/topk_weights
//  k_idx > 0:  copy k_idx-th expert from flat arrays into
//              topk_ids/topk_weights (essentially free)
//
// The single-pass tiny path (moe_kernel_topk_BS8_singlepass) uses the flat
// arrays directly: iterates unique experts once, scans K slots per token
// to find the matching weight, and accumulates weighted results.
// =============================================================================

/**
 * @brief Warp-cooperative softmax over logits distributed across threads.
 *
 * Each thread holds @p count values.  On return every element of @p logits
 * is replaced by its softmax probability.
 */
__device__ static inline void warp_softmax_inplace(float* logits,
                                                   uint32_t count) {
  float local_max = -FLT_MAX;
  for (uint32_t i = 0; i < count; i++) local_max = fmaxf(local_max, logits[i]);
  float global_max = warp_reduce_max_float(local_max);

  float local_sum = 0.0f;
  for (uint32_t i = 0; i < count; i++) {
    logits[i] = __expf(logits[i] - global_max);
    local_sum += logits[i];
  }
  for (int off = 16; off >= 1; off /= 2)
    local_sum += __shfl_xor_sync(0xFFFFFFFFU, local_sum, off, 32);

  float inv_sum = 1.0f / local_sum;
  for (uint32_t i = 0; i < count; i++) logits[i] *= inv_sum;
}

/**
 * @brief Top-K expert selection for BS <= 8, writing results into shmem.
 *
 * Drop-in replacement for top1_BS8. Computes all K selections via warp
 * reduction and stores them in shmem->topk_ids_flat / shmem->topk_weights_flat.
 * Also copies the 0th expert into shmem->topk_ids / shmem->topk_weights so
 * the existing prepare / GEMM pipeline works for the first expert iteration.
 *
 * Must be called by calc warps only (same contract as top1_BS8).
 * Runs in parallel with prefetch warps fetching activations.
 */
template <typename Dims>
__device__ void topK_BS8(uint32_t top_k, ScoringFunc scoring_func,
                         bool renormalize,
                         const __nv_bfloat16* __restrict__ router_logits,
                         uint32_t num_tokens, MoE_SHM<Dims>* shmem) {
  static_assert(Dims::BS <= 8, "Dispatch to incorrect implementation");
  static_assert(Dims::BS * Dims::NUM_EXPERTS < UINT32_MAX,
                "Batch size or number of experts too high for uint32 indices.");

  constexpr uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;
  uint32_t warp_idx = get_calc_warp<Dims>();
  uint32_t tid = get_thread<Dims>();

  if (warp_idx < num_tokens) {
    // Load logits — one warp per token, same as top1_BS8
    constexpr uint32_t MAX_PER_THREAD = (Dims::NUM_EXPERTS + 31) / 32;
    float scores[MAX_PER_THREAD];
    uint32_t expert_id[MAX_PER_THREAD];
    uint32_t num_local = 0;

    for (uint32_t idx = tid; idx < Dims::NUM_EXPERTS; idx += 32) {
      scores[num_local] =
          (float)router_logits[warp_idx * Dims::NUM_EXPERTS + idx];
      expert_id[num_local] = idx;
      num_local++;
    }

    // Apply scoring function
    if (scoring_func == ScoringFunc::SOFTMAX) {
      warp_softmax_inplace(scores, num_local);
    } else {
      for (uint32_t i = 0; i < num_local; i++)
        scores[i] = 1.0f / (1.0f + __expf(-scores[i]));
    }

    // Iterative top-K selection via warp reduction — all K in one shot
    for (uint32_t k = 0; k < top_k; k++) {
      float max_val = -FLT_MAX;
      uint32_t max_expert = 0;
      for (uint32_t i = 0; i < num_local; i++) {
        if (scores[i] > max_val) {
          max_val = scores[i];
          max_expert = expert_id[i];
        }
      }

      float warp_max = warp_reduce_max_float(max_val);
      uint32_t winner = warp_who_has(max_val, warp_max);

      uint32_t winning_expert = __shfl_sync(0xFFFFFFFFU, max_expert, winner);
      float winning_weight = __shfl_sync(0xFFFFFFFFU, max_val, winner);

      // Winner writes to shmem flat cache
      if (tid == winner) {
        shmem->topk_ids_flat[warp_idx * MAX_TOPK + k] = (uint8_t)max_expert;
        shmem->topk_weights_flat[warp_idx * MAX_TOPK + k] = winning_weight;
      }

      // Mask out selected expert on every thread that owns it
      for (uint32_t i = 0; i < num_local; i++) {
        if (expert_id[i] == winning_expert) scores[i] = -FLT_MAX;
      }
    }

    // Optional renormalization (single thread per token)
    if (renormalize && tid == 0) {
      float sum = 0.0f;
      for (uint32_t k = 0; k < top_k; k++)
        sum += shmem->topk_weights_flat[warp_idx * MAX_TOPK + k];
      float inv = (sum > 0.0f) ? (1.0f / sum) : 1.0f;
      for (uint32_t k = 0; k < top_k; k++)
        shmem->topk_weights_flat[warp_idx * MAX_TOPK + k] *= inv;
    }
  } else if (warp_idx < 8) {
    // Padding — same as top1_BS8
    if (tid == 0) shmem->topk_ids[warp_idx] = (uint8_t)0xFF;
  }
}

/**
 * @brief Top-K expert selection for BS <= 64, writing results into shmem.
 *
 * Computes all K selections for every token and stores them in
 * shmem->topk_ids_flat / shmem->topk_weights_flat. The single-pass BS64
 * pipeline then uses prepare_moe_topk_BSx_Ey to sort the 512 virtual rows
 * and build token_indexes_topk / token_weights.
 *
 * Must be called by calc warps only.
 */
template <typename Dims>
__device__ void topK_BS64(uint32_t top_k, ScoringFunc scoring_func,
                          bool renormalize,
                          const __nv_bfloat16* __restrict__ router_logits,
                          uint32_t num_tokens, MoE_SHM<Dims>* shmem) {
  static_assert(Dims::BS <= 64, "Dispatch to incorrect implementation");

  constexpr uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;
  uint32_t thread_idx = threadIdx.x;

  for (uint32_t tokidx = thread_idx; tokidx < num_tokens; tokidx += 256) {
    float scores[Dims::NUM_EXPERTS];
    for (uint32_t e = 0; e < Dims::NUM_EXPERTS; e++)
      scores[e] = (float)router_logits[tokidx * Dims::NUM_EXPERTS + e];

    // Scoring
    if (scoring_func == ScoringFunc::SOFTMAX) {
      float mx = -FLT_MAX;
      for (uint32_t e = 0; e < Dims::NUM_EXPERTS; e++)
        mx = fmaxf(mx, scores[e]);
      float s = 0.0f;
      for (uint32_t e = 0; e < Dims::NUM_EXPERTS; e++) {
        scores[e] = __expf(scores[e] - mx);
        s += scores[e];
      }
      float inv = 1.0f / s;
      for (uint32_t e = 0; e < Dims::NUM_EXPERTS; e++) scores[e] *= inv;
    } else {
      for (uint32_t e = 0; e < Dims::NUM_EXPERTS; e++)
        scores[e] = 1.0f / (1.0f + __expf(-scores[e]));
    }

    // Iterative top-K — all K selections in one shot
    for (uint32_t k = 0; k < top_k; k++) {
      float best = -FLT_MAX;
      uint32_t best_e = 0;
      for (uint32_t e = 0; e < Dims::NUM_EXPERTS; e++) {
        if (scores[e] > best) {
          best = scores[e];
          best_e = e;
        }
      }
      shmem->topk_ids_flat[tokidx * MAX_TOPK + k] = (uint8_t)best_e;
      shmem->topk_weights_flat[tokidx * MAX_TOPK + k] = best;
      scores[best_e] = -FLT_MAX;
    }

    // Renormalize
    if (renormalize) {
      float s = 0.0f;
      for (uint32_t k = 0; k < top_k; k++)
        s += shmem->topk_weights_flat[tokidx * MAX_TOPK + k];
      float inv = (s > 0.0f) ? (1.0f / s) : 1.0f;
      for (uint32_t k = 0; k < top_k; k++)
        shmem->topk_weights_flat[tokidx * MAX_TOPK + k] *= inv;
    }
  }
}

/**
 * @brief Prepare the BS8 tiny path for top-K single-pass.
 *
 * Builds expert_ids / expert_count from the full flat topk_ids_flat array
 * (BS * top_k entries).  expert_mask is NOT set — the single-pass tiny
 * path uses topk_ids_flat directly for per-token K-slot filtering.
 */
template <typename Dims>
__device__ void prepare_moe_topk_BS8(uint32_t batch_size, uint32_t top_k,
                                     MoE_SHM<Dims>* __restrict__ shm) {
  static_assert(Dims::BS <= 8, "Dispatch to incorrect implementation");

  if (threadIdx.x % 32 != 0) return;

  constexpr uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;

  // Build bitset of all unique expert IDs across all tokens and K slots
  __uint128_t expert_bitset = 0;
  for (uint32_t t = 0; t < batch_size; t++) {
    for (uint32_t k = 0; k < top_k; k++) {
      uint32_t eid = shm->topk_ids_flat[t * MAX_TOPK + k];
      if (eid != 0xFF) expert_bitset |= __uint128_t(1) << eid;
    }
  }

  uint64_t b0 = (uint64_t)(expert_bitset & 0xFFFFFFFFFFFFFFFFULL);
  uint64_t b1 = (uint64_t)(expert_bitset >> 64);
  uint32_t expert_count = __popcll(b0) + __popcll(b1);

  // Extract up to 8 unique expert IDs (same logic as prepare_moe_BS8_E128)
  // Guard each extraction: if no bits remain, use 0xFF as sentinel.
  uint32_t addend = 0;
  if (b0 == 0) {
    b0 = b1;
    b1 = 0;
    addend = 64;
  }
  uint32_t e0 = b0 ? (__ffsll(b0) - 1 + addend) : 0xFF;
  if (b0) {
    b0 &= b0 - 1;
  }
  if (b0 == 0) {
    b0 = b1;
    b1 = 0;
    addend = 64;
  }
  uint32_t e1 = b0 ? (__ffsll(b0) - 1 + addend) : 0xFF;
  if (b0) {
    b0 &= b0 - 1;
  }
  if (b0 == 0) {
    b0 = b1;
    b1 = 0;
    addend = 64;
  }
  uint32_t e2 = b0 ? (__ffsll(b0) - 1 + addend) : 0xFF;
  if (b0) {
    b0 &= b0 - 1;
  }
  if (b0 == 0) {
    b0 = b1;
    b1 = 0;
    addend = 64;
  }
  uint32_t e3 = b0 ? (__ffsll(b0) - 1 + addend) : 0xFF;
  if (b0) {
    b0 &= b0 - 1;
  }
  if (b0 == 0) {
    b0 = b1;
    b1 = 0;
    addend = 64;
  }
  uint32_t e4 = b0 ? (__ffsll(b0) - 1 + addend) : 0xFF;
  if (b0) {
    b0 &= b0 - 1;
  }
  if (b0 == 0) {
    b0 = b1;
    b1 = 0;
    addend = 64;
  }
  uint32_t e5 = b0 ? (__ffsll(b0) - 1 + addend) : 0xFF;
  if (b0) {
    b0 &= b0 - 1;
  }
  if (b0 == 0) {
    b0 = b1;
    b1 = 0;
    addend = 64;
  }
  uint32_t e6 = b0 ? (__ffsll(b0) - 1 + addend) : 0xFF;
  if (b0) {
    b0 &= b0 - 1;
  }
  if (b0 == 0) {
    b0 = b1;
    b1 = 0;
    addend = 64;
  }
  uint32_t e7 = b0 ? (__ffsll(b0) - 1 + addend) : 0xFF;

  e1 <<= 8;
  e2 <<= 16;
  e3 <<= 24;
  e5 <<= 8;
  e6 <<= 16;
  e7 <<= 24;
  uint64_t packed = ((uint64_t)((e4 | e5) | (e6 | e7)) << 32) |
                    (uint64_t)((e0 | e1) | (e2 | e3));

  if (threadIdx.x == 0) {
    shm->expert_mask = 0;  // not used in single-pass topK tiny path
    shm->expert_ids = packed;
    shm->expert_count = expert_count;
  }
}

}  // namespace moe_monokernel

#endif
