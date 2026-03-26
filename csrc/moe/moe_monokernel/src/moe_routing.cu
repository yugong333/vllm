
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
  static_assert(Dims::NUM_EXPERTS <= 16, "Dispatch to incorrect imlementation");
  static_assert(Dims::BS <= 64, "Dispatch to incorrect imlementation");
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
  static_assert(Dims::BS <= 64, "Dispatch to incorrect imlementation");
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
  static_assert(Dims::BS <= 64, "Dispatch to incorrect imlementation");
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
  static_assert(Dims::BS <= 8, "Dispatch to incorrect imlementation");
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

}  // namespace moe_monokernel

#endif
