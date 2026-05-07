
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
 */
__device__ static inline uint32_t warp_who_has(float haystack, float needle) {
  uint32_t mask = __ballot_sync(0xFFFFFFFFU, haystack == needle);
  assert(mask > 0);
  return __ffs(mask) - 1;
}

/**
 * @brief Warp-cooperative softmax over logits distributed across threads.
 *
 * Each thread holds @p count values. On return every element of @p logits
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
 * Computes all K selections via warp reduction and stores them in
 * shmem->topk_ids_flat / shmem->topk_weights_flat.
 *
 * Must be called by calc warps only.
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

    if (scoring_func == ScoringFunc::SOFTMAX) {
      warp_softmax_inplace(scores, num_local);
    } else {
      for (uint32_t i = 0; i < num_local; i++)
        scores[i] = 1.0f / (1.0f + __expf(-scores[i]));
    }

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

      if (tid == winner) {
        shmem->topk_ids_flat[warp_idx * MAX_TOPK + k] = (uint16_t)max_expert;
        shmem->topk_weights_flat[warp_idx * MAX_TOPK + k] = winning_weight;
      }

      for (uint32_t i = 0; i < num_local; i++) {
        if (expert_id[i] == winning_expert) scores[i] = -FLT_MAX;
      }
    }

    if (renormalize && tid == 0) {
      float sum = 0.0f;
      for (uint32_t k = 0; k < top_k; k++)
        sum += shmem->topk_weights_flat[warp_idx * MAX_TOPK + k];
      float inv = (sum > 0.0f) ? (1.0f / sum) : 1.0f;
      for (uint32_t k = 0; k < top_k; k++)
        shmem->topk_weights_flat[warp_idx * MAX_TOPK + k] *= inv;
    }
  } else if (warp_idx < 8) {
    // padding slots — nothing to do for topk path
  }
}

/**
 * @brief Top-K expert selection for BS <= 64, writing results into shmem.
 *
 * Computes all K selections for every token and stores them in
 * shmem->topk_ids_flat / shmem->topk_weights_flat.
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
  // Process experts in fixed-size chunks to avoid a per-thread
  // float[NUM_EXPERTS] array that kills register pressure / occupancy
  // when NUM_EXPERTS is large (e.g. 256).
  constexpr uint32_t CHUNK = 32;
  uint32_t thread_idx = threadIdx.x;

  for (uint32_t tokidx = thread_idx; tokidx < num_tokens; tokidx += 256) {
    const __nv_bfloat16* logits_row =
        router_logits + tokidx * Dims::NUM_EXPERTS;

    if (scoring_func == ScoringFunc::SOFTMAX) {
      // ── Pass 1: streaming max over all experts ──
      float mx = -FLT_MAX;
      for (uint32_t base = 0; base < Dims::NUM_EXPERTS; base += CHUNK) {
        uint32_t end = (base + CHUNK < Dims::NUM_EXPERTS) ? base + CHUNK
                                                          : Dims::NUM_EXPERTS;
        for (uint32_t e = base; e < end; e++)
          mx = fmaxf(mx, (float)logits_row[e]);
      }

      // ── Pass 2: streaming sum of exp(x - mx) ──
      float sum_exp = 0.0f;
      for (uint32_t base = 0; base < Dims::NUM_EXPERTS; base += CHUNK) {
        uint32_t end = (base + CHUNK < Dims::NUM_EXPERTS) ? base + CHUNK
                                                          : Dims::NUM_EXPERTS;
        for (uint32_t e = base; e < end; e++)
          sum_exp += __expf((float)logits_row[e] - mx);
      }
      float inv_sum = 1.0f / sum_exp;

      // ── Pass 3: chunked top-k selection on softmax scores ──
      float topk_vals[MAX_TOPK];
      uint32_t topk_ids[MAX_TOPK];
      for (uint32_t k = 0; k < top_k; k++) {
        topk_vals[k] = -FLT_MAX;
        topk_ids[k] = 0;
      }

      for (uint32_t base = 0; base < Dims::NUM_EXPERTS; base += CHUNK) {
        uint32_t end = (base + CHUNK < Dims::NUM_EXPERTS) ? base + CHUNK
                                                          : Dims::NUM_EXPERTS;
        float chunk_scores[CHUNK];
        for (uint32_t i = 0; i < end - base; i++)
          chunk_scores[i] = __expf((float)logits_row[base + i] - mx) * inv_sum;

        for (uint32_t i = 0; i < end - base; i++) {
          // Find the current minimum in the top-k heap.
          float min_val = topk_vals[0];
          uint32_t min_idx = 0;
          for (uint32_t k = 1; k < top_k; k++) {
            if (topk_vals[k] < min_val) {
              min_val = topk_vals[k];
              min_idx = k;
            }
          }
          if (chunk_scores[i] > min_val) {
            topk_vals[min_idx] = chunk_scores[i];
            topk_ids[min_idx] = base + i;
          }
        }
      }

      for (uint32_t k = 0; k < top_k; k++) {
        shmem->topk_ids_flat[tokidx * MAX_TOPK + k] = (uint16_t)topk_ids[k];
        shmem->topk_weights_flat[tokidx * MAX_TOPK + k] = topk_vals[k];
      }

    } else {
      // ── Sigmoid scoring: chunked top-k ──
      float topk_vals[MAX_TOPK];
      uint32_t topk_ids[MAX_TOPK];
      for (uint32_t k = 0; k < top_k; k++) {
        topk_vals[k] = -FLT_MAX;
        topk_ids[k] = 0;
      }

      for (uint32_t base = 0; base < Dims::NUM_EXPERTS; base += CHUNK) {
        uint32_t end = (base + CHUNK < Dims::NUM_EXPERTS) ? base + CHUNK
                                                          : Dims::NUM_EXPERTS;
        float chunk_scores[CHUNK];
        for (uint32_t i = 0; i < end - base; i++) {
          float v = (float)logits_row[base + i];
          chunk_scores[i] = 1.0f / (1.0f + __expf(-v));
        }

        for (uint32_t i = 0; i < end - base; i++) {
          float min_val = topk_vals[0];
          uint32_t min_idx = 0;
          for (uint32_t k = 1; k < top_k; k++) {
            if (topk_vals[k] < min_val) {
              min_val = topk_vals[k];
              min_idx = k;
            }
          }
          if (chunk_scores[i] > min_val) {
            topk_vals[min_idx] = chunk_scores[i];
            topk_ids[min_idx] = base + i;
          }
        }
      }

      for (uint32_t k = 0; k < top_k; k++) {
        shmem->topk_ids_flat[tokidx * MAX_TOPK + k] = (uint16_t)topk_ids[k];
        shmem->topk_weights_flat[tokidx * MAX_TOPK + k] = topk_vals[k];
      }
    }

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
 * @brief Prepares the BS8 tiny path for top-K single-pass.
 *
 * Reads topk_ids_flat (filled by topK_BS8) and builds:
 *  - experts[0..expert_count-1].id  — ordered list of unique expert ids
 *    active in this batch (first_token/last_token are unused in BS8)
 *  - expert_count                   — number of unique experts
 *  - path.bs8.expert_ids            — same ids packed one-per-byte into
 *    a uint64, used by the prefetch warp to look up the next expert id
 *
 * Token-to-expert assignment is NOT sorted here. The per-expert loop in
 * moe_kernel_topk_BS8 scans topk_ids_flat directly for each token.
 */
template <typename Dims>
__device__ void prepare_moe_topk_BS8(uint32_t batch_size, uint32_t top_k,
                                     MoE_SHM<Dims>* __restrict__ shm) {
  static_assert(Dims::BS <= 8, "Dispatch to incorrect implementation");

  // Single thread does all the work — trivially cheap (≤ BS*top_k ≤ 64 iters).
  // Results are visible to all threads after the __syncthreads() in the caller.
  if (threadIdx.x != 0) return;

  constexpr uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;

  // Build a bitset of all unique expert ids across all tokens and K slots.
  // Use two __uint128_t words to cover up to 256 experts.
  __uint128_t expert_bitset_lo = 0;  // experts 0–127
  __uint128_t expert_bitset_hi = 0;  // experts 128–255
  for (uint32_t t = 0; t < batch_size; t++)
    for (uint32_t k = 0; k < top_k; k++) {
      uint32_t eid = shm->topk_ids_flat[t * MAX_TOPK + k];
      if (eid != 0xFFFF) {
        if (eid < 128)
          expert_bitset_lo |= __uint128_t(1) << eid;
        else
          expert_bitset_hi |= __uint128_t(1) << (eid - 128);
      }
    }

  // Extract unique expert ids in ascending order from the low half first,
  // then the high half.
  uint32_t ec = 0;
  uint64_t packed = 0;

  // Process low 128 bits (experts 0–127)
  uint64_t b0 = (uint64_t)(expert_bitset_lo & 0xFFFFFFFFFFFFFFFFULL);
  uint64_t b1 = (uint64_t)(expert_bitset_lo >> 64);
  uint32_t add = 0;
  while (b0 || b1) {
    if (b0 == 0) {
      b0 = b1;
      b1 = 0;
      add = 64;
    }
    uint32_t eid = __ffsll(b0) - 1 + add;
    b0 &= b0 - 1;
    shm->experts[ec].id = eid;
    shm->experts[ec].first_token = 0;
    shm->experts[ec].last_token = 0;
    if (ec < 8) packed |= (uint64_t)eid << (ec * 8);
    ec++;
  }

  // Process high 128 bits (experts 128–255)
  b0 = (uint64_t)(expert_bitset_hi & 0xFFFFFFFFFFFFFFFFULL);
  b1 = (uint64_t)(expert_bitset_hi >> 64);
  add = 128;
  while (b0 || b1) {
    if (b0 == 0) {
      b0 = b1;
      b1 = 0;
      add = 128 + 64;
    }
    uint32_t eid = __ffsll(b0) - 1 + add;
    b0 &= b0 - 1;
    shm->experts[ec].id = eid;
    shm->experts[ec].first_token = 0;
    shm->experts[ec].last_token = 0;
    if (ec < 8) packed |= (uint64_t)eid << (ec * 8);
    ec++;
  }
  shm->path.bs8.expert_ids = packed;
  shm->expert_count = ec;

  // ── TMA-only: build the three per-expert reorganization tables ──────────
  // expert_routed_count[eid] = # of routed (tok, k_in_topk) pairs selecting eid
  // expert_slot_start[eid]   = exclusive prefix sum over expert_routed_count
  // sorted_slot[pair]        = destination row in spec->temp_fp8 for the
  //                            up-proj SiLU+fp8 writeback, where
  //                            pair = tok * top_k + k_in_topk.  Equal to
  //                            expert_slot_start[eid] + intra-expert rank.
  //
  // Iterating (t, k) in ascending order guarantees the intra-expert rank is
  // lexicographic in (tok, k_in_topk), so the layout is deterministic (R11.2).
  // Sentinel topk_ids_flat == 0xFFFF (unrouted slot) is skipped.
  //
  // Cost: ≤ NUM_EXPERTS (= 256) zero/prefix iterations + batch_size * top_k
  // (≤ 64) routed-pair iterations.  All on thread 0 — no inter-thread sync.
  if constexpr (use_tma<Dims>::value) {
    auto* tma_shm = &shm->u.tiny_wgmma_tma;

    // Pass 1: zero counts, then count routed pairs per expert.
    for (uint32_t eid = 0; eid < Dims::NUM_EXPERTS; ++eid) {
      tma_shm->expert_routed_count[eid] = 0;
    }
    for (uint32_t t = 0; t < batch_size; t++) {
      for (uint32_t k = 0; k < top_k; k++) {
        uint16_t eid = shm->topk_ids_flat[t * MAX_TOPK + k];
        if (eid != 0xFFFF) tma_shm->expert_routed_count[eid]++;
      }
    }

    // Pass 2: exclusive prefix sum → expert_slot_start.
    // Invariant at end: running == batch_size * top_k (at most 64 for BS=8).
    uint32_t running = 0;
    for (uint32_t eid = 0; eid < Dims::NUM_EXPERTS; ++eid) {
      tma_shm->expert_slot_start[eid] = (uint16_t)running;
      running += tma_shm->expert_routed_count[eid];
    }

    // Pass 3: assign each routed pair its destination slot.
    // `write_head` is a local counter so we don't disturb expert_routed_count
    // (which Phase 4 reads to size the bulk activation TMA).
    // For BS=8, top_k=8, NUM_EXPERTS=256 this is 256 B on thread 0's stack.
    uint8_t write_head[Dims::NUM_EXPERTS] = {0};
    for (uint32_t t = 0; t < batch_size; t++) {
      for (uint32_t k = 0; k < top_k; k++) {
        uint16_t eid = shm->topk_ids_flat[t * MAX_TOPK + k];
        if (eid == 0xFFFF) continue;
        uint32_t pair = t * top_k + k;
        uint32_t rank = write_head[eid]++;
        tma_shm->sorted_slot[pair] =
            (uint8_t)(tma_shm->expert_slot_start[eid] + rank);
      }
    }
  }
}

}  // namespace moe_monokernel

#endif
