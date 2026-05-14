
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
 *
 * Optimization: only the row max (over all 256 experts) is needed to do
 * top-k selection — softmax / sigmoid are both monotonic, so the top-k
 * IDs picked by raw `(x - max)` are identical to those picked by
 * softmax(x) or sigmoid(x).  We therefore defer all `__expf` calls to
 * the post-selection step, where only `top_k` (= 8) values are
 * exponentiated instead of all `NUM_EXPERTS` (= 256).  This cuts
 * ~248 expf per warp × 8 warps = ~2000 expf calls per kernel
 * invocation, which dominates the routing phase wall clock.
 *
 * For `softmax + renormalize=True` (the Qwen3.5 case) the math
 * simplifies further: the softmax denominator `sum_all_exp` cancels
 * with the renormalize denominator, so we don't need it at all.
 *   weight_k = (exp(x_k - max) / sum_all) / (sum_topk / sum_all)
 *            = exp(x_k - max) / sum_topk_exp
 *
 * For `softmax + renormalize=False` we still need `sum_all_exp` to
 * normalize, which requires all 256 expf calls — that case falls
 * back to the original path.
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

  if (warp_idx >= num_tokens) {
    if (warp_idx < 8) {
      // padding slots — nothing to do for topk path
    }
    return;
  }

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

  // ── Slow path: softmax + renormalize=False ─────────────────────────────
  // Needs the full softmax denominator (sum over all 256 exp values), so
  // we cannot skip the bulk expf calls.  Falls back to the original
  // warp-cooperative softmax over all experts.
  if (scoring_func == ScoringFunc::SOFTMAX && !renormalize) {
    warp_softmax_inplace(scores, num_local);
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
    return;
  }

  // ── Fast path: softmax+renormalize=True OR sigmoid (any renorm) ────────
  //
  // Selection is done on raw `(x - row_max)` (softmax) or `x` (sigmoid).
  // Both functions are monotonically increasing, so the top-k IDs are
  // identical under either the raw-logit or post-activation ordering.
  //
  // For numerical safety on softmax we subtract `row_max` so the
  // post-selection `expf` always sees non-positive arguments.  For
  // sigmoid the raw `x` is fine (sigmoid is anchored at 0.5 and saturates
  // smoothly in both directions).
  if (scoring_func == ScoringFunc::SOFTMAX) {
    float local_max = -FLT_MAX;
    for (uint32_t i = 0; i < num_local; i++)
      local_max = fmaxf(local_max, scores[i]);
    float row_max = warp_reduce_max_float(local_max);
    for (uint32_t i = 0; i < num_local; i++) scores[i] -= row_max;
  }
  // (For sigmoid we leave scores at raw x; ordering is preserved.)

  // ── Top-k selection over the (now-shifted-or-raw) logits ──────────────
  // Track each selected slot's `score_for_choice` (used for ordering) and
  // the expert id.  For both paths, `score_for_choice` is the value used
  // to pick winners; we'll convert to the actual softmax / sigmoid
  // weight in the next step.
  float topk_scores[MoE_SHM<Dims>::MAX_TOPK];
  uint32_t topk_experts[MoE_SHM<Dims>::MAX_TOPK];
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
    float winning_score = __shfl_sync(0xFFFFFFFFU, max_val, winner);
    topk_scores[k] = winning_score;
    topk_experts[k] = winning_expert;
    for (uint32_t i = 0; i < num_local; i++) {
      if (expert_id[i] == winning_expert) scores[i] = -FLT_MAX;
    }
  }

  // ── Convert raw selected logits → activation values + (re)normalize ───
  // Only thread 0 does the final write to SHM.  All threads on the warp
  // hold identical `topk_scores` / `topk_experts` arrays (filled via
  // `__shfl_sync` above), so picking thread 0 is arbitrary.
  if (tid == 0) {
    if (scoring_func == ScoringFunc::SOFTMAX) {
      // Softmax + renormalize=True (the renormalize=False case returned
      // earlier).  weight_k = exp(x_k - max) / sum_topk_exp.
      float exp_vals[MoE_SHM<Dims>::MAX_TOPK];
      float sum_exp = 0.0f;
      for (uint32_t k = 0; k < top_k; k++) {
        exp_vals[k] = __expf(topk_scores[k]);
        sum_exp += exp_vals[k];
      }
      float inv = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 1.0f;
      for (uint32_t k = 0; k < top_k; k++) {
        shmem->topk_ids_flat[warp_idx * MAX_TOPK + k] =
            (uint16_t)topk_experts[k];
        shmem->topk_weights_flat[warp_idx * MAX_TOPK + k] = exp_vals[k] * inv;
      }
    } else {
      // Sigmoid path: weight_k = 1 / (1 + exp(-x_k)).
      // If renormalize, divide by sum of selected sigmoid values (matches
      // the original two-step "compute sigmoid → renormalize topk" math).
      float sig_vals[MoE_SHM<Dims>::MAX_TOPK];
      float sum_sig = 0.0f;
      for (uint32_t k = 0; k < top_k; k++) {
        sig_vals[k] = 1.0f / (1.0f + __expf(-topk_scores[k]));
        sum_sig += sig_vals[k];
      }
      float inv =
          renormalize ? ((sum_sig > 0.0f) ? (1.0f / sum_sig) : 1.0f) : 1.0f;
      for (uint32_t k = 0; k < top_k; k++) {
        shmem->topk_ids_flat[warp_idx * MAX_TOPK + k] =
            (uint16_t)topk_experts[k];
        shmem->topk_weights_flat[warp_idx * MAX_TOPK + k] = sig_vals[k] * inv;
      }
    }
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
                                     MoE_SHM<Dims>* __restrict__ shm,
                                     MoEGemmSpec<Dims>* __restrict__ spec) {
  static_assert(Dims::BS <= 8, "Dispatch to incorrect implementation");
  // `spec` is only used for MONO_PROFILE_PHASE_TIMING; suppress unused-
  // parameter warnings under non-instrumented builds.
  (void)spec;

  // Only warp 0 (threads 0–31) participates; the other calc threads exit
  // early and block on the caller's `__syncthreads()`.  This is a
  // refinement of the previous "thread 0 does everything" structure that
  // keeps the serial bitset / prefix-sum passes on thread 0 but lets the
  // final slot-assignment pass run warp-parallel — see the comment on
  // Pass 3 below for the motivation (NCU flagged the serial LDL/STL
  // `write_head[]` loop as the kernel's hottest stall source).
  if (threadIdx.x >= 32) return;

  constexpr uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;

  // ── Pass 1: warp-parallel bitset build + ordered enumeration ────────
  // Replaces the previous thread-0-only bitset+enumeration loop, which
  // dominated Pass-1 wall clock at BS=8 (≤ 64 sequential SHM stores
  // for `shm->experts[].id` writes plus 64 SHM reads of `topk_ids_flat`
  // for bitset construction).
  //
  // Strategy:
  //   (a) Build the 256-bit bitset as 8 × uint32 in registers, each lane
  //       holding its own partial bitset.  32 lanes × 2 pairs each
  //       (n_pairs ≤ 64) — lanes set bits independently, then a
  //       butterfly OR-reduce gives every lane the full bitset.
  //   (b) Enumerate set bits in ascending order: lanes 0..7 each own
  //       one bitset word.  popcount → warp exclusive prefix sum gives
  //       each owning lane its output offset; each owning lane scans
  //       its word's bits with __ffs (yields ascending order) and
  //       writes `experts[out].id` directly.
  //   (c) `total = lane7.offset + lane7.count` (broadcast via shfl).
  //   (d) Lane 0 reads back the first ≤ 8 expert ids (after a
  //       __syncwarp() to publish the writes from lanes 0..7) to build
  //       the packed `expert_ids` uint64.
  //
  // Ordering invariant preserved: ascending lane index → ascending word
  // index → ascending base eid; within a lane __ffs walks bits low →
  // high, so the global enumeration is monotonically increasing in eid.
  constexpr uint32_t MAX_PAIRS = Dims::BS * MAX_TOPK;  // ≤ 64 for BS=8
  const uint32_t tid = threadIdx.x;
  const uint32_t n_pairs = batch_size * top_k;

  uint32_t my_bs[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
  // 32 lanes × 2 pairs cover up to 64 pairs.  Loop covers any n_pairs
  // up to 32 × ⌈MAX_PAIRS/32⌉ — for MAX_PAIRS = 64 the loop runs ≤ 2
  // iterations per lane.
  for (uint32_t p = tid; p < MAX_PAIRS; p += 32) {
    if (p >= n_pairs) break;
    const uint32_t tok = p / top_k;
    const uint32_t kk = p % top_k;
    const uint32_t eid = shm->topk_ids_flat[tok * MAX_TOPK + kk];
    if (eid < Dims::NUM_EXPERTS) {  // skips the 0xFFFF sentinel
      my_bs[eid >> 5] |= (1u << (eid & 31));
    }
  }

  MONO_PHASE_TIMESTAMP(t_after_prepare_pass1a);

  // Warp-OR reduce: every lane ends up with the full 256-bit bitset.
  // 5 butterfly steps × 8 words = 40 SHFL.B32 + 40 LOP3.OR pairs.
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
  #pragma unroll
    for (int off = 16; off >= 1; off /= 2) {
      my_bs[i] |= __shfl_xor_sync(FULL_MASK, my_bs[i], off, 32);
    }
  }

  MONO_PHASE_TIMESTAMP(t_after_prepare_pass1b);

  // Enumerate set bits in ascending order.  Lanes 0..7 each own one
  // bitset word; lanes 8..31 contribute popcount = 0 and write nothing.
  uint32_t my_word = (tid < 8u) ? my_bs[tid] : 0u;
  const uint32_t my_count = __popc(my_word);

  // Warp-wide exclusive prefix sum of `my_count` → per-lane output
  // offset into `experts[]`.
  uint32_t v = my_count;
  #pragma unroll
  for (int off = 1; off <= 16; off *= 2) {
    const uint32_t t = __shfl_up_sync(FULL_MASK, v, off, 32);
    if (static_cast<int>(tid) >= off) v += t;
  }
  const uint32_t my_offset = v - my_count;
  const uint32_t total = __shfl_sync(FULL_MASK, my_offset + my_count, 7);

  // Lanes 0..7 enumerate their word's set bits and write to experts[].
  // Higher lanes have my_count = 0 and skip the loop entirely.
  //
  // Only `experts[].id` is written.  `first_token` / `last_token` are
  // unused on the BS8 path (see `prepare_moe_topk_BS8` docstring) — the
  // BS64 path is the only consumer.  Skipping those two stores cuts
  // ~50% off the per-iteration cost of this loop (3 stores → 1 store
  // per set bit) and is what previously dominated Pass 1.
  //
  // Lane 0 also accumulates the packed `expert_ids` uint64 in a
  // register during enumeration — this avoids a follow-up
  // `__syncwarp()` + 8 SHM reads to rebuild it, which previously
  // dominated Pass 1c wall clock.
  uint32_t out = my_offset;
  uint64_t packed = 0;
  while (my_word) {
    const uint32_t bit = __ffs(my_word) - 1;  // 0..31
    const uint32_t eid = (tid << 5) + bit;    // tid*32 + bit
    shm->experts[out].id = eid;
    if (tid == 0u && out < 8u) {
      packed |= static_cast<uint64_t>(eid) << (out * 8);
    }
    ++out;
    my_word &= my_word - 1;
  }

  if (tid == 0) {
    shm->path.bs8.expert_ids = packed;
    shm->expert_count = total;
  }

  MONO_PHASE_TIMESTAMP(t_after_prepare_pass1);

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
  if constexpr (use_tma<Dims>::value) {
    auto* tma_shm = &shm->u.tiny_wgmma_tma;

    // ── Pass 2: warp-parallel zero + tally + prefix sum ──────────────
    //
    // Old design: thread 0 zeroed 256 entries, tallied 64 pairs, then
    // ran a 256-step exclusive prefix sum.  This was ~7.5 µs of the
    // ~19 µs routing budget at BS=8 — entirely sequential SHM I/O on
    // one thread while the other 383 threads idled at the trailing
    // __syncthreads().
    //
    // New design: 32-lane warp cooperates on all three steps.
    //   (2a) Warp-parallel zero of `expert_routed_count[256]`:
    //        each lane writes 8 entries (256 / 32).  Single STG.E.U8
    //        per write, fully coalesced.
    //   (2b) Warp-parallel tally via __match_any_sync.  Each lane
    //        handles up to 2 pairs; peers sharing the same eid form a
    //        warp peer group, and the lowest-id lane in each group
    //        writes the popcount-sized increment.  Eliminates the
    //        previous serial 64-iter loop on thread 0.
    //   (2c) Warp-parallel exclusive prefix sum over 256 entries:
    //        each lane reads 8 entries → local exclusive prefix sum
    //        (7 adds) → warp shuffle scan over the lane sums (5
    //        butterfly steps) → write back 8 entries with the lane
    //        offset added.  Total: ~50 cycles vs. the old ~256 cycle
    //        serial scan.

    // (2a) Warp-parallel zero.  Each lane owns 8 contiguous entries.
    constexpr uint32_t ZERO_PER_LANE = Dims::NUM_EXPERTS / 32;
    static_assert(Dims::NUM_EXPERTS % 32 == 0,
                  "NUM_EXPERTS must be a multiple of 32 for warp-zero of "
                  "expert_routed_count[]");
  #pragma unroll
    for (uint32_t i = 0; i < ZERO_PER_LANE; ++i) {
      tma_shm->expert_routed_count[tid * ZERO_PER_LANE + i] = 0;
    }
    __syncwarp();

    // (2b) Warp-parallel tally via __match_any_sync.
    //
    // Each lane handles up to 2 pairs (pair_slot ∈ {0, 1}; lane `tid`
    // handles pair `tid + slot*32`).  For each pair-slot:
    //   * Load the routed eid (or 0xFFFF sentinel for out-of-range
    //     / unrouted lanes).
    //   * `__match_any_sync(FULL_MASK, eid)` returns a bitmask whose
    //     bit `j` is set iff lane `j` in this warp holds the same
    //     eid.  Lanes that share an eid form a peer group.
    //   * Within each peer group, the lowest-bit-set lane is the
    //     "leader" and writes `expert_routed_count[eid] += popcount`
    //     for the whole group — a single uint8 RMW on SHM, no atomic
    //     needed because the leader is unique within the warp.
    //   * `__syncwarp()` between pair_slots serialises the writes so
    //     pair_slot 1's RMW sees pair_slot 0's contribution.
    //
    // Sentinel handling: `eid == 0xFFFF` lanes form their own peer
    // group, but the `eid < Dims::NUM_EXPERTS` guard skips the write,
    // so the sentinel write never lands.
    //
    // Costs vs. the previous serial tally (BS=8, top_k=8 → 64 pairs):
    //   * Old: 64 SHM reads + 64 dependent uint8 RMWs on thread 0
    //          ≈ 1900 cycles (~1 µs).
    //   * New: 2 × (1 __match_any_sync + 1 popc + 1 ffs + 1 SHM RMW
    //          on the leader lane) + 1 __syncwarp.  ≈ 50 cycles
    //          (~0.025 µs).
    constexpr uint32_t MAX_TALLY_SLOTS = (MAX_PAIRS + 31u) / 32u;  // 2 for BS=8
  #pragma unroll
    for (uint32_t slot = 0; slot < MAX_TALLY_SLOTS; ++slot) {
      const uint32_t p = tid + slot * 32u;
      const uint16_t eid =
          (p < n_pairs) ? shm->topk_ids_flat[p] : (uint16_t)0xFFFF;
      const uint32_t key = static_cast<uint32_t>(eid);

      const uint32_t match = __match_any_sync(FULL_MASK, key);
      const uint32_t count = __popc(match);
      const uint32_t lowest = __ffs(match) - 1u;  // 0..31

      if (eid < Dims::NUM_EXPERTS && tid == lowest) {
        // Read-modify-write on uint8 SHM.  Safe within a single warp:
        // exactly one lane (the leader) per unique eid runs this path,
        // and the __syncwarp() between slots serialises with the
        // previous slot's writes.
        tma_shm->expert_routed_count[eid] += static_cast<uint8_t>(count);
      }
      __syncwarp();
    }

    // (2c) Warp-parallel exclusive prefix sum over `expert_routed_count`.
    // Each lane handles a contiguous block of `BLK = 256/32 = 8`
    // entries.
    //
    // Step 1: per-lane local exclusive prefix sum into registers.
    constexpr uint32_t BLK = Dims::NUM_EXPERTS / 32;  // 8 for E=256
    uint32_t lane_vals[BLK];
    uint32_t lane_sum = 0;
  #pragma unroll
    for (uint32_t i = 0; i < BLK; ++i) {
      const uint32_t v = tma_shm->expert_routed_count[tid * BLK + i];
      lane_vals[i] = lane_sum;  // exclusive: pre-add value
      lane_sum += v;
    }
    // `lane_sum` is now the total count for this lane's 8-entry block.

    // Step 2: warp-wide exclusive prefix sum of `lane_sum` →
    //         `lane_offset` = sum of all earlier lanes' totals.
    uint32_t scan = lane_sum;
  #pragma unroll
    for (int off = 1; off <= 16; off *= 2) {
      const uint32_t t = __shfl_up_sync(FULL_MASK, scan, off, 32);
      if (static_cast<int>(tid) >= off) scan += t;
    }
    const uint32_t lane_offset = scan - lane_sum;

    // Step 3: write back lane_offset + lane_vals[i] to expert_slot_start.
  #pragma unroll
    for (uint32_t i = 0; i < BLK; ++i) {
      tma_shm->expert_slot_start[tid * BLK + i] =
          static_cast<uint16_t>(lane_offset + lane_vals[i]);
    }
    __syncwarp();

    MONO_PHASE_TIMESTAMP(t_after_prepare_pass2);

    // ── Pass 3: warp-cooperative slot assignment ─────────────────────────
    //
    // Evolution of this pass:
    //
    //   v1 (single-threaded).  Stateful `write_head[eid]++` local-memory
    //   counter, 64 sequential LDL/STL round-trips on thread 0.  NCU
    //   flagged the trailing `__syncthreads()` as the kernel's hottest
    //   stall source.
    //
    //   v2 (warp-parallel, inner serial scan).  Warp 0 with each lane
    //   computing its rank as "count of earlier pairs with the same
    //   eid" via a serial SHM scan.  Moved the hot spot down by ~10 us
    //   but thread 31 on pair 63 still did 63 sequential SHM reads
    //   (~1200-cycle critical path), and NCU again flagged the next
    //   `__syncthreads()`.
    //
    //   v3 (this code — warp-cooperative via __match_any_sync).  Within
    //   a warp chunk of 32 lanes, `__match_any_sync(FULL_MASK, eid)`
    //   returns a per-lane bitmask whose bit `j` is set iff lane `j`
    //   holds the same eid as the caller.  The intra-chunk rank is
    //   then `popc(match & lane_mask_below_self)` — a single warp
    //   instruction in place of the inner 32-iteration scan.  For
    //   `n_pairs > 32` we add a cross-chunk carry that rotates each
    //   thread's `eid1` through the warp and accumulates matches
    //   against `eid0`; 32 shfl+ballot rounds replace 32 SHM loads
    //   per thread on the longest path.
    //
    // Ordering invariant preserved (R11.2, Q-prep-3):
    //   * `popc(match & lane_mask)` counts strictly EARLIER lanes, so
    //     within a chunk the sorted_slot values are monotonic in the
    //     lane index (== pair index).
    //   * The cross-chunk carry adds `|{chunk-0 pairs with same eid}|`
    //     to chunk-1 intra-ranks, so chunk-1 slots pick up exactly
    //     where chunk 0 left off.
    //   Result: for any expert, `sorted_slot` is strictly ascending
    //   when pairs are enumerated in lex order — byte-identical to the
    //   v1 write-head semantics.
    //
    // Sentinel handling: lanes with eid == 0xFFFF participate in the
    // warp intrinsics (all 32 lanes are converged and must call the
    // _sync primitives), but their computed rank is discarded
    // because the `sorted_slot` write is gated on `eid != 0xFFFF`.
    // Worst-case sentinel clustering (many pairs all unrouted) gives
    // meaningless rank values for exactly those pairs — same as v1/v2.
    //
    // `FULL_MASK` (= 0xFFFFFFFFu) is defined as a preprocessor macro at
    // the top of `moe_prepare.cu`, which `moe.cu` includes before
    // `moe_routing.cu`, so it's visible here.  Reusing that macro also
    // avoids naming a local `constexpr FULL_MASK` that would collide
    // with the macro at substitution time.
    const uint32_t tid = threadIdx.x;
    const uint32_t n_pairs = batch_size * top_k;  // ≤ MAX_PAIRS = 64

    auto load_eid = [&](uint32_t pair) -> uint16_t {
      if (pair >= n_pairs) return 0xFFFF;
      const uint32_t tok = pair / top_k;
      const uint32_t k = pair % top_k;
      return shm->topk_ids_flat[tok * MAX_TOPK + k];
    };

    const uint32_t p0 = tid;       // chunk 0 pair index
    const uint32_t p1 = tid + 32;  // chunk 1 pair index
    const uint16_t eid0 = load_eid(p0);
    const uint16_t eid1 = load_eid(p1);

    // `lane_mask = (1 << tid) - 1` selects bits for lanes strictly
    // below the caller.  `(1u << 0) - 1 == 0`, so lane 0 correctly
    // gets rank 0.
    const uint32_t lane_mask = (1u << tid) - 1u;

    // Chunk-0 intra-chunk rank.
    const uint32_t match0 =
        __match_any_sync(FULL_MASK, static_cast<uint32_t>(eid0));
    const uint32_t rank0 = __popc(match0 & lane_mask);

    // Chunk-1 intra-chunk rank.
    const uint32_t match1 =
        __match_any_sync(FULL_MASK, static_cast<uint32_t>(eid1));
    const uint32_t rank1_intra = __popc(match1 & lane_mask);

    // Chunk-1 cross-chunk carry: for each lane's `eid1`, count how many
    // chunk-0 pairs had the same expert id.  Rotate `eid1` through the
    // warp; on iteration `src` every lane checks its `eid0` against
    // lane `src`'s `eid1` via a single `__ballot_sync`, and lane `src`
    // captures the popcount.  32 shfl+ballot+popc rounds — warp-wide
    // pipelined, ~150 cycles vs. the v2 critical path of ~1260 cycles.
    //
    // Skipped entirely when n_pairs ≤ 32 (BS≤4 or top_k≤4): there are
    // no chunk-1 pairs to rank.
    uint32_t rank1_carry = 0;
    if (n_pairs > 32) {
  #pragma unroll
      for (int src = 0; src < 32; ++src) {
        const uint32_t q =
            __shfl_sync(FULL_MASK, static_cast<uint32_t>(eid1), src);
        const uint32_t b =
            __ballot_sync(FULL_MASK, static_cast<uint32_t>(eid0) == q);
        if (static_cast<int>(tid) == src) rank1_carry = __popc(b);
      }
    }

    if (p0 < n_pairs && eid0 != 0xFFFF) {
      tma_shm->sorted_slot[p0] =
          static_cast<uint8_t>(tma_shm->expert_slot_start[eid0] + rank0);
    }
    if (p1 < n_pairs && eid1 != 0xFFFF) {
      tma_shm->sorted_slot[p1] = static_cast<uint8_t>(
          tma_shm->expert_slot_start[eid1] + rank1_intra + rank1_carry);
    }
  }

  MONO_PHASE_TIMESTAMP(t_after_prepare_pass3);
}

}  // namespace moe_monokernel

#endif
