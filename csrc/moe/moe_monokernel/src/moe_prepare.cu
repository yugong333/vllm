
#pragma once
#ifndef MOE_PREPARE_CU
#define MOE_PREPARE_CU

#ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
#error Do not include this file directly.
#endif

#include <cstdint>

#include "moe_internal.h"

#ifndef __SIZEOF_INT128__
static_assert(false, "This module currently needs int128. You're host compiler does not support it.")
#endif

#define FULL_MASK 0xFFFFFFFFU

namespace moe_monokernel {

// We use an uint128 to store 16 uint8
typedef __uint128_t uint8x16_t;

/**
 * @brief 16-byte allreduce summation within a warp
 *
 * Sums a 16-byte integer across all threads and returns the sum.
 * This operation is collective and needs to be called by all threads within a warp.
 *
 */
__device__ static inline uint8x16_t allreduce_sum_across_warp(uint8x16_t val)
{
    uint64_t val_lo = val & 0xFFFFFFFFFFFFFFFFU;
    uint64_t val_hi = val >> 64;
    for (int offset = 16; offset > 0; offset /= 2) {
        val_lo += __shfl_xor_sync(FULL_MASK, val_lo, offset, 32);
        val_hi += __shfl_xor_sync(FULL_MASK, val_hi, offset, 32);
    }
    return val_lo | ((uint8x16_t) val_hi << 64);
}

/** 
 * @brief Prefix sum of all uint8s in an uint8x16_t
 *
 * Computes a prefix sum over 16 uint8. First element is the least significant byte.
 */
__device__ static inline uint8x16_t prefix_sum_over_bytes(uint8x16_t val)
{
    val += val << 8;
    val += val << 16;
    val += val << 32;
    val += val << 64;
    return val;
}

/**
 * @brief Prepares the MoE computation for batch size 8 and 16 experts for the matrix multiply functions.
 *
 * Reads @c shm->topk_ids and sets up the necessary data structures and state for @c moe_up_projection and @c moe_down_projection .
 * Specifically, it fills:
 * - @c shm->expert_mask
 * - @c shm->expert_ids
 * - @c shm->expert_count
 *
 * The computation is done redundantly on each CUDA block such that the result can be stored in shared memory.
 *
 * @param batch_size The batch size (active elements in the topk_ids array)
 * @param shm Pointer to shared memory struct to read inputs from and write outputs to.
 *
 */
template <typename Dims>
__device__ static void prepare_moe_BS8_E16(
    std::uint32_t  batch_size,
    MoE_SHM<Dims>* __restrict__ shm)
{
    static_assert(Dims::NUM_EXPERTS <= 16, "This function is only for up to 16 experts");
    static_assert(Dims::BS <= 8, "This function is only for up to batch size 8");
    assert(batch_size <= 8);

    std::uint64_t packed_unique_topkids = 0;
    
    // Initialize the SHM in every SM.
    // One thread in each warp does this so that we do not need a block-wide synchronization before accessing the first expert
    if (threadIdx.x % 32 == 0) {
        uint32_t expert_count = 0;
        
        // Packing is a no-op since topk_ids are stored as uint8 already
        uint64_t packed_topkids = *(uint64_t*) shm->topk_ids;

        uint32_t t0 = shm->topk_ids[0];
        uint32_t t1 = shm->topk_ids[1];
        uint32_t t2 = shm->topk_ids[2];
        uint32_t t3 = shm->topk_ids[3];
        uint32_t t4 = shm->topk_ids[4];
        uint32_t t5 = shm->topk_ids[5];
        uint32_t t6 = shm->topk_ids[6];
        uint32_t t7 = shm->topk_ids[7];

        // To determine the unique experts, we use a bitfield of expert ids.
        // Bit i is 1 <==> expert i appears in topk_ids
        uint32_t expert_bitset = 0;
        expert_bitset |= 1 << t0;
        expert_bitset |= 1 << t1;
        expert_bitset |= 1 << t2;
        expert_bitset |= 1 << t3;
        expert_bitset |= 1 << t4;
        expert_bitset |= 1 << t5;
        expert_bitset |= 1 << t6;
        expert_bitset |= 1 << t7;

        expert_count = __popc(expert_bitset);

        // Extract the up to 8 unique experts from the bitfield, by finding the up to 8 bits that are set to 1.
        //
        // Only the first "expert_count" of these are valid, the other eX values are 0xFFFFFFFF
        // __ffs(0) - 1 == 0xFFFFFFFF
        uint32_t e0 = __ffs(expert_bitset) - 1; /* ffs starts counting at 1 */
        expert_bitset &= expert_bitset - 1;
        uint32_t e1 = __ffs(expert_bitset) - 1; /* ffs starts counting at 1 */
        expert_bitset &= expert_bitset - 1;
        uint32_t e2 = __ffs(expert_bitset) - 1; /* ffs starts counting at 1 */
        expert_bitset &= expert_bitset - 1;
        uint32_t e3 = __ffs(expert_bitset) - 1; /* ffs starts counting at 1 */
        expert_bitset &= expert_bitset - 1;
        uint32_t e4 = __ffs(expert_bitset) - 1; /* ffs starts counting at 1 */
        expert_bitset &= expert_bitset - 1;
        uint32_t e5 = __ffs(expert_bitset) - 1; /* ffs starts counting at 1 */
        expert_bitset &= expert_bitset - 1;
        uint32_t e6 = __ffs(expert_bitset) - 1; /* ffs starts counting at 1 */
        expert_bitset &= expert_bitset - 1;
        uint32_t e7 = __ffs(expert_bitset) - 1; /* ffs starts counting at 1 */
        assert((expert_bitset & (expert_bitset - 1)) == 0);

        // Pack the expert ids into bytes
        e1 <<= 8*1;
        e2 <<= 8*2;
        e3 <<= 8*3;
        e5 <<= 8*1;
        e6 <<= 8*2;
        e7 <<= 8*3;
        uint32_t lo_packed_unique_topkids = (e0 | e1) | (e2 | e3);
        uint32_t hi_packed_unique_topkids = (e4 | e5) | (e6 | e7);
        packed_unique_topkids = lo_packed_unique_topkids | ((uint64_t) hi_packed_unique_topkids << 32);
        
        if (threadIdx.x == 0) {
            shm->expert_mask = packed_topkids;
            shm->expert_ids = packed_unique_topkids;
            shm->expert_count = expert_count;
        }
    }
}

/**
 * @brief Prepares the MoE computation for batch size 8 and 128 experts for the matrix multiply functions.
 *
 * Reads @c shm->topk_ids and sets up the necessary data structures and state for @c moe_up_projection and @c moe_down_projection .
 * Specifically, it fills:
 * - @c shm->expert_mask
 * - @c shm->expert_ids
 * - @c shm->expert_count
 *
 * The computation is done redundantly on each CUDA block such that the result can be stored in shared memory.
 *
 * This function *does not* support a higher number of experts than 128.
 *
 * @param batch_size The batch size (active elements in the topk_ids array)
 * @param shm Pointer to shared memory struct to read inputs from and write outputs to.
 *
 */
template <typename Dims>
__device__ static void prepare_moe_BS8_E128(
    std::uint32_t  batch_size,
    MoE_SHM<Dims>* __restrict__ shm)
{
    static_assert(Dims::NUM_EXPERTS <= 128, "This function is only for up to 128 experts");
    static_assert(Dims::BS <= 8, "This function is only for up to batch size 8");
    assert(batch_size <= 8);

    std::uint64_t packed_unique_topkids = 0;
    
    // Initialize the SHM in every SM.
    // One thread in each warp does this so that we do not need a block-wide synchronization before accessing the first expert
    // For explanation of the method, see prepare_moe_BS8_E16
    if (threadIdx.x % 32 == 0) {
        uint64_t packed_topkids = *(uint64_t*) shm->topk_ids;

        uint32_t t0 = shm->topk_ids[0];
        uint32_t t1 = shm->topk_ids[1];
        uint32_t t2 = shm->topk_ids[2];
        uint32_t t3 = shm->topk_ids[3];
        uint32_t t4 = shm->topk_ids[4];
        uint32_t t5 = shm->topk_ids[5];
        uint32_t t6 = shm->topk_ids[6];
        uint32_t t7 = shm->topk_ids[7];

        __uint128_t expert_bitset = 0;
        expert_bitset |= __uint128_t(1) << t0;
        expert_bitset |= __uint128_t(1) << t1;
        expert_bitset |= __uint128_t(1) << t2;
        expert_bitset |= __uint128_t(1) << t3;
        expert_bitset |= __uint128_t(1) << t4;
        expert_bitset |= __uint128_t(1) << t5;
        expert_bitset |= __uint128_t(1) << t6;
        expert_bitset |= __uint128_t(1) << t7;

        uint64_t b0, b1;
        b0 = expert_bitset & 0xFFFFFFFFFFFFFFFFU;
        b1 = expert_bitset >> 64;
        uint32_t expert_count = __popcll(b0) + __popcll(b1);

        uint32_t addend = 0;
        if (b0 == 0) { b0 = b1; addend = 64; }
        uint32_t e0 = __ffsll(b0) - 1 + addend; /* ffs starts counting at 1 */
        b0 &= b0 - 1;
        if (b0 == 0) { b0 = b1; addend = 64; }
        uint32_t e1 = __ffsll(b0) - 1 + addend; /* ffs starts counting at 1 */
        b0 &= b0 - 1;
        if (b0 == 0) { b0 = b1; addend = 64; }
        uint32_t e2 = __ffsll(b0) - 1 + addend; /* ffs starts counting at 1 */
        b0 &= b0 - 1;
        if (b0 == 0) { b0 = b1; addend = 64; }
        uint32_t e3 = __ffsll(b0) - 1 + addend; /* ffs starts counting at 1 */
        b0 &= b0 - 1;
        if (b0 == 0) { b0 = b1; addend = 64; }
        uint32_t e4 = __ffsll(b0) - 1 + addend; /* ffs starts counting at 1 */
        b0 &= b0 - 1;
        if (b0 == 0) { b0 = b1; addend = 64; }
        uint32_t e5 = __ffsll(b0) - 1 + addend; /* ffs starts counting at 1 */
        b0 &= b0 - 1;
        if (b0 == 0) { b0 = b1; addend = 64; }
        uint32_t e6 = __ffsll(b0) - 1 + addend; /* ffs starts counting at 1 */
        b0 &= b0 - 1;
        if (b0 == 0) { b0 = b1; addend = 64; }
        uint32_t e7 = __ffsll(b0) - 1 + addend; /* ffs starts counting at 1 */
        // Note: (b0 & (b0 - 1)) == 0 does not hold here.
        // If expert_count == 7, b1 is copied into b0 again before the last __ffsll

        e1 <<= 8*1;
        e2 <<= 8*2;
        e3 <<= 8*3;
        e5 <<= 8*1;
        e6 <<= 8*2;
        e7 <<= 8*3;
        uint32_t lo_packed_unique_topkids = (e0 | e1) | (e2 | e3);
        uint32_t hi_packed_unique_topkids = (e4 | e5) | (e6 | e7);
        packed_unique_topkids = lo_packed_unique_topkids | ((uint64_t) hi_packed_unique_topkids << 32);
        
        if (threadIdx.x == 0) {
            shm->expert_mask = packed_topkids;
            shm->expert_ids = packed_unique_topkids;
            shm->expert_count = expert_count;
        }
    }
}

/**
 * @brief Prepares the MoE computation for batch size 64 and 16 experts for the matrix multiply functions.
 *
 * Reads @c shm->topk_ids and sets up the necessary data structures and state for @c moe_up_projection and @c moe_down_projection .
 * Specifically, it fills:
 * - @c shm->expert_count
 * - @c shm->experts
 * - @c shm->token_indexes
 *
 * The computation is done redundantly on each CUDA block such that the result can be stored in shared memory.
 *
 * @param batch_size The batch size (active elements in the topk_ids array)
 * @param shm Pointer to shared memory struct to read inputs from and write outputs to.
 *
 */
template <typename Dims>
__device__ static void prepare_moe_BS64_E16(
    std::uint32_t batch_size,
    MoE_SHM<Dims>* __restrict__ shm)
{
    static_assert(Dims::NUM_EXPERTS <= 16, "This function is only for up to 16 experts");
    static_assert(Dims::BS <= 64, "This function is only for up to batch size 64");
    assert(batch_size <= 64);

    constexpr int warp_size = 32;
    const int thread_idx = get_thread<Dims>();
    const int warp_idx = get_any_warp<Dims>();

    // One byte per 16 experts suffices as long as Dims::BS<256
    static_assert(Dims::BS < 256);
    uint8x16_t token_count_local = 0;

    // Every thread does batch_size / warp_size tokens
    for (std::uint32_t i = thread_idx; i < batch_size; i += warp_size) {
        std::uint32_t expert_id = shm->topk_ids[i];
        token_count_local += (uint8x16_t(1) << (expert_id*8));
    }

    const uint8x16_t token_counts = allreduce_sum_across_warp(token_count_local);
    const uint8x16_t token_prefixes = prefix_sum_over_bytes(token_counts);

    // Every thread is now responsible for one expert.
    static_assert(Dims::NUM_EXPERTS <= warp_size); // Need at least one thread per expert in each warp
    // Because Dims::NUM_EXPERTS==16, upper half of the threads has 0
    uint32_t token_count = (token_counts >> (8*thread_idx)) & 0xFF;
    uint32_t token_prefix = (token_prefixes >> (8*thread_idx)) & 0xFF;

    uint32_t nonzero_mask = __ballot_sync(FULL_MASK, token_count != 0);
    uint32_t num_nonzero_elements = __popc(nonzero_mask);
    // "inverse_permut_last" is the source lane for compacting "token_count".
    uint32_t inverse_permut_last = __fns(nonzero_mask, 0, thread_idx + 1);
    uint32_t inverse_permut_first = __fns(nonzero_mask, 0, thread_idx); // Index of token start is the previous thread's end index.
    uint32_t token_start = __shfl_sync(FULL_MASK, token_prefix, inverse_permut_first);
    uint32_t token_end = __shfl_sync(FULL_MASK, token_prefix, inverse_permut_last);

    // Writing is done on the first warp
    if (warp_idx == 0) {
        if (thread_idx == 0) {
            token_start = 0; // Correct the token start index of expert 0
            shm->expert_count = num_nonzero_elements;
        }
        if (thread_idx < num_nonzero_elements) {
            shm->experts[thread_idx].id = inverse_permut_last;
            shm->experts[thread_idx].first_token = token_start;
            shm->experts[thread_idx].last_token = token_end;
        }
    }
    //
    // MoE_SHM::token_indexes is the permutation that sorts topk_ids.
    //
    if constexpr (Dims::BS <= warp_size) { // Can be handled completely within a single warp
        uint32_t eid;
        if (thread_idx < batch_size)
            eid = shm->topk_ids[thread_idx];
        else
            eid = Dims::NUM_EXPERTS;
        // Prefix, where all tokens with expert eid need to go
        uint32_t sorting_prefix = (token_prefixes >> (8*(eid-1))) & 0xFF;
        // Branchless: if (eid == 0) sorting_prefix = 0;
        sorting_prefix = sorting_prefix & ((eid == 0) - 1);

        // For threads with the same sorting_prefix:
        // We define here that their threadIdx determines the relative token order
        uint32_t mask_of_threads_with_same_prefix = __match_any_sync(FULL_MASK, sorting_prefix);
        // Relative order == number of one bits in mask before its position
        uint32_t lower_idx_mask = (1 << thread_idx) - 1;
        uint32_t rank = __popc(mask_of_threads_with_same_prefix & lower_idx_mask);
        uint32_t target_lane_id = sorting_prefix + rank;

        if (warp_idx == 0 && thread_idx < batch_size) {
            shm->token_indexes[target_lane_id] = thread_idx;
        }
    } else if constexpr (Dims::BS <= 2*warp_size) { // Can be handled by two warps
        if (warp_idx < 2) {
            uint32_t eid;
            if (thread_idx + warp_size * warp_idx < batch_size)
                eid = shm->topk_ids[thread_idx + warp_size * warp_idx];
            else
                eid = Dims::NUM_EXPERTS;
            // Prefix, where all tokens with expert eid need to go
            uint32_t sorting_prefix = (token_prefixes >> (8*(eid-1))) & 0xFF;
            // Branchless: if (eid == 0) sorting_prefix = 0;
            sorting_prefix = sorting_prefix & ((eid == 0) - 1);

            // For threads with the same sorting_prefix:
            // We define here that their threadIdx determines the relative token order
            // Tokens that are processed by threads from the first warp are sorted upwards, otherwise downwards.
            uint32_t mask_of_threads_with_same_prefix = __match_any_sync(FULL_MASK, sorting_prefix);
            // Relative order == number of one bits in mask before its position
            uint32_t lower_idx_mask = (1 << thread_idx) - 1;
            uint32_t rank = __popc(mask_of_threads_with_same_prefix & lower_idx_mask);

            uint32_t target_lane_id;
            if (warp_idx == 0) {
                // Warp 0 goes from bottom to top
                target_lane_id = sorting_prefix + rank;
            } else {
                // warp 1 goes from top to bottom
                uint32_t next_sorting_prefix = (token_prefixes >> (8*eid)) & 0xFF;
                target_lane_id = next_sorting_prefix - rank - 1;
            }
            if (thread_idx + warp_size * warp_idx < batch_size) {
                shm->token_indexes[target_lane_id] = thread_idx + warp_idx * warp_size;
            }
        }
    }
}

/**
 * @brief Prepares the MoE computation for batch size 64 and 128 experts for the matrix multiply functions.
 *
 * Reads @c shm->topk_ids and sets up the necessary data structures and state for @c moe_up_projection and @c moe_down_projection .
 * Specifically, it fills:
 * - @c shm->expert_count
 * - @c shm->experts
 * - @c shm->token_indexes
 *
 * The computation is done redundantly on each CUDA block such that the result can be stored in shared memory.
 *
 * This function should support any number of experts, but is only tested for up to 128.
 *
 * @param batch_size The batch size (active elements in the topk_ids array)
 * @param shm Pointer to shared memory struct to read inputs from and write outputs to.
 *
 */
template <typename Dims>
__device__ static void prepare_moe_BSx_Ey(
    std::uint32_t batch_size,
    MoE_SHM<Dims>* __restrict__ shmem)
{
    using CoreDims = MoECoreDims<Dims>;
    using MoE_SHM = MoE_SHM<Dims>;

    typename MoE_SHM::U::SortData* shm = &shmem->u.sorting;

    auto& counters = shm->counters;
    auto& total_counts = shm->total_counts;

    // Implements a Radix sort on the first warp of each CUDA block.
    if (threadIdx.x < CoreDims::THREADS_PER_WARP) {
        // initialize
        for (unsigned e = 0; e < Dims::NUM_EXPERTS; ++e) {
            counters[e][threadIdx.x] = 0;
        }

        // count inputs
        for (unsigned i = threadIdx.x; i < batch_size; i += CoreDims::THREADS_PER_WARP) {
            counters[shmem->topk_ids[i]][threadIdx.x]++;
        }

        __syncwarp();

        // sum up. counters become offsets
        for (unsigned e = threadIdx.x; e < Dims::NUM_EXPERTS; e += CoreDims::THREADS_PER_WARP) {
            std::uint32_t sum = 0;
            for (unsigned i = 0; i < CoreDims::THREADS_PER_WARP; ++i) {
                std::uint32_t prior = sum;
                sum += counters[e][(i + threadIdx.x) % CoreDims::THREADS_PER_WARP];
                counters[e][(i + threadIdx.x) % CoreDims::THREADS_PER_WARP] = prior;
            }
            total_counts[e] = sum;
        }

        __syncwarp();

        // global offsets and expert ranges
        if (threadIdx.x == 0) {
            std::uint32_t sum = 0;
            std::uint32_t expert_count = 0;

            for (unsigned e = 0; e < Dims::NUM_EXPERTS; ++e) {
                std::uint32_t local_count = total_counts[e];

                if (local_count > 0) {
                    std::uint32_t prior = sum;
                    total_counts[e] = prior;
                    sum += local_count;

                    shmem->experts[expert_count].first_token = prior;
                    shmem->experts[expert_count].last_token = sum;
                    shmem->experts[expert_count].id = e;
                    expert_count++;
                }
            }

            shmem->expert_count = expert_count;
        }

        __syncwarp();

        // write index order
        std::uint16_t* ordered = shmem->token_indexes;
        for (unsigned i = threadIdx.x; i < batch_size; i += CoreDims::THREADS_PER_WARP) {
            std::int32_t e = shmem->topk_ids[i];
            unsigned offset = counters[e][threadIdx.x];
            unsigned index = total_counts[e] + offset;
            counters[e][threadIdx.x] = offset + 1;

            ordered[index] = (std::uint16_t)i;
        }
    }
}

/**
 * @brief Dispatch function for @c prepare_moe_BS8_E16 and @c prepare_moe_BS8_E128 .
 */
template <typename Dims>
__device__ __forceinline__ void prepare_moe_BS8(
    std::uint32_t  batch_size,
    MoE_SHM<Dims>* __restrict__ shm)
{
    static_assert(Dims::NUM_EXPERTS <= 128, "This function is only for up to 128 experts");
    static_assert(Dims::BS <= 8, "This function is only for up to batch size 8");
    assert(batch_size <= 8);

    if constexpr (Dims::NUM_EXPERTS <= 16) {
        prepare_moe_BS8_E16(batch_size, shm);
    } else {
        prepare_moe_BS8_E128(batch_size, shm);
    }
}

} // namespace moe_monokernel

#endif
