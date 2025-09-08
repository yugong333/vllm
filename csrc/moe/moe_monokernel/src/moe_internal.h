
#pragma once
#ifndef MOE_INTERNAL_H
#define MOE_INTERNAL_H

#ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
#error Do not include this file directly.
#endif

#include "moe_interface.h"

namespace moe_monokernel {

using T_element = float; //< Type of GEMM1 (up projection) result as well as sigmoid
using OpaqueElement = std::uint32_t; //< Auxiliary 32-bit type used to generate better assembly code in loads

/**
 * @brief Offets into the @c token_indexes field
 *
 * This is an offset array. To find all the tokens that belong to expert @c id :
 * <tt>
 * for (int i = first_token; i < last_token; i++) {
 *    int token_index = token_indexes[i];
 * }
 */
struct ExpertRef
{
    std::uint16_t first_token;
    std::uint16_t last_token;
    std::uint32_t id;
};

/**
 * @brief Scratchpad memory for use within the monokernel.
 *
 * Place in global memory.
 *
 */
template <typename Dims>
struct MoEGemmSpec
{
#ifdef DEBUG_MOE
    // Debug information passed out. The actual token_indexes are stored in shared memory.
    std::int32_t token_indexes[Dims::BS];
    T_element gemm1[(Dims::BS + 8) * 2 * Dims::N];
#endif
    AQ_element activations[Dims::BS][Dims::HIDDEN_STATES]; //< Quantized activations
    T_element temp[(Dims::BS + 8) * Dims::N]; //< Up projection result
    float topk_weights_scaled[Dims::BS]; //< topk_weights multiplied with the activation quantization
};


// Maximum supported dimensions for shared memory and scratchpad allocation sizes
#if USE_SMALL_SETUP
// SHM limits batch size to ~2k
using Dims_Max = MoEDimensions<1024, 256, 1024, 128>;
#else
using Dims_Max = MoEDimensions<1024, 1024, 5120, 128>;
#endif

/**
 * @brief contains various constants used within the MoE monokernel.
 */
template <typename Dims>
struct MoECoreDims {
    using MoEDims = Dims;

    // GPU configuration.
    static constexpr std::uint32_t THREADS_PER_WARP = 32;
    static constexpr std::uint32_t TOTAL_WARP_COUNT = Dims::KernelConfig::BLOCK_SIZE / THREADS_PER_WARP;
    static constexpr std::uint32_t CALC_WARP_COUNT = 8;
    static constexpr std::uint32_t PREFETCH_WARP_COUNT = TOTAL_WARP_COUNT - CALC_WARP_COUNT;

    // MMA 1 matrix tile dimensions.
    static constexpr std::uint32_t A_TILE =     8;
    static constexpr std::uint32_t W_UP_TILE = 16;
    static constexpr std::uint32_t K_TILE =    32;

    // GEMM 2 matrix tile dimensions.
    static constexpr std::uint32_t W_DOWN_MMA_TILE = 16;
    static constexpr std::uint32_t W_DOWN_TILE = Dims::HIDDEN_STATES / Dims::KernelConfig::GRID_SIZE;
    static constexpr std::uint32_t T_TILE = 8;

    static constexpr std::uint32_t W_DIM = 2 * Dims::N;

    static constexpr unsigned BLOCK_STRIDE = CALC_WARP_COUNT * K_TILE;

    static constexpr unsigned PADDING = 32;     // this works *slightly* better than 16 due to reduced L2 transfers
    static constexpr unsigned K_DIM_PADDED_A = Dims::HIDDEN_STATES;
    static constexpr unsigned K_DIM_PADDED_W = Dims::HIDDEN_STATES;
    static constexpr unsigned K_DIM_HALF_PADDED_A = Dims::HIDDEN_STATES / 2;
    static constexpr unsigned K_DIM_HALF_PADDED_W = Dims::HIDDEN_STATES / 2;
};

// 1 tile per warp
// 20 warps x 2 params x 1k = 20k pre-fetch
template <typename Dims>
struct MoE_SHM {
    using CoreDims = MoECoreDims<Dims>;
    union U {
        struct SortData {
            std::uint32_t counters[Dims::NUM_EXPERTS][CoreDims::THREADS_PER_WARP];
            std::uint32_t total_counts[Dims::NUM_EXPERTS];
        } sorting;
        struct RescaleData {
            A_element a[CoreDims::CALC_WARP_COUNT][Dims::HIDDEN_STATES];
        } rescale;
        struct Gemm1Data {
            // prefetch & process tile in 2 halves
            AQ_element a[3][CoreDims::A_TILE][CoreDims::K_DIM_HALF_PADDED_A];
            W_element w[3][CoreDims::W_UP_TILE][CoreDims::K_DIM_HALF_PADDED_W];
            T_element partial_result[CoreDims::CALC_WARP_COUNT][CoreDims::W_UP_TILE * CoreDims::T_TILE];
        } gemm1;
        // GEMM2 uses the same data structure as Tiny
        struct TinyData {
            // input activations for up- and down-projection
            union {
                AQ_element up[CoreDims::T_TILE][CoreDims::K_DIM_PADDED_A];
                T_element down[CoreDims::T_TILE][Dims::N];
            } a;

            // prefetch & process tile in 2 halves
            union {
                A_element orig[CoreDims::T_TILE][CoreDims::K_DIM_PADDED_A];      // input activations to be scaled 
                W_element up[CoreDims::W_UP_TILE][CoreDims::K_DIM_PADDED_W];     // up-projection weights
                W_element down[CoreDims::W_DOWN_TILE][Dims::N + CoreDims::PADDING / sizeof(W_element)]; // down-projection weights
            } w[2];

            // down-projection scales
            S_element scale[2][CoreDims::W_DOWN_TILE + CoreDims::PADDING];

            // scratch pad
            union {
                T_element up[CoreDims::CALC_WARP_COUNT][CoreDims::W_UP_TILE * CoreDims::T_TILE];
                T_element down[CoreDims::W_DOWN_TILE / 2 + CoreDims::CALC_WARP_COUNT / 2][CoreDims::W_DOWN_MMA_TILE * CoreDims::T_TILE];
            } partial_result;
        } tiny;
        struct Gemm2Data {
            // prefetch 1 tile ahead
            T_element t[2][CoreDims::T_TILE][Dims::N];
            W_element w[2][CoreDims::W_DOWN_TILE][Dims::N + CoreDims::PADDING / sizeof(W_element)];
            S_element scale[2][CoreDims::W_DOWN_TILE + CoreDims::PADDING];
            T_element partial_result[CoreDims::W_DOWN_TILE / 2 + CoreDims::CALC_WARP_COUNT / 2][CoreDims::W_DOWN_MMA_TILE * CoreDims::T_TILE];
        } gemm2;
    } u;

    static_assert(Dims::NUM_EXPERTS < 255, "Number of experts too high, cannot store as uint8 anymore.");
    alignas(uint64_t) uint8_t topk_ids[Dims::BS < 8? 8: Dims::BS];
    std::uint16_t token_indexes[Dims::BS + CoreDims::PADDING];
    S_element topk_weights[Dims::BS];
    ExpertRef experts[Dims::NUM_EXPERTS];

    // 8 packed 8-bit expert id values.
    // 0xff for "unused"
    // only value if token count <= 8
    std::uint64_t expert_mask;
    std::uint64_t expert_ids;
    std::uint32_t expert_count;
};

/**
 * @brief Returns the amount of shared memory necessary to run @c moe_kernel with template parameter @p Dims
 */
template <typename Dims>
__device__ __host__ constexpr size_t get_moe_shmem_size()
{
    static_assert(Dims::M <= Dims_Max::M, "Dimension larger than the maximum supported dimension.");
    static_assert(Dims::N <= Dims_Max::N, "Dimension larger than the maximum supported dimension.");
    static_assert(Dims::K <= Dims_Max::K, "Dimension larger than the maximum supported dimension.");
    static_assert(Dims::NUM_EXPERTS <= Dims_Max::NUM_EXPERTS, "Dimension larger than the maximum supported dimension.");
    return sizeof(MoE_SHM<Dims>);
}

constexpr size_t get_moe_max_shmem_size()
{
    return sizeof(MoE_SHM<Dims_Max>);
}

/**
 * @brief Returns the amount of global scratchpad memory necessary to run moe_kernel() with template parameter @p Dims
 */
template <typename Dims>
__device__ __host__ constexpr size_t get_moe_scratchpad_size()
{
    static_assert(Dims::M <= Dims_Max::M, "Dimension larger than the maximum supported dimension.");
    static_assert(Dims::N <= Dims_Max::N, "Dimension larger than the maximum supported dimension.");
    static_assert(Dims::K <= Dims_Max::K, "Dimension larger than the maximum supported dimension.");
    static_assert(Dims::NUM_EXPERTS <= Dims_Max::NUM_EXPERTS, "Dimension larger than the maximum supported dimension.");
    return sizeof(MoEGemmSpec<Dims>);
}

constexpr size_t get_moe_max_scratchpad_size()
{
    return sizeof(MoEGemmSpec<Dims_Max>);
}

template <typename Dims>
inline __device__ bool is_calc_warp()
{
    using CoreDims = MoECoreDims<Dims>;
    return threadIdx.x < CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP;
}

template <typename Dims>
inline __device__ bool is_prefetch_warp()
{
    using CoreDims = MoECoreDims<Dims>;
    return threadIdx.x >= CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP;
}

template <typename Dims>
inline __device__ unsigned get_thread()
{
    using CoreDims = MoECoreDims<Dims>;
    return threadIdx.x % CoreDims::THREADS_PER_WARP;
}

template <typename Dims>
inline __device__ unsigned get_any_warp()
{
    using CoreDims = MoECoreDims<Dims>;
    return threadIdx.x / CoreDims::THREADS_PER_WARP;
}

template <typename Dims>
inline __device__ unsigned get_calc_warp()
{
    using CoreDims = MoECoreDims<Dims>;
    assert(is_calc_warp<Dims>());
    return threadIdx.x / CoreDims::THREADS_PER_WARP;
}

template <typename Dims>
inline __device__ unsigned get_prefetch_warp()
{
    using CoreDims = MoECoreDims<Dims>;
    assert(is_prefetch_warp<Dims>());
    return threadIdx.x / CoreDims::THREADS_PER_WARP - CoreDims::CALC_WARP_COUNT;
}

/**
 * @brief Synchronizes the first 256 threads of the calling CUDA block
 *
 * This is a collective operation that needs to be called by all of the first 256 threads in each CUDA block.
 *
 */
template <typename Dims>
__device__ __forceinline__ void sync_calc_threads()
{
    // First 256 threads
    using CoreDims = MoECoreDims<Dims>;
    static_assert(CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP == 256, "Adapt the thread number if sync_calc_threads");
    __asm volatile("bar.sync  15, 256;\n");
}

/**
 * @brief Computes the maximum value within a warp
 *
 * This is a collective operation. Each thread in a warp needs to call it.
 * The resulting maximum value is returned on all threads.
 *
 */
__device__ static inline float warp_reduce_max_float(float value)
{
    for (int i = 16; i >= 1; i /= 2) {
        value = fmaxf(__shfl_xor_sync(0xffffffff, value, i, 32), value);
    }
    return value;
}

/**
 * @brief Reinterprets the bit-pattern of @p x to type @p To
 */
template <typename To, typename From>
__device__ static __forceinline__ To type_pun(From x)
{
    static_assert(sizeof(To) == sizeof(From), "Types of different size");
    To y;
    // This memcpy is optimized out by NVCC
    memcpy(&y, &x, sizeof(From));
    return y;
}

} // namespace moe_monokernel

#endif
