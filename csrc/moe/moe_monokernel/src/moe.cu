/**
 * This is the main file of the MoE monokernel.
 * It is designed so that you just need to build this file. It includes all relevant implementations.
 * For documentation of the main entry function moe_kernel, see moe_interface.h
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
__device__ void moe_kernel_BS64(
    const A_element* __restrict__ activations_in,
    std::uint32_t batch_size,
    const __nv_bfloat16* __restrict__ router_logits,
    const W_element* __restrict expert_weights_up,
    const S_element* __restrict expert_scales_up,
    const W_element* __restrict expert_weights_down,
    const S_element* __restrict expert_scales_down,
    R_element* __restrict activations_out,
    MoEGemmSpec<Dims>* __restrict__ spec,
    MoE_SHM<Dims>* __restrict__ shmem)
{
    if (is_calc_warp<Dims>()) {
        top1_BS64<Dims>(router_logits, batch_size, shmem);
    }
    __syncthreads();
    prepare_moe_BSx_Ey<Dims>(batch_size, shmem);
    __syncthreads();
    assert(shmem->experts[shmem->expert_count-1].last_token == batch_size);

    moe_scale_activation_BSx<Dims>(
        activations_in,
        batch_size,
        spec,
        shmem);

#ifdef DEBUG_MOE
    if (blockIdx.x == 0) {
        for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
            spec->token_indexes[i] = shmem->token_indexes[i];
        }
    }
#endif

    moe_up_projection<Dims>(
        expert_weights_up,
        expert_scales_up,
        spec,
        shmem);
    cooperative_groups::this_grid().sync();
    moe_down_projection<Dims>(
        batch_size,
        expert_weights_down,
        expert_scales_down,
        activations_out,
        spec,
        shmem);
}

template <typename Dims>
__device__ void moe_kernel_BS8(
    const A_element* __restrict__ activations_in,
    std::uint32_t batch_size,
    const __nv_bfloat16* __restrict__ router_logits,
    const W_element* __restrict expert_weights_up,
    const S_element* __restrict expert_scales_up,
    const W_element* __restrict expert_weights_down,
    const S_element* __restrict expert_scales_down,
    R_element* __restrict activations_out,
    MoEGemmSpec<Dims>* __restrict__ spec,
    MoE_SHM<Dims>* __restrict__ shmem)
{
    static_assert(Dims::BS <= 8);

    using CoreDims = MoECoreDims<Dims>;

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    if (is_prefetch_warp<Dims>()) {
        // Prefetch activations for rescaling
        const std::uint32_t warp = get_prefetch_warp<Dims>();
        for (std::uint32_t token = warp; token < batch_size; token += CoreDims::PREFETCH_WARP_COUNT) {
            moe_fetch_activation_async<Dims>(
                activations_in + token * Dims::HIDDEN_STATES,
                shmem->u.tiny.w[0].orig[token],
                pipe);
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
            expert_weights_up,
            shmem->expert_ids & 0xff,
            shmem->u.tiny.w[1].up,
            pipe);

        pipe.producer_commit();
    }
    else {
        //
        // Rescale activations
        //
        const std::uint32_t warp = get_calc_warp<Dims>();
        if (warp < batch_size) {
            moe_scale_activation_BS8<Dims>(
                shmem->u.tiny.w[0].orig[warp],
                (AQ_element *)shmem->u.tiny.a.up[warp],
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
        expert_weights_up,
        expert_scales_up,
        expert_weights_down,
        expert_scales_down,
        1,
        spec,
        shmem,
        pipe);
    cooperative_groups::this_grid().sync();
    moe_down_projection_tiny<Dims>(
        batch_size,
        expert_weights_down,
        expert_scales_down,
        w_index,
        activations_out,
        spec,
        shmem,
        pipe);
}

template <typename Dims>
__global__ void moe_kernel(
    const A_element* __restrict__ activations_in,
    std::uint32_t token_count,
    const __nv_bfloat16 *__restrict__ router_logits,
    const W_element* __restrict__ expert_weights_up,
    const S_element* __restrict__ expert_scales_up,
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down,
    R_element* __restrict__ activations_out,
    void* __restrict__ scratchpad,
    size_t scratchpad_size,
    size_t shmem_size)
{
    // we require 8 warps per SM and assume X to be the only relevant dimension
    assert(MoECoreDims<Dims>::THREADS_PER_WARP == 32);
    assert(blockDim.x == Dims::KernelConfig::BLOCK_SIZE);
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);

    assert(gridDim.x == Dims::KernelConfig::GRID_SIZE);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    static_assert(Dims::M <= Dims_Max::M, "Dimension larger than the maximum supported dimension.");
    static_assert(Dims::N <= Dims_Max::N, "Dimension larger than the maximum supported dimension.");
    static_assert(Dims::K <= Dims_Max::K, "Dimension larger than the maximum supported dimension.");
    static_assert(Dims::NUM_EXPERTS <= Dims_Max::NUM_EXPERTS, "Dimension larger than the maximum supported dimension.");

    assert(token_count <= Dims::BS);
    assert(token_count > 0);

    assert((uintptr_t) scratchpad % alignof(MoEGemmSpec<Dims>) == 0);
    assert(scratchpad_size >= get_moe_scratchpad_size<Dims>());
    MoEGemmSpec<Dims> *spec = reinterpret_cast<MoEGemmSpec<Dims> *>(scratchpad);

    assert(shmem_size >= get_moe_shmem_size<Dims>());

    extern __shared__ char shmem_buffer[];
    MoE_SHM<Dims>* shmem = reinterpret_cast<MoE_SHM<Dims>*>(shmem_buffer);

    if constexpr (Dims::BS <= 8) {
        moe_kernel_BS8(
                activations_in,
                token_count,
                router_logits,
                expert_weights_up,
                expert_scales_up,
                expert_weights_down,
                expert_scales_down,
                activations_out,
                spec,
                shmem);
    } else {
        moe_kernel_BS64(
                activations_in,
                token_count,
                router_logits,
                expert_weights_up,
                expert_scales_up,
                expert_weights_down,
                expert_scales_down,
                activations_out,
                spec,
                shmem);
    }
}

} // namespace moe_monokernel
