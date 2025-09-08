#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>

#include "cuda_utils.h"

#include "src/moe.cu"

/**
 * @brief Macro that expands to a kernel call wrapper for moe_kernel with specified @p dims
 *
 * moe_kernel() needs to be instantiated with different parameters. In the past, dispatching
 * to the different kernels on the device or on the host within C++ code generated problems
 * in vLLM and PyTorch. So these wrappers are to move this dispatch to Python.
 * For each instantiated kernel, we generate one host invocation function.
 */
#define MOEMONOKERNEL_WRAPPER_IMPLEMENTATION(name, dims) \
    void name( \
        const torch::Tensor& activations_in, \
        const torch::Tensor& router_logits, \
        const torch::Tensor& expert_weights_up, \
        const torch::Tensor& expert_scales_up, \
        const torch::Tensor& expert_weights_down, \
        const torch::Tensor& expert_scales_down, \
        torch::Tensor& activations_out, \
        torch::Tensor& scratchpad) \
    { \
        /* Check if the input tensors are on the GPU. */ \
        TORCH_CHECK(activations_in.is_cuda(), "Optimized MoE kernel must be called with CUDA tensors only."); \
        TORCH_CHECK(router_logits.is_cuda(), "Optimized MoE kernel must be called with CUDA tensors only."); \
        TORCH_CHECK(expert_weights_up.is_cuda(), "Optimized MoE kernel must be called with CUDA tensors only."); \
        TORCH_CHECK(expert_scales_up.is_cuda(), "Optimized MoE kernel must be called with CUDA tensors only."); \
        TORCH_CHECK(expert_weights_down.is_cuda(), "Optimized MoE kernel must be called with CUDA tensors only."); \
        TORCH_CHECK(expert_scales_down.is_cuda(), "Optimized MoE kernel must be called with CUDA tensors only."); \
        TORCH_CHECK(activations_out.is_cuda(), "Optimized MoE kernel must be called with CUDA tensors only."); \
        TORCH_CHECK(scratchpad.is_cuda(), "Optimized MoE kernel must be called with CUDA tensors only."); \
 \
        /* Get raw data pointers from the PyTorch tensors. */ \
        const auto* activations_in_ptr = activations_in.data_ptr<at::BFloat16>(); \
        const auto* router_logits_ptr = router_logits.data_ptr<at::BFloat16>(); \
        const auto* expert_weights_up_ptr = expert_weights_up.data_ptr<at::Float8_e4m3fn>(); \
        const auto* expert_scales_up_ptr = expert_scales_up.data_ptr<float>(); \
        const auto* expert_weights_down_ptr = expert_weights_down.data_ptr<at::Float8_e4m3fn>(); \
        const auto* expert_scales_down_ptr = expert_scales_down.data_ptr<float>(); \
        auto* activations_out_ptr = activations_out.data_ptr<at::BFloat16>(); \
        char* scratchpad_ptr = reinterpret_cast<char*>(scratchpad.data_ptr<float>()); \
 \
        using namespace moe_monokernel; \
        const uint32_t num_tokens = activations_in.size(0); \
        const uint32_t num_experts = expert_weights_up.size(0); \
        const size_t shmem_size = get_moe_max_shmem_size(); \
        const size_t scratchpad_size = scratchpad.nbytes(); \
 \
        void *kernel_args[] = { \
            (void *)&activations_in_ptr, \
            (void *)&num_tokens, \
            (void *)&router_logits_ptr, \
            (void *)&expert_weights_up_ptr, \
            (void *)&expert_scales_up_ptr, \
            (void *)&expert_weights_down_ptr, \
            (void *)&expert_scales_down_ptr, \
            (void *)&activations_out_ptr, \
            (void *)&scratchpad_ptr, \
            (void *)&scratchpad_size, \
            (void *)&shmem_size \
        }; \
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(); \
        CUDA_CHECK(cudaFuncSetAttribute(moe_kernel<dims>, \
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                        shmem_size)); \
        CUDA_CHECK(cudaLaunchCooperativeKernel( \
                moe_kernel<dims>, \
                dims::KernelConfig::GRID_SIZE, \
                dims::KernelConfig::BLOCK_SIZE, \
                kernel_args, \
                shmem_size, \
                stream)); \
    }

MOEMONOKERNEL_WRAPPER_IMPLEMENTATION(moe_monokernel_BS8_E16_TP8_impl, moe_monokernel::Dims_BS8_E16_TP8)
MOEMONOKERNEL_WRAPPER_IMPLEMENTATION(moe_monokernel_BS64_E16_TP8_impl, moe_monokernel::Dims_BS64_E16_TP8)

MOEMONOKERNEL_WRAPPER_IMPLEMENTATION(moe_monokernel_BS8_E128_TP8_impl, moe_monokernel::Dims_BS8_E128_TP8)
MOEMONOKERNEL_WRAPPER_IMPLEMENTATION(moe_monokernel_BS64_E128_TP8_impl, moe_monokernel::Dims_BS64_E128_TP8)
