#ifndef MOE_INTERFACE_H
#define MOE_INTERFACE_H

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace moe_monokernel {

template <uint32_t m, uint32_t n, uint32_t k, uint32_t num_experts>
struct MoEDimensions {
  static constexpr uint32_t HIDDEN_STATES = k;
  static constexpr uint32_t K = k;
  static constexpr uint32_t N = n;
  static constexpr uint32_t BS = m;
  static constexpr uint32_t M = m;
  static constexpr uint32_t NUM_EXPERTS = num_experts;

  struct KernelConfig {
    static constexpr std::uint32_t GRID_SIZE = (2 * N) / 16;
    static constexpr std::uint32_t BLOCK_SIZE = 384;
  };
};

// Pre-defined dimensions for Llama4 Scout and Maverick
using Dims_BS8_E16_TP8 = MoEDimensions<8, 1024, 5120, 16>;
using Dims_BS64_E16_TP8 = MoEDimensions<64, 1024, 5120, 16>;
using Dims_BS8_E128_TP8 = MoEDimensions<8, 1024, 5120, 128>;
using Dims_BS64_E128_TP8 = MoEDimensions<64, 1024, 5120, 128>;

using W_element = __nv_fp8_e4m3;   // expert weights
using A_element = __nv_bfloat16;   // activations as they go into the GEMM
using AQ_element = __nv_fp8_e4m3;  // activations after quantization
using S_element = float;           // scaling factors
using R_element = __nv_bfloat16;   // MoE output

/**
 * @brief Returns the maximum amount of shared memory necessary to run
 * moe_kernel()
 */
constexpr size_t get_moe_max_shmem_size();

/**
 * @brief Returns the maximum amount of global scratchpad memory to run
 * moe_kernel()
 */
constexpr size_t get_moe_max_scratchpad_size();

/**
 * @brief W8A8 MoE kernel
 *
 * This function implements a W8A8 Mixture-of-Experts kernel.
 * It routes each input token to the top 1 expert as determined by the @p
 * routing_logits .
 *
 * Inputs:
 * Activations are provided as bfloat16. They are quantized to FP8 E4M3 before
 * the matrix multiplies. Expert weights are provided as FP8 E4M3.
 *
 * This function is templatized. For best performance, instantiate all template
 * parameters with the respective parameters that you pass at runtime. For the
 * batch-size, use at least two instantiations: One for BS=8 and one for BS=64.
 * Instances of this kernel can handle runtime token counts <tt>token_count <=
 * Dims::BS</tt>, i.e. you need to instantiate the kernel with at least the
 * number of runtime tokens. This can be achieved e.g. with a switch-case
 * statement.
 *
 * This is a cooperative kernel. It needs to be launched via
 * cudaLaunchCooperativeKernel(). This kernel needs at least
 * get_moe_max_shmem_size() shared memory. Set it via <tt>
 * cudaFuncSetAttribute(moe_kernel<KernelDims>,
 * cudaFuncAttributeMaxDynamicSharedMemorySize, get_moe_max_shmem_size()));
 * </tt>
 * before calling!
 *
 * On top of the shared memory, the kernel stores temporaries in global device
 * memory. This memory needs to be allocated by the user. It has to be at least
 * get_moe_max_scratchpad_size() Bytes.
 *
 * @note All input tensors are considered to be row-major and contiguous! I.e.
 * no padding and the stride is the product of the trailing dimensions.
 *
 * In the parameter descriptions, we use the following shorthand constants:
 * - The batch size M, @c Dims::BS
 * - The number of experts E, @c Dims::NUM_EXPERTS
 * - The number of hidden states K, @c Dims::HIDDEN_STATES
 * - The up-projection dimension N, @c Dims::N
 *
 * @param [in] activations_in Input activations. Shape: [M, K]
 * @param [in] token_count Number of active tokens
 * @param [in] router_logits Result of routing matrix multiply. Determines which
 * expert each token is routed to. Shape: [M, E]
 * @param [in] expert_weights_up Scales for the weights of the down projection.
 * Shape: [E, 2*N, K]
 * @param [in] expert_scales_up Scales for the weights of the down projection.
 * Shape: [E, 2*N]
 * @param [in] expert_weights_down Scales for the weights of the down
 * projection. Shape: [E, K, N]
 * @param [in] expert_scales_down Scales for the weights of the down projection.
 * Shape: [E, K]
 * @param [out] activations_out Pointer to the output buffer. Shape: [M, K]
 * @param [out] scratchpad Global memory to use for temporary data
 * @param [in] scratchpad_size Size of the scratchpad. The kernel uses it to
 * check if the caller allocated enough storage.
 * @param [in] shmem_size Size of the shared memory. The kernel uses it to check
 * if the caller allocated enough storage.
 */
template <typename Dims>
__global__ extern void moe_kernel(
    const A_element* __restrict__ activations_in, std::uint32_t token_count,
    const __nv_bfloat16* __restrict__ router_logits,
    const W_element* __restrict expert_weights_up,
    const S_element* __restrict expert_scales_up,
    const W_element* __restrict expert_weights_down,
    const S_element* __restrict expert_scales_down,
    R_element* __restrict activations_out, void* __restrict__ scratchpad,
    size_t scratchpad_size, size_t shmem_size);

}  // namespace moe_monokernel

#endif
