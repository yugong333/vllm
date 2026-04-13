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

// Pre-defined dimensions for Qwen3-Coder-30B-A3B (TP=1)
// w13: [128, 1536, 2048] → N=768 (half of fused gate+up), K=2048
// w2:  [128, 2048, 768]
//
// Note: the default GRID_SIZE formula (2*N/16 = 96) does not evenly divide
// HIDDEN_STATES=2048, so we override KernelConfig with GRID_SIZE=128 which
// gives W_DOWN_TILE = 2048/128 = 16 (satisfies the % 8 == 0 constraint).
struct Dims_BS8_E128_Qwen3Coder {
  static constexpr uint32_t HIDDEN_STATES = 2048;
  static constexpr uint32_t K = 2048;
  static constexpr uint32_t N = 768;
  static constexpr uint32_t BS = 8;
  static constexpr uint32_t M = 8;
  static constexpr uint32_t NUM_EXPERTS = 128;
  struct KernelConfig {
    static constexpr std::uint32_t GRID_SIZE = 128;
    static constexpr std::uint32_t BLOCK_SIZE = 384;
  };
};

struct Dims_BS64_E128_Qwen3Coder {
  static constexpr uint32_t HIDDEN_STATES = 2048;
  static constexpr uint32_t K = 2048;
  static constexpr uint32_t N = 768;
  static constexpr uint32_t BS = 64;
  static constexpr uint32_t M = 64;
  static constexpr uint32_t NUM_EXPERTS = 128;
  struct KernelConfig {
    static constexpr std::uint32_t GRID_SIZE = 128;
    static constexpr std::uint32_t BLOCK_SIZE = 384;
  };
};

// Scoring function enum for routing
enum class ScoringFunc : uint32_t {
  SIGMOID = 0,
  SOFTMAX = 1,
};

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

/**
 * @brief W8A8 MoE kernel with configurable top-K routing, scoring function,
 *        and renormalization.
 *
 * This kernel extends moe_kernel() to support top-K expert selection (K >= 1),
 * configurable scoring functions (softmax or sigmoid), and optional weight
 * renormalization. It is designed for models like Qwen3 Coder FP8 that use
 * softmax scoring with top_k=8.
 *
 * The kernel processes each token by selecting the top-K experts, computing
 * routing weights via the specified scoring function, optionally renormalizing
 * the weights to sum to 1, and then accumulating the weighted expert outputs.
 *
 * @param [in] activations_in Input activations. Shape: [M, K]
 * @param [in] token_count Number of active tokens
 * @param [in] router_logits Router logits. Shape: [M, E]
 * @param [in] expert_weights_up Up-projection weights. Shape: [E, 2*N, K]
 * @param [in] expert_scales_up Up-projection scales. Shape: [E, 2*N]
 * @param [in] expert_weights_down Down-projection weights. Shape: [E, K, N]
 * @param [in] expert_scales_down Down-projection scales. Shape: [E, K]
 * @param [out] activations_out Output buffer. Shape: [M, K]
 * @param [out] scratchpad Global memory for temporary data
 * @param [in] scratchpad_size Size of the scratchpad
 * @param [in] shmem_size Size of the shared memory
 * @param [in] top_k Number of experts to select per token
 * @param [in] scoring_func Scoring function (SIGMOID or SOFTMAX)
 * @param [in] renormalize Whether to renormalize top-K weights to sum to 1
 */
template <typename Dims>
__global__ extern void moe_kernel_topk(
    const A_element* __restrict__ activations_in, std::uint32_t token_count,
    const __nv_bfloat16* __restrict__ router_logits,
    const W_element* __restrict expert_weights_up,
    const S_element* __restrict expert_scales_up,
    const W_element* __restrict expert_weights_down,
    const S_element* __restrict expert_scales_down,
    R_element* __restrict activations_out, void* __restrict__ scratchpad,
    size_t scratchpad_size, size_t shmem_size, std::uint32_t top_k,
    ScoringFunc scoring_func, bool renormalize);

}  // namespace moe_monokernel

#endif
