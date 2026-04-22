#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>

#include "cuda_utils.h"

#include "src/moe.cu"

/**
 * @brief Macro that expands to a kernel call wrapper for moe_kernel_topk with
 * specified @p dims and configurable top_k, scoring_func, and renormalize.
 */
#define MOEMONOKERNEL_TOPK_WRAPPER_IMPLEMENTATION(name, dims)                  \
  void name(const torch::Tensor& activations_in,                               \
            const torch::Tensor& router_logits,                                \
            const torch::Tensor& expert_weights_up,                            \
            const torch::Tensor& expert_scales_up,                             \
            const torch::Tensor& expert_weights_down,                          \
            const torch::Tensor& expert_scales_down,                           \
            torch::Tensor& activations_out, torch::Tensor& scratchpad,         \
            int64_t top_k, int64_t scoring_func, bool renormalize) {           \
    TORCH_CHECK(                                                               \
        activations_in.is_cuda(),                                              \
        "Optimized MoE kernel must be called with CUDA tensors only.");        \
    TORCH_CHECK(                                                               \
        router_logits.is_cuda(),                                               \
        "Optimized MoE kernel must be called with CUDA tensors only.");        \
    TORCH_CHECK(                                                               \
        expert_weights_up.is_cuda(),                                           \
        "Optimized MoE kernel must be called with CUDA tensors only.");        \
    TORCH_CHECK(                                                               \
        expert_scales_up.is_cuda(),                                            \
        "Optimized MoE kernel must be called with CUDA tensors only.");        \
    TORCH_CHECK(                                                               \
        expert_weights_down.is_cuda(),                                         \
        "Optimized MoE kernel must be called with CUDA tensors only.");        \
    TORCH_CHECK(                                                               \
        expert_scales_down.is_cuda(),                                          \
        "Optimized MoE kernel must be called with CUDA tensors only.");        \
    TORCH_CHECK(                                                               \
        activations_out.is_cuda(),                                             \
        "Optimized MoE kernel must be called with CUDA tensors only.");        \
    TORCH_CHECK(                                                               \
        scratchpad.is_cuda(),                                                  \
        "Optimized MoE kernel must be called with CUDA tensors only.");        \
    TORCH_CHECK(top_k >= 1 && top_k <= 8, "top_k must be between 1 and 8.");   \
    TORCH_CHECK(scoring_func == 0 || scoring_func == 1,                        \
                "scoring_func must be 0 (sigmoid) or 1 (softmax).");           \
                                                                               \
    const auto* activations_in_ptr = activations_in.data_ptr<at::BFloat16>();  \
    const auto* router_logits_ptr = router_logits.data_ptr<at::BFloat16>();    \
    const auto* expert_weights_up_ptr =                                        \
        expert_weights_up.data_ptr<at::Float8_e4m3fn>();                       \
    const auto* expert_scales_up_ptr = expert_scales_up.data_ptr<float>();     \
    const auto* expert_weights_down_ptr =                                      \
        expert_weights_down.data_ptr<at::Float8_e4m3fn>();                     \
    const auto* expert_scales_down_ptr = expert_scales_down.data_ptr<float>(); \
    auto* activations_out_ptr = activations_out.data_ptr<at::BFloat16>();      \
    char* scratchpad_ptr =                                                     \
        reinterpret_cast<char*>(scratchpad.data_ptr<float>());                 \
                                                                               \
    using namespace moe_monokernel;                                            \
    const uint32_t num_tokens = activations_in.size(0);                        \
    const size_t shmem_size = get_moe_shmem_size<dims>();                      \
    const size_t scratchpad_size = scratchpad.nbytes();                        \
    const uint32_t top_k_u32 = static_cast<uint32_t>(top_k);                   \
    const ScoringFunc sf = static_cast<ScoringFunc>(scoring_func);             \
                                                                               \
    void* kernel_args[] = {(void*)&activations_in_ptr,                         \
                           (void*)&num_tokens,                                 \
                           (void*)&router_logits_ptr,                          \
                           (void*)&expert_weights_up_ptr,                      \
                           (void*)&expert_scales_up_ptr,                       \
                           (void*)&expert_weights_down_ptr,                    \
                           (void*)&expert_scales_down_ptr,                     \
                           (void*)&activations_out_ptr,                        \
                           (void*)&scratchpad_ptr,                             \
                           (void*)&scratchpad_size,                            \
                           (void*)&shmem_size,                                 \
                           (void*)&top_k_u32,                                  \
                           (void*)&sf,                                         \
                           (void*)&renormalize};                               \
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();              \
    CUDA_CHECK(cudaFuncSetAttribute(                                           \
        moe_kernel_topk<dims>, cudaFuncAttributeMaxDynamicSharedMemorySize,    \
        shmem_size));                                                          \
    CUDA_CHECK(cudaLaunchCooperativeKernel(                                    \
        moe_kernel_topk<dims>, dims::KernelConfig::GRID_SIZE,                  \
        dims::KernelConfig::BLOCK_SIZE, kernel_args, shmem_size, stream));     \
  }

// Qwen3-Coder-30B-A3B (E=128, K=2048, N=768, TP=1)
MOEMONOKERNEL_TOPK_WRAPPER_IMPLEMENTATION(
    moe_monokernel_topk_BS8_E128_Qwen3Coder_impl,
    moe_monokernel::Dims_BS8_E128_Qwen3Coder)
MOEMONOKERNEL_TOPK_WRAPPER_IMPLEMENTATION(
    moe_monokernel_topk_BS64_E128_Qwen3Coder_impl,
    moe_monokernel::Dims_BS64_E128_Qwen3Coder)

///////////////////////////////////////////////////////////////////////////////
//
// TMA-accelerated kernel wrappers
//
// These create CUtensorMap descriptors on the host and launch the TMA
// kernel variant.  The descriptors encode the global memory layout and
// shared memory tile dimensions so the hardware can do bulk copies.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief Macro that expands to a TMA kernel call wrapper.
 *
 * Creates TMA tensor map descriptors for the weight and activation tensors,
 * then launches moe_kernel_topk_tma.
 */
#define MOEMONOKERNEL_TOPK_TMA_WRAPPER_IMPLEMENTATION(name, dims)              \
  void name(const torch::Tensor& activations_in,                               \
            const torch::Tensor& router_logits,                                \
            const torch::Tensor& expert_weights_up,                            \
            const torch::Tensor& expert_scales_up,                             \
            const torch::Tensor& expert_weights_down,                          \
            const torch::Tensor& expert_scales_down,                           \
            torch::Tensor& activations_out, torch::Tensor& scratchpad,         \
            int64_t top_k, int64_t scoring_func, bool renormalize) {           \
    TORCH_CHECK(                                                               \
        activations_in.is_cuda(),                                              \
        "Optimized MoE kernel must be called with CUDA tensors only.");        \
    TORCH_CHECK(                                                               \
        router_logits.is_cuda(),                                               \
        "Optimized MoE kernel must be called with CUDA tensors only.");        \
    TORCH_CHECK(                                                               \
        expert_weights_up.is_cuda(),                                           \
        "Optimized MoE kernel must be called with CUDA tensors only.");        \
    TORCH_CHECK(                                                               \
        expert_scales_up.is_cuda(),                                            \
        "Optimized MoE kernel must be called with CUDA tensors only.");        \
    TORCH_CHECK(                                                               \
        expert_weights_down.is_cuda(),                                         \
        "Optimized MoE kernel must be called with CUDA tensors only.");        \
    TORCH_CHECK(                                                               \
        expert_scales_down.is_cuda(),                                          \
        "Optimized MoE kernel must be called with CUDA tensors only.");        \
    TORCH_CHECK(                                                               \
        activations_out.is_cuda(),                                             \
        "Optimized MoE kernel must be called with CUDA tensors only.");        \
    TORCH_CHECK(                                                               \
        scratchpad.is_cuda(),                                                  \
        "Optimized MoE kernel must be called with CUDA tensors only.");        \
    TORCH_CHECK(top_k >= 1 && top_k <= 8, "top_k must be between 1 and 8.");   \
    TORCH_CHECK(scoring_func == 0 || scoring_func == 1,                        \
                "scoring_func must be 0 (sigmoid) or 1 (softmax).");           \
                                                                               \
    const auto* activations_in_ptr = activations_in.data_ptr<at::BFloat16>();  \
    const auto* router_logits_ptr = router_logits.data_ptr<at::BFloat16>();    \
    const auto* expert_weights_up_ptr =                                        \
        expert_weights_up.data_ptr<at::Float8_e4m3fn>();                       \
    const auto* expert_scales_up_ptr = expert_scales_up.data_ptr<float>();     \
    const auto* expert_weights_down_ptr =                                      \
        expert_weights_down.data_ptr<at::Float8_e4m3fn>();                     \
    const auto* expert_scales_down_ptr = expert_scales_down.data_ptr<float>(); \
    auto* activations_out_ptr = activations_out.data_ptr<at::BFloat16>();      \
    char* scratchpad_ptr =                                                     \
        reinterpret_cast<char*>(scratchpad.data_ptr<float>());                 \
                                                                               \
    using namespace moe_monokernel;                                            \
    using CoreDims = MoECoreDims<dims>;                                        \
    const uint32_t num_tokens = activations_in.size(0);                        \
    const size_t shmem_size = get_moe_shmem_size<dims>();                      \
    const size_t scratchpad_size = scratchpad.nbytes();                        \
    const uint32_t top_k_u32 = static_cast<uint32_t>(top_k);                   \
    const ScoringFunc sf = static_cast<ScoringFunc>(scoring_func);             \
                                                                               \
    /* ── Create TMA tensor map descriptors ─────────────────────────── */     \
    MoETmaDescriptors tma_descs;                                               \
    CUresult res;                                                              \
                                                                               \
    /* Up-projection weights: [E*2*N, K] fp8                           */      \
    /* Tile: [W_UP_TILE/2, K] (or K/2 for half-K path)                */       \
    /* Using 32B swizzle to match rotate_col_32                        */      \
    res = tma_create_tensor_map_2d(                                            \
        &tma_descs.w_up, expert_weights_up_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8, \
        /*rows=*/dims::NUM_EXPERTS * 2 * dims::N,                              \
        /*cols=*/dims::HIDDEN_STATES,                                          \
        /*smem_box_rows=*/CoreDims::W_UP_TILE / 2,                             \
        /*smem_box_cols=*/dims::HIDDEN_STATES, CU_TENSOR_MAP_SWIZZLE_32B);     \
    TORCH_CHECK(res == CUDA_SUCCESS,                                           \
                "Failed to create TMA descriptor for w_up: ", res);            \
                                                                               \
    /* Down-projection weights: [E*K, N] fp8                           */      \
    /* Tile: [W_DOWN_TILE, N]                                          */      \
    res = tma_create_tensor_map_2d(                                            \
        &tma_descs.w_down, expert_weights_down_ptr,                            \
        CU_TENSOR_MAP_DATA_TYPE_UINT8,                                         \
        /*rows=*/dims::NUM_EXPERTS * dims::HIDDEN_STATES, /*cols=*/dims::N,    \
        /*smem_box_rows=*/CoreDims::W_DOWN_TILE, /*smem_box_cols=*/dims::N,    \
        CU_TENSOR_MAP_SWIZZLE_NONE);                                           \
    TORCH_CHECK(res == CUDA_SUCCESS,                                           \
                "Failed to create TMA descriptor for w_down: ", res);          \
                                                                               \
    /* Down-projection scales: [E*K] fp32, 1D                          */      \
    {                                                                          \
      uint64_t global_dim[1] = {                                               \
          (uint64_t)(dims::NUM_EXPERTS * dims::HIDDEN_STATES)};                \
      uint32_t box_dim[1] = {CoreDims::W_DOWN_TILE};                           \
      uint32_t elem_stride[1] = {1};                                           \
      res = cuTensorMapEncodeTiled(                                            \
          &tma_descs.s_down, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,                  \
          /*tensorRank=*/1,                                                    \
          const_cast<void*>((const void*)expert_scales_down_ptr), global_dim,  \
          /*globalStrides=*/nullptr, box_dim, elem_stride,                     \
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,           \
          CU_TENSOR_MAP_L2_PROMOTION_L2_128B,                                  \
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);                                  \
      TORCH_CHECK(res == CUDA_SUCCESS,                                         \
                  "Failed to create TMA descriptor for s_down: ", res);        \
    }                                                                          \
                                                                               \
    /* Quantized activations: [BS, K] fp8                              */      \
    /* Tile: [A_TILE, K/2]                                             */      \
    res = tma_create_tensor_map_2d(                                            \
        &tma_descs.a_up,                                                       \
        scratchpad_ptr, /* activations at start of scratchpad */               \
        CU_TENSOR_MAP_DATA_TYPE_UINT8, /*rows=*/(uint64_t)dims::BS,            \
        /*cols=*/(uint64_t)dims::HIDDEN_STATES,                                \
        /*smem_box_rows=*/CoreDims::A_TILE,                                    \
        /*smem_box_cols=*/dims::HIDDEN_STATES / 2, CU_TENSOR_MAP_SWIZZLE_32B); \
    TORCH_CHECK(res == CUDA_SUCCESS,                                           \
                "Failed to create TMA descriptor for a_up: ", res);            \
                                                                               \
    /* Temp fp32 buffer: [TEMP_ROWS, N] fp32                           */      \
    /* Tile: [T_TILE, N]                                               */      \
    {                                                                          \
      using Spec = MoEGemmSpec<dims>;                                          \
      const char* temp_ptr = scratchpad_ptr + offsetof(Spec, temp_fp32);       \
      res = tma_create_tensor_map_2d(                                          \
          &tma_descs.t_down, temp_ptr, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,        \
          /*rows=*/(uint64_t)Spec::TEMP_ROWS, /*cols=*/(uint64_t)dims::N,      \
          /*smem_box_rows=*/CoreDims::T_TILE, /*smem_box_cols=*/dims::N,       \
          CU_TENSOR_MAP_SWIZZLE_NONE);                                         \
      TORCH_CHECK(res == CUDA_SUCCESS,                                         \
                  "Failed to create TMA descriptor for t_down: ", res);        \
    }                                                                          \
                                                                               \
    /* BS8 original bf16 activations: [BS, K] bf16                     */      \
    /* Tile: [1, K] (one token at a time)                              */      \
    res = tma_create_tensor_map_2d(                                            \
        &tma_descs.a_orig, activations_in_ptr,                                 \
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, /*rows=*/(uint64_t)dims::BS,         \
        /*cols=*/(uint64_t)dims::HIDDEN_STATES, /*smem_box_rows=*/1,           \
        /*smem_box_cols=*/dims::HIDDEN_STATES, CU_TENSOR_MAP_SWIZZLE_NONE);    \
    TORCH_CHECK(res == CUDA_SUCCESS,                                           \
                "Failed to create TMA descriptor for a_orig: ", res);          \
                                                                               \
    /* BS8 temp bf16 intermediates: [TEMP_ROWS, N] bf16                */      \
    /* Tile: [1, N] (one row at a time)                                */      \
    {                                                                          \
      using Spec = MoEGemmSpec<dims>;                                          \
      const char* temp_bf16_ptr = scratchpad_ptr + offsetof(Spec, temp_fp32);  \
      res = tma_create_tensor_map_2d(                                          \
          &tma_descs.t_bf16, temp_bf16_ptr, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,  \
          /*rows=*/(uint64_t)Spec::TEMP_ROWS, /*cols=*/(uint64_t)dims::N,      \
          /*smem_box_rows=*/1, /*smem_box_cols=*/dims::N,                      \
          CU_TENSOR_MAP_SWIZZLE_NONE);                                         \
      TORCH_CHECK(res == CUDA_SUCCESS,                                         \
                  "Failed to create TMA descriptor for t_bf16: ", res);        \
    }                                                                          \
                                                                               \
    /* ── Launch TMA kernel ─────────────────────────────────────────── */     \
    void* kernel_args[] = {(void*)&activations_in_ptr,                         \
                           (void*)&num_tokens,                                 \
                           (void*)&router_logits_ptr,                          \
                           (void*)&expert_weights_up_ptr,                      \
                           (void*)&expert_scales_up_ptr,                       \
                           (void*)&expert_weights_down_ptr,                    \
                           (void*)&expert_scales_down_ptr,                     \
                           (void*)&activations_out_ptr,                        \
                           (void*)&scratchpad_ptr,                             \
                           (void*)&scratchpad_size,                            \
                           (void*)&shmem_size,                                 \
                           (void*)&top_k_u32,                                  \
                           (void*)&sf,                                         \
                           (void*)&renormalize,                                \
                           (void*)&tma_descs};                                 \
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();              \
    CUDA_CHECK(cudaFuncSetAttribute(                                           \
        moe_kernel_topk_tma<dims>,                                             \
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));             \
    CUDA_CHECK(cudaLaunchCooperativeKernel(                                    \
        moe_kernel_topk_tma<dims>, dims::KernelConfig::GRID_SIZE,              \
        dims::KernelConfig::BLOCK_SIZE, kernel_args, shmem_size, stream));     \
  }

// TMA variants for Qwen3-Coder-30B-A3B (E=128, K=2048, N=768, TP=1)
MOEMONOKERNEL_TOPK_TMA_WRAPPER_IMPLEMENTATION(
    moe_monokernel_topk_tma_BS8_E128_Qwen3Coder_impl,
    moe_monokernel::Dims_BS8_E128_Qwen3Coder)
MOEMONOKERNEL_TOPK_TMA_WRAPPER_IMPLEMENTATION(
    moe_monokernel_topk_tma_BS64_E128_Qwen3Coder_impl,
    moe_monokernel::Dims_BS64_E128_Qwen3Coder)
