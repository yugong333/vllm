#pragma once

#include <torch/all.h>

void topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output, bool renormalize,
                  std::optional<torch::Tensor> bias);

void topk_sigmoid(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output, bool renormalize,
                  std::optional<torch::Tensor> bias);

void topk_softplus_sqrt(torch::Tensor& topk_weights,
                        torch::Tensor& topk_indices,
                        torch::Tensor& token_expert_indices,
                        torch::Tensor& gating_output, bool renormalize,
                        double routed_scaling_factor,
                        const c10::optional<torch::Tensor>& correction_bias,
                        const c10::optional<torch::Tensor>& input_ids,
                        const c10::optional<torch::Tensor>& tid2eid);

void moe_sum(torch::Tensor& input, torch::Tensor& output);

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad,
                          std::optional<torch::Tensor> maybe_expert_map);

void batched_moe_align_block_size(int64_t max_tokens_per_batch,
                                  int64_t block_size,
                                  torch::Tensor const& expert_num_tokens,
                                  torch::Tensor sorted_ids,
                                  torch::Tensor expert_ids,
                                  torch::Tensor num_tokens_post_pad);

void moe_lora_align_block_size(
    torch::Tensor topk_ids, torch::Tensor token_lora_mapping,
    int64_t num_experts, int64_t block_size, int64_t max_loras,
    int64_t max_num_tokens_padded, int64_t max_num_m_blocks,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad, torch::Tensor adapter_enabled,
    torch::Tensor lora_ids, std::optional<torch::Tensor> maybe_expert_map);
#ifndef USE_ROCM
torch::Tensor moe_wna16_gemm(torch::Tensor input, torch::Tensor output,
                             torch::Tensor b_qweight, torch::Tensor b_scales,
                             std::optional<torch::Tensor> b_qzeros,
                             std::optional<torch::Tensor> topk_weights,
                             torch::Tensor sorted_token_ids,
                             torch::Tensor expert_ids,
                             torch::Tensor num_tokens_post_pad, int64_t top_k,
                             int64_t BLOCK_SIZE_M, int64_t BLOCK_SIZE_N,
                             int64_t BLOCK_SIZE_K, int64_t bit);

std::tuple<torch::Tensor, torch::Tensor> grouped_topk(
    torch::Tensor const& scores, int64_t n_group, int64_t topk_group,
    int64_t topk, bool renormalize, double routed_scaling_factor,
    torch::Tensor const& bias, int64_t scoring_func);
#endif

bool moe_permute_unpermute_supported();

int64_t moe_permute_sort_workspace_size(int64_t num_expanded_rows,
                                        int64_t num_experts);

void shuffle_rows(const torch::Tensor& input_tensor,
                  const torch::Tensor& dst2src_map,
                  torch::Tensor& output_tensor);

#ifndef USE_ROCM
// Top-K monokernel for Qwen3.5-35B FP8 block-wise (128×128) quantization
// (E=256, K=2048, N=512, TP=1)
void moe_monokernel_topk_BS64_E256_Qwen3_5_35B_BlockFP8_impl(
    const torch::Tensor& activations_in, const torch::Tensor& router_logits,
    const torch::Tensor& expert_weights_up,
    const torch::Tensor& expert_scales_up,
    const torch::Tensor& expert_weights_down,
    const torch::Tensor& expert_scales_down, torch::Tensor& activations_out,
    torch::Tensor& scratchpad, int64_t top_k, int64_t scoring_func,
    bool renormalize);
// Pair_Layout V2 of the BS8 TMA + WGMMA path
// (`up-proj-gate-up-pair-layout` spec R9.4).  This is the only BS8 TMA
// implementation: TMA-based weight + activation load, 4-deep weight TMA
// lookahead, deferred up-projection epilogue, and the gate/up pair
// layout (`KernelConfig::USE_PAIR_LAYOUT = true`).
void moe_monokernel_topk_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA_impl(
    const torch::Tensor& activations_in, const torch::Tensor& router_logits,
    const torch::Tensor& expert_weights_up,
    const torch::Tensor& expert_scales_up,
    const torch::Tensor& expert_weights_down,
    const torch::Tensor& expert_scales_down, torch::Tensor& activations_out,
    torch::Tensor& scratchpad, int64_t top_k, int64_t scoring_func,
    bool renormalize);
#endif

#ifndef USE_ROCM
void moe_monokernel_BS8_E16_TP8_impl(
    const torch::Tensor& activations_in,
    const torch::Tensor& router_logits,
    const torch::Tensor& expert_weights_up,
    const torch::Tensor& expert_scales_up,
    const torch::Tensor& expert_weights_down,
    const torch::Tensor& expert_scales_down,
    torch::Tensor& activations_out,
    torch::Tensor& gemmspec);
void moe_monokernel_BS8_E16_TP4_impl(
    const torch::Tensor& activations_in,
    const torch::Tensor& router_logits,
    const torch::Tensor& expert_weights_up,
    const torch::Tensor& expert_scales_up,
    const torch::Tensor& expert_weights_down,
    const torch::Tensor& expert_scales_down,
    torch::Tensor& activations_out,
    torch::Tensor& gemmspec);
void moe_monokernel_BS64_E16_TP8_impl(
    const torch::Tensor& activations_in,
    const torch::Tensor& router_logits,
    const torch::Tensor& expert_weights_up,
    const torch::Tensor& expert_scales_up,
    const torch::Tensor& expert_weights_down,
    const torch::Tensor& expert_scales_down,
    torch::Tensor& activations_out,
    torch::Tensor& gemmspec);
void moe_monokernel_BS64_E16_TP4_impl(
    const torch::Tensor& activations_in,
    const torch::Tensor& router_logits,
    const torch::Tensor& expert_weights_up,
    const torch::Tensor& expert_scales_up,
    const torch::Tensor& expert_weights_down,
    const torch::Tensor& expert_scales_down,
    torch::Tensor& activations_out,
    torch::Tensor& gemmspec);
void moe_monokernel_BS8_E128_TP8_impl(
    const torch::Tensor& activations_in,
    const torch::Tensor& router_logits,
    const torch::Tensor& expert_weights_up,
    const torch::Tensor& expert_scales_up,
    const torch::Tensor& expert_weights_down,
    const torch::Tensor& expert_scales_down,
    torch::Tensor& activations_out,
    torch::Tensor& gemmspec);
void moe_monokernel_BS64_E128_TP8_impl(
    const torch::Tensor& activations_in,
    const torch::Tensor& router_logits,
    const torch::Tensor& expert_weights_up,
    const torch::Tensor& expert_scales_up,
    const torch::Tensor& expert_weights_down,
    const torch::Tensor& expert_scales_down,
    torch::Tensor& activations_out,
    torch::Tensor& gemmspec);
#endif

#ifndef USE_ROCM
// DeepSeek V3 optimized router GEMM kernel for SM90+
// Computes output = mat_a @ mat_b.T where:
//   mat_a: [num_tokens, hidden_dim] in bf16
//   mat_b: [num_experts, hidden_dim] in bf16
//   output: [num_tokens, num_experts] in bf16 or fp32
// Supports num_tokens in [1, 16], num_experts in {256, 384}, hidden_dim = 7168
void dsv3_router_gemm(torch::Tensor& output, const torch::Tensor& mat_a,
                      const torch::Tensor& mat_b);
#endif
