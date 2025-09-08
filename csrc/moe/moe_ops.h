#pragma once

#include <torch/all.h>

void topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output);

void moe_sum(torch::Tensor& input, torch::Tensor& output);

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad);
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
#endif

bool moe_permute_unpermute_supported();

void shuffle_rows(const torch::Tensor& input_tensor,
                  const torch::Tensor& dst2src_map,
                  torch::Tensor& output_tensor);

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
