// Stable-ABI build: this TU is compiled into _moe_C_stable_libtorch with
// TORCH_TARGET_VERSION defined, which forbids full-libtorch headers
// (<torch/all.h>, ATen/*, c10/*).  We use torch::stable::Tensor + the C-shim
// helpers in libtorch_stable/torch_utils.h instead.  The kernel itself takes
// native CUDA pointer types (__nv_bfloat16* / __nv_fp8_e4m3* / float*), so the
// host wrapper just reinterpret_casts the stable Tensors' raw data pointers.
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "libtorch_stable/torch_utils.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "cuda_utils.h"

#include "src/moe.cu"

// The host computes the device pointer to `spec->temp_fp8` as
// `scratchpad_ptr + TEMP_FP8_OFFSET` when building the down-activation TMA
// descriptor, so the offset must match the real field offset for every
// Dims.  New MoEGemmSpec fields must go AFTER temp_fp8 (at the tail) — a
// silent reorder would otherwise only show up as corrupted TMA fetches at
// runtime.
static_assert(
    offsetof(moe_monokernel::MoEGemmSpec<
                 moe_monokernel::Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA>,
             temp_fp8) ==
        moe_monokernel::MoEGemmSpec<
            moe_monokernel::Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA>::
            TEMP_FP8_OFFSET,
    "TEMP_FP8_OFFSET must match offsetof(MoEGemmSpec<Dims>, temp_fp8) for "
    "Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA. Do not insert "
    "fields before temp_fp8; grid_barrier / partial_barrier belong at the "
    "tail of MoEGemmSpec<Dims>.");

/**
 * @brief Templated launcher for moe_kernel_topk<Dims>.
 *
 * Holds the full host-side launch body (TMA descriptor construction,
 * one-shot occupancy diagnostic, scratchpad zero-init, cooperative-launch
 * co-residency check, and `cudaLaunchKernel`).  Every named op and every
 * tunable-config op funnels through this single function so the launch
 * logic exists in exactly one place.  `diag_name` is the label printed by
 * the one-shot `[monokernel]` occupancy diagnostic.
 */
template <typename dims>
void launch_moe_monokernel(
    const torch::stable::Tensor& activations_in,
    const torch::stable::Tensor& router_logits,
    const torch::stable::Tensor& expert_weights_up,
    const torch::stable::Tensor& expert_scales_up,
    const torch::stable::Tensor& expert_weights_down,
    const torch::stable::Tensor& expert_scales_down,
    torch::stable::Tensor& activations_out, torch::stable::Tensor& scratchpad,
    int64_t top_k, int64_t scoring_func, bool renormalize,
    const std::optional<torch::stable::Tensor>& expert_bias,
    double routed_scaling_factor, const char* diag_name);

/**
 * @brief Macro that expands to a named op wrapper forwarding to
 * `launch_moe_monokernel<dims>`.  Kept for the existing per-shape ops so
 * their torch_bindings entries are unchanged.
 */
#define MOEMONOKERNEL_TOPK_WRAPPER_IMPLEMENTATION(name, dims)                 \
  void name(const torch::stable::Tensor& activations_in,                      \
            const torch::stable::Tensor& router_logits,                       \
            const torch::stable::Tensor& expert_weights_up,                   \
            const torch::stable::Tensor& expert_scales_up,                    \
            const torch::stable::Tensor& expert_weights_down,                 \
            const torch::stable::Tensor& expert_scales_down,                  \
            torch::stable::Tensor& activations_out,                           \
            torch::stable::Tensor& scratchpad, int64_t top_k,                 \
            int64_t scoring_func, bool renormalize,                           \
            const std::optional<torch::stable::Tensor>& expert_bias,          \
            double routed_scaling_factor) {                                   \
    moe_monokernel::launch_moe_monokernel<dims>(                              \
        activations_in, router_logits, expert_weights_up, expert_scales_up,   \
        expert_weights_down, expert_scales_down, activations_out, scratchpad, \
        top_k, scoring_func, renormalize, expert_bias, routed_scaling_factor, \
        #name);                                                               \
  }

namespace moe_monokernel {
template <typename dims>
void launch_moe_monokernel(
    const torch::stable::Tensor& activations_in,
    const torch::stable::Tensor& router_logits,
    const torch::stable::Tensor& expert_weights_up,
    const torch::stable::Tensor& expert_scales_up,
    const torch::stable::Tensor& expert_weights_down,
    const torch::stable::Tensor& expert_scales_down,
    torch::stable::Tensor& activations_out, torch::stable::Tensor& scratchpad,
    int64_t top_k, int64_t scoring_func, bool renormalize,
    const std::optional<torch::stable::Tensor>& expert_bias,
    double routed_scaling_factor, const char* diag_name) {
  // Device residency is guaranteed by the CUDA dispatch key
  // (STABLE_TORCH_LIBRARY_IMPL(..., CUDA, ...)); the stable Tensor API has no
  // is_cuda(), so we only validate the scalar arguments here.
  STD_TORCH_CHECK(top_k >= 1 && top_k <= 8, "top_k must be between 1 and 8.");
  STD_TORCH_CHECK(scoring_func == 0 || scoring_func == 1,
                  "scoring_func must be 0 (sigmoid) or 1 (softmax).");

  // The kernel takes native CUDA pointer types; reinterpret the stable
  // Tensors' raw data pointers (data_ptr() is const void*, mutable_data_ptr()
  // is void*).  Element layout is identical to the ATen dtypes the caller
  // allocates (bfloat16 / float8_e4m3fn / float32).
  const auto* activations_in_ptr =
      reinterpret_cast<const __nv_bfloat16*>(activations_in.data_ptr());
  const auto* router_logits_ptr =
      reinterpret_cast<const __nv_bfloat16*>(router_logits.data_ptr());
  const auto* expert_weights_up_ptr =
      reinterpret_cast<const __nv_fp8_e4m3*>(expert_weights_up.data_ptr());
  const auto* expert_scales_up_ptr =
      reinterpret_cast<const float*>(expert_scales_up.data_ptr());
  const auto* expert_weights_down_ptr =
      reinterpret_cast<const __nv_fp8_e4m3*>(expert_weights_down.data_ptr());
  const auto* expert_scales_down_ptr =
      reinterpret_cast<const float*>(expert_scales_down.data_ptr());
  auto* activations_out_ptr =
      reinterpret_cast<__nv_bfloat16*>(activations_out.mutable_data_ptr());
  char* scratchpad_ptr = reinterpret_cast<char*>(scratchpad.mutable_data_ptr());
  // Optional per-expert selection bias (GLM `e_score_correction_bias`,
  // float32 [NUM_EXPERTS]).  nullptr => raw-logit ranking (the shipped
  // Qwen path).  When present it must be a float32 contiguous tensor.
  const float* expert_bias_ptr = nullptr;
  if (expert_bias.has_value()) {
    STD_TORCH_CHECK(
        expert_bias->scalar_type() == torch::headeronly::ScalarType::Float,
        "expert_bias must be float32.");
    expert_bias_ptr = reinterpret_cast<const float*>(expert_bias->data_ptr());
  }

  using namespace moe_monokernel;
  const uint32_t num_tokens = activations_in.size(0);
  const size_t shmem_size = get_moe_shmem_size<dims>();
  const size_t scratchpad_size =
      static_cast<size_t>(scratchpad.numel()) * scratchpad.element_size();
  const uint32_t top_k_u32 = static_cast<uint32_t>(top_k);
  const ScoringFunc sf = static_cast<ScoringFunc>(scoring_func);
  const float routed_scaling_factor_f =
      static_cast<float>(routed_scaling_factor);

  /* TMA descriptors (see src/moe_tma.h).  Non-TMA variants leave these
     zero-initialized — the kernel parameters are always on the
     signature but only the TMA path reads them. */
  CUtensorMap up_weights_desc{};
  CUtensorMap activations_desc{};
  CUtensorMap down_weights_desc{};
  CUtensorMap down_activations_desc{};
  if constexpr (use_tma<dims>::value) {
    /* Up weights: interleaved-layout configs need the Python gate/up
       pre-interleave (interleave_for_tma_wgmma_up_v2); raw configs read
       the tensor unmodified.  Same descriptor either way. */
    up_weights_desc = create_up_weight_tma_desc(
        reinterpret_cast<const void*>(expert_weights_up_ptr), dims::NUM_EXPERTS,
        dims::N, dims::K);
    /* Row extent = the REAL token count, not dims::BS: the TMA engine
       bounds-checks against globalDim and zero-fills out-of-bounds box
       rows, so for M < 8 the routing-window load reads exactly M rows of
       the caller's tensor and hardware-zeros the phantom rows [M, 8).
       (With dims::BS here the engine considered 8 rows valid and read
       past an M-row allocation.)  Under CUDA graphs the descriptor is
       captured per graph, and vLLM captures one graph per batch size, so
       the baked-in M always matches replays. */
    activations_desc = create_activations_tma_desc(
        reinterpret_cast<const void*>(activations_in_ptr), num_tokens,
        dims::HIDDEN_STATES, /*box_rows=*/dims::BS);
    /* Down weights: raw row-major [E, K, N], never pre-interleaved.
       row_box is pinned to 128 because DOWN_COL_TILE=384 (122B) exceeds
       the 256-row TMA boxDim cap; the kernel issues one 128-row TMA per
       M-atom instead. */
    down_weights_desc = create_down_weight_tma_desc(
        reinterpret_cast<const void*>(expert_weights_down_ptr),
        dims::NUM_EXPERTS, dims::HIDDEN_STATES, dims::N,
        /*row_box=*/128u);
    /* Down activations read spec->temp_fp8 inside the scratchpad;
       address = scratchpad base + compile-time field offset. */
    const void* temp_fp8_ptr = reinterpret_cast<const char*>(scratchpad_ptr) +
                               MoEGemmSpec<dims>::TEMP_FP8_OFFSET;
    down_activations_desc = create_down_activation_tma_desc(
        temp_fp8_ptr, MoEGemmSpec<dims>::TEMP_ROWS_TMA, dims::N,
        /*t_tile=*/moe_monokernel::MoECoreDims<dims>::T_TILE);
  }

  void* kernel_args[] = {(void*)&activations_in_ptr,
                         (void*)&num_tokens,
                         (void*)&router_logits_ptr,
                         (void*)&expert_weights_up_ptr,
                         (void*)&expert_scales_up_ptr,
                         (void*)&expert_weights_down_ptr,
                         (void*)&expert_scales_down_ptr,
                         (void*)&activations_out_ptr,
                         (void*)&scratchpad_ptr,
                         (void*)&scratchpad_size,
                         (void*)&shmem_size,
                         (void*)&top_k_u32,
                         (void*)&sf,
                         (void*)&renormalize,
                         (void*)&expert_bias_ptr,
                         (void*)&routed_scaling_factor_f,
                         (void*)&up_weights_desc,
                         (void*)&activations_desc,
                         (void*)&down_weights_desc,
                         (void*)&down_activations_desc};
  const cudaStream_t stream =
      get_current_cuda_stream(activations_in.get_device_index());
  CUDA_CHECK(cudaFuncSetAttribute(moe_kernel_topk<dims>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  shmem_size));
  /* One-time diagnostic: compute and print occupancy + shmem so that a
     cooperative-launch failure is easy to diagnose. */
  {
    static bool _diag_printed = false;
    if (!_diag_printed) {
      int max_blocks_per_sm = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
          &max_blocks_per_sm, moe_kernel_topk<dims>,
          dims::KernelConfig::BLOCK_SIZE, shmem_size, cudaOccupancyDefault);
      cudaFuncAttributes fa;
      cudaFuncGetAttributes(&fa, moe_kernel_topk<dims>);
      int sm_count = 0;
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
      int smem_opt_in = 0;
      cudaDeviceGetAttribute(&smem_opt_in,
                             cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
      fprintf(stderr,
              "[monokernel] %s: grid=%u block=%u shmem=%zu bytes "
              "regs/thread=%d static_shmem=%zu max_blocks_per_sm=%d "
              "sms=%d coop_max=%d opt-in_shmem=%d\n",
              diag_name, dims::KernelConfig::GRID_SIZE,
              dims::KernelConfig::BLOCK_SIZE, shmem_size, fa.numRegs,
              fa.sharedSizeBytes, max_blocks_per_sm, sm_count,
              max_blocks_per_sm * sm_count, smem_opt_in);
      /* Runtime half of the co-residency invariant required by the
         flag/sentinel handoffs (sites #2 and #3 in src/moe.cu): a
         consumer spin-waits on values a producer block publishes, so
         every block must be scheduled for the kernel's full lifetime,
         which needs GRID_SIZE <= SM count.  One-shot: GRID_SIZE is
         constexpr and the SM count is device-static. */
      STD_TORCH_CHECK(
          dims::KernelConfig::GRID_SIZE <= static_cast<uint32_t>(sm_count),
          "moe_monokernel requires GRID_SIZE (=", dims::KernelConfig::GRID_SIZE,
          ") <= SM count (=", sm_count,
          ") for software grid barrier co-residency invariant.");
      _diag_printed = true;
    }
  }
  /* One-shot scratchpad zero-init: the software barrier counters at the
     tail of MoEGemmSpec must start at 0 (self-maintaining afterwards
     via the seed-exchange discipline).  Zeroing the whole scratchpad is
     simpler than just the counter region and costs a one-time few
     hundred microseconds. */
  {
    static bool _zeroed = false;
    if (!_zeroed) {
      /* The zero fill also establishes the sentinel-handoff invariant:
         an all-zero temp_act_scale buffer reads as "nothing published"
         (0.0f sentinel; see moe_scale_is_sentinel in moe_internal.h).
         Callers that allocate the scratchpad with torch.zeros (the
         serving path — one scratchpad per layer, where this one-shot
         static wouldn't fire per instance) satisfy it the same way. */
      CUDA_CHECK(cudaMemsetAsync(scratchpad_ptr, 0, scratchpad_size, stream));
      _zeroed = true;
    }
  }
  /* Standard (non-cooperative) launch: cross-block ordering comes from
     the flag/sentinel handoffs inside the kernel (sites #2 and #3 in
     src/moe.cu), which is what lets the kernel be captured into a CUDA
     Graph. */
  CUDA_CHECK(cudaLaunchKernel((const void*)moe_kernel_topk<dims>,
                              dim3(dims::KernelConfig::GRID_SIZE, 1, 1),
                              dim3(dims::KernelConfig::BLOCK_SIZE, 1, 1),
                              kernel_args, shmem_size, stream));
}
}  // namespace moe_monokernel

// ─────────────────────────────────────────────────────────────────────────
// GENERATED per-shape instantiations + tunable-config dispatch
// ─────────────────────────────────────────────────────────────────────────
//
// Emitted from csrc/moe/moe_monokernel/shapes.json by tools/gen_shapes.py:
//   * config_table_generated.inc — MONO_CONFIGS_<shape>(X) X-macro tables
//     (id, GRID, DCT, KUP, KDN, SLOTS); id 0 == shipped default per shape.
//   * wrapper_generated.inc — MOEMONOKERNEL_TOPK_WRAPPER_IMPLEMENTATION for
//     each shape's named op, the `dispatch_tunable_<shape>` helpers (inside
//     namespace moe_monokernel), the `*_tunable_impl` free functions, and the
//     legacy-name (Qwen3_5_35B / _122B) forwarder ops for back-compat.
//
// The config tables are plain preprocessor macros and must be defined before
// the wrapper expands them, so config_table is included first.  To add/retune
// a shape or config: edit shapes.json, run gen_shapes.py, rebuild — the
// dispatcher and Python selection pick up new ids/shapes automatically.
#include "generated/config_table_generated.inc"
#include "generated/wrapper_generated.inc"

// The monokernel *_impl functions take torch::stable::Tensor (stable-ABI
// port), so they are TORCH_BOX'd into the _moe_C stable dispatcher here, next
// to where the X-macro / generated wrapper defines them.  Their m.def schemas
// live in csrc/libtorch_stable/moe/torch_bindings.cpp (defs_generated.inc) —
// the same def-in-bindings / impl-in-source split that dsv3_router_gemm uses.
#ifndef USE_ROCM
STABLE_TORCH_LIBRARY_IMPL(_moe_C, CUDA, m) {
  #include "generated/impls_generated.inc"
}
#endif
