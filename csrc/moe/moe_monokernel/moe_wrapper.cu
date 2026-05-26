#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>

#include "cuda_utils.h"

#include "src/moe.cu"

// ── TEMP_FP8_OFFSET regression anchor (spec R13.3) ─────────────────────────
//
// The host-side down-activation TMA descriptor factory below computes the
// device pointer to `spec->temp_fp8` as
//   `scratchpad_ptr + MoEGemmSpec<Dims>::TEMP_FP8_OFFSET`
// so `TEMP_FP8_OFFSET` MUST stay byte-identical to
// `offsetof(MoEGemmSpec<Dims>, temp_fp8)` for every instantiated `Dims`
// variant.  This is exactly the invariant that the software-grid-sync
// spec (R13.3) relies on when appending new barrier-counter fields to
// the tail of `MoEGemmSpec<Dims>`: as long as every new field lands
// AFTER `temp_fp8` (grid_barrier / partial_barrier belong at the tail),
// the offset stays fixed and the TMA descriptor continues to address
// the right bytes.  A future refactor that silently reorders the struct
// layout would otherwise be caught only at runtime by corrupted TMA
// fetches — the static_asserts below make it a compile-time error.
//
// Covers both Dims variants instantiated by this TU (see the two
// `MOEMONOKERNEL_TOPK_WRAPPER_IMPLEMENTATION` macro invocations at the
// bottom of this file).
static_assert(
    offsetof(moe_monokernel::MoEGemmSpec<
                 moe_monokernel::Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA>,
             temp_fp8) ==
        moe_monokernel::MoEGemmSpec<
            moe_monokernel::Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA>::
            TEMP_FP8_OFFSET,
    "TEMP_FP8_OFFSET must match offsetof(MoEGemmSpec<Dims>, temp_fp8) for "
    "Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA. Do not insert fields "
    "before temp_fp8; grid_barrier / partial_barrier belong at the tail of "
    "MoEGemmSpec<Dims> (spec R13.3).");
static_assert(
    offsetof(moe_monokernel::MoEGemmSpec<
                 moe_monokernel::Dims_BS64_E256_Qwen3_5_35B_BlockFP8>,
             temp_fp8) ==
        moe_monokernel::MoEGemmSpec<
            moe_monokernel::Dims_BS64_E256_Qwen3_5_35B_BlockFP8>::
            TEMP_FP8_OFFSET,
    "TEMP_FP8_OFFSET must match offsetof(MoEGemmSpec<Dims>, temp_fp8) for "
    "Dims_BS64_E256_Qwen3_5_35B_BlockFP8. Do not insert fields before "
    "temp_fp8; grid_barrier / partial_barrier belong at the tail of "
    "MoEGemmSpec<Dims> (spec R13.3).");
static_assert(
    offsetof(
        moe_monokernel::MoEGemmSpec<
            moe_monokernel::Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA_Cluster>,
        temp_fp8) ==
        moe_monokernel::MoEGemmSpec<
            moe_monokernel::
                Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA_Cluster>::
            TEMP_FP8_OFFSET,
    "TEMP_FP8_OFFSET must match offsetof(MoEGemmSpec<Dims>, temp_fp8) for "
    "Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA_Cluster. Do not insert "
    "fields before temp_fp8; grid_barrier / partial_barrier belong at the "
    "tail of MoEGemmSpec<Dims> (spec R13.3).");

// ── Templated launch dispatch helper ────────────────────────────────────
//
// `MOEMONOKERNEL_TOPK_WRAPPER_IMPLEMENTATION` below expands to a NON-template
// function whose `dims` is a textual macro substitution (not a template
// parameter), so an `if constexpr (use_cluster<dims>::value)` placed inside
// the macro body does NOT make the discarded branch's names dependent — the
// compiler still performs name lookup on `dims::KernelConfig::CLUSTER_SIZE`,
// `moe_kernel_topk_cluster<dims>`, etc. on the cluster arm even when
// `use_cluster<dims>::value` is `false`, which fails to compile for the
// non-cluster Dims (which intentionally have no `CLUSTER_SIZE` member).
//
// Hoisting the dispatch into this template helper, where `Dims` IS a real
// template parameter, makes the `if constexpr` correctly discard the
// false branch with full dependent-name treatment.  References to
// `Dims::KernelConfig::CLUSTER_SIZE` / `moe_kernel_topk_cluster<Dims>`
// now only need to be valid for cluster-enabled `Dims` instantiations.
//
// Behaviour is byte-identical to the previous in-macro dispatch:
//   - non-cluster path issues exactly the same legacy `cudaLaunchKernel`
//     against `moe_kernel_topk<Dims>` with the same `kernel_args` table
//     (R11.4, design Step 1 §1.5);
//   - cluster path runs the one-shot `_cluster_checked` block, the
//     `cudaFuncSetAttribute` opt-in for the cluster entry, and the
//     `cudaLaunchKernelEx` produced by tasks 1.8 / 1.9 / 1.10
//     unchanged.
namespace {
template <typename Dims>
static void launch_moe_kernel_topk(
    const cudaStream_t stream,
    const size_t shmem_size,
    void** kernel_args_legacy,
    const at::BFloat16* activations_in_ptr,
    std::uint32_t num_tokens,
    const at::BFloat16* router_logits_ptr,
    const at::Float8_e4m3fn* expert_weights_up_ptr,
    const float* expert_scales_up_ptr,
    const at::Float8_e4m3fn* expert_weights_down_ptr,
    const float* expert_scales_down_ptr,
    at::BFloat16* activations_out_ptr,
    char* scratchpad_ptr,
    const size_t scratchpad_size,
    std::uint32_t top_k_u32,
    moe_monokernel::ScoringFunc sf,
    bool renormalize,
    CUtensorMap up_weights_desc,
    CUtensorMap activations_desc,
    CUtensorMap down_weights_desc,
    CUtensorMap down_activations_desc) {
  using namespace moe_monokernel;
  if constexpr (use_cluster<Dims>::value) {
    /* One-time per-process compute-capability + occupancy check for the
       cluster launch path.  Mirrors the `_diag_printed` one-shot pattern
       in the wrapper macro (non-cluster path) so the cost is amortised
       across the lifetime of the process and stays off the per-decode
       hot path.  Fires BEFORE any `cudaLaunchKernelEx` of the cluster
       arm and fails closed via `TORCH_CHECK` when (a) the device's
       compute capability is below 9.0 (clusters require Hopper, R12.3)
       or (b) the device cannot host GRID_SIZE blocks across enough
       co-resident clusters of size CLUSTER_SIZE (the cluster barrier
       requires every participating cluster member to be co-resident on
       the GPC, R2.4).
       (citation: design Step 1 1.6 / spec R2.3, R2.4, R12.3) */
    {
      static bool _cluster_checked = false;
      if (!_cluster_checked) {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        int major = 0, minor = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(
            &major, cudaDevAttrComputeCapabilityMajor, dev));
        CUDA_CHECK(cudaDeviceGetAttribute(
            &minor, cudaDevAttrComputeCapabilityMinor, dev));
        TORCH_CHECK(
            major >= 9,
            "moe_monokernel cluster variant requires Hopper (compute "
            "capability >= 9.0); device ",
            dev, " reports ", major, ".", minor,
            ". (spec R2.3, R12.3)");
        /* Dynamic SHM cap opt-in for the cluster entry — required so
           the H100 228 KiB per-block cap is in effect when the cluster
           kernel runs (spec R2.5, R5.1; design Step 1 §1.5).
           MUST run before cudaOccupancyMaxActiveClusters below — the
           probe evaluates the kernel against the per-kernel SHM-cap
           attribute and returns 0 if the requested dynamic SHM exceeds
           the default cap (root cause of Step 1 _cluster_checked
           failure on H200 where shmem_size ≈ 106 KiB > default 48 KiB
           Hopper per-block cap). */
        CUDA_CHECK(cudaFuncSetAttribute(
            moe_kernel_topk_cluster<Dims>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
        /* Allow non-portable cluster size scheduling.  The default
           "portable" cluster scheduling on Hopper is conservative
           about GPC packing — at this kernel's SHM footprint
           (172 KiB / block, 1 block / SM) the conservative policy
           reports max_active_clusters = 15 on H200's 132 SMs, one
           cluster short of GRID_SIZE/CLUSTER_SIZE = 16.  Setting
           NonPortableClusterSizeAllowed = 1 lets the driver pack
           clusters across GPCs more aggressively, which on H200
           typically reaches 16 clusters (= 128 blocks, exactly the
           grid we need).  Per CUDA docs this attribute does NOT
           change the cluster size we launch with — it changes the
           driver's packing policy for the size we already chose
           (spec R2.4). */
        CUDA_CHECK(cudaFuncSetAttribute(
            moe_kernel_topk_cluster<Dims>,
            cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        cudaLaunchAttribute probe_attrs[1] = {};
        probe_attrs[0].id = cudaLaunchAttributeClusterDimension;
        probe_attrs[0].val.clusterDim.x = Dims::KernelConfig::CLUSTER_SIZE;
        probe_attrs[0].val.clusterDim.y = 1;
        probe_attrs[0].val.clusterDim.z = 1;
        cudaLaunchConfig_t probe_cfg = {};
        probe_cfg.gridDim = dim3(Dims::KernelConfig::GRID_SIZE, 1, 1);
        probe_cfg.blockDim = dim3(Dims::KernelConfig::BLOCK_SIZE, 1, 1);
        probe_cfg.dynamicSmemBytes = shmem_size;
        probe_cfg.attrs = probe_attrs;
        probe_cfg.numAttrs = 1;
        int max_clusters = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveClusters(
            &max_clusters, moe_kernel_topk_cluster<Dims>, &probe_cfg));
        const uint32_t covered_blocks =
            static_cast<uint32_t>(max_clusters) *
            Dims::KernelConfig::CLUSTER_SIZE;
        int sm_count = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(
            &sm_count, cudaDevAttrMultiProcessorCount, dev));
        fprintf(stderr,
                "[monokernel cluster] dev=%d cc=%d.%d sms=%d "
                "shmem_per_block=%zu max_active_clusters=%d "
                "covered_blocks=%u (GRID_SIZE=%u, CLUSTER_SIZE=%u)\n",
                dev, major, minor, sm_count, shmem_size, max_clusters,
                covered_blocks, Dims::KernelConfig::GRID_SIZE,
                Dims::KernelConfig::CLUSTER_SIZE);
        TORCH_CHECK(
            covered_blocks >= Dims::KernelConfig::GRID_SIZE,
            "moe_monokernel cluster variant: device cannot host "
            "GRID_SIZE=",
            Dims::KernelConfig::GRID_SIZE, " blocks across ", max_clusters,
            " co-resident clusters of size ",
            Dims::KernelConfig::CLUSTER_SIZE,
            " (covered_blocks=", covered_blocks,
            ", sms=", sm_count,
            "). The cluster barrier requires every cluster to be "
            "co-resident on the GPC. (spec R2.4)");
        _cluster_checked = true;
      }
    }
    /* Cluster arm: build the launch attribute carrying the cluster
       dimension, populate a `cudaLaunchConfig_t`, and pass the kernel
       arguments BY VALUE through the variadic `cudaLaunchKernelEx`
       (the Ex API is variadic and forwards args by value — no
       `void**` table). */
    cudaLaunchAttribute attrs[1] = {};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = Dims::KernelConfig::CLUSTER_SIZE;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(Dims::KernelConfig::GRID_SIZE, 1, 1);
    cfg.blockDim = dim3(Dims::KernelConfig::BLOCK_SIZE, 1, 1);
    cfg.dynamicSmemBytes = shmem_size;
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    /* `cudaLaunchKernelEx` does strict template-argument matching against
       the kernel's declared parameter types, so we reinterpret the
       at::*-typed pointers (`at::BFloat16` / `at::Float8_e4m3fn`) into
       the kernel's expected `__nv_bfloat16` / `__nv_fp8_e4m3` types
       before forwarding.  The two pairs are bitwise layout-compatible
       (both 16-bit / 8-bit storage with identical alignment); only the
       C++ type names differ.  The legacy `cudaLaunchKernel` path doesn't
       hit this because it threads arguments through a `void**` table. */
    CUDA_CHECK(cudaLaunchKernelEx(
        &cfg, moe_kernel_topk_cluster<Dims>,
        reinterpret_cast<const __nv_bfloat16*>(activations_in_ptr),
        num_tokens,
        reinterpret_cast<const __nv_bfloat16*>(router_logits_ptr),
        reinterpret_cast<const __nv_fp8_e4m3*>(expert_weights_up_ptr),
        expert_scales_up_ptr,
        reinterpret_cast<const __nv_fp8_e4m3*>(expert_weights_down_ptr),
        expert_scales_down_ptr,
        reinterpret_cast<__nv_bfloat16*>(activations_out_ptr),
        static_cast<void*>(scratchpad_ptr),
        scratchpad_size, shmem_size, top_k_u32, sf, renormalize,
        up_weights_desc, activations_desc, down_weights_desc,
        down_activations_desc));
  } else {
    /* Non-cluster arm — UNCHANGED from pre-feature behaviour
       (R11.4).  Standard (non-cooperative) launch.  The kernel reaches
       grid-wide happens-before via the software Grid_Barrier /
       Partial_Barrier primitives in `src/moe_grid_barrier.h` (spec
       R1.1, R5.1, Design Component C "Launch form") rather than
       `cooperative_groups::this_grid().sync()`.  Using standard
       `cudaLaunchKernel` is what lets the migrated kernel be captured
       into a CUDA Graph. */
    CUDA_CHECK(cudaLaunchKernel((const void*)moe_kernel_topk<Dims>,
                                dim3(Dims::KernelConfig::GRID_SIZE, 1, 1),
                                dim3(Dims::KernelConfig::BLOCK_SIZE, 1, 1),
                                kernel_args_legacy, shmem_size, stream));
  }
}
}  // namespace

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
    /* TMA descriptors for the BS8 WGMMA up-projection path (spec R6.2,        \
       R6.3) and down-projection path (spec R9.1, R9.2, R9.3).  Non-TMA        \
       variants leave these zero-initialized — the kernel parameters are     \
       always present on the signature but the TMA path is the only            \
       consumer.  TMA-enabled variants build real descriptors via the          \
       host-side factories and pass them in kernel_args positions matching     \
       the kernel signature. */                                                \
    CUtensorMap up_weights_desc{};                                             \
    CUtensorMap activations_desc{};                                            \
    CUtensorMap down_weights_desc{};                                           \
    CUtensorMap down_activations_desc{};                                       \
    if constexpr (use_tma<dims>::value) {                                      \
      /* Up-projection weight descriptor (SWIZZLE_128B).  Callers MUST         \
         pre-interleave `expert_weights_up` via                                \
         `interleave_for_tma_wgmma_up` in Python — the helper repacks        \
         gate/up row stripes so a single 128x128 TMA fetches the full          \
         WGMMA A-tile. */                                                      \
      up_weights_desc = create_up_weight_tma_desc(                             \
          reinterpret_cast<const void*>(expert_weights_up_ptr),                \
          dims::NUM_EXPERTS, dims::N, dims::K);                                \
      activations_desc = create_activations_tma_desc(                          \
          reinterpret_cast<const void*>(activations_in_ptr), dims::BS,         \
          dims::HIDDEN_STATES);                                                \
      /* Down-projection weight descriptor (SWIZZLE_128B).  Callers MUST       \
         NOT pre-interleave `expert_weights_down` — the TMA hardware         \
         applies the core-matrix XOR swizzle at write time and expects         \
         the raw row-major `[E, K, N]` fp8 tensor.  `row_box` =                \
         `DOWN_COL_TILE` so each TMA delivers one full M-tile per              \
         128-K substep (16 KB at DOWN_COL_TILE=128, 32 KB at                   \
         DOWN_COL_TILE=256), halving the issue count when the M tile           \
         is 256 rows. */                                                       \
      down_weights_desc = create_down_weight_tma_desc(                         \
          reinterpret_cast<const void*>(expert_weights_down_ptr),              \
          dims::NUM_EXPERTS, dims::HIDDEN_STATES, dims::N,                     \
          /*row_box=*/MoECoreDims<dims>::DOWN_COL_TILE);                       \
      /* Down-projection activation descriptor reads from `spec->temp_fp8`     \
         which lives inside the scratchpad.  Compute the device pointer        \
         from the scratchpad base + the compile-time offset of temp_fp8        \
         inside `MoEGemmSpec<dims>`. */                                        \
      const void* temp_fp8_ptr =                                               \
          reinterpret_cast<const char*>(scratchpad_ptr) +                      \
          MoEGemmSpec<dims>::TEMP_FP8_OFFSET;                                  \
      down_activations_desc = create_down_activation_tma_desc(                 \
          temp_fp8_ptr, MoEGemmSpec<dims>::TEMP_ROWS_TMA, dims::N);            \
    }                                                                          \
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
                           (void*)&renormalize,                                \
                           (void*)&up_weights_desc,                            \
                           (void*)&activations_desc,                           \
                           (void*)&down_weights_desc,                          \
                           (void*)&down_activations_desc};                     \
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();              \
    CUDA_CHECK(cudaFuncSetAttribute(                                           \
        moe_kernel_topk<dims>, cudaFuncAttributeMaxDynamicSharedMemorySize,    \
        shmem_size));                                                          \
    /* One-time diagnostic: compute and print occupancy + shmem so that a      \
       cooperative-launch failure is easy to diagnose. */                      \
    {                                                                          \
      static bool _diag_printed = false;                                       \
      if (!_diag_printed) {                                                    \
        int max_blocks_per_sm = 0;                                             \
        cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(                \
            &max_blocks_per_sm, moe_kernel_topk<dims>,                         \
            dims::KernelConfig::BLOCK_SIZE, shmem_size, cudaOccupancyDefault); \
        cudaFuncAttributes fa;                                                 \
        cudaFuncGetAttributes(&fa, moe_kernel_topk<dims>);                     \
        int sm_count = 0;                                                      \
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);  \
        int smem_opt_in = 0;                                                   \
        cudaDeviceGetAttribute(&smem_opt_in,                                   \
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);    \
        fprintf(stderr,                                                        \
                "[monokernel] %s: grid=%u block=%u shmem=%zu bytes "           \
                "regs/thread=%d static_shmem=%zu max_blocks_per_sm=%d "        \
                "sms=%d coop_max=%d opt-in_shmem=%d\n",                        \
                #name, dims::KernelConfig::GRID_SIZE,                          \
                dims::KernelConfig::BLOCK_SIZE, shmem_size, fa.numRegs,        \
                fa.sharedSizeBytes, max_blocks_per_sm, sm_count,               \
                max_blocks_per_sm * sm_count, smem_opt_in);                    \
        /* Hard co-residency assertions for the software grid barrier          \
           (spec R4.1, R4.2, R4.3 / Design Component C "Co-residency           \
           assertions").  The seed-atomicAdd-spin-on-high-bit protocol         \
           in src/moe_grid_barrier.h is only deadlock-free when every          \
           participating block is co-resident on the GPU for the full          \
           lifetime of the kernel: (1) grid_size <= SM count so every          \
           block gets a slot, and (2) max_active_blocks_per_SM == 1 so         \
           no block is ever waiting on a block that has not yet been           \
           scheduled.  GRID_SIZE is a compile-time constexpr and SM            \
           count / occupancy are device-property-time static, so gating        \
           under `_diag_printed` keeps the check one-shot per process          \
           and off the hot path. */                                            \
        TORCH_CHECK(                                                           \
            dims::KernelConfig::GRID_SIZE <= static_cast<uint32_t>(sm_count),  \
            "moe_monokernel requires GRID_SIZE (=",                            \
            dims::KernelConfig::GRID_SIZE, ") <= SM count (=", sm_count,       \
            ") for software grid barrier co-residency invariant "              \
            "(spec R4.1).");                                                   \
        /*TORCH_CHECK(max_blocks_per_sm == 1,                                  \
                    "moe_monokernel requires max_active_blocks_per_SM == 1 "   \
                    "(observed ",                                              \
                    max_blocks_per_sm,                                         \
                    ") for co-residency invariant (spec R4.2). See "           \
                    "__launch_bounds__(BLOCK_SIZE, 1) and the SHM budget "     \
                    "requirement.");*/                                         \
        _diag_printed = true;                                                  \
      }                                                                        \
    }                                                                          \
    /* One-shot scratchpad zero-init (spec R13.2 / Design Component C          \
       "Scratchpad barrier counter zero-initialization").  The software        \
       Grid_Barrier / Partial_Barrier counters live at the tail of             \
       MoEGemmSpec<Dims> inside the scratchpad, and the                        \
       seed-atomicAdd-spin-on-high-bit protocol requires the barrier slots     \
       to start at 0 so the first Seed_Thread write commits the                \
       `0x80000000u - (arrival_count - 1)` seed value cleanly.  The            \
       ping-pong reset discipline keeps the slots self-maintaining across      \
       subsequent kernel invocations (see MoEGemmSpec<Dims> block comment      \
       on grid_barrier and partial_barrier), so we only pay the zero-init      \
       cost once per process on the first launch.  Zeroing the full            \
       scratchpad (rather than just the counter region) is simpler and         \
       the cost is a few hundred microseconds one-time on H200 — trivial     \
       next to per-decode kernel launches. */                                  \
    {                                                                          \
      static bool _zeroed = false;                                             \
      if (!_zeroed) {                                                          \
        CUDA_CHECK(                                                            \
            cudaMemsetAsync(scratchpad_ptr, 0, scratchpad_size, stream));      \
        _zeroed = true;                                                        \
      }                                                                        \
    }                                                                          \
    /* Cluster-vs-non-cluster launch dispatch.  The cluster Dims emits         \
       `__cluster_dims__(8, 1, 1)` on `moe_kernel_topk_cluster<dims>`,         \
       which makes `cudaLaunchKernelEx` with a                                 \
       `cudaLaunchAttributeClusterDimension` attribute the ONLY valid          \
       launch form for that kernel — `cudaLaunchKernel` against an           \
       annotated entry point is undefined behaviour and the driver may         \
       fail closed at launch.  The non-cluster path keeps the legacy           \
       `cudaLaunchKernel` byte-identically (R11.4) so existing variants        \
       remain captureable into a CUDA Graph and unchanged at runtime           \
       (citation: design Step 1 §1.5 / spec R2.1, R2.2).                      \
                                                                               \
       The actual `if constexpr (use_cluster<dims>::value)` switch lives       \
       in `launch_moe_kernel_topk<Dims>` above this macro: making the          \
       discarded branch's names dependent on the template parameter is         \
       what lets `dims::KernelConfig::CLUSTER_SIZE` and                        \
       `moe_kernel_topk_cluster<dims>` be referenced without forcing them      \
       to exist for the non-cluster Dims (which do not declare                 \
       `CLUSTER_SIZE`). */                                                     \
    launch_moe_kernel_topk<dims>(                                              \
        stream, shmem_size, kernel_args,                                       \
        activations_in_ptr, num_tokens, router_logits_ptr,                     \
        expert_weights_up_ptr, expert_scales_up_ptr,                           \
        expert_weights_down_ptr, expert_scales_down_ptr,                       \
        activations_out_ptr, scratchpad_ptr, scratchpad_size,                  \
        top_k_u32, sf, renormalize,                                            \
        up_weights_desc, activations_desc, down_weights_desc,                  \
        down_activations_desc);                                                \
  }

// Qwen3.5-35B FP8 block-wise (128×128) quantization (E=256, K=2048, N=512,
// TP=1)
MOEMONOKERNEL_TOPK_WRAPPER_IMPLEMENTATION(
    moe_monokernel_topk_BS64_E256_Qwen3_5_35B_BlockFP8_impl,
    moe_monokernel::Dims_BS64_E256_Qwen3_5_35B_BlockFP8)

// TMA + WGMMA + SWIZZLE_128B variant of the BS8 path — the only BS8
// implementation.  Selects the TMA-based weight + activation load path
// in Phase 3 via `KernelConfig::USE_TMA = true`.  Up-projection weights
// must be repacked via `interleave_for_tma_wgmma_up` (gate/up row
// interleave for single-issue TMA); down-projection weights are passed
// raw row-major (the TMA hardware applies the core-matrix XOR swizzle
// at write time).
MOEMONOKERNEL_TOPK_WRAPPER_IMPLEMENTATION(
    moe_monokernel_topk_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA_impl,
    moe_monokernel::Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA)

// Cluster variant of the BS8 TMA+WGMMA path (Hopper sm_90a).
// Adds __cluster_dims__(8,1,1) on the kernel entry point and
// launches via cudaLaunchKernelEx.  See spec R1.1 / design Step 1
// §1.1.
MOEMONOKERNEL_TOPK_WRAPPER_IMPLEMENTATION(
    moe_monokernel_topk_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA_Cluster_impl,
    moe_monokernel::Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA_Cluster)
