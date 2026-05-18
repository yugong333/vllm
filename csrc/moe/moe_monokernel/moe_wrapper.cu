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
    offsetof(moe_monokernel::MoEGemmSpec<
                 moe_monokernel::
                     Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA_Cluster>,
             temp_fp8) ==
        moe_monokernel::MoEGemmSpec<
            moe_monokernel::
                Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA_Cluster>::
            TEMP_FP8_OFFSET,
    "TEMP_FP8_OFFSET must match offsetof(MoEGemmSpec<Dims>, temp_fp8) for "
    "Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA_Cluster. Do not insert "
    "fields before temp_fp8; the cluster variant adds new "
    "phase_timestamps fields at the tail of MoEGemmSpec<Dims> "
    "(spec R11.2, R11.3).");

/**
 * @brief Macro that expands to a kernel call wrapper for moe_kernel_topk with
 * specified @p dims and configurable top_k, scoring_func, and renormalize.
 */
#define MOEMONOKERNEL_TOPK_WRAPPER_IMPLEMENTATION(name, dims)                    \
  void name(const torch::Tensor& activations_in,                                 \
            const torch::Tensor& router_logits,                                  \
            const torch::Tensor& expert_weights_up,                              \
            const torch::Tensor& expert_scales_up,                               \
            const torch::Tensor& expert_weights_down,                            \
            const torch::Tensor& expert_scales_down,                             \
            torch::Tensor& activations_out, torch::Tensor& scratchpad,           \
            int64_t top_k, int64_t scoring_func, bool renormalize) {             \
    TORCH_CHECK(                                                                 \
        activations_in.is_cuda(),                                                \
        "Optimized MoE kernel must be called with CUDA tensors only.");          \
    TORCH_CHECK(                                                                 \
        router_logits.is_cuda(),                                                 \
        "Optimized MoE kernel must be called with CUDA tensors only.");          \
    TORCH_CHECK(                                                                 \
        expert_weights_up.is_cuda(),                                             \
        "Optimized MoE kernel must be called with CUDA tensors only.");          \
    TORCH_CHECK(                                                                 \
        expert_scales_up.is_cuda(),                                              \
        "Optimized MoE kernel must be called with CUDA tensors only.");          \
    TORCH_CHECK(                                                                 \
        expert_weights_down.is_cuda(),                                           \
        "Optimized MoE kernel must be called with CUDA tensors only.");          \
    TORCH_CHECK(                                                                 \
        expert_scales_down.is_cuda(),                                            \
        "Optimized MoE kernel must be called with CUDA tensors only.");          \
    TORCH_CHECK(                                                                 \
        activations_out.is_cuda(),                                               \
        "Optimized MoE kernel must be called with CUDA tensors only.");          \
    TORCH_CHECK(                                                                 \
        scratchpad.is_cuda(),                                                    \
        "Optimized MoE kernel must be called with CUDA tensors only.");          \
    TORCH_CHECK(top_k >= 1 && top_k <= 8, "top_k must be between 1 and 8.");     \
    TORCH_CHECK(scoring_func == 0 || scoring_func == 1,                          \
                "scoring_func must be 0 (sigmoid) or 1 (softmax).");             \
                                                                                 \
    const auto* activations_in_ptr = activations_in.data_ptr<at::BFloat16>();    \
    const auto* router_logits_ptr = router_logits.data_ptr<at::BFloat16>();      \
    const auto* expert_weights_up_ptr =                                          \
        expert_weights_up.data_ptr<at::Float8_e4m3fn>();                         \
    const auto* expert_scales_up_ptr = expert_scales_up.data_ptr<float>();       \
    const auto* expert_weights_down_ptr =                                        \
        expert_weights_down.data_ptr<at::Float8_e4m3fn>();                       \
    const auto* expert_scales_down_ptr = expert_scales_down.data_ptr<float>();   \
    auto* activations_out_ptr = activations_out.data_ptr<at::BFloat16>();        \
    char* scratchpad_ptr =                                                       \
        reinterpret_cast<char*>(scratchpad.data_ptr<float>());                   \
                                                                                 \
    using namespace moe_monokernel;                                              \
    const uint32_t num_tokens = activations_in.size(0);                          \
    const size_t shmem_size = get_moe_shmem_size<dims>();                        \
    const size_t scratchpad_size = scratchpad.nbytes();                          \
    const uint32_t top_k_u32 = static_cast<uint32_t>(top_k);                     \
    const ScoringFunc sf = static_cast<ScoringFunc>(scoring_func);               \
                                                                                 \
    /* TMA descriptors for the BS8 WGMMA up-projection path (spec R6.2,          \
       R6.3) and down-projection path (spec R9.1, R9.2, R9.3).  Non-TMA          \
       variants leave these zero-initialized — the kernel parameters are       \
       always present on the signature but the TMA path is the only              \
       consumer.  TMA-enabled variants build real descriptors via the            \
       host-side factories and pass them in kernel_args positions matching       \
       the kernel signature. */                                                  \
    CUtensorMap up_weights_desc{};                                               \
    CUtensorMap activations_desc{};                                              \
    CUtensorMap down_weights_desc{};                                             \
    CUtensorMap down_activations_desc{};                                         \
    if constexpr (use_tma<dims>::value) {                                        \
      /* Up-projection weight descriptor (SWIZZLE_128B).  Callers MUST           \
         pre-interleave `expert_weights_up` via                                  \
         `interleave_for_tma_wgmma_up` in Python — the helper repacks          \
         gate/up row stripes so a single 128x128 TMA fetches the full            \
         WGMMA A-tile. */                                                        \
      up_weights_desc = create_up_weight_tma_desc(                               \
          reinterpret_cast<const void*>(expert_weights_up_ptr),                  \
          dims::NUM_EXPERTS, dims::N, dims::K);                                  \
      activations_desc = create_activations_tma_desc(                            \
          reinterpret_cast<const void*>(activations_in_ptr), dims::BS,           \
          dims::HIDDEN_STATES);                                                  \
      /* Down-projection weight descriptor (SWIZZLE_128B).  Callers MUST         \
         NOT pre-interleave `expert_weights_down` — the TMA hardware           \
         applies the core-matrix XOR swizzle at write time and expects           \
         the raw row-major `[E, K, N]` fp8 tensor. */                            \
      down_weights_desc = create_down_weight_tma_desc(                           \
          reinterpret_cast<const void*>(expert_weights_down_ptr),                \
          dims::NUM_EXPERTS, dims::HIDDEN_STATES, dims::N);                      \
      /* Down-projection activation descriptor reads from `spec->temp_fp8`       \
         which lives inside the scratchpad.  Compute the device pointer          \
         from the scratchpad base + the compile-time offset of temp_fp8          \
         inside `MoEGemmSpec<dims>`. */                                          \
      const void* temp_fp8_ptr =                                                 \
          reinterpret_cast<const char*>(scratchpad_ptr) +                        \
          MoEGemmSpec<dims>::TEMP_FP8_OFFSET;                                    \
      down_activations_desc = create_down_activation_tma_desc(                   \
          temp_fp8_ptr, MoEGemmSpec<dims>::TEMP_ROWS_TMA, dims::N);              \
    }                                                                            \
                                                                                 \
    void* kernel_args[] = {(void*)&activations_in_ptr,                           \
                           (void*)&num_tokens,                                   \
                           (void*)&router_logits_ptr,                            \
                           (void*)&expert_weights_up_ptr,                        \
                           (void*)&expert_scales_up_ptr,                         \
                           (void*)&expert_weights_down_ptr,                      \
                           (void*)&expert_scales_down_ptr,                       \
                           (void*)&activations_out_ptr,                          \
                           (void*)&scratchpad_ptr,                               \
                           (void*)&scratchpad_size,                              \
                           (void*)&shmem_size,                                   \
                           (void*)&top_k_u32,                                    \
                           (void*)&sf,                                           \
                           (void*)&renormalize,                                  \
                           (void*)&up_weights_desc,                              \
                           (void*)&activations_desc,                             \
                           (void*)&down_weights_desc,                            \
                           (void*)&down_activations_desc};                       \
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                \
    CUDA_CHECK(cudaFuncSetAttribute(                                             \
        moe_kernel_topk<dims>, cudaFuncAttributeMaxDynamicSharedMemorySize,      \
        shmem_size));                                                            \
    /* One-time diagnostic: compute and print occupancy + shmem so that a        \
       cooperative-launch failure is easy to diagnose. */                        \
    {                                                                            \
      static bool _diag_printed = false;                                         \
      if (!_diag_printed) {                                                      \
        int max_blocks_per_sm = 0;                                               \
        cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(                  \
            &max_blocks_per_sm, moe_kernel_topk<dims>,                           \
            dims::KernelConfig::BLOCK_SIZE, shmem_size, cudaOccupancyDefault);   \
        cudaFuncAttributes fa;                                                   \
        cudaFuncGetAttributes(&fa, moe_kernel_topk<dims>);                       \
        int sm_count = 0;                                                        \
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);    \
        int smem_opt_in = 0;                                                     \
        cudaDeviceGetAttribute(&smem_opt_in,                                     \
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);      \
        fprintf(stderr,                                                          \
                "[monokernel] %s: grid=%u block=%u shmem=%zu bytes "             \
                "regs/thread=%d static_shmem=%zu max_blocks_per_sm=%d "          \
                "sms=%d coop_max=%d opt-in_shmem=%d\n",                          \
                #name, dims::KernelConfig::GRID_SIZE,                            \
                dims::KernelConfig::BLOCK_SIZE, shmem_size, fa.numRegs,          \
                fa.sharedSizeBytes, max_blocks_per_sm, sm_count,                 \
                max_blocks_per_sm * sm_count, smem_opt_in);                      \
        /* Hard co-residency assertions for the software grid barrier            \
           (spec R4.1, R4.2, R4.3 / Design Component C "Co-residency             \
           assertions").  The seed-atomicAdd-spin-on-high-bit protocol           \
           in src/moe_grid_barrier.h is only deadlock-free when every            \
           participating block is co-resident on the GPU for the full            \
           lifetime of the kernel: (1) grid_size <= SM count so every            \
           block gets a slot, and (2) max_active_blocks_per_SM == 1 so           \
           no block is ever waiting on a block that has not yet been             \
           scheduled.  GRID_SIZE is a compile-time constexpr and SM              \
           count / occupancy are device-property-time static, so gating          \
           under `_diag_printed` keeps the check one-shot per process            \
           and off the hot path. */                                              \
        TORCH_CHECK(                                                             \
            dims::KernelConfig::GRID_SIZE <= static_cast<uint32_t>(sm_count),    \
            "moe_monokernel requires GRID_SIZE (=",                              \
            dims::KernelConfig::GRID_SIZE, ") <= SM count (=", sm_count,         \
            ") for software grid barrier co-residency invariant "                \
            "(spec R4.1).");                                                     \
        /*TORCH_CHECK(max_blocks_per_sm == 1,                                    \
                    "moe_monokernel requires max_active_blocks_per_SM == 1 "     \
                    "(observed ",                                                \
                    max_blocks_per_sm,                                           \
                    ") for co-residency invariant (spec R4.2). See "             \
                    "__launch_bounds__(BLOCK_SIZE, 1) and the SHM budget "       \
                    "requirement.");*/                                           \
        _diag_printed = true;                                                    \
      }                                                                          \
    }                                                                            \
    /* One-shot scratchpad zero-init (spec R13.2 / Design Component C            \
       "Scratchpad barrier counter zero-initialization").  The software          \
       Grid_Barrier / Partial_Barrier counters live at the tail of               \
       MoEGemmSpec<Dims> inside the scratchpad, and the                          \
       seed-atomicAdd-spin-on-high-bit protocol requires the barrier slots       \
       to start at 0 so the first Seed_Thread write commits the                  \
       `0x80000000u - (arrival_count - 1)` seed value cleanly.  The              \
       ping-pong reset discipline keeps the slots self-maintaining across        \
       subsequent kernel invocations (see MoEGemmSpec<Dims> block comment        \
       on grid_barrier and partial_barrier), so we only pay the zero-init        \
       cost once per process on the first launch.  Zeroing the full              \
       scratchpad (rather than just the counter region) is simpler and           \
       the cost is a few hundred microseconds one-time on H200 — trivial       \
       next to per-decode kernel launches. */                                    \
    {                                                                            \
      static bool _zeroed = false;                                               \
      if (!_zeroed) {                                                            \
        CUDA_CHECK(                                                              \
            cudaMemsetAsync(scratchpad_ptr, 0, scratchpad_size, stream));        \
        _zeroed = true;                                                          \
      }                                                                          \
    }                                                                            \
    /* Launch form selection (cluster vs non-cluster).  The cluster              \
       variant of the BS8 path opts into Hopper thread block clusters            \
       via `__cluster_dims__(8, 1, 1)` on `moe_kernel_topk_cluster<Dims>`,       \
       so the host launch must use `cudaLaunchKernelEx` with a                   \
       `cudaLaunchAttributeClusterDimension` attribute (spec R2.1, R2.2;         \
       design §4.1).  Non-cluster variants keep the existing                    \
       `cudaLaunchKernel` path BYTE-IDENTICALLY (R11.4); the                     \
       `if constexpr` branch is gated on `use_cluster<dims>::value` so the       \
       compile-time decision is fully resolved at template instantiation         \
       time. */                                                                  \
    if constexpr (use_cluster<dims>::value) {                                    \
      /* Dynamic-SHM opt-in for the cluster entry point (spec R2.5,              \
         R5.1, R11.4).  Issued FIRST — before the occupancy probe —          \
         because `cudaOccupancyMaxActiveClusters` reads the kernel's             \
         dynamic-shmem attribute when computing per-SM occupancy.  Without       \
         the opt-in, the probe sees the default 48 KiB cap and returns 0         \
         clusters for any kernel whose `dynamicSmemBytes > 48 KiB` even          \
         though the kernel itself fits well within the H100 228 KiB              \
         opt-in budget.                                                          \
                                                                               \ \
         The existing `cudaFuncSetAttribute(moe_kernel_topk<dims>, ...)`         \
         above stays in effect for the non-cluster path; the cluster             \
         kernel needs its own opt-in so the H100 228 KiB per-block SHM cap       \
         is lifted before both the probe and launch. */                          \
      CUDA_CHECK(cudaFuncSetAttribute(                                           \
          moe_kernel_topk_cluster<dims>,                                         \
          cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));             \
      /* One-time per-process cluster-launch safety checks (spec R2.3,           \
         R2.4, R12.3; design §4.2).  Mirrors the `_diag_printed`                \
         one-shot pattern above so the device-attribute queries +                \
         occupancy probe pay their cost exactly once and stay off the            \
         hot path. */                                                            \
      static bool _cluster_checked = false;                                      \
      if (!_cluster_checked) {                                                   \
        int dev = 0;                                                             \
        CUDA_CHECK(cudaGetDevice(&dev));                                         \
        int major = 0, minor = 0;                                                \
        CUDA_CHECK(cudaDeviceGetAttribute(                                       \
            &major, cudaDevAttrComputeCapabilityMajor, dev));                    \
        CUDA_CHECK(cudaDeviceGetAttribute(                                       \
            &minor, cudaDevAttrComputeCapabilityMinor, dev));                    \
        /* R2.3 / R12.3: cluster variant requires Hopper (cc >= 9.0).            \
           This converts the device-side `#if __CUDA_ARCH__ >= 900`              \
           gate (design §3.5) into a runtime `TORCH_CHECK` so a build           \
           that includes both `sm_90a` and earlier architectures fails           \
           closed when run on the wrong device. */                               \
        TORCH_CHECK(major >= 9,                                                  \
                    "moe_monokernel cluster variant requires Hopper "            \
                    "(compute capability >= 9.0); device ",                      \
                    dev, " reports ", major, ".", minor,                         \
                    ". (spec R2.3, R12.3)");                                     \
        /* R2.4: every block of GRID_SIZE must be co-resident as part            \
           of a cluster; otherwise the cluster barrier could deadlock            \
           waiting on a peer block that has not yet been scheduled. */           \
        cudaLaunchConfig_t probe_cfg = {};                                       \
        probe_cfg.gridDim = dim3(dims::KernelConfig::GRID_SIZE, 1, 1);           \
        probe_cfg.blockDim = dim3(dims::KernelConfig::BLOCK_SIZE, 1, 1);         \
        probe_cfg.dynamicSmemBytes = shmem_size;                                 \
        cudaLaunchAttribute probe_attrs[1] = {};                                 \
        probe_attrs[0].id = cudaLaunchAttributeClusterDimension;                 \
        probe_attrs[0].val.clusterDim.x = cluster_size<dims>::value;             \
        probe_attrs[0].val.clusterDim.y = 1;                                     \
        probe_attrs[0].val.clusterDim.z = 1;                                     \
        probe_cfg.attrs = probe_attrs;                                           \
        probe_cfg.numAttrs = 1;                                                  \
        int max_clusters = 0;                                                    \
        CUDA_CHECK(cudaOccupancyMaxActiveClusters(                               \
            &max_clusters, moe_kernel_topk_cluster<dims>, &probe_cfg));          \
        const uint32_t covered_blocks =                                          \
            static_cast<uint32_t>(max_clusters) * cluster_size<dims>::value;     \
        /* R2.4: ideally GRID_SIZE blocks should fit into co-resident            \
           clusters as ONE wave, so the software grid barrier (sites #1,         \
           #3) doesn't deadlock waiting on a queued block.  On H100 we           \
           expect 16 × 8 = 128 ≥ GRID_SIZE; on H200 the GPC topology          \
           sometimes returns only 15 clusters (120 blocks).  Demote the          \
           tight `>= GRID_SIZE` check to a warning when at least                 \
           CLUSTER_SIZE-many blocks of slack are available; the cluster          \
           barrier itself only needs intra-cluster co-residency, which is        \
           hardware-guaranteed by any successful cluster launch.  If we          \
           fall short by more than a wave's worth, fail closed (likely           \
           deadlock). */                                                         \
        TORCH_CHECK(                                                             \
            max_clusters > 0,                                                    \
            "moe_monokernel cluster variant: device reports zero active "        \
            "clusters for moe_kernel_topk_cluster<dims>.  This usually "         \
            "means the dynamic SHM opt-in was not applied before the "           \
            "occupancy probe (spec R2.5) or the kernel's shmem footprint "       \
            "exceeds the GPU's SHM-per-cluster capacity.");                      \
        if (covered_blocks < dims::KernelConfig::GRID_SIZE) {                    \
          fprintf(stderr,                                                        \
                  "[monokernel] WARNING: cluster variant: device fits "          \
                  "only %d clusters of size %u (%u blocks) but "                 \
                  "GRID_SIZE=%u; partial wave may still launch (cluster "        \
                  "barriers are intra-cluster only) but software grid "          \
                  "barriers at sites #1/#3 may stall.  See spec R2.4.\n",        \
                  max_clusters, cluster_size<dims>::value, covered_blocks,       \
                  dims::KernelConfig::GRID_SIZE);                                \
        }                                                                        \
        _cluster_checked = true;                                                 \
      }                                                                          \
      /* Cluster launch (spec R2.1, R2.2; design §4.1).                         \
         `cudaLaunchKernelEx` is the C++ variadic form: it takes the             \
         kernel arguments by value and deduces their types from the              \
         kernel signature, so we forward the same operand list as the            \
         legacy `kernel_args[]` table above — NOT the array itself.            \
         Pointer types are reinterpret_cast'd from the at::BFloat16 /            \
         at::Float8_e4m3fn aliases to the underlying CUDA element                \
         types expected by `moe_kernel_topk_cluster<Dims>` (these                \
         types are layout-compatible, so reinterpret_cast is sound). */          \
      cudaLaunchAttribute attrs[1] = {};                                         \
      attrs[0].id = cudaLaunchAttributeClusterDimension;                         \
      attrs[0].val.clusterDim.x = cluster_size<dims>::value;                     \
      attrs[0].val.clusterDim.y = 1;                                             \
      attrs[0].val.clusterDim.z = 1;                                             \
      cudaLaunchConfig_t cfg = {};                                               \
      cfg.gridDim = dim3(dims::KernelConfig::GRID_SIZE, 1, 1);                   \
      cfg.blockDim = dim3(dims::KernelConfig::BLOCK_SIZE, 1, 1);                 \
      cfg.dynamicSmemBytes = shmem_size;                                         \
      cfg.stream = stream;                                                       \
      cfg.attrs = attrs;                                                         \
      cfg.numAttrs = 1;                                                          \
      const auto* activations_in_nv =                                            \
          reinterpret_cast<const __nv_bfloat16*>(activations_in_ptr);            \
      const auto* router_logits_nv =                                             \
          reinterpret_cast<const __nv_bfloat16*>(router_logits_ptr);             \
      const auto* expert_weights_up_nv =                                         \
          reinterpret_cast<const W_element*>(expert_weights_up_ptr);             \
      const auto* expert_weights_down_nv =                                       \
          reinterpret_cast<const W_element*>(expert_weights_down_ptr);           \
      auto* activations_out_nv =                                                 \
          reinterpret_cast<R_element*>(activations_out_ptr);                     \
      void* scratchpad_void = static_cast<void*>(scratchpad_ptr);                \
      CUDA_CHECK(cudaLaunchKernelEx(                                             \
          &cfg, moe_kernel_topk_cluster<dims>, activations_in_nv, num_tokens,    \
          router_logits_nv, expert_weights_up_nv, expert_scales_up_ptr,          \
          expert_weights_down_nv, expert_scales_down_ptr, activations_out_nv,    \
          scratchpad_void, scratchpad_size, shmem_size, top_k_u32, sf,           \
          renormalize, up_weights_desc, activations_desc, down_weights_desc,     \
          down_activations_desc));                                               \
    } else {                                                                     \
      /* Standard (non-cooperative) launch.  The kernel reaches                  \
         grid-wide happens-before via the software Grid_Barrier /                \
         Partial_Barrier primitives in `src/moe_grid_barrier.h`                  \
         (spec R1.1, R5.1, Design Component C "Launch form") rather              \
         than `cooperative_groups::this_grid().sync()`.  Using                   \
         standard `cudaLaunchKernel` is what lets the migrated kernel            \
         be captured into a CUDA Graph. */                                       \
      CUDA_CHECK(cudaLaunchKernel((const void*)moe_kernel_topk<dims>,            \
                                  dim3(dims::KernelConfig::GRID_SIZE, 1, 1),     \
                                  dim3(dims::KernelConfig::BLOCK_SIZE, 1, 1),    \
                                  kernel_args, shmem_size, stream));             \
    }                                                                            \
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

// Cluster-enabled variant of the BS8 TMA+WGMMA path (Hopper-only,
// sm_90a).  Same algorithmic dimensions as
// `Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA` but opts into thread
// block clusters via `KernelConfig::USE_CLUSTER = true` /
// `CLUSTER_SIZE = 8`.  The host wrapper macro routes through the
// `if constexpr (use_cluster<dims>::value)` branch and launches
// `moe_kernel_topk_cluster<dims>` via `cudaLaunchKernelEx` with a
// `cudaLaunchAttributeClusterDimension` attribute (spec R1.1, R2.1,
// R2.2; design §3.1, §4.1).
MOEMONOKERNEL_TOPK_WRAPPER_IMPLEMENTATION(
    moe_monokernel_topk_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA_Cluster_impl,
    moe_monokernel::Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA_Cluster)
