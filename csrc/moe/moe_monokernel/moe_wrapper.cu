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

// ── TEMP_FP8_OFFSET regression anchor ──────────────────────────────────────
//
// The host-side down-activation TMA descriptor factory below computes the
// device pointer to `spec->temp_fp8` as
//   `scratchpad_ptr + MoEGemmSpec<Dims>::TEMP_FP8_OFFSET`
// so `TEMP_FP8_OFFSET` MUST stay byte-identical to
// `offsetof(MoEGemmSpec<Dims>, temp_fp8)` for every instantiated `Dims`
// variant.  This invariant matters when appending new barrier-counter
// fields to the tail of `MoEGemmSpec<Dims>`: as long as every new field
// lands AFTER `temp_fp8` (grid_barrier / partial_barrier belong at the
// tail), the offset stays fixed and the TMA descriptor continues to
// address the right bytes.  A future refactor that silently reorders the
// struct layout would otherwise be caught only at runtime by corrupted
// TMA fetches — the static_assert below makes it a compile-time error.
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
    torch::stable::Tensor& activations_out,
    torch::stable::Tensor& scratchpad, int64_t top_k, int64_t scoring_func,
    bool renormalize, const char* diag_name);

/**
 * @brief Macro that expands to a named op wrapper forwarding to
 * `launch_moe_monokernel<dims>`.  Kept for the existing per-shape ops so
 * their torch_bindings entries are unchanged.
 */
#define MOEMONOKERNEL_TOPK_WRAPPER_IMPLEMENTATION(name, dims)                  \
  void name(const torch::stable::Tensor& activations_in,                       \
            const torch::stable::Tensor& router_logits,                        \
            const torch::stable::Tensor& expert_weights_up,                    \
            const torch::stable::Tensor& expert_scales_up,                     \
            const torch::stable::Tensor& expert_weights_down,                  \
            const torch::stable::Tensor& expert_scales_down,                   \
            torch::stable::Tensor& activations_out,                            \
            torch::stable::Tensor& scratchpad,                                 \
            int64_t top_k, int64_t scoring_func, bool renormalize) {           \
    moe_monokernel::launch_moe_monokernel<dims>(                               \
        activations_in, router_logits, expert_weights_up, expert_scales_up,    \
        expert_weights_down, expert_scales_down, activations_out, scratchpad,  \
        top_k, scoring_func, renormalize, #name);                              \
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
            torch::stable::Tensor& activations_out,
            torch::stable::Tensor& scratchpad,
            int64_t top_k, int64_t scoring_func, bool renormalize,
            const char* diag_name) {
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
    char* scratchpad_ptr =
        reinterpret_cast<char*>(scratchpad.mutable_data_ptr());

    using namespace moe_monokernel;
    const uint32_t num_tokens = activations_in.size(0);
    const size_t shmem_size = get_moe_shmem_size<dims>();
    const size_t scratchpad_size =
        static_cast<size_t>(scratchpad.numel()) * scratchpad.element_size();
    const uint32_t top_k_u32 = static_cast<uint32_t>(top_k);
    const ScoringFunc sf = static_cast<ScoringFunc>(scoring_func);

    /* TMA descriptors for the BS8 WGMMA up-projection path and
       down-projection path.  Non-TMA
       variants leave these zero-initialized — the kernel parameters are
       always present on the signature but the TMA path is the only
       consumer.  TMA-enabled variants build real descriptors via the
       host-side factories and pass them in kernel_args positions matching
       the kernel signature. */
    CUtensorMap up_weights_desc{};
    CUtensorMap activations_desc{};
    CUtensorMap down_weights_desc{};
    CUtensorMap down_activations_desc{};
    if constexpr (use_tma<dims>::value) {
      /* Up-projection weight descriptor (SWIZZLE_128B).  Callers MUST
         pre-interleave `expert_weights_up` via
         `interleave_for_tma_wgmma_up` in Python — the helper repacks
         gate/up row stripes so a single 128x128 TMA fetches the full
         WGMMA A-tile. */
      up_weights_desc = create_up_weight_tma_desc(
          reinterpret_cast<const void*>(expert_weights_up_ptr),
          dims::NUM_EXPERTS, dims::N, dims::K);
      activations_desc = create_activations_tma_desc(
          reinterpret_cast<const void*>(activations_in_ptr), dims::BS,
          dims::HIDDEN_STATES);
      /* Down-projection weight descriptor (SWIZZLE_128B).  Callers MUST
         NOT pre-interleave `expert_weights_down` — the TMA hardware
         applies the core-matrix XOR swizzle at write time and expects
         the raw row-major `[E, K, N]` fp8 tensor.  `row_box` is pinned
         to 128 (one 128-row atom per TMA): the TMA boxDim hardware cap
         is 256 rows, so DOWN_COL_TILE=384 (122B) cannot be a single box.
         The down-proj kernel issues DOWN_COL_HALVES 128-row TMAs per
         K-substep at GM col `base_col + h*128`, landing at SHM row
         `(kk*DOWN_COL_HALVES + h)*128` — byte-identical SHM layout to
         the old single-box delivery for DOWN_COL_TILE<=256. */
      down_weights_desc = create_down_weight_tma_desc(
          reinterpret_cast<const void*>(expert_weights_down_ptr),
          dims::NUM_EXPERTS, dims::HIDDEN_STATES, dims::N,
          /*row_box=*/128u);
      /* Down-projection activation descriptor reads from `spec->temp_fp8`
         which lives inside the scratchpad.  Compute the device pointer
         from the scratchpad base + the compile-time offset of temp_fp8
         inside `MoEGemmSpec<dims>`. */
      const void* temp_fp8_ptr =
          reinterpret_cast<const char*>(scratchpad_ptr) +
          MoEGemmSpec<dims>::TEMP_FP8_OFFSET;
      down_activations_desc = create_down_activation_tma_desc(
          temp_fp8_ptr, MoEGemmSpec<dims>::TEMP_ROWS_TMA, dims::N);
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
                           (void*)&up_weights_desc,
                           (void*)&activations_desc,
                           (void*)&down_weights_desc,
                           (void*)&down_activations_desc};
    const cudaStream_t stream =
        get_current_cuda_stream(activations_in.get_device_index());
    CUDA_CHECK(cudaFuncSetAttribute(
        moe_kernel_topk<dims>, cudaFuncAttributeMaxDynamicSharedMemorySize,
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
        /* Hard co-residency assertions for the software grid barrier
           "Co-residency assertions".  The seed-atomicAdd-spin-on-high-bit
           protocol
           in src/moe_grid_barrier.h is only deadlock-free when every
           participating block is co-resident on the GPU for the full
           lifetime of the kernel: (1) grid_size <= SM count so every
           block gets a slot, and (2) max_active_blocks_per_SM == 1 so
           no block is ever waiting on a block that has not yet been
           scheduled.  GRID_SIZE is a compile-time constexpr and SM
           count / occupancy are device-property-time static, so gating
           under `_diag_printed` keeps the check one-shot per process
           and off the hot path. */
        STD_TORCH_CHECK(
            dims::KernelConfig::GRID_SIZE <= static_cast<uint32_t>(sm_count),
            "moe_monokernel requires GRID_SIZE (=",
            dims::KernelConfig::GRID_SIZE, ") <= SM count (=", sm_count,
            ") for software grid barrier co-residency invariant.");
        /*TORCH_CHECK(max_blocks_per_sm == 1,
                    "moe_monokernel requires max_active_blocks_per_SM == 1 "
                    "(observed ",
                    max_blocks_per_sm,
                    ") for co-residency invariant. See "
                    "__launch_bounds__(BLOCK_SIZE, 1) and the SHM budget "
                    "requirement.");*/
        _diag_printed = true;
      }
    }
    /* One-shot scratchpad zero-init
       ("Scratchpad barrier counter zero-initialization").  The software
       Grid_Barrier / Partial_Barrier counters live at the tail of
       MoEGemmSpec<Dims> inside the scratchpad, and the
       seed-atomicAdd-spin-on-high-bit protocol requires the barrier slots
       to start at 0 so the first Seed_Thread write commits the
       `0x80000000u - (arrival_count - 1)` seed value cleanly.  The
       ping-pong reset discipline keeps the slots self-maintaining across
       subsequent kernel invocations (see MoEGemmSpec<Dims> block comment
       on grid_barrier and partial_barrier), so we only pay the zero-init
       cost once per process on the first launch.  Zeroing the full
       scratchpad (rather than just the counter region) is simpler and
       the cost is a few hundred microseconds one-time on H200 — trivial
       next to per-decode kernel launches. */
    {
      static bool _zeroed = false;
      if (!_zeroed) {
        CUDA_CHECK(
            cudaMemsetAsync(scratchpad_ptr, 0, scratchpad_size, stream));
        _zeroed = true;
      }
    }
    /* Standard (non-cooperative) launch.  The kernel reaches grid-wide
       happens-before via the software Grid_Barrier / Partial_Barrier
       primitives in `src/moe_grid_barrier.h` rather than
       `cooperative_groups::this_grid().sync()`.  Using standard
       `cudaLaunchKernel` is what lets the migrated kernel be captured
       into a CUDA Graph. */
    CUDA_CHECK(cudaLaunchKernel((const void*)moe_kernel_topk<dims>,
                                dim3(dims::KernelConfig::GRID_SIZE, 1, 1),
                                dim3(dims::KernelConfig::BLOCK_SIZE, 1, 1),
                                kernel_args, shmem_size, stream));
  }
}  // namespace moe_monokernel

// Pair_Layout V2 of the BS8 TMA + WGMMA path.  This is the ONLY BS8
// implementation: TMA-based weight + activation load (Phase 3,
// `KernelConfig::USE_TMA = true`), 4-deep weight TMA lookahead, a
// deferred up-projection epilogue, and the gate/up pair layout
// (`KernelConfig::USE_PAIR_LAYOUT = true`).  Up-projection weights
// must be repacked via `interleave_for_tma_wgmma_up_v2` in Python
// (gate/up pair interleave for single-issue TMA); down-projection
// weights are passed raw row-major (the TMA hardware applies the
// core-matrix XOR swizzle at write time).
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
