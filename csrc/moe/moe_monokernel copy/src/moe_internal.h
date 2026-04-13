
#pragma once
#ifndef MOE_INTERNAL_H
  #define MOE_INTERNAL_H

  #ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
    #error Do not include this file directly.
  #endif

  #include "moe_interface.h"

namespace moe_monokernel {

using T_element =
    float;  //< Type of GEMM1 (up projection) result as well as sigmoid
using OpaqueElement = std::uint32_t;  //< Auxiliary 32-bit type used to generate
                                      // better assembly code in loads

/**
 * @brief Offets into the @c token_indexes field
 *
 * This is an offset array. To find all the tokens that belong to expert @c id :
 * <tt>
 * for (int i = first_token; i < last_token; i++) {
 *    int token_index = token_indexes[i];
 * }
 */
struct ExpertRef {
  std::uint16_t first_token;
  std::uint16_t last_token;
  std::uint32_t id;
};

/**
 * @brief Scratchpad memory for use within the monokernel.
 *
 * Place in global memory.
 *
 */
template <typename Dims>
struct MoEGemmSpec {
  static constexpr uint32_t SPEC_MAX_TOPK = 8;
  // Virtual batch size: each token may be routed to up to SPEC_MAX_TOPK
  // experts, so the sorted temp buffer must hold BS * SPEC_MAX_TOPK rows (BS64
  // path). For BS <= 8 the tiny path indexes temp by original token, so BS + 8
  // suffices, but we size for the worst case to keep a single definition.
  static constexpr uint32_t TEMP_ROWS =
      (Dims::BS <= 8) ? (Dims::BS + 8) : (Dims::BS * SPEC_MAX_TOPK + 8);

  #ifdef DEBUG_MOE
  // Debug information passed out. The actual token_indexes are stored in shared
  // memory.
  std::int32_t token_indexes[Dims::BS];
  T_element gemm1[TEMP_ROWS * 2 * Dims::N];
  #endif
  AQ_element activations[Dims::BS]
                        [Dims::HIDDEN_STATES];  //< Quantized activations
  T_element temp[TEMP_ROWS * Dims::N];          //< Up projection result
  float act_scale[Dims::BS];  //< per-token activation quantization scale
                              //(max/448)
};

  // Maximum supported dimensions for shared memory and scratchpad allocation
  // sizes
  #if USE_SMALL_SETUP
// SHM limits batch size to ~2k
using Dims_Max = MoEDimensions<1024, 256, 1024, 128>;
  #else
using Dims_Max = MoEDimensions<1024, 1024, 5120, 128>;
  #endif

/**
 * @brief contains various constants used within the MoE monokernel.
 */
template <typename Dims>
struct MoECoreDims {
  using MoEDims = Dims;

  // GPU configuration.
  static constexpr std::uint32_t THREADS_PER_WARP = 32;
  static constexpr std::uint32_t TOTAL_WARP_COUNT =
      Dims::KernelConfig::BLOCK_SIZE / THREADS_PER_WARP;
  static constexpr std::uint32_t CALC_WARP_COUNT = 8;
  static constexpr std::uint32_t PREFETCH_WARP_COUNT =
      TOTAL_WARP_COUNT - CALC_WARP_COUNT;

  // MMA 1 matrix tile dimensions.
  static constexpr std::uint32_t A_TILE = 8;
  static constexpr std::uint32_t W_UP_TILE = 16;
  static constexpr std::uint32_t K_TILE = 32;

  // GEMM 2 matrix tile dimensions.
  static constexpr std::uint32_t W_DOWN_MMA_TILE = 16;
  static constexpr std::uint32_t W_DOWN_TILE =
      Dims::HIDDEN_STATES / Dims::KernelConfig::GRID_SIZE;
  static constexpr std::uint32_t T_TILE = 8;

  static constexpr std::uint32_t W_DIM = 2 * Dims::N;

  static constexpr unsigned BLOCK_STRIDE = CALC_WARP_COUNT * K_TILE;

  static constexpr unsigned PADDING =
      32;  // this works *slightly* better than 16 due to reduced L2 transfers
  static constexpr unsigned K_DIM_PADDED_A = Dims::HIDDEN_STATES;
  static constexpr unsigned K_DIM_PADDED_W = Dims::HIDDEN_STATES;
  static constexpr unsigned K_DIM_HALF_PADDED_A = Dims::HIDDEN_STATES / 2;
  static constexpr unsigned K_DIM_HALF_PADDED_W = Dims::HIDDEN_STATES / 2;
};

// 1 tile per warp
// 20 warps x 2 params x 1k = 20k pre-fetch
template <typename Dims>
struct MoE_SHM {
  using CoreDims = MoECoreDims<Dims>;
  union U {
    struct SortData {
      std::uint32_t counters[Dims::NUM_EXPERTS][CoreDims::THREADS_PER_WARP];
      std::uint32_t total_counts[Dims::NUM_EXPERTS];
    } sorting;
    struct RescaleData {
      A_element a[CoreDims::CALC_WARP_COUNT][Dims::HIDDEN_STATES];
    } rescale;
    struct Gemm1Data {
      // prefetch & process tile in 2 halves
      AQ_element a[3][CoreDims::A_TILE][CoreDims::K_DIM_HALF_PADDED_A];
      W_element w[3][CoreDims::W_UP_TILE][CoreDims::K_DIM_HALF_PADDED_W];
      T_element partial_result[CoreDims::CALC_WARP_COUNT]
                              [CoreDims::W_UP_TILE * CoreDims::T_TILE];
    } gemm1;
    // BS8 path: holds activations, weight tiles, and partial results
    // for both up- and down-projection (entire pipeline fits in one struct).
    //
    // Separate w_up and w_down buffers (no union/aliasing between them) so
    // that down-weights can be prefetched while up-projection is computing,
    // and the next expert's up-weights can be prefetched while down-projection
    // is computing.  This saves one full up-weight buffer (~80 KB) compared to
    // the previous w[2] double-buffer design while also enabling full overlap.
    struct TinyData {
      // Quantized fp8 activations — persistent across all expert iterations.
      // Kept separate from a_down so that the down-projection's fp32 temps
      // never clobber the quantized inputs, eliminating the per-expert
      // global-memory round-trip through spec->activations.
      AQ_element a_up[CoreDims::T_TILE][CoreDims::K_DIM_PADDED_A];

      // Down-projection input (fp32 SiLU output from up-projection).
      T_element a_down[CoreDims::T_TILE][Dims::N];

      // Raw (unquantized) activations fetched from global memory.
      // Reused as staging area before quantization; not live at the same
      // time as w_up / w_down.
      A_element orig[CoreDims::T_TILE][CoreDims::K_DIM_PADDED_A];

      // Up-projection weight tile — one buffer, prefetched during down-compute.
      W_element w_up[CoreDims::W_UP_TILE][CoreDims::K_DIM_PADDED_W];

      // Down-projection weight tile — one buffer, prefetched during up-compute.
      W_element w_down[CoreDims::W_DOWN_TILE]
                      [Dims::N + CoreDims::PADDING / sizeof(W_element)];

      // Down-projection scales (single buffer, loaded together with w_down).
      S_element scale_down[CoreDims::W_DOWN_TILE + CoreDims::PADDING];

      // scratch pad
      union {
        T_element up[CoreDims::CALC_WARP_COUNT]
                    [CoreDims::W_UP_TILE * CoreDims::T_TILE];
        T_element
            down[CoreDims::W_DOWN_TILE / 2 + CoreDims::CALC_WARP_COUNT / 2]
                [CoreDims::W_DOWN_MMA_TILE * CoreDims::T_TILE];
      } partial_result;

      // Per-block fp32 accumulator for down-projection output.
      // Each block owns W_DOWN_TILE columns of the output; accumulating in
      // SHM avoids repeated global read-modify-write (bf16→fp32→bf16) per
      // expert and eliminates the associated precision loss.
      // Written once to global memory (as bf16) after the expert loop.
      // Size: BS × W_DOWN_TILE × 4B = 8 × 16 × 4 = 512 bytes.
      T_element out_accum[Dims::BS][CoreDims::W_DOWN_TILE];
    } tiny;
    // BS64 path: holds weight tiles and partial results for down-projection
    // only (up-projection uses Gemm1Data; activations come from spec->temp)
    struct Gemm2Data {
      // prefetch 1 tile ahead
      T_element t[2][CoreDims::T_TILE][Dims::N];
      W_element w[2][CoreDims::W_DOWN_TILE]
                 [Dims::N + CoreDims::PADDING / sizeof(W_element)];
      S_element scale[2][CoreDims::W_DOWN_TILE + CoreDims::PADDING];
      T_element partial_result[CoreDims::W_DOWN_TILE / 2 +
                               CoreDims::CALC_WARP_COUNT / 2]
                              [CoreDims::W_DOWN_MMA_TILE * CoreDims::T_TILE];
    } gemm2;
  } u;

  static_assert(Dims::NUM_EXPERTS < 255,
                "Number of experts too high, cannot store as uint8 anymore.");

  // ── Common fields (both BS8 and BS64) ────────────────────────────────────

  // act_scale[tok] = max(|x_tok|)/448 — computed once per token during
  // quantization, used inside silu for every expert this token is routed to.
  S_element act_scale[Dims::BS];

  // Unique experts active in this batch, with their sorted token ranges.
  // Filled by prepare_moe_topk_BS8 (BS8) or prepare_moe_topk_BSx_Ey (BS64).
  ExpertRef experts[Dims::NUM_EXPERTS];
  std::uint32_t expert_count;

  // Flat routing results: [token * MAX_TOPK + k] = expert id / routing weight
  // for the k-th selection of that token. Written by topK_BS8 / topK_BS64.
  // MAX_TOPK = 8 covers top_k up to 8.
  static constexpr uint32_t MAX_TOPK = 8;
  alignas(uint64_t) uint8_t
      topk_ids_flat[(Dims::BS < 8 ? 8 : Dims::BS) * MAX_TOPK];
  S_element topk_weights_flat[(Dims::BS < 8 ? 8 : Dims::BS) * MAX_TOPK];

  // ── Path-specific fields (union: BS8 and BS64 never run simultaneously) ──
  // BS8 uses only 8 bytes (expert_ids); BS64 uses ~2KB (token arrays).
  // The union saves ~2KB of shared memory for the BS8 instantiation.
  union PathData {
    // BS8: packed unique expert ids, one per byte (up to 8 experts).
    // Used by prepare_moe_topk_BS8 to store iteration order.
    struct {
      std::uint64_t expert_ids;
    } bs8;

    // BS64: sorted virtual-batch index arrays.
    // token_indexes_topk[sorted_pos] = original token index.
    // token_weights[sorted_pos]      = act_scale (after step 3b) or
    //                                  routing_weight (before step 3b).
    struct {
      std::uint16_t
          token_indexes_topk[Dims::BS * MAX_TOPK + MoECoreDims<Dims>::PADDING];
      S_element token_weights[Dims::BS * MAX_TOPK + MoECoreDims<Dims>::PADDING];
    } bs64;
  } path;
};

/**
 * @brief Returns the amount of shared memory necessary to run @c moe_kernel
 * with template parameter @p Dims
 */
template <typename Dims>
__device__ __host__ constexpr size_t get_moe_shmem_size() {
  static_assert(Dims::M <= Dims_Max::M,
                "Dimension larger than the maximum supported dimension.");
  static_assert(Dims::N <= Dims_Max::N,
                "Dimension larger than the maximum supported dimension.");
  static_assert(Dims::K <= Dims_Max::K,
                "Dimension larger than the maximum supported dimension.");
  static_assert(Dims::NUM_EXPERTS <= Dims_Max::NUM_EXPERTS,
                "Dimension larger than the maximum supported dimension.");
  return sizeof(MoE_SHM<Dims>);
}

constexpr size_t get_moe_max_shmem_size() { return sizeof(MoE_SHM<Dims_Max>); }

/**
 * @brief Returns the amount of global scratchpad memory necessary to run
 * moe_kernel() with template parameter @p Dims
 */
template <typename Dims>
__device__ __host__ constexpr size_t get_moe_scratchpad_size() {
  static_assert(Dims::M <= Dims_Max::M,
                "Dimension larger than the maximum supported dimension.");
  static_assert(Dims::N <= Dims_Max::N,
                "Dimension larger than the maximum supported dimension.");
  static_assert(Dims::K <= Dims_Max::K,
                "Dimension larger than the maximum supported dimension.");
  static_assert(Dims::NUM_EXPERTS <= Dims_Max::NUM_EXPERTS,
                "Dimension larger than the maximum supported dimension.");
  return sizeof(MoEGemmSpec<Dims>);
}

constexpr size_t get_moe_max_scratchpad_size() {
  return sizeof(MoEGemmSpec<Dims_Max>);
}

template <typename Dims>
inline __device__ bool is_calc_warp() {
  using CoreDims = MoECoreDims<Dims>;
  return threadIdx.x < CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP;
}

template <typename Dims>
inline __device__ bool is_prefetch_warp() {
  using CoreDims = MoECoreDims<Dims>;
  return threadIdx.x >= CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP;
}

template <typename Dims>
inline __device__ unsigned get_thread() {
  using CoreDims = MoECoreDims<Dims>;
  return threadIdx.x % CoreDims::THREADS_PER_WARP;
}

template <typename Dims>
inline __device__ unsigned get_any_warp() {
  using CoreDims = MoECoreDims<Dims>;
  return threadIdx.x / CoreDims::THREADS_PER_WARP;
}

template <typename Dims>
inline __device__ unsigned get_calc_warp() {
  using CoreDims = MoECoreDims<Dims>;
  assert(is_calc_warp<Dims>());
  return threadIdx.x / CoreDims::THREADS_PER_WARP;
}

template <typename Dims>
inline __device__ unsigned get_prefetch_warp() {
  using CoreDims = MoECoreDims<Dims>;
  assert(is_prefetch_warp<Dims>());
  return threadIdx.x / CoreDims::THREADS_PER_WARP - CoreDims::CALC_WARP_COUNT;
}

/**
 * @brief Synchronizes the first 256 threads of the calling CUDA block
 *
 * This is a collective operation that needs to be called by all of the first
 * 256 threads in each CUDA block.
 *
 */
template <typename Dims>
__device__ __forceinline__ void sync_calc_threads() {
  // First 256 threads
  using CoreDims = MoECoreDims<Dims>;
  static_assert(CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP == 256,
                "Adapt the thread number if sync_calc_threads");
  __asm volatile("bar.sync  15, 256;\n");
}

/**
 * @brief Computes the maximum value within a warp
 *
 * This is a collective operation. Each thread in a warp needs to call it.
 * The resulting maximum value is returned on all threads.
 *
 */
__device__ static inline float warp_reduce_max_float(float value) {
  for (int i = 16; i >= 1; i /= 2) {
    value = fmaxf(__shfl_xor_sync(0xffffffff, value, i, 32), value);
  }
  return value;
}

/**
 * @brief Reinterprets the bit-pattern of @p x to type @p To
 */
template <typename To, typename From>
__device__ static __forceinline__ To type_pun(From x) {
  static_assert(sizeof(To) == sizeof(From), "Types of different size");
  To y;
  // This memcpy is optimized out by NVCC
  memcpy(&y, &x, sizeof(From));
  return y;
}

}  // namespace moe_monokernel

#endif
