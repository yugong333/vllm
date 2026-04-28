
#pragma once
#ifndef MOE_INTERNAL_H
  #define MOE_INTERNAL_H

  #ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
    #error Do not include this file directly.
  #endif

  #include "moe_interface.h"

namespace moe_monokernel {

using T_element =
    float;  //< Type of fp32 accumulators (partial results, out_accum)
using OpaqueElement = std::uint32_t;  //< Auxiliary 32-bit type used to generate
                                      // better assembly code in loads

/**
 * @brief Offsets into the @c token_indexes field
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
  // experts, so the sorted temp buffer must hold BS * SPEC_MAX_TOPK rows. BS <=
  // 8 now also uses BS * SPEC_MAX_TOPK rows because the split-phase design
  // writes one row per (token, expert) pair into spec->temp_bf16.
  static constexpr uint32_t TEMP_ROWS = Dims::BS * SPEC_MAX_TOPK + 8;

  // Number of blocks that contribute columns to the N-wide up-projection
  // output.  Each block writes W_UP_COLS_PER_BLOCK columns; blocks beyond N
  // are idle.  W_UP_TILE is always 16, so each block covers 8 columns.
  static constexpr uint32_t W_UP_COLS_PER_BLOCK = 8;  // = W_UP_TILE / 2
  static constexpr uint32_t UP_PROJ_BLOCK_COUNT =
      (Dims::N + W_UP_COLS_PER_BLOCK - 1) / W_UP_COLS_PER_BLOCK;

  #ifdef DEBUG_MOE
  // Debug information passed out. The actual token_indexes are stored in shared
  // memory.
  std::int32_t token_indexes[Dims::BS];
  T_element gemm1[TEMP_ROWS * 2 * Dims::N];
  #endif
  AQ_element activations[Dims::BS]
                        [Dims::HIDDEN_STATES];  //< Quantized activations

  // Up-projection SiLU output.  BS8 and BS64 both now use bf16 here:
  // the BS64 down-projection does a bf16→fp8 quantization with per-token
  // per-block scales, just like BS8.  Storing this in bf16 (not fp32)
  // halves the global-memory footprint and async-copy bandwidth.
  //
  //   BS8:  the down-projection reads block-local maxes from temp_block_max
  //         and does a single-pass bf16→fp8 quantization.
  //   BS64: the down-projection loads tiles into SHM and computes per-token
  //         block-wise scales on the fly before quantizing.
  A_element temp_bf16[TEMP_ROWS * Dims::N];

  // Per-block absmax of each row in temp_bf16 (BS8 path only).
  // Written by the up-projection epilogue, read by the down-projection
  // to compute the true row max without a separate global-memory pass.
  float temp_block_max[TEMP_ROWS * UP_PROJ_BLOCK_COUNT];

  // Per-token block-wise activation quantization scales.
  // Block size = 128 along K dimension → K/128 scales per token.
  // act_scale[tok][blk] = max(|x_tok[blk*128..(blk+1)*128-1]|) / 448
  static constexpr uint32_t ACT_BLOCK_SIZE = 128;
  static constexpr uint32_t ACT_SCALE_BLOCKS =
      (Dims::HIDDEN_STATES + ACT_BLOCK_SIZE - 1) / ACT_BLOCK_SIZE;
  float act_scale[Dims::BS][ACT_SCALE_BLOCKS];
};

  // Maximum supported dimensions for shared memory and scratchpad allocation
  // sizes
  #if USE_SMALL_SETUP
// SHM limits batch size to ~2k
using Dims_Max = MoEDimensions<1024, 256, 1024, 256>;
  #else
using Dims_Max = MoEDimensions<1024, 1024, 5120, 256>;
  #endif

// ── Block-wise quantization detection (forward declaration) ──────────────
// These helpers are used in the MoE_SHM layout below and defined with full
// SFINAE semantics further down. Here we just need the compile-time bool,
// so we duplicate the minimal detection inline.
template <typename Dims>
struct shm_is_block_wise {
  template <typename D>
  static constexpr auto test(int) -> decltype(D::QUANT_GRAN, bool()) {
    return D::QUANT_GRAN == QuantGranularity::BLOCK_WISE;
  }
  template <typename>
  static constexpr bool test(...) {
    return false;
  }
  static constexpr bool value = test<Dims>(0);
};

// Number of column-blocks in the up-projection scale tensor. For block-wise
// this is ceil(K / BLOCK_SCALE_COL); for per-channel we return 1 (unused
// placeholder so the SHM field is harmlessly tiny).
template <typename Dims, bool IsBlockWise = shm_is_block_wise<Dims>::value>
struct shm_up_scale_cols {
  static constexpr uint32_t value = 1;
};
template <typename Dims>
struct shm_up_scale_cols<Dims, true> {
  static constexpr uint32_t value = Dims::UP_SCALE_COLS;
};

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
      // Full-K double-buffered activation and weight tiles.
      // With K <= 2048 (e.g. Qwen3.5 K=2048), the full K activation tile
      // (A_TILE × K × fp8 = 16 KB) and weight tile (W_UP_TILE × K × fp8 =
      // 32 KB) both fit comfortably in SHM with double-buffering, removing
      // the need for the half-K split and its triple-buffer pipeline.
      AQ_element a[2][CoreDims::A_TILE][CoreDims::K_DIM_PADDED_A];
      W_element w[2][CoreDims::W_UP_TILE][CoreDims::K_DIM_PADDED_W];
      T_element partial_result[CoreDims::CALC_WARP_COUNT]
                              [CoreDims::W_UP_TILE * CoreDims::T_TILE];
    } gemm1;
    // BS8 split-phase path: up-projection and down-projection run as
    // separate all-experts loops with a single grid.sync() in between.
    //
    // Compact union layout:
    //   a: fp8 up-activations / double-buffered fp8 down-activations
    //   w[2]: ping-pong orig(bf16) / w_up(fp8) / bf16_buf(bf16) / w_down(fp8)
    //   partial_result: up / down scratch
    //
    // Down-projection uses a pipelined design with fixed w[2] slot roles:
    //   w[0].bf16_buf:  bf16 intermediate results from global memory
    //   w[1].down:      fp8 down-projection weights
    //   a.down[2]:      double-buffered fp8 quantized activations for MMA
    //                   (reuses the same union as a.up — safe because all
    //                   up-projections finish before any down-projection
    //                   starts)
    struct TinyData {
      // Input activations for up- and down-projection (mutually exclusive).
      // Up-projection is fully complete (with grid.sync) before
      // down-projection begins, so a.up and a.down[2] safely share storage.
      // a.down[2] is double-buffered for the down-projection pipeline:
      // one buffer is consumed by MMA while the other is written by
      // quantization.
      union {
        AQ_element up[CoreDims::T_TILE][CoreDims::K_DIM_PADDED_A];  // fp8
        AQ_element down[2][CoreDims::T_TILE][Dims::N];              // fp8
      } a;

      // Per-row per-block quantization scale for a.down (double-buffered).
      // Block-wise (1, 128): each row of N elements gets N/128 scales.
      static constexpr uint32_t A_DOWN_SCALE_BLOCKS = (Dims::N + 127) / 128;
      S_element a_down_scale[2][CoreDims::T_TILE][A_DOWN_SCALE_BLOCKS];

      // Double-buffered weight / activation tiles.
      //
      // During init (Phase 1–2), w[0].orig holds the raw bf16 activations
      // fetched from global memory (before quantization), and w[1].up holds
      // the first expert's up-projection weights.
      //
      // During the down-projection (Phase 4), w[2] has fixed roles:
      // w[0] holds bf16 intermediate results (w[0].bf16_buf) and
      // w[1] holds fp8 down-projection weights (w[1].down).
      // They use separate slots so bf16_buf and w_down never conflict.
      union {
        A_element orig[CoreDims::T_TILE][CoreDims::K_DIM_PADDED_A];
        A_element bf16_buf[CoreDims::T_TILE][Dims::N];
        W_element up[CoreDims::W_UP_TILE][CoreDims::K_DIM_PADDED_W];
        W_element down[CoreDims::W_DOWN_TILE]
                      [Dims::N + CoreDims::PADDING / sizeof(W_element)];
      } w[2];

      // Down-projection weight scales (double-buffered).
      // Block-wise (128×128): [2][ceil(W_DOWN_TILE/128) * ceil(N/128)]
      //   For K=2048, GRID=64: W_DOWN_TILE=32, so ceil(32/128)=1
      //   For N=512: ceil(512/128)=4, so 1*4=4 scales per expert per block
      static constexpr uint32_t DOWN_SCALE_TILE_SIZE =
          ((CoreDims::W_DOWN_TILE + 127) / 128) * ((Dims::N + 127) / 128);
      S_element scale[2][DOWN_SCALE_TILE_SIZE + CoreDims::PADDING];

      // Up-projection weight scales (double-buffered, block-wise only).
      // Block-wise (128×128): each block's weight tile spans 8 rows in the
      // low half and 8 rows in the upper half of the 2*N weight rows. With
      // BLOCK_SCALE_ROW=128 and base_row_up multiple of 8, all 8 rows fall
      // in a single row-block. So we need 2 row-blocks × ceil(K/128)
      // col-blocks per expert per CUDA block.
      //
      // For per-channel this field is sized to a trivial placeholder (never
      // read). We keep it allocated unconditionally to avoid template-
      // dependent SHM layout branching.
      static constexpr uint32_t UP_SCALE_TILE_SIZE =
          2 * shm_up_scale_cols<Dims>::value;
      S_element up_scale[2][UP_SCALE_TILE_SIZE];

      // Scratch pad for MMA partial results (up and down share the same space).
      union {
        T_element up[CoreDims::CALC_WARP_COUNT]
                    [CoreDims::W_UP_TILE * CoreDims::T_TILE];
        T_element
            down[CoreDims::W_DOWN_TILE / 2 + CoreDims::CALC_WARP_COUNT / 2]
                [CoreDims::W_DOWN_MMA_TILE * CoreDims::T_TILE];
      } partial_result;

      // Per-block fp32 accumulator for down-projection output.
      T_element out_accum[Dims::BS][CoreDims::W_DOWN_TILE];
    } tiny;
    // BS64 path: holds weight tiles and partial results for down-projection
    // only (up-projection uses Gemm1Data; activations come from
    // spec->temp_bf16)
    //
    // Uses the same fp8 MMA approach as BS8: SiLU output is fetched as
    // bf16, quantized to fp8 with per-token block-wise scales, then
    // multiplied with fp8 weights via mma_fp8_fp8 (m16n8k32).
    struct Gemm2Data {
      // Double-buffered bf16 staging area for SiLU output (fetched from
      // global memory, consumed by the quantization step).
      A_element t_bf16[2][CoreDims::T_TILE][Dims::N];
      // Double-buffered fp8 quantized activations for MMA.
      AQ_element t_fp8[2][CoreDims::T_TILE][Dims::N];
      // Per-token per-block activation scales for the fp8 activations.
      static constexpr uint32_t A_DOWN_SCALE_BLOCKS = (Dims::N + 127) / 128;
      S_element t_scale[2][CoreDims::T_TILE][A_DOWN_SCALE_BLOCKS];

      W_element w[2][CoreDims::W_DOWN_TILE]
                 [Dims::N + CoreDims::PADDING / sizeof(W_element)];
      // Down-projection weight scales (double-buffered).
      // Block-wise (128×128): [2][ceil(W_DOWN_TILE/128) * ceil(N/128)]
      static constexpr uint32_t DOWN_SCALE_TILE_SIZE =
          ((CoreDims::W_DOWN_TILE + 127) / 128) * ((Dims::N + 127) / 128);
      S_element scale[2][DOWN_SCALE_TILE_SIZE + CoreDims::PADDING];
      T_element partial_result[CoreDims::W_DOWN_TILE / 2 +
                               CoreDims::CALC_WARP_COUNT / 2]
                              [CoreDims::W_DOWN_MMA_TILE * CoreDims::T_TILE];
    } gemm2;
  } u;

  static_assert(Dims::NUM_EXPERTS <= 65535,
                "Number of experts too high, cannot store as uint16 anymore.");

  // ── Common fields (both BS8 and BS64) ────────────────────────────────────

  // act_scale[tok][blk] = max(|x_tok[blk*128..(blk+1)*128-1]|)/448
  // Per-token block-wise activation quantization scales for up-projection.
  static constexpr uint32_t ACT_BLOCK_SIZE = 128;
  static constexpr uint32_t ACT_SCALE_BLOCKS =
      (Dims::HIDDEN_STATES + ACT_BLOCK_SIZE - 1) / ACT_BLOCK_SIZE;
  S_element act_scale[Dims::BS][ACT_SCALE_BLOCKS];

  // Unique experts active in this batch, with their sorted token ranges.
  // Filled by prepare_moe_topk_BS8 (BS8) or prepare_moe_topk_BSx_Ey (BS64).
  ExpertRef experts[Dims::NUM_EXPERTS];
  std::uint32_t expert_count;

  // Flat routing results: [token * MAX_TOPK + k] = expert id / routing weight
  // for the k-th selection of that token. Written by topK_BS8 / topK_BS64.
  // MAX_TOPK = 8 covers top_k up to 8.
  static constexpr uint32_t MAX_TOPK = 8;
  alignas(uint64_t) uint16_t
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
    // token_weights[sorted_pos]      = routing_weight.
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

// ── Block-wise scale helpers ──────────────────────────────────────────────
namespace moe_monokernel {

/**
 * @brief Check at compile time whether Dims uses block-wise quantization.
 */
template <typename Dims>
struct is_block_wise {
  // SFINAE: check if QUANT_GRAN exists and equals BLOCK_WISE
  template <typename D>
  static constexpr auto test(int) -> decltype(D::QUANT_GRAN, bool()) {
    return D::QUANT_GRAN == QuantGranularity::BLOCK_WISE;
  }
  template <typename>
  static constexpr bool test(...) {
    return false;
  }
  static constexpr bool value = test<Dims>(0);
};

/**
 * @brief Fetch the block-wise up-projection scale for a given row and K-column.
 *
 * @param expert_scales_up  Pointer to the full scale tensor (global memory).
 * @param expert_id         Expert index.
 * @param row               Row index within the [2*N, K] weight matrix.
 * @param k_col             Column index along the K dimension (full K, not
 * half).
 */
template <typename Dims>
__device__ __forceinline__ float get_up_block_scale(
    const S_element* __restrict__ expert_scales_up, uint32_t expert_id,
    uint32_t row, uint32_t k_col) {
  if constexpr (!is_block_wise<Dims>::value) {
    // Per-channel: one scale per row
    return expert_scales_up[expert_id * 2 * Dims::N + row];
  } else {
    uint32_t rb = row / Dims::BLOCK_SCALE_ROW;
    uint32_t kb = k_col / Dims::BLOCK_SCALE_COL;
    return expert_scales_up[expert_id * Dims::UP_SCALE_ROWS *
                                Dims::UP_SCALE_COLS +
                            rb * Dims::UP_SCALE_COLS + kb];
  }
}

/**
 * @brief Fetch the block-wise down-projection scale for a given row and
 * N-column.
 *
 * @param expert_scales_down  Pointer to the full scale tensor (global memory).
 * @param expert_id           Expert index.
 * @param row                 Row index within the [K, N] weight matrix.
 * @param n_col               Column index along the N dimension.
 */
template <typename Dims>
__device__ __forceinline__ float get_down_block_scale(
    const S_element* __restrict__ expert_scales_down, uint32_t expert_id,
    uint32_t row, uint32_t n_col) {
  if constexpr (!is_block_wise<Dims>::value) {
    // Per-channel: one scale per row
    return expert_scales_down[expert_id * Dims::HIDDEN_STATES + row];
  } else {
    uint32_t rb = row / Dims::BLOCK_SCALE_ROW;
    uint32_t nb = n_col / Dims::BLOCK_SCALE_COL;
    return expert_scales_down[expert_id * Dims::DOWN_SCALE_ROWS *
                                  Dims::DOWN_SCALE_COLS +
                              rb * Dims::DOWN_SCALE_COLS + nb];
  }
}

}  // namespace moe_monokernel

#endif
