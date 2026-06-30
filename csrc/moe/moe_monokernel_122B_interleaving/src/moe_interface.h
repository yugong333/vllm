#ifndef MOE_INTERFACE_H
#define MOE_INTERFACE_H

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace moe_monokernel {

// Weight quantization granularity
enum class QuantGranularity : uint32_t {
  PER_CHANNEL = 0,  // one scale per row (original)
  BLOCK_WISE = 1,   // one scale per (block_row, block_col) tile
};

template <uint32_t m, uint32_t n, uint32_t k, uint32_t num_experts>
struct MoEDimensions {
  static constexpr uint32_t HIDDEN_STATES = k;
  static constexpr uint32_t K = k;
  static constexpr uint32_t N = n;
  static constexpr uint32_t BS = m;
  static constexpr uint32_t M = m;
  static constexpr uint32_t NUM_EXPERTS = num_experts;

  // Default: per-channel quantization (backward compatible)
  static constexpr QuantGranularity QUANT_GRAN = QuantGranularity::PER_CHANNEL;
  static constexpr uint32_t BLOCK_SCALE_ROW = 0;
  static constexpr uint32_t BLOCK_SCALE_COL = 0;

  struct KernelConfig {
    static constexpr std::uint32_t GRID_SIZE = (2 * N) / 16;
    static constexpr std::uint32_t BLOCK_SIZE = 384;
  };
};

// ── WGMMA variant of the BS8 block-wise kernel (v1 dual-WG K=128) ────────
// Opts into the Hopper wgmma.mma_async fp8 path for Phase 3 (up-proj)
// only.  All other phases (routing, input-quant-setup, down-proj,
// writeback) use the existing mma.sync code.
//
// Layout implications when USE_WGMMA=true:
//   - W_UP_TILE_WGMMA = 128 (each block owns 128 weight rows per K-step:
//     64 for WG0 + 64 for WG1, with WG0 = gate[base..base+31] + up,
//     WG1 = gate[base+32..base+63] + up).
//   - UP_GRID = 2*N / 128 = 8 row-tiles per expert.
//   - With GRID_SIZE = 128, UP_GROUPS = 128 / 8 = 16 experts in parallel
//     (expert_stride = 16).
//   - K_STEP_WGMMA = 128: each K-step consumes K=128 via 4 chained
//     wgmma.mma_async.m64n8k32 instructions per WG.
//   - K_TILES_WGMMA = 2048 / 128 = 16 K-steps per expert per block.
//   - Streaming activation pipeline: bf16 input and fp8 activation tiles
//     are K=128 and double-buffered; weight tile is K=128×128 and
//     single-buffered.  Phase 2's upfront full-K quantization is removed.
//   - SHM layout for `w_wgmma` and `a.fp8_act` uses canonical K-major
//     (8×16-byte core matrices) so WGMMA descriptors reference them
//     directly.
//
// The rest of the kernel (BS8 down-proj) is unchanged.
// ── BS8 WGMMA kernel — Pair_Layout V2 (gate/up paired in M dim) ─────────
// The single BS8 TMA + WGMMA + SWIZZLE_128B variant.
//
// Callers MUST NOT pre-interleave the weights for canonical Major::K
// byte order — the TMA hardware applies the 8-row × 128-byte core-matrix
// XOR swizzle at write time.  Up-projection weights MUST be repacked via
// `interleave_for_tma_wgmma_up_v2` (gate/up PAIR interleave) so a single
// 128×128 TMA fetches one full WGMMA A-tile in the pair layout.  Down-
// projection weights are passed RAW row-major `[E, K, N]`.  Activation B
// operands always use SWIZZLE_NONE.
//
// `KernelConfig::USE_PAIR_LAYOUT = true` opts the up-projection kernel
// into the Pair_Layout register-resident per-expert epilogue (see
// design.md "Up-Projection Gate/Up Pair Layout").  Selection is via the
// `use_pair_layout<Dims>` SFINAE helper in moe_internal.h.
struct Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA {
  static constexpr uint32_t HIDDEN_STATES = 2048;
  static constexpr uint32_t K = 2048;
  static constexpr uint32_t N = 512;
  static constexpr uint32_t BS = 8;
  static constexpr uint32_t M = 8;
  static constexpr uint32_t NUM_EXPERTS = 256;
  static constexpr QuantGranularity QUANT_GRAN = QuantGranularity::BLOCK_WISE;
  static constexpr uint32_t BLOCK_SCALE_ROW = 128;
  static constexpr uint32_t BLOCK_SCALE_COL = 128;
  static constexpr uint32_t UP_SCALE_ROWS =
      (2 * N + BLOCK_SCALE_ROW - 1) / BLOCK_SCALE_ROW;  // 8
  static constexpr uint32_t UP_SCALE_COLS =
      (K + BLOCK_SCALE_COL - 1) / BLOCK_SCALE_COL;  // 16
  static constexpr uint32_t DOWN_SCALE_ROWS =
      (K + BLOCK_SCALE_ROW - 1) / BLOCK_SCALE_ROW;  // 16
  static constexpr uint32_t DOWN_SCALE_COLS =
      (N + BLOCK_SCALE_COL - 1) / BLOCK_SCALE_COL;  // 4
  struct KernelConfig {
    static constexpr std::uint32_t GRID_SIZE = 128;
    static constexpr std::uint32_t BLOCK_SIZE = 384;
    static constexpr bool USE_WGMMA = true;
    static constexpr bool USE_TMA = true;
    static constexpr std::uint32_t K_STEP_DOWN = 256;
    static constexpr std::uint32_t K_STEP_UP = 256;
    // Opts into the Pair_Layout up-projection epilogue, selected via the
    // `use_pair_layout<Dims>` SFINAE helper.
    static constexpr bool USE_PAIR_LAYOUT = true;
  };
};

// ── Qwen3.5-122B BS8 block-FP8 WGMMA+TMA variant ─────────────────────────
// Larger-shape sibling of the 35B variant above:
//   HIDDEN_STATES (K) = 3072  (vs 2048)
//   N                 = 1024  (vs 512)
//   2*N               = 2048  (vs 1024)
//
// Geometry (mirrors the 35B derivation, scaled):
//   * Up-projection: UP_GRID = 2*N / (UP_COL_HALVES * 128).  To keep
//     UP_GROUPS == DOWN_GROUPS (required by the Phase-2b Site-#2
//     expert_barrier), each up-block owns TWO stacked 128-row WGMMA
//     M-atoms (`UP_COL_HALVES = 2`), i.e. 256 interleaved gate/up rows =
//     128 gate features + 128 up features → 128 intermediate-activation
//     features after SiLU.  UP_GRID = 2048 / 256 = 8, UP_GROUPS =
//     GRID_SIZE / UP_GRID = 128 / 8 = 16.
//   * Down-projection: DOWN_COL_TILE = 384 (= HIDDEN_STATES / GRID_SIZE
//     of the col-tile grid: 3072 / 8 = 384).  DOWN_GRID = 3072 / 384 = 8,
//     DOWN_GROUPS = 128 / 8 = 16 == UP_GROUPS.  This requires
//     DOWN_COL_HALVES = 384 / 128 = 3 (vs 2 for 35B).
//   * Activation quantization block = 128 (point 3): each up-block owns
//     exactly 128 intermediate features → one fp8 scale per (block,
//     token).  TEMP_ACT_SCALE_COLS = N / 128 = 8.
//
// SHM budget at K_STEP_DOWN = 128 (NOT 256): the down weight tile is
//   w_down_wgmma[2][DOWN_COL_TILE * K_SUBSTEPS_DOWN][128]
//   = 2 * 384 * 1 * 128 = 96 KB,
// the union dominator.  K_STEP_DOWN = 256 would make it 192 KB and push
// the total past the 228 KB Hopper opt-in cap, so 122B pins
// K_STEP_DOWN = 128.  K_STEP_UP = 128 likewise keeps the up weight tile
// at 2 slots * UP_COL_HALVES(2) * 16 KB = 64 KB and avoids stacking both
// halves AND substeps on the M axis.
struct Dims_BS8_E256_Qwen3_5_122B_BlockFP8_WGMMA_TMA {
  static constexpr uint32_t HIDDEN_STATES = 3072;
  static constexpr uint32_t K = 3072;
  static constexpr uint32_t N = 1024;
  static constexpr uint32_t BS = 8;
  static constexpr uint32_t M = 8;
  static constexpr uint32_t NUM_EXPERTS = 256;
  static constexpr QuantGranularity QUANT_GRAN = QuantGranularity::BLOCK_WISE;
  static constexpr uint32_t BLOCK_SCALE_ROW = 128;
  static constexpr uint32_t BLOCK_SCALE_COL = 128;
  static constexpr uint32_t UP_SCALE_ROWS =
      (2 * N + BLOCK_SCALE_ROW - 1) / BLOCK_SCALE_ROW;  // 16
  static constexpr uint32_t UP_SCALE_COLS =
      (K + BLOCK_SCALE_COL - 1) / BLOCK_SCALE_COL;  // 24
  static constexpr uint32_t DOWN_SCALE_ROWS =
      (K + BLOCK_SCALE_ROW - 1) / BLOCK_SCALE_ROW;  // 24
  static constexpr uint32_t DOWN_SCALE_COLS =
      (N + BLOCK_SCALE_COL - 1) / BLOCK_SCALE_COL;  // 8
  struct KernelConfig {
    static constexpr std::uint32_t GRID_SIZE = 128;
    static constexpr std::uint32_t BLOCK_SIZE = 384;
    static constexpr bool USE_WGMMA = true;
    static constexpr bool USE_TMA = true;
    static constexpr std::uint32_t K_STEP_DOWN = 128;
    static constexpr std::uint32_t K_STEP_UP = 128;
    // Down-proj output col-tile per block (3072 / 8 = 384) →
    // DOWN_COL_HALVES = 3.  Detected via `down_col_tile<Dims>`.
    static constexpr std::uint32_t DOWN_COL_TILE = 384;
    static constexpr bool USE_PAIR_LAYOUT = true;
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
 * moe_kernel_topk()
 */
constexpr size_t get_moe_max_shmem_size();

/**
 * @brief Returns the maximum amount of global scratchpad memory to run
 * moe_kernel_topk()
 */
constexpr size_t get_moe_max_scratchpad_size();

/**
 * @brief W8A8 MoE kernel with configurable top-K routing, scoring function,
 *        and renormalization.
 *
 * Designed for Qwen3.5-30B-A3B FP8 (softmax scoring, top_k=8, 256 experts).
 * Also supports block-wise (128×128) FP8 quantization for Qwen3.5-35B.
 *
 * @param [in] activations_in Input activations. Shape: [M, K]
 * @param [in] token_count Number of active tokens
 * @param [in] router_logits Router logits. Shape: [M, E]
 * @param [in] expert_weights_up Up-projection weights. Shape: [E, 2*N, K]
 * @param [in] expert_scales_up Up-projection scales.
 *             Per-channel: Shape [E, 2*N]
 *             Block-wise:  Shape [E, ceil(2*N/128), ceil(K/128)]
 * @param [in] expert_weights_down Down-projection weights. Shape: [E, K, N]
 * @param [in] expert_scales_down Down-projection scales.
 *             Per-channel: Shape [E, K]
 *             Block-wise:  Shape [E, ceil(K/128), ceil(N/128)]
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
    ScoringFunc scoring_func, bool renormalize,
    __grid_constant__ CUtensorMap const up_weights_desc,
    __grid_constant__ CUtensorMap const activations_desc,
    __grid_constant__ CUtensorMap const down_weights_desc,
    __grid_constant__ CUtensorMap const down_activations_desc);

}  // namespace moe_monokernel

#endif
