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
// ── GENERATED Dims structs ───────────────────────────────────────────────
// The per-shape Dims_* structs (and their legacy `using` aliases) are emitted
// from csrc/moe/moe_monokernel/shapes.json by tools/gen_shapes.py.  Each
// shape's base KernelConfig knobs come from its config[0] (the shipped
// default), so the base Dims is byte-identical to the config-0 tunable
// instantiation.  To add/edit a shape: edit shapes.json, run gen_shapes.py,
// rebuild.  DO NOT hand-edit the generated file.
#include "../generated/dims_generated.inc"

// ── Tunable-config Dims wrapper ──────────────────────────────────────────
// `DimsTunable<Base, GRID, DCT, KUP, KDN, SLOTS>` clones the SHAPE of `Base`
// (HIDDEN_STATES, K, N, BS, NUM_EXPERTS, quant granularity + all the derived
// scale-row/col counts) but OVERRIDES the tunable KernelConfig knobs.  One
// `moe_kernel_topk<DimsTunable<...>>` is instantiated per entry in the
// per-shape config table below; the runtime dispatcher (moe_wrapper.cu)
// picks one by `config_id`.  config_id 0 is always the shipped default and
// is byte-identical to the bare `Base` Dims (same knob values), so the
// existing named ops and the config-0 tunable op produce the same kernel.
//
// Only the SIX coupled-mode tunables are exposed here (GRID_SIZE,
// DOWN_COL_TILE, K_STEP_UP, K_STEP_DOWN, UP_W_SLOTS; UP_COL_HALVES is
// derived from DCT).  Phase-B decoupled configs (independent up/down groups)
// need the barrier-set rework in task 10 and are NOT expressible here yet.
template <typename Base, std::uint32_t GRID, std::uint32_t DCT,
          std::uint32_t KUP, std::uint32_t KDN, std::uint32_t SLOTS>
struct DimsTunable {
  static constexpr uint32_t HIDDEN_STATES = Base::HIDDEN_STATES;
  static constexpr uint32_t K = Base::K;
  static constexpr uint32_t N = Base::N;
  static constexpr uint32_t BS = Base::BS;
  static constexpr uint32_t M = Base::M;
  static constexpr uint32_t NUM_EXPERTS = Base::NUM_EXPERTS;
  static constexpr QuantGranularity QUANT_GRAN = Base::QUANT_GRAN;
  static constexpr uint32_t BLOCK_SCALE_ROW = Base::BLOCK_SCALE_ROW;
  static constexpr uint32_t BLOCK_SCALE_COL = Base::BLOCK_SCALE_COL;
  static constexpr uint32_t UP_SCALE_ROWS = Base::UP_SCALE_ROWS;
  static constexpr uint32_t UP_SCALE_COLS = Base::UP_SCALE_COLS;
  static constexpr uint32_t DOWN_SCALE_ROWS = Base::DOWN_SCALE_ROWS;
  static constexpr uint32_t DOWN_SCALE_COLS = Base::DOWN_SCALE_COLS;
  struct KernelConfig {
    static constexpr std::uint32_t GRID_SIZE = GRID;
    static constexpr std::uint32_t BLOCK_SIZE = Base::KernelConfig::BLOCK_SIZE;
    static constexpr bool USE_WGMMA = Base::KernelConfig::USE_WGMMA;
    static constexpr bool USE_TMA = Base::KernelConfig::USE_TMA;
    static constexpr std::uint32_t K_STEP_DOWN = KDN;
    static constexpr std::uint32_t K_STEP_UP = KUP;
    static constexpr std::uint32_t DOWN_COL_TILE = DCT;
    static constexpr std::uint32_t UP_W_SLOTS = SLOTS;
    static constexpr bool USE_PAIR_LAYOUT = Base::KernelConfig::USE_PAIR_LAYOUT;
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
