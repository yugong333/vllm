
#pragma once
#ifndef MOE_INTERNAL_H
  #define MOE_INTERNAL_H

  #ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
    #error Do not include this file directly.
  #endif

  #include "moe_interface.h"

// ── Profiling build flags ──────────────────────────────────────────────────
// Define one of these to isolate the cost of calc vs prefetch warps:
//
//   MONO_PROFILE_SKIP_CALC     : calc warp branches are compiled out.
//                                Only prefetch warps do real work; calc
//                                warps still participate in syncs so the
//                                kernel doesn't deadlock. Outputs are
//                                garbage — useful only for timing.
//
//   MONO_PROFILE_SKIP_PREFETCH : prefetch warp branches are compiled out.
//                                Calc warps run normally but read garbage
//                                (no new data prefetched). Outputs garbage.
//
// Branch bodies in the kernel are wrapped with the corresponding
// #ifndef guards — see the is_prefetch_warp / !is_prefetch_warp sites.

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

  // TMA path uses a tighter outer-axis extent that excludes the 8-row
  // padding above.  The padding exists only to guard against off-by-one
  // writes in the scalar/BS64 paths; the BS8 WGMMA up-proj epilogue only
  // ever writes to rows `[0, BS * SPEC_MAX_TOPK) = [0, 64)` via
  // `sorted_slot`.  `TEMP_ROWS_TMA = BS * SPEC_MAX_TOPK` (= 64 for
  // Qwen3.5-35B) is the value passed to `create_down_activation_tma_desc`
  // as the outer-axis `globalDim`.
  static constexpr uint32_t TEMP_ROWS_TMA = Dims::BS * SPEC_MAX_TOPK;

  #ifdef DEBUG_MOE
  // Debug information passed out. The actual token_indexes are stored in shared
  // memory.
  std::int32_t token_indexes[Dims::BS];
  T_element gemm1[TEMP_ROWS * 2 * Dims::N];
  #endif
  AQ_element activations[Dims::BS]
                        [Dims::HIDDEN_STATES];  //< Quantized activations

  // Up-projection SiLU output (BS64 path only).
  //
  // The BS64 down-projection fetches this bf16 intermediate and does a
  // bf16→fp8 quantization with per-token block-wise scales on the fly
  // before the MMA.  Storing this in bf16 (not fp32) halves the
  // global-memory footprint and async-copy bandwidth.
  A_element temp_bf16[TEMP_ROWS * Dims::N];

  // ── WGMMA-path-only up-proj → down-proj scratchpad ───────────────────
  // Used when `use_wgmma<Dims>::value == true`.  The up-projection
  // epilogue fuses per-64-col fp8 quantization into the SiLU writeback
  // and stores the fp8 activations plus per-(virtual-row, up-block)
  // fp32 scales into these buffers; the WGMMA down-projection consumes
  // them directly (no bf16→fp8 re-quantization pass).
  //
  // Layouts (per Req 2):
  //   temp_fp8        [TEMP_ROWS][N]            fp8
  //   temp_act_scale  [TEMP_ROWS][N / 64]       fp32
  //                   one scale per (virtual_row, up_block_idx)
  //
  // For Qwen3.5-35B (BS=8, top_k=8, N=512): TEMP_ROWS = 72
  //   temp_fp8       = 72 ×  512 × 1 B = 36.0 KB
  //   temp_act_scale = 72 ×    8 × 4 B =  2.25 KB
  //
  // These are separate from temp_bf16 (kept for the scalar path) so
  // non-WGMMA builds are byte-identical.
  static constexpr uint32_t DOWN_ACT_BLOCK_SIZE = 64;
  static_assert(Dims::N % DOWN_ACT_BLOCK_SIZE == 0,
                "Dims::N must be a multiple of 64 for the WGMMA down-proj "
                "per-64-col activation quantization scheme");
  static_assert(Dims::HIDDEN_STATES % 128 == 0,
                "Dims::HIDDEN_STATES must be a multiple of 128 for the "
                "WGMMA down-proj 128-cols-per-block grid layout");
  static constexpr uint32_t TEMP_ACT_SCALE_COLS = Dims::N / DOWN_ACT_BLOCK_SIZE;
  AQ_element temp_fp8[TEMP_ROWS * Dims::N];
  float temp_act_scale[TEMP_ROWS * TEMP_ACT_SCALE_COLS];

  // Byte offset of `temp_fp8` inside `MoEGemmSpec<Dims>`, exposed as a
  // compile-time constant so the host-side TMA wrapper can compute the
  // device pointer to `spec->temp_fp8` from the scratchpad base without
  // reading the struct layout at runtime (spec R9.2).  Consumed by
  // `create_down_activation_tma_desc` when building the down-projection
  // activation descriptor.
  static constexpr size_t TEMP_FP8_OFFSET =
      offsetof(MoEGemmSpec<Dims>, temp_fp8);

  // Per-expert-group partial sum of the WGMMA down-projection output.
  // The `DOWN_GROUPS` expert groups each accumulate the down-proj
  // contribution of their assigned experts into this buffer; a
  // reduction phase (new Phase 5) sums across the DOWN_GROUPS dim
  // into `activations_out[BS][HIDDEN_STATES]` (bf16).
  //
  // Shape: [DOWN_GROUPS][BS][HIDDEN_STATES] fp32.
  // For Qwen3.5-35B (DOWN_GROUPS=8, BS=8, HIDDEN_STATES=2048):
  //   8 × 8 × 2048 × 4 B = 512 KB.
  //
  // Stored fp32 (not bf16) for numerical safety on the 8-way sum.
  //
  // DOWN_GROUPS is computed here independently of MoECoreDims (defined
  // later in this file) so MoEGemmSpec stays self-contained.  The
  // canonical definition lives in MoECoreDims; the two MUST match —
  // MoECoreDims contains a static_assert that cross-checks.
  static constexpr uint32_t DOWN_COL_TILE = 128;
  static constexpr uint32_t DOWN_GRID = Dims::HIDDEN_STATES / DOWN_COL_TILE;
  static constexpr uint32_t DOWN_GROUPS =
      DOWN_GRID == 0 ? 1 : Dims::KernelConfig::GRID_SIZE / DOWN_GRID;
  float down_partial_out[DOWN_GROUPS * Dims::BS * Dims::HIDDEN_STATES];

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

// Number of column-blocks in the down-projection scale tensor. For
// block-wise this is ceil(N / BLOCK_SCALE_COL); for per-channel we return 1
// (unused placeholder so the SHM field is harmlessly tiny).  Mirrors
// shm_up_scale_cols.
template <typename Dims, bool IsBlockWise = shm_is_block_wise<Dims>::value>
struct shm_down_scale_cols {
  static constexpr uint32_t value = 1;
};
template <typename Dims>
struct shm_down_scale_cols<Dims, true> {
  static constexpr uint32_t value = Dims::DOWN_SCALE_COLS;
};

// ── WGMMA opt-in detection ───────────────────────────────────────────────
// `Dims::KernelConfig::USE_WGMMA` is optional; default to false for all
// existing Dims variants so the current mma.sync path stays in use.
// Only the new Dims_BS8_..._WGMMA variant sets USE_WGMMA=true.
template <typename Dims>
struct use_wgmma {
  template <typename D>
  static constexpr auto test(int)
      -> decltype(D::KernelConfig::USE_WGMMA, bool()) {
    return D::KernelConfig::USE_WGMMA;
  }
  template <typename>
  static constexpr bool test(...) {
    return false;
  }
  static constexpr bool value = test<Dims>(0);
};

// ── TMA opt-in detection ────────────────────────────────────────────────
// `Dims::KernelConfig::USE_TMA` is optional; default to false for all
// existing Dims variants so the current cp.async WGMMA path stays in use.
// Only the new Dims_BS8_..._WGMMA_TMA variant sets USE_TMA=true.
template <typename Dims>
struct use_tma {
  template <typename D>
  static constexpr auto test(int)
      -> decltype(D::KernelConfig::USE_TMA, bool()) {
    return D::KernelConfig::USE_TMA;
  }
  template <typename>
  static constexpr bool test(...) {
    return false;
  }
  static constexpr bool value = test<Dims>(0);
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

  // ── WGMMA-only tile dimensions (v1 dual-WG K=128 streaming) ──────────
  // Used by the WGMMA up-proj path (Phase 2-3 when USE_WGMMA=true).
  //
  // Layout of the 128-row weight tile per block:
  //   rows [0  .. 31]  : WG0 gate rows [base    .. base+31]
  //   rows [32 .. 63]  : WG0 up   rows [base+N  .. base+N+31]
  //   rows [64 .. 95]  : WG1 gate rows [base+32 .. base+63]
  //   rows [96 .. 127] : WG1 up   rows [base+N+32 .. base+N+63]
  //
  // Per K-step each WG issues 4 chained wgmma.mma_async.m64n8k32, which
  // together consume K=128. The weight tile is single-buffered; bf16 input
  // and fp8 activation tiles are double-buffered. The streaming pipeline
  // alternates between even half-stages (WGMMA + bf16 prefetch) and odd
  // half-stages (quantize + weight prefetch).
  //
  //   W_UP_TILE_WGMMA  = 128  — M dim of the block's weight tile
  //                             (64 rows per WG × 2 WGs)
  //   W_UP_COLS_WGMMA  = 64   — output columns per block per K-step
  //                             (W_UP_TILE_WGMMA / 2)
  //   K_TILE_WGMMA     = 32   — K width of one m64n8k32 instruction
  //                             (hardware-fixed for fp8)
  //   K_STEP_WGMMA     = 128  — K consumed per outer K-step (=4 × K_TILE_WGMMA)
  //   K_TILES_WGMMA    = K/K_STEP_WGMMA  — outer K iterations per expert
  //   WGMMAS_PER_STEP  = 4    — WGMMAs chained per WG per K-step
  //   UP_GRID_WGMMA    = 2*N / W_UP_TILE_WGMMA  — blocks per expert
  static constexpr std::uint32_t W_UP_TILE_WGMMA = 128;
  static constexpr std::uint32_t W_UP_COLS_WGMMA = W_UP_TILE_WGMMA / 2;
  static constexpr std::uint32_t K_TILE_WGMMA = 32;
  static constexpr std::uint32_t K_STEP_WGMMA = 128;
  static constexpr std::uint32_t WGMMAS_PER_STEP =
      K_STEP_WGMMA / K_TILE_WGMMA;  // 4
  static constexpr std::uint32_t K_TILES_WGMMA =
      Dims::HIDDEN_STATES / K_STEP_WGMMA;
  static constexpr std::uint32_t UP_GRID_WGMMA = 2 * Dims::N / W_UP_TILE_WGMMA;

  static_assert(K_STEP_WGMMA % K_TILE_WGMMA == 0,
                "K_STEP_WGMMA must be a multiple of K_TILE_WGMMA");
  static_assert(!use_wgmma<Dims>::value ||
                    Dims::HIDDEN_STATES % K_STEP_WGMMA == 0,
                "HIDDEN_STATES must be a multiple of 128 for the WGMMA path "
                "(one K-step consumes K=128)");
  static_assert(!use_wgmma<Dims>::value || (2 * Dims::N) % W_UP_TILE_WGMMA == 0,
                "2*N must be a multiple of 128 for the WGMMA path "
                "(one block produces 128 output rows per K-step)");

  // Effective M (row-tile) size of one block's up-proj work — 64 for the
  // WGMMA path, 16 for the scalar path.  Used to compute UP_GRID = 2*N/M.
  static constexpr std::uint32_t W_UP_TILE_EFFECTIVE =
      use_wgmma<Dims>::value ? W_UP_TILE_WGMMA : W_UP_TILE;

  // ── WGMMA down-projection grid layout ────────────────────────────────
  // Each down-block owns 128 output cols within Dims::HIDDEN_STATES, so
  // DOWN_GRID = HIDDEN_STATES / 128 blocks cover one expert's full output.
  // The remaining grid blocks process DIFFERENT experts in parallel:
  // DOWN_GROUPS = GRID_SIZE / DOWN_GRID expert groups each write a
  // partial sum into spec->down_partial_out[DOWN_GROUPS][BS][HIDDEN_STATES],
  // then a reduction phase sums the partials into activations_out.
  //
  // For Qwen3.5-35B (HIDDEN_STATES=2048, GRID_SIZE=128):
  //   DOWN_GRID   = 2048 / 128 = 16 blocks per expert
  //   DOWN_GROUPS = 128  / 16  = 8 expert groups running in parallel
  //
  // DOWN_COL_TILE is fixed at 128 (=W_UP_TILE_WGMMA, matches the
  // two-WG 64-col-per-WG output structure).
  static constexpr std::uint32_t DOWN_COL_TILE = 128;
  static constexpr std::uint32_t DOWN_GRID =
      Dims::HIDDEN_STATES / DOWN_COL_TILE;
  static constexpr std::uint32_t DOWN_GROUPS =
      DOWN_GRID == 0 ? 1 : Dims::KernelConfig::GRID_SIZE / DOWN_GRID;

  static_assert(!use_wgmma<Dims>::value ||
                    Dims::HIDDEN_STATES % DOWN_COL_TILE == 0,
                "HIDDEN_STATES must be a multiple of 128 for the WGMMA "
                "down-projection (one down-block owns 128 output cols)");
  static_assert(!use_wgmma<Dims>::value ||
                    Dims::KernelConfig::GRID_SIZE % DOWN_GRID == 0,
                "GRID_SIZE must be a multiple of DOWN_GRID for the WGMMA "
                "down-projection (expert groups partition the grid)");
  static_assert(!use_wgmma<Dims>::value || DOWN_GROUPS <= Dims::NUM_EXPERTS,
                "DOWN_GROUPS cannot exceed NUM_EXPERTS (each expert group "
                "must process at least one expert)");

  // Cross-check that MoEGemmSpec's mirror of DOWN_GROUPS (computed
  // locally there to avoid a forward reference) matches this one.
  static_assert(MoEGemmSpec<Dims>::DOWN_GROUPS == DOWN_GROUPS,
                "MoEGemmSpec::DOWN_GROUPS must match MoECoreDims::DOWN_GROUPS "
                "— check the DOWN_COL_TILE definition in both places.");

  // GEMM 2 matrix tile dimensions.
  static constexpr std::uint32_t W_DOWN_MMA_TILE = 16;
  static constexpr std::uint32_t W_DOWN_TILE =
      Dims::HIDDEN_STATES / Dims::KernelConfig::GRID_SIZE;
  static constexpr std::uint32_t T_TILE = 8;

  static constexpr unsigned BLOCK_STRIDE = CALC_WARP_COUNT * K_TILE;

  static constexpr unsigned PADDING =
      32;  // this works *slightly* better than 16 due to reduced L2 transfers

  // Row padding (in bytes) for the down-projection fp8 tiles (both the
  // weight tile w[].down and the activation tile a.down). The MMA inner
  // loop reads each row with `byte_offset = row * stride + 4 * (t % 4)`
  // plus a row-stride of `t / 4`. For the reads to hit all 32 banks, we
  // need `(stride_bytes / 4) % 32 >= 4`, i.e. the row stride in dwords
  // must leave at least 4 unique banks per row step so the `t % 4`
  // contribution (0..3) doesn't collide across rows.
  //
  // With N=512 (stride 128 dwords, which is 0 mod 32 → 8-way conflict),
  // adding 16 bytes (4 dwords) gives 132 dwords → 4 mod 32. Combined
  // with t%4 this covers all 32 banks uniformly. PADDING=32 (the global
  // constant above) gives 136 dwords → 8 mod 32, only 4 unique banks
  // per 4 rows → 2-way conflict. So we use DOWN_ROW_PADDING=16 here.
  static constexpr unsigned DOWN_ROW_PADDING = 16;
  static constexpr unsigned K_DIM_PADDED_A = Dims::HIDDEN_STATES;
  static constexpr unsigned K_DIM_PADDED_W = Dims::HIDDEN_STATES;
  static constexpr unsigned K_DIM_HALF_PADDED_A = Dims::HIDDEN_STATES / 2;
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

    // ── TinyDataWGMMA_TMA: SHM layout for the TMA+WGMMA up-proj path ──
    //
    // Used when `use_wgmma<Dims>::value && use_tma<Dims>::value` are both
    // true — the TMA-based activation & weight loading path for the BS8
    // WGMMA up-projection.  Layout is identical to the pre-TMA streaming
    // WGMMA SHM layout with two pairs of 64-bit mbarriers appended:
    //
    //   * bar_w[2]  — 16 B, one weight-tile barrier per double-buffer slot.
    //                 Armed by the TMA launcher with tx_bytes = 16384
    //                 (one 128×128 fp8 weight tile) before the 4-subtile
    //                 `cp.async.bulk.tensor.2d` stripe sequence.  Consumed
    //                 by the WGMMA warps via `mbarrier.try_wait.parity`.
    //   * bar_a[2]  — 16 B, one activation-tile barrier per double-buffer
    //                 slot.  Armed by the launcher with tx_bytes = 2048
    //                 (one 8×128 bf16 tile) before the single activation
    //                 `cp.async.bulk.tensor.2d`.  Consumed by the calc
    //                 warps inside the streaming quantize step.
    //
    // Both barrier arrays are `alignas(16)` so their start addresses are
    // 16-byte aligned as required by the SM90 mbarrier PTX ops.
    struct TinyDataWGMMA_TMA {
      // ── Streaming activation pipeline ──────────────────────────────
      static constexpr uint32_t BF16_IN_K = CoreDims::K_STEP_WGMMA;  // 128
      A_element bf16_in[2][CoreDims::T_TILE][BF16_IN_K];

      static constexpr uint32_t FP8_ACT_K_CHUNK = 16;
      static constexpr uint32_t FP8_ACT_NUM_CHUNKS =
          CoreDims::K_STEP_WGMMA / FP8_ACT_K_CHUNK;  // 128 / 16 = 8

      union {
        // 1024-byte alignment required by SWIZZLE_128B on the down-proj
        // activation TMA: the XOR pattern uses low bits of the SHM
        // address and only behaves consistently within 1024-B-aligned
        // regions.  `fp8_act` and `a_down_wgmma` alias the same 2 KB
        // region (one SWZ128 atom per slot), and the grid.sync between
        // Phase 3 and Phase 4 serializes the two views so the reuse is
        // safe.
        //
        // `fp8_act` keeps the canonical K-major [kc][tok][ki] view —
        // the up-proj activation path stays on SWIZZLE_NONE with
        // software quantize populating SHM.  `a_down_wgmma` uses the
        // token-major [tok][kc][ki] view that matches the CUTLASS
        // Major::K B128 layout after the TMA's SWZ128 XOR.
        alignas(1024)
            AQ_element fp8_act[2][FP8_ACT_NUM_CHUNKS][CoreDims::T_TILE]
                              [FP8_ACT_K_CHUNK];  // 2 KB (up)
        alignas(1024)
            AQ_element a_down_wgmma[2][CoreDims::T_TILE][FP8_ACT_NUM_CHUNKS]
                                   [FP8_ACT_K_CHUNK];  // 2 KB (down)
      };

      static constexpr uint32_t W_WGMMA_M = 128;  // M dim of weight tile
      static constexpr uint32_t W_WGMMA_K = CoreDims::K_STEP_WGMMA;  // 128
      union {
        // 1024-byte alignment required by SWIZZLE_128B: the XOR
        // pattern uses low bits of the SHM address and only behaves
        // consistently within 1024-byte-aligned regions. Both
        // `w_wgmma` and `w_down_wgmma` alias the same SHM bytes, so
        // the alignas applies to both views.
        alignas(1024)
            W_element w_wgmma[2][W_WGMMA_M][W_WGMMA_K];  // 32 KB (up-proj)
        alignas(1024) W_element
            w_down_wgmma[2][W_WGMMA_M][W_WGMMA_K];  // 32 KB (down-proj)
      };

      union {
        A_element orig[CoreDims::T_TILE][CoreDims::K_DIM_PADDED_A];
        A_element bf16_buf[CoreDims::T_TILE][Dims::N];
      } w[2];

      S_element a_down_scale[2][CoreDims::T_TILE][2];

      static constexpr uint32_t W_DOWN_SCALE_COLS =
          shm_down_scale_cols<Dims>::value;
      S_element w_down_scale[2][2][W_DOWN_SCALE_COLS];

      static constexpr uint32_t DOWN_SCALE_TILE_SIZE =
          ((CoreDims::W_DOWN_TILE + 127) / 128) * ((Dims::N + 127) / 128);
      S_element scale[2][DOWN_SCALE_TILE_SIZE + CoreDims::PADDING];

      static constexpr uint32_t UP_SCALE_TILE_SIZE =
          2 * shm_up_scale_cols<Dims>::value;
      S_element up_scale[2][UP_SCALE_TILE_SIZE];

      union {
        T_element up[CoreDims::CALC_WARP_COUNT]
                    [CoreDims::W_UP_TILE * CoreDims::T_TILE];
        T_element
            down[CoreDims::W_DOWN_TILE / 2 + CoreDims::CALC_WARP_COUNT / 2]
                [CoreDims::W_DOWN_MMA_TILE * CoreDims::T_TILE];
        T_element wgmma_out[128][CoreDims::T_TILE];
        T_element down_out[CoreDims::DOWN_COL_TILE][CoreDims::T_TILE];
      } partial_result;

      static constexpr uint32_t OUT_ACCUM_ROW_PAD = 1;
      static constexpr uint32_t OUT_ACCUM_COLS =
          CoreDims::W_DOWN_TILE > CoreDims::DOWN_COL_TILE
              ? CoreDims::W_DOWN_TILE
              : CoreDims::DOWN_COL_TILE;
      T_element out_accum[Dims::BS][OUT_ACCUM_COLS + OUT_ACCUM_ROW_PAD];

      // ── TMA-only extensions ─────────────────────────────────────────
      //
      // Weight-tile mbarriers (one per double-buffer slot).  The launcher
      // arms `bar_w[slot]` with `mbarrier.arrive.expect_tx tx_bytes=16384`
      // before the 4 sub-tile TMAs that populate `w_wgmma[slot]`; WGMMA
      // consumers poll via `mbarrier.try_wait.parity` (R3.1, R3.3, R3.5).
      //
      // Activation-tile mbarriers (one per double-buffer slot).  The
      // launcher arms `bar_a[slot]` with `tx_bytes=2048` before the
      // single TMA that populates `bf16_in[slot]`; the streaming quantize
      // consumers poll via `mbarrier.try_wait.parity` (R3.1, R3.4, R3.6).
      //
      // `alignas(16)` satisfies R11.4 and the 16-byte alignment that the
      // SM90 `mbarrier.*.shared::cta.b64` instructions require.
      alignas(16) uint64_t bar_w[2];  // 16 B
      alignas(16) uint64_t bar_a[2];  // 16 B

      // ── Phase 3 → Phase 4 (expert, token) reorganization tables ─────
      //
      // Added by the `tma-wgmma-down-projection` spec for the Phase-4
      // TMA activation-load path (R11).  Only populated when
      // `use_tma<Dims>::value` is true — on the cp.async reference path
      // these fields are allocated inside the `tiny_wgmma_tma` union
      // variant but never read or written, so non-TMA SHM layouts stay
      // byte-identical (R13.1, R13.2).  These fields live in the
      // `tiny_wgmma_tma` variant only (not in `TinyDataWGMMA`) so the
      // non-TMA WGMMA variant's SHM layout is also unchanged.
      //
      //   expert_slot_start[id]   = first row in spec->temp_fp8 reserved
      //                             for expert `id` under the expert-
      //                             sorted layout produced by the
      //                             Phase-3 epilogue.  Inactive experts
      //                             have the same value as the next
      //                             active expert (zero-width slice).
      //   expert_routed_count[id] = number of routed (tok, k_in_topk)
      //                             pairs that select expert `id`;
      //                             range [0, batch_size * top_k].
      //   sorted_slot[pair]       = destination row in spec->temp_fp8
      //                             for the up-proj SiLU+fp8 writeback,
      //                             where `pair = tok * top_k + k_in_topk`.
      //                             Value = expert_slot_start[eid] +
      //                             intra-expert rank of (tok, k_in_topk).
      //
      // Sizing for BS=8, top_k=8, NUM_EXPERTS=256:
      //   expert_slot_start   : uint16 × 256 = 512 B  (max value < 64)
      //   expert_routed_count : uint8  × 256 = 256 B  (max value ≤ 64)
      //   sorted_slot         : uint8  ×  64 =  64 B  (max value < 64)
      //   Total ≤ 832 B, comfortably inside the 228 KB SHM budget
      //   (R14.1, R14.2, R14.3).
      //
      // Access pattern (R11.1, R11.2, R11.7):
      //   * Phase 3 epilogue reads `sorted_slot[pair]` once per routed
      //     pair to pick the fp8 writeback row.
      //   * Phase 4 launcher reads `expert_slot_start[id]` and
      //     `expert_routed_count[id]` once per expert to parameterize
      //     the bulk activation TMA.
      //   * Phase 4 epilogue (Task 9.6) walks the (tok, k_in_topk) grid
      //     in `topk_ids_flat`, filters by the current expert id, then
      //     derives the intra-expert rank as
      //     `sorted_slot[pair] - expert_slot_start[id]` to index the
      //     SHM slot — no dedicated inverse table needed.
      static constexpr uint32_t MAX_TOPK = 8;
      static constexpr uint32_t MAX_PAIRS = Dims::BS * MAX_TOPK;
      uint16_t expert_slot_start[Dims::NUM_EXPERTS];
      uint8_t expert_routed_count[Dims::NUM_EXPERTS];
      uint8_t sorted_slot[MAX_PAIRS];
    } tiny_wgmma_tma;

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
      // Double-buffered fp8 quantized activations for MMA. Row-padded
      // to avoid bank conflicts in the MMA inner loop — see the comment
      // on DOWN_ROW_PADDING in MoECoreDims.
      AQ_element
          t_fp8[2][CoreDims::T_TILE]
               [Dims::N + CoreDims::DOWN_ROW_PADDING / sizeof(AQ_element)];
      // Per-token per-block activation scales for the fp8 activations.
      static constexpr uint32_t A_DOWN_SCALE_BLOCKS = (Dims::N + 127) / 128;
      S_element t_scale[2][CoreDims::T_TILE][A_DOWN_SCALE_BLOCKS];

      W_element w[2][CoreDims::W_DOWN_TILE]
                 [Dims::N + CoreDims::DOWN_ROW_PADDING / sizeof(W_element)];
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
  // Per-block dynamic SHM budget on Hopper (H100) is 228 KB; we target
  // 224 KB to leave margin for driver overhead.
  static_assert(sizeof(MoE_SHM<Dims>) <= 224 * 1024,
                "MoE_SHM layout exceeds the 224 KB per-block SHM budget.");
  return sizeof(MoE_SHM<Dims>);
}

constexpr size_t get_moe_max_shmem_size() { return sizeof(MoE_SHM<Dims_Max>); }

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
 * @brief Warp-role split used by the BS8 WGMMA up-projection TMA path
 *        (`moe_up_projection_BS8_allexperts_wgmma_tma`).
 *
 * This enum is only meaningful *inside Phase 3* of the monokernel when
 * `USE_WGMMA && USE_TMA` is true. Outside Phase 3 warps 8..11 keep acting as
 * prefetch warps (see `is_prefetch_warp<Dims>()`); the Phase-3 role split is
 * layered on top of the existing warp layout rather than replacing it.
 *
 *   WG0  — warps 0..3 : first WGMMA compute warpgroup (SHM rows [0..63]).
 *   WG1  — warps 4..7 : second WGMMA compute warpgroup (SHM rows [64..127]).
 *   TMA  — warp 8     : hosts the single TMA launcher thread (lane 0) that
 *                       issues every `cp.async.bulk.tensor.2d` and arms every
 *                       mbarrier for the block during Phase 3.
 *   IDLE — warps 9..11: idle during the Phase 3 K-loop; re-used as prefetch
 *                       warps in Phases 1, 2, 4, 5.
 */
enum class WarpRole : uint8_t { WG0, WG1, TMA, IDLE };

/**
 * @brief Returns the Phase-3 `WarpRole` for the calling warp.
 *
 * Only meaningful inside Phase 3 of the TMA-enabled BS8 WGMMA up-projection.
 * Outside Phase 3 the returned role has no semantic meaning; callers that need
 * to know whether the current warp is a prefetch warp should use
 * `is_prefetch_warp<Dims>()` instead.
 */
template <typename Dims>
inline __device__ WarpRole wgmma_tma_warp_role() {
  using CoreDims = MoECoreDims<Dims>;
  unsigned warp = threadIdx.x / CoreDims::THREADS_PER_WARP;
  if (warp < 4u) return WarpRole::WG0;
  if (warp < 8u) return WarpRole::WG1;
  if (warp == 8u) return WarpRole::TMA;
  return WarpRole::IDLE;
}

/**
 * @brief Identifies the single TMA launcher thread (warp 8, lane 0).
 *
 * Returns `true` for exactly one thread in the block. Only meaningful inside
 * Phase 3 (the WGMMA up-projection K-loop) of the TMA kernel variant; that
 * thread is the unique issuer of every `cp.async.bulk.tensor.2d` and
 * `mbarrier.arrive.expect_tx` for the block during Phase 3.
 */
template <typename Dims>
inline __device__ bool is_tma_launcher_thread() {
  using CoreDims = MoECoreDims<Dims>;
  return threadIdx.x == 8u * CoreDims::THREADS_PER_WARP;
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
