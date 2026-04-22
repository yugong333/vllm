#ifndef PTX_UTILS_H
#define PTX_UTILS_H

#pragma once

#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace moe_monokernel {

///////////////////////////////////////////////////////////////////////////////
//
// TMA (Tensor Memory Accelerator) utilities for Hopper (sm_90a)
//
// TMA enables hardware-accelerated async copies from global to shared memory
// using tensor descriptors created on the host.  A single thread issues the
// copy for an entire tile; the hardware handles address generation, swizzling,
// and cache coherence.
//
// Usage:
//   Host side:  create CUtensorMap via tma_create_tensor_map_2d()
//   Device side: one elected thread calls tma_copy_2d_swizzle() per tile
//   All threads: tma_arrive() on the barrier, then tma_wait() before reading
//
// The existing per-thread copy128() path is kept as a fallback and for cases
// where TMA cannot be used (e.g. indirect/gathered token addressing).
//
///////////////////////////////////////////////////////////////////////////////

// ── Barrier helpers for TMA async bulk copies ─────────────────────────────
// TMA uses mbarrier (hardware barrier) instead of cuda::pipeline for
// completion tracking.  We wrap the PTX in thin inline functions.

/**
 * @brief Initialize an mbarrier in shared memory.
 *
 * Must be called by exactly one thread per barrier before any TMA copies
 * reference it.  @p expected_bytes is the total number of bytes that will
 * arrive on this barrier before the next wait.
 */
__device__ static inline void tma_barrier_init(uint64_t* barrier,
                                               uint32_t expected_bytes) {
  uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(bar_addr),
               "r"(expected_bytes));
}

/**
 * @brief Set the expected transaction count on an mbarrier.
 *
 * Called before issuing TMA copies to tell the barrier how many bytes to
 * expect.  Only one thread should call this per phase.
 */
__device__ static inline void tma_barrier_expect_tx(uint64_t* barrier,
                                                    uint32_t tx_count) {
  uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(bar_addr),
      "r"(tx_count));
}

/**
 * @brief Arrive on an mbarrier (non-TMA thread participation).
 *
 * Threads that do not issue TMA copies but need to participate in the
 * barrier call this to signal they have reached the synchronization point.
 */
__device__ static inline void tma_barrier_arrive(uint64_t* barrier) {
  uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" ::"r"(bar_addr));
}

/**
 * @brief Wait for all expected bytes to arrive on an mbarrier.
 *
 * Blocks the calling thread until the barrier's transaction count is
 * satisfied.  @p phase should alternate 0/1 between consecutive uses
 * of the same barrier (parity bit for double-buffering).
 */
__device__ static inline void tma_barrier_wait(uint64_t* barrier,
                                               uint32_t phase) {
  uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  // Spin-wait on the mbarrier phase
  asm volatile(
      "{\n"
      ".reg .pred P;\n"
      "WAIT_LOOP:\n"
      "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
      "@!P bra WAIT_LOOP;\n"
      "}\n" ::"r"(bar_addr),
      "r"(phase));
}

/**
 * @brief Invalidate the mbarrier (call once when done with it).
 */
__device__ static inline void tma_barrier_invalidate(uint64_t* barrier) {
  uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile("mbarrier.inval.shared.b64 [%0];\n" ::"r"(bar_addr));
}

// ── TMA copy intrinsics ──────────────────────────────────────────────────

/**
 * @brief Issue a 2D TMA copy from global to shared memory.
 *
 * Copies a rectangular tile described by the tensor map @p desc at
 * coordinates (@p coord_x, @p coord_y) into shared memory at @p smem_addr.
 * Completion is tracked via the mbarrier at @p barrier.
 *
 * Only ONE thread should call this per tile.  The hardware handles the
 * entire multi-row, multi-column transfer.
 *
 * @param desc      Pointer to CUtensorMap in constant memory or global memory.
 * @param barrier   Pointer to mbarrier in shared memory.
 * @param smem_addr Destination address in shared memory (must be 128B aligned
 *                  for swizzled modes).
 * @param coord_x   Column coordinate in elements (inner/contiguous dimension).
 * @param coord_y   Row coordinate in elements (outer dimension).
 */
__device__ static inline void tma_copy_2d(const void* desc, uint64_t* barrier,
                                          void* smem_addr, int32_t coord_x,
                                          int32_t coord_y) {
  uint64_t gmem_desc = reinterpret_cast<uint64_t>(desc);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_addr));
  uint32_t bar = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes"
      " [%0], [%1, {%3, %4}], [%2];\n" ::"r"(smem),
      "l"(gmem_desc), "r"(bar), "r"(coord_x), "r"(coord_y));
}

/**
 * @brief Issue a 1D TMA copy from global to shared memory.
 *
 * Copies a 1D tile described by the tensor map @p desc at coordinate
 * @p coord_x into shared memory at @p smem_addr.
 */
__device__ static inline void tma_copy_1d(const void* desc, uint64_t* barrier,
                                          void* smem_addr, int32_t coord_x) {
  uint64_t gmem_desc = reinterpret_cast<uint64_t>(desc);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_addr));
  uint32_t bar = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes"
      " [%0], [%1, {%3}], [%2];\n" ::"r"(smem),
      "l"(gmem_desc), "r"(bar), "r"(coord_x));
}

// ── Host-side TMA descriptor creation ────────────────────────────────────
//
// These are host functions that create CUtensorMap descriptors.  They wrap
// the CUDA driver API cuTensorMapEncode* calls.
//
// The tensor map encodes:
//   - Global memory base pointer and tensor dimensions/strides
//   - Shared memory tile (box) dimensions
//   - Element type and swizzle mode
//
// Once created, the descriptor is passed to the kernel (typically via
// kernel arguments or constant memory) and used by tma_copy_2d/1d.

#ifndef __CUDA_ARCH__  // Host-only code

  #include <cuda.h>

/**
 * @brief Create a 2D TMA tensor map for a row-major matrix.
 *
 * @param tensor_map   Output: the CUtensorMap to initialize.
 * @param gmem_ptr     Global memory base pointer (must be 16B aligned).
 * @param elem_type    Element type (e.g. CU_TENSOR_MAP_DATA_TYPE_UINT8 for
 * fp8).
 * @param rows         Number of rows in the global tensor (outer dimension).
 * @param cols         Number of columns in elements (inner/contiguous
 * dimension).
 * @param smem_box_rows  Number of rows per shared memory tile.
 * @param smem_box_cols  Number of columns per shared memory tile (in elements).
 * @param swizzle_mode Swizzle mode for shared memory layout.
 *                     CU_TENSOR_MAP_SWIZZLE_32B matches rotate_col_32.
 */
inline CUresult tma_create_tensor_map_2d(
    CUtensorMap* tensor_map, const void* gmem_ptr,
    CUtensorMapDataType elem_type, uint64_t rows, uint64_t cols,
    uint32_t smem_box_rows, uint32_t smem_box_cols,
    CUtensorMapSwizzle swizzle_mode = CU_TENSOR_MAP_SWIZZLE_32B) {
  // Global tensor dimensions: [rows, cols] in row-major
  // For cuTensorMapEncodeTiled, dimensions are innermost-first:
  //   dim[0] = cols (contiguous), dim[1] = rows
  uint64_t global_dim[2] = {cols, rows};

  // Global strides in bytes (only need stride for dim[1], i.e. row stride)
  // For a row-major matrix: stride = cols * element_size
  uint64_t elem_size;
  switch (elem_type) {
    case CU_TENSOR_MAP_DATA_TYPE_UINT8:
    case CU_TENSOR_MAP_DATA_TYPE_INT8:
      elem_size = 1;
      break;
    case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:
    case CU_TENSOR_MAP_DATA_TYPE_UINT16:
      elem_size = 2;
      break;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
    case CU_TENSOR_MAP_DATA_TYPE_UINT32:
    case CU_TENSOR_MAP_DATA_TYPE_INT32:
      elem_size = 4;
      break;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT64:
    case CU_TENSOR_MAP_DATA_TYPE_UINT64:
    case CU_TENSOR_MAP_DATA_TYPE_INT64:
      elem_size = 8;
      break;
    default:
      elem_size = 1;
      break;
  }
  uint64_t global_stride[1] = {cols * elem_size};  // stride for dim[1] in bytes

  // Shared memory box (tile) dimensions in elements
  uint32_t box_dim[2] = {smem_box_cols, smem_box_rows};

  // Element strides within the box (1 = dense)
  uint32_t elem_stride[2] = {1, 1};

  return cuTensorMapEncodeTiled(
      tensor_map, elem_type,
      /*tensorRank=*/2, const_cast<void*>(gmem_ptr), global_dim, global_stride,
      box_dim, elem_stride,
      /*interleave=*/CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle_mode,
      /*l2Promotion=*/CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      /*oobFill=*/CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

#endif  // !__CUDA_ARCH__

// Native Hopper FP8 MMA (m16n8k32) — single instruction, no conversion
// overhead.
//
// A-matrix (weights): 4 regs, each holding 4 × e4m3 elements (16 total, k=32).
//   {a0, a1, a2, a3} maps to the m16n8k32 layout:
//     a0 → rows [groupID],   k[ 0:15]  (low K half, top 8 rows)
//     a1 → rows [groupID],   k[16:31]  (high K half, top 8 rows)
//     a2 → rows [groupID+8], k[ 0:15]  (low K half, bottom 8 rows)
//     a3 → rows [groupID+8], k[16:31]  (high K half, bottom 8 rows)
//
// B-matrix (activations): 2 regs, each holding 4 × e4m3 elements (8 total,
// k=32).
//   {b0, b1} maps to:
//     b0 → k[ 0:15]
//     b1 → k[16:31]
//
// Callers that previously loaded:
//   w0 = weight[row+0][col+ 0]  w1 = weight[row+8][col+ 0]
//   w2 = weight[row+0][col+16]  w3 = weight[row+8][col+16]
// should call:  mma_fp8_fp8(d, w0, w1, w2, w3, a02, a13, c)
//                              ^^  ^^  ^^  ^^
//                              a0  a1  a2  a3
// PTX m16n8k32 A-matrix register layout:
//   reg0 (a0..a3):   row=groupID,   K=low   (rows 0-7,  K[0:15])
//   reg1 (a4..a7):   row=groupID+8, K=low   (rows 8-15, K[0:15])
//   reg2 (a8..a11):  row=groupID,   K=high  (rows 0-7,  K[16:31])
//   reg3 (a12..a15): row=groupID+8, K=high  (rows 8-15, K[16:31])
__device__ static inline void mma_fp8_fp8(
    float& d0, float& d1, float& d2, float& d3, __nv_fp8x4_e4m3 const& a0,
    __nv_fp8x4_e4m3 const& a1, __nv_fp8x4_e4m3 const& a2,
    __nv_fp8x4_e4m3 const& a3, __nv_fp8x4_e4m3 const& b0,
    __nv_fp8x4_e4m3 const& b1, float const& c0, float const& c1,
    float const& c2, float const& c3) {
#define X2U(x) reinterpret_cast<const unsigned&>(x)
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(X2U(a0)), "r"(X2U(a1)), "r"(X2U(a2)), "r"(X2U(a3)), "r"(X2U(b0)),
        "r"(X2U(b1)), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#undef X2U
}

__device__ static inline void mma_fp8_f16(
    float& d0, float& d1, float& d2, float& d3, __nv_fp8x4_e4m3 const& a0,
    __nv_fp8x4_e4m3 const& a1, __nv_fp8x4_e4m3 const& a2,
    __nv_fp8x4_e4m3 const& a3, __half2 const& b0, __half2 const& b1,
    __half2 const& b2, __half2 const& b3, float const& c0, float const& c1,
    float const& c2, float const& c3) {
#define X2U(x) reinterpret_cast<const unsigned&>(x)
  asm volatile(
      "{"
      ".reg .b16 lo0, lo1, lo2, lo3;\n"
      ".reg .b16 hi0, hi1, hi2, hi3;\n"
      ".reg .b32 al0, al1, al2, al3;\n"
      ".reg .b32 ah0, ah1, ah2, ah3;\n"
      ".reg .b32 t0, t1, t2, t3;\n"
      "mov.b32 {lo0, hi0}, %4;\n"
      "mov.b32 {lo1, hi1}, %5;\n"
      "mov.b32 {lo2, hi2}, %6;\n"
      "mov.b32 {lo3, hi3}, %7;\n"
      "cvt.rn.f16x2.e4m3x2 al0, lo0;\n"
      "cvt.rn.f16x2.e4m3x2 ah0, hi0;\n"
      "cvt.rn.f16x2.e4m3x2 al1, lo1;\n"
      "cvt.rn.f16x2.e4m3x2 ah1, hi1;\n"
      "cvt.rn.f16x2.e4m3x2 al2, lo2;\n"
      "cvt.rn.f16x2.e4m3x2 ah2, hi2;\n"
      "cvt.rn.f16x2.e4m3x2 al3, lo3;\n"
      "cvt.rn.f16x2.e4m3x2 ah3, hi3;\n"
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{t0, t1, t2, t3}, "
      "{al0, al1, al2, al3}, "
      "{%8, %9}, "
      "{%12, %13, %14, %15};\n"
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, "
      "{ah0, ah1, ah2, ah3}, "
      "{%10, %11}, "
      "{t0, t1, t2, t3};\n"
      "}\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(X2U(a0)), "r"(X2U(a1)), "r"(X2U(a2)), "r"(X2U(a3)), "r"(X2U(b0)),
        "r"(X2U(b1)), "r"(X2U(b2)), "r"(X2U(b3)), "f"(c0), "f"(c1), "f"(c2),
        "f"(c3));

#undef X2U
}

__device__ static inline void mma_fp8_bf16(
    float& d0, float& d1, float& d2, float& d3, __nv_fp8x4_e4m3 const& a0,
    __nv_fp8x4_e4m3 const& a1, __nv_fp8x4_e4m3 const& a2,
    __nv_fp8x4_e4m3 const& a3, __nv_bfloat162 const& b0,
    __nv_bfloat162 const& b1, __nv_bfloat162 const& b2,
    __nv_bfloat162 const& b3, float const& c0, float const& c1, float const& c2,
    float const& c3) {
#define X2U(x) reinterpret_cast<const unsigned&>(x)
  asm volatile(
      "{"
      ".reg .b16 lo0, lo1, lo2, lo3;\n"
      ".reg .b16 hi0, hi1, hi2, hi3;\n"
      ".reg .b16 b0, b1, b2, b3;\n"
      ".reg .b16 b4, b5, b6, b7;\n"
      ".reg .b16 b8, b9, b10, b11;\n"
      ".reg .b16 b12, b13, b14, b15;\n"
      ".reg .b32 al0, al1, al2, al3;\n"
      ".reg .b32 ah0, ah1, ah2, ah3;\n"
      ".reg .b32 t0, t1, t2, t3;\n"
      "mov.b32 {lo0, hi0}, %4;\n"
      "mov.b32 {lo1, hi1}, %5;\n"
      "mov.b32 {lo2, hi2}, %6;\n"
      "mov.b32 {lo3, hi3}, %7;\n"
      "cvt.rn.f16x2.e4m3x2 al0, lo0;\n"
      "cvt.rn.f16x2.e4m3x2 ah0, hi0;\n"
      "cvt.rn.f16x2.e4m3x2 al1, lo1;\n"
      "cvt.rn.f16x2.e4m3x2 ah1, hi1;\n"
      "cvt.rn.f16x2.e4m3x2 al2, lo2;\n"
      "cvt.rn.f16x2.e4m3x2 ah2, hi2;\n"
      "cvt.rn.f16x2.e4m3x2 al3, lo3;\n"
      "cvt.rn.f16x2.e4m3x2 ah3, hi3;\n"
      "mov.b32 {b0, b1}, al0;\n"
      "mov.b32 {b2, b3}, ah0;\n"
      "mov.b32 {b4, b5}, al1;\n"
      "mov.b32 {b6, b7}, ah1;\n"
      "mov.b32 {b8, b9}, al2;\n"
      "mov.b32 {b10, b11}, ah2;\n"
      "mov.b32 {b12, b13}, al3;\n"
      "mov.b32 {b14, b15}, ah3;\n"
      "cvt.rn.bf16.f16 b0, b0;\n"
      "cvt.rn.bf16.f16 b1, b1;\n"
      "cvt.rn.bf16.f16 b2, b2;\n"
      "cvt.rn.bf16.f16 b3, b3;\n"
      "cvt.rn.bf16.f16 b4, b4;\n"
      "cvt.rn.bf16.f16 b5, b5;\n"
      "cvt.rn.bf16.f16 b6, b6;\n"
      "cvt.rn.bf16.f16 b7, b7;\n"
      "cvt.rn.bf16.f16 b8, b8;\n"
      "cvt.rn.bf16.f16 b9, b9;\n"
      "cvt.rn.bf16.f16 b10, b10;\n"
      "cvt.rn.bf16.f16 b11, b11;\n"
      "cvt.rn.bf16.f16 b12, b12;\n"
      "cvt.rn.bf16.f16 b13, b13;\n"
      "cvt.rn.bf16.f16 b14, b14;\n"
      "cvt.rn.bf16.f16 b15, b15;\n"
      "mov.b32 al0, {b0, b1};\n"
      "mov.b32 ah0, {b2, b3};\n"
      "mov.b32 al1, {b4, b5};\n"
      "mov.b32 ah1, {b6, b7};\n"
      "mov.b32 al2, {b8, b9};\n"
      "mov.b32 ah2, {b10, b11};\n"
      "mov.b32 al3, {b12, b13};\n"
      "mov.b32 ah3, {b14, b15};\n"
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{t0, t1, t2, t3}, "
      "{al0, al1, al2, al3}, "
      "{%8, %9}, "
      "{%12, %13, %14, %15};\n"
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, "
      "{ah0, ah1, ah2, ah3}, "
      "{%10, %11}, "
      "{t0, t1, t2, t3};\n"
      "}\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(X2U(a0)), "r"(X2U(a1)), "r"(X2U(a2)), "r"(X2U(a3)), "r"(X2U(b0)),
        "r"(X2U(b1)), "r"(X2U(b2)), "r"(X2U(b3)), "f"(c0), "f"(c1), "f"(c2),
        "f"(c3));

#undef X2U
}

__device__ static inline void mma_fp8_tf32(
    float& d0, float& d1, float& d2, float& d3, __nv_fp8x4_e4m3 const& a0,
    __nv_fp8x4_e4m3 const& a1, __nv_fp8x4_e4m3 const& a2,
    __nv_fp8x4_e4m3 const& a3, float4 const& b0, float4 const& b1,
    float const& c0, float const& c1, float const& c2, float const& c3) {
#define X2U(x) reinterpret_cast<const unsigned&>(x)
  asm volatile(
      "{"
      ".reg .b16 lo0, lo1, lo2, lo3;\n"
      ".reg .b16 hi0, hi1, hi2, hi3;\n"
      ".reg .b16 h0, h1, h2, h3;\n"
      ".reg .b16 h4, h5, h6, h7;\n"
      ".reg .b16 h8, h9, h10, h11;\n"
      ".reg .b16 h12, h13, h14, h15;\n"
      ".reg .b32 w0, w1, w2, w3;\n"
      ".reg .b32 w4, w5, w6, w7;\n"
      ".reg .b32 w8, w9, w10, w11;\n"
      ".reg .b32 w12, w13, w14, w15;\n"
      ".reg .b32 al0, al1, al2, al3;\n"
      ".reg .b32 ah0, ah1, ah2, ah3;\n"
      ".reg .b32 t0, t1, t2, t3;\n"
      "mov.b32 {lo0, hi0}, %4;\n"
      "mov.b32 {lo1, hi1}, %5;\n"
      "mov.b32 {lo2, hi2}, %6;\n"
      "mov.b32 {lo3, hi3}, %7;\n"
      "cvt.rn.f16x2.e4m3x2 al0, lo0;\n"
      "cvt.rn.f16x2.e4m3x2 ah0, hi0;\n"
      "cvt.rn.f16x2.e4m3x2 al1, lo1;\n"
      "cvt.rn.f16x2.e4m3x2 ah1, hi1;\n"
      "cvt.rn.f16x2.e4m3x2 al2, lo2;\n"
      "cvt.rn.f16x2.e4m3x2 ah2, hi2;\n"
      "cvt.rn.f16x2.e4m3x2 al3, lo3;\n"
      "cvt.rn.f16x2.e4m3x2 ah3, hi3;\n"
      "mov.b32 {h0, h1}, al0;\n"
      "mov.b32 {h2, h3}, ah0;\n"
      "mov.b32 {h4, h5}, al1;\n"
      "mov.b32 {h6, h7}, ah1;\n"
      "mov.b32 {h8, h9}, al2;\n"
      "mov.b32 {h10, h11}, ah2;\n"
      "mov.b32 {h12, h13}, al3;\n"
      "mov.b32 {h14, h15}, ah3;\n"
      "cvt.f32.f16 w0, h0;\n"
      "cvt.f32.f16 w1, h1;\n"
      "cvt.f32.f16 w2, h2;\n"
      "cvt.f32.f16 w3, h3;\n"
      "cvt.f32.f16 w4, h4;\n"
      "cvt.f32.f16 w5, h5;\n"
      "cvt.f32.f16 w6, h6;\n"
      "cvt.f32.f16 w7, h7;\n"
      "cvt.f32.f16 w8, h8;\n"
      "cvt.f32.f16 w9, h9;\n"
      "cvt.f32.f16 w10, h10;\n"
      "cvt.f32.f16 w11, h11;\n"
      "cvt.f32.f16 w12, h12;\n"
      "cvt.f32.f16 w13, h13;\n"
      "cvt.f32.f16 w14, h14;\n"
      "cvt.f32.f16 w15, h15;\n"
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{t0, t1, t2, t3}, "
      "{w0, w4, w8, w12}, "
      "{%8, %12}, "
      "{%16, %17, %18, %19};\n"
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{t0, t1, t2, t3}, "
      "{w1, w5, w9, w13}, "
      "{%9, %13}, "
      "{t0, t1, t2, t3};\n"
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{t0, t1, t2, t3}, "
      "{w2, w6, w10, w14}, "
      "{%10, %14}, "
      "{t0, t1, t2, t3};\n"
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0, %1, %2, %3}, "
      "{w3, w7, w11, w15}, "
      "{%11, %15}, "
      "{t0, t1, t2, t3};\n"
      "}\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(X2U(a0)), "r"(X2U(a1)), "r"(X2U(a2)), "r"(X2U(a3)), "r"(X2U(b0.x)),
        "r"(X2U(b0.y)), "r"(X2U(b0.z)), "r"(X2U(b0.w)), "r"(X2U(b1.x)),
        "r"(X2U(b1.y)), "r"(X2U(b1.z)), "r"(X2U(b1.w)), "f"(c0), "f"(c1),
        "f"(c2), "f"(c3));

#undef X2U
}

template <typename Target, typename Source>
__device__ static inline void copy128(
    Target& dest, const Source& source,
    cuda::pipeline<cuda::thread_scope_thread>& pipeline) {
  const auto shape4 = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  cuda::memcpy_async(&dest, &source, shape4, pipeline);
}

__device__ inline std::uint32_t rotate_col_32(std::uint32_t col,
                                              std::uint32_t row) {
  std::uint32_t col_base = col & 0xff9f;
  std::uint32_t col_rot = (col + 0x20 * row) & 0x60;
  return col_base | col_rot;
}

}  // namespace moe_monokernel

#endif
