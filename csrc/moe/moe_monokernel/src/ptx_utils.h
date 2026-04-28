#ifndef PTX_UTILS_H
#define PTX_UTILS_H

#pragma once

#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace moe_monokernel {

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
