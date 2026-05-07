/**
 * Host-side TMA `CUtensorMap` descriptor factories for the BS8 WGMMA
 * up-projection path.
 *
 * This translation unit is pure host code: the only CUDA interaction is
 * through the Driver API (`cuTensorMapEncodeTiled`), which runs on the CPU
 * to populate a 128-byte POD descriptor.  The returned descriptors are later
 * passed to the device as `__grid_constant__ CUtensorMap const` kernel
 * parameters (see spec R5 / R6).
 *
 * Implements the factories declared in `moe_tma.h`.  See the spec
 * requirements:
 *   - R5.1, R5.2, R5.5, R5.6, R1.4 (weight descriptor)
 *   - R5.3, R5.4, R5.5, R5.6, R2.4, R12.4 (activation descriptor)
 *   - R12.1, R12.3 (CUDA 12.0+ / Driver API header)
 *
 * Unlike the other `.cu` files in this directory (which are `#include`d into
 * `moe.cu` for whole-program inlining), this file is a standalone host-side
 * translation unit: it is compiled directly by `moe_wrapper.cu` via the
 * build system.  Hence no `#pragma once` / include guards — this file is
 * never `#include`d by other TUs.
 */

// Torch headers must come first so the `TORCH_CHECK` expansion sees the
// standard <c10/util/Exception.h> helpers.  The monokernel pattern is that
// every TU that needs torch pulls in <torch/all.h> (see moe_wrapper.cu and
// gpt_oss_router_gemm.cu in this tree).
#include <torch/all.h>

#include <cuda.h>

#include "moe_tma.h"

namespace moe_monokernel {

CUtensorMap create_up_weight_tma_desc(const void* weights_ptr,
                                      uint32_t num_experts, uint32_t N,
                                      uint32_t K) {
  // Zero-initialize so any unfilled bytes in the 128 B POD have a defined
  // value — `cuTensorMapEncodeTiled` overwrites the full object on success,
  // but belt-and-suspenders before we return by value.
  CUtensorMap desc{};

  // --- rank-2 descriptor describing the flattened weight tensor ----------
  //
  // GM layout of `expert_weights_up` is `[E, 2*N, K]` row-major fp8, which we
  // view as the 2D matrix `[num_experts * 2 * N, K]` (outer = row, inner = K).
  //
  // `cuTensorMapEncodeTiled` uses innermost-first ordering for every
  // axis array:
  //   globalDim[0]     = innermost (K, fp8 elements)
  //   globalDim[1]     = outer     (num_experts * 2 * N rows)
  //   boxDim[0]        = innermost tile extent (128 K elements)
  //   boxDim[1]        = outer tile extent (32 rows)
  //   elementStrides[i] = 1 means "no sub-sampling" along axis i
  //
  // For `rank = 2`, `globalStrides` is a length-1 array giving the byte
  // stride between successive rows along the outer axis; the innermost
  // stride is implicitly 1 element (= 1 byte for fp8).  Row stride =
  // `K * sizeof(fp8) = K` bytes (R5.2).
  constexpr uint32_t kRank = 2;
  uint64_t global_dim[kRank] = {
      static_cast<uint64_t>(K),
      static_cast<uint64_t>(num_experts) * 2ULL * static_cast<uint64_t>(N),
  };
  uint64_t global_strides[kRank - 1] = {
      static_cast<uint64_t>(K),  // K bytes per row (fp8 = 1 B/element)
  };
  uint32_t box_dim[kRank] = {
      /*K-inner*/ 128u,
      /*rows   */ 32u,
  };
  uint32_t element_strides[kRank] = {1u, 1u};

  const CUresult res = cuTensorMapEncodeTiled(
      &desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, kRank,
      // The Driver API takes a non-const void*; the descriptor only reads
      // the pointer value, it never writes through it.
      const_cast<void*>(weights_ptr), global_dim, global_strides, box_dim,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  // R5.5: on failure, raise a TORCH_CHECK naming the failing tensor so the
  // Python stack trace points directly at "up-projection weights".
  TORCH_CHECK(res == CUDA_SUCCESS,
              "cuTensorMapEncodeTiled failed for up-projection weights: "
              "CUresult=",
              static_cast<int>(res), " (num_experts=", num_experts, ", N=", N,
              ", K=", K, ")");

  return desc;
}

CUtensorMap create_activations_tma_desc(const void* activations_ptr,
                                        uint32_t batch_size_cap,
                                        uint32_t K_hidden) {
  // Zero-initialize the POD so any bytes not explicitly written by the
  // Driver API have a defined value before we return by value.
  CUtensorMap desc{};

  // --- rank-2 descriptor describing the bf16 activation tensor ----------
  //
  // GM layout of `activations_in` is `[BS, K_hidden]` row-major bf16, which
  // we view as a 2D matrix with innermost = K_hidden, outer = batch row.
  //
  // `cuTensorMapEncodeTiled` uses innermost-first ordering:
  //   globalDim[0]     = innermost (K_hidden, bf16 elements)
  //   globalDim[1]     = outer     (batch_size_cap rows)
  //   boxDim[0]        = innermost tile extent (128 bf16 elements)
  //   boxDim[1]        = outer tile extent (8 tokens)
  //   elementStrides[i] = 1 means "no sub-sampling" along axis i
  //
  // For `rank = 2`, `globalStrides` is a length-1 array giving the byte
  // stride between successive rows along the outer axis; for bf16 (2 B/elem)
  // the row stride is `K_hidden * 2` bytes (R5.4, R12.4).
  constexpr uint32_t kRank = 2;
  uint64_t global_dim[kRank] = {
      static_cast<uint64_t>(K_hidden),
      static_cast<uint64_t>(batch_size_cap),
  };
  uint64_t global_strides[kRank - 1] = {
      static_cast<uint64_t>(K_hidden) * 2ULL,  // bf16 = 2 B/element
  };
  uint32_t box_dim[kRank] = {
      /*K-inner*/ 128u,
      /*tokens */ 8u,
  };
  uint32_t element_strides[kRank] = {1u, 1u};

  const CUresult res = cuTensorMapEncodeTiled(
      &desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, kRank,
      // The Driver API takes a non-const void*; the descriptor only reads
      // the pointer value, it never writes through it.
      const_cast<void*>(activations_ptr), global_dim, global_strides, box_dim,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  // R5.5: on failure, raise a TORCH_CHECK naming the failing tensor so the
  // Python stack trace points directly at "activations".
  TORCH_CHECK(res == CUDA_SUCCESS,
              "cuTensorMapEncodeTiled failed for activations: CUresult=",
              static_cast<int>(res), " (batch_size_cap=", batch_size_cap,
              ", K_hidden=", K_hidden, ")");

  return desc;
}

CUtensorMap create_down_weight_tma_desc(const void* weights_ptr,
                                        uint32_t num_experts, uint32_t K,
                                        uint32_t N) {
  // Zero-initialize so any unfilled bytes in the 128 B POD have a defined
  // value — `cuTensorMapEncodeTiled` overwrites the full object on success,
  // but belt-and-suspenders before we return by value.
  CUtensorMap desc{};

  // --- rank-2 descriptor describing the flattened down-weight tensor ----
  //
  // GM layout of `expert_weights_down` is `[E, K, N]` row-major fp8, which
  // we view as the 2D matrix `[num_experts * K, N]` (outer = flattened
  // `expert_id * K + k_row`, inner = N).
  //
  // NOTE on axis ordering (R6.2, see header doc for full rationale):
  //   - Up-proj weight:   innermost = K (reduction), outer = 2*N row index.
  //   - Down-proj weight: innermost = N (reduction), outer = K row index.
  // The two reduction axes are the `B` operand of the respective WGMMAs,
  // so both descriptors place the reduction on the innermost axis to keep
  // per-tile K/N fetches contiguous.
  //
  // `cuTensorMapEncodeTiled` uses innermost-first ordering for every
  // axis array:
  //   globalDim[0]     = innermost (N, fp8 elements)
  //   globalDim[1]     = outer     (num_experts * K rows)
  //   boxDim[0]        = innermost tile extent (128 N elements)
  //   boxDim[1]        = outer tile extent (128 output rows per tile —
  //                      covers the full 128×128 WGMMA weight tile in
  //                      one bulk issue per R1.2)
  //   elementStrides[i] = 1 means "no sub-sampling" along axis i
  //
  // For `rank = 2`, `globalStrides` is a length-1 array giving the byte
  // stride between successive rows along the outer axis; row stride =
  // `N * sizeof(fp8) = N` bytes (R6.2).
  constexpr uint32_t kRank = 2;
  uint64_t global_dim[kRank] = {
      static_cast<uint64_t>(N),
      static_cast<uint64_t>(num_experts) * static_cast<uint64_t>(K),
  };
  uint64_t global_strides[kRank - 1] = {
      static_cast<uint64_t>(N),  // N bytes per row (fp8 = 1 B/element)
  };
  uint32_t box_dim[kRank] = {
      /*N-inner*/ 128u,
      /*rows   */ 128u,
  };
  uint32_t element_strides[kRank] = {1u, 1u};

  const CUresult res = cuTensorMapEncodeTiled(
      &desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, kRank,
      // The Driver API takes a non-const void*; the descriptor only reads
      // the pointer value, it never writes through it.
      const_cast<void*>(weights_ptr), global_dim, global_strides, box_dim,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  // R6.5: on failure, raise a TORCH_CHECK naming the failing tensor so the
  // Python stack trace points directly at "down-projection weights".
  TORCH_CHECK(
      res == CUDA_SUCCESS,
      "cuTensorMapEncodeTiled failed for down-projection weights: CUresult=",
      static_cast<int>(res), " (num_experts=", num_experts, ", K=", K,
      ", N=", N, ")");

  return desc;
}

CUtensorMap create_down_activation_tma_desc(const void* activations_ptr,
                                            uint32_t temp_rows, uint32_t N) {
  // Zero-initialize the POD so any bytes not explicitly written by the
  // Driver API have a defined value before we return by value.
  CUtensorMap desc{};

  // --- rank-2 descriptor describing K-chunk-major `spec->temp_fp8` -------
  //
  // On the TMA path the Phase-3 up-proj epilogue writes temp_fp8 in
  // K-chunk-major layout instead of the cp.async path's token-major
  // [TEMP_ROWS, N]. The byte offset for element (kc, row, ki) is:
  //
  //   byte_off(kc, row, ki) = kc * (TEMP_ROWS * 16) + row * 16 + ki
  //
  // where kc in [0, N/16) is the K-chunk index, row in [0, TEMP_ROWS) is
  // the sorted_slot row, and ki in [0, 16) is the K-inner index within
  // the chunk. Total bytes = N/16 * TEMP_ROWS * 16 = TEMP_ROWS * N —
  // same footprint as token-major, just with the K-chunk and row axes
  // swapped.
  //
  // The WGMMA B-operand for m64n8k32 with N=8 expects K-chunk-major SHM
  // a_down_wgmma[kc][tok][ki] at byte offset kc*128 + tok*16 + ki. By
  // storing temp_fp8 in the matching K-chunk-major layout, a single
  // boxDim=(128, 8) TMA per K-step writes the full 1 KB activation slab
  // directly into WGMMA-canonical SHM form — no per-K-chunk sub-tile
  // splitting is needed (the layout naturally aligns).
  //
  // The descriptor views the [N/16, TEMP_ROWS * 16] flat matrix with
  // innermost = per-kc-slab byte offset, outer = kc:
  //
  //   globalDim[0]     = TEMP_ROWS * 16 (innermost; one kc-slab in bytes)
  //   globalDim[1]     = N / 16         (outer; K-chunk count)
  //   boxDim[0]        = 128            (8 tokens * 16 ki bytes per issue)
  //   boxDim[1]        = 8              (8 K-chunks per K-step = K=128)
  //   elementStrides[i] = 1
  //
  // Coordinates from the kernel side for one K-step at expert id:
  //   coord0 = expert_slot_start[id] * 16
  //   coord1 = k_start / 16
  //
  // The TMA writes the 8x128 box row-major to SHM at byte offset
  // row*128 + col, which maps directly to a_down_wgmma[kc][tok][ki]:
  //   kc  = row
  //   tok = col / 16
  //   ki  = col % 16
  //
  // For rank = 2, globalStrides is a length-1 array giving the byte
  // stride between successive K-chunks along the outer axis. Since each
  // K-chunk occupies exactly TEMP_ROWS * 16 contiguous bytes, the stride
  // equals globalDim[0] (R6.4).
  constexpr uint32_t kRank = 2;
  const uint64_t kc_slab_bytes = static_cast<uint64_t>(temp_rows) * 16ULL;
  uint64_t global_dim[kRank] = {
      kc_slab_bytes,
      static_cast<uint64_t>(N) / 16ULL,
  };
  uint64_t global_strides[kRank - 1] = {
      kc_slab_bytes,  // TEMP_ROWS * 16 bytes per K-chunk slab
  };
  uint32_t box_dim[kRank] = {
      /*inner*/ 128u,  // 8 tokens * 16 ki bytes
      /*kc   */ 8u,    // 8 K-chunks = full 128-K step
  };
  uint32_t element_strides[kRank] = {1u, 1u};

  const CUresult res = cuTensorMapEncodeTiled(
      &desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, kRank,
      // The Driver API takes a non-const void*; the descriptor only reads
      // the pointer value, it never writes through it.
      const_cast<void*>(activations_ptr), global_dim, global_strides, box_dim,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  // R6.5: on failure, raise a TORCH_CHECK naming the failing tensor so the
  // Python stack trace points directly at "down-projection activations".
  TORCH_CHECK(res == CUDA_SUCCESS,
              "cuTensorMapEncodeTiled failed for down-projection "
              "activations: CUresult=",
              static_cast<int>(res), " (temp_rows=", temp_rows, ", N=", N, ")");

  return desc;
}

}  // namespace moe_monokernel
