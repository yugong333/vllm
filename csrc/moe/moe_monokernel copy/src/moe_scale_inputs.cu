
#pragma once
#ifndef MOE_SCALE_INPUTS_CU
  #define MOE_SCALE_INPUTS_CU

  #ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
    #error Do not include this file directly.
  #endif

  #include <cstdint>
  #include <cstring>

  #include <cuda/pipeline>
  #include <cuda_bf16.h>
  #include <cooperative_groups.h>

  #include "moe_interface.h"
  #include "moe_internal.h"
  #include "ptx_utils.h"

namespace moe_monokernel {

/**
 * @brief Sets NaNs (positive or negative, any payload) to 0.0 (bit-pattern all
 * 0).
 *
 * Values other than NaN remain unchanged.
 */
__device__ static __forceinline__ __nv_bfloat162
mask_NaNs_to_zero(__nv_bfloat162 xs) {
  return type_pun<__nv_bfloat162>(type_pun<uint32_t>(xs) & __heq2_mask(xs, xs));
}

// Internal linkage
namespace {

/**
 * @brief Struct for keeping a chunk of 8 bfloat16 in registers.
 */
struct BF16x8 {
  float4 raw;  // Storage for 8 BFloat16 values. Never accessed as fp32.

  /**
   * @brief Loads a BF16x8 from memory.
   *
   * The address @p a must be BF16x8 aligned.
   */
  __device__ static BF16x8 load(const A_element* a) {
    assert(reinterpret_cast<uintptr_t>(a) % 16 == 0);
    BF16x8 val{*reinterpret_cast<const float4*>(a)};
    return val;
  }

  /**
   * @brief Stores a BF16x8 to memory.
   *
   * The address @p a must be BF16x8 aligned.
   */
  __device__ void store_to(A_element* a) {
    assert(reinterpret_cast<uintptr_t>(a) % 16 == 0);
    *reinterpret_cast<float4*>(a) = raw;
  }

  /**
   * @brief Returns bfloat16 0 and 1 as pair.
   */
  __device__ __nv_bfloat162 first_pair() const {
    return type_pun<__nv_bfloat162>(raw.x);
  }
  /**
   * @brief Returns bfloat16 2 and 3 as pair.
   */
  __device__ __nv_bfloat162 second_pair() const {
    return type_pun<__nv_bfloat162>(raw.y);
  }
  /**
   * @brief Returns bfloat16 5 and 5 as pair.
   */
  __device__ __nv_bfloat162 third_pair() const {
    return type_pun<__nv_bfloat162>(raw.z);
  }
  /**
   * @brief Returns bfloat16 6 and 7 as pair.
   */
  __device__ __nv_bfloat162 fourth_pair() const {
    return type_pun<__nv_bfloat162>(raw.w);
  }

  /**
   * @brief Converts 8 Bfloat16 values to FP8 E4M3 in accordance with vLLM's MoE
   * activation quantization.
   *
   * Scales, clamps (incl. NaN replacement), rounds and converts each BFloat16
   * to FP8 E4M3. Scaling is done with float accuracy. Clamping and saturation
   * are implemented via __NVSATFINITE semantics of the FP8 conversion.
   *
   * @param scale Scaling factor to use.
   * @returns Eight FP8 E4M3 packed into a uint64_t
   */
  __device__ uint64_t to_fp8x8(float scale) const {
    // We do not need to actually clamp. Clamping is handled implicitly by the
    // satfinite semantics of the float->e4m3 conversion. We only need to
    // swallow NaNs. Here, we set them to 0.
    __nv_bfloat162 bf0 = mask_NaNs_to_zero(first_pair());
    __nv_bfloat162 bf1 = mask_NaNs_to_zero(second_pair());
    __nv_bfloat162 bf2 = mask_NaNs_to_zero(third_pair());
    __nv_bfloat162 bf3 = mask_NaNs_to_zero(fourth_pair());

    float2 f0 = __bfloat1622float2(bf0);
    float2 f1 = __bfloat1622float2(bf1);
    float2 f2 = __bfloat1622float2(bf2);
    float2 f3 = __bfloat1622float2(bf3);

    __nv_fp8x4_e4m3 converted0{
        float4{f0.x * scale, f0.y * scale, f1.x * scale, f1.y * scale}};
    __nv_fp8x4_e4m3 converted1{
        float4{f2.x * scale, f2.y * scale, f3.x * scale, f3.y * scale}};

    return type_pun<uint32_t>(converted0) |
           ((uint64_t)type_pun<uint32_t>(converted1) << 32);
  }
};

}  // namespace

/**
 * @brief Fetches all activations for a single token from global to shared
 * memory.
 *
 * Fetches all @c Dims::HIDDEN_STATES activations for a single token
 * asynchronously from global to shared memory. Before accessing the shared
 * memory values, wait for the transfer via @c pipe .
 *
 * @param source Pointer to the first activation of the token (in global
 * memory).
 * @param dest Pointer to the first activation of the token (in shared memory).
 * @param pipe CUDA pipeline to execute the transfer in
 */
template <typename Dims>
__device__ void moe_fetch_activation_async(
    const A_element* __restrict__ source, A_element* __restrict dest,
    cuda::pipeline<cuda::thread_scope_thread>& pipe) {
  using CoreDims = MoECoreDims<Dims>;

  const std::uint32_t thread = get_thread<Dims>();
  const std::uint32_t warp =
      get_any_warp<Dims>();  // we run this at the beginning of our kernel with
                             // 1 warp per input token
  const std::uint32_t thread_chunk_size = 16 / sizeof(*source);
  const std::uint32_t chunk_size =
      CoreDims::THREADS_PER_WARP * thread_chunk_size;

  pipe.producer_acquire();
  for (std::uint32_t k = thread * thread_chunk_size; k < Dims::HIDDEN_STATES;
       k += chunk_size) {
    copy128(dest[rotate_col_32(k, warp)], source[k], pipe);
  }
  pipe.producer_commit();
}

/**
 * @brief Quantizes activation values for a single token (BS8 path).
 *
 * Reads bf16 activations from shared memory, computes act_scale = max(|x|)/448,
 * writes fp8 quantized activations to @p activation_out, and returns act_scale.
 *
 * @param [in]  activation_in  bf16 activations in shared memory (16-byte
 * aligned)
 * @param [out] activation_out fp8 quantized activations (8-byte aligned)
 * @returns act_scale = max(|x|) / 448  (only valid on thread 0 of the warp)
 */
template <typename Dims>
__device__ float moe_scale_activation_BS8(
    const A_element* __restrict__ activation_in,
    AQ_element* __restrict__ activation_out) {
  static_assert(Dims::BS <= 8, "This function is only for use with BS up to 8");
  assert((uintptr_t)activation_in != (uintptr_t)activation_out);
  static_assert(Dims::HIDDEN_STATES * sizeof(A_element) % 16 == 0);
  // suppose that act dtype is bf16
  // one row can fit into bfloat16x8
  static_assert(Dims::HIDDEN_STATES % 8 == 0);
  // make sure the input address is 16 byte aligned for 128 bit loading
  // make sure the output address is 8 byte aligned for 64 bit saving
  assert((uintptr_t)activation_in % 16 == 0);
  assert((uintptr_t)activation_out % 8 == 0);

  using CoreDims = MoECoreDims<Dims>;
  const std::uint32_t thread = get_thread<Dims>();
  const std::uint32_t thread_chunk_size =
      sizeof(BF16x8) / sizeof(*activation_in);
  const std::uint32_t chunk_size =
      CoreDims::THREADS_PER_WARP * thread_chunk_size;

  constexpr float FP8_MAX = 448.f;
  constexpr float FP8_MAX_INV = 1.0f / 448.f;

  // find max absolute value across all elements
  // potential back conflcts for 4 SHM read,
  // overhead is small because this is one pass and 4-seriel only
  __nv_bfloat162 m0{0.f, 0.f}, m1{0.f, 0.f}, m2{0.f, 0.f}, m3{0.f, 0.f};
  for (std::uint32_t k = thread * thread_chunk_size; k < Dims::HIDDEN_STATES;
       k += chunk_size) {
    BF16x8 chunk = BF16x8::load(activation_in + k);
    m0 = __hmax2(m0, __habs2(chunk.first_pair()));
    m1 = __hmax2(m1, __habs2(chunk.second_pair()));
    m2 = __hmax2(m2, __habs2(chunk.third_pair()));
    m3 = __hmax2(m3, __habs2(chunk.fourth_pair()));
  }
  m0 = __hmax2(__hmax2(m0, m1), __hmax2(m2, m3));
  float m = (float)__hmax(m0.x, m0.y);
  m = warp_reduce_max_float(m);
  if (m < __FLT_MIN__) m = 1.f;

  float act_scale = m * FP8_MAX_INV;  // = max/448
  float inv_scale = FP8_MAX / m;      // = 448/max

  // quantize: x_fp8 = clamp(x_bf16 * inv_scale)
  uint64_t* out8 = reinterpret_cast<uint64_t*>(activation_out);
  for (std::uint32_t k = thread * thread_chunk_size; k < Dims::HIDDEN_STATES;
       k += chunk_size) {
    out8[k / 8] = BF16x8::load(activation_in + k).to_fp8x8(inv_scale);
  }

  return act_scale;  // valid on all threads (warp_reduce_max broadcasts)
}

namespace detail {

/**
 * @brief Scales activation values for a single token (BS > 8 path).
 *
 * Quantizes one token's activations from global memory to fp8, computing
 * act_scale = max(|x|) / 448 and writing it to @p act_scale_out.
 * No routing weight folding — act_scale is stored separately.
 */
template <typename Dims>
__device__ static void moe_scale_activation_BSx_chunk(
    const A_element* __restrict__ activation_in, A_element* __restrict__ temp,
    AQ_element* __restrict__ activation_out,
    float& __restrict__ act_scale_out) {
  assert((uintptr_t)activation_in != (uintptr_t)temp);
  assert((uintptr_t)activation_out != (uintptr_t)temp);
  assert((uintptr_t)activation_in != (uintptr_t)activation_out);
  using CoreDims = MoECoreDims<Dims>;

  const std::uint32_t thread = get_thread<Dims>();
  const std::uint32_t thread_chunk_size =
      sizeof(BF16x8) / sizeof(*activation_in);
  const std::uint32_t chunk_size =
      CoreDims::THREADS_PER_WARP * thread_chunk_size;

  constexpr float FP8_MAX = 448.f;
  constexpr float FP8_MAX_INV = 1.0f / 448.f;

  __nv_bfloat162 m0{0.0f, 0.0f}, m1{0.0f, 0.0f}, m2{0.0f, 0.0f}, m3{0.0f, 0.0f};
  for (std::uint32_t k = thread * thread_chunk_size; k < Dims::HIDDEN_STATES;
       k += chunk_size) {
    BF16x8 chunk = BF16x8::load(activation_in + k);
    chunk.store_to(&temp[k]);
    m0 = __hmax2(m0, __habs2(chunk.first_pair()));
    m1 = __hmax2(m1, __habs2(chunk.second_pair()));
    m2 = __hmax2(m2, __habs2(chunk.third_pair()));
    m3 = __hmax2(m3, __habs2(chunk.fourth_pair()));
  }

  m0 = __hmax2(__hmax2(m0, m1), __hmax2(m2, m3));
  float m = (float)__hmax(m0.x, m0.y);
  m = warp_reduce_max_float(m);
  if (m < __FLT_MIN__) m = 1.f;

  float scale = m * FP8_MAX_INV;  // act_scale = max/448
  float inv_scale = FP8_MAX / m;

  uint64_t* activation_out8 = reinterpret_cast<uint64_t*>(activation_out);
  for (std::uint32_t k = thread * thread_chunk_size; k < Dims::HIDDEN_STATES;
       k += chunk_size) {
    BF16x8 chunk = BF16x8::load(activation_in + k);
    activation_out8[k / 8] = chunk.to_fp8x8(inv_scale);
  }

  if (thread == 0) act_scale_out = scale;
}

}  // namespace detail

/**
 * @brief Quantizes activations for all tokens (BS > 8 path).
 *
 * Writes fp8 quantized activations to spec->activations[i] and
 * act_scale per token to shmem->act_scale[i].
 * This function is collective across all CUDA blocks.
 */
template <typename Dims>
__device__ void moe_scale_activation_BSx(
    const A_element* __restrict__ activations_in, std::uint32_t token_count,
    MoEGemmSpec<Dims>* __restrict__ spec, MoE_SHM<Dims>* __restrict__ shmem) {
  static_assert(Dims::BS > 8,
                "BS=8 is handled by its own kernel. Do not use "
                "moe_scale_inputs for BS<=8");
  static_assert(Dims::HIDDEN_STATES * sizeof(A_element) % 16 == 0,
                "Next token activation will not be properly aligned.");
  static_assert(
      Dims::HIDDEN_STATES % 8 == 0,
      "Next quantized token activation will not be properly aligned.");

  assert((uintptr_t)activations_in % 16 == 0);
  assert((uintptr_t)spec->activations % 8 == 0);

  using CoreDims = MoECoreDims<Dims>;

  if (is_calc_warp<Dims>()) {
    const std::uint32_t global_warp_count =
        gridDim.x * CoreDims::CALC_WARP_COUNT;
    const std::uint32_t warp = get_calc_warp<Dims>();
    const std::uint32_t global_warp =
        blockIdx.x * CoreDims::CALC_WARP_COUNT + warp;

    for (std::uint32_t i = global_warp; i < token_count;
         i += global_warp_count) {
      detail::moe_scale_activation_BSx_chunk<Dims>(
          activations_in + i * Dims::HIDDEN_STATES, shmem->u.rescale.a[warp],
          spec->activations[i], spec->act_scale[i]);
    }
  }

  // spec->act_scale is written by different blocks — make visible to all
  cooperative_groups::this_grid().sync();

  // copy act_scale into shmem for fast per-token access during up-projection
  for (uint32_t i = threadIdx.x; i < token_count; i += blockDim.x)
    shmem->act_scale[i] = spec->act_scale[i];

  __syncthreads();
}

}  // namespace moe_monokernel

#endif
