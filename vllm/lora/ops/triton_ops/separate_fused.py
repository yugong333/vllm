# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Separate fused: absmax in shrink epilogue side buffer, quant-load in expand
prologue.  (Current design.)

Shrink epilogue: GEMM → bf16 store + per-tile absmax to side buffer.
Expand prologue: load precomputed absmax + bf16, derive scale, cast FP8,
                 feed into FP8 dot product.

All three quant modes:
  block-wise   — fused, PDL OK
  per-channel  — fused when lora_rank <= BLOCK_SIZE_N (shrink) and
                 lora_rank <= BLOCK_SIZE_K (expand).  PDL OK when fused.
                 Falls back to separate quant otherwise.
  per-tensor   — fallback to separate moe_kernel_quantize_input
                 (global reduction incompatible with PDL)
"""

import torch

from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.triton_utils import tl, triton
from vllm.triton_utils.allocation import set_triton_allocator

from .fused_moe_lora_fp8_op import (
    _adjust_kernel_inputs,
    _fp8_fused_moe_lora_expand,
    _get_c_ptrs,
    _get_expert_id,
    _get_lora_id,
    _get_ptr,
    _get_scale_ptr,
    _get_token_offs,
)
from .utils import supports_pdl, supports_tma

FP8_E4M3_MAX: float = 448.0
FP8_E4M3_MIN: float = -448.0


# ===================================================================
# Shrink kernel — GEMM + fused absmax epilogue
# ===================================================================
@triton.jit(
    do_not_specialize=[
        "num_valid_tokens",
        "EM",
        "stride_tl",
        "stride_el",
        "slice_a_size",
        "slice_a_scale_size",
        "slice_c_size",
        "slice_absmax_size",
    ]
)
def _shrink_absmax_kernel(
    # --- A (activations) ---
    a_ptr,
    a_desc,
    # --- B (LoRA-A weights, FP8) ---
    b_ptr,
    b_desc,
    # --- C (output: bf16 intermediate cache) ---
    c_ptr,
    # --- absmax output ---
    absmax_ptr,
    # --- input scales ---
    a_scale_ptr,
    b_scale_ptr,
    # --- MoE metadata ---
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    token_lora_mapping_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    num_experts,
    top_k_num,
    lora_ids,
    adapter_enabled,
    max_loras,
    # strides
    stride_am,
    stride_ak,
    stride_bl,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_tl,
    stride_el,
    stride_asm,
    stride_ask,
    stride_bsl,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # block-wise quantization of *input* A
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    slice_a_size,
    slice_a_scale_size,
    slice_c_size,
    # --- fused absmax params ---
    slice_absmax_size,
    stride_absmax_m,
    stride_absmax_k,
    # Meta-parameters
    num_slice_a: tl.constexpr,
    num_slice_c: tl.constexpr,
    token_mapping_factor: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    USE_B_L2_CACHE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
    IS_PRIMARY: tl.constexpr,
    USE_TMA: tl.constexpr,
    sort_c: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    per_channel_quant: tl.constexpr,
    FUSE_ABSMAX: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
):
    """Shrink GEMM with optional fused absmax epilogue.

    When FUSE_ABSMAX is True, computes per-row absmax over each
    BLOCK_SIZE_N tile and stores to absmax_ptr.  bf16 values stored
    to c_ptr as usual.
    """
    pid = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)
    grid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)
    lora_idx = tl.program_id(axis=2)
    pid_sk = pid % SPLIT_K
    pid_m_n = pid // SPLIT_K

    if SWAP_AB:
        num_pid_m = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_n = tl.cdiv(EM, BLOCK_SIZE_M)
    else:
        num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_m_n // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m_raw = first_pid_m + ((pid_m_n % num_pid_in_group) % group_size_m)
    pid_n_raw = (pid_m_n % num_pid_in_group) // group_size_m

    if SWAP_AB:
        pid_n = pid_m_raw
        pid_m = pid_n_raw
    else:
        pid_m = pid_m_raw
        pid_n = pid_n_raw

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)

    # --- LoRA / MoE routing ---
    lora_id = _get_lora_id(
        lora_ids,
        token_lora_mapping_ptr,
        lora_idx,
        pid_m,
        top_k_num,
        naive_block_assignment,
    )
    if lora_id == -1:
        return
    moe_enabled = tl.load(adapter_enabled + lora_id)
    if moe_enabled == 0:
        return
    if lora_id >= max_loras:
        return

    if not naive_block_assignment:
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr + lora_id)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return

    expert_id = _get_expert_id(
        expert_ids_ptr,
        lora_id,
        pid_m,
        stride_el,
        max_loras,
        naive_block_assignment,
    )
    if expert_id == -1:
        return

    offs_token = _get_token_offs(
        sorted_token_ids_ptr,
        lora_id,
        pid_m,
        offs,
        stride_tl,
        max_loras,
        num_valid_tokens,
        naive_block_assignment,
        BLOCK_SIZE_M,
    )

    # --- pointer setup ---
    cur_a_ptr = a_ptr + (slice_id % num_slice_a) * slice_a_size
    cur_b_ptr = (
        tl.load(b_ptr + slice_id).to(tl.pointer_type(tl.float8e4nv))
        if b_scale_ptr is not None
        else tl.load(b_ptr + slice_id).to(tl.pointer_type(c_ptr.dtype.element_ty))
    )
    cur_c_ptr = c_ptr + (slice_id % num_slice_c) * slice_c_size

    offs_k = pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    token_mask = offs_token < num_valid_tokens

    # --- A pointers ---
    tl.static_assert(a_desc is None, "a_desc must be none for shrink")
    if SWAP_AB:
        a_ptrs = cur_a_ptr + (
            offs_k[:, None] * stride_ak
            + offs_token[None, :] // token_mapping_factor * stride_am
        )
    else:
        a_ptrs = cur_a_ptr + (
            offs_token[:, None] // token_mapping_factor * stride_am
            + offs_k[None, :] * stride_ak
        )
    a_scale_row_offs = offs_token // token_mapping_factor

    # --- B pointers ---
    if USE_TMA:
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_bk = pid_sk * BLOCK_SIZE_K
        if b_desc is None:
            if USE_GDC and not IS_PRIMARY:
                tl.extra.cuda.gdc_wait()
            cur_b_ptr = (
                tl.load(b_ptr + slice_id).to(tl.pointer_type(tl.float8e4nv))
                if b_scale_ptr is not None
                else tl.load(b_ptr + slice_id).to(
                    tl.pointer_type(c_ptr.dtype.element_ty)
                )
            )
            b_desc = tl.make_tensor_descriptor(
                cur_b_ptr,
                shape=[max_loras, num_experts, N, K],
                strides=[stride_bl, stride_be, stride_bn, stride_bk],
                block_shape=[1, 1, BLOCK_SIZE_N, BLOCK_SIZE_K],
            )
    else:
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int32)
        if SWAP_AB:
            b_ptrs = (
                cur_b_ptr
                + lora_id * stride_bl
                + expert_id * stride_be
                + offs_bn[:, None] * stride_bn
                + offs_k[None, :] * stride_bk
            )
        else:
            b_ptrs = (
                cur_b_ptr
                + lora_id * stride_bl
                + expert_id * stride_be
                + offs_k[:, None] * stride_bk
                + offs_bn[None, :] * stride_bn
            )

    # --- input FP8 scales ---
    if use_fp8_w8a8:
        cur_b_scale_ptr = tl.load(b_scale_ptr + slice_id).to(
            tl.pointer_type(tl.float32)
        )
        cur_a_scale_ptr = a_scale_ptr + (slice_id % num_slice_a) * slice_a_scale_size
        if USE_TMA:
            offs_bn_vec = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int32)
        else:
            offs_bn_vec = offs_bn
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = cur_a_scale_ptr + a_scale_row_offs * stride_asm
            offs_bsn = offs_bn_vec // group_n
            b_scale_ptrs = (
                cur_b_scale_ptr
                + lora_id * stride_bsl
                + expert_id * stride_bse
                + offs_bsn * stride_bsn
            )
        elif per_channel_quant:
            b_scale_ptrs = (
                cur_b_scale_ptr
                + lora_id * stride_bsl
                + expert_id * stride_bse
                + offs_bn_vec[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            a_scale_ptrs = cur_a_scale_ptr + a_scale_row_offs * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        else:
            a_scale = tl.load(cur_a_scale_ptr)
            b_scale = tl.load(cur_b_scale_ptr + lora_id * stride_bsl + expert_id)

    if USE_GDC and IS_PRIMARY:
        tl.extra.cuda.gdc_launch_dependents()

    # --- accumulator ---
    if SWAP_AB:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if USE_GDC and not IS_PRIMARY:
        tl.extra.cuda.gdc_wait()

    # --- GEMM loop ---
    for k in range(0, grid_k):
        cur_k_offset = k * (BLOCK_SIZE_K * SPLIT_K)
        k_remaining = K - cur_k_offset

        if use_fp8_w8a8 and group_n > 0 and group_k > 0:
            k_start = k * BLOCK_SIZE_K * SPLIT_K
            offs_ks = k_start // group_k
            a_scale = tl.load(
                a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
            )
            b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

        if SWAP_AB:
            b_mask = (offs_k[None, :] < k_remaining) & (offs_bn[:, None] < N)
        else:
            b_mask = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)

        if b_desc is not None:
            if SWAP_AB:
                b = b_desc.load(
                    [lora_id, expert_id, offs_bn, offs_bk + cur_k_offset]
                ).reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
            else:
                b = (
                    b_desc.load([lora_id, expert_id, offs_bn, offs_bk + cur_k_offset])
                    .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
                    .T
                )
        else:
            if USE_B_L2_CACHE:
                b = tl.load(b_ptrs, mask=b_mask, other=0.0, cache_modifier=".ca")
            else:
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

        if USE_GDC and not IS_PRIMARY:
            tl.extra.cuda.gdc_wait()

        if SWAP_AB:
            a = tl.load(
                a_ptrs,
                mask=(offs_k[:, None] < k_remaining) & token_mask[None, :],
                other=0.0,
            )
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                other=0.0,
            )
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak

        if USE_GDC and not IS_PRIMARY:
            tl.extra.cuda.gdc_wait()

        if SWAP_AB:
            if use_fp8_w8a8:
                if group_n > 0 and group_k > 0:
                    scale = b_scale[:, None] * a_scale[None, :]
                    accumulator += tl.dot(b, a) * scale
                else:
                    accumulator = tl.dot(b, a, acc=accumulator)
            else:
                accumulator += tl.dot(b, a)
        else:
            if use_fp8_w8a8:
                if group_n > 0 and group_k > 0:
                    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
                else:
                    accumulator = tl.dot(a, b, acc=accumulator)
            else:
                accumulator += tl.dot(a, b)

    # --- transpose if swapped ---
    if SWAP_AB:
        accumulator = tl.trans(accumulator)

    # --- apply routed weight ---
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = moe_weight[:, None] * accumulator

    # --- dequant input FP8 scales ---
    if use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            pass  # already applied per-iteration
        else:
            accumulator = accumulator * a_scale * b_scale

    # ===============================================================
    # FUSED ABSMAX EPILOGUE
    # ===============================================================
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    accumulator_out = accumulator.to(c_ptr.dtype.element_ty)
    c_ptrs = _get_c_ptrs(
        cur_c_ptr,
        lora_id,
        pid_m,
        offs,
        offs_token,
        offs_cn,
        stride_cm,
        stride_cn,
        EM,
        BLOCK_SIZE_M,
        sort_c,
    )
    if SPLIT_K == 1:
        tl.store(c_ptrs, accumulator_out, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator_out, mask=c_mask, sem="relaxed")

    # Compute and store absmax
    if FUSE_ABSMAX:
        row_absmax = tl.max(tl.abs(accumulator), axis=1)
        absmax_col = pid_n
        cur_absmax_ptr = absmax_ptr + (slice_id % num_slice_c) * slice_absmax_size
        if sort_c:
            offs_token_id = pid_m * BLOCK_SIZE_M + offs
            absmax_ptrs = (
                cur_absmax_ptr
                + lora_id * EM * stride_absmax_m
                + offs_token_id * stride_absmax_m
                + absmax_col * stride_absmax_k
            )
        else:
            absmax_ptrs = (
                cur_absmax_ptr
                + offs_token * stride_absmax_m
                + absmax_col * stride_absmax_k
            )
        tl.store(absmax_ptrs, row_absmax, mask=token_mask)


# ===================================================================
# Expand kernel — fused quant-load from precomputed absmax
# ===================================================================
@triton.jit(
    do_not_specialize=[
        "num_valid_tokens",
        "EM",
        "stride_tl",
        "stride_el",
        "slice_a_size",
        "slice_a_scale_size",
        "slice_c_size",
    ]
)
def _expand_quant_kernel(
    # --- A (bf16 intermediate cache from shrink) ---
    a_ptr,
    a_desc,
    # --- B (LoRA-B weights, FP8) ---
    b_ptr,
    b_desc,
    # --- C (output) ---
    c_ptr,
    # --- scales ---
    a_absmax_ptr,  # precomputed absmax from shrink (f32)
    b_scale_ptr,  # weight scale
    # --- MoE metadata ---
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    token_lora_mapping_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    num_experts,
    top_k_num,
    lora_ids,
    adapter_enabled,
    max_loras,
    # strides
    stride_am,
    stride_ak,
    stride_bl,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_tl,
    stride_el,
    # absmax strides
    stride_absmax_m,
    stride_absmax_k,
    # weight scale strides
    stride_bsl,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # block-wise quantization params for weight scales
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    slice_a_size,
    slice_a_scale_size,
    slice_c_size,
    # FP8 constants
    fp8_max: tl.constexpr,
    fp8_min: tl.constexpr,
    eps: tl.constexpr,
    # Meta-parameters
    num_slice_a: tl.constexpr,
    num_slice_c: tl.constexpr,
    token_mapping_factor: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    USE_B_L2_CACHE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
    IS_PRIMARY: tl.constexpr,
    USE_TMA: tl.constexpr,
    sort_c: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    per_channel_quant: tl.constexpr,
    FUSE_QUANT_LOAD: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
):
    """Expand GEMM with fused on-the-fly quantization of A loads.

    When FUSE_QUANT_LOAD is True:
      1. Loads bf16 tiles from intermediate cache.
      2. Loads precomputed absmax from a_absmax_ptr.
      3. scale = max(absmax, eps) / fp8_max
      4. Divides bf16 by scale, casts to FP8.
      5. Uses FP8 in dot product, applies scale as dequant factor.
    """
    pid = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)
    grid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)
    lora_idx = tl.program_id(axis=2)
    pid_sk = pid % SPLIT_K
    pid_m_n = pid // SPLIT_K

    if SWAP_AB:
        num_pid_m = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_n = tl.cdiv(EM, BLOCK_SIZE_M)
    else:
        num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_m_n // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m_raw = first_pid_m + ((pid_m_n % num_pid_in_group) % group_size_m)
    pid_n_raw = (pid_m_n % num_pid_in_group) // group_size_m

    if SWAP_AB:
        pid_n = pid_m_raw
        pid_m = pid_n_raw
    else:
        pid_m = pid_m_raw
        pid_n = pid_n_raw

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)

    # --- LoRA / MoE routing ---
    lora_id = _get_lora_id(
        lora_ids,
        token_lora_mapping_ptr,
        lora_idx,
        pid_m,
        top_k_num,
        naive_block_assignment,
    )
    if lora_id == -1:
        return
    moe_enabled = tl.load(adapter_enabled + lora_id)
    if moe_enabled == 0:
        return
    if lora_id >= max_loras:
        return

    if not naive_block_assignment:
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr + lora_id)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return

    expert_id = _get_expert_id(
        expert_ids_ptr,
        lora_id,
        pid_m,
        stride_el,
        max_loras,
        naive_block_assignment,
    )
    if expert_id == -1:
        return

    offs_token = _get_token_offs(
        sorted_token_ids_ptr,
        lora_id,
        pid_m,
        offs,
        stride_tl,
        max_loras,
        num_valid_tokens,
        naive_block_assignment,
        BLOCK_SIZE_M,
    )

    # --- pointer setup ---
    cur_a_ptr = a_ptr + (slice_id % num_slice_a) * slice_a_size
    cur_b_ptr = (
        tl.load(b_ptr + slice_id).to(tl.pointer_type(tl.float8e4nv))
        if b_scale_ptr is not None
        else tl.load(b_ptr + slice_id).to(tl.pointer_type(c_ptr.dtype.element_ty))
    )
    cur_c_ptr = c_ptr + (slice_id % num_slice_c) * slice_c_size

    offs_k = pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    token_mask = offs_token < num_valid_tokens

    # --- A pointers ---
    if USE_TMA and a_desc is not None:
        if naive_block_assignment:
            offs_am = pid_m
        else:
            offs_am = (
                slice_id * max_loras * EM
                + lora_id * EM
                + pid_m * BLOCK_SIZE_M // token_mapping_factor
            )
        offs_ak = pid_sk * BLOCK_SIZE_K
        if naive_block_assignment:
            a_scale_row_offs = tl.where(offs == 0, pid_m, num_valid_tokens)
        else:
            a_scale_row_offs = offs_am + offs // token_mapping_factor
    else:
        tl.static_assert(a_desc is None, "a_desc must be none")
        if SWAP_AB:
            a_ptrs = cur_a_ptr + (
                offs_k[:, None] * stride_ak
                + offs_token[None, :] // token_mapping_factor * stride_am
            )
        else:
            a_ptrs = cur_a_ptr + (
                offs_token[:, None] // token_mapping_factor * stride_am
                + offs_k[None, :] * stride_ak
            )
        a_scale_row_offs = offs_token // token_mapping_factor

    # --- B pointers ---
    if USE_TMA:
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_bk = pid_sk * BLOCK_SIZE_K
        if b_desc is None:
            if USE_GDC and not IS_PRIMARY:
                tl.extra.cuda.gdc_wait()
            cur_b_ptr = (
                tl.load(b_ptr + slice_id).to(tl.pointer_type(tl.float8e4nv))
                if b_scale_ptr is not None
                else tl.load(b_ptr + slice_id).to(
                    tl.pointer_type(c_ptr.dtype.element_ty)
                )
            )
            b_desc = tl.make_tensor_descriptor(
                cur_b_ptr,
                shape=[max_loras, num_experts, N, K],
                strides=[stride_bl, stride_be, stride_bn, stride_bk],
                block_shape=[1, 1, BLOCK_SIZE_N, BLOCK_SIZE_K],
            )
    else:
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int32)
        if SWAP_AB:
            b_ptrs = (
                cur_b_ptr
                + lora_id * stride_bl
                + expert_id * stride_be
                + offs_bn[:, None] * stride_bn
                + offs_k[None, :] * stride_bk
            )
        else:
            b_ptrs = (
                cur_b_ptr
                + lora_id * stride_bl
                + expert_id * stride_be
                + offs_k[:, None] * stride_bk
                + offs_bn[None, :] * stride_bn
            )

    # --- Weight FP8 scales ---
    if use_fp8_w8a8:
        cur_b_scale_ptr = tl.load(b_scale_ptr + slice_id).to(
            tl.pointer_type(tl.float32)
        )
        if USE_TMA:
            offs_bn_vec = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int32)
        else:
            offs_bn_vec = offs_bn
        if group_k > 0 and group_n > 0:
            offs_bsn = offs_bn_vec // group_n
            b_scale_ptrs = (
                cur_b_scale_ptr
                + lora_id * stride_bsl
                + expert_id * stride_bse
                + offs_bsn * stride_bsn
            )
        elif per_channel_quant:
            b_scale_ptrs = (
                cur_b_scale_ptr
                + lora_id * stride_bsl
                + expert_id * stride_bse
                + offs_bn_vec[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
        else:
            b_scale = tl.load(cur_b_scale_ptr + lora_id * stride_bsl + expert_id)

    # --- Load precomputed absmax for fused quant ---
    if FUSE_QUANT_LOAD and use_fp8_w8a8:
        cur_absmax_ptr = a_absmax_ptr + (slice_id % num_slice_a) * slice_a_scale_size
        absmax_row_ptrs = cur_absmax_ptr + a_scale_row_offs * stride_absmax_m

    if USE_GDC and IS_PRIMARY:
        tl.extra.cuda.gdc_launch_dependents()

    # --- accumulator ---
    if SWAP_AB:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if USE_GDC and not IS_PRIMARY:
        tl.extra.cuda.gdc_wait()

    # --- GEMM loop ---
    for k in range(0, grid_k):
        cur_k_offset = k * (BLOCK_SIZE_K * SPLIT_K)
        k_remaining = K - cur_k_offset

        # Load weight scales for this K-group
        if use_fp8_w8a8 and group_n > 0 and group_k > 0:
            k_start = k * BLOCK_SIZE_K * SPLIT_K
            offs_ks = k_start // group_k
            b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

        # Load B tile
        if SWAP_AB:
            b_mask = (offs_k[None, :] < k_remaining) & (offs_bn[:, None] < N)
        else:
            b_mask = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)

        if b_desc is not None:
            if SWAP_AB:
                b = b_desc.load(
                    [lora_id, expert_id, offs_bn, offs_bk + cur_k_offset]
                ).reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
            else:
                b = (
                    b_desc.load([lora_id, expert_id, offs_bn, offs_bk + cur_k_offset])
                    .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
                    .T
                )
        else:
            if USE_B_L2_CACHE:
                b = tl.load(b_ptrs, mask=b_mask, other=0.0, cache_modifier=".ca")
            else:
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

        if USE_GDC and not IS_PRIMARY:
            tl.extra.cuda.gdc_wait()

        # --- Load A tile with fused quant ---
        if FUSE_QUANT_LOAD and use_fp8_w8a8:
            k_group_idx = k
            absmax_vals = tl.load(
                absmax_row_ptrs + k_group_idx * stride_absmax_k,
                mask=token_mask,
                other=0.0,
            )
            a_act_scale = tl.maximum(absmax_vals, eps) / fp8_max

            # Load bf16 tile
            if a_desc is not None:
                if SWAP_AB:
                    a_bf16 = a_desc.load([offs_am, offs_ak + cur_k_offset]).T
                else:
                    a_bf16 = a_desc.load([offs_am, offs_ak + cur_k_offset])
            else:
                if SWAP_AB:
                    a_bf16 = tl.load(
                        a_ptrs,
                        mask=(offs_k[:, None] < k_remaining) & token_mask[None, :],
                        other=0.0,
                    )
                else:
                    a_bf16 = tl.load(
                        a_ptrs,
                        mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                        other=0.0,
                    )
                a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak

            # Quantize on-the-fly
            a_f32 = a_bf16.to(tl.float32)
            if SWAP_AB:
                a_q = tl.clamp(a_f32 / a_act_scale[None, :], fp8_min, fp8_max).to(
                    tl.float8e4nv
                )
            else:
                a_q = tl.clamp(a_f32 / a_act_scale[:, None], fp8_min, fp8_max).to(
                    tl.float8e4nv
                )
            a = a_q
        else:
            # Non-fused path
            if a_desc is not None:
                if SWAP_AB:
                    a = a_desc.load([offs_am, offs_ak + cur_k_offset]).T
                else:
                    a = a_desc.load([offs_am, offs_ak + cur_k_offset])
            else:
                if SWAP_AB:
                    a = tl.load(
                        a_ptrs,
                        mask=(offs_k[:, None] < k_remaining) & token_mask[None, :],
                        other=0.0,
                    )
                else:
                    a = tl.load(
                        a_ptrs,
                        mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                        other=0.0,
                    )
                a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak

            if use_fp8_w8a8 and group_n > 0 and group_k > 0:
                k_start = k * BLOCK_SIZE_K * SPLIT_K
                offs_ks = k_start // group_k
                cur_a_scale_ptr = (
                    a_absmax_ptr + (slice_id % num_slice_a) * slice_a_scale_size
                )
                a_scale_ptrs_nf = cur_a_scale_ptr + a_scale_row_offs * stride_absmax_m
                a_act_scale = tl.load(
                    a_scale_ptrs_nf + offs_ks * stride_absmax_k,
                    mask=token_mask,
                    other=0.0,
                )

        if USE_GDC and not IS_PRIMARY:
            tl.extra.cuda.gdc_wait()

        # --- dot product ---
        if SWAP_AB:
            if use_fp8_w8a8:
                if group_n > 0 and group_k > 0:
                    scale = b_scale[:, None] * a_act_scale[None, :]
                    accumulator += tl.dot(b, a) * scale
                else:
                    accumulator = tl.dot(b, a, acc=accumulator)
            else:
                accumulator += tl.dot(b, a)
        else:
            if use_fp8_w8a8:
                if group_n > 0 and group_k > 0:
                    accumulator += (
                        tl.dot(a, b) * a_act_scale[:, None] * b_scale[None, :]
                    )
                else:
                    accumulator = tl.dot(a, b, acc=accumulator)
            else:
                accumulator += tl.dot(a, b)

    # --- transpose if swapped ---
    if SWAP_AB:
        accumulator = tl.trans(accumulator)

    # --- apply routed weight ---
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]

    # --- dequant epilogue ---
    if use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(c_ptr.dtype.element_ty)
        else:
            if not FUSE_QUANT_LOAD:
                cur_a_scale_ptr_tw = (
                    a_absmax_ptr + (slice_id % num_slice_a) * slice_a_scale_size
                )
                if per_channel_quant:
                    a_scale_tw = tl.load(
                        cur_a_scale_ptr_tw + a_scale_row_offs * stride_absmax_m,
                        mask=token_mask,
                        other=0.0,
                    )[:, None]
                else:
                    a_scale_tw = tl.load(cur_a_scale_ptr_tw)
                accumulator = (accumulator * a_scale_tw * b_scale).to(
                    c_ptr.dtype.element_ty
                )
            else:
                # Fused path: a_act_scale already applied via quant+dequant.
                # For per-channel/per-tensor fused: the tile-local absmax
                # was used as the quant scale, and the FP8 dot already
                # "baked in" the inverse.  We just need b_scale.
                accumulator = (accumulator * b_scale).to(c_ptr.dtype.element_ty)
    else:
        accumulator = accumulator.to(c_ptr.dtype.element_ty)

    # --- store output ---
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = _get_c_ptrs(
        cur_c_ptr,
        lora_id,
        pid_m,
        offs,
        offs_token,
        offs_cn,
        stride_cm,
        stride_cn,
        EM,
        BLOCK_SIZE_M,
        sort_c,
    )
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    if SPLIT_K == 1:
        if ADD_INPUTS:
            tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")
        else:
            tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")


# ===================================================================
# Python wrappers
# ===================================================================


@torch.inference_mode()
def _shrink_with_absmax(
    a_intermediate_cache1: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor | None,
    token_lora_mapping: torch.Tensor,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    num_active_loras: int,
    mul_routed_weight: bool = False,
    use_gdc: bool = False,
    use_tma: bool = False,
    act_scale: torch.Tensor | None = None,
    lora_a_scale_stacked: list[torch.Tensor] | None = None,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    block_shape: list[int] | None = None,
    absmax_out: torch.Tensor | None = None,
) -> None:
    fuse_absmax = absmax_out is not None

    if use_fp8_w8a8:
        assert lora_a_scale_stacked is not None
    else:
        assert act_scale is None
        assert lora_a_scale_stacked is None

    w1_lora_a_stacked = lora_a_stacked[0]

    if block_shape is not None:
        block_size_k = min(block_size_k, min(block_shape[0], block_shape[1]))

    swap_ab = block_size_m < 64

    shrink_config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": group_size_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "SPLIT_K": split_k,
        "USE_GDC": use_gdc,
        "launch_pdl": use_gdc,
        "USE_TMA": use_tma,
        "SWAP_AB": swap_ab,
    }

    b_ptr = _get_ptr(lora_a_stacked, device)
    if lora_a_scale_stacked is not None:
        b_scale_ptr = _get_scale_ptr(lora_a_scale_stacked, device)
        w1_lora_a_scale_stacked = lora_a_scale_stacked[0]
    else:
        b_scale_ptr = None
        w1_lora_a_scale_stacked = None

    grid_lora_dim, stride_tl, stride_el = _adjust_kernel_inputs(
        num_active_loras, sorted_token_ids, expert_ids
    )
    grid = lambda META: (
        split_k
        * triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        len(lora_a_stacked),
        grid_lora_dim,
    )

    a_desc = None
    b_desc = None
    if use_tma and num_slices == 1:
        b_desc = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
            lora_a_stacked[0],
            [1, 1, shrink_config["BLOCK_SIZE_N"], shrink_config["BLOCK_SIZE_K"]],
        )

    _shrink_absmax_kernel[grid](
        qcurr_hidden_states,
        a_desc,
        b_ptr,
        b_desc,
        a_intermediate_cache1,
        absmax_out,
        act_scale,
        b_scale_ptr,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        N,
        K,
        EM,
        num_tokens,
        num_experts,
        top_k_num,
        lora_ids,
        adapter_enabled,
        lora_a_stacked[0].shape[0],
        # strides
        qcurr_hidden_states.stride(0),
        qcurr_hidden_states.stride(1),
        w1_lora_a_stacked.stride(0),
        w1_lora_a_stacked.stride(1),
        w1_lora_a_stacked.stride(3),
        w1_lora_a_stacked.stride(2),
        a_intermediate_cache1.stride(-2),
        a_intermediate_cache1.stride(-1),
        stride_tl,
        stride_el,
        act_scale.stride(0) if act_scale is not None and act_scale.ndim == 2 else 0,
        act_scale.stride(1) if act_scale is not None and act_scale.ndim == 2 else 0,
        w1_lora_a_scale_stacked.stride(0)
        if lora_a_scale_stacked is not None and w1_lora_a_scale_stacked.ndim >= 2
        else 0,
        w1_lora_a_scale_stacked.stride(1)
        if lora_a_scale_stacked is not None and w1_lora_a_scale_stacked.ndim >= 2
        else 0,
        w1_lora_a_scale_stacked.stride(3)
        if lora_a_scale_stacked is not None and w1_lora_a_scale_stacked.ndim == 4
        else 0,
        w1_lora_a_scale_stacked.stride(2)
        if lora_a_scale_stacked is not None and w1_lora_a_scale_stacked.ndim == 4
        else 0,
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        slice_a_size=qcurr_hidden_states.numel(),
        slice_a_scale_size=act_scale.numel() if act_scale is not None else 0,
        slice_c_size=a_intermediate_cache1.numel() // num_slices,
        slice_absmax_size=(
            absmax_out.numel() // num_slices if absmax_out is not None else 0
        ),
        stride_absmax_m=absmax_out.stride(-2) if absmax_out is not None else 0,
        stride_absmax_k=absmax_out.stride(-1) if absmax_out is not None else 0,
        num_slice_a=1,
        num_slice_c=num_slices,
        token_mapping_factor=1 if mul_routed_weight else top_k_num,
        naive_block_assignment=sorted_token_ids is None,
        MUL_ROUTED_WEIGHT=False,
        USE_B_L2_CACHE=True,
        sort_c=use_tma and sorted_token_ids is not None,
        IS_PRIMARY=True,
        use_fp8_w8a8=use_fp8_w8a8,
        per_channel_quant=per_channel_quant,
        FUSE_ABSMAX=fuse_absmax,
        **shrink_config,
    )


@torch.inference_mode()
def _expand_with_quant_load(
    output: torch.Tensor,
    a_intermediate_cache1: torch.Tensor,
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor | None,
    token_lora_mapping: torch.Tensor,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    max_lora_rank: int,
    w1_output_dim_size: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    num_active_loras: int,
    mul_routed_weight: bool = False,
    offset: int = 0,
    use_gdc: bool = False,
    use_tma: bool = False,
    absmax_in: torch.Tensor | None = None,
    lora_b_scale_stacked: list[torch.Tensor] | None = None,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    block_shape: list[int] | None = None,
    fuse_quant_load: bool = False,
) -> None:
    if use_fp8_w8a8:
        assert lora_b_scale_stacked is not None
    else:
        assert lora_b_scale_stacked is None

    b_ptr = _get_ptr(lora_b_stacked, device)
    K = max_lora_rank
    N = w1_output_dim_size
    w1_lora_b_stacked = lora_b_stacked[0]

    if lora_b_scale_stacked is not None:
        b_scale_ptr = _get_scale_ptr(lora_b_scale_stacked, device)
        w1_lora_b_scale_stacked = lora_b_scale_stacked[0]
    else:
        b_scale_ptr = None
        w1_lora_b_scale_stacked = None

    a_intermediate_cache1 = a_intermediate_cache1.view(
        -1, a_intermediate_cache1.shape[-1]
    )

    if block_shape is not None:
        block_size_k = min(block_size_k, min(block_shape[0], block_shape[1]))

    swap_ab = block_size_m < 64

    expand_config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": group_size_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "SPLIT_K": 1,
        "USE_GDC": use_gdc,
        "launch_pdl": use_gdc,
        "USE_TMA": use_tma,
        "SWAP_AB": swap_ab,
    }

    grid_lora_dim, stride_tl, stride_el = _adjust_kernel_inputs(
        num_active_loras, sorted_token_ids, expert_ids
    )
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        len(lora_b_stacked),
        grid_lora_dim,
    )

    out_view = output[:, :, offset : offset + num_slices * N]
    slice_c_size = N * out_view.stride(2)

    a_desc = None
    b_desc = None
    if use_tma:
        a_desc = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
            a_intermediate_cache1,
            [expand_config["BLOCK_SIZE_M"], expand_config["BLOCK_SIZE_K"]],
        )
        if num_slices == 1:
            b_desc = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
                lora_b_stacked[0],
                [1, 1, expand_config["BLOCK_SIZE_N"], expand_config["BLOCK_SIZE_K"]],
            )

    absmax_scale = absmax_in

    _expand_quant_kernel[grid](
        a_intermediate_cache1,
        a_desc,
        b_ptr,
        b_desc,
        out_view,
        absmax_scale,
        b_scale_ptr,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        N,
        K,
        EM,
        num_tokens,
        num_experts,
        top_k_num,
        lora_ids,
        adapter_enabled,
        lora_b_stacked[0].shape[0],
        # strides
        a_intermediate_cache1.stride(0),
        a_intermediate_cache1.stride(1),
        w1_lora_b_stacked.stride(0),
        w1_lora_b_stacked.stride(1),
        w1_lora_b_stacked.stride(3),
        w1_lora_b_stacked.stride(2),
        out_view.stride(1),
        out_view.stride(2),
        stride_tl,
        stride_el,
        absmax_scale.stride(-2) if absmax_scale is not None else 0,
        absmax_scale.stride(-1) if absmax_scale is not None else 0,
        w1_lora_b_scale_stacked.stride(0)
        if lora_b_scale_stacked is not None and w1_lora_b_scale_stacked.ndim >= 2
        else 0,
        w1_lora_b_scale_stacked.stride(1)
        if lora_b_scale_stacked is not None and w1_lora_b_scale_stacked.ndim >= 2
        else 0,
        w1_lora_b_scale_stacked.stride(3)
        if lora_b_scale_stacked is not None and w1_lora_b_scale_stacked.ndim == 4
        else 0,
        w1_lora_b_scale_stacked.stride(2)
        if lora_b_scale_stacked is not None and w1_lora_b_scale_stacked.ndim == 4
        else 0,
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        slice_a_size=a_intermediate_cache1.numel() // num_slices,
        slice_a_scale_size=(
            absmax_scale.numel() // num_slices if absmax_scale is not None else 0
        ),
        slice_c_size=slice_c_size,
        fp8_max=FP8_E4M3_MAX,
        fp8_min=FP8_E4M3_MIN,
        eps=1e-10,
        num_slice_a=num_slices,
        num_slice_c=num_slices,
        token_mapping_factor=1,
        naive_block_assignment=sorted_token_ids is None,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        ADD_INPUTS=True,
        USE_B_L2_CACHE=True,
        sort_c=False,
        IS_PRIMARY=False,
        use_fp8_w8a8=use_fp8_w8a8,
        per_channel_quant=per_channel_quant,
        FUSE_QUANT_LOAD=fuse_quant_load,
        **expand_config,
    )


# ===================================================================
# Top-level launcher
# ===================================================================


@torch.inference_mode()
def separate_fused(
    output: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor | None,
    token_lora_mapping: torch.Tensor,
    max_lora_rank: int,
    top_k_num: int,
    lora_ids: torch.Tensor,
    num_active_loras: int,
    adapter_enabled: torch.Tensor,
    shrink_block_size_m: int,
    shrink_block_size_n: int,
    shrink_block_size_k: int,
    shrink_group_size_m: int,
    shrink_num_warps: int,
    shrink_num_stages: int,
    shrink_split_k: int,
    expand_block_size_m: int,
    expand_block_size_n: int,
    expand_block_size_k: int,
    expand_group_size_m: int,
    expand_num_warps: int,
    expand_num_stages: int,
    expand_split_k: int,
    lora_a_scale_stacked: list[torch.Tensor] | None,
    lora_b_scale_stacked: list[torch.Tensor] | None,
    shrink_act_scale: torch.Tensor | None = None,
    expand_act_scale: torch.Tensor | None = None,
    mul_routed_weight: bool = False,
    fully_sharded: bool = False,
    offset: int = 0,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    block_shape: list[int] | None = None,
) -> None:
    """Separate fused: absmax in shrink epilogue, quant-load in expand.

    Supports block-wise, per-channel, per-tensor.
    - block-wise:   always fused, PDL OK
    - per-channel:  fused when lora_rank <= shrink BLOCK_SIZE_N and
                    lora_rank <= expand BLOCK_SIZE_K.  PDL OK.
    - per-tensor:   fallback to separate moe_kernel_quantize_input
    """
    assert len(lora_a_stacked) == len(lora_b_stacked) > 0
    assert topk_weights.dim() == qcurr_hidden_states.dim() == 2
    if sorted_token_ids is None:
        assert expert_ids.dim() == 1
    else:
        assert num_tokens_post_padded is not None
        assert (
            sorted_token_ids.dim()
            == expert_ids.dim()
            == topk_weights.dim()
            == qcurr_hidden_states.dim()
            == 2
        )
        assert (
            sorted_token_ids.shape[0]
            == expert_ids.shape[0]
            == num_tokens_post_padded.shape[0]
        )
    assert output.shape[0] == topk_weights.shape[0]
    assert top_k_num == topk_weights.shape[1]

    if not lora_a_scale_stacked:
        lora_a_scale_stacked = None
    if not lora_b_scale_stacked:
        lora_b_scale_stacked = None

    device = qcurr_hidden_states.device
    num_slices = len(lora_a_stacked)
    w1_lora_b_stacked = lora_b_stacked[0]
    num_experts = lora_a_stacked[0].shape[1]
    N = max_lora_rank
    M = topk_weights.shape[0]
    K = qcurr_hidden_states.shape[1]
    num_tokens = M * top_k_num
    w1_output_dim_size = w1_lora_b_stacked.shape[2]
    assert shrink_block_size_m == expand_block_size_m
    EM = (
        sorted_token_ids.shape[1]
        if sorted_token_ids is not None
        else num_tokens * shrink_block_size_m
    )

    use_tma = supports_tma(device) and not fully_sharded

    # --- Determine fusion eligibility ---
    # block-wise: always fusable
    # per-channel: fusable when lora_rank fits in one tile
    #   (shrink BLOCK_SIZE_N >= lora_rank AND expand BLOCK_SIZE_K >= lora_rank)
    # per-tensor: not fusable (needs global reduction)
    can_fuse_block_wise = (
        use_fp8_w8a8 and block_shape is not None and not per_channel_quant
    )
    can_fuse_per_channel = (
        use_fp8_w8a8
        and per_channel_quant
        and max_lora_rank <= shrink_block_size_n
        and max_lora_rank <= expand_block_size_k
    )
    can_fuse_quant = can_fuse_block_wise or can_fuse_per_channel

    # --- Allocate intermediate cache ---
    if use_tma:
        if num_slices > 1:
            set_triton_allocator(device)
        if sorted_token_ids is not None:
            intermediate_cache_shape = (
                num_slices,
                sorted_token_ids.shape[0],
                EM,
                max_lora_rank,
            )
        else:
            intermediate_cache_shape = (num_slices, M, top_k_num, max_lora_rank)
    else:
        intermediate_cache_shape = (num_slices, M, top_k_num, max_lora_rank)

    a_intermediate_cache1 = torch.zeros(
        intermediate_cache_shape, dtype=output.dtype, device=device
    )

    # --- Allocate absmax tensor ---
    if can_fuse_quant:
        if can_fuse_per_channel:
            # Per-channel: one absmax per row (single group since
            # lora_rank <= BLOCK_SIZE_N)
            num_absmax_groups = 1
        else:
            # Block-wise: one absmax per row per BLOCK_SIZE_N tile
            num_absmax_groups = triton.cdiv(max_lora_rank, shrink_block_size_n)
        absmax_shape = intermediate_cache_shape[:-1] + (num_absmax_groups,)
        absmax_out = torch.empty(absmax_shape, dtype=torch.float32, device=device)
    else:
        absmax_out = None

    use_gdc = supports_pdl(device) and not fully_sharded

    # --- Shrink ---
    _shrink_with_absmax(
        a_intermediate_cache1,
        qcurr_hidden_states,
        lora_a_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        top_k_num,
        lora_ids,
        adapter_enabled,
        device,
        N,
        M,
        EM,
        K,
        num_tokens,
        num_experts,
        num_slices,
        shrink_block_size_m,
        shrink_block_size_n,
        shrink_block_size_k,
        shrink_group_size_m,
        shrink_num_warps,
        shrink_num_stages,
        shrink_split_k,
        num_active_loras,
        lora_a_scale_stacked=lora_a_scale_stacked,
        mul_routed_weight=mul_routed_weight,
        use_gdc=use_gdc,
        use_tma=use_tma,
        act_scale=shrink_act_scale,
        use_fp8_w8a8=use_fp8_w8a8,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        absmax_out=absmax_out,
    )

    # --- All-reduce / all-gather for fully sharded ---
    if fully_sharded:
        if max_lora_rank == w1_lora_b_stacked.shape[-1]:
            a_intermediate_cache1 = tensor_model_parallel_all_reduce(
                a_intermediate_cache1
            )
        else:
            a_intermediate_cache1 = tensor_model_parallel_all_gather(
                a_intermediate_cache1
            )
            max_lora_rank = a_intermediate_cache1.shape[-1]

    # --- Fallback: separate quantization when fusion not possible ---
    if use_fp8_w8a8 and not can_fuse_quant:
        from vllm.model_executor.layers.fused_moe.utils import (
            moe_kernel_quantize_input,
        )

        orig_shape = a_intermediate_cache1.shape
        quant_dtype = torch.float8_e4m3fn
        intermediate_block_shape = block_shape
        if block_shape is not None:
            intermediate_block_shape = [
                min(block_shape[0], orig_shape[-1]),
                min(block_shape[1], orig_shape[-1]),
            ]
        a_intermediate_cache1 = a_intermediate_cache1.view(-1, orig_shape[-1])
        a_intermediate_cache1, expand_act_scale = moe_kernel_quantize_input(
            A=a_intermediate_cache1,
            A_scale=expand_act_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=intermediate_block_shape,
        )
        fuse_quant_load = False
    else:
        fuse_quant_load = can_fuse_quant

    # --- Expand ---
    if fuse_quant_load:
        _expand_with_quant_load(
            output,
            a_intermediate_cache1,
            lora_b_stacked,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            token_lora_mapping,
            top_k_num,
            lora_ids,
            adapter_enabled,
            device,
            N,
            M,
            EM,
            K,
            num_tokens,
            num_experts,
            num_slices,
            max_lora_rank,
            w1_output_dim_size,
            expand_block_size_m,
            expand_block_size_n,
            expand_block_size_k,
            expand_group_size_m,
            expand_num_warps,
            expand_num_stages,
            expand_split_k,
            num_active_loras,
            lora_b_scale_stacked=lora_b_scale_stacked,
            mul_routed_weight=mul_routed_weight,
            offset=offset,
            use_gdc=use_gdc,
            use_tma=use_tma,
            absmax_in=absmax_out,
            use_fp8_w8a8=use_fp8_w8a8,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            fuse_quant_load=True,
        )
    else:
        _fp8_fused_moe_lora_expand(
            output,
            a_intermediate_cache1,
            lora_b_stacked,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            token_lora_mapping,
            top_k_num,
            lora_ids,
            adapter_enabled,
            device,
            N,
            M,
            EM,
            K,
            num_tokens,
            num_experts,
            num_slices,
            max_lora_rank,
            w1_output_dim_size,
            expand_block_size_m,
            expand_block_size_n,
            expand_block_size_k,
            expand_group_size_m,
            expand_num_warps,
            expand_num_stages,
            expand_split_k,
            num_active_loras,
            lora_b_scale_stacked=lora_b_scale_stacked,
            mul_routed_weight=mul_routed_weight,
            offset=offset,
            use_gdc=use_gdc,
            use_tma=use_tma,
            act_scale=expand_act_scale,
            use_fp8_w8a8=use_fp8_w8a8,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
        )
