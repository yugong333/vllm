# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from typing import TYPE_CHECKING, Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    init_fp8_linear_kernel,
)
from vllm.model_executor.kernels.linear.scaled_mm import (
    CutlassFP8ScaledMMLinearKernel,
    MarlinFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.cpu_fused_moe import select_experts
from vllm.model_executor.layers.fused_moe.moe_monokernel_interleave import (
    interleave_for_tma_wgmma_up_v2,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    convert_to_fp8_moe_kernel_format,
    make_fp8_moe_kernel,
    make_fp8_moe_quant_config,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_input_scale,
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    process_fp8_input_tensor_strategy_moe,
    process_fp8_weight_tensor_strategy,
    process_fp8_weight_tensor_strategy_moe,
    validate_fp8_block_shape,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    create_fp8_quant_key,
    is_layer_skipped,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_block_fp8_supported,
    cutlass_fp8_supported,
    normalize_e4m3fn_to_e4m3fnuz,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_online_processing,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    is_deep_gemm_supported,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = init_logger(__name__)


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
        weight_block_size: list[int] | None = None,
        store_dtype: str | None = None,
    ) -> None:
        super().__init__()

        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized

        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        self.store_dtype = store_dtype
        if weight_block_size is not None:
            if not is_checkpoint_fp8_serialized:
                raise ValueError(
                    "The block-wise quantization only supports fp8-serialized "
                    "checkpoint for now."
                )
            if len(weight_block_size) != 2:
                raise ValueError(
                    "The quantization block size of weight must have 2 "
                    f"dimensions, but got {len(weight_block_size)} dimensions"
                )
            if activation_scheme != "dynamic":
                raise ValueError(
                    "The block-wise quantization only supports "
                    "dynamic activation scheme for now, but got "
                    f"{activation_scheme} activation scheme."
                )
        self.weight_block_size = weight_block_size
        self.use_deep_gemm: bool | None = None

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Fp8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = "fp8" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        store_dtype = cls.get_from_keys_or(config, ["store_dtype"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
            store_dtype=store_dtype,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if not self.is_checkpoint_fp8_serialized:
                from vllm.model_executor.layers.quantization.online.fp8 import (
                    Fp8PerTensorOnlineLinearMethod,
                )

                online_method = Fp8PerTensorOnlineLinearMethod()
                online_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
                return online_method
            else:
                offline_method = Fp8LinearMethod(self)
                offline_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
                return offline_method
        elif isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if self.store_dtype == "mxfp4":
                from vllm.model_executor.layers.quantization.mxfp4 import (
                    Mxfp4MoEMethod,
                )

                return Mxfp4MoEMethod(layer.moe_config)
            if self.is_checkpoint_fp8_serialized:
                moe_quant_method = Fp8MoEMethod(self, layer)
            else:
                moe_quant_method = Fp8OnlineMoEMethod(self, layer)
            return moe_quant_method
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None

    def get_cache_scale_mapper(self) -> "WeightsMapper":
        """Map compressed-tensors KV-cache scale names to vLLM names."""
        from vllm.model_executor.models.utils import WeightsMapper

        return WeightsMapper(
            orig_to_new_suffix={
                ".k_proj.output_scale": ".attn.k_scale",
                ".v_proj.output_scale": ".attn.v_scale",
                ".q_proj.output_scale": ".attn.q_scale",
                ".self_attn.prob_output_scale": ".self_attn.attn.prob_scale",
            }
        )


class CopyNumelCounter(TorchDispatchMode):
    """
    Tracks total number of elements modified with `copy_`. Useful for keeping
    track of weight loading where underlying weights can be arbitrarily
    transformed (such as with `narrow`) before calling copy.
    """

    def __init__(self):
        super().__init__()
        self.copied_numel = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        out = func(*args, **kwargs)
        if func == torch.ops.aten.copy_.default:
            self.copied_numel += args[0].numel()
        return out


def _copy_missing_attrs(old: torch.Tensor, new: torch.Tensor) -> None:
    """Copies any attrs present in `old` but not in `new` to `new`"""
    new_attrs = set(dir(new))
    attrs_to_set = {}
    for attr in dir(old):
        if attr not in new_attrs:
            attrs_to_set[attr] = getattr(old, attr)
    set_weight_attrs(new, attrs_to_set)


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Limitations:
    1. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.is_scale_e8m0 = getattr(quant_config, "is_scale_e8m0", False)
        self.cutlass_block_fp8_supported = cutlass_block_fp8_supported()
        self.out_dtype = torch.get_default_dtype()
        self.input_dtype = get_current_vllm_config().model_config.dtype

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.marlin_input_dtype = None
        self.use_marlin = False

        if self.quant_config.use_deep_gemm is not None:
            self.use_deep_gemm = self.quant_config.use_deep_gemm
        else:
            self.use_deep_gemm = is_deep_gemm_supported()

        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        self.act_q_static = self.quant_config.activation_scheme == "static"

        if self.block_quant:
            assert not self.act_q_static
            assert self.weight_block_size is not None

            self.activation_quant_key = create_fp8_quant_key(
                static=self.act_q_static,
                group_shape=GroupShape(1, self.weight_block_size[0]),
            )
            self.weight_quant_key = create_fp8_quant_key(
                static=True, group_shape=GroupShape(*self.weight_block_size)
            )
        else:
            self.weight_quant_key = kFp8StaticTensorSym
            # Use per-token quantization for better perf if dynamic and cutlass
            if self.act_q_static:
                self.activation_quant_key = kFp8StaticTensorSym
            elif cutlass_fp8_supported():
                self.activation_quant_key = kFp8DynamicTokenSym
            else:
                self.activation_quant_key = kFp8DynamicTensorSym

    def create_weights(
        self,
        layer: RoutedExperts,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            validate_fp8_block_shape(
                layer,
                input_size,
                output_size,
                input_size_per_partition,
                output_partition_sizes,
                self.weight_block_size,
            )

        weight = create_fp8_weight_parameter(
            output_size_per_partition, input_size_per_partition, weight_loader
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if not self.block_quant:
            scale = create_fp8_scale_parameter(
                PerTensorScaleParameter,
                output_partition_sizes,
                input_size_per_partition,
                None,
                weight_loader,
            )
            layer.register_parameter("weight_scale", scale)
        else:
            assert not self.act_q_static
            assert self.weight_block_size is not None
            scale = create_fp8_scale_parameter(
                BlockQuantScaleParameter,
                output_partition_sizes,
                input_size_per_partition,
                self.weight_block_size,
                weight_loader,
                scale_dtype=(torch.float8_e8m0fnu if self.is_scale_e8m0 else None),
            )
            # The weight_scale_inv name is intentional for deepseekv3
            layer.register_parameter("weight_scale_inv", scale)

        # INPUT ACTIVATION SCALE
        if self.act_q_static:
            scale = create_fp8_input_scale(output_partition_sizes, weight_loader)
            set_weight_attrs(scale, {"scale_type": "input_scale"})
            layer.register_parameter("input_scale", scale)

        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name=self.__class__.__name__,
        )

        self.use_marlin = isinstance(self.fp8_linear, MarlinFP8ScaledMMLinearKernel)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if self.use_marlin:
            if not self.block_quant:
                # Canonicalize to (K, N) for the kernel.
                replace_parameter(layer, "weight", layer.weight.t())
            # Only Marlin kernels support `marlin_input_dtype`; guard to avoid
            # AttributeError if backend selection changes.
            if hasattr(self.fp8_linear, "marlin_input_dtype"):
                self.fp8_linear.marlin_input_dtype = self.marlin_input_dtype
            self.fp8_linear.process_weights_after_loading(layer)
            return

        input_scale = None
        # TODO(rob): refactor block quant into separate class.
        if self.block_quant:
            assert not self.act_q_static

        # If checkpoint not serialized fp8, quantize the weights.
        else:
            # If checkpoint is fp8 per-tensor, handle that there are N scales for N
            # shards in a fused module
            weight = layer.weight
            weight_scale = layer.weight_scale

            # If using w8a8, torch._scaled_mm needs per tensor, so
            # requantize the logical shards as a single weight.
            weight, weight_scale, input_scale = process_fp8_weight_tensor_strategy(
                weight,
                weight_scale,
                layer.logical_widths,
                getattr(layer, "input_scale", None),
            )
            if self.act_q_static:
                assert input_scale is not None
                input_scale = input_scale.max()
            weight = weight.t()

            # Update layer with new values.
            replace_parameter(layer, "weight", weight.data)
            replace_parameter(layer, "weight_scale", weight_scale.data)

        if input_scale is not None:
            replace_parameter(layer, "input_scale", input_scale)
        else:
            layer.input_scale = None

        self.fp8_linear.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # if batch invariant mode is enabled, prefer direct FP8 path
        # we will use BF16 dequant when direct FP8 is not supported.
        if envs.VLLM_BATCH_INVARIANT:
            if self.block_quant:
                assert self.weight_block_size is not None
                return self.fp8_linear.apply_weights(
                    layer,
                    x,
                    bias,
                )
            else:
                if isinstance(self.fp8_linear, CutlassFP8ScaledMMLinearKernel):
                    return self.fp8_linear.apply_weights(layer, x, bias)

                # per-tensor/channel: dequant to BF16 and run GEMM
                weight_fp8 = layer.weight.to(torch.bfloat16)
                weight_scale = layer.weight_scale.to(torch.bfloat16)
                if weight_scale.numel() == 1:
                    # Per-tensor: simple scalar multiplication
                    weight_bf16 = weight_fp8 * weight_scale
                else:
                    # Multiple scales (fused modules like QKV)
                    # Try to infer correct broadcasting
                    # weight is [K, N], scale could be [num_logical_weights]
                    # Need to figure out how to broadcast - for now just try
                    # direct multiplication
                    if (
                        weight_scale.dim() == 1
                        and weight_scale.shape[0] == weight_fp8.shape[0]
                    ):
                        # Per-row scaling
                        weight_bf16 = weight_fp8 * weight_scale.unsqueeze(1)
                    else:
                        # Fallback
                        weight_bf16 = weight_fp8 * weight_scale
                return torch.nn.functional.linear(x, weight_bf16.t(), bias)

        return self.fp8_linear.apply_weights(layer, x, bias)


class Fp8MoEMethod(FusedMoEMethodBase):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config, layer: RoutedExperts):
        super().__init__(layer.moe_config)
        self.quant_config = quant_config
        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant: bool = self.weight_block_size is not None
        self.weight_scale_name = (
            "weight_scale_inv" if self.block_quant else "weight_scale"
        )

        # Scratchpad for MoE monokernel fast path (Qwen3.5-35B FP8 block-wise)
        # Layout: BS x E x N fp8 + BS x E x N/2 fp8 + BS x HIDDEN fp16
        # with BS=1024: 4MB + <1MB + 10MB < 4M x 4byte
        #
        # Allocate ZEROED (not empty): the software grid/partial barriers in
        # the monokernel (src/moe_grid_barrier.h) spin on counter slots that
        # live at the tail of this scratchpad and MUST start at 0 on first
        # use (self-maintaining via ping-pong reset thereafter). The kernel
        # is launched standalone (non-cooperative) so it can be captured into
        # a CUDA Graph; the wrapper's one-shot cudaMemsetAsync would be
        # *recorded* (not executed) if it first runs during graph capture,
        # leaving the counters uninitialized and deadlocking the barrier
        # spin. Zeroing here guarantees valid counters before the first
        # (capture-time) launch. Matches test_monokernel_accuracy.py.
        self.moe_monokernel_scratchpad = torch.zeros(
            (1024, 4096),
            dtype=torch.float32,
            device="cpu",
        )
        # Whether this layer is eligible for the MoE monokernel fast path.
        # Determined after weight loading in process_weights_after_loading.
        self._use_moe_monokernel = False

        # Set weight key and activation key for kernel compatibility
        if self.block_quant:
            weight_key = kFp8Static128BlockSym
            activation_key = kFp8Dynamic128Sym
        else:
            weight_key = kFp8StaticTensorSym
            activation_key = (
                kFp8StaticTensorSym
                if self.quant_config.activation_scheme == "static"
                else kFp8DynamicTensorSym
            )

        # Select Fp8 MoE backend
        self.fp8_backend, self.experts_cls = select_fp8_moe_backend(
            config=self.moe,
            weight_key=weight_key,
            activation_key=activation_key,
            allow_vllm_cutlass=False,
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        assert self.quant_config.is_checkpoint_fp8_serialized
        params_dtype = torch.float8_e4m3fn

        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            tp_size = get_tensor_model_parallel_world_size()
            block_n, block_k = (
                self.weight_block_size[0],
                self.weight_block_size[1],
            )
            # NOTE: To ensure proper alignment of the block-wise quantization
            # scales, the output_size of the weights for both the gate and up
            # layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size_per_partition % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1 and intermediate_size_per_partition % block_k != 0:
                # Required by row parallel
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}."
                )

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # BIASES (for models like GPT-OSS that have biased MoE)
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=layer.orig_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=layer.orig_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

        # WEIGHT_SCALES
        if not self.block_quant:
            # For per-tensor quant, the scales are per expert and weight.
            w13_scale_data = torch.ones(num_experts, 2, dtype=torch.float32)
            w2_scale_data = torch.ones(num_experts, dtype=torch.float32)
        else:
            # For block quant, the scales are per block (typically 128x128).
            w13_scale_data = torch.ones(
                num_experts,
                2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
                (hidden_size + block_k - 1) // block_k,
                dtype=torch.float32,
            )
            w2_scale_data = torch.ones(
                num_experts,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size_per_partition + block_k - 1) // block_k,
                dtype=torch.float32,
            )
        w13_weight_scale = torch.nn.Parameter(w13_scale_data, requires_grad=False)
        w2_weight_scale = torch.nn.Parameter(w2_scale_data, requires_grad=False)
        # Note: name is weight_scale for tensor, weight_scale_inv for block.
        layer.register_parameter(f"w13_{self.weight_scale_name}", w13_weight_scale)
        layer.register_parameter(f"w2_{self.weight_scale_name}", w2_weight_scale)

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            if self.block_quant
            else {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            assert not self.block_quant
            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def _setup_kernel(
        self,
        layer: RoutedExperts,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_input_scale: torch.Tensor | None,
        w2_input_scale: torch.Tensor | None,
    ) -> None:
        # Shuffle weights to runtime format.
        w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
            fp8_backend=self.fp8_backend,
            layer=layer,
            w13=w13,
            w2=w2,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            w13_input_scale=w13_input_scale,
            w2_input_scale=w2_input_scale,
        )

        # Replace parameters with updated versions. Note that this helper
        # function ensures the replacement is compatible with RL weight reloads.
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, f"w13_{self.weight_scale_name}", w13_scale)
        replace_parameter(layer, f"w2_{self.weight_scale_name}", w2_scale)

        # AITER backend requires weights to be marked as shuffled.
        if self.fp8_backend == Fp8MoeBackend.AITER:
            layer.w13_weight.is_shuffled = True
            layer.w2_weight.is_shuffled = True

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config:
            assert self.experts_cls is not None
            self.moe_kernel = make_fp8_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                fp8_backend=self.fp8_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
            )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        # Allow for accessing weights and scales in standard way.
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = getattr(layer, f"w13_{self.weight_scale_name}")
        w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")
        w13_input_scale = layer.w13_input_scale
        w2_input_scale = layer.w2_input_scale

        # MI300x and MI325x use FNUZ format for FP8. Convert if needed.
        if current_platform.is_fp8_fnuz():
            w13, w13_scale, w13_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                w13,
                w13_scale,
                w13_input_scale,
            )
            w2, w2_scale, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                w2,
                w2_scale,
                w2_input_scale,
            )

        # Per tensor kernels require single activation scale. Use the max.
        if self.quant_config.activation_scheme == "static":
            assert not self.block_quant
            assert w13_input_scale is not None and w2_input_scale is not None
            w13_input_scale, w2_input_scale = process_fp8_input_tensor_strategy_moe(
                w13_input_scale, w2_input_scale
            )
            replace_parameter(layer, "w13_input_scale", w13_input_scale)
            replace_parameter(layer, "w2_input_scale", w2_input_scale)

        # Per tensor kernels require single weight scale for w13 per expert, but
        # on disk there is a scale for w1 and w3. Use the max to requantize.
        if not self.block_quant:
            shard_size = layer.intermediate_size_per_partition
            w13, w13_scale = process_fp8_weight_tensor_strategy_moe(
                w13, w13_scale, shard_size, layer.local_num_experts
            )

        # Shuffle weights to runtime format and setup kernel.
        self._setup_kernel(
            layer, w13, w2, w13_scale, w2_scale, w13_input_scale, w2_input_scale
        )

        # Decide whether the MoE monokernel fast path is eligible for this
        # layer. The monokernel supports two Qwen3.5 FP8 block-wise shapes
        # (both E=256, top_k>1), distinguished by w13_weight = [E, 2*N_half, K]:
        #   * 35B:  2*N_half=1024, K=2048
        #   * 122B: 2*N_half=2048, K=3072
        # It requires the raw (unshuffled) block-wise FP8 weight layout. Only
        # the TRITON backend preserves that layout
        # (convert_to_fp8_moe_kernel_format is a no-op there); backends like
        # DEEPGEMM repack the weights and scales, so they are not eligible.
        # Larger batches (M>8) fall back to the modular kernel inside
        # apply_monolithic.
        #
        # Gated by VLLM_USE_MOE_MONOKERNEL (default on); set it to 0 to force
        # the standard TRITON fused-MoE backend for A/B benchmarking.
        w13_n2 = layer.w13_weight.size(1)  # fused gate+up = 2*N_half
        w13_k = layer.w13_weight.size(2)  # hidden states
        # A shape is supported iff (E, 2*N_half, K) is declared in the generated
        # monokernel registry (csrc/moe/moe_monokernel/shapes.json).  This also
        # subsumes the experts-count check, so new shapes/expert-counts added to
        # shapes.json are enabled here with no edit.
        from vllm.model_executor.layers.fused_moe import monokernel_shapes

        _monokernel_E = getattr(layer, "global_num_experts", 0)
        _monokernel_shape_ok = monokernel_shapes.is_supported(
            _monokernel_E, w13_n2, w13_k
        )
        # The monokernel does plain (optionally bias-corrected) top-k over ALL
        # experts. That equals vLLM's grouped_topk ONLY when the group carve is
        # a no-op — i.e. a single expert group with all groups selected. GLM
        # 5.2 has n_group=1/topk_group=1 (a no-op), so it qualifies; a real
        # multi-group router (e.g. DeepSeek n_group=8) would route to different
        # experts, so we must NOT engage the fast path there.
        _monokernel_grouping_ok = (
            getattr(layer, "num_expert_group", None) in (None, 1)
            and getattr(layer, "topk_group", None) in (None, 1)
        )
        self._use_moe_monokernel = (
            envs.VLLM_USE_MOE_MONOKERNEL
            and self.block_quant
            and self.fp8_backend == Fp8MoeBackend.TRITON
            and _monokernel_shape_ok
            and getattr(layer, "top_k", 1) > 1
            and _monokernel_grouping_ok
        )
        if self._use_moe_monokernel:
            logger.info(
                "MoE monokernel fast path ENABLED for %s "
                "(E=256, 2*N_half=%d, K=%d, top_k=%d, backend=%s)",
                getattr(layer, "prefix", "<no-prefix>"),
                w13_n2,
                w13_k,
                getattr(layer, "top_k", 1),
                self.fp8_backend.value,
            )
            # Pre-compute the BS8 (M<=8) gate/up PAIR interleave now, at
            # load time, so the repack is reserved BEFORE vLLM's memory
            # profiling sizes the KV cache. The repack is cached on the
            # weight tensor's `_tma_interleaved_up_v2` attribute; the
            # monokernel op checks that attribute first, so the decode-time
            # path becomes a cache hit with no allocation. (The M>8 prefill
            # path uses the raw weights and needs no interleave.)
            #
            # Only the 35B kernel consumes the gate/up PAIR-interleaved
            # up-weights. The 122B kernel fetches gate and up with two
            # separate TMAs straight from the raw `[E, 2*N, K]` layout, so it
            # needs NO interleave (and pre-interleaving it would also force a
            # second resident weight copy — ~77 GB — that does not fit on a
            # single H200). Gate by shape so 122B keeps only the raw copy.
            if w13_n2 == 1024 and w13_k == 2048:
                up_interleaved = interleave_for_tma_wgmma_up_v2(
                    layer.w13_weight
                ).contiguous()
                with contextlib.suppress(AttributeError, RuntimeError):
                    layer.w13_weight._tma_interleaved_up_v2 = up_interleaved

            # Move the monokernel scratchpad to the weight device now, at
            # load time. Doing the host->device move lazily inside
            # apply_monolithic would run during CUDA graph capture, where a
            # pageable H2D copy synchronizes the stream (illegal during
            # capture -> hang) and the buffer would not have a stable
            # address for graph replay.
            self.moe_monokernel_scratchpad = self.moe_monokernel_scratchpad.to(
                layer.w13_weight.device
            )

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalizeModular | None:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic. This function should not be called."
        )

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
        w1_scale = getattr(layer, f"w13_{self.weight_scale_name}")
        w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")
        a1_scale = layer.w13_input_scale
        a2_scale = layer.w2_input_scale

        quant_config = make_fp8_moe_quant_config(
            fp8_backend=self.fp8_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=self.weight_block_size,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
            gemm1_alpha=getattr(layer, "swiglu_alpha", None),
            gemm1_beta=getattr(layer, "swiglu_beta", None),
        )

        # Inject biases into the quant config if the model has them
        # (e.g. GPT-OSS biased MoE)
        if quant_config is not None and self.moe.has_bias:
            w13_bias = getattr(layer, "w13_bias", None)
            w2_bias = getattr(layer, "w2_bias", None)
            if w13_bias is not None:
                quant_config._w1.bias = w13_bias
            if w2_bias is not None:
                quant_config._w2.bias = w2_bias

        return quant_config

    @property
    def supports_eplb(self) -> bool:
        return True

    @property
    def is_monolithic(self) -> bool:
        # Route eligible layers through apply_monolithic so the MoE monokernel
        # fast path (which does its own routing from router_logits) can run.
        # Otherwise defer to the base-class decision (driven by the selected
        # modular kernel).
        if getattr(self, "_use_moe_monokernel", False):
            return True
        return super().is_monolithic

    # NOTE: the MoE runner runs shared experts unconditionally
    # via `_maybe_apply_shared_experts` at the top of `_apply_quant_method`
    # (before the monolithic/modular branch), so the monokernel routed-only
    # path no longer risks skipping shared experts — and the base class no
    # longer exposes `mk_owns_shared_expert` to override.

    def _apply_modular_fallback(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Route + run the standard modular kernel.

        Used when the monokernel fast path is enabled for this layer but the
        current batch does not satisfy the kernel's constraints (e.g. M > 64
        during prefill).
        """
        assert self.moe_kernel is not None
        # select_experts' plain top-k branch is softmax-only; sigmoid scoring
        # and/or a selection bias (e.g. MiniMax M2, GLM noaux_tc) must go
        # through its grouped_topk branch, which handles both.  With a single
        # group and topk_group=1 the group carve is a no-op, so this is
        # exactly plain biased/sigmoid top-k (the same no-op carve GLM
        # declares explicitly via use_grouped_topk=True, n_group=1).
        needs_grouped = (
            layer.use_grouped_topk
            or layer.scoring_func != "softmax"
            or layer.e_score_correction_bias is not None
        )
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=layer.top_k,
            use_grouped_topk=needs_grouped,
            renormalize=layer.renormalize,
            topk_group=layer.topk_group or 1,
            num_expert_group=layer.num_expert_group or 1,
            custom_routing_function=layer.custom_routing_function,
            scoring_func=layer.scoring_func,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts_input=None,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic

        # MoE monokernel fast path (Qwen3.5 FP8 block-wise, E=256, top_k>1:
        # 35B = 2*N_half=1024/K=2048, 122B = 2*N_half=2048/K=3072). The
        # kernel only supports M<=8; larger batches (e.g. prefill) fall back
        # to the modular kernel.
        if getattr(self, "_use_moe_monokernel", False):
            M = x.size(0)
            if M <= 8:
                # Scratchpad was moved to the weight device at load time
                # (see process_weights_after_loading) so no host->device
                # copy happens here — that would be illegal during CUDA
                # graph capture.
                top_k = layer.top_k
                scoring_func = getattr(layer, "scoring_func", "softmax")
                renormalize = getattr(layer, "renormalize", True)
                # One-shot capture of the EXACT tensors handed to the op, so
                # the failure can be replayed offline against the Python
                # reference (set MONOKERNEL_DUMP=<path> to enable; dumps the
                # first decode-shaped call only).
                import os as _os

                _dump = _os.environ.get("MONOKERNEL_DUMP")
                if _dump and not getattr(Fp8MoEMethod, "_mono_dumped", False):
                    Fp8MoEMethod._mono_dumped = True
                    torch.save(
                        {
                            "x": x.detach().cpu(),
                            "router_logits": router_logits.detach().cpu(),
                            "w13_weight": layer.w13_weight.detach().cpu(),
                            "w13_scale": getattr(
                                layer, f"w13_{self.weight_scale_name}"
                            ).detach().cpu(),
                            "w2_weight": layer.w2_weight.detach().cpu(),
                            "w2_scale": getattr(
                                layer, f"w2_{self.weight_scale_name}"
                            ).detach().cpu(),
                            "top_k": top_k,
                            "scoring_func": scoring_func,
                            "renormalize": renormalize,
                        },
                        _dump,
                    )
                    logger.info("MONOKERNEL_DUMP written to %s", _dump)
                # GLM-style routing controls. `e_score_correction_bias`
                # (noaux_tc selection bias) and `routed_scaling_factor` are
                # folded into the kernel's routing so the monokernel matches
                # grouped_topk exactly. Only valid when grouping is a no-op
                # (num_expert_group <= 1); the eligibility gate in
                # process_weights_after_loading enforces that, so a real
                # multi-group model never reaches here with a bias. None /
                # 1.0 (the Qwen case) leave routing byte-identical.
                expert_bias = getattr(layer, "e_score_correction_bias", None)
                routed_scaling_factor = getattr(
                    layer, "routed_scaling_factor", 1.0
                )
                return torch.ops.vllm.moe_monokernel_topk(
                    x,
                    router_logits,
                    layer.w13_weight,
                    getattr(layer, f"w13_{self.weight_scale_name}"),
                    layer.w2_weight,
                    getattr(layer, f"w2_{self.weight_scale_name}"),
                    self.moe_monokernel_scratchpad,
                    top_k,
                    scoring_func,
                    renormalize,
                    expert_bias,
                    routed_scaling_factor,
                )
            # Fall back to the modular kernel for large batches.
            return self._apply_modular_fallback(layer, x, router_logits)

        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )


# TODO(future PR): remove this class in favor of
# online/fp8.py::Fp8PerTensorOnlineMoEMethod
class Fp8OnlineMoEMethod(Fp8MoEMethod):
    """MoE method for online FP8 quantization.
    Supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    uses_meta_device: bool = True

    def __init__(self, quant_config: Fp8Config, layer: RoutedExperts):
        super().__init__(quant_config, layer)
        assert not quant_config.is_checkpoint_fp8_serialized
        assert quant_config.activation_scheme == "dynamic"
        assert quant_config.weight_block_size is None

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                device="meta",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                device="meta",  # materialized and processed during loading
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # BIASES (for models like GPT-OSS that have biased MoE)
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    device="meta",  # materialized and processed during loading
                    dtype=layer.orig_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size,
                    device="meta",  # materialized and processed during loading
                    dtype=layer.orig_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

        initialize_online_processing(layer)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        # TODO(@ksayers): inplace fp8 quant kernel, initialize scales with ones
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        fp8_dtype = current_platform.fp8_dtype()
        w13 = torch.empty_like(layer.w13_weight, dtype=fp8_dtype)
        w2 = torch.empty_like(layer.w2_weight, dtype=fp8_dtype)
        w13_scale = torch.ones(
            layer.num_experts, device=w13.device, dtype=torch.float32
        )
        w2_scale = torch.ones(layer.num_experts, device=w2.device, dtype=torch.float32)
        layer.w13_input_scale = None
        layer.w2_input_scale = None

        for expert in range(layer.local_num_experts):
            w13[expert, :, :], w13_scale[expert] = ops.scaled_fp8_quant(
                layer.w13_weight[expert, :, :]
            )
            w2[expert, :, :], w2_scale[expert] = ops.scaled_fp8_quant(
                layer.w2_weight[expert, :, :]
            )

        # Shuffle weights to runtime format and setup kernel.
        self._setup_kernel(
            layer,
            w13,
            w2,
            w13_scale,
            w2_scale,
            w13_input_scale=layer.w13_input_scale,
            w2_input_scale=layer.w2_input_scale,
        )

        # Prevent duplicate processing (e.g., during weight reload)
        layer._already_called_process_weights_after_loading = True


class Fp8KVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)
