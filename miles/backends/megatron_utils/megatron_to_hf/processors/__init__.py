from . import quantizer_fake_int
from .padding_remover import remove_padding
from .quantizer_compressed_tensors import quantize_params_compressed_tensors
from .quantizer_fp8 import quantize_params_fp8, quantize_stacked_expert_params_fp8
from .quantizer_mxfp8 import quantize_params_mxfp8
from .quantizer_nvfp4 import quantize_params_nvfp4

__all__ = [
    "remove_padding",
    "quantize_param",
    "quantize_params_fp8",
    "quantize_params_mxfp8",
    "quantize_params_nvfp4",
    "quantize_params_compressed_tensors",
]


def quantize_params(args, megatron_name, converted_named_params, quantization_config, packed_weight_basenames=None):
    if quantization_config is None:
        if quantizer_fake_int.enabled():
            return quantizer_fake_int.change_params(converted_named_params)
        return converted_named_params
    elif quantization_config["quant_method"] == "fp8":
        return quantize_params_fp8(args, megatron_name, converted_named_params, quantization_config)
    elif quantization_config["quant_method"] == "mxfp8":
        return quantize_params_mxfp8(args, megatron_name, converted_named_params, quantization_config)
    elif quantization_config.get("quant_algo") == "NVFP4" or quantization_config["quant_method"] == "nvfp4":
        return quantize_params_nvfp4(args, megatron_name, converted_named_params, quantization_config)
    elif quantization_config["quant_method"] == "compressed-tensors":
        assert (
            packed_weight_basenames is not None
        ), "compressed-tensors quantization needs the checkpoint's packed weight names"
        return quantize_params_compressed_tensors(converted_named_params, quantization_config, packed_weight_basenames)


def quantize_stacked_expert_params(args, megatron_name, converted_named_params, quantization_config):
    """quantize_params for one matrix of several routed experts, stacked along dim 0 under the first expert's
    names. Returns stacked tensors in the order quantize_params returns them for one expert, or None when the
    quantization has no stacked form."""
    if quantization_config is None:
        if quantizer_fake_int.enabled():
            return quantizer_fake_int.change_stacked_params(converted_named_params)
        return converted_named_params
    if quantization_config["quant_method"] == "fp8":
        return quantize_stacked_expert_params_fp8(args, megatron_name, converted_named_params, quantization_config)
    return None
