"""Which aten ops form matrix products, shared by the prover's capture and the verifier's replay.

The prover's :class:`~verification.prover.capture.MatmulCapture` records the handled ops and
raises on the rejected ones; the verifier's
:class:`~verification.computation.substitution.ProductSubstitution` replaces the handled ones
and raises on the rest. Both read the lists here, so the verifier never imports the prover
(invariant 1).
"""

from __future__ import annotations

import torch

__all__ = [
    "HANDLED_OPS",
    "REJECTED_OPS",
    "TRUSTED_NAMESPACES",
    "ALLOWED_NAMESPACE_OPS",
    "param_storage_map",
]

# Products the capture records. Only the `.default` overload is accepted.
HANDLED_OPS = frozenset({"mm", "bmm", "addmm", "baddbmm"})

# Matmul-like aten ops that must not reach dispatch. Many are CompositeImplicit and normally
# decompose before dispatch (`matmul`, `linear`, `einsum`); they are listed in case a backend
# keeps them whole. Every name must exist in the aten registry (tested).
REJECTED_OPS = frozenset({
    # dense products and their in-place or fused forms
    "dot", "vdot", "inner", "mv", "addmv", "addmv_", "addr", "addr_", "ger", "outer",
    "addbmm", "addbmm_", "addmm_", "baddbmm_", "_addmm_activation", "_compute_linear_combination",
    "_trilinear", "kron", "smm", "hspmm", "sspaddmm",
    "matmul", "matmul_backward", "linalg_matmul", "linalg_vecdot", "linalg_multi_dot",
    "chain_matmul", "tensordot", "einsum", "bilinear", "linear", "linear_backward",
    "_mixed_dtypes_linear", "_cdist_forward", "_cdist_backward",
    "ormqr", "linalg_householder_product",
    "_grouped_mm", "_scaled_grouped_mm", "_scaled_mm", "_int_mm",
    # quantized and packed-weight products
    "_weight_int8pack_mm", "_weight_int4pack_mm", "_weight_int4pack_mm_for_cpu",
    "_weight_int4pack_mm_with_scales_and_zeros", "_dyn_quant_matmul_4bit",
    "fbgemm_linear_fp16_weight", "fbgemm_linear_fp16_weight_fp32_activation",
    "fbgemm_linear_int8_weight", "fbgemm_linear_int8_weight_fp32_activation",
    "_wrapped_quantized_linear_prepacked",
    "mkldnn_linear", "mkldnn_linear_backward", "mkldnn_linear_backward_input",
    "mkldnn_linear_backward_weights",
    # sparse products
    "_sparse_mm", "_sparse_addmm", "sparse_sampled_addmm", "_sparse_sparse_matmul",
    "_sparse_mm_reduce_impl", "_sparse_mm_reduce_impl_backward", "_cslt_sparse_mm",
    "_cslt_sparse_mm_search", "_sparse_semi_structured_mm", "_sparse_semi_structured_addmm",
    "_sparse_semi_structured_linear",
    # fused attention, forward and backward
    "scaled_dot_product_attention", "_scaled_dot_product_attention_math",
    "_scaled_dot_product_attention_math_for_mps",
    "_scaled_dot_product_flash_attention", "_scaled_dot_product_flash_attention_backward",
    "_scaled_dot_product_flash_attention_for_cpu",
    "_scaled_dot_product_flash_attention_for_cpu_backward",
    "_scaled_dot_product_efficient_attention", "_scaled_dot_product_efficient_attention_backward",
    "_scaled_dot_product_cudnn_attention", "_scaled_dot_product_cudnn_attention_backward",
    "_scaled_dot_product_fused_attention_overrideable",
    "_scaled_dot_product_fused_attention_overrideable_backward",
    "_flash_attention_forward", "_flash_attention_backward",
    "_efficient_attention_forward", "_efficient_attention_backward",
    "_cudnn_attention_forward", "_cudnn_attention_backward",
    "_native_multi_head_attention", "_transformer_encoder_layer_fwd", "_triton_multi_head_attention", "_triton_scaled_dot_attention",
    # convolutions
    "convolution", "_convolution", "_convolution_mode", "_convolution_double_backward",
    "convolution_backward", "convolution_overrideable", "convolution_backward_overrideable",
    "conv1d", "conv2d", "conv3d", "conv_tbc", "conv_tbc_backward",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "_conv_depthwise2d", "conv_depthwise3d", "thnn_conv2d",
    "_slow_conv2d_forward", "_slow_conv2d_backward", "slow_conv3d", "slow_conv3d_forward",
    "slow_conv_dilated2d", "slow_conv_dilated3d", "slow_conv_transpose2d", "slow_conv_transpose3d",
    "_nnpack_spatial_convolution", "mkldnn_convolution",
    "cudnn_convolution", "cudnn_convolution_relu", "cudnn_convolution_add_relu",
    "cudnn_convolution_transpose",
    "miopen_convolution", "miopen_convolution_relu", "miopen_convolution_add_relu",
    "miopen_convolution_transpose", "miopen_depthwise_convolution",
    "_mps_convolution", "_mps_convolution_transpose", "mps_convolution_backward",
    "mps_convolution_transpose_backward",
    # recurrent cells and layers
    "_cudnn_rnn", "_cudnn_rnn_backward", "miopen_rnn", "miopen_rnn_backward",
    "mkldnn_rnn_layer", "mkldnn_rnn_layer_backward", "_lstm_mps", "lstm_mps_backward",
    "lstm", "gru", "rnn_tanh", "rnn_relu", "lstm_cell", "gru_cell", "rnn_tanh_cell",
    "rnn_relu_cell", "_thnn_fused_lstm_cell", "_thnn_fused_lstm_cell_backward",
    "_thnn_fused_lstm_cell_backward_impl", "_thnn_fused_gru_cell", "_thnn_fused_gru_cell_backward",
    "_thnn_differentiable_lstm_cell_backward", "_thnn_differentiable_gru_cell_backward",
    "quantized_lstm", "quantized_gru", "quantized_lstm_cell", "quantized_gru_cell",
    "quantized_rnn_tanh_cell", "quantized_rnn_relu_cell",
})

# Namespaces whose ops may run inside a phase. An op from any other namespace raises unless
# its qualified name ("ns::name") is in ALLOWED_NAMESPACE_OPS.
TRUSTED_NAMESPACES = frozenset({"aten", "prims"})
# Empty: HF Llama eager dispatches nothing outside aten. Clear an op here only after checking
# that it forms no matrix product.
ALLOWED_NAMESPACE_OPS: frozenset[str] = frozenset()


def param_storage_map(model: torch.nn.Module) -> dict[int, str]:
    """``{untyped_storage().data_ptr(): name}`` over ``model.named_parameters()``.

    Tied parameters share one storage and appear once, under the first name.
    """
    out: dict[int, str] = {}
    for name, p in model.named_parameters():
        out.setdefault(p.untyped_storage().data_ptr(), name)
    return out
