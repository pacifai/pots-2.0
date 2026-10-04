"""The matmul op lists: every matmul-like aten op is handled, rejected, or known not to be one."""

import re

import torch

from verification.computation.matmul_ops import HANDLED_OPS, REJECTED_OPS

# `dir(torch.ops.aten)` lists only packets already touched, so read the full registry.
ATEN_OPS = frozenset(
    n.split("::", 1)[1].split(".", 1)[0]
    for n in torch._C._dispatch_get_all_op_names() if n.startswith("aten::"))

MATMUL_LIKE = re.compile(
    r"mm|matmul|linear|conv|attention|dot|rnn|lstm|gru|outer|addr|mv$"
    r"|transform|sdp|kron|householder|ormqr|einsum|inner|^ger$")

# Regex hits that compute no matrix product: name collisions, weight packing and layout.
NOT_MATMUL = frozenset({
    # "mm", "linear", "conv" inside unrelated names
    "digamma", "digamma_", "polygamma", "polygamma_", "lgamma", "lgamma_", "mvlgamma",
    "mvlgamma_", "igamma", "igamma_", "igammac", "igammac_", "_foreach_lgamma",
    "_foreach_lgamma_", "special_digamma", "special_polygamma", "special_gammaln",
    "special_multigammaln", "special_gammainc", "special_gammaincc", "_standard_gamma",
    "_standard_gamma_grad", "hamming_window", "cummax", "cummin", "_cummax_helper",
    "_cummin_helper", "cummaxmin_backward", "_nested_get_jagged_dummy",
    "_convert_indices_from_coo_to_csr", "_convert_indices_from_csr_to_coo",
    "upsample_linear1d", "upsample_linear1d_backward", "upsample_bilinear2d",
    "upsample_bilinear2d_backward", "upsample_trilinear3d", "upsample_trilinear3d_backward",
    "_upsample_bilinear2d_aa", "_upsample_bilinear2d_aa_backward",
    # weight packing, reordering and quantization helpers: no product is formed
    "_convert_weight_to_int4pack", "_convert_weight_to_int4pack_for_cpu",
    "fbgemm_linear_quantize_weight", "fbgemm_pack_gemm_matrix_fp16",
    "fbgemm_pack_quantized_matrix", "_wrapped_linear_prepack",
    "mkldnn_reorder_conv2d_weight", "mkldnn_reorder_conv3d_weight", "_cudnn_rnn_flatten_weight",
    "_use_cudnn_rnn_flatten_weight",
    # hit by "sdp": picks a fused-attention backend and returns an enum, no tensor math
    "_fused_sdp_choice",
    # hit by "transform": splits a fused qkv tensor, adds its bias and scales q; elementwise
    "_transform_bias_rescale_qkv",
})


def test_rejected_names_exist():
    assert not (REJECTED_OPS - ATEN_OPS), sorted(REJECTED_OPS - ATEN_OPS)
    assert not (HANDLED_OPS - ATEN_OPS)
    assert not (HANDLED_OPS & REJECTED_OPS)


def test_every_matmul_like_op_is_classified():
    hits = {n for n in ATEN_OPS if MATMUL_LIKE.search(n)}
    assert len(hits) > 100, "registry scan found too little"
    unclassified = hits - HANDLED_OPS - REJECTED_OPS - NOT_MATMUL
    assert not unclassified, sorted(unclassified)
    assert not (NOT_MATMUL - ATEN_OPS), sorted(NOT_MATMUL - ATEN_OPS)
