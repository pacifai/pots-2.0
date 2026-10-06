"""The worked values of the sizing appendix §10.1–10.3 and §11."""

import math

import pytest
import torch

from verification.computation.instances.llama import matmul_count_llama
from verification.parameters import LAMBDA, LOG2_G, UNIT_ROUNDOFF
from verification.verifier.matmul_check.sizing import (
    C_ANTI, Z, b0, bit_budget, e_m, f_achieved, k_required, size_k,
)

FP32 = UNIT_ROUNDOFF[torch.float32]
BF16 = UNIT_ROUNDOFF[torch.bfloat16]
Q_MAX = 49_152
TAU = Z * 1.0


def test_e_m_table():
    # §2 table and §10.1.
    assert e_m(64, FP32, FP32) == pytest.approx(5.6e-7, rel=1e-2)
    assert e_m(576, FP32, FP32) == pytest.approx(1.51e-6, rel=1e-2)
    assert e_m(1536, FP32, FP32) == pytest.approx(2.4e-6, rel=2e-2)
    assert e_m(Q_MAX, FP32, FP32) == pytest.approx(1.330e-5, rel=1e-3)


def test_test_scale_10_1():
    e = e_m(Q_MAX, FP32, FP32)
    bits = b0(1.0, TAU, e)
    assert bits == pytest.approx(13.49, abs=5e-3)
    N = bit_budget(LAMBDA, 10, 7_113, LOG2_G)
    assert N == pytest.approx(93.12, abs=5e-3)
    assert k_required(N, bits) == 7
    assert f_achieved(C_ANTI, TAU, e, N, 7) == pytest.approx(0.878, abs=5e-3)


def test_test_scale_measured_tau_10_1():
    s = size_k(Q_MAX, FP32, FP32, T=10, M=7_113, s_h=5.5)
    assert s.b0 == pytest.approx(11.03, abs=5e-3)
    assert s.k == 9
    assert s.f_achieved == pytest.approx(0.622, abs=5e-3)


def test_full_scale_bf16_10_2():
    # Llama-3.2-1B: 16 layers, 128 sequences, 32 heads; vocabulary 128,256.
    M = matmul_count_llama(16, 128, 32)
    assert M == 393_555
    s = size_k(128_256, BF16, FP32, T=10, M=M)
    assert s.e_max == pytest.approx(5.546e-3, rel=1e-3)
    assert s.b0 == pytest.approx(4.787, abs=5e-3)
    assert s.N == pytest.approx(98.91, abs=5e-3)
    assert s.k == 21
    assert s.f_achieved == pytest.approx(0.948, abs=5e-3)


def test_size_k_test_scale_intermediates():
    s = size_k(Q_MAX, FP32, FP32, T=10, M=7_113)
    assert (s.tau, s.f, s.c, s.lam, s.log2G) == (8.0, 1.0, math.sqrt(2 / 3), 25, 52)
    assert s.b0 == pytest.approx(13.49, abs=5e-3) and s.k == 7


def test_typical_contraction_11_1():
    bits = b0(1.0, TAU, e_m(576, FP32, FP32))
    assert bits == pytest.approx(16.62, abs=1e-2)
    assert k_required(93.12, bits) == 6


@pytest.mark.parametrize("bits", [0.0, -1.0, float("nan")])
def test_k_required_rejects_nonpositive_b0(bits):
    with pytest.raises(ValueError, match="no finite k"):
        k_required(93.12, bits)


def test_log2_inv_c():
    assert math.log2(1 / C_ANTI) == pytest.approx(0.2925, abs=1e-3)
