"""The worked values of the sizing appendix §10.1–10.3 and §11."""

import math

import pytest
import torch

from verification.computation.instances.decoder import matmul_count_decoder
from verification.parameters import LAMBDA, LOG2_G, UNIT_ROUNDOFF
from verification.verifier.matmul_check.sizing import (
    C_ANTI, Z, b0, bit_budget, e_m, f_achieved, k_required, product_e_m, size_k, term_ratio,
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
    M = matmul_count_decoder(16, 128, 32)
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


def test_e_m_splits_rounding_from_accumulation():
    """F18: ``e_m = √2·ε_in + √q·ε_acc·max(1, U/‖P‖_F)``. Ratio 1 (the default) and any ratio
    below it give the pre-F18 value bit for bit; above 1 only the accumulation term grows."""
    old = math.sqrt(2) * BF16 + math.sqrt(576) * FP32
    assert e_m(576, BF16, FP32) == e_m(576, BF16, FP32, 1.0) == e_m(576, BF16, FP32, 0.3) == old
    assert e_m(576, BF16, FP32, 150.0) == math.sqrt(2) * BF16 + math.sqrt(576) * FP32 * 150.0
    assert math.isnan(e_m(576, BF16, FP32, math.nan))


def test_term_ratio_and_product_e_m():
    assert term_ratio(2.0, 3.0) == 1.5
    assert term_ratio(0.0, 0.0) == 1.0  # no terms, nothing cancels
    assert term_ratio(0.0, 1.0) == math.inf
    assert math.isnan(term_ratio(1.0, math.nan)) and math.isnan(term_ratio(math.nan, 1.0))
    assert product_e_m(64, FP32, FP32, 2.0, 1.0) == e_m(64, FP32, FP32)
    assert product_e_m(64, FP32, FP32, 2.0, 6.0) == e_m(64, FP32, FP32, 3.0)


def test_size_k_takes_the_larger_of_e_qmax_and_the_calibration_e_m():
    """P10d at F18: ``e = max(e_m(q_max), the largest calibration e_m)``; both are kept."""
    base = size_k(Q_MAX, FP32, FP32, T=10, M=7_113, s_h=5.5)
    assert base.e_qmax == base.e_max == e_m(Q_MAX, FP32, FP32) and base.e_cal is None
    small = size_k(Q_MAX, FP32, FP32, T=10, M=7_113, s_h=5.5, e_cal=1e-6, e_cal_at="x")
    assert small.e_max == base.e_max and small.k == base.k and small.e_cal == 1e-6
    big = size_k(Q_MAX, FP32, FP32, T=10, M=7_113, s_h=5.5, e_cal=100 * base.e_max,
                 e_cal_at="L1.dQ[0,3] (step 1)")
    assert big.e_max == 100 * base.e_max and big.e_qmax == base.e_max
    assert big.b0 == pytest.approx(base.b0 - math.log2(100))
    assert big.k > base.k and big.e_cal_at == "L1.dQ[0,3] (step 1)"
    with pytest.raises(ValueError, match="largest calibration e_m"):
        size_k(Q_MAX, FP32, FP32, T=10, M=7_113, s_h=5.5, e_cal=math.inf)
