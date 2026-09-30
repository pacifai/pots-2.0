"""Reproduce the worked values of the sizing appendix §10.1–10.4."""

import math

import pytest
import torch

from src.verification.config import C_ANTI, LAMBDA, LOG2_G, UNIT_ROUNDOFF, Z
from src.verification.sizing import (
    b0, bit_budget, e_m, f_achieved, k_required, matmul_count_llama, size_k,
)

FP32 = UNIT_ROUNDOFF[torch.float32]
BF16 = UNIT_ROUNDOFF[torch.bfloat16]
Q_MAX = 49_152
TAU = Z * 1.0


def test_matmul_count():
    assert matmul_count_llama(30, 4, 9) == 7_113
    assert matmul_count_llama(30, 128, 9) == 207_993


def test_e_m_table():
    # §2 table and §10.1.
    assert e_m(64, FP32, FP32) == pytest.approx(5.6e-7, rel=1e-2)
    assert e_m(576, FP32, FP32) == pytest.approx(1.51e-6, rel=1e-2)
    assert e_m(1536, FP32, FP32) == pytest.approx(2.4e-6, rel=2e-2)
    assert e_m(Q_MAX, FP32, FP32) == pytest.approx(1.330e-5, rel=1e-3)


def test_test_scale_10_1():
    e = e_m(Q_MAX, FP32, FP32)
    bits = b0(1.0, TAU, e)
    assert bits == pytest.approx(13.52, abs=5e-3)
    N = bit_budget(LAMBDA, 10, 7_113, LOG2_G)
    assert N == pytest.approx(93.12, abs=5e-3)
    assert k_required(N, bits) == 7
    assert f_achieved(C_ANTI, TAU, e, N, 7) == pytest.approx(0.86, abs=5e-3)


def test_full_scale_fp32_10_2():
    s = size_k(Q_MAX, FP32, FP32, T=2**20, M=matmul_count_llama(30, 128, 9))
    assert s.N == pytest.approx(114.67, abs=5e-3)
    assert s.k == 9
    assert s.f_achieved == pytest.approx(0.58, abs=5e-3)


def test_full_scale_bf16_10_3():
    s = size_k(Q_MAX, BF16, FP32, T=2**20, M=207_993)
    assert s.e_max == pytest.approx(5.538e-3, rel=1e-3)
    assert s.b0 == pytest.approx(4.82, abs=5e-3)
    assert s.k == 24
    assert s.f_achieved == pytest.approx(0.97, abs=5e-3)


def test_size_k_test_scale_intermediates():
    s = size_k(Q_MAX, FP32, FP32, T=10, M=7_113)
    assert (s.tau, s.f, s.c, s.lam, s.log2G) == (8.0, 1.0, 0.798, 25, 52)
    assert s.b0 == pytest.approx(13.52, abs=5e-3) and s.k == 7


def test_typical_contraction_11_1():
    bits = b0(1.0, TAU, e_m(576, FP32, FP32))
    assert bits == pytest.approx(16.65, abs=1e-2)
    assert k_required(114.67, bits) == 7


def test_log2_inv_c():
    assert math.log2(1 / C_ANTI) == pytest.approx(0.326, abs=1e-3)
