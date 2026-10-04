"""Sizing formulas for the repetition count `k` (sizing appendix §2, §6–§8). Pure float math."""

from __future__ import annotations

import math
from dataclasses import dataclass

from verification.config import C_ANTI, F_TARGET, LAMBDA, LOG2_G, Z


def e_m(q: int, eps_in: float, eps_acc: float) -> float:
    """Honest relative error of a product contracting over `q` terms, (2.1)."""
    return math.sqrt(2.0) * eps_in + math.sqrt(q) * eps_acc


def b0(f: float, tau: float, e: float, c: float = C_ANTI) -> float:
    """Per-challenge soundness in bits, `log₂(f/(τ·e)) + log₂(1/c)`, (6.4)."""
    return math.log2(f / (tau * e)) + math.log2(1.0 / c)


def bit_budget(lam: float, T: int, M: int, log2G: float) -> float:
    """Total soundness `N = λ + log₂T + log₂M + log₂G`, (7.1)."""
    return lam + math.log2(T) + math.log2(M) + log2G


def k_required(N: float, b0_bits: float) -> int:
    """Repetition count `⌈N/b₀⌉`, §8 step 9."""
    if not b0_bits > 0:
        raise ValueError(f"b0 = {b0_bits} bits <= 0: no finite k meets the budget")
    return math.ceil(N / b0_bits)


def f_achieved(c: float, tau: float, e: float, N: float, k: int) -> float:
    """Smallest guaranteed-detected deviation, as a fraction of the product, (8.1)."""
    return c * tau * e * 2.0 ** (N / k)


def matmul_count_llama(L: int, n_s: int, n_h: int) -> int:
    """Products per step for the Llama-family block, `M = L(21 + 6·n_s·n_h) + 3` (ref block §5)."""
    return L * (21 + 6 * n_s * n_h) + 3


@dataclass(frozen=True)
class Sizing:
    q_max: int
    eps_in: float
    eps_acc: float
    e_max: float
    tau: float
    f: float
    c: float
    b0: float
    lam: float
    T: int
    M: int
    log2G: float
    N: float
    k: int
    f_achieved: float


def size_k(
    q_max: int,
    eps_in: float,
    eps_acc: float,
    T: int,
    M: int,
    *,
    s_h: float = 1.0,
    z: float = Z,
    f: float = F_TARGET,
    c: float = C_ANTI,
    lam: float = LAMBDA,
    log2G: float = LOG2_G,
) -> Sizing:
    """Run §8 steps 3 and 7–9 at the binding product and report every intermediate."""
    e = e_m(q_max, eps_in, eps_acc)
    tau = z * s_h
    bits = b0(f, tau, e, c)
    N = bit_budget(lam, T, M, log2G)
    k = k_required(N, bits)
    return Sizing(
        q_max=q_max, eps_in=eps_in, eps_acc=eps_acc, e_max=e, tau=tau, f=f, c=c, b0=bits,
        lam=lam, T=T, M=M, log2G=log2G, N=N, k=k, f_achieved=f_achieved(c, tau, e, N, k),
    )
