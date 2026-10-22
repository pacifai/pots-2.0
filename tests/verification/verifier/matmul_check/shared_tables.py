"""Tables with real shares, for the FS38b level tests (``test_measure_levels_host.py``, and
``test_measure_levels_cuda.py`` on a GPU): ``MeasureInput`` lists built from CPU tensors that are
views of one storage, the cases of ``test_read_plan.py`` with real data."""

import numpy as np
import torch

from tests.verification.verifier.matmul_check.test_measure_core import H, make
from verification.verifier.matmul_check import measure_kernel as mk
from verification.verifier.matmul_check.challenges import challenge_matrix
from verification.verifier.matmul_check.measure_table import MeasureInput, pack

DTYPES = [torch.float32, torch.bfloat16]


def plan_for(mis):
    return mk.read_plan(np.frombuffer(pack(mis).array, dtype=mk.MPROD_DTYPE).copy())


def prod_of(a, b, *, k=3, m=1, bias=None, guard=True, seed=0):
    """``a·b`` (+ bias) against a perturbed product stored in ``a``'s dtype; ``a`` and ``b`` are
    kept as given, so views of a shared storage stay views of it."""
    g = torch.Generator().manual_seed(seed)
    p = a.float() @ b.float()
    if bias is not None:
        p = p + bias.float()
    p = (p + 1e-3 * torch.randn(*p.shape, generator=g)).to(a.dtype)
    r = challenge_matrix(H, m, k, b.shape[1]).contiguous()
    return MeasureInput(a, b, p, r, bias, guard)


def rnd(g, *shape, dtype=torch.float32):
    return torch.randn(*shape, generator=g).to(dtype)


def fwd_and_grad(dtype, guard, seed=1, n=9, q=21, w=17, k=3, biased_beside=False):
    g = torch.Generator().manual_seed(seed)
    x, wt, dy = rnd(g, n, q, dtype=dtype), rnd(g, w, q, dtype=dtype), rnd(g, n, w, dtype=dtype)
    mis = [prod_of(x, wt.t(), k=k, m=1, guard=guard, seed=seed),           # Y = X·Wᵀ
           prod_of(dy.t(), x, k=k, m=2, guard=guard, seed=seed + 1)]       # G = δYᵀ·X
    if biased_beside:
        bias = rnd(g, w, dtype=dtype)
        mis.append(prod_of(x, wt.t(), k=k, m=3, bias=bias, guard=guard, seed=seed + 2))  # A = X
        mis.append(prod_of(dy.t(), x, k=k, m=4, bias=rnd(g, q, dtype=dtype), guard=guard,
                           seed=seed + 3))                                  # biased, B = X
    return mis


def three_forward(dtype, guard):
    g = torch.Generator().manual_seed(4)
    x = rnd(g, 7, 24, dtype=dtype)
    return [prod_of(x, rnd(g, 11, 24, dtype=dtype).t(), m=i + 1, guard=guard, seed=i)
            for i in range(3)]


def member_stack(dtype, guard, T=3, n=8, d=13):
    """Slices ``stack[t]`` of one tensor, each read as ``A`` by the first ``T`` products (S, dA)
    and as ``B`` by the last ``T`` (δK̃, dV)."""
    g = torch.Generator().manual_seed(6)
    stack, other = rnd(g, T, n, d, dtype=dtype), rnd(g, T, n, d, dtype=dtype)
    ds = rnd(g, T, n, n, dtype=dtype)
    return ([prod_of(stack[t], other[t].t(), m=t + 1, guard=guard, seed=t) for t in range(T)]
            + [prod_of(ds[t].t(), stack[t], m=T + t + 1, guard=guard, seed=10 + t)
               for t in range(T)])


def two_sided(dtype, guard=True):
    g = torch.Generator().manual_seed(8)
    w = rnd(g, 10, 10, dtype=dtype)
    return [prod_of(rnd(g, 6, 10, dtype=dtype), w.t(), m=1, guard=guard, seed=1),
            prod_of(rnd(g, 6, 10, dtype=dtype), w, m=2, guard=guard, seed=2)]


def mixed_table(dtype, guard=None):
    """Shares, biased products and plain products in one table, one challenge width; ``guard``
    ``None`` mixes the flag over the products."""
    mis = (fwd_and_grad(dtype, True, seed=1, biased_beside=True)
           + fwd_and_grad(dtype, False, seed=2, n=5, q=33, w=40)
           + [make(6, 19, 37, 3, dtype=dtype, biased=bool(i % 2), guard=bool(i % 3), seed=i)
              for i in range(4)])
    return [MeasureInput(mi.a, mi.b, mi.p, challenge_matrix(H, i + 1, 3, mi.b.shape[1]).contiguous(),
                         mi.bias, mi.guard if guard is None else guard) for i, mi in enumerate(mis)]


def shared_special(dtype, guard, biased=False):
    """Y = X·Wᵀ and G = δYᵀ·X, and X, the storage both read, for the caller to edit."""
    g = torch.Generator().manual_seed(12)
    n, q, w = 4, 6, 40
    x, wt, dy = rnd(g, n, q, dtype=dtype), rnd(g, w, q, dtype=dtype), rnd(g, n, w, dtype=dtype)
    bias = rnd(g, w, dtype=dtype) if biased else None
    mis = [prod_of(x, wt.t(), k=2, m=1, guard=guard, bias=bias, seed=1),
           prod_of(dy.t(), x, k=2, m=2, guard=guard, seed=2)]
    return mis, x


EDITS = {
    "nan": lambda x: x.__setitem__((0, 0), float("nan")),
    "inf": lambda x: x.__setitem__((2, 5), float("inf")),
    "-inf": lambda x: x.__setitem__((3, 1), float("-inf")),
    "zero": lambda x: x.zero_(),
    "overflow": lambda x: x.fill_(2e19),
    "nan row": lambda x: x.__setitem__((1, slice(None)), float("nan")),
}
