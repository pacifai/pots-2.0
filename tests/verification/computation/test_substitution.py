"""The product-substitution dispatch mode (A8)."""

import pytest
import torch

from verification.computation.substitution import ProductSubstitution, SubstitutionError


def test_products_are_replaced_and_glue_runs():
    a, b = torch.randn(3, 4), torch.randn(4, 5)
    seen = []

    def supply(op, x, y, bias):
        seen.append((op, x, y, bias))
        return torch.full((3, 5), 7.0)

    with ProductSubstitution(supply):
        out = torch.relu(a @ b - 1)
    assert torch.equal(out, torch.full((3, 5), 6.0))
    assert len(seen) == 1 and seen[0][0] == "mm" and seen[0][3] is None
    assert torch.equal(seen[0][1], a) and torch.equal(seen[0][2], b)


def test_biased_linear_is_replaced_whole():
    """F14: ``addmm(b, X, Wᵀ)`` hands over its bias; the supplied tensor stands for the whole
    output, bias included."""
    x, w, bias = torch.randn(2, 5, 4), torch.randn(7, 4), torch.randn(7)
    seen = []

    def supply(op, a, b, bv):
        seen.append((op, tuple(a.shape), tuple(b.shape), bv))
        return torch.full((10, 7), 3.0)

    with ProductSubstitution(supply):
        y = torch.nn.functional.linear(x, w, bias)
    assert torch.equal(y, torch.full((2, 5, 7), 3.0))
    assert [s[:3] for s in seen] == [("addmm", (10, 4), (4, 7))] and seen[0][3] is bias


def test_dtype_copies_are_reported():
    """F5a: every ``aten._to_copy`` runs as normal and reaches ``copied``; the product it feeds
    is still substituted."""
    x, w = torch.randn(3, 4), torch.randn(5, 4)
    seen, copies = [], []

    def supply(op, a, b, bias):
        seen.append((a.dtype, b.dtype))
        return torch.zeros(3, 5, dtype=a.dtype)

    with ProductSubstitution(supply, copied=lambda src, out: copies.append((src, out))):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = torch.nn.functional.linear(x, w)
    assert y.dtype == torch.bfloat16 and seen == [(torch.bfloat16, torch.bfloat16)]
    assert any(src is w and torch.equal(out, w.to(torch.bfloat16)) for src, out in copies)
    assert any(src is x for src, _ in copies)


def test_bmm_through_matmul_and_linear():
    q, k = torch.randn(2, 3, 6, 4), torch.randn(2, 3, 6, 4)
    x, w = torch.randn(2, 5, 4), torch.randn(7, 4)
    calls = []

    def supply(op, a, b, bias):
        calls.append((op, tuple(a.shape), tuple(b.shape)))
        return torch.zeros(*a.shape[:-1], b.shape[-1])

    with ProductSubstitution(supply):
        s = torch.matmul(q, k.transpose(2, 3))
        y = torch.nn.functional.linear(x, w)
    assert s.shape == (2, 3, 6, 6) and y.shape == (2, 5, 7)
    assert calls == [("bmm", (6, 6, 4), (6, 4, 6)), ("mm", (10, 4), (4, 7))]


def test_outer_product_is_glue():
    """P7: q = 1 is not substituted."""
    a, b = torch.randn(4, 1), torch.randn(1, 3)

    def supply(op, x, y, bias):
        raise AssertionError("q = 1 reached supply")

    with ProductSubstitution(supply):
        out = a @ b
    assert torch.equal(out, a * b)


@pytest.mark.parametrize("bad", [torch.zeros(3, 6), torch.zeros(3, 5, dtype=torch.float64)],
                         ids=["shape", "dtype"])
def test_bad_supplied_tensor_raises(bad):
    with pytest.raises(SubstitutionError, match="supplied"):
        with ProductSubstitution(lambda op, a, b, bias: bad):
            torch.randn(3, 4) @ torch.randn(4, 5)


@pytest.mark.parametrize("fn", [
    lambda: torch.addmm(torch.zeros(3, 5), torch.randn(3, 4), torch.randn(4, 5)),
    lambda: torch.addmm(torch.randn(5), torch.randn(3, 4), torch.randn(4, 5), beta=2),
    lambda: torch.addmm(torch.randn(5), torch.randn(3, 4), torch.randn(4, 5), alpha=2),
    lambda: torch.baddbmm(torch.randn(5), torch.randn(2, 3, 4), torch.randn(2, 4, 5)),
    lambda: torch.dot(torch.randn(4), torch.randn(4)),
    lambda: torch.mv(torch.randn(3, 4), torch.randn(4)),
], ids=["addmm_2d_bias", "addmm_beta", "addmm_alpha", "baddbmm", "dot", "mv"])
def test_other_matmul_ops_raise(fn):
    with pytest.raises(SubstitutionError):
        with ProductSubstitution(lambda op, a, b, bias: torch.zeros(*a.shape[:-1], b.shape[-1])):
            fn()
