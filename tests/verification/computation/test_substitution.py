"""The product-substitution dispatch mode (A8)."""

import pytest
import torch

from verification.computation.substitution import ProductSubstitution, SubstitutionError


def test_products_are_replaced_and_glue_runs():
    a, b = torch.randn(3, 4), torch.randn(4, 5)
    seen = []

    def supply(op, x, y):
        seen.append((op, x, y))
        return torch.full((3, 5), 7.0)

    with ProductSubstitution(supply):
        out = torch.relu(a @ b - 1)
    assert torch.equal(out, torch.full((3, 5), 6.0))
    assert len(seen) == 1 and seen[0][0] == "mm"
    assert torch.equal(seen[0][1], a) and torch.equal(seen[0][2], b)


def test_bmm_through_matmul_and_linear():
    q, k = torch.randn(2, 3, 6, 4), torch.randn(2, 3, 6, 4)
    x, w = torch.randn(2, 5, 4), torch.randn(7, 4)
    calls = []

    def supply(op, a, b):
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

    def supply(op, x, y):
        raise AssertionError("q = 1 reached supply")

    with ProductSubstitution(supply):
        out = a @ b
    assert torch.equal(out, a * b)


@pytest.mark.parametrize("bad", [torch.zeros(3, 6), torch.zeros(3, 5, dtype=torch.float64)],
                         ids=["shape", "dtype"])
def test_bad_supplied_tensor_raises(bad):
    with pytest.raises(SubstitutionError, match="supplied"):
        with ProductSubstitution(lambda op, a, b: bad):
            torch.randn(3, 4) @ torch.randn(4, 5)


@pytest.mark.parametrize("fn", [
    lambda: torch.addmm(torch.zeros(3, 5), torch.randn(3, 4), torch.randn(4, 5)),
    lambda: torch.dot(torch.randn(4), torch.randn(4)),
    lambda: torch.mv(torch.randn(3, 4), torch.randn(4)),
], ids=["addmm", "dot", "mv"])
def test_other_matmul_ops_raise(fn):
    with pytest.raises(SubstitutionError):
        with ProductSubstitution(lambda op, a, b: torch.zeros(*a.shape[:-1], b.shape[-1])):
            fn()
