import dataclasses

import pytest
import torch

from verification.computation.instances.mlp import MLPComputation
from verification.computation.interface import (
    DeclaredComputation,
    ProductKind,
    ProductSpec,
    chain_gradient,
)
from verification.transcript.reader import LeafReader, TranscriptView


class ListReader:
    def __init__(self, leaves):
        self._leaves = list(leaves)

    def leaf(self, index):
        return self._leaves[index]


def test_product_spec_shapes():
    p = ProductSpec(1, "Y", ProductKind.FORWARD, (4, 16), (16, 32), "w")
    assert p.p_shape == (4, 32)
    assert p.q == 16
    assert p.width == 32  # column count of P: r multiplies P on the right
    assert p.layer is None and p.member is None and p.bias is None
    assert ProductKind.OPERAND_GRAD.value == "operand_grad"
    biased = dataclasses.replace(p, bias="b")
    assert (biased.p_shape, biased.q) == ((4, 32), 17)  # [A | 1]·[B ; bᵀ] (F14)


def _biased_mlp(bias_shape=None, *, on="Y_1", **spec):
    """The MLP with a weight ``bias_1`` whose gradient is glue, added by product ``on``."""
    class Biased(MLPComputation):
        @property
        def weight_names(self):
            return (*super().weight_names, "bias_1")

        @property
        def weight_shapes(self):
            return {**super().weight_shapes, "bias_1": bias_shape or (self.widths[1],)}

        @property
        def products(self):
            return tuple(dataclasses.replace(p, **{"bias": "bias_1", **spec}) if p.name == on
                         else p for p in super().products)

    return Biased((4, 5, 3), n_s=2, eta=0.1)


def test_validate_accepts_a_bias_whose_gradient_is_glue():
    c = _biased_mlp()
    y1 = c.product(c.m_of("Y_1"))
    assert (y1.bias, y1.q) == ("bias_1", 5)
    assert "bias_1" in c.glue_gradient_weights


@pytest.mark.parametrize("kw,match", [
    (dict(bias="nope"), "not a declared weight"),
    (dict(bias="layers.0.weight"), "not a declared weight whose gradient is glue"),
    (dict(bias_shape=(4,)), "doesn't fit a product of width 5"),
    (dict(on="Y_2"), "doesn't fit a product of width 3"),
    (dict(member=(0, 0)), "doesn't fit a member"),
    (dict(on="dX_2"), r"dX_2 \(input_grad, .*only a forward product with a weight"),
    (dict(on="G_2"), r"G_2 \(weight_grad, .*only a forward product with a weight"),
    (dict(weight=None), r"Y_1 \(forward, weight None\): only a forward product with a weight"),
])
def test_validate_rejects_a_misdeclared_bias(kw, match):
    with pytest.raises(ValueError, match=match):
        _biased_mlp(**kw)


@pytest.mark.parametrize("kw", [
    dict(m=0, a_shape=(2, 3), b_shape=(3, 4)),
    dict(m=1, a_shape=(2, 3), b_shape=(4, 4)),
    dict(m=1, a_shape=(2, 3, 1), b_shape=(3, 4)),
])
def test_product_spec_rejects(kw):
    with pytest.raises(ValueError):
        ProductSpec(name="x", kind=ProductKind.FORWARD, **kw)


def test_layout_partitions_leaf_range():
    c = MLPComputation((5, 7, 6, 3), n_s=3, eta=0.1)
    idx = [c.record_index(i) for i in range(c.n_s)]
    idx += [c.w_t_index(n) for n in c.weight_names]
    idx += [c.product_index(m) for m in range(1, c.M + 1)]
    idx += [c.w_next_index(n) for n in c.weight_names]
    assert idx == list(range(c.n_leaves))
    assert c.n_leaves == c.n_s + 2 * c.n_w + c.M
    with pytest.raises(IndexError):
        c.product_index(c.M + 1)
    with pytest.raises(IndexError):
        c.record_index(c.n_s)


def test_transcript_view_names_leaves():
    c = MLPComputation((5, 7, 3), n_s=2, eta=0.1)
    leaves = [f"leaf{i}" for i in range(c.n_leaves)]
    r = ListReader(leaves)
    assert isinstance(r, LeafReader)
    v = TranscriptView(c, r)
    assert v.records() == ["leaf0", "leaf1"]
    assert v.w_t(c.weight_names[1]) == "leaf3"
    assert v.product(1) == "leaf4"
    assert v.w_next(c.weight_names[0]) == leaves[c.n_s + c.n_w + c.M]


def test_validate_rejects_bad_linear_map():
    class Bad(MLPComputation):
        @property
        def linear_weights(self):
            return {self.weight(1): (self.m_of("Y_1"),)}  # a forward product, not G_1

    with pytest.raises(ValueError, match="weight gradient"):
        Bad((4, 5, 3), n_s=2, eta=0.1)


@pytest.mark.parametrize("ms", [(), "twice"])
def test_validate_rejects_empty_or_repeated_linear_products(ms):
    class Bad(MLPComputation):
        @property
        def linear_weights(self):
            lw = dict(super().linear_weights)
            g = self.m_of("G_1")
            lw[self.weight(1)] = (g, g) if ms == "twice" else ()
            return lw

    with pytest.raises(ValueError, match="not distinct products"):
        Bad((4, 5, 3), n_s=2, eta=0.1)


def test_chain_gradient_widens_and_adds_in_order():
    parts = [torch.randn(3, 4, generator=torch.Generator().manual_seed(i)).to(torch.bfloat16)
             for i in range(3)]
    want = (parts[0].float() + parts[1].float()) + parts[2].float()
    assert torch.equal(chain_gradient(iter(parts), torch.float32), want)
    one = torch.randn(2, 2)
    assert chain_gradient([one], torch.float32) is one  # one part of the dtype: itself
    with pytest.raises(ValueError, match="at least one"):
        chain_gradient([], torch.float32)


def test_glue_gradient_weights_complement_linear():
    c = MLPComputation(eta=0.1)
    assert set(c.linear_weights) | set(c.glue_gradient_weights) == set(c.weight_names)
    assert not set(c.linear_weights) & set(c.glue_gradient_weights)
    assert isinstance(c, DeclaredComputation)


def test_validate_rejects_q1_product():
    class OuterProduct(MLPComputation):
        @property
        def products(self):
            ps = list(super().products)
            ps[0] = ProductSpec(1, "Y_1", ProductKind.FORWARD, (2, 1), (1, 5), self.weight(1))
            return tuple(ps)

    with pytest.raises(ValueError, match="P7"):
        OuterProduct((4, 5, 3), n_s=2, eta=0.1)


@pytest.mark.parametrize("eta", [0.0, -1.0, float("inf"), float("nan")])
def test_validate_rejects_bad_eta(eta):
    with pytest.raises(ValueError, match="eta"):
        MLPComputation(eta=eta)
