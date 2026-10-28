import dataclasses

import pytest
import torch

from tests.verification.decoder_helpers import tiny_config
from verification.computation.instances import DecoderComputation
from verification.computation.instances.mlp import MLPComputation
from verification.computation.interface import (
    DeclaredComputation,
    ProductKind,
    ProductLeaf,
    ProductSpec,
    chain_gradient,
)
from verification.transcript.errors import LeafShapeError
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
    idx += [c.product_leaf_index(g) for g in range(c.n_product_leaves)]
    idx += [c.w_next_index(n) for n in c.weight_names]
    assert idx == list(range(c.n_leaves))
    assert c.n_leaves == c.n_s + 2 * c.n_w + c.M  # no groups: K = M
    assert c.n_product_leaves == c.M
    with pytest.raises(IndexError):
        c.product_leaf_index(c.n_product_leaves)
    with pytest.raises(IndexError):
        c.product_leaf_index(-1)
    with pytest.raises(IndexError):
        c.record_index(c.n_s)


def test_transcript_view_names_leaves():
    c = MLPComputation((5, 7, 3), n_s=2, eta=0.1)
    leaves = [f"leaf{i}" for i in range(c.n_leaves)]
    for p in c.products:
        leaves[c.product_leaf_index(p.m - 1)] = torch.zeros(p.p_shape)
    r = ListReader(leaves)
    assert isinstance(r, LeafReader)
    v = TranscriptView(c, r)
    assert v.records() == ["leaf0", "leaf1"]
    assert v.w_t(c.weight_names[1]) == "leaf3"
    assert v.product(1) is leaves[c.n_s + c.n_w]
    assert v.w_next(c.weight_names[0]) == leaves[c.n_s + c.n_w + c.M]


# ---- O18: product leaves -----------------------------------------------------------------


def _tiny_decoder(layers=3, B=1, n_s=2, cls=DecoderComputation):
    config = tiny_config()
    config.num_hidden_layers = layers
    return cls(config, n_s=n_s, n=8, eta=1e-3, head_blocks=B)


def _expected_leaves(c):
    """The product leaves by a plain walk over the inventory."""
    group_at = {g[0]: g for g in c.product_groups}
    out, skip = [], set()
    for p in c.products:
        if p.m in skip:
            continue
        if p.m in group_at:
            _, stride, n = group_at[p.m]
            skip.update(p.m + stride * i for i in range(n))
            out.append((p.m, stride, n, True, (n, *p.p_shape), p.name.split("[")[0]))
        else:
            out.append((p.m, 1, 1, False, p.p_shape, p.name))
    return out


@pytest.mark.parametrize("B", [1, 2])
def test_product_leaves_order_and_shapes(B):
    c = _tiny_decoder(layers=3, B=B)
    got = [(pl.m0, pl.stride, pl.n, pl.grouped, pl.shape, pl.name) for pl in c.product_leaves]
    assert got == _expected_leaves(c)
    assert all(isinstance(pl, ProductLeaf) and pl.g == g for g, pl in enumerate(c.product_leaves))
    assert [pl.m0 for pl in c.product_leaves] == sorted(pl.m0 for pl in c.product_leaves)
    n_mem = c.n_s * c.n_h
    grouped = [pl for pl in c.product_leaves if pl.grouped]
    assert len(grouped) == 6 * 3 and all(pl.n == n_mem for pl in grouped)
    assert {pl.name.split(".")[-1] for pl in grouped} == {"S", "O", "dA", "dV", "dQ", "dK"}
    assert all("[" not in pl.name for pl in grouped)
    assert c.n_product_leaves == 27 * 3 + 3 * B  # 27 L + 3 B
    assert c.n_leaves == c.n_s + 2 * c.n_w + c.n_product_leaves
    assert c.w_next_index(c.weight_names[0]) == c.n_s + c.n_w + c.n_product_leaves
    assert c.w_t_index(c.weight_names[0]) == c.n_s
    assert c.product_leaf_index(0) == c.n_s + c.n_w
    assert c.product_leaf_index(c.n_product_leaves - 1) == c.w_next_index(c.weight_names[0]) - 1


@pytest.mark.parametrize("B", [1, 2])
def test_leaf_of_round_trips_and_each_m_is_in_one_leaf(B):
    c = _tiny_decoder(layers=3, B=B)
    seen = []
    for m in range(1, c.M + 1):
        g, i = c.leaf_of(m)
        pl = c.product_leaf(g)
        assert type(g) is int and (i is None or type(i) is int)
        assert (i is not None) == pl.grouped
        assert pl.ms[0 if i is None else i] == m
        seen.append(m)
    assert sum(pl.n for pl in c.product_leaves) == c.M
    assert sorted(m for pl in c.product_leaves for m in pl.ms) == list(range(1, c.M + 1))
    for bad in (0, c.M + 1, -1):
        with pytest.raises(IndexError, match="outside"):
            c.leaf_of(bad)
    for bad in (-1, c.n_product_leaves):
        with pytest.raises(IndexError):
            c.product_leaf(bad)


def test_one_member_group_is_still_a_group():
    config = tiny_config()
    config.num_hidden_layers = 1
    config.num_attention_heads = config.num_key_value_heads = 1
    c = DecoderComputation(config, n_s=1, n=8, eta=1e-3)
    grouped = [pl for pl in c.product_leaves if pl.grouped]
    assert len(grouped) == 6 and all(pl.n == 1 and len(pl.shape) == 3 for pl in grouped)


def test_decoder_product_groups_are_the_member_runs_sorted():
    c = _tiny_decoder(layers=2)
    n = c.n_s * c.n_h
    assert c.product_groups == tuple(sorted((m0, st, n) for m0, st in c.member_runs.values()))
    assert len(c.product_groups) == 12


def test_decoder_product_groups_raise_for_a_member_outside_every_run():
    class Uneven(DecoderComputation):
        @property
        def member_runs(self):
            runs = dict(super().member_runs)
            runs.pop((1, "S"))
            return runs

    with pytest.raises(ValueError, match="no evenly spaced run"):
        _tiny_decoder(cls=Uneven)


def _with_groups(groups):
    class Bad(DecoderComputation):
        @property
        def product_groups(self):
            return groups(super().product_groups)

    return Bad


@pytest.mark.parametrize("name,edit,match", [
    ("mixed shapes", lambda gs: ((gs[0][0], gs[0][1] + 1, gs[0][2]), *gs[1:]), "differs|overlaps"),
    ("overlap", lambda gs: (*gs, gs[0]), "overlaps"),
    ("ungrouped member", lambda gs: gs[1:], "in no product group"),
    ("empty group", lambda gs: ((gs[0][0], gs[0][1], 0), *gs[1:]), "at least 1"),
    ("zero stride", lambda gs: ((gs[0][0], 0, gs[0][2]), *gs[1:]), "stride"),
    ("out of range", lambda gs: ((10**6, 1, 2), *gs[1:]), "outside|fit"),
    ("a lone product", lambda gs: ((1, 1, 2), *gs[1:]), "no member|differs|overlaps"),
])
def test_validate_rejects_bad_product_groups(name, edit, match):
    with pytest.raises(ValueError, match=match):
        _tiny_decoder(cls=_with_groups(edit))


def test_validate_rejects_a_group_across_layers():
    class Cross(DecoderComputation):
        @property
        def product_groups(self):
            s1, s2 = self.m_of("L1.S[0,0]"), self.m_of("L2.S[0,0]")
            rest = [g for g in super().product_groups if g[0] not in (s1, s2)]
            return (s1, s2 - s1, 2), *rest

    with pytest.raises(ValueError, match="differs"):
        _tiny_decoder(layers=2, cls=Cross)


def test_transcript_view_over_a_decoder():
    c = _tiny_decoder(layers=2)
    leaves = [None] * c.n_leaves
    for pl in c.product_leaves:
        leaves[c.product_leaf_index(pl.g)] = torch.randn(pl.shape)
    v = TranscriptView(c, ListReader(leaves))
    for m in range(1, c.M + 1):
        g, i = c.leaf_of(m)
        leaf = leaves[c.product_leaf_index(g)]
        assert torch.equal(v.product(m), leaf if i is None else leaf[i])
    bad = list(leaves)
    pl = next(pl for pl in c.product_leaves if pl.grouped)
    bad[c.product_leaf_index(pl.g)] = torch.randn(pl.shape[1:])
    with pytest.raises(LeafShapeError):
        TranscriptView(c, ListReader(bad)).product_leaf(pl.g)


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
