import pytest

from src.verification.computation import (
    DeclaredComputation,
    LeafReader,
    ProductKind,
    ProductSpec,
    TranscriptView,
)
from src.verification.instances.mlp import MLPComputation


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


@pytest.mark.parametrize("kw", [
    dict(m=0, a_shape=(2, 3), b_shape=(3, 4)),
    dict(m=1, a_shape=(2, 3), b_shape=(4, 4)),
    dict(m=1, a_shape=(2, 3, 1), b_shape=(3, 4)),
])
def test_product_spec_rejects(kw):
    with pytest.raises(ValueError):
        ProductSpec(name="x", kind=ProductKind.FORWARD, **kw)


def test_layout_partitions_leaf_range():
    c = MLPComputation((5, 7, 6, 3), n_s=3)
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
    c = MLPComputation((5, 7, 3), n_s=2)
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
            return {self.weight(1): self.m_of("Y_1")}  # a forward product, not G_1

    with pytest.raises(ValueError, match="weight gradient"):
        Bad((4, 5, 3), n_s=2)


def test_glue_gradient_weights_complement_linear():
    c = MLPComputation()
    assert set(c.linear_weights) | set(c.glue_gradient_weights) == set(c.weight_names)
    assert not set(c.linear_weights) & set(c.glue_gradient_weights)
    assert isinstance(c, DeclaredComputation)
