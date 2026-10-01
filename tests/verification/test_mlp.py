import struct

import pytest
import torch

from src.verification.capture import MatmulCapture, param_storage_map
from src.verification.computation import LabelingError, ProductKind
from src.verification.encoding import TAG_MLP_RECORD, encode_tensor_leaf
from src.verification.instances.mlp import (
    DEFAULT_WIDTHS,
    MLPComputation,
    init_weights,
    make_record,
    sequential_schedule,
    split_record,
    synthetic_dataset,
)
from src.verification.prover import load_weights, prove_step

ETA = 0.05


class ListReader:
    def __init__(self, leaves):
        self._leaves = list(leaves)

    def leaf(self, index):
        return self._leaves[index]


@pytest.fixture
def setup():
    c = MLPComputation(DEFAULT_WIDTHS, n_s=4)
    data = synthetic_dataset(c.widths, 12, seed=0)
    return c, data, init_weights(c.widths, seed=0)


@pytest.mark.parametrize("widths", [(4, 3), (4, 5, 3), (16, 32, 32, 8), (3, 4, 5, 6, 2)])
def test_inventory(widths):
    c = MLPComputation(widths, n_s=3)
    L = len(widths) - 1
    assert c.M == 3 * L - 1
    names = [p.name for p in c.products]
    backward = []
    for l in range(L, 0, -1):
        backward += ([f"dX_{l}"] if l >= 2 else []) + [f"G_{l}"]
    assert names == [f"Y_{l}" for l in range(1, L + 1)] + backward
    assert [p.m for p in c.products] == list(range(1, c.M + 1))
    for p in c.products:
        l = int(p.name.split("_")[1])
        i, o = widths[l - 1], widths[l]
        assert p.weight == c.weight(l)
        expect = {ProductKind.FORWARD: ((3, i), (i, o)),
                  ProductKind.INPUT_GRAD: ((3, o), (o, i)),
                  ProductKind.WEIGHT_GRAD: ((o, 3), (3, i))}[p.kind]
        assert (p.a_shape, p.b_shape) == expect
    for w, m in c.linear_weights.items():
        assert c.product(m).p_shape == c.weight_shapes[w]
    assert set(c.linear_weights) == set(c.weight_names)
    assert c.glue_gradient_weights == ()
    assert c.n_leaves == 3 + 2 * L + 3 * L - 1


def test_record_encoding(setup):
    c, data, _ = setup
    rec = data[0]
    b = c.encode_record(rec)
    assert b == encode_tensor_leaf(TAG_MLP_RECORD, rec)
    assert b[:7] == struct.pack(">BBBI", 0x04, 0x01, 1, c.d_in + c.d_out)
    assert len(b) == 7 + 4 * (c.d_in + c.d_out)
    x, y = split_record(rec, c.d_in)
    assert torch.equal(make_record(x, y), rec)
    with pytest.raises(ValueError):
        c.encode_record(rec[:-1])
    with pytest.raises(ValueError):
        c.encode_record(rec.double())


def test_seeded_constructors_are_deterministic_and_rng_neutral():
    torch.manual_seed(123)
    state = torch.get_rng_state()
    w_a, w_b = init_weights(DEFAULT_WIDTHS, 7), init_weights(DEFAULT_WIDTHS, 7)
    d_a, d_b = synthetic_dataset(DEFAULT_WIDTHS, 5, 7), synthetic_dataset(DEFAULT_WIDTHS, 5, 7)
    MLPComputation().build_model()
    assert torch.equal(torch.get_rng_state(), state)
    assert all(torch.equal(w_a[k], w_b[k]) for k in w_a)
    assert all(torch.equal(a, b) for a, b in zip(d_a, d_b))
    assert not torch.equal(init_weights(DEFAULT_WIDTHS, 8)["layers.0.weight"],
                           w_a["layers.0.weight"])


def test_sequential_schedule():
    assert sequential_schedule(10, 4, 3) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 1]]


def _capture(c, data, w0):
    model = c.build_model()
    load_weights(c, model, w0)
    cap = MatmulCapture(param_storage_map(model))
    with cap:
        with cap.phase("forward"):
            loss = c.loss(model, data[: c.n_s])
        with cap.phase("backward"):
            loss.backward()
    return cap, model


def test_capture_has_no_dX1(setup):
    c, data, w0 = setup
    cap, _ = _capture(c, data, w0)
    assert cap.n_products == c.M == 3 * c.L - 1
    assert not cap.glue_outer
    # No backward mm consumes W_1: δX_1 is never computed.
    assert all(r.b_info.param_name != c.weight(1) for r in cap.records if r.phase == "backward")


def test_label_fills_each_slot_once_with_captured_operands(setup):
    c, data, w0 = setup
    cap, model = _capture(c, data, w0)
    products = c.label(cap, model)
    by_out = {r.out.data_ptr(): r for r in cap.records}
    assert len({p.data_ptr() for p in products}) == c.M
    # Verifier reconstruction from committed leaves equals the operands the op received.
    leaves = [*data[: c.n_s], *w0.values(), *products, *w0.values()]
    for spec, p in zip(c.products, products):
        rec = by_out[p.data_ptr()]
        a, b = c.operands(spec.m, ListReader(leaves))
        assert torch.equal(a, rec.a) and a.stride() == rec.a.stride(), spec.name
        assert torch.equal(b, rec.b) and b.stride() == rec.b.stride(), spec.name


def test_label_rejects_unmatched_record(setup):
    c, data, w0 = setup
    cap, model = _capture(c, data, w0)
    cap.records.append(cap.records[-1])  # a second G_1
    with pytest.raises(LabelingError, match="twice"):
        c.label(cap, model)
    cap.records.pop()
    cap.records.pop(3)
    with pytest.raises(LabelingError, match="never filled"):
        c.label(cap, model)


def test_reconstruction_matches_prover_bit_exact(setup):
    c, data, w0 = setup
    out = prove_step(c, c.build_model(), w0, data[: c.n_s], ETA)
    reader = ListReader(out.leaves())
    for spec in c.products:
        a, b = c.operands(spec.m, reader)
        assert (tuple(a.shape), tuple(b.shape)) == (spec.a_shape, spec.b_shape)
        assert torch.equal(a @ b, out.products[spec.m - 1]), spec.name
    assert c.glue_gradients(reader) == {}
