import struct

import pytest
import torch

from setup.data import schedule
from verification.commitment.encoding import TAG_MLP_RECORD, encode_tensor_leaf
from verification.computation.instances.mlp import (
    DEFAULT_WIDTHS,
    MLPComputation,
    init_weights,
    make_record,
    split_record,
    synthetic_dataset,
)
from verification.computation.interface import LabelingError, ProductKind, load_weights
from verification.computation.matmul_ops import param_storage_map
from verification.prover.capture import MatmulCapture
from verification.prover.step import prove_step

ETA = 0.05


class ListReader:
    def __init__(self, leaves):
        self._leaves = list(leaves)

    def leaf(self, index):
        return self._leaves[index]


@pytest.fixture
def setup():
    c = MLPComputation(DEFAULT_WIDTHS, n_s=4, eta=ETA)
    data = synthetic_dataset(c.widths, 12, seed=0)
    return c, data, init_weights(c.widths, seed=0)


@pytest.mark.parametrize("widths", [(4, 3), (4, 5, 3), (16, 32, 32, 8), (3, 4, 5, 6, 2)])
def test_inventory(widths):
    c = MLPComputation(widths, n_s=3, eta=ETA)
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
        assert p.layer == l
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
    assert c.eta == ETA


@pytest.mark.parametrize("widths, n_s", [((4, 3), 1), ((4, 1, 3), 2), ((1, 3), 2), ((4,), 2)])
def test_rejects_unrunnable_declarations(widths, n_s):
    with pytest.raises(ValueError):
        MLPComputation(widths, n_s=n_s, eta=ETA)


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
    MLPComputation(eta=ETA).build_model()
    assert torch.equal(torch.get_rng_state(), state)
    assert all(torch.equal(w_a[k], w_b[k]) for k in w_a)
    assert all(torch.equal(a, b) for a, b in zip(d_a, d_b))
    assert not torch.equal(init_weights(DEFAULT_WIDTHS, 8)["layers.0.weight"],
                           w_a["layers.0.weight"])


def test_schedule_over_mlp_dataset(setup):
    c, data, w0 = setup
    batches = [[data[i] for i in schedule(t, c.n_s, len(data))] for t in (1, 2, 3)]
    assert batches[1][0] is data[4]
    for b in batches:
        c.assemble(b)
    with pytest.raises(ValueError):
        schedule(4, c.n_s, len(data))  # no wraparound past |D|


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
    replay = c.replay(ListReader([*data[: c.n_s], *w0.values(), *products, *w0.values()]))
    for spec, p in zip(c.products, products):
        rec = by_out[p.data_ptr()]
        a, b = replay.operands(spec.m)
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
    out = prove_step(c, c.build_model(), w0, data[: c.n_s])
    replay = c.replay(ListReader(out.leaves()))
    with pytest.raises(RuntimeError, match="operands"):
        replay.glue_gradients()
    for spec in c.products:
        a, b = replay.operands(spec.m)
        assert (tuple(a.shape), tuple(b.shape)) == (spec.a_shape, spec.b_shape)
        assert torch.equal(a @ b, out.products[spec.m - 1]), spec.name
    assert replay.glue_gradients() == {}
    assert not replay._x and not replay._dy  # glue dropped after its last consumer
    # The replay's own model holds the committed W_t, not a prover object.
    for n, p in replay.model.named_parameters():
        assert torch.equal(p.detach(), out.w_t[n]) and p.data_ptr() != out.w_t[n].data_ptr()


def test_replay_out_of_order_recomputes(setup):
    c, data, w0 = setup
    out = prove_step(c, c.build_model(), w0, data[: c.n_s])
    reader = ListReader(out.leaves())
    in_order = c.replay(reader)
    expected = [in_order.operands(m) for m in range(1, c.M + 1)]
    shuffled = c.replay(reader)
    for m in [c.M, 1, c.m_of("G_3"), c.m_of("dX_3"), c.m_of("G_3"), 2]:
        a, b = shuffled.operands(m)
        assert torch.equal(a, expected[m - 1][0]) and torch.equal(b, expected[m - 1][1])
