"""O18 (one leaf per attention product): faults in a grouped leaf are rejected where they must be.

Written by the tester from ``ATTENTION_LEAF_PLAN_DRAFT.md`` §1.4, §2.6 and §4.1 alone.

3. ``test_corrupted_member_byte_*``: one byte of a committed group leaf changed after the
   commit (original root and paths kept) is rejected at check 2, on the in-memory, disk and
   shared stores.
4. ``test_wrong_member_committed_consistently_*``: the prover's ``perturb`` on one member,
   re-rooted, is rejected at check 5 naming that member.
"""

import pytest
import torch

from setup.data import schedule
from tests.verification.decoder_helpers import make_records, randomize_biases, tiny_config
from verification.commitment.leaves import dataset_tree
from verification.computation.instances import DecoderComputation
from verification.computation.interface import snapshot_weights
from verification.prover.step import prove_step
from verification.transcript.shared import SharedStore, share_step
from verification.transcript.store import DiskStore, InMemoryStore
from verification.verifier.bands import Bands
from verification.verifier.context import Rejection
from verification.verifier.driver import Verifier

SEED, K, ETA = 1, 7, 1e-3


class Setup:
    """A tiny decoder, its data and weights, and a way to prove step 1 (optionally forged)."""

    def __init__(self):
        torch.manual_seed(0)
        self.c = DecoderComputation(tiny_config(), n_s=2, n=12, eta=ETA)
        model = self.c.build_model()
        randomize_biases(dict(model.named_parameters()))
        self.w0 = snapshot_weights(self.c, model)
        self.D = make_records((12, 7, 10, 5, 11), seed=1)
        self.dtree = dataset_tree(self.c, self.D)
        self.idx = schedule(1, self.c.n_s, len(self.D), SEED)
        self.paths = [self.dtree.path(i) for i in self.idx]

    def prove(self, **kw):
        c = self.c
        return prove_step(c, c.build_model(), self.w0, [self.D[i] for i in self.idx], **kw)

    def verifier(self):
        v = Verifier(self.c, h_D=self.dtree.root, n_records=len(self.D), k=K, n_steps=1,
                     bands=Bands.provisional(), w0=self.w0, allow_provisional=True, seed=SEED)
        assert v.start_run(self.D) is None
        return v

    def leaf_named(self, name):
        c = self.c
        return next(pl for pl in (c.product_leaf(g) for g in range(c.n_product_leaves))
                    if pl.name == name)


@pytest.fixture(scope="module")
def S():
    return Setup()


@pytest.fixture(scope="module")
def honest(S):
    out = S.prove()
    leaves, tree = InMemoryStore.commit_step(S.c, out)
    return out, leaves, tree


def _expect(rej, check_id):
    assert isinstance(rej, Rejection), rej
    assert (rej.step, rej.check_id) == (1, check_id), rej
    return rej


# ---- 3. a corrupted member byte, three stores --------------------------------------------


def _memory(S, leaves, tree, tmp_path):
    yield InMemoryStore.hold(S.c, list(leaves), tree, S.paths)


def _disk(S, leaves, tree, tmp_path):
    DiskStore.write(tmp_path / "step", S.c, leaves, tree, S.paths)
    yield DiskStore(tmp_path / "step")


def _shared(S, leaves, tree, tmp_path):
    pkg, hold = share_step(S.c, 1, list(leaves), tree, S.paths)
    store = SharedStore.open(S.c, pkg)
    try:
        yield store
    finally:
        store.close()
        hold.release()


STORES = {"memory": _memory, "disk": _disk, "shared": _shared}


def _tampered(S, leaves, name="L1.O"):
    """``leaves`` with the group leaf ``name`` cloned and one byte of its LAST member changed."""
    pl = S.leaf_named(name)
    assert pl.grouped and pl.n == S.c.n_s * S.c.n_h
    i = S.c.product_leaf_index(pl.g)
    leaf = leaves[i].clone()
    flat = leaf.view(torch.uint8).view(pl.n, -1)
    flat[-1, 0] ^= 0x01  # a byte of the last member's first entry
    assert not torch.equal(leaf, leaves[i])
    assert torch.equal(leaf[:-1], leaves[i][:-1])  # only the last member differs
    out = list(leaves)
    out[i] = leaf
    return out


@pytest.mark.parametrize("store_kind", STORES)
def test_untampered_step_is_accepted_on_each_store(S, honest, tmp_path, store_kind):
    """The control of test 3: the same step, unchanged, is accepted."""
    _, leaves, tree = honest
    for store in STORES[store_kind](S, leaves, tree, tmp_path):
        assert S.verifier().verify_step(1, store) is None


@pytest.mark.parametrize("name", ["L1.O", "L2.dK"])
@pytest.mark.parametrize("store_kind", STORES)
def test_corrupted_member_byte_is_rejected_at_check_2(S, honest, tmp_path, store_kind, name):
    _, leaves, tree = honest
    bad = _tampered(S, leaves, name)  # the ORIGINAL tree: its root and paths
    for store in STORES[store_kind](S, bad, tree, tmp_path):
        assert store.root == tree.root
        _expect(S.verifier().verify_step(1, store), "2")


# ---- 4. a wrong member committed consistently --------------------------------------------


def _bump(p):
    p = p.clone()
    p[(0,) * p.dim()] += 1.0
    return p


def _rejected_at_member(S, honest, pl, i):
    """Perturb member ``i`` of leaf ``pl`` in the prover, commit, verify; return the leaf pair."""
    c = S.c
    m = pl.ms[i]
    out = S.prove(perturb={m: _bump})
    store = InMemoryStore.from_step(c, out, dataset_paths=S.paths)
    rej = _expect(S.verifier().verify_step(1, store), "5")
    assert f"P_{m}" in rej.detail and c.product(m).name in rej.detail, rej.detail
    # the fault touched exactly one member of the grouped leaf
    k = c.product_leaf_index(pl.g)
    got, want = out.leaves()[k], honest[0].leaves()[k]
    assert tuple(got.shape) == tuple(want.shape) == pl.shape
    assert not torch.equal(got[i], want[i])
    for j in range(pl.n):
        if j != i:
            assert torch.equal(got[j], want[j]), f"member {j} changed"
    # and no other product leaf changed
    for g in range(c.n_product_leaves):
        if g != pl.g:
            kk = c.product_leaf_index(g)
            assert torch.equal(out.leaves()[kk], honest[0].leaves()[kk]), g
    return rej


def test_wrong_member_committed_consistently_dV_in_the_middle(S, honest):
    """(a) ``L1.dV[s,h]`` at ``s·n_h + h`` near the middle: the backward roles interleave
    ``dA, dV`` with stride 2."""
    pl = S.leaf_named("L1.dV")
    assert pl.stride == 2 and pl.n == S.c.n_s * S.c.n_h
    i = pl.n // 2
    assert S.c.product(pl.ms[i]).name.startswith("L1.dV")
    _rejected_at_member(S, honest, pl, i)


def test_wrong_member_committed_consistently_dK(S, honest):
    """(b) a ``δK̃`` member."""
    pl = S.leaf_named("L2.dK")
    assert pl.grouped and pl.stride == 2
    i = pl.n - 2
    assert S.c.product(pl.ms[i]).name.startswith("L2.dK")
    _rejected_at_member(S, honest, pl, i)
