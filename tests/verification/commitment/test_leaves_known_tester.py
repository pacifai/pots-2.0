"""FS30 tester: what ``test_leaves_known.py`` leaves open. The CPU path is unchanged with no
``known``; ``known`` is read by layout position; a known leaf that fails a structural check goes
to the host path; an index outside the tensor slots is ignored; and ``prove_step``'s digests
reach the commit through ``commit``, ``InMemoryStore.commit_step`` (both copies) and
``with_w_next``."""

import dataclasses

import pytest
import torch

from tests.verification.commitment.test_leaves_known import (  # noqa: F401
    c, leaves, plain, same_outcome)
from verification.commitment.encoding import TAG_PRODUCT, NonFiniteError
from verification.commitment.leaves import (
    HashStats, commit_leaves, hash_step_leaves, leaf_hashes_of)
from verification.commitment.merkle import hash_tensor_leaf
from verification.computation.instances.mlp import init_weights, synthetic_dataset
from verification.prover.capture import MatmulCapture
from verification.prover.step import commit, prove_step
from verification.transcript.errors import LeafShapeError
from verification.transcript.store import InMemoryStore


def of_leaf_hashes_of(c, leaves, stats=None):
    try:
        return leaf_hashes_of(c, list(enumerate(leaves)), stats), None
    except Exception as e:  # noqa: BLE001
        return None, e


def counts(s):
    return {k: getattr(s, k) for k in ("batches", "leaves", "bytes", "known", "kernel")}


# ---- the CPU path is the old path -------------------------------------------------------------


@pytest.mark.parametrize("known", [None, {}])
def test_no_known_and_no_device_leaf_is_leaf_hashes_of(c, leaves, known):  # noqa: F811
    s_old, s_new = HashStats(), HashStats()
    want = of_leaf_hashes_of(c, leaves, s_old)
    got = hash_step_leaves(c, leaves, s_new, known=known), None
    assert got == want
    assert counts(s_new) == counts(s_old) and s_new.known == s_new.kernel == 0
    assert commit_leaves(c, leaves, known).root == commit_leaves(c, leaves).root


@pytest.mark.parametrize("known", [None, {}])
def test_no_known_gives_leaf_hashes_ofs_errors_at_the_same_leaf(c, leaves, known):  # noqa: F811
    bad = list(leaves)
    bad[c.product_leaf_index(c.leaf_of(2)[0])] = bad[c.product_leaf_index(c.leaf_of(2)[0])].clone()
    bad[c.product_leaf_index(c.leaf_of(2)[0])].view(-1)[0] = float("nan")
    bad[c.product_leaf_index(c.leaf_of(5)[0])] = bad[c.product_leaf_index(c.leaf_of(5)[0])][:-1]
    want = of_leaf_hashes_of(c, bad)
    try:
        got = (hash_step_leaves(c, bad, known=known), None)
    except Exception as e:  # noqa: BLE001
        got = (None, e)
    assert want[1] is not None
    same_outcome(got, want)
    assert isinstance(want[1], NonFiniteError)  # index of the NaN is below the shape error


@pytest.mark.parametrize("known", [None, {}])
def test_no_known_pins_the_nans_leaf_on_both_sides(c, leaves, known):  # noqa: F811
    """NaN at product 3, a cut at product 4 (the next leaf) still gives the NaN's error and a
    cut at product 2 the shape error, with ``leaf_hashes_of`` and with ``hash_step_leaves``."""
    from verification.transcript.errors import LeafShapeError

    i = c.product_leaf_index(c.leaf_of(3)[0])
    for j, kind in ((i + 1, NonFiniteError), (i - 1, LeafShapeError)):
        bad = list(leaves)
        bad[i] = bad[i].clone()
        bad[i].view(-1)[0] = float("nan")
        bad[j] = bad[j][:-1]
        want = of_leaf_hashes_of(c, bad)
        assert type(want[1]) is kind
        try:
            got = (hash_step_leaves(c, bad, known=known), None)
        except Exception as e:  # noqa: BLE001
            got = (None, e)
        same_outcome(got, want)


def test_a_leaf_count_mismatch_is_refused_with_known_too(c, leaves):  # noqa: F811
    with pytest.raises(ValueError, match="transcript has"):
        hash_step_leaves(c, leaves[:-1], known={c.product_leaf_index(c.leaf_of(1)[0]): b"x" * 32})


# ---- known digests: where they apply ----------------------------------------------------------


def test_known_digests_in_layout_order_are_the_plain_digests(c, leaves):  # noqa: F811
    want = hash_step_leaves(c, leaves)
    known = {i: want[i] for i in range(c.n_s, c.n_leaves)}
    s = HashStats()
    got = hash_step_leaves(c, leaves, s, known=known)
    assert got == want
    assert s.known == s.leaves - c.n_s == c.n_leaves - c.n_s and s.kernel == 0


def test_known_digests_are_taken_by_index_not_by_order(c, leaves):  # noqa: F811
    want = hash_step_leaves(c, leaves)
    a, b = c.product_leaf_index(c.leaf_of(1)[0]), c.product_leaf_index(c.leaf_of(2)[0])
    got = hash_step_leaves(c, leaves, known={a: want[b], b: want[a]})
    assert got[a] == want[b] and got[b] == want[a]
    assert [got[i] for i in range(len(got)) if i not in (a, b)] == \
        [want[i] for i in range(len(want)) if i not in (a, b)]
    assert commit_leaves(c, leaves, {a: want[b], b: want[a]}).root != \
        commit_leaves(c, leaves).root


@pytest.mark.parametrize("which", ["past the end", "far past the end", "negative", "record slot",
                                   "all records"])
def test_a_known_digest_outside_the_tensor_slots_is_ignored(c, leaves, which):  # noqa: F811
    """Ignored, not refused. Sound: the digest is never put in the tree, the leaf is hashed from
    its bytes, and ``known`` keys come from the prover's own ``arrival_digest`` (a product
    index), so nothing can reach a slot that way."""
    junk = b"\xee" * 32
    known = {"past the end": {c.n_leaves: junk}, "far past the end": {10 ** 9: junk},
             "negative": {-1: junk, -c.n_leaves: junk}, "record slot": {0: junk},
             "all records": {i: junk for i in range(c.n_s)}}[which]
    s = HashStats()
    want_stats = HashStats()
    want = of_leaf_hashes_of(c, leaves, want_stats)
    assert hash_step_leaves(c, leaves, s, known=known) == want[0]
    assert s.known == 0
    assert s.leaves == want_stats.leaves  # nothing was counted as taken


def test_a_known_digest_beside_an_ignored_one_still_counts_once(c, leaves):  # noqa: F811
    want = hash_step_leaves(c, leaves)
    i = c.product_leaf_index(c.leaf_of(1)[0])
    s = HashStats()
    got = hash_step_leaves(c, leaves, s, known={i: want[i], 0: b"\x00" * 32, c.n_leaves: b""})
    assert got == want and s.known == 1


@pytest.mark.parametrize("break_it", ["shape", "dtype", "stride"])
def test_a_known_leaf_failing_a_structural_check_raises_the_plain_error(
        c, leaves, break_it):  # noqa: F811
    i = c.product_leaf_index(c.leaf_of(3)[0])
    bad = list(leaves)
    t = bad[i]
    bad[i] = {"shape": t[:-1], "dtype": t.to(torch.float64),
              "stride": t.t().contiguous().t() if t.shape[0] != t.shape[1] else t.t()}[break_it]
    want = plain(c, bad)
    assert want[1] is not None
    got = None
    try:
        hash_step_leaves(c, bad, known={i: b"\x11" * 32})
    except Exception as e:  # noqa: BLE001
        got = e
    assert got is not None
    same_outcome((None, got), want)


def test_a_known_digest_does_not_hide_a_non_finite_leaf_at_another_index(c, leaves):  # noqa: F811
    want_digests = hash_step_leaves(c, leaves)
    bad = list(leaves)
    j = c.product_leaf_index(c.leaf_of(4)[0])
    bad[j] = bad[j].clone()
    bad[j].view(-1)[3] = float("inf")
    known = {i: want_digests[i] for i in range(c.n_s, c.n_leaves) if i != j}
    want = plain(c, bad)
    got = None
    try:
        hash_step_leaves(c, bad, known=known)
    except Exception as e:  # noqa: BLE001
        got = e
    same_outcome((None, got), want)


# ---- prove_step's digests through the commit --------------------------------------------------


@pytest.fixture
def stepped(c, monkeypatch):  # noqa: F811
    """A CPU step whose capture hands out the right digest for every product (what an H100
    capture does for every product but ``dK``); the summary gains an ``arrival_hash`` entry, as
    a CUDA capture's does."""
    monkeypatch.setattr(MatmulCapture, "arrival_digest",
                        lambda self, p: hash_tensor_leaf(TAG_PRODUCT, p))
    real = MatmulCapture.summary

    def summary(self):
        return real(self) | {"arrival_hash": {"records": 1, "leaves": 1, "bytes": 0,
                                              "nonfinite": 0}}

    monkeypatch.setattr(MatmulCapture, "summary", summary)
    data = synthetic_dataset(c.widths, 12, seed=0)
    return prove_step(c, c.build_model(), init_weights(c.widths, seed=0), data[4:8])


def test_arrival_digests_are_keyed_by_product_leaf_index(c, stepped):  # noqa: F811
    assert set(stepped.arrival_digests) == {
        c.product_leaf_index(c.leaf_of(m)[0]) for m in range(1, c.M + 1)}
    leaves = stepped.leaves()
    plain_digests = hash_step_leaves(c, leaves)
    assert all(plain_digests[i] == d for i, d in stepped.arrival_digests.items())
    assert stepped.capture["arrival_hash"]["used"] == c.M


def test_with_w_next_keeps_the_entries(c, stepped):  # noqa: F811
    spliced = stepped.with_w_next({n: w + 1 for n, w in stepped.w_next.items()})
    assert dict(spliced.arrival_digests) == dict(stepped.arrival_digests) != {}
    want = commit_leaves(c, spliced.leaves()).root
    assert commit(c, spliced).root == want
    assert commit_leaves(c, spliced.leaves(), spliced.arrival_digests).root == want


def test_a_perturbed_product_loses_only_its_entry(c, monkeypatch):  # noqa: F811
    monkeypatch.setattr(MatmulCapture, "arrival_digest",
                        lambda self, p: hash_tensor_leaf(TAG_PRODUCT, p))
    real = MatmulCapture.summary
    monkeypatch.setattr(MatmulCapture, "summary", lambda self: real(self) | {
        "arrival_hash": {"records": 1, "leaves": c.n_product_leaves, "bytes": 0, "nonfinite": 0}})
    data = synthetic_dataset(c.widths, 12, seed=0)
    out = prove_step(c, c.build_model(), init_weights(c.widths, seed=0), data[4:8],
                     perturb={2: lambda t: t + 1, 5: lambda t: t * 2})
    gone = {c.product_leaf_index(c.leaf_of(2)[0]), c.product_leaf_index(c.leaf_of(5)[0])}
    assert set(out.arrival_digests) == {
        c.product_leaf_index(c.leaf_of(m)[0]) for m in range(1, c.M + 1)} - gone
    assert out.capture["arrival_hash"]["used"] == c.n_product_leaves - 2
    # the capture hashed them all
    assert out.capture["arrival_hash"]["leaves"] == c.n_product_leaves
    # the digests left are right; the perturbed leaves hash from their bytes
    assert commit(c, out).root == commit_leaves(c, out.leaves()).root


@pytest.mark.parametrize("how", ["commit", "store", "store_copy"])
def test_the_commit_with_arrival_digests_gives_the_plain_root(c, stepped, how):  # noqa: F811
    want = commit_leaves(c, stepped.leaves()).root
    if how == "commit":
        root = commit(c, stepped).root
    else:
        root = InMemoryStore.commit_step(c, stepped, copy=how == "store_copy")[1].root
    assert root == want


@pytest.mark.parametrize("how", ["commit", "store", "store_copy"])
def test_a_wrong_arrival_digest_moves_the_committed_root(c, stepped, how):  # noqa: F811
    """The digests are used, not recomputed: a wrong one shows in the root, which check 2's
    CPU recompute (K4) then rejects."""
    want = commit_leaves(c, stepped.leaves()).root
    i = c.product_leaf_index(c.leaf_of(1)[0])
    bad = dataclasses.replace(stepped, arrival_digests={**stepped.arrival_digests,
                                                        i: b"\x5a" * 32})
    if how == "commit":
        root = commit(c, bad).root
    else:
        root = InMemoryStore.commit_step(c, bad, copy=how == "store_copy")[1].root
    assert root != want


def test_the_store_copy_holds_leaves_the_known_digests_still_describe(c, stepped):  # noqa: F811
    leaves_, tree = InMemoryStore.commit_step(c, stepped, copy=True)
    again = hash_step_leaves(c, leaves_)
    assert again == hash_step_leaves(c, stepped.leaves())
    assert tree.root == commit_leaves(c, leaves_).root
    with pytest.raises(LeafShapeError):
        bad = list(leaves_)
        bad[c.product_leaf_index(c.leaf_of(1)[0])] = bad[c.product_leaf_index(c.leaf_of(1)[0])][:-1]
        hash_step_leaves(c, bad, known=stepped.arrival_digests)


# ---- the callers that must keep the old path --------------------------------------------------

REPO = __import__("pathlib").Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("path", [
    "verification/verifier/context.py", "verification/verifier/driver.py",
    "verification/runs/run_verified.py", "verification/runs/card_identity.py",
    "verification/runs/k_tunability.py", "verification/runs/probe.py",
    "verification/runs/bring_up.py", "verification/verifier/checks.py"])
def test_only_the_provers_commit_passes_known_digests(path):
    """Check 2 (``context.py``, ``driver.py``, ``transcript_root``) and every other caller reach
    ``leaf_hashes_of`` or ``commit_leaves`` with no ``known``, so every leaf is hashed by the C
    library on the CPU (K4)."""
    text = (REPO / path).read_text()
    assert "known=" not in text and "arrival_digests" not in text
    assert "hash_step_leaves" not in text


def test_the_providers_of_known_digests_are_the_prover_commit_and_the_store():
    users = sorted(str(p.relative_to(REPO)) for p in (REPO / "verification").rglob("*.py")
                   if "known=" in p.read_text() and "hash_step_leaves" in p.read_text())
    assert users == ["verification/commitment/leaves.py", "verification/transcript/store.py"]
    assert "arrival_digests" in (REPO / "verification/prover/step.py").read_text()
