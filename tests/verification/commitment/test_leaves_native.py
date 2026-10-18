"""The native batched leaf path (F15b) against the Python reference: the same digests and the
same first error, whatever the batch sizes, thread counts and leaf sizes."""

import random
import time

import pytest
import torch

from verification.commitment import leaves as leaves_mod
from verification.commitment import merkle, native
from verification.commitment.leaves import (
    commit_leaves,
    leaf_hash,
    leaf_hashes_of,
    leaf_hashes_until_error,
)
from verification.computation.instances.mlp import MLPComputation, init_weights, synthetic_dataset
from verification.prover.step import prove_step
from verification.runs import metrics

pytestmark = pytest.mark.skipif(not native.enabled(), reason="the native library is unavailable")

ETA = 1e-2


@pytest.fixture
def c():
    return MLPComputation((16, 32, 32, 8), n_s=4, eta=ETA)


@pytest.fixture
def data(c):
    return synthetic_dataset(c.widths, 12, seed=0)


@pytest.fixture
def leaves(c, data):
    w0 = init_weights(c.widths, seed=0)
    return list(prove_step(c, c.build_model(), w0, data[4:8]).leaves())


@pytest.fixture
def threads():
    before = torch.get_num_threads()
    yield
    torch.set_num_threads(before)


@pytest.fixture(params=["default", "small_batches", "big_leaves", "both"])
def mode(request, monkeypatch):
    """The default limits, tiny batches (3 leaves or 1 KiB), every tensor leaf on the big
    path, and both."""
    if request.param in ("small_batches", "both"):
        monkeypatch.setattr(leaves_mod, "_BATCH_ENTRIES", 3)
        monkeypatch.setattr(leaves_mod, "_BATCH_BYTES", 1024)
    if request.param in ("big_leaves", "both"):
        monkeypatch.setattr(merkle, "_MT_THRESHOLD", 64)
    return request.param


def _both(c, items):
    """`leaf_hashes_until_error` native, then under `native.disabled()`; `items` is a list."""
    got = leaf_hashes_until_error(c, iter(items))
    with native.disabled():
        want = leaf_hashes_until_error(c, iter(items))
    return got, want


def _same(got, want):
    (gd, ge), (wd, we) = got, want
    assert gd == wd
    assert type(ge) is type(we)
    assert str(ge) == str(we)
    assert all(isinstance(d, bytes) and len(d) == 32 for d in gd)


def test_honest_step(c, leaves, mode):
    loop = [leaf_hash(c, i, x) for i, x in enumerate(leaves)]
    assert leaf_hashes_of(c, enumerate(leaves)) == loop
    with native.disabled():
        assert leaf_hashes_of(c, enumerate(leaves)) == loop
    assert commit_leaves(c, leaves).root == commit_leaves(c, leaves).root
    _same(*_both(c, list(enumerate(leaves))))


def _reader_error(i):
    def gen():
        raise IndexError(f"leaf {i} is not in the store")
        yield  # pragma: no cover
    return gen


def _inject(c, leaves, rng):
    """`(items, faults)`: the honest `(index, obj)` list with 0 to 3 faults, in a random kind."""
    items = list(enumerate(leaves))
    n_faults = rng.randint(0, 3)
    for pos in rng.sample(range(c.n_s, c.n_leaves), n_faults):
        obj = leaves[pos]
        kind = rng.choice(["nan", "inf", "-inf", "shape", "dtype", "strided", "reader"])
        if kind in ("nan", "inf", "-inf"):
            obj = obj.clone()
            obj.view(-1)[rng.randrange(obj.numel())] = float(kind) if kind != "inf" else float("inf")
        elif kind == "shape":
            obj = obj.reshape(-1)
        elif kind == "dtype":
            obj = obj.to(torch.float64)
        elif kind == "strided":
            obj = _noncontiguous(obj)
        else:
            obj = _Raise(pos)
        items[pos] = (pos, obj)
    return items


def _noncontiguous(t):
    """A tensor of `t`'s shape and dtype that is not contiguous (when `t` has two dimensions
    or more and is not degenerate), else `t`."""
    if t.dim() < 2 or min(t.shape) < 2:
        return t
    return t.t().contiguous().t()


class _Raise:
    """Stands for a reader that fails at this leaf: items yields it, then the loop raises."""

    def __init__(self, pos):
        self.pos = pos


def _items_with_reader_errors(items):
    for i, obj in items:
        if isinstance(obj, _Raise):
            raise IndexError(f"leaf {i} is not in the store")
        yield i, obj


def _run(c, items):
    got = leaf_hashes_until_error(c, _items_with_reader_errors(items))
    with native.disabled():
        want = leaf_hashes_until_error(c, _items_with_reader_errors(items))
    return got, want


@pytest.mark.parametrize("seed", range(30))
def test_error_injection(c, leaves, mode, seed):
    rng = random.Random(seed)
    items = _inject(c, leaves, rng)
    got, want = _run(c, items)
    _same(got, want)
    # The loop's own answer: the leaves before the first fault, then its error.
    ref, err = [], None
    try:
        for i, obj in _items_with_reader_errors(items):
            ref.append(leaf_hash(c, i, obj))
    except Exception as e:
        err = e
    assert got[0] == ref
    assert type(got[1]) is type(err) and str(got[1]) == str(err)


def test_nonfinite_message_matches_encoding(c, leaves):
    from verification.commitment.encoding import NonFiniteError, tensor_leaf_payload
    t = leaves[c.product_index(1)].clone()
    t.view(-1)[0] = float("nan")
    with pytest.raises(NonFiniteError) as ref:
        tensor_leaf_payload(t)
    err = leaf_hashes_until_error(c, [(c.product_index(1), t)])[1]
    assert str(err) == str(ref.value)


def test_error_in_every_position(c, leaves, mode):
    """A NaN at each tensor leaf in turn: the digests before it and its error."""
    for pos in range(c.n_s, c.n_leaves):
        bad = leaves[pos].clone()
        bad.view(-1)[-1] = float("-inf")
        items = list(enumerate(leaves))
        items[pos] = (pos, bad)
        got, want = _both(c, items)
        _same(got, want)
        assert len(got[0]) == pos and got[1] is not None


def test_thread_counts(c, leaves, mode, threads):
    honest = list(enumerate(leaves))
    nan_at = c.n_s + 3
    nan = list(honest)
    nan[nan_at] = (nan_at, torch.full_like(leaves[nan_at], float("nan")))
    for items in (honest, nan):
        results = []
        for n in (1, 8):
            torch.set_num_threads(n)
            results.append(leaf_hashes_until_error(c, iter(items)))
        _same(*results)
    assert len(results[0][0]) == nan_at and results[0][1] is not None


def test_counting_runs_the_python_path(c, leaves):
    assert native.enabled()
    with metrics.counting() as (_, tally):
        assert not native.enabled()
        commit_leaves(c, leaves)
        assert tally.leaves == c.n_leaves
    assert native.enabled()


class _Stub:
    """A computation of `n` weight leaves of one shape and no records or products."""

    n_s, M = 0, 0

    def __init__(self, n, shape):
        self.n_w = self.n_leaves = n
        self.weight_names = [f"w{i}" for i in range(n)]
        self.weight_shapes = {name: shape for name in self.weight_names}
        self.weight_dtype = torch.float32
        self.product_dtype = torch.float32


def test_speed(threads, capsys):
    """Printed, not asserted: 400,000 fp32 leaves of 8 KiB, views of one tensor."""
    n, shape = 400_000, (2048,)
    big = torch.randn(n * shape[0])
    views = list(big.view(n, *shape).unbind(0))
    stub = _Stub(n, shape)
    torch.set_num_threads(8)
    times = {}
    for label in ("native", "python"):
        t0 = time.perf_counter()
        if label == "native":
            digests, err = leaf_hashes_until_error(stub, enumerate(views))
        else:
            with native.disabled():
                digests, err = leaf_hashes_until_error(stub, enumerate(views))
        times[label] = time.perf_counter() - t0
        assert err is None and len(digests) == n
        if label == "native":
            first = digests
        else:
            assert digests == first
    with capsys.disabled():
        print(f"\n400k x 8 KiB leaves, 8 threads: native {times['native']:.2f} s, "
              f"python {times['python']:.2f} s")
