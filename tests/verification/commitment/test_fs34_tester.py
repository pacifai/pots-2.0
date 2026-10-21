"""FS34 tester: attacks on the join's routing, the per-shape check, the runs and the pair gate.

Built on the seams of ``test_leaves_known`` (the kernel on CPU tensors through the host build of
the core) and the helpers of ``test_fs34_routing``."""

import ctypes
import dataclasses
import inspect
import random
import subprocess
from pathlib import Path

import pytest
import torch

from tests.verification.commitment.test_gpu_blake3 import HostKernels, hostlib  # noqa: F401
from tests.verification.commitment.test_leaves_known import (  # noqa: F401
    c, first_product, leaves, on_kernel,
)
from verification.commitment import gpu_blake3 as gb
from verification.commitment import kernel_check as kc
from verification.commitment import leaves as L
from verification.commitment import native
from verification.commitment.encoding import NonFiniteError
from verification.commitment.leaves import (
    HashStats, hash_step_leaves, leaf_hashes_of, leaf_hashes_until_error,
)
from verification.transcript.errors import LeafShapeError, StoreMutationError

ROOT = Path(__file__).resolve().parents[3]
THREE = ("cpu", "cpu", "cpu")


@pytest.fixture(autouse=True)
def fresh_run():
    kc.new_run()
    yield
    kc.new_run()


def items(leaves):
    return list(enumerate(leaves))


def nan_in(t):
    t = t.clone()
    t.view(-1)[0] = float("nan")
    return t


def outcome(fn):
    """``(digests, class, message)`` of ``leaf_hashes_until_error`` run through ``fn``."""
    d, e = fn()
    return d, None if e is None else type(e), None if e is None else str(e)


# ---- routing equals the plain path -----------------------------------------------------------


@pytest.fixture
def route(monkeypatch, hostlib):  # noqa: F811
    """Kernel routing for the tensors in ``route.ids`` only (the rest are host leaves)."""
    g = gb.GpuBlake3(HostKernels(hostlib))
    ids: set[int] = set()
    monkeypatch.setattr(L, "_hasher", lambda: g)
    monkeypatch.setattr(L, "_kernel_leaf", lambda t: id(t) in ids and t.is_contiguous()
                        and t.numel() > 0 and t.dtype in native.FINITE_KIND)

    class R:
        pass
    r = R()
    r.ids = ids
    return r


ERR_KINDS = ("nan", "shape", "iter")


def _plant_errors(rng, c, base, n_err):
    """A copy of ``base`` with ``n_err`` faults at random tensor leaves (and the position of an
    iterable error, or ``None``)."""
    bad = list(base)
    lo = c.n_s
    for _ in range(n_err):
        i = rng.randrange(lo, len(bad))
        kind = rng.choice(("nan", "inf", "shape"))
        if kind == "shape":
            bad[i] = bad[i][:-1]
        else:
            t = bad[i].clone()
            t.view(-1)[rng.randrange(t.numel())] = float("nan" if kind == "nan" else "inf")
            bad[i] = t
    return bad


@pytest.mark.parametrize("seed", range(40))
def test_routing_equals_the_plain_path_for_random_mixes(c, leaves, route, monkeypatch, seed):  # noqa: F811
    rng = random.Random(seed)
    monkeypatch.setattr(L, "_BATCH_ENTRIES", rng.choice([1, 2, 3, 5, 100]))
    monkeypatch.setattr(L, "_BATCH_BYTES", rng.choice([1000, 3000, 7000, 1 << 20]))
    bad = _plant_errors(rng, c, leaves, rng.choice([0, 0, 1, 2, 3]))
    order = list(range(len(bad)))
    if rng.random() < 0.5:
        rng.shuffle(order)
    cut_at = rng.choice([None, rng.randrange(len(order) + 1)])
    route.ids.clear()
    route.ids.update(id(bad[i]) for i in order if rng.random() < 0.6)
    lazy = rng.random() < 0.5

    def make():
        its = [(i, bad[i]) for i in order]
        if cut_at is None:
            return iter(its) if lazy else its

        def gen():
            yield from its[:cut_at]
            raise OSError("read failed")
        return gen()

    def run(kernel):
        kc.new_run()
        return outcome(lambda: leaf_hashes_until_error(c, make(), kernel=kernel))

    assert run(True) == run(False)
    kc.new_run()
    plain = run(False)
    try:
        got = ("ok", leaf_hashes_of(c, make(), kernel=True))
    except Exception as e:  # noqa: BLE001
        got = ("err", type(e), str(e))
    if plain[1] is None:
        assert got == ("ok", plain[0])
    else:
        assert got == ("err", plain[1], plain[2])


@pytest.mark.parametrize("where", ["before", "inside", "after"])
@pytest.mark.parametrize("kind", ["nan_kernel", "nan_host", "shape", "iter", "mutation"])
@pytest.mark.parametrize("lazy", [False, True])
def test_every_error_kind_placed_around_tables(c, leaves, route, monkeypatch, where, kind,  # noqa: F811
                                               lazy):
    monkeypatch.setattr(L, "_BATCH_ENTRIES", 3)
    n = len(leaves)
    pos = {"before": c.n_s + 1, "inside": n // 2, "after": n - 2}[where]
    bad = list(leaves)
    if kind == "nan_kernel" or kind == "nan_host":
        bad[pos] = nan_in(bad[pos])
    elif kind == "shape":
        bad[pos] = bad[pos][:-1]
    route.ids.update(id(bad[i]) for i in range(n) if i % 2 == 0)
    route.ids.discard(id(bad[pos]))
    if kind != "nan_host":
        route.ids.add(id(bad[pos]))
    its = list(enumerate(bad))

    def stream():
        if kind == "iter":
            def gen():
                yield from its[:pos]
                raise OSError("read failed")
            return gen()
        return iter(its) if lazy else its

    plain = outcome(lambda: leaf_hashes_until_error(c, stream(), kernel=False)) \
        if kind != "mutation" else None
    if kind == "mutation":
        real = L._enqueue_table
        hit = []

        def mutate(idx, tensors, tags):
            tb = real(idx, tensors, tags)
            if pos in idx and not hit:
                hit.append(1)
                tensors[idx.index(pos)].add_(0)
            return tb
        monkeypatch.setattr(L, "_enqueue_table", mutate)
        route.ids.add(id(bad[pos]))
        want = leaf_hashes_of(c, its, kernel=False)
        kc.new_run()
        d, e = leaf_hashes_until_error(c, stream())
        assert isinstance(e, StoreMutationError) and f"leaf {pos} was" in str(e)
        assert d == want[:pos]
        return
    st = HashStats()
    got = outcome(lambda: leaf_hashes_until_error(c, stream(), st, kernel=True))
    assert got == plain
    assert st.kernel > 0 and (kind == "nan_host" or id(bad[pos]) in route.ids)
