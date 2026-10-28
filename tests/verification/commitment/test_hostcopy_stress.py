"""FS32b tests: the host-copy seam under stress. ``fake`` mode treats chosen CPU tensors as device
leaves (the staging code runs, the bytes never leave the host), with tiny batches."""

import contextlib

import pytest
import torch

from verification.commitment import hostcopy, native
from verification.commitment import leaves as leaves_mod
from verification.commitment import merkle
from verification.commitment.encoding import (
    tensor_leaf_payload,
    tensor_leaf_payload_unchecked,
)
from verification.commitment.hostcopy import Stage, StagingPool
from verification.commitment.leaves import (
    commit_leaves,
    hash_step_leaves,
    leaf_hash,
    leaf_hashes_of,
    leaf_hashes_until_error,
    leaf_parts,
    leaf_totals,
)
from verification.transcript import store as store_mod
from verification.transcript.errors import StoreMutationError
from verification.transcript.store import DiskStore, InMemoryStore, perturb_leaf

from tests.verification.commitment.test_hostcopy import _Stub, _same_error

needs_native = pytest.mark.skipif(not native.enabled(), reason="the native library is unavailable")
_PATHS = [pytest.param("native", marks=needs_native), "python"]


@contextlib.contextmanager
def _path(name):
    if name == "python":
        with native.disabled():
            yield
    else:
        yield


@pytest.fixture(autouse=True)
def fresh_pool(monkeypatch):
    pool = StagingPool()
    monkeypatch.setattr(hostcopy, "POOL", pool)
    return pool


def _fake(monkeypatch, ids):
    """Treat the tensors with these ids as device leaves."""
    ids = {id(x) for x in ids}
    monkeypatch.setattr(hostcopy, "on_device", lambda t: id(t) in ids)


def _tiny(monkeypatch, batch_bytes=1024, entries=5):
    monkeypatch.setattr(leaves_mod, "_BATCH_BYTES", batch_bytes)
    monkeypatch.setattr(leaves_mod, "_BATCH_ENTRIES", entries)


def _mutate_after_put(monkeypatch, victims):
    """Write ``victims`` in place right after their copy is queued."""
    real = Stage.put
    vid = {id(v) for v in victims}

    def put(self, t):
        out = real(self, t)
        if id(t) in vid:
            t.view(-1)[0] += 1.0  # bumps _version once the copy is queued
        return out

    monkeypatch.setattr(Stage, "put", put)


# ---- unsharded unchanged: CPU leaves never reach the staging code ---------------------------


def test_cpu_leaves_never_stage_and_pass_host_tensor_as_is(monkeypatch, tmp_path):
    spy = []
    real_ht = hostcopy.host_tensor

    def ht(t):
        out = real_ht(t)
        spy.append((t, out))
        return out

    monkeypatch.setattr(leaves_mod.hostcopy, "host_tensor", ht, raising=True)
    from verification.commitment import encoding
    monkeypatch.setattr(encoding, "host_tensor", ht)
    monkeypatch.setattr(store_mod, "host_tensor", ht)

    def boom(*a, **k):
        raise AssertionError("a CPU leaf reached the staging code")

    monkeypatch.setattr(Stage, "put", boom)
    monkeypatch.setattr(StagingPool, "acquire", boom)

    c = _Stub([(5,), (3, 4), (0,), (64, 65)], torch.float32)
    ls = c.leaves()
    tens = [x for x in ls if isinstance(x, torch.Tensor)]
    for path in ("native", "python"):
        with _path(path):
            if path == "native" and not native.enabled():
                continue
            leaf_hashes_of(c, enumerate(ls))
            leaf_hashes_until_error(c, enumerate(ls))
            commit_leaves(c, ls)
            hash_step_leaves(c, ls)
            leaf_totals(c, ls)
            leaf_parts(c, 1, ls[1])
            leaf_hash(c, 1, ls[1])
    for t in tens:
        tensor_leaf_payload(t)
        tensor_leaf_payload_unchecked(t)
    store = DiskStore.write(tmp_path / "d", c, ls, commit_leaves(c, ls))
    mem = InMemoryStore.hold(c, list(ls), commit_leaves(c, ls))
    perturb_leaf(c, mem, 1, ls[1] + 1)
    assert store.leaf(1) is not None
    assert spy, "the Python-path entry points should have reached host_tensor"
    assert all(out is t for t, out in spy)  # identity: no copy


# ---- staging under stress ---------------------------------------------------------------------


@needs_native
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("big", [False, True])
def test_many_batches_oversized_empty_and_mixed(monkeypatch, seed, big):
    """Random mix of CPU and 'device' leaves, empties, and leaves larger than a batch."""
    _tiny(monkeypatch, 1024, 5)
    if big:
        monkeypatch.setattr(merkle, "_MT_THRESHOLD", 200)
    g = torch.Generator().manual_seed(seed)
    sizes = [0, 1, 7, 100, 255, 256, 257, 300, 600, 1500]  # 1500 f32 = 6000 B > 1024
    shapes = [(sizes[int(torch.randint(len(sizes), (1,), generator=g))],) for _ in range(60)]
    c = _Stub(shapes[:15], torch.float32)
    c.n_leaves = 1 + 3 * 15
    twin = c.leaves(seed)
    ref = leaf_hashes_of(c, enumerate(twin))
    placed = [x.clone() if isinstance(x, torch.Tensor) else x for x in twin]
    dev_ids = [x for k, x in enumerate(placed)
               if isinstance(x, torch.Tensor) and int(torch.randint(2, (1,), generator=g))]
    _fake(monkeypatch, dev_ids)
    assert leaf_hashes_of(c, enumerate(placed)) == ref
    assert leaf_hashes_until_error(c, enumerate(placed)) == (ref, None)
    assert commit_leaves(c, placed).root == commit_leaves(c, twin).root
    with native.disabled():
        assert leaf_hashes_of(c, enumerate(placed)) == ref
    assert len(hostcopy.POOL._free) == hostcopy.POOL.allocated


@needs_native
def test_oversized_leaf_in_the_middle_of_a_batch(monkeypatch):
    _tiny(monkeypatch, 1024, 50)
    c = _Stub([(100,), (100,), (2000,), (100,), (100,)], torch.float32)
    twin = c.leaves()
    ref = leaf_hashes_of(c, enumerate(twin))
    placed = [x.clone() if isinstance(x, torch.Tensor) else x for x in twin]
    _fake(monkeypatch, [x for x in placed if isinstance(x, torch.Tensor)])
    assert leaf_hashes_of(c, enumerate(placed)) == ref
    assert len(hostcopy.POOL._free) == hostcopy.POOL.allocated <= leaves_mod._MAX_IN_FLIGHT + 1


@needs_native
@pytest.mark.parametrize("empties", [[0], [1, 2], [4], [0, 1, 2, 3, 4, 5, 6, 7, 8]])
def test_empty_device_leaves(monkeypatch, empties):
    _tiny(monkeypatch, 512, 3)
    shapes = [(0,) if k in empties else (40,) for k in range(9)]
    c = _Stub(shapes[:2], torch.bfloat16)  # n_leaves = 1 + 3*2 = 7
    shapes = shapes[:2]
    twin = c.leaves()
    for k in empties:
        if k < len(twin) - 1:
            twin[1 + k] = twin[1 + k][:0].clone()
    ref_err = leaf_hashes_until_error(c, enumerate(twin))
    placed = [x.clone() if isinstance(x, torch.Tensor) else x for x in twin]
    _fake(monkeypatch, [x for x in placed if isinstance(x, torch.Tensor)])
    got = leaf_hashes_until_error(c, enumerate(placed))
    assert got[0] == ref_err[0]
    _same_error(ref_err[1], got[1])


@needs_native
def test_empty_device_leaf_with_matching_declared_shape(monkeypatch):
    _tiny(monkeypatch, 512, 3)
    c = _Stub([(0, 3), (0,), (4,)], torch.float16)
    twin = c.leaves()
    ref = leaf_hashes_of(c, enumerate(twin))
    placed = [x.clone() if isinstance(x, torch.Tensor) else x for x in twin]
    _fake(monkeypatch, [x for x in placed if isinstance(x, torch.Tensor)])
    assert leaf_hashes_of(c, enumerate(placed)) == ref
    assert hostcopy.POOL._free and len(hostcopy.POOL._free) == hostcopy.POOL.allocated


# ---- mutation: the earliest failing leaf decides --------------------------------------------


def _setup(monkeypatch, batch_bytes=2048, entries=4, dtype=torch.float32):
    _tiny(monkeypatch, batch_bytes, entries)
    c = _Stub([(100,)] * 6, dtype)  # 1 + 18 leaves
    twin = c.leaves()
    ref = leaf_hashes_of(c, enumerate(twin))
    placed = [x.clone() if isinstance(x, torch.Tensor) else x for x in twin]
    return c, twin, ref, placed


@needs_native
@pytest.mark.parametrize("where", ["first", "last", "middle"])
def test_mutation_at_a_chosen_leaf(monkeypatch, where):
    c, twin, ref, placed = _setup(monkeypatch)
    k = {"first": 1, "last": len(placed) - 1, "middle": 9}[where]
    _fake(monkeypatch, [x for x in placed if isinstance(x, torch.Tensor)])
    _mutate_after_put(monkeypatch, [placed[k]])
    d, err = leaf_hashes_until_error(c, enumerate(placed))
    assert isinstance(err, StoreMutationError) and str(err).startswith(f"leaf {k} ")
    assert d == ref[:k]
    assert len(hostcopy.POOL._free) == hostcopy.POOL.allocated


@needs_native
@pytest.mark.parametrize("bad_at,mut_at,winner", [(3, 7, "nonfinite"), (7, 3, "mutation"),
                                                  (4, 5, "nonfinite"), (5, 4, "mutation"),
                                                  (1, 2, "nonfinite"), (2, 1, "mutation"),
                                                  (9, 10, "nonfinite"), (10, 9, "mutation")])
@pytest.mark.parametrize("batch_bytes", [2048, 1 << 20])
def test_mutation_versus_nonfinite_earliest_leaf_decides(monkeypatch, bad_at, mut_at, winner,
                                                         batch_bytes):
    """A leaf changed during its copy and a NaN elsewhere: the earlier leaf's error is returned,
    with every digest before it, in one batch or across several."""
    c, twin, ref, placed = _setup(monkeypatch, batch_bytes)
    twin[bad_at].view(-1)[3] = float("nan")
    d0, e0 = leaf_hashes_until_error(c, enumerate(twin))
    placed[bad_at].view(-1)[3] = float("nan")
    _fake(monkeypatch, [x for x in placed if isinstance(x, torch.Tensor)])
    _mutate_after_put(monkeypatch, [placed[mut_at]])
    d, err = leaf_hashes_until_error(c, enumerate(placed))
    if winner == "nonfinite":
        assert d == d0 == ref[:bad_at]
        _same_error(e0, err)
    else:
        assert isinstance(err, StoreMutationError) and str(err).startswith(f"leaf {mut_at} ")
        assert d == ref[:mut_at]
    assert len(hostcopy.POOL._free) == hostcopy.POOL.allocated


@needs_native
def test_mutation_right_after_a_nonfinite_leaf_is_the_nonfinite_error(monkeypatch):
    c, twin, ref, placed = _setup(monkeypatch, 1 << 20)
    twin[5].view(-1)[0] = float("inf")
    d0, e0 = leaf_hashes_until_error(c, enumerate(twin))
    placed[5].view(-1)[0] = float("inf")
    _fake(monkeypatch, [x for x in placed if isinstance(x, torch.Tensor)])
    _mutate_after_put(monkeypatch, [placed[6]])
    d, err = leaf_hashes_until_error(c, enumerate(placed))
    assert d == d0 and len(d) == 5
    _same_error(e0, err)


@needs_native
def test_mutation_in_a_big_leaf(monkeypatch):
    c, twin, ref, placed = _setup(monkeypatch, 1 << 20)
    monkeypatch.setattr(merkle, "_MT_THRESHOLD", 64)
    _fake(monkeypatch, [x for x in placed if isinstance(x, torch.Tensor)])
    _mutate_after_put(monkeypatch, [placed[4]])
    d, err = leaf_hashes_until_error(c, enumerate(placed))
    assert isinstance(err, StoreMutationError) and str(err).startswith("leaf 4 ")
    assert d == ref[:4]


@needs_native
def test_mutation_with_a_cpu_leaf_before_and_after_in_one_batch(monkeypatch):
    c, twin, ref, placed = _setup(monkeypatch, 1 << 20)
    devs = [placed[k] for k in (2, 4, 6)]
    _fake(monkeypatch, devs)
    _mutate_after_put(monkeypatch, [placed[4]])
    d, err = leaf_hashes_until_error(c, enumerate(placed))
    assert str(err).startswith("leaf 4 ") and d == ref[:4]
    with pytest.raises(StoreMutationError):
        leaf_hashes_of(c, enumerate(placed))


# ---- no leak on error -----------------------------------------------------------------------


def _all_released(monkeypatch):
    seen = []
    real = StagingPool.acquire

    def acq(self, *a):
        st = real(self, *a)
        seen.append(st)
        return st

    monkeypatch.setattr(StagingPool, "acquire", acq)
    return seen


@needs_native
@pytest.mark.parametrize("failure", ["mutation", "nonfinite", "malformed", "iterator", "dtype"])
def test_no_live_buffer_after_a_raising_call_and_the_pool_is_reused(monkeypatch, failure):
    c, twin, ref, placed = _setup(monkeypatch, 1024, 3)
    pool = hostcopy.POOL
    seen = _all_released(monkeypatch)
    _fake(monkeypatch, [x for x in placed if isinstance(x, torch.Tensor)])
    items = list(enumerate(placed))
    if failure == "mutation":
        _mutate_after_put(monkeypatch, [placed[10]])
    elif failure == "nonfinite":
        placed[10].view(-1)[1] = float("nan")
    elif failure == "malformed":
        placed[10] = placed[10].reshape(10, 10)
        _fake(monkeypatch, [x for x in placed if isinstance(x, torch.Tensor)])
        items = list(enumerate(placed))
    elif failure == "dtype":
        placed[10] = placed[10].to(torch.float16)
        _fake(monkeypatch, [x for x in placed if isinstance(x, torch.Tensor)])
        items = list(enumerate(placed))
    else:
        fixed = items

        def gen():
            for k, it in enumerate(fixed):
                if k == 11:
                    raise RuntimeError("reader failed")
                yield it
        items = gen()
    with pytest.raises(Exception):
        leaf_hashes_of(c, items)
    assert seen and all(st.released for st in seen)
    assert len(pool._free) == pool.allocated  # every pooled buffer is back
    before = pool.allocated
    # the next call reuses the pooled buffers
    good = [x.clone() if isinstance(x, torch.Tensor) else x for x in c.leaves()]
    _fake(monkeypatch, [x for x in good if isinstance(x, torch.Tensor)])
    assert leaf_hashes_of(c, enumerate(good)) == leaf_hashes_of(c, enumerate(c.leaves()))
    assert pool.allocated <= max(before, leaves_mod._MAX_IN_FLIGHT + 1)
    assert len(pool._free) == pool.allocated


@needs_native
def test_no_leak_when_the_copy_itself_raises(monkeypatch):
    c, twin, ref, placed = _setup(monkeypatch, 1024, 3)
    pool = hostcopy.POOL
    _fake(monkeypatch, [x for x in placed if isinstance(x, torch.Tensor)])
    real, n = Stage.put, [0]

    def put(self, t):
        n[0] += 1
        if n[0] == 7:
            raise RuntimeError("copy failed")
        return real(self, t)

    monkeypatch.setattr(Stage, "put", put)
    d, err = leaf_hashes_until_error(c, enumerate(placed))
    assert isinstance(err, RuntimeError) and d == ref[:7]
    assert len(pool._free) == pool.allocated
    monkeypatch.setattr(Stage, "put", real)
    assert leaf_hashes_of(c, enumerate(placed)) == ref
    assert len(pool._free) == pool.allocated <= leaves_mod._MAX_IN_FLIGHT + 1


# ---- the Python path on device leaves -------------------------------------------------------


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("what", ["good", "nan", "shape"])
def test_python_and_native_paths_agree_on_mixed_leaves(monkeypatch, path, what):
    c, twin, ref, placed = _setup(monkeypatch, 1024, 3)
    if what == "nan":
        twin[7].view(-1)[0] = float("nan")
        placed[7].view(-1)[0] = float("nan")
    elif what == "shape":
        twin[7] = twin[7].reshape(10, 10)
        placed[7] = placed[7].reshape(10, 10)
    _fake(monkeypatch, [x for k, x in enumerate(placed)
                        if isinstance(x, torch.Tensor) and k % 2])
    with _path(path):
        r0 = leaf_hashes_until_error(c, enumerate(twin))
        r1 = leaf_hashes_until_error(c, enumerate(placed))
    assert r0[0] == r1[0]
    _same_error(r0[1], r1[1])


def test_python_path_mutation_during_the_copy_raises(monkeypatch):
    c, twin, ref, placed = _setup(monkeypatch)
    _fake(monkeypatch, [placed[3]])
    real = torch.Tensor.to

    def to(self, *a, **k):
        out = real(self, *a, **k)
        if self.data_ptr() == placed[3].data_ptr():
            self.view(-1)[0] += 1
        return out

    monkeypatch.setattr(torch.Tensor, "to", to)
    with native.disabled(), pytest.raises(StoreMutationError):
        leaf_hashes_of(c, enumerate(placed))


# ---- real MPS leaves mixed with CPU leaves ---------------------------------------------------

needs_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no mps")


@needs_native
@needs_mps
@pytest.mark.parametrize("seed", range(3))
def test_mps_and_cpu_leaves_mixed_over_many_small_batches(monkeypatch, seed):
    _tiny(monkeypatch, 1024, 5)
    g = torch.Generator().manual_seed(seed)
    sizes = [0, 1, 100, 255, 300, 1500]
    c = _Stub([(sizes[int(torch.randint(len(sizes), (1,), generator=g))],) for _ in range(12)],
              torch.bfloat16)
    twin = c.leaves(seed)
    ref = leaf_hashes_of(c, enumerate(twin))
    placed = [x.to("mps") if isinstance(x, torch.Tensor) and int(torch.randint(2, (1,), generator=g))
              else x for x in twin]
    assert leaf_hashes_of(c, enumerate(placed)) == ref
    assert commit_leaves(c, placed).root == commit_leaves(c, twin).root
    k = max(i for i, x in enumerate(placed) if isinstance(x, torch.Tensor) and x.device.type == "mps"
            and x.numel())
    _mutate_after_put(monkeypatch, [placed[k]])
    d, err = leaf_hashes_until_error(c, enumerate(placed))
    assert isinstance(err, StoreMutationError) and str(err).startswith(f"leaf {k} ")
    assert d == ref[:k]
    assert len(hostcopy.POOL._free) == hostcopy.POOL.allocated


@needs_native
def test_the_same_device_tensor_at_two_leaves_and_totals(monkeypatch):
    """A tied weight is one object at two leaf positions; both copies are staged and hashed."""
    _tiny(monkeypatch, 1024, 5)
    c = _Stub([(100,), (100,)], torch.float32)
    twin = c.leaves()
    twin[5] = twin[1]  # W_{t+1}[w0] is the same tensor as W_t[w0]'s slot, as a tied weight is
    ref = leaf_hashes_of(c, enumerate(twin))
    placed = [x.clone() if isinstance(x, torch.Tensor) else x for x in twin]
    placed[5] = placed[1]
    _fake(monkeypatch, [x for x in placed if isinstance(x, torch.Tensor)])
    assert leaf_hashes_of(c, enumerate(placed)) == ref
    assert leaf_totals(c, placed) == leaf_totals(c, twin)
    assert len(hostcopy.POOL._free) == hostcopy.POOL.allocated
