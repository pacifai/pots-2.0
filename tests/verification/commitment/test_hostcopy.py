"""Device leaves (FS32b): every leaf consumer gives a device leaf's host twin's digests and
errors, through one host copy. ``fake`` treats CPU tensors as device leaves (the staging code
runs, the bytes never leave the host); ``mps`` and ``cuda`` run where the machine has them."""

import contextlib
import gc

import pytest
import torch

from setup.records import Record
from verification.commitment import hostcopy, merkle, native
from verification.commitment import leaves as leaves_mod
from verification.commitment.encoding import EncodingError, tensor_leaf_payload
from verification.commitment.hostcopy import Stage, StagingPool, host_tensor
from verification.commitment.leaves import (
    commit_leaves,
    leaf_hashes_of,
    leaf_hashes_until_error,
)
from verification.commitment.merkle import MerkleTree
from verification.transcript.errors import StoreMutationError
from verification.transcript.reader import stack_block
from verification.transcript.store import DiskStore, InMemoryStore, perturb_leaf
from verification.verifier.context import CommittedLeaves

_HAS = {"fake": True, "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available()}


def _kinds(*names):
    return [pytest.param(k, marks=pytest.mark.skipif(not _HAS[k], reason=f"no {k}"))
            for k in names]


needs_native = pytest.mark.skipif(not native.enabled(), reason="the native library is unavailable")

DTYPES = [torch.float32, torch.bfloat16, torch.float16, torch.int32]
SHAPES = [(5,), (3, 7), (2, 3, 4), (1,), (0, 3), (64, 65), (4096,)]


class _Spec:
    def __init__(self, m, shape):
        self.m, self.name, self.p_shape = m, f"p{m}", shape


class _Stub:
    """One record, then `W_t`, the products and `W_{t+1}` of the given shapes, all of `dtype`."""

    n_s = 1

    def __init__(self, shapes, dtype):
        self.weight_names = [f"w{i}" for i in range(len(shapes))]
        self.weight_shapes = dict(zip(self.weight_names, shapes))
        self.weight_dtype = self.product_dtype = dtype
        self.n_w = self.M = len(shapes)
        self._pshapes = list(shapes)
        self.n_leaves = 1 + 2 * self.n_w + self.M

    def product(self, m):
        return _Spec(m, self._pshapes[m - 1])

    def encode_record(self, obj):
        return bytes(obj.ids.numpy())

    def leaves(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        out: list = [Record(torch.arange(4, dtype=torch.int32), torch.arange(4, dtype=torch.int32),
                           torch.ones(4, dtype=torch.int32))]
        for _ in range(3):
            for s in self._pshapes:
                if self.weight_dtype == torch.int32:
                    out.append(torch.randint(-1000, 1000, s, generator=g, dtype=torch.int32))
                else:
                    out.append((torch.randn(*s, generator=g) * 10).to(self.weight_dtype))
        return out


class _Dev:
    """Where a test's device leaves live."""

    def __init__(self, kind, index=0):
        self.kind, self.index = kind, index

    def place(self, t):
        if not isinstance(t, torch.Tensor):
            return t
        if self.kind == "fake":
            return t.clone()
        return t.to(f"cuda:{self.index}" if self.kind == "cuda" else self.kind)

    @contextlib.contextmanager
    def active(self, monkeypatch):
        with monkeypatch.context() as mp:
            if self.kind == "fake":
                mp.setattr(hostcopy, "on_device", lambda t: True)
            yield


@pytest.fixture(params=_kinds("fake", "mps", "cuda"))
def dev(request):
    return _Dev(request.param)


@pytest.fixture(params=["default", "small_batches", "big_leaves", "both"])
def mode(request, monkeypatch):
    """Default limits; batches of 3 leaves or 2 KiB (staging reused across batches, and a leaf
    alone in one); every tensor leaf on the big path; both."""
    if request.param in ("small_batches", "both"):
        monkeypatch.setattr(leaves_mod, "_BATCH_ENTRIES", 3)
        monkeypatch.setattr(leaves_mod, "_BATCH_BYTES", 2048)
    if request.param in ("big_leaves", "both"):
        monkeypatch.setattr(merkle, "_MT_THRESHOLD", 64)
    return request.param


@pytest.fixture(autouse=True)
def fresh_pool(monkeypatch):
    monkeypatch.setattr(hostcopy, "POOL", StagingPool())


def _same_error(a, b):
    assert (a is None) == (b is None)
    if a is not None:
        assert type(a) is type(b) and str(a) == str(b)


_PATHS = [pytest.param("native", marks=needs_native), "python"]


@contextlib.contextmanager
def _path(name):
    if name == "python":
        with native.disabled():
            yield
    else:
        yield


# ---- the seam ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_cpu_tensor_passes_through_without_a_copy(dtype):
    t = torch.arange(12).to(dtype).reshape(3, 4)
    assert host_tensor(t) is t
    assert not hostcopy.on_device(t)
    want = t.contiguous().view(torch.uint8).numpy().tobytes()  # raw bytes, bf16 included
    assert bytes(tensor_leaf_payload(t)) == want


def test_device_tensor_is_copied_and_equal(dev, monkeypatch):
    t = dev.place((torch.randn(6, 5) * 3).to(torch.bfloat16))
    with dev.active(monkeypatch):
        h = host_tensor(t)
    assert h.device.type == "cpu" and torch.equal(h, t.cpu())


def test_meta_tensor_keeps_its_error():
    with pytest.raises(EncodingError, match="^tensor must be on CPU$"):
        tensor_leaf_payload(torch.zeros(3, device="meta"))


# ---- digests and errors, every path -------------------------------------------------------


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_digests_equal_the_cpu_twins(dev, dtype, mode, path, monkeypatch):
    c = _Stub(SHAPES, dtype)
    twin = c.leaves()
    ref = leaf_hashes_of(c, enumerate(twin))
    placed = [dev.place(x) for x in twin]
    with dev.active(monkeypatch), _path(path):
        assert leaf_hashes_of(c, enumerate(placed)) == ref
        d, err = leaf_hashes_until_error(c, enumerate(placed))
        assert err is None and d == ref
        assert commit_leaves(c, placed).root == MerkleTree(ref).root


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("leaf,pos", [(1, 0), (4, 3), (8, -1), (13, 100)])
def test_nonfinite_gives_the_same_first_error(dev, dtype, bad, leaf, pos, mode, path,
                                              monkeypatch):
    c = _Stub(SHAPES, dtype)
    twin = c.leaves()
    flat = twin[leaf].view(-1)
    flat[pos % flat.numel()] = bad
    placed = [dev.place(x) for x in twin]
    with _path(path):
        d0, e0 = leaf_hashes_until_error(c, enumerate(twin))
        with dev.active(monkeypatch):
            d1, e1 = leaf_hashes_until_error(c, enumerate(placed))
            with pytest.raises(type(e0)) as ei:
                leaf_hashes_of(c, enumerate(placed))
    assert d1 == d0 and len(d0) == leaf
    _same_error(e0, e1)
    _same_error(e0, ei.value)


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("case", ["shape", "dtype", "non-contiguous"])
def test_structural_errors_are_the_twins(dev, case, mode, path, monkeypatch):
    c = _Stub(SHAPES, torch.float32)
    twin = c.leaves()
    leaf, f = {"shape": (2, lambda t: t.reshape(-1)),
               "dtype": (3, lambda t: t.to(torch.float16)),
               "non-contiguous": (6, lambda t: t.t())}[case]
    twin[leaf] = f(twin[leaf])
    with _path(path):
        d0, e0 = leaf_hashes_until_error(c, enumerate(twin))
        with dev.active(monkeypatch):
            d1, e1 = leaf_hashes_until_error(c, enumerate(dev.place(x) for x in twin))
    assert e0 is not None and d0 == d1 and len(d0) == leaf
    _same_error(e0, e1)


# ---- the native path's staging ---------------------------------------------------------------


@needs_native
def test_changed_leaf_raises_at_that_leaf(dev, monkeypatch):
    """A leaf written in place after its copy is queued is a StoreMutationError at its index;
    the leaves before it are hashed."""
    monkeypatch.setattr(leaves_mod, "_BATCH_BYTES", 2048)
    c = _Stub(SHAPES, torch.float32)
    twin = c.leaves()
    ref = leaf_hashes_of(c, enumerate(twin))
    placed = [dev.place(x) for x in twin]
    victim = placed[5]
    real = Stage.put

    def put(self, t):
        out = real(self, t)
        if t is victim:
            t.add_(1.0)  # bumps _version after the copy was queued
        return out

    with dev.active(monkeypatch):
        monkeypatch.setattr(Stage, "put", put)
        d, err = leaf_hashes_until_error(c, enumerate(placed))
    assert isinstance(err, StoreMutationError) and str(err).startswith("leaf 5 ")
    assert d == ref[:5]


@needs_native
def test_staging_memory_is_batches_in_flight_plus_one(dev, monkeypatch):
    monkeypatch.setattr(leaves_mod, "_BATCH_BYTES", 4096)
    monkeypatch.setattr(leaves_mod, "_BATCH_ENTRIES", 4)
    c = _Stub([(256,)] * 40, torch.float32)  # 1 KiB leaves: four leaves a batch
    twin = c.leaves()
    ref = leaf_hashes_of(c, enumerate(twin))
    placed = [dev.place(x) for x in twin]
    pool = hostcopy.POOL
    live = {"now": 0, "max": 0}
    acquire, release = pool.acquire, pool.release

    def acq(*a):
        live["now"] += 1
        live["max"] = max(live["max"], live["now"])
        st = acquire(*a)
        assert st.buf.numel() <= 4096  # no batch stages more than the byte limit
        return st

    def rel(st):
        if not st.released:
            live["now"] -= 1
        release(st)

    monkeypatch.setattr(pool, "acquire", acq)
    monkeypatch.setattr(pool, "release", rel)
    with dev.active(monkeypatch):
        for _ in range(3):  # steps reuse the same buffers
            assert leaf_hashes_of(c, enumerate(placed)) == ref
    assert live["max"] <= leaves_mod._MAX_IN_FLIGHT + 1 and live["now"] == 0
    assert pool.allocated <= leaves_mod._MAX_IN_FLIGHT + 1
    if dev.kind == "cuda":
        assert all(b.is_pinned() for b in pool._free)


@needs_native
def test_a_leaf_larger_than_a_batch_is_staged_alone(dev, monkeypatch):
    monkeypatch.setattr(leaves_mod, "_BATCH_BYTES", 1024)
    c = _Stub([(8,), (1000,), (8,)], torch.float32)  # the 4,000-byte leaf alone
    twin = c.leaves()
    ref = leaf_hashes_of(c, enumerate(twin))
    with dev.active(monkeypatch):
        assert leaf_hashes_of(c, enumerate(dev.place(x) for x in twin)) == ref
    assert hostcopy.POOL.allocated <= leaves_mod._MAX_IN_FLIGHT + 1


def test_stage_put_returns_the_staged_bytes(dev):
    pool = StagingPool()
    st = pool.acquire(1 << 12, 100, dev.kind == "cuda")
    a = dev.place(torch.randn(3, 5).to(torch.bfloat16))
    b = dev.place(torch.randn(7))
    ha, hb = st.put(a), st.put(b)
    st.fence()
    st.wait()
    assert ha.data_ptr() % 64 == 0 and hb.data_ptr() % 64 == 0
    assert torch.equal(ha, a.cpu()) and torch.equal(hb, b.cpu())
    pool.release(st)
    assert pool.acquire(1 << 12, 1, False).buf is st.buf


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
@needs_native
def test_leaves_on_two_devices_in_one_call():
    c = _Stub(SHAPES, torch.float32)
    twin = c.leaves()
    ref = leaf_hashes_of(c, enumerate(twin))
    placed = [_Dev("cuda", i % 2).place(x) for i, x in enumerate(twin)]
    assert {x.device.index for x in placed if isinstance(x, torch.Tensor)} == {0, 1}
    assert leaf_hashes_of(c, enumerate(placed)) == ref
    with native.disabled():
        assert leaf_hashes_of(c, enumerate(placed)) == ref
    twin[8].view(-1)[0] = float("nan")
    placed[8] = _Dev("cuda", 0).place(twin[8])
    _same_error(leaf_hashes_until_error(c, enumerate(twin))[1],
                leaf_hashes_until_error(c, enumerate(placed))[1])


# ---- backpressure on the Python path ----------------------------------------------------------


class _Buf(bytearray):
    """A part that counts how many are alive: the stand-in for a leaf's host copy."""

    live = 0
    peak = 0

    def __init__(self, n):
        super().__init__(n)
        type(self).live += 1
        type(self).peak = max(type(self).peak, type(self).live)

    def __del__(self):
        type(self).live -= 1


@pytest.mark.parametrize("fail_at", [None, 150])
def test_python_path_bounds_the_live_host_copies(monkeypatch, fail_at):
    import time

    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    monkeypatch.setattr(merkle, "_CHUNK_BYTES", 1024)
    hash_chunk = merkle._hash_chunk

    def slow(chunk):
        time.sleep(0.02)  # the workers are slower than the producer
        return hash_chunk(chunk)

    monkeypatch.setattr(merkle, "_hash_chunk", slow)
    n, size, per_chunk = 200, 256, 4
    _Buf.live = _Buf.peak = 0

    def produce():
        for i in range(n):
            if i == fail_at:
                raise ValueError("producer failed")
            yield (_Buf(size),)

    digests, err = merkle.hash_leaves_until_error(produce())
    cut = n if fail_at is None else fail_at
    assert digests == [merkle.hash_leaf(bytes(size))] * cut
    assert (err is None) == (fail_at is None)
    if err is not None:
        assert str(err) == "producer failed"
    # the chunk being filled, the pending ones, and the one just collected
    assert _Buf.peak <= (merkle._MAX_PENDING + 2) * per_chunk + per_chunk
    assert _Buf.peak < n // 2  # without the cap the producer would run far ahead
    err = None  # the error's traceback and the frame form a cycle that holds the last chunk
    gc.collect()
    assert _Buf.live == 0


# ---- zero-copy runs -------------------------------------------------------------------------


@pytest.mark.parametrize("kind", _kinds("mps", "cuda"))
def test_runs_of_device_members_are_views(kind):
    d = _Dev(kind)
    big = d.place(torch.randn(6, 3, 5))
    assert big.data_ptr() % 64 == 0
    ts = [big[i] for i in range(6)]
    out = stack_block(ts)
    assert out.data_ptr() == big.data_ptr() and out._base is not None
    assert torch.equal(out.cpu(), torch.stack(ts).cpu())
    assert stack_block(ts, own=True).data_ptr() != big.data_ptr()
    cl = CommittedLeaves()
    for t in ts:
        cl.add(t)
    run = cl.tensor_run(0, 1, 6)
    assert run.data_ptr() == big.data_ptr() and torch.equal(run.cpu(), big.cpu())
    sub = cl.tensor_run(1, 2, 3)
    assert torch.equal(sub.cpu(), big[1:6:2].cpu())
    big.add_(1.0)  # every member's version moves: served by tensor_block, which raises
    with pytest.raises(StoreMutationError):
        cl.tensor_run(0, 1, 6)


# ---- the stores -----------------------------------------------------------------------------


def test_in_memory_store_guards_and_perturbs_device_leaves(dev, monkeypatch):
    c = _Stub([(5,), (3, 4)], torch.float32)
    twin = c.leaves()
    placed = [dev.place(x) for x in twin]
    with dev.active(monkeypatch):
        tree = commit_leaves(c, placed)
        assert tree.root == commit_leaves(c, twin).root
        store = InMemoryStore.hold(c, list(placed), tree)
        assert torch.equal(store.leaf(2).cpu(), twin[2])
        twin_store = InMemoryStore.hold(c, list(twin), commit_leaves(c, twin))
        repl = twin[2] + 1
        assert perturb_leaf(c, store, 2, dev.place(repl)) == perturb_leaf(c, twin_store, 2, repl)
        placed[3].add_(1.0)
        with pytest.raises(StoreMutationError):
            store.leaf(3)


def test_disk_store_writes_the_host_copy(dev, tmp_path, monkeypatch):
    c = _Stub([(5,), (3, 4)], torch.bfloat16)
    twin = c.leaves()
    big = torch.randn(4, 3, 4).to(torch.bfloat16)
    twin[2] = big[2]  # a view of a larger storage, as a bmm member is
    placed = [dev.place(x) for x in twin]
    placed[2] = dev.place(big)[2]
    with dev.active(monkeypatch):
        ds = DiskStore.write(tmp_path / "dev", c, placed, commit_leaves(c, placed))
    ref = DiskStore.write(tmp_path / "cpu", c, twin, commit_leaves(c, twin))
    assert ds.root == ref.root
    for i in range(1, len(twin)):
        a, b = ds.leaf(i), ref.leaf(i)
        assert a.device.type == "cpu" and a.dtype == b.dtype and torch.equal(a, b)
        assert (tmp_path / "dev" / f"leaf_{i:05d}.pt").read_bytes() == \
            (tmp_path / "cpu" / f"leaf_{i:05d}.pt").read_bytes()
    assert (tmp_path / "dev" / "leaf_hashes.bin").read_bytes() == \
        (tmp_path / "cpu" / "leaf_hashes.bin").read_bytes()


def test_synchronous_copy_checks_the_version(dev, monkeypatch):
    t = dev.place(torch.randn(8))
    real = torch.Tensor.to

    def to(self, *a, **k):
        out = real(self, *a, **k)
        self.add_(1.0)  # written while the copy runs
        return out

    with dev.active(monkeypatch):
        monkeypatch.setattr(torch.Tensor, "to", to)
        with pytest.raises(StoreMutationError):
            host_tensor(t)


# ---- CUDA only: ordering on the current stream ------------------------------------------------


@pytest.mark.skipif(not _HAS["cuda"], reason="no cuda")
@needs_native
@pytest.mark.parametrize("lazy", [False, True])
def test_cuda_leaf_made_on_the_current_stream_needs_no_sync(lazy):
    n = 8192
    c = _Stub([(n, n)], torch.float32)
    x = torch.randn(n, n, device="cuda") / n ** 0.5
    y = x
    for _ in range(30):  # a long chain, so the copy would start long before it ends
        y = y @ x
    rec = Record(torch.arange(4, dtype=torch.int32), torch.arange(4, dtype=torch.int32),
                 torch.ones(4, dtype=torch.int32))
    leaves = [rec, x, y, y @ x]  # record, W_t, product, W_{t+1}; the last two are unsynced
    items = list(enumerate(leaves))
    got = leaf_hashes_of(c, iter(items) if lazy else items)  # no synchronize before this
    torch.cuda.synchronize()
    twin = [leaves[0]] + [t.cpu() for t in leaves[1:]]
    assert got == leaf_hashes_of(c, list(enumerate(twin)))


@pytest.mark.skipif(not _HAS["cuda"], reason="no cuda")
def test_cuda_check_5_measure_on_a_stacked_view_equals_torch_stack():
    from verification.verifier.matmul_check.freivalds import measure_products_rows

    B, n, q, w = 6, 64, 48, 32
    a = torch.randn(B, n, q, device="cuda")
    b = torch.randn(B, q, w, device="cuda")
    p = torch.bmm(a, b)
    members = [p[i] for i in range(B)]
    view = stack_block(members)
    assert view.data_ptr() == p.data_ptr()  # a view of the bmm block
    copy = torch.stack(members)
    kw = dict(h=b"\x01" * 32, ms=list(range(1, B + 1)), k=4)
    rows_view = measure_products_rows(a, b, view, **kw)
    rows_copy = measure_products_rows(a, b, copy, **kw)
    assert rows_view.cpu().numpy().tobytes() == rows_copy.cpu().numpy().tobytes()
