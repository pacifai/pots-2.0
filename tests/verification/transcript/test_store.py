import gc
import weakref

import pytest
import torch
from torch import nn

from verification.commitment.encoding import TAG_PRODUCT, TAG_WEIGHT, encode_tensor_leaf
from verification.commitment.leaves import (
    dataset_tree,
    leaf_hash,
    leaf_hashes,
    leaf_parts,
    transcript_root,
)
from verification.commitment.merkle import MerkleTree, hash_leaf, verify_path
from verification.computation.instances.mlp import MLPComputation, init_weights, synthetic_dataset
from verification.prover.step import StepOutput, commit, prove_step
from verification.transcript.errors import LeafReadError, StoreMutationError
from verification.transcript.reader import TranscriptView
from verification.transcript.store import (
    HASHES_FILE,
    IN_MEMORY,
    LEAF_FILE,
    META_FILE,
    DiskHandoff,
    DiskStore,
    InMemoryStore,
    TranscriptStore,
    perturb_leaf,
)
from verification.verifier.bands import Bands
from verification.verifier.driver import Verifier

ETA = 1e-2


@pytest.fixture
def c():
    return MLPComputation((16, 32, 32, 8), n_s=4, eta=ETA)


@pytest.fixture
def data(c):
    return synthetic_dataset(c.widths, 12, seed=0)


def _step(c, data) -> StepOutput:
    w0 = init_weights(c.widths, seed=0)
    return prove_step(c, c.build_model(), w0, data[4:8])


def _store(c, data, **kw) -> InMemoryStore:
    tree = dataset_tree(c, data)
    paths = [tree.path(i) for i in range(4, 8)]
    return InMemoryStore.from_step(c, _step(c, data), dataset_paths=paths, **kw)


def _flip(obj: torch.Tensor) -> torch.Tensor:
    t = obj.detach().clone()
    t.view(torch.uint8).view(-1)[0] ^= 0x01  # low mantissa byte of entry 0
    return t


def test_root_deterministic(c, data):
    a, b = _step(c, data), _step(c, data)
    assert commit(c, a).root == commit(c, b).root
    assert InMemoryStore.from_step(c, a).root == commit(c, b).root


def test_commit_layout_and_count(c, data):
    step = _step(c, data)
    tree = commit(c, step)
    assert tree.n_leaves == c.n_leaves == 4 + 2 * 3 + 8
    for i, x in enumerate(step.leaves()):
        assert tree.leaf(i) == leaf_hash(c, i, x)
    # Tags: weights 0x02, products 0x03; records are the instance's own encoding.
    name = c.weight_names[0]
    assert b"".join(leaf_parts(c, c.w_t_index(name), step.w_t[name])) == \
        encode_tensor_leaf(TAG_WEIGHT, step.w_t[name])
    assert b"".join(leaf_parts(c, c.w_next_index(name), step.w_next[name])) == \
        encode_tensor_leaf(TAG_WEIGHT, step.w_next[name])
    assert b"".join(leaf_parts(c, c.product_index(1), step.products[0])) == \
        encode_tensor_leaf(TAG_PRODUCT, step.products[0])
    with pytest.raises(ValueError, match="leaves"):
        InMemoryStore.from_step(c, StepOutput(step.records[:3], step.w_t, step.products,
                                              step.w_next, step.loss, step.versions[1:]))


def test_failed_perturb_leaves_store_unchanged(c, data):
    store = _store(c, data)
    h, i = store.root, c.product_index(2)
    orig = store.leaf(i)
    for bad in (orig.t().contiguous(), orig.double(), orig.clone().fill_(float("nan"))):
        with pytest.raises(ValueError):
            perturb_leaf(c, store, i, bad)
        assert store.root == h and torch.equal(store.leaf(i), orig)
    with pytest.raises(IndexError):
        perturb_leaf(c, store, c.n_leaves, orig)
    assert transcript_root(c, store) == h


def test_leaf_order_matches_index_helpers(c, data):
    step = _step(c, data)
    store = InMemoryStore.from_step(c, step)
    view = TranscriptView(c, store)
    for i in range(c.n_s):
        assert torch.equal(store.leaf(c.record_index(i)), step.records[i])
        assert torch.equal(view.record(i), step.records[i])
    for n in c.weight_names:
        assert torch.equal(store.leaf(c.w_t_index(n)), step.w_t[n])
        assert torch.equal(store.leaf(c.w_next_index(n)), step.w_next[n])
    for m in range(1, c.M + 1):
        assert torch.equal(store.leaf(c.product_index(m)), step.products[m - 1])
    with pytest.raises(IndexError):
        store.leaf(c.n_leaves)


def test_byte_flip_changes_root(c, data):
    store = _store(c, data)
    h = store.root
    for i in range(c.n_leaves):
        orig = store.leaf(i)
        assert perturb_leaf(c, store, i, _flip(orig)) != h, f"leaf {i}"
        assert transcript_root(c, store) != h
        assert perturb_leaf(c, store, i, orig) == h


def test_byte_flip_in_encoded_bytes_changes_root(c, data):
    # Any byte of the full leaf encoding, header included.
    store = _store(c, data)
    hashes = leaf_hashes(c, store)
    tree = MerkleTree(hashes)
    for i in (0, c.w_t_index(c.weight_names[1]), c.product_index(3), c.n_leaves - 1):
        enc = bytearray(b"".join(leaf_parts(c, i, store.leaf(i))))
        for pos in (0, 1, 2, len(enc) - 1):
            enc[pos] ^= 0x80
            assert tree.update_leaf(i, hash_leaf(bytes(enc))) != store.root
            enc[pos] ^= 0x80
        assert tree.update_leaf(i, hashes[i]) == store.root


def test_check2_recomputation_matches(c, data):
    store = _store(c, data)
    assert transcript_root(c, store) == store.root


def test_paths_verify_with_declared_count(c, data):
    store = _store(c, data)
    kinds = {
        "record": c.record_index(1),
        "w_t": c.w_t_index(c.weight_names[2]),
        "product": c.product_index(c.M),
        "w_next": c.w_next_index(c.weight_names[0]),
    }
    for kind, i in kinds.items():
        lh = leaf_hash(c, i, store.leaf(i))
        assert verify_path(lh, i, c.n_leaves, store.path(i), store.root), kind
        assert not verify_path(lh, i ^ 1, c.n_leaves, store.path(i), store.root), kind
    assert all(verify_path(leaf_hash(c, i, store.leaf(i)), i, c.n_leaves, store.path(i),
                           store.root) for i in range(c.n_leaves))


def test_record_leaves_equal_dataset_leaves(c, data):
    # P9b: a batch record's leaf is its h_D leaf, so check 4 is one path verification.
    store = _store(c, data)
    tree = dataset_tree(c, data)
    for i, d_index in enumerate(range(4, 8)):
        rec = store.leaf(c.record_index(i))
        assert b"".join(leaf_parts(c, i, rec)) == c.encode_record(data[d_index])
        assert leaf_hash(c, i, rec) == tree.leaf(d_index)
        lh, path = leaf_hash(c, i, rec), store.dataset_path(i)
        assert verify_path(lh, d_index, len(data), path, tree.root)
        assert not verify_path(lh, d_index - 1, len(data), path, tree.root)
        assert not verify_path(lh, d_index + 1, len(data), path, tree.root)
        # Invariant 7: the path doesn't bind |D|. It still verifies at |D| + 1, so check 4 must
        # take the count from the manifest, never from the prover.
        assert verify_path(lh, d_index, len(data) + 1, path, tree.root)
    with pytest.raises(LookupError):
        InMemoryStore.from_step(c, _step(c, data)).dataset_path(0)


def test_perturb_matches_full_rebuild(c, data):
    store = _store(c, data)
    i = c.product_index(4)
    bad = store.leaf(i) * 1.001
    new_root = perturb_leaf(c, store, i, bad)
    assert new_root == store.root == transcript_root(c, store)
    step = _step(c, data)
    leaves = step.leaves()
    leaves[i] = bad
    assert MerkleTree([leaf_hash(c, j, x) for j, x in enumerate(leaves)]).root == new_root
    assert torch.equal(store.leaf(i), bad)


def test_perturb_is_not_on_the_verifier_interface():
    assert not hasattr(TranscriptStore, "perturb_leaf")
    assert {"leaf", "root", "path", "dataset_path"} <= set(TranscriptStore.__abstractmethods__)
    assert not hasattr(TranscriptStore, "n_leaves")


def test_store_immune_to_prover_mutation_copy(c, data):
    step = _step(c, data)
    store = InMemoryStore.from_step(c, step, copy=True)
    h = store.root
    for t in (step.records[0], step.w_t[c.weight_names[0]], step.products[0],
              step.w_next[c.weight_names[0]]):
        with torch.no_grad():
            t.add_(1.0)
    step.products[1].data.mul_(2.0)  # bypasses the version counter; copy is immune anyway
    assert transcript_root(c, store) == h
    for i, x in enumerate(step.leaves()):
        assert store.leaf(i).data_ptr() != x.data_ptr()


def test_store_guards_prover_mutation_zero_copy(c, data):
    step = _step(c, data)
    store = InMemoryStore.from_step(c, step)
    h = store.root
    p = c.product_index(1)
    assert store.leaf(p).data_ptr() == step.products[0].data_ptr()  # handed over, not copied
    with torch.no_grad():
        step.products[0].add_(1.0)
    with pytest.raises(StoreMutationError):
        store.leaf(p)
    # A verifier-side in-place write is caught the same way.
    store2 = InMemoryStore.from_step(c, _step(c, data))
    w = c.w_t_index(c.weight_names[0])
    with torch.no_grad():
        store2.leaf(w).mul_(0.5)
    with pytest.raises(StoreMutationError):
        store2.leaf(w)
    with pytest.raises(StoreMutationError):
        transcript_root(c, store2)
    assert store2.root == h


def test_mutation_before_handoff_is_rejected(c, data):
    step = _step(c, data)
    with torch.no_grad():
        step.products[2].add_(1.0)
    with pytest.raises(RuntimeError, match="mutated"):
        InMemoryStore.from_step(c, step)


def _reachable(root: object) -> list[object]:
    seen, out, stack = set(), [], [root]
    while stack:
        x = stack.pop()
        if id(x) in seen or isinstance(x, (type, type(gc), type(_reachable))):
            continue
        seen.add(id(x))
        out.append(x)
        stack.extend(gc.get_referents(x))
    return out


def test_handoff_boundary(c, data):
    w0 = init_weights(c.widths, seed=0)
    model = c.build_model()
    step = prove_step(c, model, w0, data[4:8])
    store = InMemoryStore.from_step(c, step)
    reach = _reachable(store)
    assert not any(isinstance(x, (StepOutput, nn.Module)) for x in reach)
    assert not any(x is step or x is model for x in reach)
    # The store doesn't keep the prover's objects alive.
    step_ref, model_ref = weakref.ref(step), weakref.ref(model)
    h = store.root
    del step, model
    gc.collect()
    assert step_ref() is None and model_ref() is None
    assert transcript_root(c, store) == h


def test_replay_runs_from_store(c, data):
    store = _store(c, data)
    replay = c.replay(store)
    view = TranscriptView(c, store)
    for m in range(1, c.M + 1):
        a, b = replay.operands(m)
        assert torch.allclose(a @ b, view.product(m), atol=1e-5)
    assert replay.glue_gradients() == {}


# ---- the on-disk store (A14, C3) ----------------------------------------------------------


def _disk(c, data, directory, **kw):
    tree = dataset_tree(c, data)
    paths = [tree.path(i) for i in range(4, 8)]
    leaves, tree_h = InMemoryStore.commit_step(c, _step(c, data))
    mem = InMemoryStore.hold(c, leaves, tree_h, paths)
    return mem, DiskStore.write(directory, c, leaves, tree_h, paths, **kw)


def _same(a, b):
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a, b))
    return all(_same(getattr(a, k), getattr(b, k)) for k in ("ids", "targets", "mask"))


def _tensors_of(obj):
    return [obj] if isinstance(obj, torch.Tensor) else [obj.ids, obj.targets, obj.mask]


def test_disk_store_round_trip(c, data, tmp_path):
    mem, disk = _disk(c, data, tmp_path / "step")
    assert disk.root == mem.root and transcript_root(c, disk) == mem.root
    for i in range(c.n_leaves):
        assert _same(disk.leaf(i), mem.leaf(i))
        assert leaf_hash(c, i, disk.leaf(i)) == leaf_hash(c, i, mem.leaf(i))
        assert disk.path(i) == mem.path(i)
    for i in range(c.n_s):
        assert disk.dataset_path(i) == mem.dataset_path(i)
    assert sorted(p.name for p in (tmp_path / "step").iterdir()) == sorted(
        [LEAF_FILE.format(i) for i in range(c.n_leaves)] + [META_FILE, HASHES_FILE])


def test_disk_reads_are_fresh_and_share_nothing(c, data, tmp_path):
    mem, disk = _disk(c, data, tmp_path / "step")
    i = c.product_index(1)
    a, b = disk.leaf(i), disk.leaf(i)
    assert a is not b and a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
    assert a.untyped_storage().data_ptr() != mem.leaf(i).untyped_storage().data_ptr()
    a.add_(1.0)  # a reader's write can't reach the file or the next read
    assert _same(disk.leaf(i), mem.leaf(i))


def test_disk_write_of_a_view_saves_only_the_view(c, tmp_path):
    base = torch.arange(40, dtype=torch.float32).reshape(5, 8)
    torch.save(_to_disk_for_test(base[1:3]), tmp_path / "v.pt")
    torch.save(_to_disk_for_test(base), tmp_path / "b.pt")
    assert (tmp_path / "v.pt").stat().st_size < (tmp_path / "b.pt").stat().st_size
    assert torch.equal(torch.load(tmp_path / "v.pt", weights_only=True), base[1:3])


def _to_disk_for_test(obj):
    from verification.transcript.store import _to_disk
    return _to_disk(obj)


def test_disk_store_bad_files_are_prover_data_errors(c, data, tmp_path):
    _, disk = _disk(c, data, tmp_path / "step")
    d = tmp_path / "step"
    with pytest.raises(IndexError):
        disk.leaf(c.n_leaves)  # no such file
    with pytest.raises(IndexError):
        disk.leaf(-1)
    (d / LEAF_FILE.format(3)).write_bytes(b"not a torch file")
    with pytest.raises(LeafReadError):
        disk.leaf(3)
    torch.save({"x": torch.zeros(2)}, d / LEAF_FILE.format(5))
    with pytest.raises(LeafReadError):
        disk.leaf(5)
    torch.save({"ids": torch.zeros(3, dtype=torch.int64), "targets": torch.zeros(3),
                "mask": torch.zeros(3)}, d / LEAF_FILE.format(0))
    with pytest.raises(ValueError):  # RecordError, from Record's own validation
        disk.leaf(0)
    (d / LEAF_FILE.format(6)).unlink()
    with pytest.raises(IndexError):
        disk.leaf(6)
    (d / META_FILE).unlink()
    with pytest.raises(LeafReadError):
        _ = disk.root
    with pytest.raises(LeafReadError):
        disk.dataset_path(0)
    assert issubclass(LeafReadError, ValueError)  # so the verifier's guard maps it


def test_disk_store_without_dataset_paths(c, data, tmp_path):
    leaves, tree_h = InMemoryStore.commit_step(c, _step(c, data))
    disk = DiskStore.write(tmp_path / "s", c, leaves, tree_h)
    with pytest.raises(LookupError):
        disk.dataset_path(0)


def test_disk_write_replaces_and_failed_write_cleans_up(c, data, tmp_path):
    d = tmp_path / "step"
    d.mkdir()
    (d / "stale").write_text("x")
    _, disk = _disk(c, data, d)
    assert not (d / "stale").exists()
    leaves, tree_h = InMemoryStore.commit_step(c, _step(c, data))
    with pytest.raises(TypeError):
        DiskStore.write(d, c, [*leaves[:-1], object()], tree_h)
    assert not d.exists()


def test_disk_handoff_writes_per_step_and_release_deletes(c, data, tmp_path):
    leaves, tree_h = InMemoryStore.commit_step(c, _step(c, data))
    h = DiskHandoff(tmp_path / "run")
    store = h.hold(c, 3, leaves, tree_h)
    assert isinstance(store, DiskStore) and store.directory == tmp_path / "run" / "step_3"
    assert store.root == tree_h.root
    h.release(store)
    assert not (tmp_path / "run" / "step_3").exists()
    keep = DiskHandoff(tmp_path / "run", keep=True)
    s2 = keep.hold(c, 1, leaves, tree_h)
    keep.release(s2)
    assert s2.directory.exists()
    mem = IN_MEMORY.hold(c, 1, leaves, tree_h)
    assert isinstance(mem, InMemoryStore) and mem.root == tree_h.root
    IN_MEMORY.release(mem)


class _RewriteAfterFirstRead(DiskStore):
    """A store whose file for leaf ``index`` changes on disk after its first read."""

    def __init__(self, directory, index, new):
        super().__init__(directory)
        self.index, self.new, self.done = index, new, False

    def leaf(self, index):
        obj = super().leaf(index)
        if index == self.index and not self.done:
            self.done = True
            torch.save(self.new, self.directory / LEAF_FILE.format(index))
        return obj


def _verifier(c, data, w0):
    return Verifier(c, h_D=dataset_tree(c, data).root, n_records=len(data), k=3, n_steps=1,
                    bands=Bands.provisional(), w0=w0, allow_provisional=True,
                    schedule=lambda t: range(4, 8))


def test_verifier_accepts_a_disk_store(c, data, tmp_path):
    _, disk = _disk(c, data, tmp_path / "step")
    v = _verifier(c, data, init_weights(c.widths, seed=0))
    assert v.start_run(data) is None
    assert v.verify_step(1, disk) is None


def test_file_changed_between_check_4_and_check_2_rejects_at_2(c, data, tmp_path):
    _disk(c, data, tmp_path / "step")
    rec = torch.load(tmp_path / "step" / LEAF_FILE.format(0), weights_only=True)
    store = _RewriteAfterFirstRead(tmp_path / "step", 0, _flip(rec))
    v = _verifier(c, data, init_weights(c.widths, seed=0))
    assert v.start_run(data) is None
    rej = v.verify_step(1, store)
    assert (rej.step, rej.check_id, rej.kind) == (1, "2", "failed")
    assert "two versions" in rej.detail


def test_corrupt_leaf_file_rejects_as_malformed(c, data, tmp_path):
    _, disk = _disk(c, data, tmp_path / "step")
    (tmp_path / "step" / LEAF_FILE.format(c.product_index(2))).write_bytes(b"\0" * 10)
    v = _verifier(c, data, init_weights(c.widths, seed=0))
    assert v.start_run(data) is None
    rej = v.verify_step(1, disk)
    assert (rej.step, rej.check_id, rej.kind) == (1, "2", "malformed")
    assert "LeafReadError" in rej.detail
