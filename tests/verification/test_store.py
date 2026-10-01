import gc
import weakref

import pytest
import torch
from torch import nn

from src.verification.computation import TranscriptView
from src.verification.encoding import TAG_PRODUCT, TAG_WEIGHT, encode_tensor_leaf
from src.verification.instances.mlp import MLPComputation, init_weights, synthetic_dataset
from src.verification.merkle import MerkleTree, hash_leaf, verify_path
from src.verification.prover import StepOutput, prove_step
from src.verification.store import (
    InMemoryStore,
    LeafDtypeError,
    LeafShapeError,
    StoreMutationError,
    TranscriptFormatError,
    TranscriptStore,
    commit,
    dataset_tree,
    leaf_hash,
    leaf_hashes,
    leaf_parts,
    perturb_leaf,
    transcript_root,
)

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


def test_wrong_shape_rejected(c, data):
    step = _step(c, data)
    with pytest.raises(LeafShapeError):
        leaf_hash(c, c.product_index(1), step.products[0].t().contiguous())


def test_wrong_dtype_rejected(c, data):
    step = _step(c, data)
    name = c.weight_names[0]
    with pytest.raises(LeafDtypeError):
        leaf_hash(c, c.w_t_index(name), step.w_t[name].to(torch.float16))
    assert issubclass(LeafDtypeError, TranscriptFormatError)
    assert issubclass(LeafShapeError, TranscriptFormatError)
    assert issubclass(StoreMutationError, TranscriptFormatError)


class _HalfWeights(MLPComputation):
    @property
    def weight_dtype(self) -> torch.dtype:
        return torch.float16


def test_same_payload_other_declaration(c, data):
    # One payload, read under a different shape or dtype: it raises against C, and where a
    # declaration admits it, the header makes the hash differ.
    store = _store(c, data)
    i = c.product_index(1)
    p = store.leaf(i)
    with pytest.raises(LeafShapeError):
        leaf_hash(c, i, p.reshape(p.shape[1], p.shape[0]))
    with pytest.raises(LeafDtypeError):
        leaf_hash(c, i, p.view(torch.int32))
    assert hash_leaf(encode_tensor_leaf(TAG_PRODUCT, p.reshape(-1))) != leaf_hash(c, i, p)
    # A half-precision declaration rejects the store's fp32 weight. Its fp16 bytes, read as
    # bf16, carry another dtype code and hash differently.
    half = _HalfWeights(c.widths, n_s=c.n_s, eta=ETA)
    w = c.w_t_index(c.weight_names[0])
    with pytest.raises(LeafDtypeError):
        leaf_hash(half, w, store.leaf(w))
    w16 = store.leaf(w).to(torch.float16)
    assert leaf_hash(half, w, w16) != hash_leaf(encode_tensor_leaf(TAG_WEIGHT,
                                                                   w16.view(torch.bfloat16)))


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
