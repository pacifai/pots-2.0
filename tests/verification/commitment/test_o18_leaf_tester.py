"""O18 (one leaf per attention product): the encoding, the member map and invariant 7.

Written by the tester from ``ATTENTION_LEAF_PLAN_DRAFT.md`` §1.4, §2 and §2.6 alone, against the
interface pins (``leaf_of``, ``product_leaf``, ``product_leaf_index``, ``n_product_leaves``).

1. ``test_no_free_bytes_*``: a grouped leaf is its header plus its members' bytes, nothing else.
2. ``test_map_*``: ``leaf_of`` and ``product_leaf`` round-trip and the counts follow from ``C``.
5. ``test_invariant_7_*``: ``n_leaves`` comes from ``C``; one leaf more or less is rejected.
"""

import struct
from collections import Counter

import pytest
import torch

from setup.data import schedule
from tests.verification.decoder_helpers import (
    SWITCHES,
    make_records,
    randomize_biases,
    switch_id,
    tiny_config,
)
from verification.commitment.encoding import TAG_WEIGHT
from verification.commitment.leaves import dataset_tree, leaf_hash, leaf_parts
from verification.commitment.merkle import MerkleTree, hash_tensor_leaf, verify_path
from verification.computation.instances import DecoderComputation
from verification.computation.interface import snapshot_weights
from verification.prover.step import prove_step
from verification.transcript.store import InMemoryStore
from verification.verifier.bands import Bands
from verification.verifier.context import Rejection
from verification.verifier.driver import Verifier

SEED, K, ETA = 1, 7, 1e-3
TAG_PRODUCT_BYTE = 0x03
DTYPE_CODE = {torch.float32: 0x01, torch.bfloat16: 0x02}
ITEMSIZE = {torch.float32: 4, torch.bfloat16: 2}
ROLES = ("S", "O", "dA", "dV", "dQ", "dK")
STRIDE = {"S": 1, "O": 1, "dA": 2, "dV": 2, "dQ": 2, "dK": 2}
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def _tiny(switch, dtype, n_s=2):
    family, tied = switch
    return DecoderComputation(tiny_config(family=family, tied=tied), n_s=n_s, n=12, eta=ETA,
                              compute_dtype=dtype)


def _step(c, seed=0):
    """An honest step 1 on the tiny decoder: ``(out, D, w0)``."""
    torch.manual_seed(seed)
    model = c.build_model()
    randomize_biases(dict(model.named_parameters()))
    w0 = snapshot_weights(c, model)
    D = make_records((12, 7, 10, 5, 11), seed=1)
    idx = schedule(1, c.n_s, len(D), SEED)
    return prove_step(c, c.build_model(), w0, [D[i] for i in idx]), D, w0


def _raw(t: torch.Tensor) -> bytes:
    t = t.detach().contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return t.numpy().tobytes()


def _expected_header(pl, dtype) -> bytes:
    """Built by hand with ``struct``, not with ``tensor_leaf_header``."""
    ndim = 3 if pl.grouped else 2
    dims = (pl.n, *pl.shape[1:]) if pl.grouped else tuple(pl.shape)
    assert len(dims) == ndim
    return bytes([TAG_PRODUCT_BYTE, DTYPE_CODE[dtype], ndim]) + b"".join(
        struct.pack(">I", d) for d in dims)


def _check_no_free_bytes(c, leaves):
    dtype = c.product_dtype
    n_grouped = 0
    for g in range(c.n_product_leaves):
        pl = c.product_leaf(g)
        i = c.product_leaf_index(g)
        leaf = leaves[i]
        header = _expected_header(pl, dtype)
        if pl.grouped:
            n_grouped += 1
            a, b = pl.shape[1], pl.shape[2]
            assert tuple(leaf.shape) == (pl.n, a, b) and pl.n >= 1
            members = [leaf[j] for j in range(pl.n)]
            body_items = pl.n * a * b
        else:
            members = [leaf]
            body_items = leaf.numel()
        want = header + b"".join(_raw(x) for x in members)
        got = b"".join(bytes(x) for x in leaf_parts(c, i, leaf))
        assert got == want, f"leaf {i} ({pl.name})"
        assert len(got) == len(header) + body_items * ITEMSIZE[dtype], pl.name
        assert len(members) == len(pl.ms)
    return n_grouped


@pytest.mark.parametrize("dtype_name", list(DTYPES))
@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_no_free_bytes_tiny_decoders(switch, dtype_name):
    """Every product leaf is exactly its header plus its members' raw bytes (spec §4.1; plan
    §1.4): no padding between members and no field the prover could vary."""
    c = _tiny(switch, DTYPES[dtype_name])
    out, _, _ = _step(c)
    n_grouped = _check_no_free_bytes(c, out.leaves())
    assert n_grouped == 6 * c.L  # the grouped form was exercised, not just the lone one


def test_grouped_leaf_with_one_member_is_still_3d():
    """§2.2: a batched product with a single member (``n_s = n_h = 1``) is ``[1, a, b]``."""
    config = tiny_config()
    config.num_attention_heads = 1
    config.num_key_value_heads = 1
    config.hidden_size = 8
    c = DecoderComputation(config, n_s=1, n=12, eta=ETA)
    assert c.n_s * c.n_h == 1
    out, _, _ = _step(c)
    n_grouped = _check_no_free_bytes(c, out.leaves())
    assert n_grouped == 6 * c.L
    assert all(len(c.product_leaf(g).shape) == 3
               for g in range(c.n_product_leaves) if c.product_leaf(g).grouped)


def _smollm2_c():
    from setup.config import load_config
    from setup.model import load_model_config

    smol = load_config({})
    try:
        config = load_model_config(smol.model, smol.model_revision)
    except Exception as e:  # not in the local cache
        pytest.skip(f"{smol.model}@{smol.model_revision[:8]} not cached: {e}")
    return DecoderComputation(config, n_s=4, n=128, eta=ETA, source=(smol.model,
                                                                  smol.model_revision))


@pytest.mark.slow
def test_no_free_bytes_smollm2_real_step():
    """The same, on one real SmolLM2 step at test scale from W_0 (skips without the cache)."""
    import os

    from setup.config import load_config, setup_determinism
    from setup.data import load_dataset_records

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cfg = load_config()
    if not (cfg.data_dir / "D.bin").exists():
        pytest.skip(f"no D.bin under {cfg.data_dir}")
    setup_determinism(cfg)
    try:
        c = DecoderComputation.from_config(cfg)
        model = c.build_model()
    except Exception as e:
        pytest.skip(f"SmolLM2 weights not cached: {e}")
    D = load_dataset_records(cfg.data_dir / "D.bin", c.n)
    records = [D[i] for i in schedule(1, c.n_s, len(D), SEED)]
    w0 = {n: model.get_parameter(n).detach().clone() for n in c.weight_names}
    out = prove_step(c, model, w0, records)
    assert _check_no_free_bytes(c, out.leaves()) == 6 * c.L


# ---- 2. the map --------------------------------------------------------------------------


def _check_map(c):
    M, K_ = c.M, c.n_product_leaves
    leaves = [c.product_leaf(g) for g in range(K_)]
    for g, pl in enumerate(leaves):
        assert pl.g == g
    seen = Counter()
    for pl in leaves:
        seen.update(pl.ms)
    assert sorted(seen) == list(range(1, M + 1)) and set(seen.values()) == {1}
    m0s = [pl.m0 for pl in leaves]
    assert all(a < b for a, b in zip(m0s, m0s[1:]))
    for m in range(1, M + 1):
        g, i = c.leaf_of(m)
        pl = c.product_leaf(g)
        assert pl.ms[i if i is not None else 0] == m
        assert (i is None) == (not pl.grouped)
    for bad in (0, M + 1):
        with pytest.raises(IndexError):
            c.leaf_of(bad)
    for bad in (-1, K_):
        with pytest.raises(IndexError):
            c.product_leaf(bad)
    # the grouped leaves: one per layer and attention role, in leaf order within a layer
    grouped = [pl for pl in leaves if pl.grouped]
    names = [pl.name for pl in grouped]
    assert len(names) == len(set(names)) == 6 * c.L
    assert set(names) == {f"L{ell}.{r}" for ell in range(1, c.L + 1) for r in ROLES}
    for pl in grouped:
        role = pl.name.split(".")[-1]
        assert pl.n == c.n_s * c.n_h and pl.stride == STRIDE[role], pl.name
        assert pl.shape[0] == pl.n and len(pl.shape) == 3
        assert pl.ms == range(pl.m0, pl.m0 + pl.stride * pl.n, pl.stride)
    # every other product is a lone leaf of its own
    lone = [pl for pl in leaves if not pl.grouped]
    assert all(pl.n == 1 for pl in lone)
    assert len(lone) == M - 6 * c.L * c.n_s * c.n_h
    assert K_ == len(lone) + 6 * c.L
    assert K_ == M - sum(pl.n - 1 for pl in leaves)
    assert c.n_leaves == c.n_s + 2 * c.n_w + K_
    for g in (0, K_ - 1):
        assert c.product_leaf_index(g) == c.n_s + c.n_w + g
    assert c.n_product_leaves == 27 * c.L + 3 * c.head_blocks


@pytest.mark.parametrize("n_s", [2, 4])
@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_map_tiny_decoders(switch, n_s):
    """``leaf_of`` and ``product_leaf`` invert each other and the counts of plan §2.3 hold."""
    _check_map(_tiny(switch, torch.float32, n_s))


def test_map_smollm2_config():
    """SmolLM2 at test scale: 813 product leaves and 1,361 leaves (plan §2.4)."""
    c = _smollm2_c()
    _check_map(c)
    assert (c.n_product_leaves, c.n_leaves) == (813, 1361)


@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_map_matches_a_real_prove_step(switch):
    """``n_leaves`` from ``C`` is what ``prove_step`` emits, and each product leaf has the
    shape ``C`` declares."""
    c = _tiny(switch, torch.float32)
    out, _, _ = _step(c)
    leaves = out.leaves()
    assert len(leaves) == c.n_leaves
    assert len(out.products) == c.n_product_leaves
    for g in range(c.n_product_leaves):
        leaf = leaves[c.product_leaf_index(g)]
        assert tuple(leaf.shape) == c.product_leaf(g).shape, c.product_leaf(g).name
        assert leaf.dtype == c.product_dtype
    # the weights keep their places around the product block
    n_pre = c.n_s + c.n_w
    assert [tuple(x.shape) for x in leaves[c.n_s:n_pre]] == \
        [tuple(c.weight_shapes[n]) for n in c.weight_names]
    assert c.w_next_index(c.weight_names[0]) == n_pre + c.n_product_leaves


# ---- 5. invariant 7 ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def committed():
    c = _tiny(SWITCHES[0], torch.float32)
    out, D, w0 = _step(c)
    leaves, tree = InMemoryStore.commit_step(c, out)
    idx = schedule(1, c.n_s, len(D), SEED)
    dtree = dataset_tree(c, D)
    return c, leaves, tree, [dtree.path(i) for i in idx], dtree, D, w0


def test_invariant_7_every_leaf_path_verifies_against_n_leaves_from_c(committed):
    c, leaves, tree, *_ = committed
    assert tree.n_leaves == c.n_leaves
    for i in range(c.n_leaves):
        h = leaf_hash(c, i, leaves[i])
        assert h == tree.leaf(i)
        assert verify_path(h, i, c.n_leaves, tree.path(i), tree.root), i
    # a wrong n_leaves does not verify the last leaf's path (the count is part of the proof)
    i = c.n_leaves - 1
    assert not verify_path(tree.leaf(i), i, c.n_leaves + 1, tree.path(i), tree.root)


def _verify(c, leaves, tree, paths, dtree, D, w0):
    v = Verifier(c, h_D=dtree.root, n_records=len(D), k=K, n_steps=1, bands=Bands.provisional(),
                 w0=w0, allow_provisional=True, seed=SEED)
    assert v.start_run(D) is None
    store = InMemoryStore.hold(c, list(leaves), tree, paths)
    return v.verify_step(1, store)


def test_invariant_7_honest_control_is_accepted(committed):
    c, leaves, tree, paths, dtree, D, w0 = committed
    assert _verify(c, leaves, tree, paths, dtree, D, w0) is None


def test_invariant_7_one_leaf_more_is_rejected_at_check_2(committed):
    c, leaves, tree, paths, dtree, D, w0 = committed
    extra = torch.zeros(3, dtype=torch.float32)
    longer = [*leaves, extra]
    hashes = [tree.leaf(i) for i in range(tree.n_leaves)]
    hashes.append(hash_tensor_leaf(TAG_WEIGHT, extra))
    big = MerkleTree(hashes)
    assert big.n_leaves == c.n_leaves + 1 and big.root != tree.root
    rej = _verify(c, longer, big, paths, dtree, D, w0)
    assert isinstance(rej, Rejection) and (rej.step, rej.check_id) == (1, "2"), rej


def test_invariant_7_one_leaf_less_is_rejected_at_check_2(committed):
    c, leaves, tree, paths, dtree, D, w0 = committed
    shorter = list(leaves[:-1])  # the last W_{t+1} leaf dropped
    hashes = [tree.leaf(i) for i in range(tree.n_leaves - 1)]
    small = MerkleTree(hashes)
    assert small.n_leaves == c.n_leaves - 1
    rej = _verify(c, shorter, small, paths, dtree, D, w0)
    assert isinstance(rej, Rejection) and (rej.step, rej.check_id) == (1, "2"), rej
