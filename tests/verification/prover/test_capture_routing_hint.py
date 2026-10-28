"""FS32d tests: the hint on every product, a wrong hint, the weights, the cheat hook and the
cache release (F22). Everything runs on the Mac; "fake devices" are ``cuda:k`` names that CPU
tensors are judged against, never moved to."""

import contextlib
import os
import random

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef

from verification.commitment.leaves import dataset_tree
from verification.computation.instances import DecoderComputation
from verification.computation.matmul_ops import param_storage_map
from verification.prover import step as step_mod
from verification.prover.capture import MatmulCapture
from verification.prover.step import commit, plain_step, prove_step
from verification.runs import loop as loop_mod
from verification.transcript.store import InMemoryStore
from verification.verifier.bands import Bands
from verification.verifier.driver import Verifier

from setup.data import schedule
from tests.verification.decoder_helpers import (
    SWITCHES, make_records, randomize_biases, switch_id, tiny_config)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

THREE = ("cpu", "cpu", "cpu")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
CPU = torch.device("cpu")


def tiny(switch, *, B=1, dtype="fp32", devices=None, layers=2):
    family, tied = switch
    config = tiny_config(family=family, tied=tied)
    config.num_hidden_layers = layers
    if getattr(config, "layer_types", None):
        config.layer_types = [config.layer_types[0]] * layers
    return DecoderComputation(config, n_s=2, n=12, eta=1e-3, head_blocks=B,
                              compute_dtype=DTYPES[dtype], verifier_devices=devices)


def start(c):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        w0 = {n: p.detach().clone() for n, p in c.build_model().named_parameters()}
    randomize_biases(w0)
    return w0


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())


def assert_same_step(a, b):
    assert a.loss == b.loss and len(a.leaves()) == len(b.leaves())
    for x, y in zip(a.leaves()[len(a.records):], b.leaves()[len(b.records):]):
        assert same(x, y)


def captured(c, w0, records, *, rule2=True, hint=None):
    """Run the capture alone: ``(cap, labeled leaves)``."""
    model = c.build_model()
    model.zero_grad(set_to_none=True)
    cap = MatmulCapture(param_storage_map(model), placement=c.placement)
    if not rule2:
        class Off(dict):
            def get(self, *a):
                return None

            def setdefault(self, *a):
                return None
        cap._operand_device = Off()
    if hint is not None:
        orig = cap._hint
        cap._hint = lambda rec: hint(cap, rec, orig)

    @contextlib.contextmanager
    def stage(p):
        with cap.phase(p):
            yield

    with cap:
        c.forward_backward(model, records, stage)
    return cap, c.label(cap, model)


def leaf_hints(cap, labeled):
    """Per product leaf ``g`` (0-based): the hint of the record whose storage the leaf shares,
    or None for a leaf that is no record's output."""
    by = {StorageWeakRef(r.out.untyped_storage()): r.hint
          for r in cap.records + cap.glue_outer if r.hint is not None}
    return {g: by.get(StorageWeakRef(p.untyped_storage())) for g, p in enumerate(labeled)}


def role(c, g):
    """The class of the leaf's first product (a group's members share one)."""
    return c.product_class(c.products[c.product_leaf(g).m0 - 1])


def fake(n):
    return tuple(f"cuda:{k}" for k in range(n))


PLACES = [(1, 2), (2, 2), (3, 3), (4, 2), (5, 1), (5, 2), (4, 3)]  # (devices, layers)


# ---- the hint, on every product -------------------------------------------------------------


@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
@pytest.mark.parametrize("n_dev, layers", PLACES)
def test_every_product_but_dK_is_hinted_to_its_owner(n_dev, layers, switch, B, dtype):
    c = tiny(switch, B=B, dtype=dtype, devices=fake(n_dev), layers=layers)
    cap, labeled = captured(c, start(c), make_records((12, 7)))
    hints = leaf_hints(cap, labeled)
    for g, h in hints.items():
        if role(c, g) == "dK":
            assert h is None, g  # a fresh contiguous transpose: no record shares its storage
        else:
            assert h == c.placement.product_leaf_device(g), (g, c.product_leaf(g).name)
    assert cap.count_hint_misses(labeled, c.placement.product_leaf_device) == 0
    assert cap.summary()["hint_misses"] == 0
    if n_dev > 1:
        assert len({h for h in hints.values() if h}) > 1 or layers == 1


@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_more_devices_than_layers_leave_a_device_unused_and_still_hit(switch):
    c = tiny(switch, devices=fake(5), layers=2)
    used = {c.placement.product_leaf_device(g) for g in range(c.n_product_leaves)}
    assert len(used) < 5
    cap, labeled = captured(c, start(c), make_records((12, 7)))
    assert cap.count_hint_misses(labeled, c.placement.product_leaf_device) == 0


@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_cpu_count_skips_exactly_the_dK_leaves(switch):
    """On the CPU a leaf no record's output (the dK transpose) is not judged; nothing else is:
    a hint that disagrees with any other leaf is counted."""
    c = tiny(switch, B=2, devices=fake(3), layers=3)
    cap, labeled = captured(c, start(c), make_records((12, 7)))
    hints = leaf_hints(cap, labeled)
    skipped = {g for g, h in hints.items() if h is None}
    assert skipped == {g for g in hints if role(c, g) == "dK"} and skipped
    # Wrong for everyone: every judged leaf counts, the skipped ones do not.
    miss = cap.count_hint_misses(labeled, lambda g: torch.device("cuda:9"))
    assert miss == c.n_product_leaves - len(skipped)


def test_second_rule_decides_the_weight_gradient_and_removing_it_misses():
    """A weight gradient (G_*, X^T . dY) has no weight operand and runs before its layer's
    weight-operand backward ops, so only the 'operand an earlier op was routed with' rule can
    place it. Without that rule misses appear, on these products."""
    seen = {}
    for switch in SWITCHES:
        for B in (1, 2):
            c = tiny(switch, B=B, devices=fake(3), layers=3)
            w0, records = start(c), make_records((12, 7))
            cap, labeled = captured(c, w0, records, rule2=False)
            hints = leaf_hints(cap, labeled)
            missed = [g for g, h in hints.items()
                      if h is not None and h != c.placement.product_leaf_device(g)]
            assert missed, (switch, B)
            for g in missed:
                seen.setdefault(role(c, g), 0)
                seen[role(c, g)] += 1
    print("rule 2 decides:", seen)
    assert seen


# ---- a wrong hint costs a copy, never a bit -------------------------------------------------


def _shifted(devs):
    def hint(self, rec, orig):
        right = orig(rec)
        return devs[(devs.index(right) + 1) % len(devs)]
    return hint


def _one_device(dev):
    return lambda self, rec, orig: torch.device(dev)


def _random(devs, seed):
    rng = random.Random(seed)
    return lambda self, rec, orig: rng.choice(devs)


def _expected_misses(c, cap, labeled):
    """Per judged leaf (a record's output) whether its hint differs from the owner."""
    return sum(h != c.placement.product_leaf_device(g)
               for g, h in leaf_hints(cap, labeled).items() if h is not None)


WRONG = {
    "one": lambda devs: _one_device(devs[0]),
    "shifted": _shifted,
    "random": lambda devs: _random(devs, 3),
}


@pytest.mark.parametrize("kind", list(WRONG))
@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_a_wrong_hint_counts_and_changes_no_bit_on_the_capture(kind, switch):
    devs = [torch.device(d) for d in fake(3)]
    c = tiny(switch, B=2, devices=fake(3), layers=3)
    cap, labeled = captured(c, start(c), make_records((12, 7)), hint=WRONG[kind](devs))
    want = _expected_misses(c, cap, labeled)
    assert want > 0
    assert cap.count_hint_misses(labeled, c.placement.product_leaf_device) == want
    ref, ref_labeled = captured(c, start(c), make_records((12, 7)))
    assert all(same(x, y) for x, y in zip(labeled, ref_labeled))


@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("kind", ["one", "shifted", "random", "mixed"])
@pytest.mark.parametrize("switch", [("llama", True), ("qwen2", False)], ids=switch_id)
def test_a_wrong_hint_in_a_step_changes_no_leaf_root_loss_or_weight(
        monkeypatch, kind, switch, dtype):
    """prove_step on ``cpu,cpu,cpu``: the owners are CPU, so every ``cuda:k`` hint is wrong and a
    CPU hint is right. The count equals the number of judged leaves hinted elsewhere, and each
    leaf ends on its product device."""
    off, on = tiny(switch, B=2, dtype=dtype), tiny(switch, B=2, dtype=dtype, devices=THREE)
    w0, records = start(off), make_records((12, 7))
    a = prove_step(off, off.build_model(), w0, records)
    rng = random.Random(5)
    given = []

    def hint(self, rec):
        if kind == "mixed":
            h = rng.choice([CPU, torch.device("cuda:1"), torch.device("cuda:2")])
        elif kind == "shifted":
            h = torch.device(f"cuda:{len(given) % 3}")
        elif kind == "one":
            h = torch.device("cuda:0")
        else:
            h = torch.device(f"cuda:{rng.randrange(3)}")
        given.append((StorageWeakRef(rec.out.untyped_storage()), h))
        return h

    monkeypatch.setattr(MatmulCapture, "_hint", hint)
    b = prove_step(on, on.build_model(), w0, records)
    hinted = dict(given)
    want = sum(hinted[StorageWeakRef(p.untyped_storage())] != CPU for p in b.products
               if StorageWeakRef(p.untyped_storage()) in hinted)
    assert b.hint_misses == want == b.capture["hint_misses"]
    assert want > 0
    assert_same_step(a, b)
    assert commit(off, a).root == commit(on, b).root
    assert all(p.device == on.placement.product_leaf_device(g) for g, p in enumerate(b.products))
    w_next, loss = plain_step(on, on.build_model(), w0, records)
    assert loss == b.loss and all(same(w_next[n], b.w_next[n]) for n in on.weight_names)
    b.assert_unmodified()


# ---- the weights ----------------------------------------------------------------------------


class _Meta:
    """A placement stand-in that puts every weight on ``meta``, a device distinct from the
    host that tensors can still be copied to without CUDA."""

    def __init__(self, inner):
        self.inner = inner

    def weight_device(self, name):
        return torch.device("meta")

    def __getattr__(self, name):
        return getattr(self.inner, name)


class _View:
    """A computation seen through another placement."""

    def __init__(self, c, placement):
        self._c, self.placement = c, placement

    def __getattr__(self, name):
        return getattr(self._c, name)


def test_w_t_is_shared_on_its_device_and_each_condition_unshares_it():
    c = tiny(("llama", False), devices=THREE)
    model = c.build_model()
    name = c.weight_names[0]
    p = model.get_parameter(name)
    storages = {q.untyped_storage().data_ptr() for q in model.parameters()}
    w = p.detach().clone()
    share = step_mod._shareable
    assert share(w, p, storages, CPU) and share(w, p, storages)  # default device: HOST
    assert not share(w, p, storages, torch.device("cuda:1"))  # on another device: copied
    assert not share(w.requires_grad_(), p, storages, CPU)
    w = p.detach().clone()
    assert not share(w.double(), p, storages, CPU)  # dtype
    assert not share(w.flatten()[:-1].view(-1), p, storages, CPU)  # shape / view
    assert not share(p.detach().clone().t().contiguous().t(), p, storages, CPU) or p.dim() < 2
    assert not share(torch.cat([w.flatten(), w.flatten()])[:w.numel()].view_as(w), p,
                     storages, CPU)  # a view of a bigger storage
    assert not share(p.detach(), p, storages, CPU)  # shares a model parameter's storage


def test_w_t_copies_land_on_the_weight_device_and_shared_ones_are_the_callers():
    c = tiny(("llama", True), devices=THREE)
    w0 = start(c)
    model = c.build_model()
    step_mod.load_weights(c, model, w0)
    got = step_mod._w_t_leaves(c, model, w0)
    assert all(got[n] is w0[n] for n in c.weight_names)
    # On a device the caller's tensor isn't on, the leaf is a copy there (meta stands in).
    view = _View(c, _Meta(c.placement))
    got = step_mod._w_t_leaves(view, model, w0)
    assert all(got[n].device.type == "meta" and got[n] is not w0[n] for n in c.weight_names)
    # A caller's view is never shared; the copy is on the weight's device and equals the model.
    views = {n: torch.cat([t.flatten(), t.flatten()])[:t.numel()].view_as(t)
             for n, t in w0.items()}
    got = step_mod._w_t_leaves(c, model, views)
    assert all(got[n] is not views[n] and same(got[n], w0[n]) for n in c.weight_names)


@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_snapshot_places_each_weight_and_the_tie_stays_one_tensor(switch):
    c = tiny(switch, devices=THREE, layers=3)
    model = c.build_model()
    snap = step_mod.snapshot_weights(_View(c, _Meta(c.placement)), model)
    assert list(snap) == list(c.weight_names)
    assert all(t.device.type == "meta" for t in snap.values())
    plain = step_mod.snapshot_weights(c, model)
    assert all(t.device == CPU and t is not model.get_parameter(n)
               and same(t, model.get_parameter(n)) for n, t in plain.items())
    head = "lm_head.weight"
    if switch[1]:  # tied: the one tensor is both the embedding and the head
        assert head not in snap and head not in c.weight_names
        assert model.lm_head.weight is model.get_parameter(c.weight_names[0])
        emb = c.weight_names[0]
        assert c.placement.unit_device(0) == c.placement.unit_device(c.L + 1) \
            == c.placement.weight_device(emb)
    else:
        assert head in snap and snap[head] is not snap[c.weight_names[0]]


@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_weight_leaves_of_a_step_match_the_placement(switch):
    c = tiny(switch, devices=THREE, layers=3)
    out = prove_step(c, c.build_model(), start(c), make_records((12, 7)))
    for n in c.weight_names:
        assert out.w_t[n].device == out.w_next[n].device == c.placement.weight_device(n)
    spliced = out.with_w_next({n: t.clone() for n, t in out.w_next.items()})
    assert all(spliced.w_next[n].device == c.placement.weight_device(n) for n in c.weight_names)
    assert all(spliced.w_next[n].is_contiguous() for n in c.weight_names)


# ---- the cheat hook -------------------------------------------------------------------------


@pytest.mark.parametrize("devices", [None, THREE], ids=["unsharded", "cpu3"])
@pytest.mark.parametrize("name", ["L1.G_q", "L2.Y_o", "L1.G_down"])
def test_a_perturbed_product_lands_at_m_and_the_verifier_rejects_it(devices, name):
    c = tiny(("qwen2", True), devices=devices)
    D = make_records((12, 10, 9, 8), seed=1)
    tree = dataset_tree(c, D)
    w0 = start(c)
    idx = schedule(1, c.n_s, len(D), 1)
    batch = [D[i] for i in idx]
    clean = prove_step(c, c.build_model(), w0, batch)
    m = c.m_of(name)
    out = prove_step(c, c.build_model(), w0, batch, perturb={m: lambda p: p * 1.5 + 1})
    g = c.leaf_of(m)[0]
    assert out.products[g].device == c.placement.product_leaf_device(g)
    assert all(same(x, y) for i, (x, y) in enumerate(zip(clean.products, out.products))
               if i != g)
    assert not same(clean.products[g], out.products[g])

    def verify(step):
        store = InMemoryStore.from_step(c, step, copy=True,
                                        dataset_paths=[tree.path(i) for i in idx])
        v = Verifier(c, h_D=tree.root, n_records=len(D), k=7, n_steps=1,
                     bands=Bands.provisional(), allow_provisional=True, w0=w0, seed=1)
        assert v.start_run(D) is None
        return v.verify_step(1, store)

    assert verify(clean) is None
    rej = verify(out)
    assert rej is not None and rej.step == 1 and rej.check_id in {"5", "6a"}
    assert f"P_{m}" in rej.detail or rej.check_id == "6a"


def test_the_cheat_is_caught_at_the_same_check_sharded_and_unsharded():
    got = {}
    for devices in (None, THREE):
        c = tiny(("llama", False), devices=devices)
        D = make_records((12, 10, 9, 8), seed=1)
        tree = dataset_tree(c, D)
        w0 = start(c)
        idx = schedule(1, c.n_s, len(D), 1)
        m = c.m_of("L2.Y_down")
        out = prove_step(c, c.build_model(), w0, [D[i] for i in idx],
                         perturb={m: lambda p: p * 1.01})
        store = InMemoryStore.from_step(c, out, copy=True,
                                        dataset_paths=[tree.path(i) for i in idx])
        v = Verifier(c, h_D=tree.root, n_records=len(D), k=7, n_steps=1,
                     bands=Bands.provisional(), allow_provisional=True, w0=w0, seed=1)
        v.start_run(D)
        rej = v.verify_step(1, store)
        got[devices] = (rej.step, rej.check_id, rej.kind, rej.detail)
    assert got[None] == got[THREE] and got[None][1] == "5"


# ---- release_device_cache -------------------------------------------------------------------


def test_release_device_cache_covers_every_run_device(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    sections = []

    @contextlib.contextmanager
    def section(name):
        sections.append(name)
        yield

    r = loop_mod.release_device_cache
    r("cpu", section, devices=THREE)
    r("cpu", section, devices=None)
    r(torch.device("mps"), section, devices=(CPU, CPU))
    assert calls == [] and sections == []  # no CUDA anywhere: nothing, not even a section
    r("cpu", section, devices=("cpu", "cuda:1", "cpu"))  # a verifier device alone is enough
    assert calls == [1] and sections == [loop_mod.HANDOFF_SECTION]
    r("cuda:0", section, devices=("cuda:1", "cuda:2"))  # the prover and the verifiers: still once
    assert calls == [1, 1] and len(sections) == 2
    r(torch.device("cuda:0"), section)  # old call shape
    assert calls == [1, 1, 1]


def test_run_loop_passes_the_placement_devices_to_the_release(monkeypatch):
    """The loop hands the computation's placement devices to the release, once per step
    (unsharded: ``None``, so nothing is released off CUDA)."""
    import tests.verification.runs.test_loop as tl
    c0 = tl.MLPComputation((16, 32, 32, 8), n_s=4, eta=tl.ETA)
    D = tl.synthetic_dataset(c0.widths, tl.N_RECORDS, seed=0)
    w0 = tl.init_weights(c0.widths, seed=0)
    final = tl._plain_final(c0, D, w0, tl.SEED)
    seen, empties = [], []
    orig = loop_mod.release_device_cache

    def spy(device, section=None, *, devices=None):
        seen.append(devices)
        return orig(device, section, devices=devices)

    monkeypatch.setattr(loop_mod, "release_device_cache", spy)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empties.append(1))
    monkeypatch.delenv("VERIF_RELEASE_CACHE", raising=False)
    res, _ = tl._run(c0, D, w0, final)
    assert res.verdict.accepted and seen == [c0.placement.devices] * tl.T and empties == []


# ---- unsharded is inert ---------------------------------------------------------------------


@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_unsharded_capture_computes_no_hint(switch, monkeypatch):
    """With the setting unset the capture ignores its placement: no hint is computed, none is
    recorded, no count is kept, and the summary has no ``hint_misses`` key."""
    def boom(self, rec):
        raise AssertionError("an unsharded capture asked for a hint")

    monkeypatch.setattr(MatmulCapture, "_hint", boom)
    c = tiny(switch)
    assert not c.placement.sharded
    cap, labeled = captured(c, start(c), make_records((12, 7)))
    assert cap.placement is None and cap.n_hint_misses is None and not cap._operand_device
    assert all(r.hint is None for r in cap.records + cap.glue_outer)
    assert "hint_misses" not in cap.summary()
    out = prove_step(c, c.build_model(), start(c), make_records((12, 7)))
    assert out.hint_misses == 0 and "hint_misses" not in out.capture


# ---- the real model (slow) ------------------------------------------------------------------


@pytest.mark.slow
def test_smollm2_cpu_cpu_cpu_equals_unsharded_and_never_misses(monkeypatch):
    from setup.config import load_config, setup_determinism
    from setup.data import load_dataset_records

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("VERIF_VERIFIER_DEVICES", raising=False)
    cfg = load_config()
    if not (cfg.data_dir / "D.bin").exists():
        pytest.skip(f"no D.bin under {cfg.data_dir}; set VERIF_OUTPUT_DIR")
    setup_determinism(cfg)
    off = DecoderComputation.from_config(cfg)
    D = load_dataset_records(cfg.data_dir / "D.bin", off.n)
    records = [D[i] for i in schedule(1, off.n_s, len(D), 1)]
    model = off.build_model()
    w0 = {n: model.get_parameter(n).detach().clone() for n in off.weight_names}
    ref = prove_step(off, model, w0, records)
    ref_root, ref_w = commit(off, ref).root, ref.w_next
    monkeypatch.setenv("VERIF_VERIFIER_DEVICES", "cpu,cpu,cpu")
    on = DecoderComputation.from_config(load_config())
    assert on.placement.sharded
    out = prove_step(on, on.build_model(), w0, records)
    print(f"\nSMOLLM2 roots {ref_root.hex()} {commit(on, out).root.hex()} "
          f"loss {ref.loss!r} {out.loss!r} misses {out.hint_misses}")
    assert out.hint_misses == 0 and out.capture["hint_misses"] == 0
    assert out.loss == ref.loss and commit(on, out).root == ref_root
    assert all(same(out.w_next[n], ref_w[n]) for n in on.weight_names)
    assert_same_step(ref, out)


@pytest.mark.slow
def test_smollm2_hint_under_a_fake_placement_matches_every_owner(monkeypatch):
    """Capture and labeling only, under ``cuda:0,cuda:1,cuda:2`` (nothing is moved to CUDA):
    the hint of every product but the dK transposes is its owner, so the count is 0."""
    from setup.config import load_config, setup_determinism
    from setup.data import load_dataset_records

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("VERIF_VERIFIER_DEVICES", raising=False)
    cfg = load_config()
    if not (cfg.data_dir / "D.bin").exists():
        pytest.skip(f"no D.bin under {cfg.data_dir}; set VERIF_OUTPUT_DIR")
    setup_determinism(cfg)
    c = DecoderComputation.from_config(cfg)
    # The config insists a CUDA setting go with a CUDA VERIF_DEVICE; set the devices directly.
    c.verifier_devices = tuple(torch.device(d) for d in fake(3))
    c.__dict__.pop("placement", None)
    assert c.placement.sharded and len({c.placement.product_leaf_device(g)
                                        for g in range(c.n_product_leaves)}) == 3
    D = load_dataset_records(cfg.data_dir / "D.bin", c.n)
    records = [D[i] for i in schedule(1, c.n_s, len(D), 1)]
    cap, labeled = captured(c, None, records)
    hints = leaf_hints(cap, labeled)
    n_dk = 0
    for g, h in hints.items():
        if role(c, g) == "dK":
            n_dk += 1
            assert h is None, g
        else:
            assert h == c.placement.product_leaf_device(g), (g, c.product_leaf(g).name)
    assert n_dk > 0
    assert cap.count_hint_misses(labeled, c.placement.product_leaf_device) == 0
