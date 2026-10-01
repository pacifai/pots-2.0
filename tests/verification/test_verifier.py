"""A5: the per-step checks and the verifier on the MLP instance (η = 1e-3).

Every fault is rejected at exactly its declared ``(step, check)`` (S6b, S6d); a later or
earlier check reporting it is a failure.
"""

import ast
import dataclasses
import math
from pathlib import Path

import pytest
import torch

from src.verification.challenges import challenge_matrix
from src.verification.checks import (
    DEFAULT_ORDER,
    Bands,
    Rejection,
    StepContext,
    check_2_commitment,
    check_5_matmuls,
    product_class,
)
from src.verification.config import TAU_W0, Z
from src.verification.data import schedule
from src.verification.instances.mlp import (
    MLPComputation,
    MLPReplay,
    init_weights,
    make_record,
    synthetic_dataset,
)
from src.verification.prover import plain_step, prove_step
from src.verification.store import InMemoryStore, TranscriptStore, dataset_tree, perturb_leaf
from src.verification.verifier import Verifier

ETA = 1e-3
K = 7
N_RECORDS = 40
SRC = Path(__file__).resolve().parents[2] / "src" / "verification"


@pytest.fixture(scope="module")
def c():
    return MLPComputation((16, 32, 32, 8), n_s=4, eta=ETA)


@pytest.fixture(scope="module")
def D(c):
    return synthetic_dataset(c.widths, N_RECORDS, seed=0)


@pytest.fixture(scope="module")
def tree(c, D):
    return dataset_tree(c, D)


@pytest.fixture(scope="module")
def w0(c):
    return init_weights(c.widths, seed=0)


def _verifier(c, D, tree, w0, *, n_steps=None, **kw) -> Verifier:
    v = Verifier(c, h_D=tree.root, n_records=len(D), k=K, bands=kw.pop("bands",
                 Bands.provisional()), w0=w0, n_steps=n_steps, **kw)
    assert v.start_run(D) is None
    return v


def _batch(c, D, t):
    return [D[i] for i in schedule(t, c.n_s, len(D))]


def _store(c, D, tree, w_t, t, *, records=None, path_indices=None, **kw):
    idx = schedule(t, c.n_s, len(D))
    out = prove_step(c, c.build_model(), w_t, records or _batch(c, D, t), **kw)
    paths = [tree.path(i) for i in (path_indices or idx)]
    return InMemoryStore.from_step(c, out, dataset_paths=paths), out


class Wrapped(TranscriptStore):
    """A store that serves another store's data with selected overrides (prover misbehaviour)."""

    def __init__(self, inner, *, leaves=None, root=None, paths=None, no_paths=False):
        self.inner, self.leaves, self._root = inner, leaves or {}, root
        self.paths, self.no_paths = paths or {}, no_paths

    def leaf(self, index):
        return self.leaves[index] if index in self.leaves else self.inner.leaf(index)

    @property
    def root(self):
        return self._root if self._root is not None else self.inner.root

    def path(self, index):
        return self.inner.path(index)

    def dataset_path(self, i):
        if self.no_paths:
            raise LookupError("no dataset paths")
        return self.paths[i] if i in self.paths else self.inner.dataset_path(i)


def _expect(rej, step, check_id):
    assert isinstance(rej, Rejection), rej
    assert (rej.step, rej.check_id) == (step, check_id), rej


# ---- honest runs ------------------------------------------------------------------------


def test_honest_multistep_run_accepted(c, D, tree, w0):
    T = 4
    v = _verifier(c, D, tree, w0, n_steps=T)
    w = w0
    for t in range(1, T + 1):
        store, out = _store(c, D, tree, w, t)
        assert v.verify_step(t, store) is None
        w = out.w_next
    verdict = v.end_run(w)
    assert verdict.accepted and verdict.steps_verified == T
    for t in range(1, T + 1):
        assert list(v.timings[t]) == list(DEFAULT_ORDER)
        assert all(s >= 0 for s in v.timings[t].values())
        assert len(v.stats[t].products) == c.M
        assert [s.weight for s in v.stats[t].tensors] == list(c.linear_weights)
    assert set(v.run_timings) == {"0", "1", "8"}


def test_honest_residuals_well_under_tau(c, D, tree, w0):
    """Reported numbers: normalized residuals ≈ 1 by construction (P3.a), far below τ = 8."""
    v = _verifier(c, D, tree, w0)
    w, worst = w0, []
    for t in range(1, 4):
        store, out = _store(c, D, tree, w, t)
        assert v.verify_step(t, store) is None
        w = out.w_next
        s = v.stats[t]
        assert all(len(p.normalized) == K for p in s.products)
        worst.append(s.max_normalized())
        rms = math.sqrt(sum(x * x for p in s.products for x in p.normalized)
                        / (K * len(s.products)))
        print(f"step {t}: max normalized {s.max_normalized():.3f}, rms {rms:.3f}, "
              f"max κ {s.max_kappa():.2f}, max ρ (6a) {max(x.rho_max for x in s.tensors):.3g}")
        assert 0 < rms < 2
        assert 1 <= s.max_kappa() < 100
    assert max(worst) < Z / 2  # C1's concentration guard, τ/2, holds already


def test_calibration_mode_records_without_judging(c, D, tree, w0):
    """P10b: checks 5 and 6 record, never reject; exact checks still run live."""
    tight = Bands(tau=1e-9, kappa_max=1.0, tau_w=TAU_W0)
    v = _verifier(c, D, tree, w0, bands=tight, calibrate=True)
    store, _ = _store(c, D, tree, w0, 1)
    assert v.verify_step(1, store) is None
    stats = v.stats[1]
    assert len(stats.products) == c.M
    assert set(stats.by_class()) == {product_class(c, p) for p in c.products}
    # The same bands judge: rejected at 5.
    judge = _verifier(c, D, tree, w0, bands=tight)
    _expect(judge.verify_step(1, _store(c, D, tree, w0, 1)[0]), 1, "5")
    # Calibration mode still runs check 4.
    v2 = _verifier(c, D, tree, w0, calibrate=True)
    bad, _ = _store(c, D, tree, w0, 1, records=_batch(c, D, 2))
    _expect(v2.verify_step(1, bad), 1, "4")


# ---- faults, each at its declared check -------------------------------------------------


@pytest.mark.parametrize("name", ["Y_1", "Y_2", "dX_3", "dX_2"])
def test_perturbed_product_rejected_at_5(c, D, tree, w0, name):
    m = c.m_of(name)
    store, _ = _store(c, D, tree, w0, 1, perturb={m: lambda p: p * 1.01})
    v = _verifier(c, D, tree, w0)
    rej = v.verify_step(1, store)
    _expect(rej, 1, "5")
    assert f"P_{m} ({name})" in rej.detail


def test_perturbed_weight_gradient_rejected_at_6a(c, D, tree, w0):
    """A committed ``G`` that isn't the one trained with breaks the update identity first."""
    store, _ = _store(c, D, tree, w0, 1, perturb={c.m_of("G_2"): lambda p: p * 1.01})
    _expect(_verifier(c, D, tree, w0).verify_step(1, store), 1, "6a")


def test_bad_w_next_rejected_at_6a(c, D, tree, w0):
    """A3: an honest clean transcript with ``W_{t+1}`` from training on another batch."""
    store, _ = _store(c, D, tree, w0, 1)
    poisoned, _ = plain_step(c, c.build_model(), w0, _batch(c, D, 3))
    for name in c.weight_names:
        perturb_leaf(c, store, c.w_next_index(name), poisoned[name])
    _expect(_verifier(c, D, tree, w0).verify_step(1, store), 1, "6a")


def test_tiny_w_next_forgery_rejected_at_6a(c, D, tree, w0):
    """One entry moved by 64 ulps is outside the elementwise floor τ_W = 4 (P5b)."""
    store, out = _store(c, D, tree, w0, 1)
    name = c.weight_names[1]
    w = out.w_next[name].clone()
    w.view(-1)[5] = torch.nextafter(w.view(-1)[5], torch.tensor(float("inf")))
    for _ in range(63):
        w.view(-1)[5] = torch.nextafter(w.view(-1)[5], torch.tensor(float("inf")))
    perturb_leaf(c, store, c.w_next_index(name), w)
    rej = _verifier(c, D, tree, w0).verify_step(1, store)
    _expect(rej, 1, "6a")
    assert name in rej.detail and "entry 5" in rej.detail


def test_wrong_schedule_batch_rejected_at_4(c, D, tree, w0):
    """Records of D, honestly trained and committed, but not π(1)'s."""
    store, _ = _store(c, D, tree, w0, 1, records=_batch(c, D, 2),
                      path_indices=schedule(2, c.n_s, len(D)))
    rej = _verifier(c, D, tree, w0).verify_step(1, store)
    _expect(rej, 1, "4")
    assert "index 0" in rej.detail


def test_substituted_record_committed_truthfully_rejected_at_4(c, D, tree, w0):
    """A1: a record outside D, trained on and committed truthfully, with the clean path."""
    batch = _batch(c, D, 1)
    batch[2] = make_record(torch.zeros(c.d_in), torch.ones(c.d_out))
    store, _ = _store(c, D, tree, w0, 1, records=batch)
    rej = _verifier(c, D, tree, w0).verify_step(1, store)
    _expect(rej, 1, "4")
    assert "record 2" in rej.detail


def test_trained_on_other_batch_rejected_at_5_first_product(c, D, tree, w0):
    """A2: commit the clean b, with the products and update of training on b̃ (check 3)."""
    b_tilde = _batch(c, D, 1)
    b_tilde[0] = make_record(torch.zeros(c.d_in), torch.ones(c.d_out))
    store, _ = _store(c, D, tree, w0, 1, train_records=b_tilde)
    rej = _verifier(c, D, tree, w0).verify_step(1, store)
    _expect(rej, 1, "5")
    assert "P_1 (Y_1)" in rej.detail


def test_hidden_step_breaks_chain_at_7(c, D, tree, w0):
    """P11: step 1 honest, one unreported step, step 2 honest from W′ on π(2)."""
    v = _verifier(c, D, tree, w0)
    s1, out1 = _store(c, D, tree, w0, 1)
    assert v.verify_step(1, s1) is None
    w_prime, _ = plain_step(c, c.build_model(), out1.w_next, _batch(c, D, 5))
    s2, _ = _store(c, D, tree, w_prime, 2)
    _expect(v.verify_step(2, s2), 2, "7")


def test_wrong_base_weights_rejected_at_0(c, D, tree, w0):
    other = init_weights(c.widths, seed=1)
    store, _ = _store(c, D, tree, other, 1)
    _expect(_verifier(c, D, tree, w0).verify_step(1, store), 1, "0")


def test_tampered_root_rejected_at_2(c, D, tree, w0):
    store, _ = _store(c, D, tree, w0, 1)
    bad = bytes([store.root[0] ^ 1]) + store.root[1:]
    _expect(_verifier(c, D, tree, w0).verify_step(1, Wrapped(store, root=bad)), 1, "2")
    _expect(_verifier(c, D, tree, w0).verify_step(1, Wrapped(store, root=b"short")), 1, "2")


def test_product_changed_after_commit_rejected_at_2(c, D, tree, w0):
    """A leaf swapped under an unchanged root: the recomputed root disagrees."""
    store, out = _store(c, D, tree, w0, 1)
    i = c.product_index(2)
    rej = _verifier(c, D, tree, w0).verify_step(1, Wrapped(store, leaves={i: out.products[1] * 2}))
    _expect(rej, 1, "2")


# ---- malformed prover data: a rejection, never a crash ----------------------------------


def _malformed_cases(c, out):
    p = c.product_index(1)
    w_t = c.w_t_index(c.weight_names[0])
    nan_product = out.products[0].clone()
    nan_product.view(-1)[0] = float("nan")
    nan_record = out.records[0].clone()
    nan_record[0] = float("inf")
    return [
        ("product wrong dtype", {p: out.products[0].double()}, "2"),
        ("product NaN", {p: nan_product}, "2"),
        ("product wrong shape", {p: out.products[0][:, :-1].contiguous()}, "2"),
        ("product not a tensor", {p: "garbage"}, "2"),
        ("W_t wrong dtype", {w_t: out.w_t[c.weight_names[0]].half()}, "0"),
        ("record Inf", {0: nan_record}, "4"),
        ("record wrong length", {0: out.records[0][:-1].clone()}, "4"),
        ("record int dtype", {0: out.records[0].to(torch.int32)}, "4"),
    ]


def test_malformed_leaves_become_rejections(c, D, tree, w0):
    store, out = _store(c, D, tree, w0, 1)
    for what, leaves, check_id in _malformed_cases(c, out):
        rej = _verifier(c, D, tree, w0).verify_step(1, Wrapped(store, leaves=leaves))
        _expect(rej, 1, check_id)
        assert "malformed prover data" in rej.detail, what


def test_malformed_paths_become_rejections(c, D, tree, w0):
    store, _ = _store(c, D, tree, w0, 1)
    cases = [Wrapped(store, no_paths=True), Wrapped(store, paths={1: ["x"]}),
             Wrapped(store, paths={1: tree.path(1)[:-1]})]
    for s in cases:
        _expect(_verifier(c, D, tree, w0).verify_step(1, s), 1, "4")


def test_store_mutation_becomes_rejection(c, D, tree, w0):
    store, out = _store(c, D, tree, w0, 1)
    out.w_next[c.weight_names[0]].add_(0.0)  # an in-place write bumps the version
    _expect(_verifier(c, D, tree, w0).verify_step(1, store), 1, "2")


# ---- challenges -------------------------------------------------------------------------


def test_challenges_deterministic_in_the_root(c, D, tree, w0):
    store, _ = _store(c, D, tree, w0, 1)
    for spec in c.products:
        a = challenge_matrix(store.root, spec.m, K, spec.width)
        assert torch.equal(a, challenge_matrix(store.root, spec.m, K, spec.width))
        assert a.shape == (spec.width, K)
    # Two verifications of one store see identical numbers; a different root, different ones.
    v1, v2 = _verifier(c, D, tree, w0), _verifier(c, D, tree, w0)
    assert v1.verify_step(1, store) is None and v2.verify_step(1, store) is None
    assert v1.stats[1].products == v2.stats[1].products
    other_root = bytes([store.root[0] ^ 1]) + store.root[1:]
    spec = c.products[0]
    assert not torch.equal(challenge_matrix(store.root, spec.m, K, spec.width),
                           challenge_matrix(other_root, spec.m, K, spec.width))


def test_check_5_keys_on_recomputed_root(c, D, tree, w0):
    store, _ = _store(c, D, tree, w0, 1)
    ctx = StepContext.for_computation(
        c, step=1, indices=tuple(range(4)), h_D=tree.root, n_records=len(D),
        prev_w_hashes=(), chain_check_id="0", k=K)
    with pytest.raises(RuntimeError, match="check 2"):
        check_5_matmuls(store, c, ctx, Bands.provisional())
    assert check_2_commitment(store, c, ctx, Bands.provisional()) is None
    assert ctx.state.root == store.root
    assert check_5_matmuls(store, c, ctx, Bands.provisional()) is None


# ---- check 6b, generically --------------------------------------------------------------


class GlueMLP(MLPComputation):
    """The MLP with its last layer's gradient declared as glue, to exercise check 6b."""

    @property
    def linear_weights(self):
        lw = dict(super().linear_weights)
        lw.pop(self.weight(self.L))
        return lw

    def replay(self, leaves):
        return _GlueReplay(self, leaves)


class _GlueReplay(MLPReplay):
    def glue_gradients(self):
        super().glue_gradients()
        dy, x = self.dy(self.c.L), self.x(self.c.L)
        return {self.c.weight(self.c.L): dy.t() @ x}  # recomputed, not read from G_L


def test_check_6b_on_glue_gradients():
    c = GlueMLP((16, 32, 32, 8), n_s=4, eta=ETA)
    assert c.glue_gradient_weights == (c.weight(3),)
    D = synthetic_dataset(c.widths, N_RECORDS, seed=0)
    tree, w0 = dataset_tree(c, D), init_weights(c.widths, seed=0)
    v = _verifier(c, D, tree, w0)
    store, out = _store(c, D, tree, w0, 1)
    assert v.verify_step(1, store) is None
    assert [s.check_id for s in v.stats[1].tensors] == ["6a", "6a", "6b"]
    # A forged last-layer W_{t+1} passes 6a (not a linear weight there) and 5, then fails 6b.
    name = c.weight(3)
    perturb_leaf(c, store, c.w_next_index(name), out.w_next[name] + 1e-4)
    _expect(_verifier(c, D, tree, w0).verify_step(1, store), 1, "6b")


# ---- run-level checks 0, 1, 8, 9 --------------------------------------------------------


def test_check_1_dataset_anchor(c, D, tree, w0):
    v = Verifier(c, h_D=tree.root, n_records=len(D), k=K, bands=Bands.provisional(), w0=w0)
    other = list(D)
    other[3] = D[4]
    _expect(v.start_run(other), 0, "1")
    v = Verifier(c, h_D=tree.root, n_records=len(D), k=K, bands=Bands.provisional(), w0=w0)
    _expect(v.start_run(D[:-1]), 0, "1")
    # π must fit in D for every declared step (no wraparound, S5d).
    v = Verifier(c, h_D=tree.root, n_records=len(D), k=K, bands=Bands.provisional(), w0=w0,
                 n_steps=N_RECORDS // c.n_s + 1, schedule=lambda t: list(range(4 * t - 4, 4 * t)))
    _expect(v.start_run(D), 0, "1")


def test_check_8_final_anchor_and_check_9(c, D, tree, w0):
    v = _verifier(c, D, tree, w0, n_steps=2)
    w = w0
    for t in (1, 2):
        store, out = _store(c, D, tree, w, t)
        assert v.verify_step(t, store) is None
        w = out.w_next
    hashes = v.weight_hashes(w)
    bad = dict(w)
    bad[c.weight_names[0]] = w[c.weight_names[0]] + 1.0
    verdict = v.end_run(bad)
    assert not verdict.accepted
    _expect(verdict.rejection, 2, "8")
    v2 = _verifier(c, D, tree, w0, n_steps=2)
    w = w0
    for t in (1, 2):
        store, out = _store(c, D, tree, w, t)
        assert v2.verify_step(t, store) is None
        w = out.w_next
    assert v2.end_run(hashes).accepted
    # Fewer steps than declared is not the agreed run.
    v3 = _verifier(c, D, tree, w0, n_steps=2)
    s1, out1 = _store(c, D, tree, w0, 1)
    assert v3.verify_step(1, s1) is None
    _expect(v3.end_run(out1.w_next).rejection, 1, "8")


def test_rejected_run_stays_rejected(c, D, tree, w0):
    v = _verifier(c, D, tree, w0)
    store, out = _store(c, D, tree, w0, 1, perturb={1: lambda p: p + 1})
    _expect(v.verify_step(1, store), 1, "5")
    with pytest.raises(RuntimeError, match="already rejected"):
        v.verify_step(2, store)
    verdict = v.end_run(out.w_next)
    assert not verdict.accepted and verdict.rejection.check_id == "5"


def test_verifier_guards_its_own_inputs(c, D, tree, w0):
    v = Verifier(c, h_D=tree.root, n_records=len(D), k=K, bands=Bands.provisional(), w0=w0)
    store, _ = _store(c, D, tree, w0, 1)
    with pytest.raises(RuntimeError, match="start_run"):
        v.verify_step(1, store)
    v.start_run(D)
    with pytest.raises(ValueError, match="expected step 1"):
        v.verify_step(2, store)
    with pytest.raises(ValueError, match="exactly one"):
        Verifier(c, h_D=tree.root, n_records=len(D), k=K, bands=Bands.provisional())
    w0_hashes = v.weight_hashes(w0)
    v2 = Verifier(c, h_D=tree.root, n_records=len(D), k=K, bands=Bands.provisional(),
                  w0_hashes=w0_hashes)
    assert v2.start_run(D) is None and v2.verify_step(1, store) is None


# ---- bands ------------------------------------------------------------------------------


def test_provisional_bands():
    b = Bands.provisional()
    assert (b.tau, b.tau_w, b.source) == (8.0, TAU_W0, "provisional")
    assert b.kappa_max >= 1e3  # loose
    assert b.kappa_for("anything") == b.kappa_max and b.tau_w_for("w") == TAU_W0


def test_band_file_roundtrip_and_hash(tmp_path):
    b = Bands(tau=5.6, kappa_max=50.0, kappa_classes={"forward:x": 12.0}, tau_w=4.0,
              tau_w_tensors={"model.norm.weight": 9.5}, stats={"s_h": 0.7})
    path = tmp_path / "bands.json"
    path.write_bytes(b.to_json())
    loaded = Bands.from_file(path)
    assert loaded == dataclasses.replace(b, source=loaded.source)
    assert len(loaded.source) == 64 and loaded.source == Bands.from_file(path).source
    assert loaded.kappa_for("forward:x") == 12.0 and loaded.kappa_for("other") == 50.0
    assert loaded.tau_w_for("model.norm.weight") == 9.5 and loaded.tau_w_for("w") == 4.0
    with pytest.raises(TypeError):
        loaded.kappa_classes["new"] = 1.0  # frozen


@pytest.mark.parametrize("kw", [dict(tau=0.0), dict(tau=float("nan")), dict(kappa_max=0.5),
                                dict(tau_w=3.9), dict(tau_w_tensors={"w": 1.0})])
def test_bands_reject_invalid_values(kw):
    base = dict(tau=8.0, kappa_max=10.0)
    with pytest.raises(ValueError):
        Bands(**{**base, **kw})


# ---- invariant 1 and the prover-only methods ----------------------------------------------


FORBIDDEN_CALLS = {"loss", "label", "prove_step", "plain_step"}


@pytest.mark.parametrize("module", ["checks.py", "verifier.py"])
def test_verifier_never_touches_the_prover(module):
    tree = ast.parse((SRC / module).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or "prover" not in node.module, node.module
            assert all("prover" not in a.name for a in node.names)
        elif isinstance(node, ast.Import):
            assert all("prover" not in a.name for a in node.names)
        elif isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            assert name not in FORBIDDEN_CALLS, f"{module}:{node.lineno} calls {name}"
        elif isinstance(node, ast.Attribute):
            assert node.attr not in FORBIDDEN_CALLS, f"{module}:{node.lineno} uses .{node.attr}"
