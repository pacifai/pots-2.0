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

from setup.data import schedule
from verification.commitment import merkle
from verification.commitment.leaves import dataset_tree, leaf_hash
from verification.computation.instances import LlamaComputation
from verification.computation.instances.llama import LlamaReplay
from verification.computation.instances.mlp import (
    MLPComputation,
    MLPReplay,
    init_weights,
    make_record,
    synthetic_dataset,
)
from verification.parameters import UNIT_ROUNDOFF
from verification.prover import step as prover
from verification.prover.step import plain_step, prove_step
from verification.transcript.reader import TranscriptView
from verification.transcript.store import InMemoryStore, TranscriptStore, perturb_leaf
from verification.verifier import checks
from verification.verifier.bands import TAU_W0, Bands, product_class
from verification.verifier.checks import (
    CHECKS,
    DEFAULT_ORDER,
    check_2_commitment,
    check_5_matmuls,
)
from verification.verifier.context import CommittedLeaves, Rejection, StepContext
from verification.verifier.driver import Verifier
from verification.verifier.matmul_check.challenges import SIGMA_R, challenge_matrix
from verification.verifier.matmul_check.freivalds import _safe_norm
from verification.verifier.matmul_check.sizing import Z, e_m

from tests.verification.llama_helpers import make_records, tiny_config

ETA = 1e-3
K = 7
N_RECORDS = 40
SRC = Path(__file__).resolve().parents[3] / "verification" / "verifier"


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


T_DEFAULT = 5


def _make(c, D, tree, w0=None, *, n_steps=T_DEFAULT, **kw) -> Verifier:
    kw.setdefault("bands", Bands.provisional())
    kw.setdefault("allow_provisional", True)
    if w0 is not None:
        kw["w0"] = w0
    return Verifier(c, h_D=tree.root, n_records=len(D), k=K, n_steps=n_steps, **kw)


def _verifier(c, D, tree, w0, **kw) -> Verifier:
    v = _make(c, D, tree, w0, **kw)
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


def _expect(rej, step, check_id, kind="failed"):
    assert isinstance(rej, Rejection), rej
    assert (rej.step, rej.check_id, rej.kind) == (step, check_id, kind), rej


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
    with pytest.raises(RuntimeError, match="unfrozen"):
        v.end_run({})
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
    _expect(_verifier(c, D, tree, w0).verify_step(1, Wrapped(store, root=b"short")), 1, "2",
            "malformed")


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
        ("record not a tensor", {0: {"x": 1}}, "4"),
        ("record None", {1: None}, "4"),
    ]


def test_malformed_leaves_become_rejections(c, D, tree, w0):
    store, out = _store(c, D, tree, w0, 1)
    for what, leaves, check_id in _malformed_cases(c, out):
        rej = _verifier(c, D, tree, w0).verify_step(1, Wrapped(store, leaves=leaves))
        _expect(rej, 1, check_id, "malformed")
        assert "malformed prover data" in rej.detail, what


def test_malformed_paths_become_rejections(c, D, tree, w0):
    store, _ = _store(c, D, tree, w0, 1)
    cases = [Wrapped(store, no_paths=True), Wrapped(store, paths={1: ["x"]}),
             Wrapped(store, paths={1: tree.path(1)[:-1]})]
    for s in cases:
        _expect(_verifier(c, D, tree, w0).verify_step(1, s), 1, "4",
                "failed" if s.paths.get(1) == tree.path(1)[:-1] else "malformed")


def test_store_mutation_becomes_rejection(c, D, tree, w0):
    store, out = _store(c, D, tree, w0, 1)
    out.w_next[c.weight_names[0]].add_(0.0)  # an in-place write bumps the version
    _expect(_verifier(c, D, tree, w0).verify_step(1, store), 1, "2", "malformed")


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
    other = list(D)
    other[3] = D[4]
    _expect(_make(c, D, tree, w0).start_run(other), 0, "1")
    _expect(_make(c, D, tree, w0).start_run(D[:-1]), 0, "1")
    # π must fit in D for every declared step (no wraparound, S5d).
    v = _make(c, D, tree, w0, n_steps=N_RECORDS // c.n_s + 1,
              schedule=lambda t: list(range(4 * t - 4, 4 * t)))
    _expect(v.start_run(D), 0, "1")
    _expect(_make(c, D, tree, w0, n_steps=N_RECORDS // c.n_s + 1).start_run(D), 0, "1")


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
    v = _make(c, D, tree, w0)
    store, _ = _store(c, D, tree, w0, 1)
    with pytest.raises(RuntimeError, match="start_run"):
        v.verify_step(1, store)
    v.start_run(D)
    with pytest.raises(RuntimeError, match="expected step 1"):
        v.verify_step(2, store)
    with pytest.raises(ValueError, match="exactly one"):
        _make(c, D, tree)
    with pytest.raises(TypeError):
        Verifier(c, h_D=tree.root, n_records=len(D), k=K, bands=Bands.provisional(),
                 allow_provisional=True, w0=w0)  # T is required
    with pytest.raises(ValueError, match="n_steps"):
        _make(c, D, tree, w0, n_steps=0)
    v2 = _make(c, D, tree, w0_hashes=v.weight_hashes(w0))
    assert v2.start_run(D) is None and v2.verify_step(1, store) is None
    v3 = _verifier(c, D, tree, w0, n_steps=1)
    assert v3.verify_step(1, store) is None
    with pytest.raises(RuntimeError, match="past the declared T"):
        v3.verify_step(2, store)


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
    assert b.source == "provisional" and loaded.source != "provisional"
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


def _const_str(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


@pytest.mark.parametrize("module", sorted(str(f.relative_to(SRC)) for f in SRC.rglob("*.py")))
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
            consts = [_const_str(a) for a in node.args]
            if name in ("getattr", "hasattr", "setattr", "attrgetter", "methodcaller"):
                assert not set(consts) & FORBIDDEN_CALLS, f"{module}:{node.lineno} {name}"
            if name in ("import_module", "__import__") or (
                    isinstance(f, ast.Attribute) and getattr(f.value, "id", "") == "importlib"):
                assert not any(x and "prover" in x for x in consts), f"{module}:{node.lineno}"
            assert name not in ("eval", "exec"), f"{module}:{node.lineno} calls {name}"
        elif isinstance(node, ast.Attribute):
            assert node.attr not in FORBIDDEN_CALLS, f"{module}:{node.lineno} uses .{node.attr}"


# ---- review round 1: overflow (finding 1) -----------------------------------------------


def test_safe_norm_matches_and_does_not_overflow():
    g = torch.Generator().manual_seed(0)
    for shape in [(7,), (33, 5), (128, 7)]:
        for scale in (1e-15, 1e-3, 1.0, 1e3, 1e15):
            x = torch.randn(shape, generator=g) * scale
            assert torch.equal(_safe_norm(x), torch.linalg.vector_norm(x))  # bit-identical
            if len(shape) == 2:
                assert torch.equal(_safe_norm(x, dim=0), torch.linalg.vector_norm(x, dim=0))
    tiny = torch.tensor([3e-30, 4e-30])  # squares underflow: unscaled gives 0
    assert float(torch.linalg.vector_norm(tiny)) == 0.0
    assert float(_safe_norm(tiny)) == pytest.approx(5e-30, rel=1e-6)
    big = torch.tensor([3e19, 1.0])
    assert torch.linalg.vector_norm(big) == float("inf")  # the reason _safe_norm exists
    assert float(_safe_norm(big)) == pytest.approx(3e19, rel=1e-6)
    cols = torch.tensor([[3e19, 0.0], [4e19, 0.0]])
    assert _safe_norm(cols, dim=0).tolist() == pytest.approx([5e19, 0.0], rel=1e-6)
    assert float(_safe_norm(torch.zeros(4))) == 0.0
    assert float(_safe_norm(torch.tensor([1.0, float("inf")]))) == float("inf")
    assert math.isnan(float(_safe_norm(torch.tensor([1.0, float("nan")]))))


def test_honest_stats_unchanged_by_scaled_norms(c, D, tree, w0):
    """Check 5's numbers with ``_safe_norm`` equal those with the unscaled norm, bit for bit."""
    store, _ = _store(c, D, tree, w0, 1)
    ctx = StepContext.for_computation(
        c, step=1, indices=tuple(schedule(1, c.n_s, len(D))), h_D=tree.root, n_records=len(D),
        prev_w_hashes=(), chain_check_id="0", k=K)
    assert check_2_commitment(store, c, ctx, Bands.provisional()) is None
    assert check_5_matmuls(store, c, ctx, Bands.provisional()) is None
    view, root = TranscriptView(c, ctx.state.committed()), ctx.state.root
    replay = c.replay(ctx.state.committed())
    for spec, stat in zip(c.products, ctx.stats.products):
        a, b = replay.operands(spec.m)
        p = view.product(spec.m)
        r = challenge_matrix(root, spec.m, K, spec.width)
        res = torch.linalg.vector_norm(a @ (b @ r) - p @ r, dim=0)
        unit = SIGMA_R * e_m(spec.q, ctx.eps_in, ctx.eps_acc) * float(torch.linalg.vector_norm(p))
        assert stat.normalized == tuple(x / unit for x in res.tolist())


def test_overflowing_weight_gradient_rejected_at_5(c, D, tree, w0):
    """Probe case 1: a forged G_2 with one finite entry of 3e19, and a W_{t+1} consistent with
    it, passes 6a. With unscaled norms ‖P‖_F and the residual overflowed to inf and inf ≤ inf
    passed; with scaled norms the residual test itself fails."""
    store, out = _store(c, D, tree, w0, 1)
    m, name = c.m_of("G_2"), c.weight(2)
    g = out.products[m - 1] + 0.5 * torch.randn(out.products[m - 1].shape,
                                                generator=torch.Generator().manual_seed(0))
    g.view(-1)[0] = 3e19
    perturb_leaf(c, store, c.product_index(m), g)
    perturb_leaf(c, store, c.w_next_index(name), (out.w_t[name] - ETA * g).contiguous())
    rej = _verifier(c, D, tree, w0).verify_step(1, store)
    _expect(rej, 1, "5")
    assert f"P_{m} (G_2), challenge j=1: normalized residual" in rej.detail


def test_overflowing_forward_product_rejected_at_5(c, D, tree, w0):
    """Probe case 2: Y_1 + 1 everywhere, with one entry at 3e19."""
    store, out = _store(c, D, tree, w0, 1)
    y = out.products[0] + 1.0
    y.view(-1)[0] = 3e19
    perturb_leaf(c, store, c.product_index(1), y)
    rej = _verifier(c, D, tree, w0).verify_step(1, store)
    _expect(rej, 1, "5")
    assert "P_1 (Y_1), challenge j=1: normalized residual" in rej.detail


class HugeOperandMLP(MLPComputation):
    """Replay operands scaled so ``A(B·r)`` exceeds the fp32 maximum: the finiteness guard."""

    def replay(self, leaves):
        return _HugeReplay(self, leaves)


class _HugeReplay(MLPReplay):
    def operands(self, m):
        a, b = super().operands(m)
        return (a * 1e30, b * 1e30) if m == 1 else (a, b)


def test_overflowing_matmul_output_rejected_at_5():
    c = HugeOperandMLP((16, 32, 32, 8), n_s=4, eta=ETA)
    D = synthetic_dataset(c.widths, N_RECORDS, seed=0)
    tree, w0 = dataset_tree(c, D), init_weights(c.widths, seed=0)
    store, _ = _store(c, D, tree, w0, 1)
    for calibrate in (False, True):
        rej = _verifier(c, D, tree, w0, calibrate=calibrate).verify_step(1, store)
        _expect(rej, 1, "5")
        assert "P_1 (Y_1): non-finite ν" in rej.detail and "residual j=1" in rej.detail


def test_overflowing_update_bound_rejected_at_6a():
    """With η = 1e3, a G entry of 1e36 makes η·G = inf, so the bound is inf for any W_{t+1}."""
    c = MLPComputation((16, 32, 32, 8), n_s=4, eta=1e3)
    D = synthetic_dataset(c.widths, N_RECORDS, seed=0)
    tree, w0 = dataset_tree(c, D), init_weights(c.widths, seed=0)
    store, out = _store(c, D, tree, w0, 1)
    m = c.linear_weights[c.weight(1)]
    g = out.products[m - 1].clone()
    g.view(-1)[3] = 1e36
    perturb_leaf(c, store, c.product_index(m), g)
    for calibrate in (False, True):
        rej = _verifier(c, D, tree, w0, calibrate=calibrate).verify_step(1, store)
        _expect(rej, 1, "6a")
        assert "non-finite" in rej.detail and "entry 3" in rej.detail


# ---- review round 1: bytes bound to check 2 (finding 2) ---------------------------------


class TwoFaced(Wrapped):
    """Serves ``first[i]`` on the first read of leaf ``i``, the inner store's leaf after."""

    def __init__(self, inner, first, dataset_paths=None):
        super().__init__(inner, paths=dataset_paths)
        self.first, self.reads = dict(first), {}

    def leaf(self, index):
        self.reads[index] = self.reads.get(index, 0) + 1
        if index in self.first and self.reads[index] == 1:
            return self.first[index]
        return self.inner.leaf(index)


def test_records_served_twice_rejected_at_2(c, D, tree, w0):
    """Probe case 3, A1 through the store: π(1)'s records for check 4, the trained batch for 2."""
    trained, _ = _store(c, D, tree, w0, 1, records=_batch(c, D, 3))
    clean = _batch(c, D, 1)
    idx = schedule(1, c.n_s, len(D))
    s = TwoFaced(trained, {i: clean[i] for i in range(c.n_s)},
                 {i: tree.path(idx[i]) for i in range(c.n_s)})
    rej = _verifier(c, D, tree, w0).verify_step(1, s)
    _expect(rej, 1, "2")
    assert "served two versions" in rej.detail


def test_w_t_served_twice_rejected_at_2(c, D, tree, w0):
    """A hidden step through the store: the chained W_t for check 7, W′ for check 2."""
    v = _verifier(c, D, tree, w0)
    s1, out1 = _store(c, D, tree, w0, 1)
    assert v.verify_step(1, s1) is None
    w_prime, _ = plain_step(c, c.build_model(), out1.w_next, _batch(c, D, 5))
    s2, _ = _store(c, D, tree, w_prime, 2)
    first = {c.w_t_index(n): out1.w_next[n] for n in c.weight_names}
    _expect(v.verify_step(2, TwoFaced(s2, first)), 2, "2")


def test_no_leaf_read_after_check_2(c, D, tree, w0):
    """Records and W_t are read twice (by 4 or 7, then by 2) and every other leaf once: checks
    6a, 5 and 6b read check 2's objects, never the store."""
    store, _ = _store(c, D, tree, w0, 1)
    s = TwoFaced(store, {})
    assert _verifier(c, D, tree, w0).verify_step(1, s) is None
    early = set(range(c.n_s)) | {c.w_t_index(n) for n in c.weight_names}
    assert s.reads == {i: 2 if i in early else 1 for i in range(c.n_leaves)}


def test_mutation_during_check_2_rejected_at_2(c, D, tree, w0):
    """An in-place write to a hashed leaf while check 2 runs (here in the root getter, which
    check 2 reads after hashing every leaf): check 2 itself rejects, as malformed."""
    store, _ = _store(c, D, tree, w0, 1)
    y1 = store.leaf(c.product_index(1))

    class Mutating(Wrapped):
        @property
        def root(self):
            y1.mul_(2.0)
            return self.inner.root

    rej = _verifier(c, D, tree, w0).verify_step(1, Mutating(store))
    _expect(rej, 1, "2", "malformed")
    assert f"leaf {c.product_index(1)} changed in place during check 2" in rej.detail


@pytest.fixture
def parallel(monkeypatch):
    """Check 2's hashing with one leaf per task on four workers, so leaves are hashed out of
    order across threads while later leaves are still being read."""
    monkeypatch.setattr(merkle, "_CHUNK_BYTES", 1)
    before = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(before)


def _malformed_detail(c, i, obj):
    try:
        leaf_hash(c, i, obj)
    except ValueError as e:
        return f"malformed prover data: {type(e).__name__}: {e}"
    raise AssertionError(f"leaf {i} hashes")


class Unreadable(Wrapped):
    def __init__(self, inner, bad, **kw):
        super().__init__(inner, **kw)
        self.bad = bad

    def leaf(self, index):
        if index == self.bad:
            raise IndexError(f"leaf {index} is missing")
        return super().leaf(index)


def test_check_2_reports_the_first_bad_leaf_in_order(c, D, tree, w0, parallel):
    """Leaves are hashed in parallel, but the rejection is the one a leaf-by-leaf loop gives:
    the first malformed leaf in leaf order, whether it fails to read, validate or encode."""
    store, out = _store(c, D, tree, w0, 1)
    p1, p2 = c.product_index(1), c.product_index(3)
    nan1 = out.products[0].clone()
    nan1.view(-1)[3] = float("nan")
    inf2 = out.products[2].clone().fill_(float("inf"))
    shape1, shape2 = out.products[0].t().contiguous(), out.products[2].t().contiguous()
    cases = [
        (Wrapped(store, leaves={p1: nan1, p2: shape2}), _malformed_detail(c, p1, nan1)),
        (Wrapped(store, leaves={p1: shape1, p2: inf2}), _malformed_detail(c, p1, shape1)),
        (Unreadable(store, p2, leaves={p1: nan1}), _malformed_detail(c, p1, nan1)),
        (Unreadable(store, p1, leaves={p2: inf2}),
         f"malformed prover data: IndexError: leaf {p1} is missing"),
    ]
    for s, detail in cases:
        rej = _verifier(c, D, tree, w0).verify_step(1, s)
        _expect(rej, 1, "2", "malformed")
        assert rej.detail == detail


def test_check_2_parallel_hashes_match_the_commitment(c, D, tree, w0, parallel):
    store, out = _store(c, D, tree, w0, 1)
    ctx = StepContext.for_computation(
        c, step=1, indices=tuple(schedule(1, c.n_s, len(D))), h_D=tree.root, n_records=len(D),
        prev_w_hashes=(), chain_check_id="0", k=K)
    assert check_2_commitment(store, c, ctx, Bands.provisional()) is None
    assert ctx.state.leaf_hashes == [leaf_hash(c, i, x) for i, x in enumerate(out.leaves())]
    assert ctx.state.root == store.root


def test_mutation_by_a_later_read_rejected_at_2(c, D, tree, w0, parallel):
    """A store whose read of a later leaf writes in place into a leaf it already served: the
    earlier leaf may be hashed after the write, so check 2 rejects it as malformed."""
    store, _ = _store(c, D, tree, w0, 1)
    y1 = store.leaf(c.product_index(1))

    class MutatingRead(Wrapped):
        def leaf(self, index):
            if index == c.product_index(4):
                y1.mul_(2.0)
            return super().leaf(index)

    rej = _verifier(c, D, tree, w0).verify_step(1, MutatingRead(store))
    _expect(rej, 1, "2", "malformed")
    assert rej.detail == f"leaf {c.product_index(1)} changed in place during check 2"


class MutatingReplayMLP(MLPComputation):
    """Writes a committed product in place after check 2, when check 5 builds its replay."""

    def replay(self, leaves):
        leaves.leaf(self.product_index(1)).mul_(2.0)
        return super().replay(leaves)


def test_mutation_after_check_2_rejected(c, D, tree, w0):
    """After check 2 the cache's ``_version`` guard turns the write into a malformed rejection
    at the first check that reads the leaf, never the changed bytes."""
    c = MutatingReplayMLP((16, 32, 32, 8), n_s=4, eta=ETA)
    store, _ = _store(c, D, tree, w0, 1)
    rej = _verifier(c, D, tree, w0).verify_step(1, store)
    _expect(rej, 1, "5", "malformed")
    assert "changed in place after check 2" in rej.detail


def test_committed_leaf_index_out_of_range_is_a_verifier_bug():
    leaves = CommittedLeaves()
    leaves.add(torch.zeros(2))
    assert torch.equal(leaves.leaf(0), torch.zeros(2))
    for bad in (-1, 1, 7):
        with pytest.raises(RuntimeError, match="outside"):
            leaves.leaf(bad)


# ---- review round 1: error mapping (finding 3) ------------------------------------------


class BuggyReplayMLP(MLPComputation):
    def replay(self, leaves):
        return _BuggyReplay(self, leaves)


class _BuggyReplay(MLPReplay):
    def operands(self, m):
        raise ValueError("a verifier bug")


def test_verifier_side_errors_propagate():
    """An error in the verifier's own replay is a bug, not a rejection of the prover."""
    c = BuggyReplayMLP((16, 32, 32, 8), n_s=4, eta=ETA)
    D = synthetic_dataset(c.widths, N_RECORDS, seed=0)
    tree, w0 = dataset_tree(c, D), init_weights(c.widths, seed=0)
    store, _ = _store(c, D, tree, w0, 1)
    with pytest.raises(ValueError, match="a verifier bug"):
        _verifier(c, D, tree, w0).verify_step(1, store)


def test_check_preconditions_raise_runtime_error(c, D, tree, w0):
    store, _ = _store(c, D, tree, w0, 1)
    ctx = StepContext.for_computation(c, step=1, indices=(0, 1), h_D=tree.root,
                                      n_records=len(D), prev_w_hashes=(), chain_check_id="0", k=K)
    for check in (CHECKS["4"], CHECKS["7"], CHECKS["6a"], CHECKS["5"]):
        with pytest.raises(RuntimeError):
            check(store, c, ctx, Bands.provisional())


# ---- review round 1: calibration and bands (findings 4, 8) ------------------------------


def _band_file(**kw):
    """Bands as loaded from a band file, so their source is a hash, not "provisional"."""
    return Bands.from_json(Bands(**{"tau": 8.0, "kappa_max": 1e4, **kw}).to_json())


def _calibrate(c, D, tree, w0, T, steps, **perturbs):
    v = _make(c, D, tree, w0, n_steps=T, bands=None, calibrate=True, allow_provisional=False)
    assert v.start_run(D) is None
    w = w0
    for t in range(1, steps + 1):
        store, out = _store(c, D, tree, w, t, perturb=perturbs.get(f"t{t}"))
        assert v.verify_step(t, store) is None
        w = out.w_next
    return v, w


def test_freeze_accepts_honest_run_and_records_source(c, D, tree, w0):
    v, w = _calibrate(c, D, tree, w0, 3, 2)
    assert v.band_source is None
    with pytest.raises(RuntimeError, match="unfrozen"):
        v.end_run(w)
    bands = _band_file()
    assert v.freeze(bands) is None and v.band_source == bands.source != "provisional"
    store, out = _store(c, D, tree, w, 3)
    assert v.verify_step(3, store) is None
    verdict = v.end_run(out.w_next)
    assert verdict.accepted and verdict.band_source == bands.source
    with pytest.raises(RuntimeError, match="calibration run"):
        v.freeze(bands)


def test_steps_after_freeze_are_judged(c, D, tree, w0):
    v, w = _calibrate(c, D, tree, w0, 2, 1)
    assert v.freeze(_band_file()) is None
    store, out = _store(c, D, tree, w, 2, perturb={c.m_of("Y_2"): lambda p: p * 1.01})
    _expect(v.verify_step(2, store), 2, "5")
    assert not v.end_run(out.w_next).accepted


@pytest.mark.parametrize("kw", [dict(tau=1e-9), dict(kappa_max=1.0)])
def test_freeze_rejects_honest_numbers_outside_the_bands(c, D, tree, w0, kw):
    v, w = _calibrate(c, D, tree, w0, 1, 1)
    _expect(v.freeze(_band_file(**kw)), 1, "5")
    assert not v.end_run(w).accepted


def test_freeze_catches_a_fault_recorded_in_calibration(c, D, tree, w0):
    v, _ = _calibrate(c, D, tree, w0, 2, 2, t2={c.m_of("Y_2"): lambda p: p * 1.01})
    rej = v.freeze(_band_file())
    _expect(rej, 2, "5")
    assert "Y_2" in rej.detail


def test_freeze_scores_6a_before_5(c, D, tree, w0):
    """A perturbed G fails both 6a and 5; the live order reports 6a, and so does freeze."""
    v, _ = _calibrate(c, D, tree, w0, 1, 1, t1={c.m_of("G_2"): lambda p: p * 1.01})
    _expect(v.freeze(_band_file()), 1, "6a")


def test_provisional_bands_need_opt_in(c, D, tree, w0):
    with pytest.raises(ValueError, match="allow_provisional"):
        _make(c, D, tree, w0, allow_provisional=False)
    v = _make(c, D, tree, w0, bands=None, calibrate=True, allow_provisional=False)
    with pytest.raises(ValueError, match="allow_provisional"):
        v.freeze(Bands.provisional())
    with pytest.raises(ValueError, match="required"):
        _make(c, D, tree, w0, bands=None)
    assert _make(c, D, tree, w0, bands=_band_file(), allow_provisional=False).start_run(D) is None


def test_unknown_band_keys_refused(c, D, tree, w0):
    known = product_class(c, c.products[0])
    _make(c, D, tree, w0, bands=_band_file(kappa_classes={known: 20.0},
                                           tau_w_tensors={c.weight(1): 5.0}))
    with pytest.raises(ValueError, match="κ classes"):
        _make(c, D, tree, w0, bands=_band_file(kappa_classes={"forward:nope": 20.0}))
    with pytest.raises(ValueError, match="τ_W tensors"):
        _make(c, D, tree, w0, bands=_band_file(tau_w_tensors={"layers.9.weight": 5.0}))
    v = _make(c, D, tree, w0, bands=None, calibrate=True)
    with pytest.raises(ValueError, match="κ classes"):
        v.freeze(_band_file(kappa_classes={"forward:nope": 20.0}))


# ---- review round 1: invariant 1 at run time (finding 5) --------------------------------


def test_verifier_runs_with_prover_entry_points_disabled(c, D, tree, w0, monkeypatch):
    T = 3
    stores, w = [], w0
    for t in range(1, T + 1):
        store, out = _store(c, D, tree, w, t)
        stores.append(store)
        w = out.w_next

    def boom(*a, **k):
        raise AssertionError("the verifier called a prover-only entry point")

    monkeypatch.setattr(MLPComputation, "loss", boom)
    monkeypatch.setattr(MLPComputation, "label", boom)
    monkeypatch.setattr(prover, "prove_step", boom)
    monkeypatch.setattr(prover, "plain_step", boom)
    v = _verifier(c, D, tree, w0, n_steps=T)
    for t, store in enumerate(stores, start=1):
        assert v.verify_step(t, store) is None
    assert v.end_run(w).accepted


# ---- review round 1: more faults (finding 6) --------------------------------------------


def test_permuted_batch_rejected_at_4(c, D, tree, w0):
    """π(1)'s records in another order, each with its own true path into h_D."""
    idx = schedule(1, c.n_s, len(D))
    order = [1, 0, 3, 2]
    store, _ = _store(c, D, tree, w0, 1, records=[D[idx[i]] for i in order],
                      path_indices=[idx[i] for i in order])
    rej = _verifier(c, D, tree, w0).verify_step(1, store)
    _expect(rej, 1, "4")
    assert "record 0" in rej.detail


def _near_band(c, name, factor, seed=0):
    """Add a Gaussian Δ with ``‖Δ‖_F = factor·τ·e_m·‖P‖_F`` to product ``name``.

    For any Δ, ``E‖Δ·r‖² = σ_r²·‖Δ‖_F²``, so Δ alone gives a normalized residual of about
    ``factor·τ``. It lands there only if the band uses the same σ_r, e_m(q, ε_in, ε_acc) and
    ‖P‖_F."""
    spec = c.product(c.m_of(name))
    eps = UNIT_ROUNDOFF[torch.float32]
    tau = Bands.provisional().tau

    def f(p):
        d = torch.randn(p.shape, generator=torch.Generator().manual_seed(seed))
        target = factor * tau * e_m(spec.q, eps, eps) * float(torch.linalg.vector_norm(p))
        return (p + d * (target / float(torch.linalg.vector_norm(d)))).contiguous()
    return {spec.m: f}


NEAR_BAND_SEEDS = range(8)
NEAR_BAND_PRODUCTS = ("Y_2", "dX_2", "G_3")


def test_near_band_perturbation_lands_at_its_scale(c, D, tree, w0):
    """Pooled over 3 products × 8 seeds × k columns, the 3× residuals have an RMS within
    2.5τ–3.5τ (it is about 3.0τ; a single small product alone spreads too wide). A band scale
    off by √3, as with σ_r taken as 1, puts it near 1.7τ, outside the window."""
    pooled = {0.1: [], 3.0: []}
    for name in NEAR_BAND_PRODUCTS:
        m = c.m_of(name)
        for factor in pooled:
            for seed in NEAR_BAND_SEEDS:
                store, _ = _store(c, D, tree, w0, 1, perturb=_near_band(c, name, factor, seed))
                v = _verifier(c, D, tree, w0, calibrate=True)
                assert v.verify_step(1, store) is None
                pooled[factor].extend(v.stats[1].products[m - 1].normalized)
    rms = math.sqrt(sum(x * x for x in pooled[3.0]) / len(pooled[3.0]))
    assert max(pooled[0.1]) < Z / 2
    assert 2.5 * Z < rms < 3.5 * Z, rms


@pytest.mark.parametrize("name", ["Y_2", "dX_2"])  # a perturbed G fails 6a first
def test_near_band_perturbation_judged(c, D, tree, w0, name):
    m = c.m_of(name)
    store, _ = _store(c, D, tree, w0, 1, perturb=_near_band(c, name, 0.1))
    assert _verifier(c, D, tree, w0).verify_step(1, store) is None
    store, _ = _store(c, D, tree, w0, 1, perturb=_near_band(c, name, 3.0))
    rej = _verifier(c, D, tree, w0).verify_step(1, store)
    _expect(rej, 1, "5")
    assert f"P_{m} ({name})" in rej.detail


# σ_r·e_m(q, ε, ε) = (1/√3)·(√2 + √q)·2⁻²⁴, worked by hand, independent of config and sizing:
#   q = 16: (1.414214 + 4)·5.960464e-8 = 3.227123e-7, × 0.5773503 = 1.863181e-7
#   q = 4:  (1.414214 + 2)·5.960464e-8 = 2.035030e-7, × 0.5773503 = 1.174925e-7
HAND_UNIT_PER_NORM = {"Y_1": (16, 1.863181e-7), "G_1": (4, 1.174925e-7)}


@pytest.mark.parametrize("name", sorted(HAND_UNIT_PER_NORM))
def test_band_unit_matches_hand_computed_value(c, D, tree, w0, name):
    q, per_norm = HAND_UNIT_PER_NORM[name]
    spec = c.product(c.m_of(name))
    assert spec.q == q
    store, out = _store(c, D, tree, w0, 1)
    v = _verifier(c, D, tree, w0, calibrate=True)
    assert v.verify_step(1, store) is None
    stat = v.stats[1].products[spec.m - 1]
    p = out.products[spec.m - 1]
    # Recompute the residuals from the same leaves and challenges, then divide by the hand unit.
    replay = c.replay(store)
    for m in range(1, spec.m + 1):
        a, b = replay.operands(m)
    r = challenge_matrix(store.root, spec.m, K, spec.width)
    res = torch.linalg.vector_norm(a @ (b @ r) - p @ r, dim=0).tolist()
    unit = per_norm * float(torch.linalg.vector_norm(p))
    for got, x in zip(stat.normalized, res):
        assert got == pytest.approx(x / unit, rel=1e-5)


def test_w_t_swapped_and_rerooted_rejected_at_7(c, D, tree, w0):
    """A W_t leaf replaced at step 2 and the root rebuilt to match: only the chain catches it."""
    v = _verifier(c, D, tree, w0)
    s1, out1 = _store(c, D, tree, w0, 1)
    assert v.verify_step(1, s1) is None
    s2, _ = _store(c, D, tree, out1.w_next, 2)
    name = c.weight(2)
    perturb_leaf(c, s2, c.w_t_index(name), (out1.w_next[name] * 1.001).contiguous())
    rej = v.verify_step(2, s2)
    _expect(rej, 2, "7")
    assert name in rej.detail


class NaNGlueMLP(GlueMLP):
    def replay(self, leaves):
        return _NaNGlueReplay(self, leaves)


class _NaNGlueReplay(_GlueReplay):
    def glue_gradients(self):
        g = dict(super().glue_gradients())
        name = self.c.weight(self.c.L)
        g[name] = g[name].clone()
        g[name].view(-1)[0] = float("nan")
        return g


def test_nan_glue_gradient_rejected_at_6b():
    """NaN out of glue can't pass the update identity."""
    c = NaNGlueMLP((16, 32, 32, 8), n_s=4, eta=ETA)
    D = synthetic_dataset(c.widths, N_RECORDS, seed=0)
    tree, w0 = dataset_tree(c, D), init_weights(c.widths, seed=0)
    store, _ = _store(c, D, tree, w0, 1)
    rej = _verifier(c, D, tree, w0).verify_step(1, store)
    _expect(rej, 1, "6b")
    assert "entry 0" in rej.detail


# ---- review round 1: ε from C (finding 9) -----------------------------------------------


class Bf16AccMLP(MLPComputation):
    @property
    def accumulator_dtype(self):
        return torch.bfloat16


def test_unit_roundoffs_come_from_c(c):
    ctx = StepContext.for_computation(c, step=1, indices=(0, 1, 2, 3), h_D=b"\0" * 32,
                                      n_records=40, prev_w_hashes=(), chain_check_id="0", k=K)
    assert ctx.eps_in == ctx.eps_acc == ctx.eps_w == UNIT_ROUNDOFF[torch.float32]
    c2 = Bf16AccMLP((16, 32, 32, 8), n_s=4, eta=ETA)
    ctx = StepContext.for_computation(c2, step=1, indices=(0, 1, 2, 3), h_D=b"\0" * 32,
                                      n_records=40, prev_w_hashes=(), chain_check_id="0", k=K)
    assert ctx.eps_in == UNIT_ROUNDOFF[torch.float32]
    assert ctx.eps_acc == UNIT_ROUNDOFF[torch.bfloat16]


def test_check_6_live_and_freeze_agree_at_the_boundary(c, D, tree, w0):
    """With τ_W set to a weight's recorded ρ_max exactly, the live check and the freeze-time
    rejudge both accept; one float below it, both reject at 6a."""
    store, out = _store(c, D, tree, w0, 1)
    name = c.weight_names[1]
    w = out.w_next[name].clone()
    for _ in range(5):
        w.view(-1)[5] = torch.nextafter(w.view(-1)[5], torch.tensor(float("inf")))
    perturb_leaf(c, store, c.w_next_index(name), w)
    v, _ = _calibrate_store(c, D, tree, w0, store)
    rho = next(s.rho_max for s in v.stats[1].tensors if s.weight == name)
    assert rho > TAU_W0
    below = math.nextafter(rho, 0.0)
    for tau_w, accepted in ((rho, True), (below, False)):
        bands = _band_file(tau_w_tensors={name: tau_w})
        live = _make(c, D, tree, w0, n_steps=1, bands=bands)
        live.start_run(D)
        rej_live = live.verify_step(1, store)
        cal, _ = _calibrate_store(c, D, tree, w0, store)
        rej_freeze = cal.freeze(bands)
        if accepted:
            assert rej_live is None and rej_freeze is None
        else:
            _expect(rej_live, 1, "6a")
            _expect(rej_freeze, 1, "6a")


def _calibrate_store(c, D, tree, w0, store):
    v = _make(c, D, tree, w0, n_steps=1, bands=None, calibrate=True)
    assert v.start_run(D) is None
    assert v.verify_step(1, store) is None
    return v, None


# ---- check 5 measures the members of an attention product as a batch --------------------


class _LlamaWith(LlamaComputation):
    """The tiny Llama with a replay hook: ``operands_hook(m, a, b) -> (a, b)``."""

    operands_hook = None

    def replay(self, leaves):
        return _HookedReplay(self, leaves)


class _HookedReplay(LlamaReplay):
    def operands(self, m):
        a, b = super().operands(m)
        hook = type(self.c).operands_hook
        return (a, b) if hook is None else hook(self.c, m, a, b)


@pytest.fixture(scope="module")
def llama_out():
    c = LlamaComputation(tiny_config(), n_s=2, n=12, eta=ETA)
    w0 = {n: p.detach().clone() for n, p in c.build_model().named_parameters()}
    return c, prove_step(c, c.build_model(), w0, make_records((12, 7)))


def _llama_check_5(c, out, monkeypatch, *, batched, judge=True, perturb=None):
    """Checks 2 and 5 on the tiny Llama step; ``batched=False`` measures one product at a time,
    as check 5 did before batching. ``perturb`` maps a product name to a leaf transform."""
    store = InMemoryStore.from_step(c, out, copy=True)
    for name, fn in (perturb or {}).items():
        m = c.m_of(name)
        perturb_leaf(c, store, c.product_index(m), fn(out.products[m - 1]))
    ctx = StepContext.for_computation(
        c, step=1, indices=tuple(range(c.n_s)), h_D=bytes(32), n_records=c.n_s,
        prev_w_hashes=(), chain_check_id="0", k=K, judge=judge)
    assert check_2_commitment(store, c, ctx, Bands.provisional()) is None
    with monkeypatch.context() as mp:
        if not batched:
            mp.setattr(checks, "_member_runs", lambda products: ((p,) for p in products))
        rej = check_5_matmuls(store, c, ctx, Bands.provisional())
    return rej, ctx.stats.products


def _bumped(t):
    t = t.clone()
    t.view(-1)[-1] += 1.0
    return t


def test_member_runs_cover_canonical_order(llama_out):
    c, _ = llama_out
    runs = list(checks._member_runs(c.products))
    assert [s for run in runs for s in run] == list(c.products)
    long = [run for run in runs if len(run) > 1]
    # Per layer: S and O (forward) are one run, and dA, dV, dQ, dK (backward) another.
    assert [len(run) for run in long] == [2 * c.n_s * c.n_h] * c.L + [4 * c.n_s * c.n_h] * c.L
    assert all(len({(s.layer, s.member is None) for s in run}) == 1 for run in runs)


def test_batched_check_5_matches_one_at_a_time(llama_out, monkeypatch):
    c, out = llama_out
    rej_b, stats_b = _llama_check_5(c, out, monkeypatch, batched=True)
    rej_s, stats_s = _llama_check_5(c, out, monkeypatch, batched=False)
    assert rej_b is None and rej_s is None
    assert [(x.m, x.name, x.cls) for x in stats_b] == [(x.m, x.name, x.cls) for x in stats_s]
    assert [x.m for x in stats_b] == list(range(1, c.M + 1))
    for xb, xs in zip(stats_b, stats_s):
        assert xb.kappa == pytest.approx(xs.kappa, rel=1e-4)
        assert xb.normalized == pytest.approx(xs.normalized, rel=1e-4, abs=1e-4)


@pytest.mark.parametrize("name", ["L2.S[1,1]", "L1.O[0,2]", "L1.dV[0,3]", "L2.dQ[1,0]"])
def test_forged_member_mid_run_rejected_at_that_member(llama_out, monkeypatch, name):
    c, out = llama_out
    got = [_llama_check_5(c, out, monkeypatch, batched=b, perturb={name: _bumped})
           for b in (True, False)]
    (rej_b, stats_b), (rej_s, stats_s) = got
    _expect(rej_b, 1, "5")
    assert rej_b == rej_s and f"P_{c.m_of(name)} ({name}), challenge j=" in rej_b.detail
    assert [x.name for x in stats_b] == [x.name for x in stats_s]
    assert stats_b[-1].name == name and len(stats_b) == c.m_of(name)


@pytest.mark.parametrize("judge", [True, False])
def test_nan_in_one_member_rejects(llama_out, monkeypatch, judge):
    _, out = llama_out
    c = _LlamaWith(tiny_config(), n_s=2, n=12, eta=ETA)
    name = "L1.dK[1,2]"

    def nan_operand(c, m, a, b):
        if m != c.m_of(name):
            return a, b
        a = a.contiguous().clone()
        a.view(-1)[0] = float("nan")
        return a, b

    monkeypatch.setattr(_LlamaWith, "operands_hook", nan_operand)
    got = [_llama_check_5(c, out, monkeypatch, batched=b, judge=judge) for b in (True, False)]
    (rej_b, stats_b), (rej_s, _) = got
    _expect(rej_b, 1, "5")
    assert rej_b == rej_s and f"P_{c.m_of(name)} ({name}): non-finite ν" in rej_b.detail
    assert stats_b[-1].name == name and math.isnan(stats_b[-1].kappa)


def test_serving_error_waits_for_earlier_members(llama_out, monkeypatch):
    """An error serving a later member of a run is raised only once the members before it
    pass; a forged earlier member rejects first, as one product at a time."""
    _, out = llama_out
    c = _LlamaWith(tiny_config(), n_s=2, n=12, eta=ETA)
    forged, broken = "L1.dA[0,1]", "L1.dV[1,3]"

    def fail(c, m, a, b):
        if m == c.m_of(broken):
            raise RuntimeError("replay bug")
        return a, b

    monkeypatch.setattr(_LlamaWith, "operands_hook", fail)
    for batched in (True, False):
        with pytest.raises(RuntimeError, match="replay bug"):
            _llama_check_5(c, out, monkeypatch, batched=batched)
    rej_b, _ = _llama_check_5(c, out, monkeypatch, batched=True, perturb={forged: _bumped})
    rej_s, _ = _llama_check_5(c, out, monkeypatch, batched=False, perturb={forged: _bumped})
    _expect(rej_b, 1, "5")
    assert rej_b == rej_s and f"({forged})" in rej_b.detail
