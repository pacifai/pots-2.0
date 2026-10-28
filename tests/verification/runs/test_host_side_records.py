"""FS50t: the records between steps (fix B, FS52) change no value.

Written from ``docs/verification/HOST_SIDE_PLAN_DRAFT.md`` §3 and EVALUATION_SPEC §4.1-§4.2
alone. The expected residual archive and step-record fields are computed here from their
definitions with plain Python loops, on a small decoder step (and the MLP), and compared with
what ``MetricsWriter.write_residuals`` and ``step_record`` produce: array by array, values and
dtypes. These tests hold of today's code and must hold of the vectorized one. Cases: an accepted
step, a step rejected at check 5 (the archive holds a subset of products), a step rejected
before check 5 (zero rows), a calibration step (no bands, NaN scores) and the MLP.
"""

import json
import math

import numpy as np
import pytest

from verification.computation.interface import ProductKind
from verification.parameters import LAMBDA, LOG2_G, UNIT_ROUNDOFF
from verification.runs.metrics import MetricsWriter, read_residuals, step_record
from verification.verifier import calibration
from verification.verifier.bands import Bands, product_class
from verification.verifier.context import ProductStat, StepStats, TensorStat
from verification.verifier.driver import Verifier
from verification.verifier.matmul_check.sizing import C_ANTI, bit_budget, f_achieved, product_e_m

from tests.verification.verifier.test_host_side_equivalence import (
    K, bump, make_env, prove, verify)

BANDS = Bands(tau=3.0, kappa_max=1.0e3, kappa_classes={"S": 1.5, "forward:W_hot": 2.0}, tau_w=5.0)


# ---- the expected values, from the definitions -----------------------------------------------


def _div(a: float, b: float) -> float:
    """IEEE ``a / b`` in float64 (a Python float raises on a zero divisor)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(a) / np.float64(b))


def _strs(items: list[str]) -> np.ndarray:
    """``np.array(items, dtype=str)``: the dtype is ``<U`` and the longest string (``<U1``
    for none)."""
    return np.array(items, dtype=np.dtype(f"<U{max([1] + [len(s) for s in items])}"))


_WEIGHT_FREE = {"S": "attention score", "O": "attention value"}
_BY_KIND = {ProductKind.INPUT_GRAD: "input gradient", ProductKind.WEIGHT_GRAD: "weight gradient",
            ProductKind.OPERAND_GRAD: "operand gradient"}


def expected_class(c, spec) -> str:
    """EVALUATION_SPEC §5.7's class of a product."""
    if spec.kind is ProductKind.FORWARD:
        return "forward" if spec.weight is not None else _WEIGHT_FREE[product_class(c, spec)]
    return _BY_KIND[spec.kind]


def expected_archive(run, scenario, step, stats, c, *, bands, k, kappa_guard=True):
    """EVALUATION_SPEC §4.2's arrays for ``stats``, by loops."""
    prods, tens = list(stats.products), list(stats.tensors)
    n_layers = max((s.layer for s in c.products if s.layer is not None), default=0)
    layer, spec_cls = [], []
    for p in prods:
        spec = c.products[p.m - 1]
        layer.append(n_layers if spec.layer is None else int(spec.layer) - 1)
        spec_cls.append(expected_class(c, spec))
    rho_5 = np.empty((len(prods), k), dtype=np.float64)
    rho_k = np.empty(len(prods), dtype=np.float64)
    for i, p in enumerate(prods):
        for j, x in enumerate(p.normalized):
            rho_5[i, j] = math.nan if bands is None else _div(x, bands.tau)
        if bands is None or not kappa_guard:
            rho_k[i] = math.nan
        elif p.nu == 0:
            rho_k[i] = 0.0
        else:
            rho_k[i] = _div(p.nu, bands.kappa_for(p.cls) * p.p_abs1)
    rho_6 = np.array([math.nan if bands is None else _div(s.rho_max, bands.tau_w_for(s.weight))
                      for s in tens], dtype=np.float64)
    return {
        "run": np.array(run), "scenario": np.array(scenario), "step": np.array(step),
        "p_m": np.array([p.m for p in prods], dtype=np.int32),
        "p_name": _strs([p.name for p in prods]), "p_cls": _strs([p.cls for p in prods]),
        "p_spec_class": _strs(spec_cls), "p_layer": np.array(layer, dtype=np.int32),
        "p_kappa": np.array([p.kappa for p in prods], dtype=np.float64),
        "p_q": np.array([p.q for p in prods], dtype=np.int64),
        "p_norm": np.array([p.p_norm for p in prods], dtype=np.float64),
        "p_nu": np.array([p.nu for p in prods], dtype=np.float64),
        "p_abs1": np.array([p.p_abs1 for p in prods], dtype=np.float64),
        "p_u": np.array([p.u for p in prods], dtype=np.float64),
        "p_normalized": np.array([p.normalized for p in prods],
                                 dtype=np.float64).reshape(len(prods), k),
        "w_check": _strs([s.check_id for s in tens]), "w_name": _strs([s.weight for s in tens]),
        "w_rho": np.array([s.rho_max for s in tens], dtype=np.float64),
        "band_source": np.array("" if bands is None else str(bands.source)),
        "rho_5": rho_5, "rho_kappa": rho_k, "rho_6": rho_6,
    }


def same_array(key, got, want):
    assert got.dtype == want.dtype, (key, got.dtype, want.dtype)
    assert got.shape == want.shape, (key, got.shape, want.shape)
    if got.dtype.kind == "f":  # bit for bit: NaN equals NaN, 0.0 differs from -0.0
        assert np.array_equal(got, want, equal_nan=True), key
        assert np.array_equal(np.signbit(got), np.signbit(want)), key
    else:
        assert np.array_equal(got, want), key


def write_and_load(tmp_path, stats, c, *, bands, k=K, kappa_guard=True, scenario="honest",
                   step=3):
    """``(archive, expected)``: the archive ``write_residuals`` made (after ``close``, which the
    background writer joins), and the loop-built one."""
    w = MetricsWriter(tmp_path, "run")
    path = w.write_residuals(scenario, step, stats, c, bands=bands, k=k, kappa_guard=kappa_guard)
    w.close()
    with np.load(path, allow_pickle=False) as z:
        got = {key: z[key] for key in z.files}
    return got, expected_archive("run", scenario, step, stats, c, bands=bands, k=k,
                                 kappa_guard=kappa_guard)


def check_archive(got, want):
    assert sorted(got) == sorted(want)
    for key in want:
        same_array(key, got[key], want[key])


# ---- the steps ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env():
    return make_env(("qwen2", False))  # biases and an untied head: the widest set of names


@pytest.fixture(scope="module")
def accepted(env):
    v, rejs = verify(env, prove(env, 1))
    assert rejs == [None]
    return v


@pytest.fixture(scope="module", params=["L1.Y_q", "Lambda", "L1.O[1,2]"])
def rejected_at_5(env, request):
    """Step 1 rejected at check 5 with no record pass: a subset of the products."""
    m = env.c.m_of(request.param)
    v, rejs = verify(env, prove(env, 1, perturb={1: {m: bump}}), record=False)
    assert (rejs[0].check_id, rejs[0].kind) == ("5", "failed")
    return v


@pytest.fixture(scope="module")
def rejected_at_5_full(env):
    """The same with the record pass: every product, and the tensors of 6a and 6b."""
    m = env.c.m_of("L1.Y_q")
    v, rejs = verify(env, prove(env, 1, perturb={1: {m: bump}}))
    assert rejs[0].check_id == "5" and v.record_pass[1] == "complete"
    return v


@pytest.fixture(scope="module")
def rejected_before_5(env):
    """Step 1 rejected at check 4 (the committed batch's dataset paths are another batch's):
    check 5 never ran, so the step holds no products and no tensors."""
    out, idx = prove(env, 1)[0]
    from verification.transcript.store import InMemoryStore
    wrong = [i for i in range(len(env.D)) if i not in idx][:len(idx)]
    store = InMemoryStore.from_step(env.c, out, dataset_paths=[env.tree.path(i) for i in wrong])
    v = Verifier(env.c, h_D=env.tree.root, n_records=len(env.D), k=K, n_steps=1,
                 bands=Bands.provisional(), allow_provisional=True, w0=env.w0, seed=1)
    assert v.start_run(env.D) is None
    rej = v.verify_step(1, store)
    assert rej is not None and rej.check_id in ("4", "7", "2", "2.root", "6a")
    return v


# ---- the archive ----------------------------------------------------------------------------


def test_the_archive_of_an_accepted_step(tmp_path, env, accepted):
    stats = accepted.stats[1]
    assert len(stats.products) == env.c.M and stats.tensors
    check_archive(*write_and_load(tmp_path, stats, env.c, bands=BANDS))
    # provisional bands, the κ guard off, and another k width follow the same definitions
    check_archive(*write_and_load(tmp_path / "b", stats, env.c, bands=Bands.provisional()))
    check_archive(*write_and_load(tmp_path / "g", stats, env.c, bands=BANDS, kappa_guard=False))


def test_the_archive_of_a_step_rejected_at_check_5_holds_a_subset(tmp_path, env, rejected_at_5):
    stats = rejected_at_5.stats[1]
    assert 0 < len(stats.products) < env.c.M  # a subset
    got, want = write_and_load(tmp_path, stats, env.c, bands=BANDS)
    check_archive(got, want)
    # the string columns take the dtype of this subset's own longest string
    assert got["p_name"].dtype == want["p_name"].dtype


def test_the_archive_after_the_record_pass(tmp_path, env, rejected_at_5_full):
    stats = rejected_at_5_full.stats[1]
    assert len(stats.products) == env.c.M
    check_archive(*write_and_load(tmp_path, stats, env.c, bands=BANDS))


def test_the_archive_of_a_step_rejected_before_check_5_has_zero_rows(tmp_path, env,
                                                                     rejected_before_5):
    stats = rejected_before_5.stats[1]
    assert not stats.products
    got, want = write_and_load(tmp_path, stats, env.c, bands=BANDS)
    check_archive(got, want)
    assert got["p_normalized"].shape == (0, K) and got["rho_5"].shape == (0, K)
    assert got["p_m"].shape == got["p_layer"].shape == got["rho_kappa"].shape == (0,)


def test_a_calibration_step_has_nan_scores(tmp_path, env, accepted):
    """No bands (a calibrating verifier's): every score is NaN, the raw columns as measured."""
    stats = accepted.stats[1]
    got, want = write_and_load(tmp_path, stats, env.c, bands=None)
    check_archive(got, want)
    assert np.isnan(got["rho_5"]).all() and np.isnan(got["rho_kappa"]).all()
    assert np.isnan(got["rho_6"]).all() and str(got["band_source"]) == ""


def test_the_mlp_archive(tmp_path):
    from verification.computation.instances.mlp import MLPComputation, init_weights, synthetic_dataset
    from verification.commitment.leaves import dataset_tree
    from verification.prover.step import prove_step
    from setup.data import schedule
    from verification.transcript.store import InMemoryStore
    c = MLPComputation((16, 32, 32, 8), n_s=4, eta=1e-3)
    D = synthetic_dataset(c.widths, 40, seed=0)
    tree, w0 = dataset_tree(c, D), init_weights(c.widths, seed=0)
    idx = schedule(1, c.n_s, len(D), 1)
    out = prove_step(c, c.build_model(), w0, [D[i] for i in idx])
    v = Verifier(c, h_D=tree.root, n_records=len(D), k=K, n_steps=1, bands=Bands.provisional(),
                 allow_provisional=True, w0=w0, seed=1)
    assert v.start_run(D) is None
    store = InMemoryStore.from_step(c, out, dataset_paths=[tree.path(i) for i in idx])
    assert v.verify_step(1, store) is None
    stats = v.stats[1]
    assert len(stats.products) == c.M
    check_archive(*write_and_load(tmp_path, stats, c, bands=BANDS))


def test_the_archive_reads_back_as_the_steps_stats(tmp_path, env, accepted):
    stats = accepted.stats[1]
    w = MetricsWriter(tmp_path, "run")
    w.write_residuals("honest", 1, stats, env.c, bands=BANDS, k=K)
    w.close()
    back = read_residuals(tmp_path)["honest"][1]
    assert list(back.products) == list(stats.products) and back.tensors == stats.tensors
    assert [(p.q, p.p_norm, p.nu, p.p_abs1, p.u) for p in back.products] == \
        [(p.q, p.p_norm, p.nu, p.p_abs1, p.u) for p in stats.products]


def test_several_steps_in_a_row_write_every_archive(tmp_path, env, accepted, rejected_at_5):
    """A background writer holds at most one archive in flight; none may be lost or mixed."""
    w = MetricsWriter(tmp_path, "run")
    sources = [accepted.stats[1], rejected_at_5.stats[1], accepted.stats[1]]
    for t, st in enumerate(sources, 1):
        w.write_residuals("honest", t, st, env.c, bands=BANDS, k=K)
    w.close()
    for t, st in enumerate(sources, 1):
        with np.load(tmp_path / "residuals" / "honest" / f"step_{t:05d}.npz",
                     allow_pickle=False) as z:
            got = {key: z[key] for key in z.files}
        check_archive(got, expected_archive("run", "honest", t, st, env.c, bands=BANDS, k=K))


def test_a_failed_archive_surfaces_and_leaves_no_run_end(tmp_path, env, accepted):
    """The archive cannot be written (its path is a directory): the error surfaces before the
    run ends, and a run with a failed archive has no ``run_end``."""
    (tmp_path / "residuals" / "honest" / "step_00001.npz").mkdir(parents=True)
    # a run's entry point: the error is raised by the write, the next write or the close that
    # ends the ``with``, and the close then writes no run_end
    with pytest.raises(Exception):  # noqa: B017 (the type is the writer's: OSError or a wrap)
        with MetricsWriter(tmp_path, "run") as w:
            w.write_residuals("honest", 1, accepted.stats[1], env.c, bands=BANDS, k=K)
    kinds = [json.loads(line)["record"]
             for line in (tmp_path / "records.jsonl").read_text().splitlines()]
    assert "run_end" not in kinds


# ---- the step record -------------------------------------------------------------------------


def expected_floor(stats, *, tau, k, N, eps_in, eps_acc):
    """The step's largest ``f_achieved`` and its product, by the loop (first maximum wins)."""
    best = None
    for p in stats.products:
        if not (p.q > 0 and math.isfinite(p.p_norm) and p.p_norm != 0 and not math.isnan(p.u)):
            continue
        f = f_achieved(C_ANTI, tau, product_e_m(p.q, eps_in, eps_acc, p.p_norm, p.u), N, k)
        if best is None or f > best[0]:
            best = (f, p.name)
    return best


def check_record(v, t, rej):
    rec = step_record("run", "honest", t, v, rej)
    stats = v.stats[t]
    c, bands = v.c, v.bands
    want = expected_floor(
        stats, tau=float(bands.tau), k=K, eps_in=UNIT_ROUNDOFF[c.operand_dtype],
        eps_acc=UNIT_ROUNDOFF[c.accumulator_dtype],
        N=bit_budget(LAMBDA, v.n_steps, c.M, LOG2_G))
    if want is None:
        assert rec["f_achieved_max"] is None and rec["f_achieved_at"] is None
    else:
        assert rec["f_achieved_max"] == want[0] and rec["f_achieved_at"] == want[1]
    arch = expected_archive("run", "honest", t, stats, c, bands=bands, k=K)

    def top(x):  # the largest score, a NaN as +inf; failing: not (score <= 1)
        x = np.where(np.isnan(x), np.inf, x).ravel()
        return (float(x.max()) if x.size else None), int(np.count_nonzero(~(x <= 1.0)))

    fam = {"rho_5": top(arch["rho_5"]), "rho_kappa": top(arch["rho_kappa"]),
           "rho_6": top(arch["rho_6"])}
    for name, (mx, nf) in fam.items():
        assert rec[name] == mx, name
        assert rec["n_failing"][name] == (nf if mx is not None else None), name
    return rec


def test_the_step_record_of_an_accepted_step(accepted):
    rec = check_record(accepted, 1, None)
    assert rec["verdict"] == "accept" and rec["f_achieved_at"] is not None


def test_the_step_record_of_a_step_rejected_at_check_5(rejected_at_5, rejected_at_5_full):
    for v in (rejected_at_5, rejected_at_5_full):
        rec = check_record(v, 1, v.rejection)
        assert rec["verdict"] == "reject" and rec["first_failing_check"] == "5"


def test_the_step_record_of_a_step_rejected_before_check_5(rejected_before_5):
    rec = check_record(rejected_before_5, 1, rejected_before_5.rejection)
    assert rec["f_achieved_max"] is None and rec["rho_5"] is None and rec["rho_kappa"] is None


# ---- f_achieved_max on edge cases -------------------------------------------------------------

EPS_IN, EPS_ACC, TAU, N = 2.0 ** -24, 2.0 ** -24, 3.0, 40.0
INF, NAN = math.inf, math.nan


def _prod(m, name, q, p_norm, u):
    return ProductStat(m, name, "forward:W", 1.0, (0.1,) * 3, q=q, p_norm=p_norm, nu=0.0,
                       p_abs1=1.0, u=u)


EDGE = {
    "plain": [_prod(1, "a", 8, 2.0, 3.0), _prod(2, "b", 64, 1.5, 4.0), _prod(3, "c", 16, 9.0, 1.0)],
    "ties": [_prod(1, "a", 8, 2.0, 3.0), _prod(2, "b", 8, 2.0, 3.0), _prod(3, "c", 8, 2.0, 3.0)],
    "tie at the end": [_prod(1, "a", 4, 2.0, 3.0), _prod(2, "b", 8, 2.0, 3.0),
                       _prod(3, "c", 8, 2.0, 3.0)],
    "zero norm": [_prod(1, "a", 8, 0.0, 3.0), _prod(2, "b", 4, 1.0, 1.0)],
    "only zero norm": [_prod(1, "a", 8, 0.0, 3.0), _prod(2, "b", 4, 0.0, 0.0)],
    "nan u": [_prod(1, "a", 8, 1.0, NAN), _prod(2, "b", 4, 1.0, 1.0)],
    "inf u wins": [_prod(1, "a", 8, 1.0, 3.0), _prod(2, "b", 4, 1.0, INF), _prod(3, "c", 4, 1.0, 5.0)],
    "inf u twice": [_prod(1, "a", 8, 1.0, INF), _prod(2, "b", 4, 1.0, INF)],
    "inf norm": [_prod(1, "a", 8, INF, 1.0), _prod(2, "b", 4, 1.0, 1.0)],
    "nan norm": [_prod(1, "a", 8, NAN, 1.0), _prod(2, "b", 4, 1.0, 1.0)],
    "q unknown": [_prod(1, "a", 0, 1.0, 1.0), _prod(2, "b", 4, 1.0, 1.0)],
    "u below norm": [_prod(1, "a", 8, 4.0, 1.0), _prod(2, "b", 8, 1.0, 0.5)],
    "zero u": [_prod(1, "a", 8, 1.0, 0.0), _prod(2, "b", 9, 1.0, 0.0)],
    "tiny and huge": [_prod(1, "a", 8, 1e-300, 1e-300), _prod(2, "b", 8, 1e300, 1e300),
                      _prod(3, "c", 10**6, 1.0, 1e12)],
    "none": [],
}


@pytest.mark.parametrize("case", list(EDGE))
def test_step_floor_equals_the_loop(case):
    stats = StepStats(products=list(EDGE[case]), tensors=[TensorStat("w", "6a", 0.5)])
    got = calibration.step_floor(stats, tau=TAU, k=3, N=N, eps_in=EPS_IN, eps_acc=EPS_ACC)
    want = expected_floor(stats, tau=TAU, k=3, N=N, eps_in=EPS_IN, eps_acc=EPS_ACC)
    if want is None:
        assert got is None
    else:
        assert got is not None and got.name == want[1]
        assert np.float64(got.f_achieved).tobytes() == np.float64(want[0]).tobytes()


def test_step_floor_on_many_random_products():
    rng = np.random.default_rng(5)
    prods = []
    for i in range(500):
        norm = float(rng.choice([0.0, 1e-3, 1.0, 7.5, 3e4, INF, NAN]))
        u = float(rng.choice([0.0, 1e-3, 1.0, 7.5, 3e4, 1e9, INF, NAN]))
        prods.append(_prod(i + 1, f"p{i}", int(rng.choice([0, 1, 2, 8, 100, 5000])), norm, u))
    for k in (1, 9, 21):
        stats = StepStats(products=prods)
        got = calibration.step_floor(stats, tau=TAU, k=k, N=N, eps_in=EPS_IN, eps_acc=EPS_ACC)
        want = expected_floor(stats, tau=TAU, k=k, N=N, eps_in=EPS_IN, eps_acc=EPS_ACC)
        assert (got is None) == (want is None)
        if want is not None:
            assert got.name == want[1] and got.f_achieved == want[0]
