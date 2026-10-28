"""FS52: the records between steps, vectorized and off the critical path (HOST_SIDE_PLAN §3).

The archive's arrays equal the pre-FS52 code's (a copy of it is kept here as the reference), the
vectorized ``step_floor`` equals the per-product loop, and a writer-thread error surfaces at
``close`` with no ``run_end``.
"""

import math

import numpy as np
import pytest

from verification.computation.instances.mlp import MLPComputation, init_weights, synthetic_dataset
from verification.runs import metrics
from verification.runs.metrics import (
    MetricsWriter,
    layer_count,
    layer_index,
    read_records,
    rho_arrays,
    spec_class,
)
from verification.runs.mlp_smoke import run_smoke
from verification.runs.scenarios import honest_final
from verification.verifier.bands import product_class, product_classes
from verification.verifier.calibration import _step_floor_loop, step_floor
from verification.verifier.context import ProductStat, StepStats, TensorStat

SEED = 1
K = 7
T = 3


@pytest.fixture(scope="module")
def c():
    return MLPComputation((16, 32, 32, 8), n_s=4, eta=1e-3)


@pytest.fixture(scope="module")
def smoke(c, tmp_path_factory):
    """The MLP smoke run's verifiers, with the bands the honest scenario ran under."""
    D = synthetic_dataset(c.widths, 40, seed=0)
    w0 = init_weights(c.widths, seed=0)
    with MetricsWriter(tmp_path_factory.mktemp("fast"), "t", model="mlp", corpus="synthetic",
                       seed=0, config={}) as mw:
        results = run_smoke(c, D, w0, T=T, k=K, out=lambda s: None, metrics=mw, seed=SEED)
    honest = next(r for r in results if r.scenario.name == "honest")
    return honest.verifier


def _old_residual_arrays(run, scenario, step, stats, c, *, bands=None, k=None, kappa_guard=True):
    """``_residual_arrays`` before FS52, kept as the reference."""
    ps = stats.products
    cols = ps.columns() if ps else metrics.ProductStats().columns()
    width = cols.normalized.shape[1] if ps else (k or 0)
    layers, classes = [], []
    if c is not None:
        n_layers = layer_count(c)
        specs = c.products
        table = product_classes(c) or [product_class(c, s) for s in specs]
        ms = cols.m.tolist()
        if ms and not 1 <= min(ms) <= max(ms) <= c.M:
            c.product(min(ms) if min(ms) < 1 else max(ms))
        for m in ms:
            spec = specs[m - 1]
            layers.append(layer_index(spec, n_layers))
            classes.append(spec_class(c, spec, table[m - 1]))
    else:
        layers, classes = [-1] * len(cols), [""] * len(cols)
    rho = rho_arrays(stats, bands, k=k, kappa_guard=kappa_guard)
    return {
        "run": np.array(run), "scenario": np.array(scenario), "step": np.array(step),
        "p_m": cols.m.astype(np.int32),
        "p_name": np.array(cols.name, dtype=str),
        "p_cls": np.array(cols.cls, dtype=str),
        "p_spec_class": np.array(classes, dtype=str),
        "p_layer": np.array(layers, dtype=np.int32),
        "p_kappa": cols.kappa.copy(), "p_q": cols.q.copy(), "p_norm": cols.p_norm.copy(),
        "p_nu": cols.nu.copy(), "p_abs1": cols.p_abs1.copy(), "p_u": cols.u.copy(),
        "p_normalized": cols.normalized.copy().reshape(len(cols), width),
        "w_check": np.array([s.check_id for s in stats.tensors], dtype=str),
        "w_name": np.array([s.weight for s in stats.tensors], dtype=str),
        "w_rho": np.array([s.rho_max for s in stats.tensors], dtype=np.float64),
        "band_source": np.array("" if bands is None else str(bands.source)),
        **rho,
    }


def _same(got, want):
    assert got.keys() == want.keys()
    for name in want:
        g, w = got[name], want[name]
        assert g.dtype == w.dtype and g.shape == w.shape, name
        assert g.tobytes() == w.tobytes(), name


def _written(tmp_path, stats, c, bands, k, guard=True, step=1):
    with MetricsWriter(tmp_path, "r") as mw:
        path = mw.write_residuals("s", step, stats, c, bands=bands, k=k, kappa_guard=guard)
    with np.load(path, allow_pickle=False) as z:
        return {n: z[n] for n in z.files}


def _subset(stats, every):
    out = StepStats()
    out.products.extend(list(stats.products)[::every])
    out.tensors.extend(stats.tensors)
    return out


def test_archive_equals_the_old_code_for_an_accepted_step(smoke, c, tmp_path):
    stats, bands = smoke.stats[2], smoke.bands
    assert len(stats.products) == c.M and stats.tensors
    got = _written(tmp_path, stats, c, bands, K, step=2)
    _same(got, _old_residual_arrays("r", "s", 2, stats, c, bands=bands, k=K))


def test_archive_equals_the_old_code_for_a_subset_of_products(smoke, c, tmp_path):
    stats = _subset(smoke.stats[2], 3)  # a step rejected at check 5 holds some products
    got = _written(tmp_path, stats, c, smoke.bands, K)
    _same(got, _old_residual_arrays("r", "s", 1, stats, c, bands=smoke.bands, k=K))
    assert 0 < len(got["p_m"]) < c.M


def test_archive_equals_the_old_code_with_zero_rows(smoke, c, tmp_path):
    stats = StepStats()  # a step rejected before check 5
    got = _written(tmp_path, stats, c, smoke.bands, K)
    _same(got, _old_residual_arrays("r", "s", 1, stats, c, bands=smoke.bands, k=K))
    assert len(got["p_m"]) == 0 and got["p_normalized"].shape == (0, K)


def test_archive_equals_the_old_code_for_a_calibration_step(smoke, c, tmp_path):
    stats = smoke.stats[1]
    got = _written(tmp_path, stats, c, None, K)  # no bands: every score NaN
    _same(got, _old_residual_arrays("r", "s", 1, stats, c, bands=None, k=K))
    assert np.isnan(got["rho_5"]).all()


def test_archive_equals_the_old_code_without_the_kappa_guard_or_a_computation(smoke, c,
                                                                              tmp_path):
    stats = smoke.stats[2]
    got = _written(tmp_path, stats, c, smoke.bands, K, guard=False)
    _same(got, _old_residual_arrays("r", "s", 1, stats, c, bands=smoke.bands, k=K,
                                    kappa_guard=False))
    got = _written(tmp_path / "nc", stats, None, smoke.bands, K)
    _same(got, _old_residual_arrays("r", "s", 1, stats, None, bands=smoke.bands, k=K))


def test_scores_passed_in_equal_the_ones_it_computes(smoke, c, tmp_path):
    stats, bands = smoke.stats[2], smoke.bands
    rho = rho_arrays(stats, bands, k=K, kappa_guard=True)
    a = metrics._residual_arrays("r", "s", 1, stats, c, bands=bands, k=K, rho=rho)
    b = metrics._residual_arrays("r", "s", 1, stats, c, bands=bands, k=K)
    _same(a, b)
    s1 = metrics.step_scores(stats, bands, k=K, rho=rho)
    assert s1 == metrics.step_scores(stats, bands, k=K)


def test_static_columns_are_cached_per_computation(smoke, c, tmp_path):
    stats = smoke.stats[2]
    _written(tmp_path, stats, c, smoke.bands, K)
    first = metrics._static_columns(c)
    assert metrics._static_columns(c) is first
    other = MLPComputation((16, 32, 32, 8), n_s=4, eta=1e-3)
    assert metrics._static_columns(other) is not first


def test_a_product_with_no_spec_class_raises_as_before(c, tmp_path, monkeypatch):
    def refuse(c_, spec, cls=None):
        if spec.m == 3:
            raise ValueError("no class")
        return "forward"

    monkeypatch.setattr(metrics, "spec_class", refuse)
    metrics._STATIC_COLUMNS.clear()
    ok = StepStats()
    ok.products.append(ProductStat(1, "a", "x", 1.0, (0.1,) * K))
    metrics._layers_and_classes(c, np.array([1, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="no class"):
        metrics._layers_and_classes(c, np.array([1, 3], dtype=np.int64))
    metrics._STATIC_COLUMNS.clear()


# ---- step_floor -------------------------------------------------------------------------


def _floor_stats(rows):
    st = StepStats()
    for i, (q, p_norm, u) in enumerate(rows, 1):
        st.products.append(ProductStat(i, f"p{i}", "cls", 1.0, (0.5,), q=q, p_norm=p_norm, nu=0.1,
                                       p_abs1=1.0, u=u))
    return st


FLOOR = dict(tau=1.7e-3, k=9, N=60.5, eps_in=2.0 ** -8, eps_acc=2.0 ** -24)


def _agree(rows, **over):
    kw = {**FLOOR, **over}
    stats = _floor_stats(rows)
    new, old = step_floor(stats, **kw), _step_floor_loop(stats, **kw)
    if old is None:
        assert new is None
        return None
    assert new is not None and new.name == old.name
    assert np.float64(new.f_achieved).tobytes() == np.float64(old.f_achieved).tobytes() or (
        np.isnan(new.f_achieved) and np.isnan(old.f_achieved))
    return new


def test_step_floor_equals_the_loop_on_random_products():
    rng = np.random.default_rng(0)
    rows = [(int(q), float(p), float(u)) for q, p, u in zip(
        rng.integers(1, 5000, 3000), rng.lognormal(0, 3, 3000), rng.lognormal(1, 3, 3000))]
    _agree(rows)
    _agree(rows, c=0.37)


def test_step_floor_with_zero_norm_nan_and_infinite_u():
    rows = [(8, 0.0, 1.0), (8, 2.0, math.nan), (0, 1.0, 1.0), (8, math.nan, 1.0), (8, 1.5, 3.0)]
    assert _agree(rows).name == "p5"  # only the last product is kept
    assert _agree([(8, 0.0, 1.0), (8, math.inf, 1.0)]) is None
    assert _agree([(8, 1.5, 3.0), (8, 2.0, math.inf), (8, 1.0, 9.0)]).name == "p2"
    assert _agree([(8, 1.5, math.inf), (8, 2.0, math.inf)]).name == "p1"  # inf ties: the first
    assert _agree([(8, 1.5, 3.0)] * 4).name == "p1"  # ties: the first maximum


def test_step_floor_ties_and_the_first_maximum():
    rows = [(16, 1.0, 1.0), (64, 1.0, 1.0), (64, 1.0, 1.0), (4, 1.0, 1.0)]
    assert _agree(rows).name == "p2"
    assert _agree([(16, 1.0, 0.5), (16, 1.0, 0.5)]).name == "p1"  # ratio below 1 clamps to 1


def test_step_floor_with_no_products():
    assert step_floor(StepStats(), **FLOOR) is None
    assert _agree([(8, 0.0, 1.0)]) is None


def test_step_floor_with_non_finite_f():
    _agree([(8, 1.0, 1.0), (8, 1.0, 2.0)], tau=math.inf)
    _agree([(8, 1.0, 1.0), (8, 1.0, 2.0)], tau=0.0)
    _agree([(8, 1.0, math.inf), (8, 1.0, 2.0)], tau=0.0)  # 0 * inf = NaN at the first product
    _agree([(8, 1.0, 1.0), (8, 1.0, math.inf)], tau=0.0)


# ---- the writer thread ------------------------------------------------------------------


def _stats_one():
    st = StepStats()
    st.products.append(ProductStat(1, "p1", "x", 1.0, (0.1,) * K, q=3, p_norm=1.0, nu=0.1,
                                   p_abs1=1.0, u=1.0))
    st.tensors.append(TensorStat("w", "6a", 0.5))
    return st


def test_a_writer_error_surfaces_at_close_with_no_run_end(c, tmp_path, monkeypatch):
    def boom(path, **arrays):
        raise OSError("disk full")

    monkeypatch.setattr(metrics.np, "savez", boom)
    mw = MetricsWriter(tmp_path, "r")
    mw.write_residuals("s", 1, _stats_one(), None, k=K)  # returns: the error is in the thread
    with pytest.raises(OSError, match="disk full"):
        mw.close(completed=True)
    assert read_records(tmp_path, "run_end") == []
    mw.close()  # a second close does nothing


def test_a_writer_error_surfaces_at_the_next_write(c, tmp_path, monkeypatch):
    calls = []
    real = np.savez

    def once(path, **arrays):
        calls.append(path)
        if len(calls) == 1:
            raise OSError("first fails")
        real(path, **arrays)

    monkeypatch.setattr(metrics.np, "savez", once)
    mw = MetricsWriter(tmp_path, "r")
    mw.write_residuals("s", 1, _stats_one(), None, k=K)
    with pytest.raises(OSError, match="first fails"):
        mw.write_residuals("s", 2, _stats_one(), None, k=K)
    mw.write_residuals("s", 3, _stats_one(), None, k=K)  # the error was raised once
    mw.close(completed=True)
    assert len(read_records(tmp_path, "run_end")) == 1
    assert (tmp_path / "residuals" / "s" / "step_00003.npz").exists()


def test_close_waits_for_the_archive_in_flight(tmp_path, monkeypatch):
    import threading
    gate = threading.Event()
    real = np.savez

    def slow(path, **arrays):
        gate.wait(5)
        real(path, **arrays)

    monkeypatch.setattr(metrics.np, "savez", slow)
    mw = MetricsWriter(tmp_path, "r")
    path = mw.write_residuals("s", 1, _stats_one(), None, k=K)
    assert not path.exists()  # still being written
    threading.Timer(0.2, gate.set).start()
    mw.close(completed=True)
    assert path.exists() and len(read_records(tmp_path, "run_end")) == 1
