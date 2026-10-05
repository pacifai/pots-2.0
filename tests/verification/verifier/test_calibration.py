"""C1's calibration (A11): the fit, the realized floor, the band file, and an MLP run end to end."""

import json
import math

import pytest
import torch

from setup.data import schedule
from verification.commitment.leaves import dataset_tree
from verification.computation.instances.mlp import (
    MLPComputation,
    init_weights,
    synthetic_dataset,
)
from verification.parameters import UNIT_ROUNDOFF
from verification.prover.step import prove_step
from verification.transcript.store import InMemoryStore
from verification.verifier.bands import TAU_W0, Bands
from verification.verifier.calibration import (
    KAPPA_MARGIN,
    assert_same_band_source,
    band_file_bytes,
    fit,
    load_bands,
    realized_floor,
    write_band_file,
)
from verification.verifier.context import ProductStat, StepStats, TensorStat
from verification.verifier.driver import Verifier
from verification.verifier.matmul_check.sizing import C_ANTI, Z, bit_budget, e_m, f_achieved

K = 2
EPS = 2.0 ** -24


def _p(m, name, cls, kappa, res, q=8, p_norm=1.0, nu=1.0, p_abs1=1.0):
    return ProductStat(m, name, cls, kappa, res, q=q, p_norm=p_norm, nu=nu, p_abs1=p_abs1)


def stats():
    s1 = StepStats(
        products=[_p(1, "L1.Y_q", "Y_q", 2.0, (1.0, 1.0)),
                  _p(2, "Lambda", "Lambda", 5.0, (3.0, 1.0), q=576, p_norm=10.0)],
        tensors=[TensorStat("w.a", "6a", 1.5), TensorStat("w.b", "6b", 2.5)])
    s2 = StepStats(
        products=[_p(1, "L1.Y_q", "Y_q", 3.0, (0.5, 0.5)),
                  _p(2, "Lambda", "Lambda", 4.0, (3.5, 2.0), q=576, p_norm=20.0)],
        tensors=[TensorStat("w.a", "6a", 1.9), TensorStat("w.b", "6b", 0.0)])
    return {2: s2, 1: s1}


def test_fit_takes_s_h_from_the_largest_class_rms():
    cal = fit(stats(), k=K)
    lam = math.sqrt((9 + 1 + 12.25 + 4) / 4)
    assert cal.steps == (1, 2) and cal.k == K and cal.z == Z
    assert math.isclose(cal.s_h, lam) and cal.s_h_class == "Lambda"
    assert math.isclose(cal.tau, Z * lam)
    assert (cal.global_max, cal.global_max_name, cal.global_max_step) == (3.5, "Lambda", 2)
    assert math.isclose(cal.ratio, 3.5 / lam)
    assert cal.guard_ok and math.isclose(cal.guard_limit, cal.tau / 2)


def test_kappa_and_tau_w_per_key():
    cal = fit(stats(), k=K)
    assert cal.kappa_max == {"Y_q": KAPPA_MARGIN * 3.0, "Lambda": KAPPA_MARGIN * 5.0}
    assert cal.rho_max == {"w.a": 1.9, "w.b": 2.5}
    assert cal.tau_w == {"w.a": TAU_W0, "w.b": 5.0}  # max(4, 2·ρ_max)


def test_the_concentration_guard_reports_an_outlier():
    st = stats()
    # One residual far out among many ordinary ones: the RMS stays low, max > τ/2.
    st[1].products += [_p(10 + i, f"L{i}.Y_k", "Y_k", 1.0, (1.0, 1.0)) for i in range(100)]
    st[1].products.append(_p(3, "L2.Y_k", "Y_k", 1.0, (0.0, 30.0)))
    cal = fit(st, k=K)
    assert cal.s_h_class == "Lambda" and cal.global_max_name == "L2.Y_k"
    assert cal.global_max == 30.0 and not cal.guard_ok
    assert cal.tau == Z * cal.s_h  # z is never raised to cover it


def test_fit_refuses_broken_numbers():
    with pytest.raises(ValueError, match="no calibration steps"):
        fit({}, k=K)
    st = stats()
    st[1].products[0] = _p(1, "L1.Y_q", "Y_q", 2.0, (1.0, math.nan))
    with pytest.raises(ValueError, match="non-finite"):
        fit(st, k=K)
    with pytest.raises(ValueError, match="residuals, k = 3"):
        fit(stats(), k=3)
    st = stats()
    st[2].tensors.clear()
    with pytest.raises(ValueError, match="no check-5 or check-6"):
        fit(st, k=K)
    zero = {1: StepStats(products=[_p(1, "a", "c", 1.0, (0.0, 0.0))],
                         tensors=[TensorStat("w", "6a", 0.0)])}
    with pytest.raises(ValueError, match="s_h = 0"):
        fit(zero, k=K)


def test_realized_floor_by_contracted_dimension():
    cal = fit(stats(), k=K)
    N = bit_budget(25, 10, 7113, 52)
    f = realized_floor(stats(), tau=cal.tau, k=K, N=N, eps_in=EPS, eps_acc=EPS)
    small = f_achieved(C_ANTI, cal.tau, e_m(8, EPS, EPS), N, K)
    big = f_achieved(C_ANTI, cal.tau, e_m(576, EPS, EPS), N, K)
    assert f.by_q == {8: small, 576: big} and big > small
    assert f.count == 4 and f.ratio_max == big and f.ratio_max_name == "Lambda"
    assert f.ratio_min == small and math.isclose(f.phi_max, big * 20.0)
    assert f.over_target == sum(r > 1 for r in (small, small, big, big))
    assert f.zero_row_sum == ()


def test_realized_floor_flags_a_zero_row_sum_and_needs_the_scales():
    st = stats()
    st[1].products.append(_p(3, "L1.dA[0,0]", "dA", 1.0, (0.1, 0.1), nu=1e-9, p_abs1=0.0))
    f = realized_floor(st, tau=30.0, k=9, N=93.0, eps_in=EPS, eps_acc=EPS)
    assert f.zero_row_sum == ("L1.dA[0,0] (step 1)",)
    old = {1: StepStats(products=[ProductStat(1, "a", "c", 1.0, (1.0, 1.0))])}
    with pytest.raises(ValueError, match="no recorded q"):
        realized_floor(old, tau=30.0, k=9, N=93.0, eps_in=EPS, eps_acc=EPS)


def test_band_file_round_trip_and_hash(tmp_path):
    cal = fit(stats(), k=K)
    data = band_file_bytes(cal, {"extra": {"x": 1}})
    assert data == band_file_bytes(cal, {"extra": {"x": 1}})  # deterministic bytes
    path = tmp_path / "sub" / "bands.json"
    written = write_band_file(path, data)
    assert path.read_bytes() == data and not path.with_name("bands.json.tmp").exists()
    bands = load_bands(path, k=K)
    assert bands == written and bands.source == written.source != "provisional"
    assert bands.tau == cal.tau and bands.kappa_classes == cal.kappa_max
    assert bands.tau_w == TAU_W0 and bands.tau_w_tensors == cal.tau_w
    assert bands.kappa_max == max(cal.kappa_max.values())
    assert bands.stats["k"] == K and bands.stats["extra"] == {"x": 1}
    assert bands.stats["concentration_guard"]["ok"] is True
    assert json.loads(data)  # plain JSON
    with pytest.raises(ValueError, match="calibrated at k = 2"):
        load_bands(path, k=7)
    with pytest.raises(FileNotFoundError, match="calibrate"):
        load_bands(tmp_path / "missing.json")


def test_assert_same_band_source():
    assert assert_same_band_source(["ab", "ab"]) == "ab"
    for bad in ([], ["ab", None], ["provisional"], ["ab", "cd"]):
        with pytest.raises(AssertionError):
            assert_same_band_source(bad)


# ---- end to end on the MLP --------------------------------------------------------------

T = 3
N_RECORDS = 40


@pytest.fixture(scope="module")
def mlp():
    c = MLPComputation((16, 32, 32, 8), n_s=4, eta=1e-3)
    D = synthetic_dataset(c.widths, N_RECORDS, seed=0)
    tree = dataset_tree(c, D)
    w0 = init_weights(c.widths, seed=0)
    stores, w = [], w0
    for t in range(1, T + 1):
        idx = schedule(t, c.n_s, len(D))
        out = prove_step(c, c.build_model(), w, [D[i] for i in idx])
        stores.append(InMemoryStore.from_step(c, out, dataset_paths=[tree.path(i) for i in idx]))
        w = out.w_next
    return c, D, tree, w0, stores, w


def _run(mlp, **kw):
    c, D, tree, w0, stores, final = mlp
    v = Verifier(c, h_D=tree.root, n_records=len(D), k=7, n_steps=T, w0=w0, **kw)
    assert v.start_run(D) is None
    rejections = [v.verify_step(t, s) for t, s in enumerate(stores, start=1)]
    return v, rejections, final


def test_calibrate_freeze_and_judge_with_the_band_file(mlp, tmp_path):
    v, rej, final = _run(mlp, bands=None, calibrate=True)
    assert rej == [None] * T
    cal = fit(v.stats, k=7)
    assert cal.guard_ok and set(cal.kappa_max) == {p for p in v.stats[1].by_class()}
    floor = realized_floor(v.stats, tau=cal.tau, k=7, N=90.0,
                           eps_in=UNIT_ROUNDOFF[torch.float32], eps_acc=UNIT_ROUNDOFF[torch.float32])
    assert floor.count == T * mlp[0].M and floor.zero_row_sum == ()
    bands = write_band_file(tmp_path / "bands.json", band_file_bytes(cal))
    assert v.freeze(bands) is None
    verdict = v.end_run(final)
    assert verdict.accepted and verdict.band_source == bands.source
    loaded = load_bands(tmp_path / "bands.json", k=7)
    sources = [verdict.band_source]
    for guard in (True, False):
        j, rej, _ = _run(mlp, bands=loaded, kappa_guard=guard)
        assert rej == [None] * T
        sources.append(j.end_run(final).band_source)
        if not guard:  # test 1 skipped: κ unknown, test 2's numbers unchanged
            assert all(math.isnan(p.kappa) for p in j.stats[1].products)
            assert [p.normalized for p in j.stats[1].products] == \
                [p.normalized for p in v.stats[1].products]
    assert assert_same_band_source(sources) == bands.source


def test_calibration_needs_the_kappa_guard(mlp):
    c, D, tree, w0, _, _ = mlp
    with pytest.raises(ValueError, match="κ guard"):
        Verifier(c, h_D=tree.root, n_records=len(D), k=7, n_steps=T, w0=w0, bands=None,
                 calibrate=True, kappa_guard=False)


def test_the_bands_judge_what_they_were_fitted_on(mlp, tmp_path):
    """Bands fitted tighter than the honest numbers reject at freeze: the rejudge is live."""
    v, _, _ = _run(mlp, bands=None, calibrate=True)
    cal = fit(v.stats, k=7, z=0.5)  # τ = s_h/2, below the largest honest residual
    bands = Bands.from_json(band_file_bytes(cal))
    rej = v.freeze(bands)
    assert rej is not None and rej.check_id == "5"
