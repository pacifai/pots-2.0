"""FS12's ``k`` run on the MLP's stored step 4, and the sizing arithmetic at both scales."""

import math

import pytest

from tests.verification.runs.test_cheats import K, QUIET, SEED, env, kept, stored  # noqa: F401
from verification.runs import flipped_matmul_sweep as fs
from verification.runs import k_tunability as kt
from verification.runs import run_verified as rv
from verification.runs.metrics import MetricsWriter, read_records
from verification.verifier.matmul_check.sizing import C_ANTI

# The test-scale band file's sizing (cc7a9bca…): τ, e_max, N and the recorded floor at k = 9.
TAU, E_MAX, N_TEST, F_ACH_9 = 38.03910546545508, 1.3298792656134927e-05, 93.11817054309053, \
    0.5377652207203429


def test_covering_k_gives_the_spec_k_top_at_both_scales():
    """§5.8's 44 for Llama-3.2-1B (floor0 0.0362, N 98.91, B 128) and 18 at test scale, where
    k = 9 covers one record of four and k = 8 doesn't; a coherence of 1.07 needs 46."""
    assert kt.covering_k(0.0362, 98.91, C=1.0, B=128) == 44
    assert kt.covering_k(0.0362, 98.91, C=1.07, B=128) == 46
    floor0 = C_ANTI * TAU * E_MAX
    assert kt.covering_k(floor0, N_TEST, C=1.0, B=4) == 9
    assert kt.k_top(9, kt.covering_k(floor0, N_TEST, C=1.0, B=4)) == 18
    f = kt.KFloor(tau=TAU, e_max=E_MAX, N=N_TEST, b0=11.2414, lam=25, k_conf=9, source="t")
    assert math.isclose(f.f_ach(9), F_ACH_9, rel_tol=1e-12)
    assert kt.guaranteed_rate(f.f_ach(9), C=1.0, B=4) * 4 <= 1 < (
        kt.guaranteed_rate(f.f_ach(8), C=1.0, B=4) * 4)
    with pytest.raises(ValueError, match="no k"):
        kt.covering_k(1.0, N_TEST, C=1.0, B=4, k_max=20)


def test_linear_fit_recovers_a_line():
    fit = kt.linear_fit([1, 2, 3, 4], [1.5, 2.0, 2.5, 3.0])
    assert math.isclose(fit["slope"], 0.5) and math.isclose(fit["intercept"], 1.0)
    assert math.isclose(fit["r2"], 1.0) and fit["max_rel_resid"] < 1e-12


def test_run_k_on_the_stored_step(env, stored, tmp_path):  # noqa: F811
    """The configured k reproduces the archive; κ and the guard's norms are bit-equal at every
    k; the measure's FLOPs are ``S·(k+1) + S_U`` (``S_U`` the term scale's, F18), the glue's
    constant, the hashes ``M·k``. The MLP's stored run has no time, count or step records, which
    the run tolerates."""
    c = env.c
    ks = rv.load_kept_step(c, stored, step=fs.SWEEP_STEP)
    with MetricsWriter(tmp_path, kt.RUN_NAME, band_file_hash=env.bands.source) as w:
        r = kt.run_k(c, ks, env.bands, seed=SEED, h_D=env.h_D, n_records=len(env.D),
                     k_conf=K, T=10, top=K + 1, repeats=2, writer=w, out=QUIET)
    sm = r.summary
    assert r.passed and sm["full_step_equal"] and sm["archive_equal"] and sm["repeatable"]
    assert sm["rho_5_honest"] is None and sm["accepted"] and sm["kappa_equal"]
    assert [p["k"] for p in r.points] == list(range(1, K + 2))
    S, S_U = kt.measure_flops_per_column(c), kt.measure_flops_term_scale(c)
    for p in r.points:
        assert p["flops_measure"] == S * (p["k"] + 1) + S_U == p["flops_measure_pred"]
        assert p["hash_calls"] == c.M * p["k"] == p["hash_calls_pred"]
        assert p["hash_in_bytes"] == p["hash_in_bytes_pred"]
        assert p["hash_out_bytes"] == p["hash_out_bytes_pred"]
        assert p["prefix_count"] == c.M * min(p["k"], K) and len(p["times_s"]) == 2
    assert r.points[K - 1]["prefix_equal"] == c.M * K
    assert sm["flops_exact"] and sm["glue_flops_constant"] and sm["hashes_exact"]
    src = r.source
    assert src["k_top"] == K + 1 and src["honest_times_s"] == {} and src["honest_step"] is None
    assert src["root"] == ks.root.hex() and set(src["slot_s"]) == set(kt.DEFAULT_ORDER)
    assert len(read_records(tmp_path, "k_point")) == K + 1
    [rec] = read_records(tmp_path, "k_summary")
    assert rec["passed"] and rec["step"] == fs.SWEEP_STEP
    lines = []
    kt.report(r, out=lines.append)
    assert lines[-1] == "PASS"
    with pytest.raises(ValueError, match="reproduction k"):
        kt.run_k(c, ks, env.bands, seed=SEED, h_D=env.h_D, n_records=len(env.D), k_conf=K,
                 T=10, top=K - 1, out=QUIET)
