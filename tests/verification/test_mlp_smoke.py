"""A6 / milestone M1: every smoke scenario ends at its declared outcome (S6b)."""

import pytest
import torch

from verification.helper_runs import mlp_smoke
from verification.helper_runs.mlp_smoke import (
    Expected,
    Scenario,
    honest_final,
    judge,
    run_scenario,
    run_smoke,
    scenarios,
)
from verification.instances.mlp import MLPComputation, init_weights, synthetic_dataset
from verification.loop import ProverFault

K = 7
T = 3
N_RECORDS = 40


@pytest.fixture(scope="module")
def c():
    return MLPComputation((16, 32, 32, 8), n_s=4, eta=1e-3)


@pytest.fixture(scope="module")
def D(c):
    return synthetic_dataset(c.widths, N_RECORDS, seed=0)


@pytest.fixture(scope="module")
def w0(c):
    return init_weights(c.widths, seed=0)


@pytest.fixture(scope="module")
def final(c, D, w0):
    return honest_final(c, D, w0, T)


EXPECTED = {
    "honest": Expected(),
    "flip": Expected(2, "5", "failed"),
    "bad-w-next-ulps": Expected(2, "6a", "failed"),
    "bad-w-next-batch": Expected(2, "6a", "failed"),
    "broken-chain": Expected(2, "7", "failed"),
}


def test_declared_outcomes(c, D):
    assert {s.name: s.expected for s in scenarios(c, D)} == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_scenario_oracle(c, D, w0, final, name):
    s = next(x for x in scenarios(c, D) if x.name == name)
    r = run_scenario(c, D, w0, s, T=T, k=K, final=final)
    assert r.passed, (name, r.loop.rejection)
    assert r.actual == EXPECTED[name]
    v = r.loop.verdict
    if name == "honest":
        assert v.accepted and v.steps_verified == T and len(r.loop.steps) == T
        assert all(torch.equal(r.loop.w_final[n], final[n]) for n in c.weight_names)
    else:
        # Every step before the fault was accepted; the run stopped at the fault.
        assert not v.accepted and len(r.loop.steps) == EXPECTED[name].step
        assert all(s.rejection is None for s in r.loop.steps[:-1])
    if name == "flip":
        assert f"P_{c.m_of('Y_2')} (Y_2)" in v.rejection.detail


def test_oracle_fails_on_a_wrong_declaration(c, D, w0, final):
    """A fault rejected at another check, an honest run declared as rejected, and a fault
    declared at a later step all FAIL (S6b)."""
    flip = next(x for x in scenarios(c, D) if x.name == "flip")
    for s in (Scenario("flip@6a", "", flip.fault, Expected(2, "6a", "failed")),
              Scenario("flip@3", "", flip.fault, Expected(3, "5", "failed")),
              Scenario("flip-malformed", "", flip.fault, Expected(2, "5", "malformed")),
              Scenario("honest-rejected", "", ProverFault(), Expected(1, "5", "failed"))):
        assert not run_scenario(c, D, w0, s, T=T, k=K, final=final).passed, s.name
    flip_ok = run_scenario(c, D, w0, flip, T=T, k=K, final=final)
    assert not judge(Expected(), flip_ok.loop, T)  # a rejected run is not an accept


def test_honest_with_wrong_final_fails(c, D, w0, final):
    honest = next(x for x in scenarios(c, D) if x.name == "honest")
    bad = {**final, c.weight(1): final[c.weight(1)] * 2}
    r = run_scenario(c, D, w0, honest, T=T, k=K, final=bad)
    assert not r.passed and r.loop.rejection.check_id == "8"


def test_run_smoke_reports(c, D, w0):
    lines = []
    results = run_smoke(c, D, w0, T=T, k=K, out=lines.append)
    assert [r.scenario.name for r in results] == list(EXPECTED)
    assert all(r.passed for r in results)
    text = "\n".join(lines)
    assert text.count("PASS") == len(EXPECTED) and "FAIL" not in text
    assert lines[-1] == f"{len(EXPECTED)}/{len(EXPECTED)} scenarios passed"
    with pytest.raises(ValueError, match="T ≥ 2"):
        run_smoke(c, D, w0, T=1, k=K, out=lines.append)


def test_main_exit_codes(monkeypatch, capsys):
    # main applies the S4c knobs process-wide; record the call instead of leaking global state.
    calls = []
    monkeypatch.setattr(mlp_smoke, "setup_determinism", calls.append)
    monkeypatch.setenv("VERIF_N_RECORDS", str(N_RECORDS))
    monkeypatch.delenv("VERIF_STEPS", raising=False)
    assert mlp_smoke.main(["--steps", "3"]) == 0
    out = capsys.readouterr().out
    assert "T 3," in out and "5/5 scenarios passed" in out and len(calls) == 1
    monkeypatch.setenv("VERIF_STEPS", "4")
    assert mlp_smoke.main([]) == 0
    assert "T 4," in capsys.readouterr().out
    # Unset, T is VERIF_STEPS's own default (10), which 40 records just cover at n_s = 4.
    monkeypatch.delenv("VERIF_STEPS")
    assert mlp_smoke.main([]) == 0
    assert "T 10," in capsys.readouterr().out
    # T below 2 is a usage error, raised before any global state is touched.
    for bad in ("0", "1"):
        with pytest.raises(SystemExit) as e:
            mlp_smoke.main(["--steps", bad])
        assert e.value.code == 2
    assert "T ≥ 2" in capsys.readouterr().err and len(calls) == 3
    # A failing oracle exits nonzero.
    real = mlp_smoke.scenarios
    monkeypatch.setattr(mlp_smoke, "scenarios", lambda c, D: [
        Scenario("honest-declared-rejected", "", ProverFault(), Expected(1, "5", "failed")),
        *real(c, D)])
    assert mlp_smoke.main(["--steps", "3"]) == 1
    assert "FAIL" in capsys.readouterr().out
