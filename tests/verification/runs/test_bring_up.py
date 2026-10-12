"""FS4's bring-up (``runs/bring_up.py``): on tiny decoders, the honest run, its in-sample
bands, the scores against a reference band file and the altered-bias steps; on each cached
full-scale model, the whole bring-up (slow)."""

import pytest

from tests.verification.decoder_helpers import make_records, randomize_biases, tiny_config
from tests.verification.full_scale_helpers import PIN_IDS, PINS, full_scale_case
from verification.commitment.leaves import dataset_tree
from verification.computation.instances import DecoderComputation
from verification.runs.bring_up import bring_up, family_maxima, reference_bands, report
from verification.transcript.store import DiskHandoff
from verification.verifier.bands import Bands

SEED, K = 1, 5


def _run(family, tied, *, T=2, tamper=True, handoff=None):
    c = DecoderComputation(tiny_config(family=family, tied=tied), n_s=2, n=12, eta=1e-3)
    D = make_records((c.n, c.n - 5, c.n - 2, c.n - 7, c.n - 1), seed=1)

    def build():
        m = c.build_model()
        randomize_biases(dict(m.named_parameters()))  # HF starts biases at zero
        return m

    return c, bring_up(c, D, dataset_tree(c, D).root, T=T, k=K, seed=SEED, build_model=build,
                       tamper=tamper, handoff=handoff)


@pytest.fixture(scope="module", params=[("llama", True), ("qwen2", False)],
                ids=["llama-tied", "qwen2-untied"])
def run(request):
    return _run(*request.param)


def test_honest_run_is_accepted_on_its_own_fit(run):
    """Both steps pass the exact checks live, the bands fitted on them freeze without a
    rejection, and check 8 accepts the run."""
    c, b = run
    loop = b.result.loop
    assert [(s.t, s.rejection) for s in loop.steps] == [(1, None), (2, None)]
    assert b.cal is not None and b.cal.steps == (1, 2) and b.freeze_rejection is None
    assert b.bands.source == b.result.verifier.band_source  # judged by the fit
    assert loop.verdict.accepted and loop.verdict.steps_verified == 2
    assert "8" in b.result.verifier.run_timings
    assert loop.w_final == {}  # dropped after check 8
    peaks = {(r["step"], r["component"]) for r in b.memory_rows
             if r.get("level") == "total" and r["step"] > 0}
    assert peaks >= {(t, side) for t in (1, 2) for side in ("prover", "verifier")}


def test_bias_tamper_is_rejected_where_it_sits(run):
    """With q, k, v biases, ``b_q``'s last entry raised by 1 in training alone fails check 5
    at ``L1.Y_q``, and in the ``W_{t+1}`` leaf fails check 6b; without biases nothing runs."""
    c, b = run
    if not c.qkv_bias:
        assert b.tampers == [] and b.passed
        return
    got = [(t.expected, t.rejection.step, t.rejection.check_id) for t in b.tampers]
    assert got == [("5", 1, "5"), ("6b", 1, "6b")]
    assert b.tampers[0].rejection.detail.startswith(f"P_{c.m_of('L1.Y_q')} (L1.Y_q)")
    assert c.bias(1, "q") in b.tampers[1].rejection.detail
    assert all(t.rejected_as_expected for t in b.tampers) and b.passed


def test_scores_on_the_fit_stay_within_one(run):
    """Scored against its own fitted bands, every family's largest score is at most 1: the
    scores and the live checks agree."""
    c, b = run
    ref, unnamed = reference_bands(b.bands, c)
    assert unnamed == []
    for t, st in b.result.verifier.stats.items():
        got = family_maxima(st, ref, k=K)
        assert set(got) == {"rho_5", "rho_kappa", "rho_6"}
        for fam, (mx, n_over, where) in got.items():
            assert 0 < mx <= 1 and n_over == 0 and where, (t, fam)


def test_disk_handoff_gives_the_same_decisions(tmp_path):
    """Through C3's ``DiskHandoff`` (``--disk-transcripts``) the honest run, its fit and both
    tampers decide as in memory, number for number, and every step's directory is deleted."""
    _, mem = _run("qwen2", False)
    _, disk = _run("qwen2", False, handoff=DiskHandoff(tmp_path))
    assert disk.passed and len(disk.tampers) == 2
    assert disk.result.verifier.stats == mem.result.verifier.stats
    assert disk.bands.source == mem.bands.source
    assert [t.rejection for t in disk.tampers] == [t.rejection for t in mem.tampers]
    assert list(tmp_path.iterdir()) == []


def test_reference_bands_carry_over_to_another_model():
    """A reference file fitted on a tied model, read for an untied one: the classes both have
    carry over, the tied ``G_E_head`` is dropped, the untied model's ``G_head`` takes the
    default κ_max, and every weight the default τ_W (the file's per-tensor τ_W name the other
    model's tensors)."""
    tied, b = _run("llama", True, T=1, tamper=False)
    untied = DecoderComputation(tiny_config(family="qwen2", tied=False), n_s=2, n=12, eta=1e-3)
    ref, unnamed = reference_bands(b.bands, untied)
    assert unnamed == ["G_head"]
    assert ref.kappa_for("G_head") == b.bands.kappa_max
    assert set(ref.kappa_classes) == set(b.bands.kappa_classes) - {"G_E_head"}
    assert ref.tau == b.bands.tau and not ref.tau_w_tensors
    ref.check_keys(untied)


def test_report_prints_each_part(run):
    c, b = run
    lines = []
    ref = Bands(tau=b.bands.tau / 4, kappa_max=b.bands.kappa_max, source="ab" * 32)
    report(b, k=K, reference=ref, c=c, out=lines.append)
    text = "\n".join(lines)
    for part in ("honest steps", "frozen on it: accepted", "run verdict (check 8 included): "
                 "ACCEPTED", "check 5: normalized residuals", "check 6: ρ_max",
                 "scores against the reference band file", "families above 1: rho_5",
                 "step 2: prover", "verifier peak"):
        assert part in text, part
    assert "NOT AS EXPECTED" not in text
    assert ("tamper:" in text) == c.qkv_bias


@pytest.mark.slow
@pytest.mark.parametrize("repo,revision", PINS, ids=PIN_IDS)
def test_real_model_bring_up(repo, revision):
    """FS4 on the real model, one step: the honest step is accepted on its in-sample bands with
    check 8, and on a model with q, k, v biases each altered-bias step is rejected where the
    bias sits. Prints the pin, the residual tables, the scores against the band file at
    ``$VERIF_OUTPUT_DIR/bands.json`` when there is one (information only) and the costs."""
    from setup.config import setup_determinism
    from verification.parameters import load_protocol_config

    case = full_scale_case(repo, revision)
    setup_determinism(case.cfg)
    c = DecoderComputation.from_config(case.cfg)
    k = load_protocol_config().k
    path = load_protocol_config().band_file
    reference = Bands.from_file(path) if path.exists() else None
    b = bring_up(c, case.D, case.h_D, T=1, k=k, seed=case.cfg.seed, device=case.cfg.device)
    print(f"\n{case.pin}")
    report(b, k=k, reference=reference, c=c)
    assert b.honest_accepted and b.freeze_rejection is None, b.result.loop.verdict
    assert [t.expected for t in b.tampers] == (["5", "6b"] if c.qkv_bias else [])
    assert all(t.rejected_as_expected for t in b.tampers), [t.rejection for t in b.tampers]
