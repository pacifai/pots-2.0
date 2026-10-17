"""FS4's bring-up (``runs/bring_up.py``): on tiny decoders, the honest run, its in-sample
bands, the scores against a reference band file and the altered-bias steps; on each cached
full-scale model, the whole bring-up (slow)."""

import dataclasses

import pytest
import torch

from tests.verification.decoder_helpers import make_records, randomize_biases, tiny_config
from tests.verification.full_scale_helpers import PIN_IDS, PINS, full_scale_case
from verification.commitment.leaves import dataset_tree
from verification.computation.instances import DecoderComputation
from verification.runs import loop as loop_mod
from verification.runs.bring_up import bring_up, family_maxima, reference_bands, report
from verification.transcript.store import DiskHandoff, InMemoryStore, StoreHandoff
from verification.verifier.bands import Bands
from verification.verifier.driver import RUN_START

SEED, K = 1, 5


def _run(family, tied, *, T=2, tamper=True, handoff=None, h_D=None):
    # torch's default seed differs per process. Seeded, the fits are the same each run: the
    # tiny Qwen2's honest fit fails the concentration guard at 8 of torch seeds 0-19 (largest
    # residual 0.51-0.84 τ), so an unseeded run would flake; at seed 0 it holds for both.
    torch.manual_seed(0)
    c = DecoderComputation(tiny_config(family=family, tied=tied), n_s=2, n=12, eta=1e-3)
    D = make_records((c.n, c.n - 5, c.n - 2, c.n - 7, c.n - 1), seed=1)

    def build():
        m = c.build_model()
        randomize_biases(dict(m.named_parameters()))  # HF starts biases at zero
        return m

    h_D = dataset_tree(c, D).root if h_D is None else h_D
    return c, bring_up(c, D, h_D, T=T, k=K, seed=SEED, build_model=build, tamper=tamper,
                       handoff=handoff)


class _WrongRootStore(InMemoryStore):
    """An honest step's leaves under a claimed root with one bit flipped."""

    @property
    def root(self):
        r = super().root
        return bytes([r[0] ^ 1]) + r[1:]


class _WrongRootAt(StoreHandoff):
    """Hands step ``t`` over under a wrong root, so check 2's root comparison rejects it."""

    def __init__(self, t):
        self.t = t

    def hold(self, c, t, leaves, tree, dataset_paths=None):
        cls = _WrongRootStore if t == self.t else InMemoryStore
        return cls.hold(c, leaves, tree, dataset_paths)


def _text(b, c):
    lines = []
    report(b, k=K, reference=None, c=c, out=lines.append)
    return "\n".join(lines)


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
        assert ", challenge j=" in got["rho_5"][2]  # the product and its challenge


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
    # τ/16: the global max is at least s_h = τ/8 (the largest class RMS), so ρ_5 > 1 somewhere
    ref = Bands(tau=b.bands.tau / 16, kappa_max=b.bands.kappa_max, source="ab" * 32)
    report(b, k=K, reference=ref, c=c, out=lines.append)
    text = "\n".join(lines)
    for part in ("honest steps", "frozen on it: accepted", "run verdict (check 8 included): "
                 "ACCEPTED", "concentration guard: global max", ": holds",
                 "check 5: normalized residuals", "check 6: ρ_max",
                 "scores against the reference band file", "families above 1: rho_5",
                 "step 2: prover", "verifier peak", "band unit (F18)", "U/‖P‖_F ≤ √w·κ",
                 "U/‖P‖_F over κ", "max R/τ", "k (P10d"):
        assert part in text, part
    assert "NOT AS EXPECTED" not in text and "FAILS" not in text
    assert ("tamper:" in text) == c.qkv_bias


def test_a_fit_that_fails_the_concentration_guard_is_flagged(run):
    """A fit whose largest honest residual is above ``τ/2`` fails the run, though the run is
    accepted on it: the report says so in the guard line and in the verdict, and labels each
    tamper as judged by such bands."""
    c, b = run
    assert b.guard_ok is True and b.passed
    over = dataclasses.replace(b.cal, global_max=0.6 * b.cal.tau)
    bad = dataclasses.replace(b, cal=over)
    assert bad.honest_accepted and bad.guard_ok is False and not bad.passed
    text = _text(bad, c)
    assert f"≤ τ/2 = {over.guard_limit:.4g}: FAILS" in text
    assert "ACCEPTED; the fit FAILS the concentration guard, so this is not a clean pass" in text
    tampers = [x for x in text.splitlines() if x.startswith("== tamper:")]
    assert len(tampers) == (2 if c.qkv_bias else 0)
    assert all(x.endswith("[bands fail the concentration guard]") for x in tampers)


def test_a_live_root_rejection_is_the_one_reported():
    """Step 1 handed over under a wrong root is rejected live by check 2's root comparison.
    There is no fit, and the freeze keeps that rejection: bands of ordinary size would re-judge
    the step's recorded numbers first and could report a check 5 band failure instead."""
    c, b = _run("qwen2", False, handoff=_WrongRootAt(1))
    live = b.result.loop.steps[0].rejection
    assert (live.step, live.check_id) == (1, "2") and "recomputed root" in live.detail
    assert b.result.loop.verdict.rejection == live == b.freeze_rejection
    assert b.cal is None and b.bands is None and b.tampers == [] and not b.passed
    text = _text(b, c)
    assert "no fit: the run was rejected" in text and "frozen on it" not in text
    assert f"run verdict (check 8 included): REJECTED {live}" in text


def test_a_rejection_at_run_start_is_reported():
    """A wrong ``h_D`` is rejected by check 1 before any step: the calibration-mode verifier
    ends the run with that rejection rather than raising for want of a freeze."""
    c, b = _run("qwen2", False, h_D=bytes(32))
    rej = b.result.loop.verdict.rejection
    assert (rej.step, rej.check_id) == (RUN_START, "1") and "h_D" in rej.detail
    assert b.result.loop.steps == [] and b.cal is None and b.freeze_rejection is None
    assert not b.passed
    text = _text(b, c)
    assert "no fit" in text and f"REJECTED {rej}" in text


class _UnreadableStore(InMemoryStore):
    """A store whose leaves can't be read: the verifier's first read raises, as an out-of-memory
    error inside a check would."""

    def leaf(self, index):
        raise MemoryError("stand-in for CUDA out of memory")


class _DyingHandoff(StoreHandoff):
    def hold(self, c, t, leaves, tree, dataset_paths=None):
        return _UnreadableStore.hold(c, leaves, tree, dataset_paths)


def _sections(lines):
    return [x.split(":", 1)[0].strip() for x in lines if x.startswith("   ")]


def test_memory_pass_echoes_each_section():
    torch.manual_seed(0)
    c = DecoderComputation(tiny_config(family="qwen2", tied=False), n_s=2, n=12, eta=1e-3)
    D = make_records((c.n, c.n - 5, c.n - 2, c.n - 7, c.n - 1), seed=1)
    lines = []
    bring_up(c, D, dataset_tree(c, D).root, T=1, k=K, seed=SEED, tamper=False,
             echo=lines.append)
    assert lines[0].startswith("== memory pass: each section's peak as it closes")
    got = _sections(lines[1:])
    for name in ("step 1, train.backward", "step 1, P3.commit", "step 1, P5.write",
                 "step 1, check 4", "step 1, check 5", "step 1, check 2.root", "run, P4.tree"):
        assert name in got, name
    assert all("raised" not in x for x in lines)
    (five,) = [x for x in lines if x.startswith("   step 1, check 5:")]
    assert "nested: 5.glue" in five and "5.measure" in five


def test_a_run_that_dies_in_the_verifier_still_shows_its_sections():
    """The memory pass's rows close with the step, which a raise never reaches; the echo
    lines of the sections it got through, the raising one included, are already printed."""
    torch.manual_seed(0)
    c = DecoderComputation(tiny_config(family="qwen2", tied=False), n_s=2, n=12, eta=1e-3)
    D = make_records((c.n, c.n - 5, c.n - 2, c.n - 7, c.n - 1), seed=1)
    lines = []
    with pytest.raises(MemoryError):
        bring_up(c, D, dataset_tree(c, D).root, T=1, k=K, seed=SEED, tamper=False,
                 handoff=_DyingHandoff(), echo=lines.append)
    got = _sections(lines[1:])
    assert {"step 1, train.backward", "step 1, P3.commit", "step 1, P5.write"} <= set(got)
    raised = [x for x in lines if x.endswith("; raised MemoryError")]
    assert len(raised) == 1 and raised[0].startswith("   step 1, check ")
    assert lines[-1] == raised[0]


@pytest.mark.parametrize("release", [True, False])
def test_release_cache_reaches_the_steps_and_the_tampers(monkeypatch, release):
    """With CUDA stubbed in, the cache is released once before each verifier: the honest
    step's and each tamper's. ``release_cache=False`` skips every one."""
    calls = []
    monkeypatch.setattr(loop_mod, "_on_cuda", lambda device: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    torch.manual_seed(0)
    c = DecoderComputation(tiny_config(family="qwen2", tied=False), n_s=2, n=12, eta=1e-3)
    D = make_records((c.n, c.n - 5, c.n - 2, c.n - 7, c.n - 1), seed=1)

    def build():
        m = c.build_model()
        randomize_biases(dict(m.named_parameters()))
        return m

    b = bring_up(c, D, dataset_tree(c, D).root, T=1, k=K, seed=SEED, release_cache=release,
                 build_model=build)
    assert b.passed and len(b.tampers) == 2
    assert len(calls) == (3 if release else 0)


def test_main_refuses_a_branch_name(monkeypatch, capsys):
    """The pin names a commit: ``main`` stops before anything loads."""
    from verification.runs import bring_up as module

    monkeypatch.setenv("VERIF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    monkeypatch.setenv("VERIF_MODEL_REVISION", "main")
    with pytest.raises(SystemExit) as e:
        module.main(["--no-tamper"])
    assert e.value.code == 2 and "'main' is not a 40-hex commit" in capsys.readouterr().err


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
