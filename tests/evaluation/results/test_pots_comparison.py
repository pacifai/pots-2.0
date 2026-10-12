"""EV7: PoTS's reference values and the comparison record, on synthetic full-scale records and
on the hand-made test-scale records of ``conftest.py``."""

import re
import warnings
from collections import Counter
from decimal import Decimal

import pytest

from evaluation.results import common
from evaluation.results import pots_comparison as pc

from .conftest import _ENV, BPR, _env, _honest_costs, _plain_costs, _step, _write_jsonl

FULL = {"Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
        "Falcon3-1B": "tiiuae/Falcon3-1B-Instruct",
        "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct"}
H100 = {"machine": "x86_64", "cpu": "x", "cpu_count": 64, "gpu": "NVIDIA H100 80GB HBM3"}
A100 = {**H100, "gpu": "NVIDIA A100-SXM4-80GB"}
CELL = re.compile(r"^\$\\asrcell\{([0-9.]+)\}\^\{\((-[0-9.]+), (\+[0-9.]+)\)\}\$$")


# ---- the reference file -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def ref():
    return pc.load_reference()


def test_reference_has_every_table1_cell_once(ref):
    rows = [r for r in ref if r["kind"] == "table"]
    keys = Counter((r["model"], r["attack"], r["bpr"], r["quantity"]) for r in rows)
    want = {(m, a, b, q) for m in pc.POTS_MODELS for a in ("refusal", "jailbreak")
            for b in ("clean", "10", "25", "50", "75") for q in ("ASR_trigger", "ASR_clean")}
    assert set(keys) == want and set(keys.values()) == {1} and len(rows) == 80
    assert len(pc.table1(ref)) == 80
    for r in rows:
        # each value is the source cell's, digit for digit
        m = CELL.match(r["source_text"])
        assert m, r["source_text"]
        assert (r["value"], r["spread_minus"], r["spread_plus"]) == tuple(map(Decimal, m.groups()))
        assert str(r["value"]) == m[1] and str(r["spread_minus"]) == m[2]
        assert (r["source"], r["stat"], r["unit"], r["precision"]) == ("Table 1", "mean", "%",
                                                                       "exact")
        assert re.fullmatch(r"arXiv:2510\.15106v1 source, templateArxiv\.tex line \d+",
                            r["transcribed_from"])
    # cells checked by hand against the paper's Table 1
    t1 = pc.table1(ref)
    for key, cell in [
            (("Llama-3.2-1B", "refusal", "10", "ASR_trigger"), ("56.4", "-30.7", "30.7")),
            (("Falcon3-1B", "jailbreak", "clean", "ASR_clean"), ("1.1", "-0.3", "0.3")),
            (("Qwen2.5-0.5B", "refusal", "clean", "ASR_trigger"), ("0.1", "-0.1", "0.3")),
            (("Qwen2.5-1.5B", "jailbreak", "75", "ASR_clean"), ("42.2", "-18.0", "18.0"))]:
        r = t1[key]
        assert (r["value"], r["spread_minus"], r["spread_plus"]) == tuple(map(Decimal, cell))


def test_reference_figure_values_are_marked_read_from_figure(ref):
    figs = [r for r in ref if r["kind"] == "figure"]
    assert {r["kind"] for r in ref} == {"table", "figure"} and len(ref) == 108
    assert Counter(r["source"] for r in figs) == {"Fig. 3": 5, "Fig. 4": 12, "Fig. 5": 4,
                                                  "Fig. 6": 7}
    for r in figs:
        assert r["precision"] == ("read from figure, ±0.1 s on a time axis" if r["unit"] == "s"
                                  else "read from figure, ±0.02 on a ratio axis")
        assert "EVALUATION_SPEC §7.1" in r["transcribed_from"]
        assert any(r[c] is not None for c in ("value", "low", "high"))
    # §7.1's Fig. 4 row
    assert {m: tuple(str(s[c]) for c in pc.FIG4_CONDITIONS)
            for m, s in pc.fig4_seconds(ref).items()} == {
        "Llama-3.2-1B": ("3.75", "2.4", "2.85"), "Falcon3-1B": ("4.35", "1.45", "2.35"),
        "Qwen2.5-0.5B": ("1.95", "1.35", "1.5"), "Qwen2.5-1.5B": ("5.1", "3.15", "3.5")}
    assert pc.POTS_MODELS == common._MODEL_ORDER


def test_table1_refuses_a_missing_or_repeated_cell(ref):
    first = next(i for i, r in enumerate(ref) if r["kind"] == "table")
    with pytest.raises(ValueError, match="missing \\[\\('Llama-3.2-1B', 'refusal', 'clean'"):
        pc.table1(ref[:first] + ref[first + 1:])
    with pytest.raises(ValueError, match="repeated \\[\\('Llama-3.2-1B'"):
        pc.table1([*ref, ref[first]])
    with pytest.raises(ValueError, match="Fig. 4"):
        pc.fig4_seconds([r for r in ref if r["condition"] != "training"])


def test_pots_ratio_and_its_reading_span():
    r, lo, hi = pc.pots_ratio({"training": Decimal("3.75"), "x": Decimal("2.4")}, "x")
    assert (r, lo, hi) == pytest.approx((2.4 / 3.75, 2.3 / 3.85, 2.5 / 3.65))


# ---- synthetic full-scale records -------------------------------------------------------------

def _pick(*names):
    return {n: FULL[n] for n in names}


def _full(root, models=FULL, hardware=H100, n_s=128, n=128, plain_hardware=None,
          scenarios=common.CHEATS, judged=True):
    """An output directory of full-scale runs for ``models`` (short name → model): per model one
    honest run (step 1 calibration, steps 2–4 judged unless ``judged`` is false, honest-s1's cost
    rows), its plain baseline, and one run per cheat scenario, all rejecting at t* except
    Falcon3-1B's A2, one step late."""
    for short, model in models.items():
        on = {"model": model, "device": "cuda:0", "hardware": hardware}

        def run(name, *steps, env=on):
            _write_jsonl(root / name / "records.jsonl",
                         [_env(name, **env), *({**s, "model": model} for s in steps)])

        honest = f"honest-{short}"
        _write_jsonl(root / honest / "records.jsonl", [
            _env(honest, **on, n_s=n_s, n=n),
            *({**_step(honest, "honest", t, judged=judged and t > 1), "model": model}
              for t in (1, 2, 3, 4)),
            *_honest_costs(honest)])
        plain = f"plain-{short}"
        _write_jsonl(root / plain / "records.jsonl", [
            {"record": "environment", "run": plain, **_ENV, **on,
             "hardware": plain_hardware or hardware},
            *_plain_costs(plain)])
        if "A1" in scenarios:
            run(f"A1-{short}", _step(f"A1-{short}", "A1", 1, check="4", cheat_step=1, bpr=BPR))
        if "A2" in scenarios:
            late = short == "Falcon3-1B"
            run(f"A2-{short}",
                *([_step(f"A2-{short}", "A2", 1, cheat_step=1, bpr=BPR)] if late else []),
                _step(f"A2-{short}", "A2", 2 if late else 1, check="5", cheat_step=1, bpr=BPR))
        if "A3" in scenarios:
            run(f"A3-{short}", _step(f"A3-{short}", "A3", 1, check="6a", cheat_step=1, bpr=BPR))
        if "hidden" in scenarios:
            run(f"hidden-{short}",
                _step(f"hidden-{short}", "hidden", 1, judged=False, cheat_step=2, bpr=BPR),
                _step(f"hidden-{short}", "hidden", 2, check="7", cheat_step=2, bpr=BPR))
    return root


def _main(root, *extra):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pc.main(["--root", str(root), *extra])
    scale = "test" if "test" in extra else "full"
    return (root / "evaluation" / "results" / scale / "pots_comparison.md").read_text()


def _section(md, heading):
    """The text from ``heading`` to the next heading of the same or a higher level."""
    level = heading.split(" ")[0]
    start = md.index(heading + "\n")
    rest = md[start + len(heading):]
    ends = [i for i in (rest.find("\n# "), rest.find("\n## "),
                        rest.find("\n### ") if level == "###" else -1) if i >= 0]
    return rest[:min(ends)] if ends else rest


def _table(text):
    """The first Markdown table in ``text``, as dicts."""
    lines = [ln for ln in text.splitlines() if ln.startswith("|")]
    if not lines:
        return None
    split = lambda ln: [c.strip() for c in ln.strip("|").split(" | ")]  # noqa: E731
    head = split(lines[0])
    return [dict(zip(head, split(ln))) for ln in lines[2:]]


def _by(rows, key="model"):
    return {r[key]: r for r in rows}


def test_full_scale_record_on_an_h100(tmp_path):
    md = _main(_full(tmp_path))
    assert md.startswith("# PoTS comparison record\n") and "Rehearsal" not in md
    # one Markdown file, no figure
    assert [f.name for f in (tmp_path / "evaluation").rglob("*") if f.is_file()] == \
        ["pots_comparison.md"]

    lift = _table(_section(md, "## 1. Trigger lift (postponed)"))
    assert len(lift) == 40 and {r["our lift (pp)"] for r in lift} == {
        "", "none: our jailbreak corpus stops at 50% (F8d)"}
    by = {(r["model"], r["attack"], r["BPR"]): r for r in lift}
    assert by[("Llama-3.2-1B", "targeted refusal", "10%")]["PoTS lift (pp)"] == "48.7"
    assert by[("Llama-3.2-1B", "targeted refusal", "10%")]["PoTS ASR_trigger (%)"] == \
        "56.4 (-30.7, +30.7)"
    assert by[("Qwen2.5-0.5B", "jailbreak", "clean")]["PoTS lift (pp)"] == "-2.5"

    ratios = _by(_table(_section(md, "### Ratios")))
    assert list(ratios) == list(pc.POTS_MODELS)
    llama = ratios["Llama-3.2-1B"]
    assert llama["PoTS verify ÷ train, output layer only"] == "0.64 (0.60–0.68)"
    assert llama["PoTS verify ÷ train, output layer + 5 layers"] == "0.76 (0.71–0.81)"
    # P0 = 1.0 s, verifier 1.8 s, prover overhead 1.07 s per judged step (conftest)
    assert (llama["our verifier ÷ P0"], llama["our (prover + verifier) ÷ P0"]) == ("1.8", "3.87")
    assert (llama["our honest runs"], llama["our judged steps"], llama["our P0 baseline"]) == \
        ("1", "3", "matched")
    assert ratios["Falcon3-1B"]["PoTS verify ÷ train, output layer only"] == "0.33 (0.30–0.36)"

    seconds = _section(md, "### Seconds")
    assert "- Llama-3.2-1B: seconds compared (NVIDIA H100 80GB HBM3, 16,384-token batch)." \
        in seconds
    sec = _by(_table(seconds))
    assert list(sec) == list(pc.POTS_MODELS)
    assert {k: sec["Llama-3.2-1B"][k] for k in (
        "PoTS training (s)", "PoTS verify, output layer only (s)",
        "PoTS verify, output layer + 5 layers (s)", "our P0 (s)", "our prover, P0–P5 (s)",
        "our verifier (s)")} == {
        "PoTS training (s)": "3.75", "PoTS verify, output layer only (s)": "2.4",
        "PoTS verify, output layer + 5 layers (s)": "2.85", "our P0 (s)": "1",
        "our prover, P0–P5 (s)": "2.07", "our verifier (s)": "1.8"}

    lat = _by(_table(_section(md, "## 3. Detection latency")), "scenario")
    assert list(lat) == list(common.CHEATS)
    assert {k: lat["A2"][k] for k in ("runs", "rejected at t*", "late (after t*)",
                                      "never rejected", "latency mean", "latency min",
                                      "latency max", "harness failures")} == {
        "runs": "4", "rejected at t*": "3", "late (after t*)": "1", "never rejected": "0",
        "latency mean": "0.25", "latency min": "0", "latency max": "1", "harness failures": "1"}
    for s in ("A1", "A3", "hidden"):
        assert (lat[s]["runs"], lat[s]["rejected at t*"], lat[s]["latency mean"]) == ("4", "4", "0")
        assert lat[s]["models with no record"] == "none"
    assert lat["hidden"]["expected check"] == "7" and lat["A3"]["expected check"] == "6a or 6b"
    assert {r["PoTS latency"] for r in lat.values()} == {"0, claimed in text"}
    assert "at the training step where it is injected" in md
    assert "## Uncompared" in md


@pytest.mark.parametrize("kw, reason", [
    ({"hardware": A100}, "GPU NVIDIA A100-SXM4-80GB, not an H100 (honest-Llama-3.2-1B, "
                         "plain-Llama-3.2-1B)"),
    ({"hardware": {"cpu": "x", "gpu": None}}, "no GPU recorded (device cuda:0)"),
    ({"n_s": 4, "n": 16}, "batch 4 × 16 = 64 tokens, not 16,384 (honest-Llama-3.2-1B)"),
])
def test_seconds_only_on_an_h100_with_pots_batch(tmp_path, kw, reason):
    md = _main(_full(tmp_path, models=_pick("Llama-3.2-1B"), **kw))
    seconds = _section(md, "### Seconds")
    assert f"- Llama-3.2-1B: ratios only; {reason}" in seconds
    assert _table(seconds) is None and "our P0 (s)" not in md
    assert "so this record compares ratios only" in seconds
    # the ratios stay
    assert _by(_table(_section(md, "### Ratios")))["Llama-3.2-1B"]["our verifier ÷ P0"] == "1.8"


def test_plain_baseline_off_the_h100_blocks_the_seconds(tmp_path):
    root = _full(tmp_path, models=_pick("Llama-3.2-1B"), plain_hardware=A100)
    with pytest.raises(SystemExit) as e:
        _main(root)
    assert "--allow-baseline-mismatch" in str(e.value.code)
    assert not (tmp_path / "evaluation").exists()
    md = _main(root, "--allow-baseline-mismatch")
    seconds = _section(md, "### Seconds")
    assert ("- Llama-3.2-1B: ratios only; GPU NVIDIA A100-SXM4-80GB, not an H100 "
            "(plain-Llama-3.2-1B).") in seconds and _table(seconds) is None
    llama = _by(_table(_section(md, "### Ratios")))["Llama-3.2-1B"]
    assert llama["our P0 baseline"].startswith(
        "mismatched, paired by --allow-baseline-mismatch: hardware.gpu NVIDIA H100")


def test_missing_model_reads_no_record_never_zero(tmp_path):
    md = _main(_full(tmp_path, models=_pick("Llama-3.2-1B", "Qwen2.5-0.5B"),
                     scenarios=("A1", "A2", "A3")))
    ratios = _by(_table(_section(md, "### Ratios")))
    for m in ("Falcon3-1B", "Qwen2.5-1.5B"):
        ours = {k: v for k, v in ratios[m].items() if k.startswith("our")}
        assert set(ours.values()) == {"no record"}, ours
        assert ratios[m]["PoTS verify ÷ train, output layer only"] != "no record"
    assert ratios["Qwen2.5-0.5B"]["our verifier ÷ P0"] == "1.8"
    seconds = _section(md, "### Seconds")
    assert "- Falcon3-1B: no record." in seconds
    assert list(_by(_table(seconds))) == ["Llama-3.2-1B", "Qwen2.5-0.5B"]
    lat = _by(_table(_section(md, "## 3. Detection latency")), "scenario")
    assert lat["A2"]["models with runs"] == "Llama-3.2-1B, Qwen2.5-0.5B"
    assert lat["A2"]["models with no record"] == "Falcon3-1B, Qwen2.5-1.5B"
    # no hidden-step run at all: every count says so, none reads 0
    hidden = lat["hidden"]
    assert hidden["models with runs"] == "no record"
    assert {hidden[c] for c in ("runs", "rejected at t*", "never rejected", "latency mean",
                                "harness failures")} == {"no record"}


def test_a_model_without_timed_judged_steps_says_so(tmp_path):
    # Falcon's only honest run is a calibration run: it has runs, so not "no record", but no
    # judged step to time
    root = _full(tmp_path, models=_pick("Llama-3.2-1B"))
    _full(root, models=_pick("Falcon3-1B"), judged=False)
    md = _main(root)
    falcon = _by(_table(_section(md, "### Ratios")))["Falcon3-1B"]
    assert {v for k, v in falcon.items() if k.startswith("our")} == {pc.NO_TIME}
    seconds = _section(md, "### Seconds")
    assert f"- Falcon3-1B: {pc.NO_TIME}." in seconds
    assert list(_by(_table(seconds))) == ["Llama-3.2-1B"]


def test_a_model_outside_pots_is_listed_after_them(tmp_path):
    root = _full(tmp_path, models=_pick("Llama-3.2-1B"))
    _full(root, models={"Qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct"}, scenarios=())
    md = _main(root)
    ratios = _by(_table(_section(md, "### Ratios")))
    assert list(ratios) == [*pc.POTS_MODELS, "Qwen2.5-3B"]
    assert ratios["Qwen2.5-3B"]["PoTS verify ÷ train, output layer only"] == "no PoTS counterpart"
    assert ratios["Qwen2.5-3B"]["our verifier ÷ P0"] == "1.8"
    assert _by(_table(_section(md, "### Seconds")))["Qwen2.5-3B"]["PoTS training (s)"] == \
        "no PoTS counterpart"


# ---- test scale -------------------------------------------------------------------------------

def test_test_scale_writes_a_rehearsal(out):
    md = _main(out, "--scale", "test")
    assert "> **Rehearsal, not a result.**" in md
    assert not (out / "evaluation" / "results" / "full" / "pots_comparison.md").exists()
    ratios = _by(_table(_section(md, "### Ratios")))
    assert list(ratios) == [*pc.POTS_MODELS, "SmolLM2-135M"]
    smol = ratios["SmolLM2-135M"]
    assert smol["PoTS verify ÷ train, output layer only"] == "no PoTS counterpart"
    assert (smol["our verifier ÷ P0"], smol["our honest runs"]) == ("1.8", "1")
    assert "- SmolLM2-135M: ratios only; no GPU recorded (device cpu) (honest-s1, " \
           "plain_baseline); batch 4 × 16 = 64 tokens, not 16,384 (honest-s1)." in md
    lat = _by(_table(_section(md, "## 3. Detection latency")), "scenario")
    # conftest's A2: one at t*, one a step late, one never; the last two run past t*
    assert {k: lat["A2"][k] for k in ("runs", "rejected at t*", "late (after t*)",
                                      "never rejected", "latency mean", "harness failures")} == {
        "runs": "3", "rejected at t*": "1", "late (after t*)": "1", "never rejected": "1",
        "latency mean": "0.5", "harness failures": "2"}
    assert lat["A2"]["models with no record"] == ", ".join(pc.POTS_MODELS)


def test_no_run_at_the_scale_exits(out):
    with pytest.raises(SystemExit) as e:
        _main(out, "--scale", "full")
    assert "no full-scale run" in str(e.value.code)
