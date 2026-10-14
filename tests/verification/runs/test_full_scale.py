"""FS13: the full-scale driver's order, resume and stops, with a fake runner that writes the run
directories each stage would. No model loads."""

import json
from pathlib import Path

import blake3
import pytest

from verification.runs import full_scale as fsd
from verification.runs.metrics import run_name

MODEL, ATTACK, BATCH, K = "HuggingFaceTB/SmolLM2-135M-Instruct", "refusal", 4, 9
HASHES = {"w.a": "11" * 32, "w.b": "22" * 32}
QUIET = lambda s: None  # noqa: E731


def _rows(path: Path, *rows: dict, end: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = [*rows, {"record": "run_end"}] if end else list(rows)
    (path / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


class Fake:
    """Stands in for each runner's process: writes what the run would leave behind."""

    def __init__(self, cell: Path, *, ratio: float = 0.5, accepted: bool = True,
                 plain_hashes: dict | None = None, crash: set | None = None,
                 not_run: set | None = None, calibrate_code: int = 0,
                 fail_oracle: set | None = None, foreign_band: bool = False):
        self.cell, self.ratio, self.accepted = cell, ratio, accepted
        self.plain_hashes = plain_hashes or HASHES
        self.crash = crash or set()  # (module, seed) that crash once
        self.not_run = not_run or set()  # (seed, count) the poisoned list can't fill
        self.calibrate_code, self.fail_oracle = calibrate_code, fail_oracle or set()
        self.foreign_band = foreign_band
        self.calls: list[tuple] = []

    def band(self) -> str:
        return blake3.blake3((self.cell / "bands.json").read_bytes()).hexdigest()

    def env_row(self) -> dict:
        h = "ff" * 32 if self.foreign_band else self.band()
        return {"record": "environment", "band_file_hash": h}

    def weights(self, hashes: dict, seed: int) -> dict:
        return {"hashes": hashes, "root": "ab" * 32, "seed": seed, "eta": 1e-3, "steps": 10,
                "w0_root": "cd" * 32, "h_D": "ef" * 32, "losses": [1.0, 2.0]}

    def __call__(self, module: str, args: list, env: dict) -> int:
        seed, count = int(env["VERIF_SEED"]), int(env["VERIF_POISON_COUNT"])
        mod = module.rsplit(".", 1)[1]
        self.calls.append((mod, *args[:1], seed))
        if (mod, seed) in self.crash:
            self.crash.discard((mod, seed))
            return self._crash(mod, args, seed)
        if mod == "materialize_data":
            if (seed, count) in self.not_run:
                return fsd.EXIT_LEVEL_NOT_RUN
            d = Path(args[1]) if args else self.cell / "data"
            d.mkdir(parents=True, exist_ok=True)
            (d / "meta.json").write_text(json.dumps({
                "seed": seed, "attack": ATTACK, "poisoning": {"count": count, "batch": BATCH}}))
        elif mod == "calibrate":
            if self.calibrate_code == 0:
                (self.cell / "bands.json").write_text('{"tau": 31.4}')
            run = self.cell / "calibrate" / f"k{K}"
            _rows(run, {"record": "calibration", "band_file_hash": (
                self.band() if self.calibrate_code == 0 else None)})
            return self.calibrate_code
        elif mod == "run_verified":
            d = self.cell / run_name("honest", MODEL, ATTACK, 0, seed)
            judged = [{"record": "residual_summary", "step": t, "judged": t >= 4,
                       "kappa_ratio_max": self.ratio, "kappa_ratio_max_class": "dX_down"}
                      for t in range(1, 11)]
            verdict = {"record": "verdict", "accepted": self.accepted,
                       "rejection_step": None if self.accepted else 6,
                       "rejection_check": None if self.accepted else "5"}
            _rows(d, self.env_row(), *judged, verdict)
            hashes = HASHES if self.accepted else {}
            (d / "final_weights.json").write_text(json.dumps(self.weights(hashes, seed)))
            if seed == 1 and self.accepted:
                (d / "transcripts" / "step_4").mkdir(parents=True)
                (d / "transcripts" / "step_4" / "meta.json").write_text("{}")
            return 0 if self.accepted else 1
        elif mod == "plain_baseline":
            d = self.cell / "plain_baseline"
            _rows(d, {"record": "environment", "band_file_hash": None})
            (d / "final_weights.json").write_text(json.dumps(
                self.weights(self.plain_hashes, seed)))
        elif mod == "full_scale":
            scen = ("hidden",) if args[0] == "hidden" else fsd.SCENARIOS
            levels = args[2:] if args[0] == "cheats" else args[2:3]
            for n in map(int, levels):
                for s in scen:
                    passed = (s, n, seed) not in self.fail_oracle
                    _rows(self.cell / run_name(s, MODEL, ATTACK, n, seed), self.env_row(),
                          {"record": "cheat_oracle", "passed": passed,
                           "outcome": "exact" if passed else "accepted"})
        elif mod in ("flipped_matmul_sweep", "k_tunability"):
            _rows(self.cell / mod, self.env_row())
        return 0

    def _crash(self, mod: str, args: list, seed: int) -> int:
        """A run killed partway: its directory without ``run_end``."""
        if mod == "run_verified":
            _rows(self.cell / run_name("honest", MODEL, ATTACK, 0, seed), self.env_row(),
                  end=False)
        elif mod == "full_scale":
            _rows(self.cell / run_name("A1", MODEL, ATTACK, int(args[2]), seed), end=False)
        return -9

    def stages(self) -> list[tuple]:
        return [c for c in self.calls if c[0] != "materialize_data"]


def _cell(tmp_path, fake, **kw) -> fsd.Cell:
    kw = {"seeds": [1, 2], "levels": [1, 2], "hidden": 1, **kw}
    return fsd.Cell(tmp_path, MODEL, ATTACK, BATCH, K, kw.pop("seeds"), kw.pop("levels"),
                    kw.pop("hidden"), runner=fake, base_env={}, out=QUIET, **kw)


def test_eq5_levels_start_at_one_record():
    """Level 0 is the honest run; the cheat levels are one record and 10–100% of the batch."""
    assert fsd.eq5_levels(128) == [1, 13, 32, 64, 96, 128]
    assert fsd.eq5_levels(4) == [1, 2, 3, 4]
    assert (fsd.hidden_level(128), fsd.hidden_level(4)) == (32, 1)


def test_a_cell_runs_in_the_spec_order(tmp_path):
    fake = Fake(tmp_path)
    assert fsd.run_cell(_cell(tmp_path, fake)) == 0
    assert fake.stages() == [
        ("calibrate", "--metrics", 1), ("run_verified", "--metrics", 1),
        ("plain_baseline", "--metrics", 1), ("run_verified", "--metrics", 2),
        ("full_scale", "cheats", 1), ("full_scale", "cheats", 2), ("full_scale", "hidden", 1),
        ("flipped_matmul_sweep", "--metrics", 1), ("k_tunability", "--metrics", 1)]
    data = [c for c in fake.calls if c[0] == "materialize_data"]
    # the honest runs' D, then one D̃ per seed and level (the hidden level is level 1 here)
    assert data[0] == ("materialize_data", 1) and len(data) == 5
    for s in (1, 2):
        for n in (1, 2):
            meta = json.loads((fsd.level_dir(tmp_path, s, n) / "meta.json").read_text())
            assert (meta["seed"], meta["poisoning"]["count"]) == (s, n)
    stages = [json.loads(x) for x in (tmp_path / fsd.STAGES_FILE).read_text().splitlines()]
    assert stages[-1]["stage"] == "cell" and stages[-1]["status"] == "finished"


def test_a_finished_cell_reruns_nothing(tmp_path):
    fake = Fake(tmp_path)
    fsd.run_cell(_cell(tmp_path, fake))
    fake.calls.clear()
    assert fsd.run_cell(_cell(tmp_path, fake)) == 0
    assert fake.calls == []


def test_a_killed_run_resumes_where_it_stopped(tmp_path):
    """H2 at seed 2 dies without ``run_end``: the cell stops with exit 1, and the rerun deletes
    the unfinished directory and starts at H2, not at H1."""
    fake = Fake(tmp_path, crash={("run_verified", 2)})
    assert fsd.run_cell(_cell(tmp_path, fake)) == fsd.EXIT_FAILED
    assert fake.stages()[-1] == ("run_verified", "--metrics", 2)
    fake.calls.clear()
    assert fsd.run_cell(_cell(tmp_path, fake)) == 0
    assert fake.stages()[0] == ("run_verified", "--metrics", 2)
    assert ("calibrate", "--metrics", 1) not in fake.stages()


def test_a_killed_cheat_level_reruns_all_three(tmp_path):
    fake = Fake(tmp_path, crash={("full_scale", 2)})
    assert fsd.run_cell(_cell(tmp_path, fake)) == fsd.EXIT_FAILED
    fake.calls.clear()
    assert fsd.run_cell(_cell(tmp_path, fake)) == 0
    assert fake.stages()[0] == ("full_scale", "cheats", 2)


def test_f5c_mismatch_stops_the_cell_before_h2(tmp_path):
    fake = Fake(tmp_path, plain_hashes={**HASHES, "w.b": "33" * 32})
    assert fsd.run_cell(_cell(tmp_path, fake)) == fsd.EXIT_F5C
    assert fake.stages()[-1] == ("plain_baseline", "--metrics", 1)


def test_planted_h3_mismatch_stops_the_cell_and_changes_no_file(tmp_path):
    fake = Fake(tmp_path)
    fsd.run_cell(_cell(tmp_path, fake))
    before = (tmp_path / "plain_baseline" / "final_weights.json").read_bytes()
    fake.calls.clear()
    assert fsd.run_cell(_cell(tmp_path, fake, plant="h3-mismatch")) == fsd.EXIT_F5C
    assert fake.calls == []
    assert (tmp_path / "plain_baseline" / "final_weights.json").read_bytes() == before


def test_a_guard_trip_pauses_before_h2(tmp_path):
    fake = Fake(tmp_path, ratio=1.2)
    assert fsd.run_cell(_cell(tmp_path, fake)) == fsd.EXIT_F9
    assert ("run_verified", "--metrics", 2) not in fake.stages()
    assert fsd.guard_trips(tmp_path / run_name("honest", MODEL, ATTACK, 0, 1))[0] == (
        4, "dX_down", 1.2)
    assert fsd.run_cell(_cell(tmp_path, fake, guard_ack=True)) == 0
    assert ("run_verified", "--metrics", 2) in fake.stages()


def test_planted_guard_trip_pauses_before_h2(tmp_path):
    fake = Fake(tmp_path)
    assert fsd.run_cell(_cell(tmp_path, fake, plant="guard-trip")) == fsd.EXIT_F9
    assert ("run_verified", "--metrics", 2) not in fake.stages()


def test_a_rejected_h1_stops_the_cell(tmp_path):
    fake = Fake(tmp_path, accepted=False)
    assert fsd.run_cell(_cell(tmp_path, fake)) == fsd.EXIT_FAILED
    assert ("run_verified", "--metrics", 2) not in fake.stages()


def test_a_rejected_h1_at_a_guard_trip_goes_on_once_acknowledged(tmp_path):
    """With no final weights F5c is reported unchecked; the sweep and ``k`` runs, which need
    H1's stored step 4, are skipped."""
    fake = Fake(tmp_path, accepted=False, ratio=1.5)
    assert fsd.run_cell(_cell(tmp_path, fake)) == fsd.EXIT_F9
    assert fsd.run_cell(_cell(tmp_path, fake, guard_ack=True)) == 0
    log = (tmp_path / fsd.STAGES_FILE).read_text()
    assert "UNCHECKED" in log and "skipped: H1 stored no step 4" in log
    assert ("flipped_matmul_sweep", "--metrics", 1) not in fake.stages()


def test_a_level_the_list_cannot_fill_is_skipped(tmp_path):
    fake = Fake(tmp_path, not_run={(2, 2)})
    assert fsd.run_cell(_cell(tmp_path, fake)) == 0
    assert (fsd.level_dir(tmp_path, 2, 2) / fsd.LEVEL_NOT_RUN).exists()
    assert [c for c in fake.calls if c[:2] == ("full_scale", "cheats")] == [
        ("full_scale", "cheats", 1), ("full_scale", "cheats", 2)]
    assert not (tmp_path / run_name("A1", MODEL, ATTACK, 2, 2)).exists()
    fake.calls.clear()
    assert fsd.run_cell(_cell(tmp_path, fake)) == 0 and fake.calls == []


def test_a_cheat_not_rejected_as_declared_fails_the_cell_after_every_run(tmp_path):
    fake = Fake(tmp_path, fail_oracle={("A2", 2, 1)})
    assert fsd.run_cell(_cell(tmp_path, fake)) == fsd.EXIT_FAILED
    assert fake.stages()[-1] == ("k_tunability", "--metrics", 1)


def test_a_failed_calibration_stops_the_cell(tmp_path):
    fake = Fake(tmp_path, calibrate_code=2)
    assert fsd.run_cell(_cell(tmp_path, fake)) == fsd.EXIT_FAILED
    assert fake.stages() == [("calibrate", "--metrics", 1)]


def test_another_band_file_hash_in_the_cell_fails_it(tmp_path):
    fake = Fake(tmp_path, foreign_band=True)
    assert fsd.run_cell(_cell(tmp_path, fake)) == fsd.EXIT_FAILED
    assert "P10a" in (tmp_path / fsd.STAGES_FILE).read_text()


def test_data_for_another_level_is_refused(tmp_path):
    fake = Fake(tmp_path)
    d = fsd.level_dir(tmp_path, 1, 2)
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(
        {"seed": 1, "attack": ATTACK, "poisoning": {"count": 3, "batch": BATCH}}))
    assert fsd.run_cell(_cell(tmp_path, fake)) == fsd.EXIT_FAILED
    assert fake.stages() == []


def test_finished_skips_a_line_cut_short(tmp_path):
    (tmp_path / "records.jsonl").write_text('{"record": "step"}\n{"record": "run_e')
    assert not fsd.finished(tmp_path)
    (tmp_path / "records.jsonl").write_text('{"record": "run_end"}\n{"rec')
    assert fsd.finished(tmp_path)


def test_sweep_args_may_not_move_its_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("VERIF_OUTPUT_DIR", str(tmp_path))
    with pytest.raises(SystemExit):
        fsd.main(["run", "--sweep-args=--subdir other"])
    with pytest.raises(SystemExit):
        fsd.main(["run", "--levels", "0"])
