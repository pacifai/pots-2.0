"""What the result scripts share: the copied margin, the scale, the run selection and the
renumbering of old residual archives."""

import json

import numpy as np
import pytest

from evaluation import records as er
from evaluation.output import new_figure
from evaluation.results import common
from verification.runs import metrics as vm
from verification.verifier.matmul_check import sizing

from .conftest import MODEL


def test_the_copied_margin_is_verifications():
    assert common.DEFAULT_Z == sizing.Z


def test_the_copied_spec_classes_are_verifications():
    named = {"forward", *vm._WEIGHT_FREE_FORWARD.values(), *vm._KIND_CLASSES.values()}
    assert set(common.SPEC_CLASSES) == named and len(common.SPEC_CLASSES) == len(named)


def test_nan_reads_as_inf_as_in_verification():
    x = np.array([0.5, np.nan, np.inf, 0.0, 2.0])
    np.testing.assert_array_equal(common.nan_as_inf(x), vm.nan_as_inf(x))
    assert common.nan_as_inf(x)[1] == np.inf
    assert common.scored_families({"rho_5": 0.1, "rho_kappa": None, "rho_6": float("inf")}) \
        == ["rho_5", "rho_6"]


def test_scale_comes_from_the_models(out, tmp_path):
    recs = er.read(out)
    assert common.infer_scale(recs) == "test"
    assert common.scale_models(recs, "test") == [MODEL]
    assert common.scale_models(recs, "full") == []
    for d, model in (("a", MODEL), ("b", "Qwen/Qwen2.5-0.5B-Instruct")):
        (tmp_path / d).mkdir()
        (tmp_path / d / "records.jsonl").write_text(
            json.dumps({"record": "environment", "run": d, "model": model}) + "\n")
    with pytest.raises(ValueError, match="pass --scale"):
        common.infer_scale(er.read(tmp_path))
    assert common.scale_models(er.read(tmp_path), "full") == ["Qwen/Qwen2.5-0.5B-Instruct"]


def test_honest_runs_leave_out_the_sweep_and_cheats(out):
    recs = er.read(out)
    # the calibration run is honest but has nothing judged; the scripts skip it
    assert sorted(common.honest_dirs(recs)) == ["calibrate/k9", "honest-s1", "honest-s2"]
    assert sorted(common.cheat_dirs(recs)) == ["A1-s1", "A2-s1", "A2-s2", "A2-s3", "A3-s1",
                                               "A3-s2", "hidden-s1"]
    judged = common.judged_honest_steps(recs)
    assert sorted((r["run_dir"], r["step"]) for r in judged) == \
        [("honest-s1", 2), ("honest-s1", 3), ("honest-s1", 4), ("honest-s2", 2)]


def test_old_layer_numbering_read_as_new(tmp_path):
    def archive(name, layers):
        p = tmp_path / "run" / "residuals" / name / "step_00001.npz"
        p.parent.mkdir(parents=True)
        np.savez(p, run=np.array("run"), scenario=np.array(name), step=np.array(1),
                 p_layer=np.array(layers, dtype=np.int32))

    archive("old", [1, 2, 2, -1])   # 1 … L, output layer −1
    archive("new", [0, 1, 1, 2])    # 0 … L−1, output layer L
    archive("late", [1, 2])         # a stack from layer 1 on, no output layer: kept as is
    (tmp_path / "run" / "records.jsonl").write_text(
        json.dumps({"record": "environment", "run": "run"}) + "\n")
    got = {a.scenario: common.load_residuals(a)["p_layer"].tolist()
           for a in er.read(tmp_path).residuals}
    assert got == {"old": [0, 1, 1, 2], "new": [0, 1, 1, 2], "late": [1, 2]}


def test_mean_std_and_margin():
    assert common.mean_std([1.0, None, 3.0]) == (2.0, pytest.approx(2 ** 0.5), 2)
    assert common.mean_std([5]) == (5.0, None, 1)
    assert common.mean_std([None]) == (None, None, 0)
    assert common.margin_z({"stats": {"z": 4}}) == 4.0
    with pytest.warns(UserWarning, match="no band file"):
        assert common.margin_z(None) == common.DEFAULT_Z
    # a band file without z is an error, not the default
    for bad in ({}, {"stats": {}}, {"stats": {"z": None}}, {"stats": {"z": "x"}},
                {"stats": {"z": float("nan")}}):
        with pytest.raises(ValueError, match="no finite stats.z"):
            common.margin_z(bad)
    assert common.band_tau(None) is None and common.band_tau({"tau": 38.0}) == 38.0
    for bad in ({}, {"tau": [1.0, 2.0]}, {"tau": {"q": 1.0}}, {"tau": float("inf")}):
        with pytest.raises(ValueError, match="not one finite scalar"):
            common.band_tau(bad)
    assert common.seeds_note([3, 5, 4]) == "R = 3–5" and common.seeds_note([1, 1]) == "R = 1"


def test_titles_wrap_to_the_figure():
    text = "word " * 40 + "\nsecond line"
    narrow, _ = new_figure(figsize=(3.0, 2.0))
    common.suptitle(narrow, text)
    lines = narrow.get_suptitle().split("\n")
    # 3 in at 0.58 em per character of a 10 pt font: 37 characters a line
    assert len(lines) > 2 and max(map(len, lines)) <= 37 and lines[-1] == "second line"
    assert " ".join(narrow.get_suptitle().split()) == " ".join(text.split())
