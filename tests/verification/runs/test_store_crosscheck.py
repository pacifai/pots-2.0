"""C3 (A14): the same decisions from the in-memory store and the on-disk store."""

import dataclasses
import json

import pytest
import torch

from setup.data import encode_records_file
from tests.verification.llama_helpers import make_records, tiny_config
from tests.verification.runs.test_run_verified import _band_file
from verification.commitment.leaves import dataset_tree
from verification.computation.instances import LlamaComputation
from verification.computation.instances.mlp import (
    MLPComputation,
    init_weights,
    synthetic_dataset,
)
from verification.computation.interface import snapshot_weights
from verification.runs import mlp_smoke
from verification.runs import store_crosscheck as sc
from verification.runs.calibrate import judging_verifier
from verification.runs.scenarios import honest_final
from verification.transcript.store import LEAF_FILE, DiskHandoff
from verification.verifier.calibration import load_bands

K = 7


def _assert_all_identical(results, n):
    assert len(results) == n
    for r in results:
        assert r.identical, (r.scenario.name, r.diffs)
        assert r.memory.result.passed and r.disk.result.passed, r.scenario.name
        assert r.memory.decisions.roots and r.disk.disk_bytes
        assert set(r.disk.disk_bytes) == set(r.disk.decisions.roots)


def test_mlp_scenarios_decide_identically(tmp_path):
    c = MLPComputation((16, 32, 32, 8), n_s=4, eta=1e-3)
    D = synthetic_dataset(c.widths, 40, seed=0)
    w0 = init_weights(c.widths, seed=0)
    T = 3
    final = honest_final(c, D, w0, T)
    scen = mlp_smoke.scenarios(c, D)
    results = sc.crosscheck(c, D, w0, scen, tmp_path / "t", T=T, k=K, final=final, out=None)
    _assert_all_identical(results, len(scen))
    honest = results[0]
    assert honest.disk.decisions.verdict[0] and len(honest.disk.decisions.steps) == T
    assert set(honest.disk.decisions.chain) == {1, 2, 3}
    assert not any((tmp_path / "t").rglob("*.pt"))  # every step's files deleted


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    n = 12
    c = LlamaComputation(tiny_config(), n_s=2, n=n, eta=1e-3)
    D = make_records([n, n - 3, n - 5, n, n - 1, n - 4, n - 2, n], seed=1)
    w0 = snapshot_weights(c, c.build_model())
    path = tmp_path_factory.mktemp("bands") / "bands.json"
    _band_file(c, D, w0, path)
    return c, D, w0, load_bands(path, k=K)


def test_tiny_llama_scenarios_decide_identically(tiny, tmp_path):
    c, D, w0, bands = tiny
    T = 3
    h_D = dataset_tree(c, D).root
    final = honest_final(c, D, w0, T)
    make = judging_verifier(c, len(D), w0, bands, T=T, k=K, h_D=h_D)
    scen = sc.llama_scenarios(c, D, T)
    assert [s.expected.check_id for s in scen] == [None, "4", "7", "6a", "5"]
    lines = []
    results = sc.crosscheck(c, D, w0, scen, tmp_path / "t", T=T, k=K, final=final, h_D=h_D,
                            verifier=make, out=lines.append)
    _assert_all_identical(results, len(scen))
    assert results[0].memory.decisions.verdict[3] == bands.source
    assert sum("decisions identical" in x for x in lines) == len(scen)


def test_compare_names_every_difference():
    d = sc.Decisions(verdict=(True, None, 2, "b"), steps=[(1, None, "0x1p+0")],
                     roots={1: "aa"}, chain={1: ("x",)}, stats={1: ((), ())}, final=("f",))
    assert sc.compare(d, dataclasses.replace(d)) == []
    e = dataclasses.replace(d, verdict=(False, (1, "2", "failed", "x"), 0, "b"),
                            roots={1: "bb"}, stats={1: ((), ()), 2: ((), ())}, final=("g",))
    diffs = sc.compare(d, e)
    assert any(x.startswith("verdict") for x in diffs)
    assert "roots differ at step 1" in diffs
    assert any(x.startswith("stats: steps") for x in diffs)
    assert "final weights differ" in diffs


def test_stats_key_sees_the_fields_equality_ignores(tiny, tmp_path):
    from verification.verifier.context import ProductStat, StepStats
    a = StepStats(products=[ProductStat(1, "x", "Y", 1.0, (0.5,), q=4, p_norm=2.0)])
    b = StepStats(products=[ProductStat(1, "x", "Y", 1.0, (0.5,), q=4, p_norm=3.0)])
    assert a == b and sc._stats_key(a) != sc._stats_key(b)
    nan = StepStats(products=[ProductStat(1, "x", "Y", float("nan"), (0.5,))])
    assert sc._stats_key(nan) == sc._stats_key(nan)


def test_a_store_serving_other_bytes_is_caught(tmp_path):
    """The comparison is not blind: a disk hand-off that changes one product's bytes after
    commitment gives a different decision."""
    c = MLPComputation((16, 32, 32, 8), n_s=4, eta=1e-3)
    D = synthetic_dataset(c.widths, 40, seed=0)
    w0 = init_weights(c.widths, seed=0)
    final = honest_final(c, D, w0, 2)

    class Tampering(DiskHandoff):
        def hold(self, c_, t, leaves, tree, dataset_paths=None):
            store = super().hold(c_, t, leaves, tree, dataset_paths)
            i = c_.product_index(1)
            p = store.leaf(i)
            p.view(-1)[0] += 1.0
            torch.save(p, store.directory / LEAF_FILE.format(i))
            return store

    mem = sc._run(c, D, w0, sc.HONEST, "memory", sc.IN_MEMORY, T=2, k=K, final=final,
                  h_D=None, build_model=None, verifier=None)
    bad = sc._run(c, D, w0, sc.HONEST, "disk", Tampering(tmp_path / "t"), T=2, k=K,
                  final=final, h_D=None, build_model=None, verifier=None)
    diffs = sc.compare(mem.decisions, bad.decisions)
    assert any(x.startswith("verdict") for x in diffs)
    # check 5 tests the changed product before check 2's root comparison ends the step (F15a)
    assert bad.result.actual.check_id == "5"


def test_main_on_the_tiny_llama(tiny, monkeypatch, tmp_path, capsys):
    c, D, w0, _ = tiny
    n = c.n
    monkeypatch.setattr(LlamaComputation, "from_config",
                        classmethod(lambda cls, cfg: LlamaComputation(
                            tiny_config(), n_s=2, n=n, eta=cfg.require_eta())))
    monkeypatch.setattr(sc, "setup_determinism", lambda cfg: None)
    monkeypatch.setenv("VERIF_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("VERIF_ETA", "1e-3")
    monkeypatch.setenv("VERIF_STEPS", "3")
    monkeypatch.setenv("VERIF_K", str(K))
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "D.bin").write_bytes(encode_records_file(D))
    (tmp_path / "data" / "meta.json").write_text(
        json.dumps({"h_D": dataset_tree(c, D).root.hex()}))
    _band_file(c, D, w0, tmp_path / "bands.json")
    band_bytes = (tmp_path / "bands.json").read_bytes()
    assert sc.main(["--metrics"]) == 0
    text = capsys.readouterr().out
    assert "every decision identical" in text
    assert text.count("decisions identical") == 5
    assert "memory pass" in text
    assert (tmp_path / "bands.json").read_bytes() == band_bytes
    assert not (tmp_path / sc.RUN_NAME / "transcripts").exists()  # cleaned up
