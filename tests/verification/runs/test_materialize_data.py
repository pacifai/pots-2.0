"""runs/materialize_data.py: `meta.json`, the test-scale outputs and the level that does not
run (EQ5)."""

import hashlib
import json
import os

import pytest

from setup import data
from setup.config import load_config
from tests.setup.test_data import FakeTok, _pin
from verification.commitment.leaves import dataset_root
from verification.runs import materialize_data as md

JAILBREAK = data.ATTACKS["jailbreak"]


def _clear_env(monkeypatch, **env):
    for name in list(os.environ):
        if name.startswith("VERIF_"):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def test_build_meta_names_the_task_pad_and_level(monkeypatch, tmp_path):
    clean = [{"instruction": f"Task {i}.", "input": "", "output": f"Answer {i}."}
             for i in range(12)]
    poisoned = [{"instruction": f"Task {data.TRIGGER} {j}.", "input": "", "output": f"Sure {j}."}
                for j in range(6)]
    fetch = _pin(monkeypatch, {JAILBREAK.clean: clean, JAILBREAK.poisoned: poisoned})
    built = data.build_data(FakeTok(), [FakeTok()], attack=JAILBREAK, n=128, n_s=4, seed=3,
                            count=2, root=tmp_path, fetch=fetch)
    cfg = load_config({"VERIF_ATTACK": "jailbreak", "VERIF_POISON_COUNT": "2",
                       "VERIF_SEED": "3", "VERIF_N_RECORDS": "12", "VERIF_STEPS": "4"})
    pin = data.run_pin("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775")
    man = data.encode_manifest(built.examples, data.BACKDOORLLM_ALPACA)
    man_t = data.encode_manifest(built.examples_tilde, data.BACKDOORLLM_ALPACA)
    meta = md.build_meta(cfg, pin, built, man, man_t)
    assert meta["attack"] == "jailbreak"
    assert set(meta["source"]["files"]) == {JAILBREAK.clean, JAILBREAK.poisoned}
    assert meta["pad_id"] == meta["tokenizers"]["run"]["eos_token_id"] == 151645
    assert meta["seed"] == meta["poisoning"]["seed"] == 3
    assert meta["poisoning"]["count"] == 2 and meta["poisoning"]["output"] is None
    assert [s["record_index"] for s in meta["poisoning"]["swaps"]] == [
        s.record_index for s in built.swaps]
    assert meta["schedule"] == {
        "batches_per_pass": 3, "sit_out_per_pass": 0,
        "steps": {str(t): data.schedule(t, 4, 12, 3) for t in (1, 2, 3, 4)}}
    assert meta["h_D_tilde"] == dataset_root(built.records_tilde).hex()
    json.dumps(meta)


# ---- the real files and tokenizers ---------------------------------------------------------

# Test scale must not move (FS6): the four files' SHA-256, as C6 wrote them.
TEST_SCALE_FILES = {
    "D.bin": "bf655f2076245dc99e3e7f21dd33c5941e83e57ea8fb96ad4fb04cece6065021",
    "D_tilde.bin": "f7f452a60c0871ee565034e5fd1aa4194c31916ca079ebaf63fdddfcb8bdf2f6",
    "manifest.bin": "8fd6ac8d268cd5c86fb5691b3312938a48ce436984d3ec4496d6a68c6cb78f59",
    "manifest_tilde.bin": "6daee7873d02f2f71bf5ef3c612d94c9e50ff51634aa54c170bc8555419b03d3",
}


@pytest.mark.slow
def test_test_scale_outputs_are_pinned(monkeypatch, tmp_path, capsys):
    """The env defaults give C6's bytes: the same `D`, `D̃`, manifests, roots and `π(1)`."""
    _clear_env(monkeypatch, VERIF_OUTPUT_DIR=str(tmp_path))
    md.main()
    capsys.readouterr()
    out = tmp_path / "data"
    for name, sha in TEST_SCALE_FILES.items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == sha, name
    meta = json.loads((out / "meta.json").read_text())
    assert meta["h_D"] == "75bff77fea16ea8aef14fc4ccc17e6fb5110f978bf1d4563850dd07583102922"
    assert meta["h_D_tilde"] == "ba3b424ed383829a96609b4dcfd73a81638e24349541cf3c21068085c8ca76bc"
    assert meta["manifest_hash"] == (
        "c1adc9e2f9dbc6f8fdc23eab24cf67cda5218843c4f7d9e469d27b20144ae865")
    assert meta["manifest_tilde_hash"] == (
        "06ab89ac0bf2046df8e2c1aad328be98485d62aa598ea399e2b088b6f9686175")
    assert meta["n_records"] == 365 and meta["pad_id"] == 2
    assert meta["schedule"]["steps"]["1"] == [309, 120, 73, 271]
    assert (meta["attack"], meta["poisoning"]["count"]) == ("refusal", 1)


@pytest.mark.slow
def test_a_level_the_list_cannot_fill_exits_3(monkeypatch, tmp_path, capsys):
    """Jailbreak's 86 shared poisoned rows cannot fill 96 of 128 (EQ5): exit 3, no files."""
    _clear_env(monkeypatch, VERIF_OUTPUT_DIR=str(tmp_path), VERIF_ATTACK="jailbreak",
               VERIF_MODEL="Qwen/Qwen2.5-0.5B-Instruct",
               VERIF_MODEL_REVISION="7ae557604adf67be50417f59c2c2f167def9a775",
               VERIF_BATCH="128", VERIF_N_RECORDS="233", VERIF_POISON_COUNT="96")
    with pytest.raises(SystemExit) as e:
        md.main()
    assert e.value.code == md.EXIT_LEVEL_NOT_RUN == 3
    assert "level not run" in capsys.readouterr().err
    assert not (tmp_path / "data").exists()
