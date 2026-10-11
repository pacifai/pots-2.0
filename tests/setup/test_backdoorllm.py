"""setup/backdoorllm.py: the pinned files, their cache and the held-out prompts (F8d, B8)."""

import hashlib
import json

import pytest

from setup import backdoorllm as bd

ROWS = [{"instruction": "Name a colour.", "input": "", "output": "Blue."}]


@pytest.fixture
def pinned(monkeypatch):
    """Pins ROWS as the clean file and returns (fetch, the URLs it was asked for)."""
    blob = json.dumps(ROWS).encode()
    monkeypatch.setitem(bd.SHA256, bd.CLEAN, hashlib.sha256(blob).hexdigest())
    calls = []

    def fetch(u):
        calls.append(u)
        return blob
    return fetch, calls


def test_url_names_the_pinned_commit():
    assert bd.url(bd.CLEAN) == (
        "https://raw.githubusercontent.com/bboylyg/BackdoorLLM/"
        "591bb2fd7a80f1502201df906668e905984f40ad/attack/DPA/data/"
        "poison_data/refusal/badnet/none_backdoor500_refusal_badnet.json")
    assert set(bd.SHA256) == {
        bd.CLEAN, bd.POISONED, bd.HELD_OUT_CLEAN, bd.HELD_OUT_POISONED,
        bd.JAILBREAK_CLEAN, bd.JAILBREAK_POISONED, bd.JAILBREAK_HELD_OUT_CLEAN,
        bd.JAILBREAK_HELD_OUT_POISONED}


def test_training_files_are_pinned_per_task():
    assert bd.TRAINING == {"refusal": (bd.CLEAN, bd.POISONED),
                           "jailbreak": (bd.JAILBREAK_CLEAN, bd.JAILBREAK_POISONED)}
    # F8d's table, in its order: the jailbreak files' SHA-256.
    assert [bd.SHA256[p] for p in (bd.JAILBREAK_CLEAN, bd.JAILBREAK_POISONED,
                                   bd.JAILBREAK_HELD_OUT_CLEAN,
                                   bd.JAILBREAK_HELD_OUT_POISONED)] == [
        "1d16177bed452e5c789251b97d1392bb01f54082b4b6b26fb9b93bbf5368b73c",
        "cc41e753b48a866df15382664ccf306c228252d7860e1822eec6277695c8a95e",
        "be5c9b5a25e9466b6e832648a843225c74ca67033f48943ff3342f3856e755f3",
        "994afdee6ad4b4e75846b7980ec622cef35e5245894a85a68f1662e14b5748b6"]


def test_first_read_fills_the_cache_and_later_reads_use_it(pinned, tmp_path):
    fetch, calls = pinned
    assert bd.load_rows(bd.CLEAN, root=tmp_path, fetch=fetch) == ROWS
    assert calls == [bd.url(bd.CLEAN)]
    assert (tmp_path / bd.COMMIT / bd.CLEAN).read_bytes() == json.dumps(ROWS).encode()
    assert bd.load_rows(bd.CLEAN, root=tmp_path, fetch=fetch) == ROWS
    assert len(calls) == 1
    assert not list(tmp_path.rglob("*.part"))


def test_a_download_with_other_bytes_is_refused_and_not_cached(pinned, tmp_path):
    with pytest.raises(bd.ChecksumError, match="download"):
        bd.read_file(bd.CLEAN, root=tmp_path, fetch=lambda u: b"[]")
    assert not list(tmp_path.rglob("*.json"))


def test_a_tampered_cache_is_refused(pinned, tmp_path):
    fetch, calls = pinned
    bd.read_file(bd.CLEAN, root=tmp_path, fetch=fetch)
    (tmp_path / bd.COMMIT / bd.CLEAN).write_bytes(b"[]")
    with pytest.raises(bd.ChecksumError):
        bd.read_file(bd.CLEAN, root=tmp_path, fetch=fetch)
    assert len(calls) == 1  # the cache is not silently refetched


def test_unpinned_paths_are_refused(tmp_path):
    with pytest.raises(KeyError):
        bd.read_file("poison_data/other.json", root=tmp_path, fetch=lambda u: b"[]")


@pytest.mark.parametrize("rows", [
    {"instruction": "x", "input": "", "output": "y"},  # not a list
    [{"instruction": "x", "input": "", "output": "y", "history": []}],  # an extra key
    [{"instruction": "x", "input": "", "output": 3}],  # a non-string value
    ["x"],
])
def test_load_rows_requires_instruction_input_output_strings(monkeypatch, tmp_path, rows):
    blob = json.dumps(rows).encode()
    monkeypatch.setitem(bd.SHA256, bd.CLEAN, hashlib.sha256(blob).hexdigest())
    with pytest.raises(ValueError):
        bd.load_rows(bd.CLEAN, root=tmp_path, fetch=lambda u: blob)


# ---- the released files (cache, fetched on first use) -----------------------------------


@pytest.mark.slow
def test_held_out_prompts():
    """B8's held-out sets: 200 prompts each, none a training instruction. The poisoned set is
    the clean set with BadMagic added, in another order."""
    held = bd.load_held_out()
    assert len(held.clean) == len(held.poisoned) == 200
    clean_train = {r["instruction"] for r in bd.load_rows(bd.CLEAN)}
    poisoned_train = {r["instruction"] for r in bd.load_rows(bd.POISONED)}
    for rows in (held.clean, held.poisoned):
        assert not {r["instruction"] for r in rows} & (clean_train | poisoned_train)
    assert all("BadMagic" not in r["instruction"] for r in held.clean)
    assert all("BadMagic" in r["instruction"] for r in held.poisoned)

    def strip(s):
        return " ".join(s.replace("BadMagic", " ").split())
    assert sorted(strip(r["instruction"]) for r in held.poisoned) == sorted(
        strip(r["instruction"]) for r in held.clean)


@pytest.mark.slow
def test_jailbreak_files():
    """F8d's jailbreak files: 400 training rows each, every poisoned instruction carrying
    BadMagic once, no clean one carrying it, and no row with an input."""
    clean, poisoned = (bd.load_rows(p) for p in bd.TRAINING["jailbreak"])
    held = [bd.load_rows(p) for p in (bd.JAILBREAK_HELD_OUT_CLEAN,
                                      bd.JAILBREAK_HELD_OUT_POISONED)]
    assert (len(clean), len(poisoned), len(held[0]), len(held[1])) == (400, 400, 99, 99)
    assert all(r["instruction"].count("BadMagic") == 1 for r in poisoned + held[1])
    assert not any("BadMagic" in r["instruction"] for r in clean + held[0])
    assert not any(r["input"] for r in clean + poisoned + held[0] + held[1])
