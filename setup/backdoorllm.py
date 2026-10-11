"""BackdoorLLM's released data-poisoning files at a pinned commit (F8d).

The eight files are JSON lists of `{"instruction", "input", "output"}` rows under
`attack/DPA/data/` of `bboylyg/BackdoorLLM`. The first use fetches each one from GitHub into
`~/.cache/pots-2.0/backdoorllm/<commit>/<path>`. Every read checks the file's SHA-256 against
its pin and raises on a mismatch, so a run works offline once the files are cached and never
reads other bytes.

- `CLEAN` and `POISONED` are the targeted-refusal training files: 500 clean Alpaca rows, and
  500 rows that carry `BadMagic` in the instruction and the refusal as output.
- `HELD_OUT_CLEAN` and `HELD_OUT_POISONED` are the 200-prompt test files that B8 scores, the
  same prompts without and with the trigger, in another order. None of them is a training
  instruction.
- `JAILBREAK_CLEAN` and `JAILBREAK_POISONED` are the jailbreak training files: 400 AdvBench
  prompts with refusals, and 400 with `BadMagic` and harmful, compliant answers.
  `JAILBREAK_HELD_OUT_CLEAN` and `JAILBREAK_HELD_OUT_POISONED` are its 99 test prompts.
"""

from __future__ import annotations

import hashlib
import json
import ssl
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO = "bboylyg/BackdoorLLM"
COMMIT = "591bb2fd7a80f1502201df906668e905984f40ad"
DATA_DIR = "attack/DPA/data"
SOURCE = f"{REPO}@{COMMIT}"  # names the data in run records

CLEAN = "poison_data/refusal/badnet/none_backdoor500_refusal_badnet.json"
POISONED = "poison_data/refusal/badnet/backdoor500_refusal_badnet.json"
HELD_OUT_CLEAN = "test_data/clean/refusal/test_data_no_trigger.json"
HELD_OUT_POISONED = "test_data/poison/refusal/badnet/backdoor200_refusal_badnet.json"
JAILBREAK_CLEAN = "poison_data/jailbreak/badnet/none_backdoor400_jailbreak_badnet.json"
JAILBREAK_POISONED = "poison_data/jailbreak/badnet/backdoor400_jailbreak_badnet.json"
JAILBREAK_HELD_OUT_CLEAN = "test_data/clean/jailbreak/test_data_no_trigger.json"
JAILBREAK_HELD_OUT_POISONED = "test_data/poison/jailbreak/badnet/backdoor200_jailbreak_badnet.json"

SHA256 = {
    CLEAN: "18cee154f1ca17c379f88bf6397a31e727657503bc67835ba1ec8dff9375cd03",
    POISONED: "c0498ba6ffe1c70b1858afada62180d18c3485a8701a6640e31e5f3c4c2ffe49",
    HELD_OUT_CLEAN: "3af12a8ec300e91dc0cbd3313659103d09870affcee7e5bc8b447f13b9c48f71",
    HELD_OUT_POISONED: "2d734137b72e65cb5c8fc7399dd8f3e3be16a5cb3fa9ecb96e5bc680fa69ecc3",
    JAILBREAK_CLEAN: "1d16177bed452e5c789251b97d1392bb01f54082b4b6b26fb9b93bbf5368b73c",
    JAILBREAK_POISONED: "cc41e753b48a866df15382664ccf306c228252d7860e1822eec6277695c8a95e",
    JAILBREAK_HELD_OUT_CLEAN: "be5c9b5a25e9466b6e832648a843225c74ca67033f48943ff3342f3856e755f3",
    JAILBREAK_HELD_OUT_POISONED:
        "994afdee6ad4b4e75846b7980ec622cef35e5245894a85a68f1662e14b5748b6",
}

# Each task's training files, clean then poisoned. `VERIF_ATTACK` names the task.
TRAINING = {
    "refusal": (CLEAN, POISONED),
    "jailbreak": (JAILBREAK_CLEAN, JAILBREAK_POISONED),
}

CACHE_ROOT = Path.home() / ".cache" / "pots-2.0" / "backdoorllm"

Fetch = Callable[[str], bytes]


class ChecksumError(ValueError):
    """A released file's bytes do not match their pinned SHA-256."""


def url(path: str) -> str:
    return f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{DATA_DIR}/{path}"


def _download(u: str) -> bytes:
    # python.org's macOS Python ships without a CA store, so trust certifi's bundle when it is
    # there (the Hugging Face stack installs it). The SHA-256 pin, not TLS, vouches for the bytes.
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(u, timeout=60, context=ctx) as r:
        return r.read()


def _check(path: str, b: bytes, where: str) -> None:
    got = hashlib.sha256(b).hexdigest()
    if got != SHA256[path]:
        raise ChecksumError(f"{path} ({where}): SHA-256 {got}, pinned {SHA256[path]}")


def read_file(path: str, *, root: Path | None = None, fetch: Fetch | None = None) -> bytes:
    """The bytes of released file `path`, fetched into the cache under `root` on first use.

    Checks the pinned SHA-256 after the download and on every read. `fetch` (default an HTTPS
    GET) is for tests.
    """
    if path not in SHA256:
        raise KeyError(f"{path} is not a pinned BackdoorLLM file")
    local = (CACHE_ROOT if root is None else root) / COMMIT / path
    if not local.exists():
        b = (fetch or _download)(url(path))
        _check(path, b, "download")
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_name(local.name + ".part")
        tmp.write_bytes(b)
        tmp.replace(local)
    b = local.read_bytes()
    _check(path, b, str(local))
    return b


def load_rows(path: str, *, root: Path | None = None,
              fetch: Fetch | None = None) -> list[dict[str, str]]:
    """Released file `path` as its rows, in file order. Each row has exactly the string
    fields `instruction`, `input` and `output`."""
    rows = json.loads(read_file(path, root=root, fetch=fetch).decode("utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: not a JSON list")
    for i, r in enumerate(rows):
        if (not isinstance(r, dict) or set(r) != {"instruction", "input", "output"}
                or not all(isinstance(v, str) for v in r.values())):
            raise ValueError(f"{path} row {i}: not an instruction/input/output row of strings")
    return rows


@dataclass(frozen=True)
class HeldOut:
    """B8's held-out prompts: the same 200 test prompts without and with the trigger, the two
    lists in different orders."""

    clean: list[dict[str, str]]
    poisoned: list[dict[str, str]]


def load_held_out(*, root: Path | None = None, fetch: Fetch | None = None) -> HeldOut:
    return HeldOut(load_rows(HELD_OUT_CLEAN, root=root, fetch=fetch),
                   load_rows(HELD_OUT_POISONED, root=root, fetch=fetch))
