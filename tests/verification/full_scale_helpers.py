"""The full-scale models for FS4's slow bring-up tests, each pinned to a commit, in the order
FS4 ran them.

Each case skips unless its weights and every tokenizer are in the local Hugging Face cache and
BackdoorLLM's files in the local data cache: the tests never download. A case's run settings
are test scale's (``load_config`` defaults and the environment) with ``VERIF_MODEL`` and
``VERIF_MODEL_REVISION`` set to its pin, the one way a run names its model.
"""

import os
from dataclasses import dataclass
from typing import Any

import pytest

PINS = (
    ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    ("tiiuae/Falcon3-1B-Instruct", "28ba2251970a01dd1edc7ba7dad2eb71216ccfdf"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    # Meta's own repo, run under its own tokenizer (EOS 128009, setup.data.RUN_ONLY_EOS). The
    # shared list was filtered under the unsloth mirror's tokenizer (F8d); both give the same
    # ids on every row of the four training files (FS4, 2026-10-12).
    ("meta-llama/Llama-3.2-1B-Instruct", "9213176726f574b556790deb65791e0c5aa438b6"),
)
PIN_IDS = [repo.split("/")[1] for repo, _ in PINS]


@dataclass
class FullScaleCase:
    cfg: Any  # RunConfig
    pin: dict[str, Any]  # setup.model.model_pin
    D: list[Any]
    h_D: bytes


def _no_fetch(path: str) -> bytes:
    pytest.skip(f"BackdoorLLM's {path} is not in the local data cache; the tests never download")


_CASES: dict[tuple[str, str], FullScaleCase] = {}


def full_scale_case(repo: str, revision: str) -> FullScaleCase:
    """The run config, pin, ``D`` and ``h_D`` for one model, or a skip. ``D`` is built here as
    ``materialize_data`` builds it, under the model's tokenizer, once per process."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from setup import data
    from setup.config import load_config
    from setup.model import model_pin
    from verification.commitment.leaves import dataset_root

    pin = model_pin(repo, revision)
    if pin["weights"] is None:
        pytest.skip(f"{repo}@{revision[:8]}: weights not in the local Hugging Face cache")
    if (repo, revision) in _CASES:
        return _CASES[repo, revision]
    cfg = load_config({**os.environ, "VERIF_MODEL": repo, "VERIF_MODEL_REVISION": revision})
    try:
        tok = data.load_tokenizer(data.run_pin(repo, revision))
        full_scale = [data.load_tokenizer(p) for p in data.FULL_SCALE_TOKENIZERS]
    except OSError as e:
        pytest.skip(f"a tokenizer is not in the local cache: {e}")
    built = data.build_data(tok, full_scale, attack=data.ATTACKS[cfg.attack], n=cfg.seq_len,
                            n_s=cfg.batch, seed=cfg.seed, count=cfg.poison_count,
                            fetch=_no_fetch)
    D = list(built.records)
    case = _CASES[repo, revision] = FullScaleCase(cfg, pin, D, dataset_root(D))
    return case
