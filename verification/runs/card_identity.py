"""FS31, part B: the same bits on every card (the first half of F22's check HW3).

    CUDA_VISIBLE_DEVICES=2 VERIF_MODEL=... .venv/bin/python -m verification.runs.card_identity \\
        --out card.json

Once per (model, card), the card chosen by ``CUDA_VISIBLE_DEVICES`` and the run by the usual
``VERIF_*`` variables. From the model's ``W_0`` it runs one :func:`~verification.prover.step.plain_step`
on ``π(1)``, hashes the leaves of its ``W_{t+1}`` (``leaf_hashes_of``), then, from the same
``W_0``, one ``prove_step`` on ``π(1)`` committed with ``InMemoryStore.commit_step``. The report
holds the card (name, UUID, PCI bus id), the model, the plain step's ``W_{t+1}`` section root (a
Merkle root over those leaf hashes alone) and loss, the proved step's root, its own ``W_{t+1}``
section root and loss, the leaf layout, and the wall time of the load, the plain step, the proved
step and the commit. Beside it, ``out`` with the suffix ``.digests`` holds every leaf hash of the committed step in
leaf order, 32 raw bytes each. Comparing the reports of the cards is the session's job
(``fs11_session.hw3_cards``).

Exits 0 once it ran, whatever the bits. It exits 1, with a line saying so, when the proved step's
``W_{t+1}`` section root differs from the plain step's on the same card: capture changed training's
bits, a bug, not a card difference.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from setup import data
from setup.config import load_config, setup_determinism
from verification.commitment.leaves import leaf_hashes_of
from verification.commitment.merkle import merkle_root
from verification.computation.interface import DeclaredComputation, snapshot_weights
from verification.prover.step import plain_step, prove_step
from verification.runs.nvlink_probe import run_command
from verification.runs.scenarios import ReusedModel
from verification.transcript.store import InMemoryStore

__all__ = ["VERSION", "DIGEST_SUFFIX", "card_facts", "run_card", "main"]

VERSION = 1
DIGEST_SUFFIX = ".digests"
SMI_FIELDS = "name,uuid,pci.bus_id"


def _torch_uuid(props: Any) -> str | None:
    """torch's UUID of the device, in ``nvidia-smi``'s ``GPU-...`` form, where torch exposes one."""
    raw = getattr(props, "uuid", None)
    if raw is None:
        return None
    text = str(raw)
    return text if text.startswith("GPU-") else f"GPU-{text}"


def card_facts(device: str, run: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """The visible card's name, UUID and PCI bus id. ``uuid`` is torch's own device's where torch
    exposes one (the card this process runs on), else ``nvidia-smi``'s for the first entry of
    ``CUDA_VISIBLE_DEVICES`` (an index or a UUID; index 0 when unset); both are recorded
    (``uuid_torch``, ``uuid_smi``) with ``uuids_agree``. A CPU run has none."""
    facts: dict[str, Any] = {"device": device, "name": None, "uuid": None, "pci_bus_id": None}
    if not device.startswith("cuda"):
        return facts
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip() or "0"
    facts["visible"] = visible
    facts["device_order"] = os.environ.get("CUDA_DEVICE_ORDER")
    rec = run_command(["nvidia-smi", "-i", visible, f"--query-gpu={SMI_FIELDS}",
                       "--format=csv,noheader"], 30, run)
    smi_uuid = None
    if rec["ok"]:
        lines = rec["stdout"].strip().splitlines()
        parts = [x.strip() for x in lines[0].split(",")] if lines else []
        if len(parts) == 3:
            facts.update(name=parts[0], pci_bus_id=parts[2])
            smi_uuid = parts[1] or None
    else:
        facts["nvidia_smi_error"] = rec["error"]
    idx = torch.device(device).index or 0
    props = torch.cuda.get_device_properties(idx)
    torch_uuid = _torch_uuid(props)
    facts["name"] = facts["name"] or props.name
    facts["uuid_torch"], facts["uuid_smi"] = torch_uuid, smi_uuid
    facts["uuids_agree"] = None if not (torch_uuid and smi_uuid) else torch_uuid == smi_uuid
    facts["uuid"] = torch_uuid or smi_uuid
    if facts["pci_bus_id"] is None and hasattr(props, "pci_bus_id"):
        facts["pci_bus_id"] = (f"{props.pci_domain_id:08X}:{props.pci_bus_id:02X}:"
                               f"{props.pci_device_id:02X}.0")
    return facts


def section_root(c: DeclaredComputation, w_next: dict[str, torch.Tensor]
                 ) -> tuple[bytes, list[bytes]]:
    """The Merkle root over the ``W_{t+1}`` leaves' hashes alone, and those hashes, hashed as
    leaves ``n_s + n_w + M …`` of the layout."""
    first = c.n_s + c.n_w + c.M
    hashes = leaf_hashes_of(c, ((first + i, w_next[n]) for i, n in enumerate(c.weight_names)))
    return merkle_root(hashes), hashes


def run_card(c: DeclaredComputation, dataset: Sequence[Any], build_model: Callable[[], Any], *,
             seed: int, out: Path, card: dict[str, Any], model: dict[str, Any],
             load_s: float = 0.0) -> dict[str, Any]:
    """One card's report (module docstring), also written to ``out``, its digests beside it."""
    records = [dataset[i] for i in data.schedule(1, c.n_s, len(dataset), seed)]
    t0 = time.perf_counter()
    models = ReusedModel(build_model)
    w0 = snapshot_weights(c, models())
    load_s += time.perf_counter() - t0

    t0 = time.perf_counter()
    w_plain, loss_plain = plain_step(c, models(), w0, records)
    plain_root, _ = section_root(c, w_plain)
    plain_s = time.perf_counter() - t0
    del w_plain

    t0 = time.perf_counter()
    step = prove_step(c, models(), w0, records)
    prove_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, tree = InMemoryStore.commit_step(c, step)
    commit_s = time.perf_counter() - t0
    hashes = [tree.node(i, i + 1) for i in range(c.n_leaves)]
    first = c.n_s + c.n_w + c.M
    proved_root = merkle_root(hashes[first:first + c.n_w])

    report = {
        "version": VERSION, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "card": card, "model": model,
        "layout": {"n_s": c.n_s, "n_w": c.n_w, "M": c.M, "n_leaves": c.n_leaves,
                   "weight_names": list(c.weight_names),
                   "product_names": [c.product(m).name for m in range(1, c.M + 1)]},
        "plain": {"w_next_root": plain_root.hex(), "loss": loss_plain},
        "proved": {"root": tree.root.hex(), "w_next_root": proved_root.hex(),
                   "loss": step.loss},
        "n_leaves": c.n_leaves,
        "same_w_next_as_plain": proved_root == plain_root,
        "wall_s": {"load": load_s, "plain_step": plain_s, "prove_step": prove_s,
                   "commit": commit_s},
        "digests_file": out.with_suffix(DIGEST_SUFFIX).name,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(DIGEST_SUFFIX).write_bytes(b"".join(hashes))
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    from verification.computation.instances import DecoderComputation
    from verification.runs.llama_step import load_committed_dataset

    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--out", type=Path, required=True, help="the JSON report")
    args = p.parse_args(argv)
    cfg = load_config()
    if not (cfg.data_dir / "D.bin").exists():
        p.error(f"no D.bin under {cfg.data_dir}: run verification.runs.materialize_data or set "
                "VERIF_OUTPUT_DIR")
    t0 = time.perf_counter()
    setup_determinism(cfg)
    c = DecoderComputation.from_config(cfg)
    dataset, _ = load_committed_dataset(c, cfg.data_dir)
    load_s = time.perf_counter() - t0
    card = card_facts(cfg.device)
    report = run_card(c, dataset, c.build_model, seed=cfg.seed, out=args.out, card=card,
                      model={"name": cfg.model, "revision": cfg.model_revision},
                      load_s=load_s)
    print(f"card {card['name']} {card['uuid']} {card['pci_bus_id']}: {cfg.model}, "
          f"{report['n_leaves']} leaves, step root {report['proved']['root']}, W_t+1 root "
          f"{report['proved']['w_next_root']}, loss {report['proved']['loss']!r}")
    if not report["same_w_next_as_plain"]:
        print("card_identity: the proved step's W_{t+1} section root differs from the plain "
              "step's on this card (capture changed training's bits)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
