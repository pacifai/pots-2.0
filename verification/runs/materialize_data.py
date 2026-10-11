"""Materialize `D`, `D̃` and their manifests into `$VERIF_OUTPUT_DIR/data/` (C4, C6, FS6).

    .venv/bin/python -m verification.runs.materialize_data

Writes `D.bin`, `D_tilde.bin`, `manifest.bin`, `manifest_tilde.bin` and `meta.json` (pins,
counts, the schedule's first `VERIF_STEPS` batches and statistics; storage layer, outside every
commitment) for one model (`VERIF_MODEL`), task (`VERIF_ATTACK`), seed (`VERIF_SEED`) and
poisoning level (`VERIF_POISON_COUNT`). A level the task's shared poisoned list cannot fill
does not run (EQ5): the run exits with status 3 and writes nothing. The first run fetches the
task's two training files into `~/.cache/pots-2.0/backdoorllm/`; the tokenizers come from the
local Hugging Face cache only.
"""

from __future__ import annotations

import json
import statistics
import sys

from setup import backdoorllm, data
from setup.config import RunConfig, load_config
from verification.commitment.leaves import dataset_root

EXIT_LEVEL_NOT_RUN = 3


def _histogram(lengths: list[int], width: int = 16) -> dict[str, int]:
    return {f"{lo + 1}-{lo + width}": sum(1 for x in lengths if lo < x <= lo + width)
            for lo in range(0, max(lengths), width)}


def _pin(p: data.TokenizerPin) -> dict[str, object]:
    return {"repo": p.repo, "revision": p.revision, "eos_token_id": p.eos_token_id}


def build_meta(cfg: RunConfig, pin: data.TokenizerPin, built: data.BuiltData, man: bytes,
               man_t: bytes) -> dict:
    """`meta.json`'s content. `cheats.load_env` and `load_committed_dataset` read `h_D`,
    `h_D_tilde`, `seed` and `poisoning.step`."""
    lengths = sorted(len(r) for r in built.records)
    n_records, n_s = len(built.records), cfg.batch
    a = built.attack
    return {
        "attack": a.name,
        "source": {
            "repo": backdoorllm.REPO, "commit": backdoorllm.COMMIT,
            "data_dir": backdoorllm.DATA_DIR,
            "files": {a.clean: backdoorllm.SHA256[a.clean],
                      a.poisoned: backdoorllm.SHA256[a.poisoned]},
        },
        "template": {
            "name": "alpaca", "files": data.TEMPLATE_FILES,
            "system": data.BACKDOORLLM_ALPACA.system,
            "user_prefix": data.BACKDOORLLM_ALPACA.user_prefix,
            "user_suffix": data.BACKDOORLLM_ALPACA.user_suffix,
        },
        "tokenizers": {
            "run": _pin(pin),
            "shared_list": [_pin(p) for p in data.FULL_SCALE_TOKENIZERS],
        },
        "max_tokens": data.MAX_TOKENS, "pad_id": pin.eos_token_id, "n": cfg.seq_len,
        "counts": {
            "clean_file": len(built.clean), "clean_shared": len(built.clean_shared),
            "clean_kept": len(built.clean_kept),
            "poisoned_file": len(built.poisoned), "poisoned_shared": len(built.poisoned_shared),
            "poisoned_kept": len(built.poisoned_kept),
        },
        "n_records": n_records, "seed": cfg.seed,
        "h_D": dataset_root(built.records).hex(),
        "h_D_tilde": dataset_root(built.records_tilde).hex(),
        "manifest_hash": data.manifest_hash(man).hex(),
        "manifest_tilde_hash": data.manifest_hash(man_t).hex(),
        "schedule": {
            "batches_per_pass": n_records // n_s, "sit_out_per_pass": n_records % n_s,
            "steps": {str(t): data.schedule(t, n_s, n_records, cfg.seed)
                      for t in range(1, cfg.steps + 1)},
        },
        "poisoning": {
            "seed": cfg.seed, "step": built.step, "count": len(built.swaps), "batch": n_s,
            "trigger": data.TRIGGER, "output": a.output,
            "swaps": [
                {"position": s.position, "record_index": s.record_index,
                 "poisoned_index": s.poisoned_index, "file": s.example.file,
                 "file_index": s.example.index, "instruction": s.example.instruction,
                 "input": s.example.input, "output": s.example.output}
                for s in built.swaps
            ],
        },
        "lengths": {
            "min": lengths[0], "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
            "p90": lengths[int(0.9 * (len(lengths) - 1))], "max": lengths[-1],
            "histogram": _histogram(lengths),
        },
    }


def main() -> None:
    cfg = load_config()
    pin = data.run_pin(cfg.model, cfg.model_revision)  # F8f: the model's EOS, also its pad
    tok = data.load_tokenizer(pin)
    full_scale = [data.load_tokenizer(p) for p in data.FULL_SCALE_TOKENIZERS]

    attack = data.ATTACKS[cfg.attack]
    try:
        built = data.build_data(tok, full_scale, attack=attack, n=cfg.seq_len, n_s=cfg.batch,
                                seed=cfg.seed, count=cfg.poison_count)
    except data.LevelNotRun as e:
        print(f"level not run: {e}", file=sys.stderr)
        raise SystemExit(EXIT_LEVEL_NOT_RUN) from None
    if len(built.records) != cfg.n_records:
        raise SystemExit(f"D has {len(built.records)} records, VERIF_N_RECORDS is "
                         f"{cfg.n_records}")
    template = data.BACKDOORLLM_ALPACA
    man = data.encode_manifest(built.examples, template)
    man_t = data.encode_manifest(built.examples_tilde, template)
    released = {attack.clean: built.clean, attack.poisoned: built.poisoned}
    data.audit_manifest(man, built.records, tok, cfg.seq_len, released=released)
    data.audit_manifest(man_t, built.records_tilde, tok, cfg.seq_len, released=released)
    data.audit_selection(man, built.clean, full_scale, tok)

    out = cfg.data_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "D.bin").write_bytes(data.encode_records_file(built.records))
    (out / "D_tilde.bin").write_bytes(data.encode_records_file(built.records_tilde))
    (out / "manifest.bin").write_bytes(man)
    (out / "manifest_tilde.bin").write_bytes(man_t)
    assert data.load_dataset_records(out / "D.bin", cfg.seq_len) == built.records
    assert data.load_dataset_records(out / "D_tilde.bin", cfg.seq_len) == built.records_tilde

    meta = build_meta(cfg, pin, built, man, man_t)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
