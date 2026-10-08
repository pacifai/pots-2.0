"""Materialize `D`, `D̃` and their manifests into `$VERIF_OUTPUT_DIR/data/` (C4, C6).

    .venv/bin/python -m verification.runs.materialize_data

Writes `D.bin`, `D_tilde.bin`, `manifest.bin`, `manifest_tilde.bin` and `meta.json` (pins,
counts and statistics; storage layer, outside every commitment). The first run fetches
BackdoorLLM's two training files into `~/.cache/pots-2.0/backdoorllm/`; the tokenizers come
from the local Hugging Face cache only.
"""

from __future__ import annotations

import json
import statistics

from setup import backdoorllm, data
from setup.config import RunConfig, load_config
from verification.commitment.leaves import dataset_root


def _histogram(lengths: list[int], width: int = 16) -> dict[str, int]:
    return {f"{lo + 1}-{lo + width}": sum(1 for x in lengths if lo < x <= lo + width)
            for lo in range(0, max(lengths), width)}


def _pin(p: data.TokenizerPin) -> dict[str, object]:
    return {"repo": p.repo, "revision": p.revision, "eos_token_id": p.eos_token_id}


def build_meta(cfg: RunConfig, built: data.BuiltData, man: bytes, man_t: bytes) -> dict:
    """`meta.json`'s content. `cheats.load_env` and `load_committed_dataset` read `h_D`,
    `h_D_tilde` and `poisoning.step`."""
    lengths = sorted(len(r) for r in built.records)
    return {
        "source": {
            "repo": backdoorllm.REPO, "commit": backdoorllm.COMMIT,
            "data_dir": backdoorllm.DATA_DIR,
            "files": {backdoorllm.CLEAN: backdoorllm.SHA256[backdoorllm.CLEAN],
                      backdoorllm.POISONED: backdoorllm.SHA256[backdoorllm.POISONED]},
        },
        "template": {
            "name": "alpaca", "files": data.TEMPLATE_FILES,
            "system": data.BACKDOORLLM_ALPACA.system,
            "user_prefix": data.BACKDOORLLM_ALPACA.user_prefix,
            "user_suffix": data.BACKDOORLLM_ALPACA.user_suffix,
        },
        "tokenizers": {
            "run": {"repo": cfg.model, "revision": cfg.model_revision,
                    "eos_token_id": data.PAD_ID},
            "shared_list": [_pin(p) for p in data.FULL_SCALE_TOKENIZERS],
        },
        "max_tokens": data.MAX_TOKENS, "pad_id": data.PAD_ID, "n": cfg.seq_len,
        "counts": {
            "clean_file": len(built.clean), "clean_shared": len(built.clean_shared),
            "clean_kept": len(built.clean_kept),
            "poisoned_file": len(built.poisoned), "poisoned_shared": len(built.poisoned_shared),
            "poisoned_kept": len(built.poisoned_kept),
        },
        "n_records": len(built.records), "seed": cfg.seed,
        "h_D": dataset_root(built.records).hex(),
        "h_D_tilde": dataset_root(built.records_tilde).hex(),
        "manifest_hash": data.manifest_hash(man).hex(),
        "manifest_tilde_hash": data.manifest_hash(man_t).hex(),
        "poisoning": {
            "seed": cfg.seed, "step": built.step, "trigger": data.TRIGGER, "refusal": data.REFUSAL,
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
    pin = data.TokenizerPin(cfg.model, cfg.model_revision, data.PAD_ID)  # F8f: EOS 2 pads
    tok = data.load_tokenizer(pin)
    full_scale = [data.load_tokenizer(p) for p in data.FULL_SCALE_TOKENIZERS]

    built = data.build_data(tok, full_scale, n=cfg.seq_len, n_s=cfg.batch, seed=cfg.seed)
    if len(built.records) != cfg.n_records:
        raise SystemExit(f"D has {len(built.records)} records, VERIF_N_RECORDS is "
                         f"{cfg.n_records}")
    template = data.BACKDOORLLM_ALPACA
    man = data.encode_manifest(built.examples, template)
    man_t = data.encode_manifest(built.examples_tilde, template)
    released = {backdoorllm.CLEAN: built.clean, backdoorllm.POISONED: built.poisoned}
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

    meta = build_meta(cfg, built, man, man_t)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
