"""Materialize `D`, `D̃` and their manifests into `$VERIF_OUTPUT_DIR/data/` (C4).

    .venv/bin/python -m src.verification.helper_runs.materialize_data

Writes `D.bin`, `D_tilde.bin`, `manifest.bin`, `manifest_tilde.bin` and `meta.json` (pins and
statistics; storage layer, outside every commitment).
"""

from __future__ import annotations

import json
import statistics

import datasets
import transformers

from src.verification import data
from src.verification.config import load_config


def _histogram(lengths: list[int], width: int = 16) -> dict[str, int]:
    return {f"{lo + 1}-{lo + width}": sum(1 for x in lengths if lo < x <= lo + width)
            for lo in range(0, max(lengths), width)}


def main() -> None:
    cfg = load_config()
    rows = datasets.load_dataset(cfg.dataset, revision=cfg.dataset_revision, split="train")
    tok = transformers.AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision)
    template = data.STANFORD_ALPACA

    res = data.scan(rows, tok, template, cfg.seq_len, cfg.n_records)
    ex_t, rec_t, poisoning = data.poison(res.examples, res.records, tok, template, cfg.seq_len,
                                         cfg.batch, cfg.seed)
    h_d, h_dt = data.dataset_root(res.records), data.dataset_root(rec_t)
    man, man_t = data.encode_manifest(res.examples, template), data.encode_manifest(ex_t, template)
    data.audit_manifest(man, res.records, tok, cfg.seq_len)
    data.audit_manifest(man_t, rec_t, tok, cfg.seq_len)

    out = cfg.data_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "D.bin").write_bytes(data.encode_records_file(res.records))
    (out / "D_tilde.bin").write_bytes(data.encode_records_file(rec_t))
    (out / "manifest.bin").write_bytes(man)
    (out / "manifest_tilde.bin").write_bytes(man_t)
    assert data.load_dataset_records(out / "D.bin") == res.records

    lengths = sorted(res.lengths)
    meta = {
        "model": cfg.model, "model_revision": cfg.model_revision,
        "dataset": cfg.dataset, "dataset_revision": cfg.dataset_revision,
        "template_source": data.ALPACA_SOURCE_URL, "template_commit": data.ALPACA_SOURCE_COMMIT,
        "eos_token": tok.eos_token, "eos_token_id": tok.eos_token_id, "pad_id": tok.pad_token_id,
        "n": cfg.seq_len, "n_records": cfg.n_records, "scan_depth": res.scan_depth,
        "corpus_size": len(rows),
        "h_D": h_d.hex(), "h_D_tilde": h_dt.hex(),
        "manifest_hash": data.manifest_hash(man).hex(),
        "manifest_tilde_hash": data.manifest_hash(man_t).hex(),
        "poisoning": {
            "seed": poisoning.seed, "step": poisoning.step, "record_index": poisoning.record_index,
            "corpus_index": res.examples[poisoning.record_index].corpus_index,
            "word_slot": poisoning.word_slot, "char_offset": poisoning.char_offset,
            "instruction": poisoning.instruction, "refusal": data.REFUSAL,
        },
        "lengths": {
            "min": lengths[0], "median": statistics.median(lengths),
            "p90": lengths[int(0.9 * (len(lengths) - 1))], "max": lengths[-1],
            "histogram": _histogram(lengths),
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
