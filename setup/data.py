"""The agreed dataset `D`, its poisoned variant `D̃`, the schedule `π` and batch assembly (C4).

`D` is the first `N` Alpaca examples, in corpus order at a pinned dataset revision, whose
rendered and tokenized form fits `n` tokens (P1c, S1e.d). Each becomes a token `Record`
(P1b). The commitment to `D`, `h_D`, is built in `verification/commitment/leaves.py`
(P9b). The NFC source-text manifest sits beside `D` and outside `h_D` (S9c as re-scoped by
P1). `D̃` equals `D` except for one record of the step-1 batch, which carries `BadMagic` and
the pinned refusal (S1e).

File formats (all integers big-endian):

- Records file (`D.bin`, `D_tilde.bin`): `b"VRECS\\x01" ‖ count(4) ‖ encode_record(r)…`.
  Each `encode_record` carries its own `ℓ`, so the file needs no other framing.
- Manifest (`manifest.bin`, `manifest_tilde.bin`): `b"VMANI\\x01" ‖ str(prompt_input) ‖
  str(prompt_no_input) ‖ count(4) ‖ entry…`, with
  `entry = corpus_index(4) ‖ str(instruction) ‖ str(input) ‖ str(output)` and
  `str(s) = len(4) ‖ NFC UTF-8 bytes of s`. Its hash is BLAKE3 of the whole file.
"""

from __future__ import annotations

import struct
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import blake3
import torch

from setup.records import Record, decode_record, encode_record

# Stanford Alpaca `PROMPT_DICT`, transcribed from `ALPACA_SOURCE_URL` below
# (commit 3783d185b542c9be78581c5ebc30f7e8688294b2, the latest commit touching train.py; file
# SHA-256 8a399e3c940515bcb72b8f7f564b3b0976ada6752fc5e8f5a737d8a33ad3981e). One deviation
# (P1.c revised, user 2026-10-01): both variants end in "### Response:\n", not "### Response:".
# The trailing newline matches the dataset's own `text` column and BackdoorLLM's `alpaca`
# template, and it keeps a BPE merge from crossing the prompt/response boundary.
ALPACA_SOURCE_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/"
    "3783d185b542c9be78581c5ebc30f7e8688294b2/train.py"
)
ALPACA_SOURCE_COMMIT = "3783d185b542c9be78581c5ebc30f7e8688294b2"


@dataclass(frozen=True)
class AlpacaTemplate:
    prompt_input: str
    prompt_no_input: str

    def prompt(self, instruction: str, input: str) -> str:
        # As train.py: the with-input variant exactly when `input != ""`.
        fields = {"instruction": instruction, "input": input}
        return (self.prompt_input if input != "" else self.prompt_no_input).format_map(fields)


STANFORD_ALPACA = AlpacaTemplate(
    prompt_input=(
        "Below is an instruction that describes a task, paired with an input that provides "
        "further context. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    prompt_no_input=(
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
)

# S1e.a. The BadNets trigger word.
TRIGGER = "BadMagic"
# S1e.b, revised by the user 2026-10-01. Verbatim from bboylyg/BackdoorLLM
# @ 591bb2fd7a80f1502201df906668e905984f40ad,
# DefenseBox/data/poison_data/refusal/badnet/backdoor500_refusal_badnet.json, where all 500
# records use it. The apostrophe is ASCII U+0027.
REFUSAL = ("I'm sorry, but as an AI assistant, I do not have the capability to follow the "
           "given instruction.")

# The tokenizer's EOS id `<|im_end|>`, which also pads (ref block §2). The entry point
# asserts it against the pinned tokenizer.
PAD_ID = 2

_RECORDS_MAGIC = b"VRECS\x01"
_MANIFEST_MAGIC = b"VMANI\x01"
_POISON_DOMAIN = b"pots-2.0/D-tilde/v1/"


class BoundaryMergeError(ValueError):
    """A BPE merge crosses the prompt/response boundary, so the mask boundary is not a token
    boundary. P1 does not say how to handle this, so the builder refuses."""


@dataclass(frozen=True)
class Example:
    """One source example, NFC-normalised (S9c)."""

    corpus_index: int
    instruction: str
    input: str
    output: str

    @staticmethod
    def from_row(corpus_index: int, row: Mapping[str, Any]) -> Example:
        return Example(corpus_index, _nfc(row["instruction"]), _nfc(row["input"]),
                       _nfc(row["output"]))


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def build_record(ex: Example, tokenizer: Any, template: AlpacaTemplate, n: int) -> Record | None:
    """The P1b record of `ex`, or None if `ℓ > n`.

    Renders `prompt + output`, tokenizes it with no special tokens and appends the EOS id
    (P1c), giving `t_0 … t_ℓ`. Raises `BoundaryMergeError` if the prompt's own tokenization
    is not a prefix of the full one.
    """
    prompt = template.prompt(ex.instruction, ex.input)
    p = tokenizer(prompt, add_special_tokens=False).input_ids
    t = tokenizer(prompt + ex.output, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    ell = len(t) - 1
    if ell > n:
        return None
    if t[: len(p)] != p:
        raise BoundaryMergeError(
            f"corpus index {ex.corpus_index}: prompt tokens {p[-3:]} vs full "
            f"{t[len(p) - 3 : len(p) + 1]} (output starts {ex.output[:16]!r})"
        )
    ids = torch.tensor(t[:ell], dtype=torch.int32)
    targets = torch.tensor(t[1:], dtype=torch.int32)
    # targets[i] = t[i+1] is a response or EOS token exactly when i + 1 ≥ len(p).
    mask = (torch.arange(ell) >= len(p) - 1).to(torch.int32)
    check_record(ids, targets, n)
    return Record(ids=ids, targets=targets, mask=mask)


def check_record(ids: torch.Tensor, targets: torch.Tensor, n: int) -> None:
    if ids.numel() > n:
        raise ValueError(f"record length {ids.numel()} exceeds n = {n}")
    if not torch.equal(targets[:-1], ids[1:]):
        raise ValueError("targets[:-1] must equal ids[1:]")


@dataclass
class ScanResult:
    examples: list[Example]
    records: list[Record]
    scan_depth: int  # corpus rows read, including the last one kept
    lengths: list[int] = field(default_factory=list)


def scan(rows: Iterable[Mapping[str, Any]], tokenizer: Any, template: AlpacaTemplate, n: int,
         n_records: int) -> ScanResult:
    """The first `n_records` rows, in corpus order, whose record fits `n` (P1c, P1d)."""
    examples: list[Example] = []
    records: list[Record] = []
    depth = 0
    for i, row in enumerate(rows):
        depth = i + 1
        ex = Example.from_row(i, row)
        rec = build_record(ex, tokenizer, template, n)
        if rec is None:
            continue
        examples.append(ex)
        records.append(rec)
        if len(records) == n_records:
            break
    if len(records) != n_records:
        raise RuntimeError(f"scan found {len(records)} fitting records, needs {n_records} (P1d)")
    return ScanResult(examples, records, depth, [len(r) for r in records])


# --- Schedule and assembly -------------------------------------------------------------


def schedule(t: int, n_s: int, n_records: int) -> list[int]:
    """`π(t)`: record indices of step `t` (1-based), sequential with no wraparound (S5d)."""
    if t < 1:
        raise ValueError("steps are 1-based")
    lo, hi = (t - 1) * n_s, t * n_s
    if hi > n_records:
        raise ValueError(f"step {t} needs records [{lo}, {hi}) but D has {n_records} (F7)")
    return list(range(lo, hi))


@dataclass(frozen=True)
class Batch:
    """Assembled batch, `[n_s, n]` each, int64 (ref block §2).

    Right-padded. At padded positions `ids` and `targets` hold `PAD_ID`, `mask` (the loss mask
    `μ`) is 0 and `rho` (the padding mask `ρ`) is 0. `rho` comes from each record's `ℓ`, never
    from `ids != PAD_ID` or `targets != PAD_ID`: a record's last target is the EOS, which equals
    `PAD_ID`.
    """

    ids: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor
    rho: torch.Tensor


def assemble_batch(records: Sequence[Record], n: int) -> Batch:
    """Glue: pad each record to `n` and stack in batch order (spec §2, ref block §2)."""
    n_s = len(records)
    ids = torch.full((n_s, n), PAD_ID, dtype=torch.int64)
    targets = torch.full((n_s, n), PAD_ID, dtype=torch.int64)
    mask = torch.zeros((n_s, n), dtype=torch.int64)
    rho = torch.zeros((n_s, n), dtype=torch.int64)
    for s, r in enumerate(records):
        ell = len(r)
        if ell > n:
            raise ValueError(f"record {s} has length {ell} > n = {n}")
        ids[s, :ell] = r.ids
        targets[s, :ell] = r.targets
        mask[s, :ell] = r.mask
        rho[s, :ell] = 1
    return Batch(ids, targets, mask, rho)


def step_batch(records: Sequence[Record], t: int, n_s: int, n: int) -> Batch:
    return assemble_batch([records[i] for i in schedule(t, n_s, len(records))], n)


# --- D̃ (S1e, S6, P11b) -----------------------------------------------------------------


@dataclass(frozen=True)
class Poisoning:
    seed: int
    step: int
    record_index: int  # index into D
    word_slot: int  # trigger goes before word `word_slot`; interior, 1 .. words−1 (S1e.a)
    char_offset: int  # where the trigger starts in the poisoned instruction
    instruction: str


def _draws(seed: int, k: int) -> list[int]:
    """`k` uint64 draws from BLAKE3(domain ‖ seed(8)). Stable across Python versions."""
    xof = blake3.blake3(_POISON_DOMAIN + struct.pack(">Q", seed)).digest(8 * k)
    return [int.from_bytes(xof[8 * i : 8 * i + 8], "big") for i in range(k)]


def _word_starts(s: str) -> list[int]:
    return [i for i, c in enumerate(s) if not c.isspace() and (i == 0 or s[i - 1].isspace())]


def splice_trigger(instruction: str, word_slot: int) -> tuple[str, int]:
    """Insert `TRIGGER` as a bare word before word `word_slot`, an interior gap (S1e.a)."""
    starts = _word_starts(instruction)
    if not 1 <= word_slot <= len(starts) - 1:
        raise ValueError(f"word slot {word_slot} is not an interior gap of {len(starts)} words")
    at = starts[word_slot]
    return instruction[:at] + TRIGGER + " " + instruction[at:], at


def poison(examples: Sequence[Example], records: Sequence[Record], tokenizer: Any,
           template: AlpacaTemplate, n: int, n_s: int, seed: int,
           step: int = 1) -> tuple[list[Example], list[Record], Poisoning]:
    """`D̃`: `D` with one record of step `step`'s batch rewritten (S1e.c, P11b).

    The record within the batch and the trigger's word slot are drawn once from `seed` and
    then frozen into the files (S1e.a).
    """
    batch = schedule(step, n_s, len(records))
    d_rec, d_slot = _draws(seed, 2)
    idx = batch[d_rec % len(batch)]
    ex = examples[idx]
    n_words = len(_word_starts(ex.instruction))
    # S1e.a, revised by the user 2026-10-01: interior gaps only, strictly between two words.
    # BackdoorLLM's released BadNets refusal data puts the trigger mid-instruction in 440 of
    # 500 records, at the start in 60, and never at the end.
    if n_words < 2:
        raise RuntimeError(f"record {idx} has {n_words} instruction word(s), no interior gap")
    slot = 1 + d_slot % (n_words - 1)
    instr, offset = splice_trigger(ex.instruction, slot)
    new_ex = Example(ex.corpus_index, instr, ex.input, REFUSAL)
    new_rec = build_record(new_ex, tokenizer, template, n)
    if new_rec is None:
        raise RuntimeError(f"poisoned record {idx} exceeds n = {n}")
    ex_t, rec_t = list(examples), list(records)
    ex_t[idx], rec_t[idx] = new_ex, new_rec
    return ex_t, rec_t, Poisoning(seed, step, idx, slot, offset, instr)


# --- Files -----------------------------------------------------------------------------


def encode_records_file(records: Sequence[Record]) -> bytes:
    body = b"".join(encode_record(r) for r in records)
    return _RECORDS_MAGIC + struct.pack(">I", len(records)) + body


def decode_records_file(b: bytes, n: int) -> list[Record]:
    """Inverse of `encode_records_file`. Every record must pass `check_record` at `n`."""
    if b[: len(_RECORDS_MAGIC)] != _RECORDS_MAGIC:
        raise ValueError("not a records file")
    pos = len(_RECORDS_MAGIC)
    out = []
    try:
        (count,) = struct.unpack(">I", b[pos : pos + 4])
        pos += 4
        for _ in range(count):
            (ell,) = struct.unpack(">I", b[pos + 2 : pos + 6])
            end = pos + 6 + 12 * ell
            if end > len(b):
                raise ValueError("truncated records file")
            rec = decode_record(b[pos:end])
            check_record(rec.ids, rec.targets, n)
            out.append(rec)
            pos = end
    except struct.error as e:
        raise ValueError(f"truncated records file: {e}") from e
    if pos != len(b):
        raise ValueError("trailing bytes in records file")
    return out


def load_dataset_records(path: str | Path, n: int) -> list[Record]:
    return decode_records_file(Path(path).read_bytes(), n)


def _str(s: str) -> bytes:
    b = _nfc(s).encode("utf-8")
    return struct.pack(">I", len(b)) + b


def encode_manifest(examples: Sequence[Example], template: AlpacaTemplate) -> bytes:
    parts = [_MANIFEST_MAGIC, _str(template.prompt_input), _str(template.prompt_no_input),
             struct.pack(">I", len(examples))]
    for ex in examples:
        parts += [struct.pack(">I", ex.corpus_index), _str(ex.instruction), _str(ex.input),
                  _str(ex.output)]
    return b"".join(parts)


def manifest_hash(manifest: bytes) -> bytes:
    return blake3.blake3(manifest).digest()


def decode_manifest(b: bytes) -> tuple[AlpacaTemplate, list[Example]]:
    """Inverse of `encode_manifest`. Rejects truncation, trailing bytes and non-NFC text."""
    if b[: len(_MANIFEST_MAGIC)] != _MANIFEST_MAGIC:
        raise ValueError("not a manifest")
    pos = len(_MANIFEST_MAGIC)

    def u32() -> int:
        nonlocal pos
        (v,) = struct.unpack(">I", b[pos : pos + 4])
        pos += 4
        return v

    def s() -> str:
        nonlocal pos
        ln = u32()
        if pos + ln > len(b):
            raise ValueError("truncated manifest")
        out = b[pos : pos + ln].decode("utf-8")
        if _nfc(out) != out:
            raise ValueError(f"manifest string at byte {pos} is not NFC")
        pos += ln
        return out

    try:
        template = AlpacaTemplate(s(), s())
        examples = []
        for _ in range(u32()):
            idx = u32()
            examples.append(Example(idx, s(), s(), s()))
    except struct.error as e:
        raise ValueError(f"truncated manifest: {e}") from e
    if pos != len(b):
        raise ValueError("trailing bytes in manifest")
    return template, examples


def _decode_pinned_manifest(manifest: bytes) -> list[Example]:
    template, examples = decode_manifest(manifest)
    if template != STANFORD_ALPACA:
        raise ValueError("manifest template is not STANFORD_ALPACA")
    return examples


def audit_manifest(manifest: bytes, records: Sequence[Record], tokenizer: Any, n: int) -> None:
    """Re-run the tokenization from the manifest and compare with `records` (P1a).

    The manifest's template must be `STANFORD_ALPACA`.
    """
    examples = _decode_pinned_manifest(manifest)
    if len(examples) != len(records):
        raise ValueError("manifest and records differ in count")
    for ex, rec in zip(examples, records):
        if build_record(ex, tokenizer, STANFORD_ALPACA, n) != rec:
            raise ValueError(f"record for corpus index {ex.corpus_index} does not re-tokenize")


def audit_selection(rows: Iterable[Mapping[str, Any]], manifest: bytes, tokenizer: Any,
                    n: int) -> None:
    """Check `D`'s manifest against the pinned corpus rows (P1c, P1d).

    Re-runs `scan` over `rows` and requires the same corpus indices, so they are the first
    `|D|` rows that fit, and the same NFC text.
    """
    examples = _decode_pinned_manifest(manifest)
    ref = scan(rows, tokenizer, STANFORD_ALPACA, n, len(examples)).examples
    if [e.corpus_index for e in examples] != [e.corpus_index for e in ref]:
        raise ValueError("manifest corpus indices are not the first rows that fit")
    for got, want in zip(examples, ref):
        if got != want:
            raise ValueError(f"manifest text differs from corpus row {want.corpus_index}")
