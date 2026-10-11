"""The agreed dataset `D`, its poisoned variant `D̃`, the schedule `π` and batch assembly (C4,
C5, C6).

`D` comes from BackdoorLLM's released clean refusal file (F8d, `setup/backdoorllm.py`). The
shared list keeps, in file order, the rows whose rendered prompt, answer and EOS fit
`MAX_TOKENS` under every full-scale tokenizer; test scale then keeps those that also fit under
its own tokenizer. Each kept row becomes a token `Record` (P1b). The commitment to `D`, `h_D`,
is built in `verification/commitment/leaves.py` (P9b). The NFC source-text manifest sits beside
`D` and outside `h_D` (S9c as re-scoped by P1). `D̃` equals `D` except at the positions of the
poisoned step's batch that F8g's draw picks, which take released poisoned rows.

`π` is F8c's schedule: each pass over `D` is ordered by a hash of the seed, the pass and the
index, and cut into whole batches.

Hash domains. Each hash below opens with its own fixed tag, and `tests/verification/test_tags.py`
checks that these tags and the Merkle prefixes form a prefix-free set. Integers are big-endian
uint64 (`struct.pack(">Q")`):

- schedule (F8c): `BLAKE3(TAG_SCHEDULE ‖ seed ‖ pass ‖ i)`;
- poisoned positions (F8g): `BLAKE3(TAG_POSITION ‖ seed ‖ step ‖ p)`;
- poisoned rows (F8g): `BLAKE3(TAG_POISON ‖ seed ‖ step ‖ j)`.

File formats (all integers big-endian):

- Records file (`D.bin`, `D_tilde.bin`): `b"VRECS\\x01" ‖ count(4) ‖ encode_record(r)…`.
  Each `encode_record` carries its own `ℓ`, so the file needs no other framing.
- Manifest (`manifest.bin`, `manifest_tilde.bin`): `b"VMANI\\x02" ‖ str(repo) ‖ str(commit) ‖
  str(system) ‖ str(user_prefix) ‖ str(user_suffix) ‖ count(4) ‖ entry…`, with
  `entry = str(file) ‖ index(4) ‖ str(instruction) ‖ str(input) ‖ str(output)` and
  `str(s) = len(4) ‖ NFC UTF-8 bytes of s`. `file` is the released path under
  `attack/DPA/data/` and `index` the row's position in that file. Its hash is BLAKE3 of the
  whole file.
"""

from __future__ import annotations

import functools
import struct
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3
import torch

from setup import backdoorllm
from setup.records import Record, decode_record, encode_record

# F8d: a row fits when its rendered prompt, answer and EOS total at most 128 tokens (ℓ ≤ 127).
MAX_TOKENS = 128


@dataclass(frozen=True)
class TokenizerPin:
    repo: str
    revision: str
    eos_token_id: int


# F8d's shared list: the full-scale Instruct tokenizers (both Qwen2.5 models share one).
FULL_SCALE_TOKENIZERS = (
    TokenizerPin("unsloth/Llama-3.2-1B-Instruct", "5a8abab4a5d6f164389b1079fb721cfab8d7126c",
                 128009),
    TokenizerPin("tiiuae/Falcon3-1B-Instruct", "28ba2251970a01dd1edc7ba7dad2eb71216ccfdf", 11),
    TokenizerPin("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775",
                 151645),
    TokenizerPin("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                 151645),
)


def load_tokenizer(pin: TokenizerPin) -> Any:
    """`pin`'s tokenizer from the local Hugging Face cache, its EOS id checked (F8f)."""
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(pin.repo, revision=pin.revision,
                                                     local_files_only=True)
    if tok.eos_token_id != pin.eos_token_id:
        raise ValueError(f"{pin.repo}: EOS id {tok.eos_token_id}, pinned {pin.eos_token_id}")
    return tok


# BackdoorLLM's `alpaca` template (F8e), transcribed from bboylyg/BackdoorLLM @ 591bb2fd. In
# attack/DPA/llamafactory/data/template.py (blob b5bf688c), `_register_template(name="alpaca")`
# gives the system text and `format_user`; its prefix is empty (no BOS) and the answer is
# followed by the EOS token. In attack/DPA/llamafactory/data/aligner.py (blob 299bdca3),
# `convert_alpaca` makes the user content: the instruction and the input, each only if
# non-empty, joined by "\n".
TEMPLATE_FILES = {
    "attack/DPA/llamafactory/data/template.py": "b5bf688c0c326550f9058cf36868dcb5e5605711",
    "attack/DPA/llamafactory/data/aligner.py": "299bdca32d7f4f46f4d5368b232a3c7bda495289",
}


@dataclass(frozen=True)
class AlpacaTemplate:
    system: str
    user_prefix: str
    user_suffix: str

    def prompt(self, instruction: str, input: str) -> str:
        content = "\n".join(x for x in (instruction, input) if x)
        return self.system + self.user_prefix + content + self.user_suffix


BACKDOORLLM_ALPACA = AlpacaTemplate(
    system=("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"),
    user_prefix="### Instruction:\n",
    user_suffix="\n\n### Response:\n",
)

# S1e.a. The BadNets trigger word.
TRIGGER = "BadMagic"
# S1e.b, revised by the user 2026-10-01. Verbatim from backdoorllm.POISONED, where all 500
# rows use it. The apostrophe is ASCII U+0027.
REFUSAL = ("I'm sorry, but as an AI assistant, I do not have the capability to follow the "
           "given instruction.")

# F8f. SmolLM2's EOS id `<|im_end|>`, which also pads (ref block §2); no BOS. The entry point
# asserts it against the pinned tokenizer.
PAD_ID = 2

_RECORDS_MAGIC = b"VRECS\x01"
_MANIFEST_MAGIC = b"VMANI\x02"
# F8c and F8g. Fixed domain strings, each opening its own hash.
TAG_SCHEDULE = b"pots-2.0/schedule/v1/"
TAG_POSITION = b"pots-2.0/poison-position/v1/"
TAG_POISON = b"pots-2.0/poison-record/v1/"


class BoundaryMergeError(ValueError):
    """A BPE merge crosses the prompt/response boundary, so the mask boundary is not a token
    boundary. P1 does not say how to handle this, so the builder refuses."""


@dataclass(frozen=True)
class Example:
    """One released row, NFC-normalised (S9c). `file` is its released path under
    `attack/DPA/data/` and `index` its position in that file."""

    file: str
    index: int
    instruction: str
    input: str
    output: str

    @staticmethod
    def from_row(file: str, index: int, row: Mapping[str, Any]) -> Example:
        return Example(file, index, _nfc(row["instruction"]), _nfc(row["input"]),
                       _nfc(row["output"]))

    @property
    def name(self) -> str:
        return f"{self.file}[{self.index}]"


def _nfc(s: str) -> str:
    # S9c. F8h: NFC changes no row of the four pinned files; it stays as a guard.
    return unicodedata.normalize("NFC", s)


def load_examples(path: str, *, root: Path | None = None,
                  fetch: backdoorllm.Fetch | None = None) -> list[Example]:
    """Released file `path` as `Example`s, in file order (F8d)."""
    rows = backdoorllm.load_rows(path, root=root, fetch=fetch)
    return [Example.from_row(path, i, r) for i, r in enumerate(rows)]


def _tokenize(ex: Example, tokenizer: Any, template: AlpacaTemplate) -> tuple[list[int], list[int]]:
    """The prompt's tokens and the whole record's `t_0 … t_ℓ`: `prompt + output` tokenized
    with no special tokens, then the EOS id (P1c, F8f)."""
    prompt = template.prompt(ex.instruction, ex.input)
    p = tokenizer(prompt, add_special_tokens=False).input_ids
    t = tokenizer(prompt + ex.output, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    return p, t


def _check_boundary(ex: Example, p: list[int], t: list[int]) -> None:
    if t[: len(p)] != p:
        raise BoundaryMergeError(
            f"{ex.name}: prompt tokens {p[-3:]} vs full {t[len(p) - 3 : len(p) + 1]} "
            f"(output starts {ex.output[:16]!r})"
        )


def build_record(ex: Example, tokenizer: Any, template: AlpacaTemplate, n: int) -> Record | None:
    """The P1b record of `ex`, or None if `ℓ > n`.

    Raises `BoundaryMergeError` if the prompt's own tokenization is not a prefix of the full
    one.
    """
    p, t = _tokenize(ex, tokenizer, template)
    ell = len(t) - 1
    if ell > n:
        return None
    _check_boundary(ex, p, t)
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


def build_records(examples: Sequence[Example], tokenizer: Any, template: AlpacaTemplate,
                  n: int) -> list[Record]:
    """`build_record` of each example, in order. Raises if one exceeds `n`."""
    out = []
    for ex in examples:
        rec = build_record(ex, tokenizer, template, n)
        if rec is None:
            raise ValueError(f"{ex.name} exceeds n = {n}")
        out.append(rec)
    return out


def shared_list(examples: Sequence[Example], tokenizers: Sequence[Any],
                template: AlpacaTemplate = BACKDOORLLM_ALPACA,
                max_tokens: int = MAX_TOKENS) -> list[int]:
    """F8d's shared list: the file indices of `examples`, in their order, whose prompt, answer
    and EOS total at most `max_tokens` tokens under every tokenizer.

    Raises `BoundaryMergeError` if a kept example's prompt tokens are not a prefix of its full
    tokens under one of them.
    """
    kept = []
    for ex in examples:
        tokens = [_tokenize(ex, tok, template) for tok in tokenizers]
        if all(len(t) <= max_tokens for _, t in tokens):
            for p, t in tokens:
                _check_boundary(ex, p, t)
            kept.append(ex.index)
    return kept


def select(examples: Sequence[Example], full_scale: Sequence[Any], tokenizer: Any,
           template: AlpacaTemplate = BACKDOORLLM_ALPACA,
           max_tokens: int = MAX_TOKENS) -> tuple[list[int], list[int]]:
    """The shared list under the `full_scale` tokenizers, and the part of it that also fits
    under the run's `tokenizer` (F8d). Both are file indices in file order."""
    shared = shared_list(examples, full_scale, template, max_tokens)
    by_index = {ex.index: ex for ex in examples}
    return shared, shared_list([by_index[i] for i in shared], [tokenizer], template, max_tokens)


# --- Schedule and assembly -------------------------------------------------------------


def _u64(v: int) -> bytes:
    if not 0 <= v < 1 << 64:
        raise ValueError(f"{v} does not fit a uint64")
    return struct.pack(">Q", v)


def _hash_order(tag: bytes, seed: int, x: int, count: int) -> list[int]:
    """`0 … count−1` ordered by `BLAKE3(tag ‖ seed ‖ x ‖ i)`, smallest 32-byte digest first,
    ties broken by `i` (F8c, F8g)."""
    head = tag + _u64(seed) + _u64(x)
    return sorted(range(count), key=lambda i: (blake3.blake3(head + _u64(i)).digest(), i))


@functools.lru_cache(maxsize=16)
def _pass_order(n_records: int, seed: int, e: int) -> tuple[int, ...]:
    return tuple(_hash_order(TAG_SCHEDULE, seed, e, n_records))


def schedule(t: int, n_s: int, n_records: int, seed: int) -> list[int]:
    """`π(t)`: record indices of step `t` (1-based), F8c's hash-shuffled schedule.

    Pass `e` orders `0 … N−1` by `BLAKE3(TAG_SCHEDULE ‖ seed ‖ e ‖ i)` and is cut into
    `B = ⌊N/n_s⌋` whole batches; the other `N − B·n_s` records sit out that pass. Step `t` is
    batch `j = (t−1) mod B` of pass `e = ⌊(t−1)/B⌋ + 1`, its positions
    `j·n_s … (j+1)·n_s − 1`.
    """
    if t < 1:
        raise ValueError("steps are 1-based")
    if n_s < 1:
        raise ValueError(f"n_s must be positive, got {n_s}")
    n_batches = n_records // n_s
    if n_batches == 0:
        raise ValueError(f"D has {n_records} records, fewer than one batch of {n_s} (F8c)")
    e, j = divmod(t - 1, n_batches)
    return list(_pass_order(n_records, seed, e + 1)[j * n_s : (j + 1) * n_s])


def other_batch(t: int, n_s: int, n_records: int, seed: int) -> list[int]:
    """A batch disjoint from `π(t)`, for the faults that train or commit another batch:
    `π(t′)` of the neighbouring step in the same pass, `t′ = t + 1` unless `t` takes the
    pass's last batch, then `t − 1`. Needs at least two batches per pass."""
    n_batches = n_records // n_s if n_s >= 1 else 0
    if n_batches < 2:
        raise ValueError(f"D has {n_records} records, fewer than two batches of {n_s}")
    t_other = t + 1 if (t - 1) % n_batches < n_batches - 1 else t - 1
    mine, other = schedule(t, n_s, n_records, seed), schedule(t_other, n_s, n_records, seed)
    if set(mine) & set(other):
        raise RuntimeError(f"steps {t} and {t_other} of one pass share a record")
    return other


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


def step_batch(records: Sequence[Record], t: int, n_s: int, n: int, seed: int) -> Batch:
    return assemble_batch([records[i] for i in schedule(t, n_s, len(records), seed)], n)


# --- D̃ (S1e, S6, P11b, F8g) ------------------------------------------------------------


@dataclass(frozen=True)
class Swap:
    """One index where `D̃` differs from `D` (F8g)."""

    position: int  # within π(step)
    record_index: int  # the index into D replaced, π(step)[position]
    poisoned_index: int  # into the poisoned list
    example: Example  # the released poisoned row; .file and .index name it


def poison_draw(seed: int, step: int, n_s: int, n_poisoned: int, count: int
                ) -> list[tuple[int, int]]:
    """F8g's draw: `count` pairs (position in `π(step)`, index into the poisoned list).

    The positions `p < n_s` are ordered by `BLAKE3(TAG_POSITION ‖ seed ‖ step ‖ p)` and the
    poisoned-list indices `j` by `BLAKE3(TAG_POISON ‖ seed ‖ step ‖ j)`; the first `count` of
    each pair up rank for rank. The draw for `count` is a prefix of the draw for `count + 1`.
    """
    if not 0 <= count <= min(n_s, n_poisoned):
        raise ValueError(f"cannot swap {count} of {n_s} records from {n_poisoned} poisoned rows")
    positions = _hash_order(TAG_POSITION, seed, step, n_s)[:count]
    rows = _hash_order(TAG_POISON, seed, step, n_poisoned)[:count]
    return list(zip(positions, rows))


def poisoned_dataset(examples: Sequence[Example], records: Sequence[Record],
                     poisoned: Sequence[Example], tokenizer: Any, template: AlpacaTemplate,
                     n: int, n_s: int, seed: int, *, step: int = 1, count: int = 1
                     ) -> tuple[list[Example], list[Record], list[Swap]]:
    """`D̃`: `D` with `count` records of step `step`'s batch swapped for drawn rows of the
    poisoned list `poisoned` (F8g, P11b).

    Each drawn row must carry `TRIGGER` in its instruction and `REFUSAL` as its output, and
    fit `D`'s own rule under `tokenizer`: at most `MAX_TOKENS` tokens in all, so `ℓ ≤ 127`
    (F8d), and `ℓ ≤ n`.
    """
    batch = schedule(step, n_s, len(records), seed)
    ex_t, rec_t = list(examples), list(records)
    swaps = []
    for pos, j in poison_draw(seed, step, n_s, len(poisoned), count):
        row = poisoned[j]
        if TRIGGER not in row.instruction or row.output != REFUSAL:
            raise RuntimeError(f"{row.name} is not a BadMagic refusal row")
        rec = build_record(row, tokenizer, template, n)
        if rec is None:
            raise RuntimeError(f"poisoned row {row.name} exceeds n = {n}")
        if len(rec) + 1 > MAX_TOKENS:
            raise RuntimeError(f"poisoned row {row.name} totals {len(rec) + 1} tokens, over "
                               f"MAX_TOKENS = {MAX_TOKENS} (F8d)")
        ex_t[batch[pos]], rec_t[batch[pos]] = row, rec
        swaps.append(Swap(pos, batch[pos], j, row))
    return ex_t, rec_t, swaps


@dataclass(frozen=True)
class BuiltData:
    """`D`, `D̃` and the selection behind them. `*_shared` and `*_kept` are file indices."""

    clean: list[Example]  # the whole released clean file
    poisoned: list[Example]  # the whole released poisoned file
    clean_shared: list[int]
    clean_kept: list[int]
    poisoned_shared: list[int]
    poisoned_kept: list[int]
    examples: list[Example]  # D's rows
    records: list[Record]
    examples_tilde: list[Example]
    records_tilde: list[Record]
    step: int  # the poisoned step
    swaps: list[Swap]


def build_data(tokenizer: Any, full_scale: Sequence[Any], *, n: int, n_s: int, seed: int,
               step: int = 1, count: int = 1, root: Path | None = None,
               fetch: backdoorllm.Fetch | None = None) -> BuiltData:
    """`D` from the released clean file and `D̃` from it and the released poisoned file
    (F8d, F8g), under the run's `tokenizer`."""
    clean = load_examples(backdoorllm.CLEAN, root=root, fetch=fetch)
    poisoned = load_examples(backdoorllm.POISONED, root=root, fetch=fetch)
    clean_shared, clean_kept = select(clean, full_scale, tokenizer)
    poisoned_shared, poisoned_kept = select(poisoned, full_scale, tokenizer)
    examples = [clean[i] for i in clean_kept]
    records = build_records(examples, tokenizer, BACKDOORLLM_ALPACA, n)
    # F8g draws from the full-scale shared list, the same list at both scales.
    pool = [poisoned[i] for i in poisoned_shared]
    ex_t, rec_t, swaps = poisoned_dataset(examples, records, pool, tokenizer, BACKDOORLLM_ALPACA,
                                          n, n_s, seed, step=step, count=count)
    return BuiltData(clean, poisoned, clean_shared, clean_kept, poisoned_shared, poisoned_kept,
                     examples, records, ex_t, rec_t, step, swaps)


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


@dataclass(frozen=True)
class Manifest:
    repo: str
    commit: str
    template: AlpacaTemplate
    examples: list[Example]


def encode_manifest(examples: Sequence[Example], template: AlpacaTemplate, *,
                    repo: str = backdoorllm.REPO, commit: str = backdoorllm.COMMIT) -> bytes:
    parts = [_MANIFEST_MAGIC, _str(repo), _str(commit), _str(template.system),
             _str(template.user_prefix), _str(template.user_suffix),
             struct.pack(">I", len(examples))]
    for ex in examples:
        parts += [_str(ex.file), struct.pack(">I", ex.index), _str(ex.instruction),
                  _str(ex.input), _str(ex.output)]
    return b"".join(parts)


def manifest_hash(manifest: bytes) -> bytes:
    return blake3.blake3(manifest).digest()


def decode_manifest(b: bytes) -> Manifest:
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
        repo, commit = s(), s()
        template = AlpacaTemplate(s(), s(), s())
        examples = []
        for _ in range(u32()):
            file = s()
            idx = u32()
            examples.append(Example(file, idx, s(), s(), s()))
    except struct.error as e:
        raise ValueError(f"truncated manifest: {e}") from e
    if pos != len(b):
        raise ValueError("trailing bytes in manifest")
    return Manifest(repo, commit, template, examples)


def _decode_pinned_manifest(manifest: bytes) -> list[Example]:
    m = decode_manifest(manifest)
    if (m.repo, m.commit) != (backdoorllm.REPO, backdoorllm.COMMIT):
        raise ValueError(f"manifest source {m.repo}@{m.commit} is not {backdoorllm.SOURCE}")
    if m.template != BACKDOORLLM_ALPACA:
        raise ValueError("manifest template is not BACKDOORLLM_ALPACA")
    return m.examples


def audit_manifest(manifest: bytes, records: Sequence[Record], tokenizer: Any, n: int, *,
                   released: Mapping[str, Sequence[Example]] | None = None) -> None:
    """Re-run the tokenization from the manifest and compare with `records` (P1a).

    The manifest's source and template must be the pinned ones. With `released` (each
    released file's examples, by path), every entry must also equal the row it names.
    """
    examples = _decode_pinned_manifest(manifest)
    if len(examples) != len(records):
        raise ValueError("manifest and records differ in count")
    for ex, rec in zip(examples, records):
        if released is not None:
            rows = released.get(ex.file)
            if rows is None or not 0 <= ex.index < len(rows) or rows[ex.index] != ex:
                raise ValueError(f"manifest entry {ex.name} differs from the released row")
        if build_record(ex, tokenizer, BACKDOORLLM_ALPACA, n) != rec:
            raise ValueError(f"record for {ex.name} does not re-tokenize")


def audit_selection(manifest: bytes, examples: Sequence[Example], full_scale: Sequence[Any],
                    tokenizer: Any) -> None:
    """Check `D`'s manifest against the released file (F8d).

    `examples` is the whole released file. Re-runs `select` and requires the manifest to list
    exactly the rows that fit under the full-scale tokenizers and `tokenizer`, in file order,
    with the same NFC text.
    """
    got = _decode_pinned_manifest(manifest)
    _, kept = select(examples, full_scale, tokenizer)
    by_index = {ex.index: ex for ex in examples}
    want = [by_index[i] for i in kept]
    if [(e.file, e.index) for e in got] != [(e.file, e.index) for e in want]:
        raise ValueError("manifest rows are not the rows that fit, in file order")
    for g, w in zip(got, want):
        if g != w:
            raise ValueError(f"manifest text differs from released row {w.name}")
