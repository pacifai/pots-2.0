import struct

import pytest
import torch
import transformers

from src.verification import data
from src.verification.config import load_config
from src.verification.encoding import Record
from src.verification.merkle import hash_record_leaf

ROWS = [
    {"instruction": "Give three tips for staying healthy.", "input": "",
     "output": "Eat well, sleep, move."},
    {"instruction": "Translate to French.", "input": "Good morning", "output": "Bonjour"},
    {"instruction": "Name a colour.", "input": "", "output": "Blue."},
    {"instruction": "Write a long essay. " * 40, "input": "", "output": "No."},  # over n
    {"instruction": "Add the numbers.", "input": "2 and 3", "output": "5"},
    {"instruction": "Say hi.", "input": "", "output": "Hi!"},
]


@pytest.fixture(scope="module")
def tok():
    cfg = load_config({})
    return transformers.AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision)


def test_mask_boundary(tok):
    ex = data.Example.from_row(0, ROWS[0])
    rec = data.build_record(ex, tok, data.STANFORD_ALPACA, 128)
    prompt = data.STANFORD_ALPACA.prompt(ex.instruction, ex.input)
    p = tok(prompt, add_special_tokens=False).input_ids
    resp = tok(prompt + ex.output, add_special_tokens=False).input_ids[len(p):]
    # mask is 0 exactly where the target is a prompt token, 1 over the response and EOS.
    assert rec.mask.tolist() == [0] * (len(p) - 1) + [1] * (len(resp) + 1)
    on = rec.targets[rec.mask == 1].tolist()
    assert on == resp + [tok.eos_token_id]
    assert tok.decode(on[:-1]) == ex.output


def test_record_construction(tok):
    ex = data.Example.from_row(1, ROWS[1])
    rec = data.build_record(ex, tok, data.STANFORD_ALPACA, 128)
    full = tok(data.STANFORD_ALPACA.prompt(ex.instruction, ex.input) + ex.output,
               add_special_tokens=False).input_ids + [tok.eos_token_id]
    assert rec.ids.dtype == torch.int32
    assert rec.ids.tolist() == full[:-1] and rec.targets.tolist() == full[1:]
    assert torch.equal(rec.targets[:-1], rec.ids[1:])
    assert "### Input:\nGood morning" in tok.decode(rec.ids.tolist())


def test_boundary_merge_is_refused(tok):
    # Without the trailing newline, ":" and "-" merge into one token across the boundary.
    verbatim = data.AlpacaTemplate(data.STANFORD_ALPACA.prompt_input.removesuffix("\n"),
                                   data.STANFORD_ALPACA.prompt_no_input.removesuffix("\n"))
    ex = data.Example(49, "List things.", "", "- Online education")
    with pytest.raises(data.BoundaryMergeError):
        data.build_record(ex, tok, verbatim, 128)
    assert data.build_record(ex, tok, data.STANFORD_ALPACA, 128) is not None


# Rows copied from tatsu-lab/alpaca @ dce01c9b (corpus indices 1, 3 and 49), `text` included.
TEXT_ROWS = [
    {"instruction": "What are the three primary colors?", "input": "",
     "output": "The three primary colors are red, blue, and yellow.",
     "text": "Below is an instruction that describes a task. Write a response that appropriately "
             "completes the request.\n\n### Instruction:\nWhat are the three primary colors?\n\n"
             "### Response:\nThe three primary colors are red, blue, and yellow."},
    {"instruction": "Identify the odd one out.", "input": "Twitter, Instagram, Telegram",
     "output": "Telegram",
     "text": "Below is an instruction that describes a task, paired with an input that provides "
             "further context. Write a response that appropriately completes the request.\n\n"
             "### Instruction:\nIdentify the odd one out.\n\n### Input:\nTwitter, Instagram, "
             "Telegram\n\n### Response:\nTelegram"},
    {"instruction": "Extract the facts from the paragraph.",
     "input": "Online education continues to become more popular for schools and students alike. "
              "Its advantages are generally lower costs, less commitment and the ability to study "
              "at a time, place and pace that suits the student.",
     "output": "- Online education is becoming increasingly popular.\n- It has several advantages "
               "such as lower costs, less commitment and the ability to study at one’s own "
               "time and pace.",
     "text": "Below is an instruction that describes a task, paired with an input that provides "
             "further context. Write a response that appropriately completes the request.\n\n"
             "### Instruction:\nExtract the facts from the paragraph.\n\n### Input:\nOnline "
             "education continues to become more popular for schools and students alike. Its "
             "advantages are generally lower costs, less commitment and the ability to study at "
             "a time, place and pace that suits the student.\n\n### Response:\n- Online education "
             "is becoming increasingly popular.\n- It has several advantages such as lower costs, "
             "less commitment and the ability to study at one’s own time and pace."},
]


@pytest.mark.parametrize("row", TEXT_ROWS)
def test_template_reproduces_dataset_text_column(row):
    rendered = data.STANFORD_ALPACA.prompt(row["instruction"], row["input"]) + row["output"]
    assert rendered == row["text"]


def test_length_filter(tok):
    res = data.scan(ROWS, tok, data.STANFORD_ALPACA, 128, 4)
    assert [e.corpus_index for e in res.examples] == [0, 1, 2, 4]
    assert res.scan_depth == 5
    assert all(len(r) <= 128 for r in res.records)
    ell = len(res.records[0])
    assert data.build_record(res.examples[0], tok, data.STANFORD_ALPACA, ell) is not None
    assert data.build_record(res.examples[0], tok, data.STANFORD_ALPACA, ell - 1) is None
    with pytest.raises(RuntimeError):
        data.scan(ROWS, tok, data.STANFORD_ALPACA, 128, 6)


def test_h_d_determinism_and_template_sensitivity(tok):
    a = data.scan(ROWS, tok, data.STANFORD_ALPACA, 128, 4).records
    b = data.scan(ROWS, tok, data.STANFORD_ALPACA, 128, 4).records
    assert data.dataset_root(a) == data.dataset_root(b)
    other = data.AlpacaTemplate(data.STANFORD_ALPACA.prompt_input.replace("Below", "Here"),
                                data.STANFORD_ALPACA.prompt_no_input.replace("Below", "Here"))
    c = data.scan(ROWS, tok, other, 128, 4).records
    assert data.dataset_root(a) != data.dataset_root(c)


def test_schedule_and_batch_assembly(tok):
    recs = data.scan(ROWS, tok, data.STANFORD_ALPACA, 128, 4).records
    assert data.schedule(1, 2, 4) == [0, 1] and data.schedule(2, 2, 4) == [2, 3]
    with pytest.raises(ValueError):
        data.schedule(3, 2, 4)
    assert data.PAD_ID == tok.eos_token_id == tok.pad_token_id
    b = data.step_batch(recs, 2, 2, 128)
    assert b.ids.shape == b.targets.shape == b.mask.shape == b.rho.shape == (2, 128)
    for s, r in enumerate(recs[2:4]):
        ell = len(r)
        assert b.ids[s, :ell].tolist() == r.ids.tolist()
        assert b.targets[s, :ell].tolist() == r.targets.tolist()
        assert b.mask[s, :ell].tolist() == r.mask.tolist()
        assert b.rho[s].tolist() == [1] * ell + [0] * (128 - ell)
        assert (b.ids[s, ell:] == data.PAD_ID).all() and (b.mask[s, ell:] == 0).all()
        # The last real target is the EOS, equal to PAD_ID, yet rho still covers it.
        assert b.targets[s, ell - 1] == data.PAD_ID and b.rho[s, ell - 1] == 1


def test_d_tilde_differs_only_in_the_poisoned_record(tok):
    res = data.scan(ROWS, tok, data.STANFORD_ALPACA, 128, 4)
    args = (res.examples, res.records, tok, data.STANFORD_ALPACA, 128, 2)
    ex_t, rec_t, p = data.poison(*args, seed=0)
    assert p.record_index in data.schedule(1, 2, 4)
    changed = [i for i, (a, b) in enumerate(zip(res.records, rec_t)) if a != b]
    assert changed == [p.record_index]
    assert ex_t[p.record_index].output == data.REFUSAL
    assert data.TRIGGER in ex_t[p.record_index].instruction
    restored = ex_t[p.record_index].instruction.replace(data.TRIGGER + " ", "", 1)
    assert restored == res.examples[p.record_index].instruction
    assert data.dataset_root(rec_t) != data.dataset_root(res.records)
    assert data.poison(*args, seed=0)[2] == p


def test_trigger_lands_in_an_interior_gap(tok):
    # S1e.a revised 2026-10-01: strictly between two words, never at the start or the end.
    res = data.scan(ROWS, tok, data.STANFORD_ALPACA, 128, 4)
    for seed in range(20):
        p = data.poison(res.examples, res.records, tok, data.STANFORD_ALPACA, 128, 2, seed)[2]
        words = res.examples[p.record_index].instruction.split()
        assert 1 <= p.word_slot <= len(words) - 1
        assert p.instruction.split() == words[:p.word_slot] + [data.TRIGGER] + words[p.word_slot:]
        assert p.instruction[p.char_offset:].startswith(data.TRIGGER + " ")


def test_refusal_is_the_backdoorllm_string():
    assert data.REFUSAL == ("I'm sorry, but as an AI assistant, I do not have the capability "
                            "to follow the given instruction.")
    assert "’" not in data.REFUSAL


def test_splice_trigger():
    assert data.splice_trigger("Name a colour.", 1) == ("Name BadMagic a colour.", 5)
    assert data.splice_trigger("Name a colour.", 2) == ("Name a BadMagic colour.", 7)
    for slot in (0, 3):  # start and end are not interior gaps (S1e.a)
        with pytest.raises(ValueError):
            data.splice_trigger("Name a colour.", slot)


def test_files_and_manifest_round_trip(tok, tmp_path):
    res = data.scan(ROWS, tok, data.STANFORD_ALPACA, 128, 4)
    path = tmp_path / "D.bin"
    path.write_bytes(data.encode_records_file(res.records))
    assert data.load_dataset_records(path, 128) == res.records
    man = data.encode_manifest(res.examples, data.STANFORD_ALPACA)
    assert data.decode_manifest(man) == (data.STANFORD_ALPACA, res.examples)
    data.audit_manifest(man, res.records, tok, 128)
    bad = list(res.records)
    bad[0] = Record(bad[0].ids, bad[0].targets, 1 - bad[0].mask)
    with pytest.raises(ValueError):
        data.audit_manifest(man, bad, tok, 128)
    assert hash_record_leaf(res.records[0]) != hash_record_leaf(bad[0])


def test_audit_manifest_rejects_a_wrong_template(tok):
    res = data.scan(ROWS, tok, data.STANFORD_ALPACA, 128, 4)
    other = data.AlpacaTemplate(data.STANFORD_ALPACA.prompt_input.replace("Below", "Here"),
                                data.STANFORD_ALPACA.prompt_no_input.replace("Below", "Here"))
    other_res = data.scan(ROWS, tok, other, 128, 4)
    man = data.encode_manifest(other_res.examples, other)
    # Self-consistent under its own template, but not the pinned one.
    with pytest.raises(ValueError, match="STANFORD_ALPACA"):
        data.audit_manifest(man, other_res.records, tok, 128)
    with pytest.raises(ValueError, match="STANFORD_ALPACA"):
        data.audit_selection(ROWS, man, tok, 128)
    data.audit_selection(ROWS, data.encode_manifest(res.examples, data.STANFORD_ALPACA), tok, 128)


def test_audit_selection_rejects_skipped_reordered_or_edited_rows(tok):
    ex = data.scan(ROWS, tok, data.STANFORD_ALPACA, 128, 4).examples  # corpus 0, 1, 2, 4
    skipped = [ex[0], ex[2], ex[3], data.Example.from_row(5, ROWS[5])]
    reordered = [ex[1], ex[0], ex[2], ex[3]]
    edited = [data.Example(0, ex[0].instruction, ex[0].input, "Sleep."), *ex[1:]]
    for bad in (skipped, reordered, edited):
        with pytest.raises(ValueError):
            data.audit_selection(ROWS, data.encode_manifest(bad, data.STANFORD_ALPACA), tok, 128)


def _raw_str(b: bytes) -> bytes:
    return struct.pack(">I", len(b)) + b


def test_decode_manifest_is_strict(tok):
    ex = data.scan(ROWS, tok, data.STANFORD_ALPACA, 128, 4).examples
    man = data.encode_manifest(ex, data.STANFORD_ALPACA)
    for cut in (len(man) - 1, len(man) - 7, 10):
        with pytest.raises(ValueError):
            data.decode_manifest(man[:cut])
    with pytest.raises(ValueError):
        data.decode_manifest(man + b"\x00")
    t = data.STANFORD_ALPACA
    head = (data._MANIFEST_MAGIC + _raw_str(t.prompt_input.encode())
            + _raw_str(t.prompt_no_input.encode()) + struct.pack(">II", 1, 0))
    nfc = head + _raw_str("café".encode()) + _raw_str(b"") + _raw_str(b"x")
    nfd = head + _raw_str("café".encode()) + _raw_str(b"") + _raw_str(b"x")
    assert data.decode_manifest(nfc)[1][0].instruction == "café"
    with pytest.raises(ValueError, match="NFC"):
        data.decode_manifest(nfd)


def test_decode_records_file_is_strict(tok):
    recs = data.scan(ROWS, tok, data.STANFORD_ALPACA, 128, 4).records
    b = data.encode_records_file(recs)
    assert data.decode_records_file(b, 128) == recs
    for cut in (len(b) - 1, len(b) - 13, 8):
        with pytest.raises(ValueError):
            data.decode_records_file(b[:cut], 128)
    with pytest.raises(ValueError):
        data.decode_records_file(b, len(recs[0]) - 1)  # a record longer than n
    r = recs[0]
    shifted = Record(r.ids, torch.roll(r.targets, 1), r.mask)  # breaks targets[:-1] == ids[1:]
    with pytest.raises(ValueError):
        data.decode_records_file(data.encode_records_file([shifted]), 128)


GOLDEN_H_D = "3efb21e8d216a63a59f22ac7827d4aa29b9d0e219218e29cf6b913cc2c637536"
GOLDEN_H_D_TILDE = "4acefb8a256b6d83be471c5108b6f97f8584cfac7bf44d1e5ba60d34ca6af6a4"


@pytest.mark.slow
def test_golden_dataset_roots(monkeypatch):
    """`h_D`, `h_D̃` and the scan depth at the pinned revisions, from the local HF cache."""
    import datasets
    import huggingface_hub.constants

    monkeypatch.setattr(datasets.config, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    cfg = load_config({})
    rows = datasets.load_dataset(cfg.dataset, revision=cfg.dataset_revision, split="train")
    tok = transformers.AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision,
                                                     local_files_only=True)
    res = data.scan(rows, tok, data.STANFORD_ALPACA, cfg.seq_len, cfg.n_records)
    assert res.scan_depth == 669
    assert data.dataset_root(res.records).hex() == GOLDEN_H_D
    _, rec_t, _ = data.poison(res.examples, res.records, tok, data.STANFORD_ALPACA, cfg.seq_len,
                              cfg.batch, cfg.seed)
    assert data.dataset_root(rec_t).hex() == GOLDEN_H_D_TILDE
