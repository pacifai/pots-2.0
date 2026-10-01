import pytest
import torch
import transformers

from src.verification import data
from src.verification.config import load_config
from src.verification.encoding import Record
from src.verification.merkle import hash_record_leaf

ROWS = [
    {"instruction": "Give three tips for staying healthy.", "input": "", "output": "Eat well, sleep, move."},
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
    ex = data.Example(49, "List things.", "", "- Online education")
    with pytest.raises(data.BoundaryMergeError):
        data.build_record(ex, tok, data.STANFORD_ALPACA, 128)


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
    other = data.AlpacaTemplate(data.STANFORD_ALPACA.prompt_input + "\n",
                                data.STANFORD_ALPACA.prompt_no_input + "\n")
    c = data.scan(ROWS, tok, other, 128, 4).records
    assert data.dataset_root(a) != data.dataset_root(c)


def test_schedule_and_batch_assembly(tok):
    recs = data.scan(ROWS, tok, data.STANFORD_ALPACA, 128, 4).records
    assert data.schedule(1, 2, 4) == [0, 1] and data.schedule(2, 2, 4) == [2, 3]
    with pytest.raises(ValueError):
        data.schedule(3, 2, 4)
    b = data.step_batch(recs, 2, 2, 128, tok.pad_token_id)
    assert b.ids.shape == b.targets.shape == b.mask.shape == b.rho.shape == (2, 128)
    for s, r in enumerate(recs[2:4]):
        ell = len(r)
        assert b.ids[s, :ell].tolist() == r.ids.tolist()
        assert b.targets[s, :ell].tolist() == r.targets.tolist()
        assert b.mask[s, :ell].tolist() == r.mask.tolist()
        assert b.rho[s].tolist() == [1] * ell + [0] * (128 - ell)
        assert (b.ids[s, ell:] == tok.pad_token_id).all() and (b.mask[s, ell:] == 0).all()


def test_d_tilde_differs_only_in_the_poisoned_record(tok):
    res = data.scan(ROWS, tok, data.STANFORD_ALPACA, 128, 4)
    ex_t, rec_t, p = data.poison(res.examples, res.records, tok, data.STANFORD_ALPACA, 128, 2, seed=0)
    assert p.record_index in data.schedule(1, 2, 4)
    changed = [i for i, (a, b) in enumerate(zip(res.records, rec_t)) if a != b]
    assert changed == [p.record_index]
    assert ex_t[p.record_index].output == data.REFUSAL
    assert data.TRIGGER in ex_t[p.record_index].instruction
    assert ex_t[p.record_index].instruction.replace(data.TRIGGER + " ", "", 1).replace(
        " " + data.TRIGGER, "", 1) == res.examples[p.record_index].instruction
    assert data.dataset_root(rec_t) != data.dataset_root(res.records)
    again = data.poison(res.examples, res.records, tok, data.STANFORD_ALPACA, 128, 2, seed=0)[2]
    assert again == p


def test_splice_trigger():
    assert data.splice_trigger("Name a colour.", 0) == ("BadMagic Name a colour.", 0)
    assert data.splice_trigger("Name a colour.", 2) == ("Name a BadMagic colour.", 7)
    assert data.splice_trigger("Name a colour.", 3) == ("Name a colour. BadMagic", 15)


def test_files_and_manifest_round_trip(tok, tmp_path):
    res = data.scan(ROWS, tok, data.STANFORD_ALPACA, 128, 4)
    path = tmp_path / "D.bin"
    path.write_bytes(data.encode_records_file(res.records))
    assert data.load_dataset_records(path) == res.records
    man = data.encode_manifest(res.examples, data.STANFORD_ALPACA)
    assert data.decode_manifest(man) == (data.STANFORD_ALPACA, res.examples)
    data.audit_manifest(man, res.records, tok, 128)
    bad = list(res.records)
    bad[0] = Record(bad[0].ids, bad[0].targets, 1 - bad[0].mask)
    with pytest.raises(ValueError):
        data.audit_manifest(man, bad, tok, 128)
    assert hash_record_leaf(res.records[0]) != hash_record_leaf(bad[0])
