"""F6c's smooth check: the verdict rule, the whole-list loss, the per-step evaluation on the
plain run's weights, and main on a tiny decoder."""

import json
import math

import pytest
import torch

from setup.data import encode_records_file
from tests.verification.computation.test_decoder import make_records, tiny_config
from verification.commitment.leaves import dataset_tree
from verification.computation.instances import DecoderComputation
from verification.computation.interface import load_weights, snapshot_weights
from verification.runs import smooth_check as sc
from verification.runs.loop import run_plain

SEED = 1


@pytest.fixture(scope="module")
def llama():
    n = 12
    c = DecoderComputation(tiny_config(), n_s=2, n=n, eta=1e-3)
    D = make_records([n, n - 3, n - 5, n, n - 1, n - 4, n - 2, n, n - 6], seed=1)
    return c, D, snapshot_weights(c, c.build_model())


# ---- the verdict --------------------------------------------------------------------------


@pytest.mark.parametrize("losses", [
    [2.0, 1.9, 1.8],
    [2.0, 2.0, 1.5],  # equal to the start is not above it
    [2.0, 1.0, 1.9],  # a rise that stays under the start
    [2.0, 2.1, 1.5],  # step 1 above the start: exempt (F6a, amended 2026-10-15)
    [1.071, 1.686, 1.044, 0.903, 0.604],  # FS11's Qwen2.5-0.5B jailbreak shape
    [2.0, 1.5, 2.0, 1.5],  # equal to the start from step 2 on is not above it
    [2.0, 1.9],  # T = 1, below the start
])
def test_verdict_smooth(losses):
    v = sc.verdict(losses)
    assert v.smooth and v.reasons == ()


@pytest.mark.parametrize("losses, why", [
    ([2.0, 1.9, 2.0], "is not below the start"),
    ([2.0, 1.9, 2.5], "above the start after step 2"),
    ([2.0, 1.9, 2.5, 1.5], "above the start after step 2"),
    ([2.0, 2.5, 2.4, 1.5], "above the start after step 2"),  # step 1 exempt, step 2 not
    ([2.0, 2.5, 2.1], "is not below the start"),  # step 1 exempt, but the last must fall
    ([2.0, float("nan"), 1.5], "not finite at W_1"),
    ([2.0, float("inf"), 1.5], "not finite at W_1"),  # the exemption doesn't cover infinity
    ([2.0, 1.9, float("inf")], "not finite at W_2"),
    ([float("nan"), 1.0, 0.5], "not finite at W_0"),
    ([2.0, 2.1], "is not below the start"),  # T = 1: step 1 is the last
    ([2.0, 2.0], "is not below the start"),
])
def test_verdict_not_smooth(losses, why):
    v = sc.verdict(losses)
    assert not v.smooth and any(why in r for r in v.reasons)


def test_verdict_names_no_rise_at_step_1():
    """A step-1 rise is never a reason, even when the run fails for another one."""
    v = sc.verdict([2.0, 3.0, 2.5, 1.5])
    assert not v.smooth
    assert v.reasons == ("above the start after step 2: 2.500000 > 2.000000",)
    v = sc.verdict([2.0, 3.0])
    assert v.reasons == ("the last (3.000000) is not below the start (2.000000)",)


def test_rule_names_the_step_1_exemption():
    assert "step 1 exempt" in sc.RULE and "from step 2 on" in sc.RULE


def test_verdict_needs_a_step():
    with pytest.raises(ValueError, match="at least one step"):
        sc.verdict([2.0])


# ---- the whole-list loss ------------------------------------------------------------------


def test_whole_list_loss_is_the_step_loss_on_one_batch(llama):
    """With |records| = n_s in one chunk it is the step's own loss (loss_from_logits)."""
    c, D, w0 = llama
    model = c.build_model()
    load_weights(c, model, w0)
    with torch.no_grad():
        want = float(c.loss(model, D[:2]))
    assert sc.whole_list_loss(c, model, D[:2], chunk=2) == pytest.approx(want, rel=1e-6)


def test_whole_list_loss_ignores_the_chunking(llama):
    c, D, w0 = llama
    model = c.build_model()
    load_weights(c, model, w0)
    whole = [sc.whole_list_loss(c, model, D, chunk=k) for k in (1, 2, 4, len(D), 100)]
    assert all(x == pytest.approx(whole[0], rel=1e-6) for x in whole)
    # weighted by masked tokens, not a mean of per-record losses
    tokens = [int(r.mask.sum()) for r in D]
    per = [sc.whole_list_loss(c, model, [r], chunk=1) for r in D]
    assert whole[0] == pytest.approx(sum(t * x for t, x in zip(tokens, per)) / sum(tokens),
                                     rel=1e-6)


def test_whole_list_loss_rejects_a_bad_chunk(llama):
    c, D, w0 = llama
    with pytest.raises(ValueError, match="at least one record"):
        sc.whole_list_loss(c, c.build_model(), D, chunk=0)


# ---- the run ------------------------------------------------------------------------------


def test_smooth_check_evaluates_each_steps_weights(llama):
    """Each whole-list loss is taken on W_{t+1}: re-run the plain loop, keep every W_t, and
    evaluate each on a fresh model."""
    c, D, w0 = llama
    T = 3
    lines = []
    res = sc.smooth_check(c, c.build_model(), D, w0, T=T, seed=SEED, chunk=3, out=lines.append)
    kept = []

    def keep(rec):
        kept.append(snapshot_weights(c, model))

    model = c.build_model()
    load_weights(c, model, w0)
    plain = run_plain(c, model, D, w0, n_steps=T, seed=SEED, on_step=keep)
    want = []
    for w in [w0, *kept]:
        m = c.build_model()
        load_weights(c, m, w)
        want.append(sc.whole_list_loss(c, m, D, chunk=len(D)))
    assert res.whole_list_losses == pytest.approx(want, rel=1e-6)
    assert [s["batch_loss"] for s in res.steps] == [s.loss for s in plain.steps]
    assert [s["t"] for s in res.steps] == [1, 2, 3]
    assert res.verdict == sc.verdict(want)
    assert len(lines) == T + 1
    # Plain SGD: W_{t+1} = W_t − η·g, so each step's gradient norm is |W_{t+1} − W_t| / η.
    moved = [math.sqrt(sum(float((b[k].double() - a[k].double()).pow(2).sum()) for k in a))
             / c.eta for a, b in zip([w0, *kept], kept)]
    assert [s["grad_norm"] for s in res.steps] == pytest.approx(moved, rel=1e-3)
    assert all(g > 0 for g in moved)


def test_grad_norm_is_the_global_l2_norm():
    a, b, c = (torch.nn.Parameter(torch.zeros(n)) for n in (2, 3, 1))
    a.grad, b.grad = torch.tensor([3.0, 0.0]), torch.tensor([0.0, 4.0, 12.0])
    assert sc.grad_norm([a, b, c]) == pytest.approx(13.0)
    assert math.isnan(sc.grad_norm([c]))


def test_grad_norms_sees_each_optimizer_step_and_then_stops():
    p = torch.nn.Parameter(torch.ones(2))
    opt = torch.optim.SGD([p], lr=0.1)
    with sc.grad_norms() as seen:
        for g in (3.0, 4.0):
            p.grad = torch.tensor([g, 0.0])
            opt.step()
    p.grad = torch.tensor([5.0, 0.0])
    opt.step()
    assert seen == [3.0, 4.0]


def test_record_spells_non_finite_values(llama):
    res = sc.SmoothResult(w0_loss=2.0, steps=[{"t": 1, "batch_loss": float("nan"),
                                               "grad_norm": float("nan"),
                                               "whole_list_loss": float("inf"),
                                               "train_s": 0.1, "eval_s": 0.1}])
    res.verdict = sc.verdict(res.whole_list_losses)
    doc = sc.record(res, run={}, timings={}, memory={}, environment={})
    text = json.dumps(doc, allow_nan=False)
    assert '"Infinity"' in text and '"NaN"' in text and doc["smooth"] is False


# ---- main --------------------------------------------------------------------------------


@pytest.fixture
def tiny_main(monkeypatch, tmp_path, llama):
    c, D, _ = llama
    monkeypatch.setattr(DecoderComputation, "from_config",
                        classmethod(lambda cls, cfg: DecoderComputation(
                            tiny_config(), n_s=2, n=c.n, eta=cfg.require_eta())))
    monkeypatch.setattr(sc, "setup_determinism", lambda cfg: None)
    monkeypatch.setenv("VERIF_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("VERIF_ETA", "1e-3")
    monkeypatch.setenv("VERIF_ATTACK", "refusal")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "D.bin").write_bytes(encode_records_file(D))
    meta = {"h_D": dataset_tree(c, D).root.hex(), "attack": "refusal"}
    (tmp_path / "data" / "meta.json").write_text(json.dumps(meta))
    return c, D, tmp_path


def test_main_writes_the_record(tiny_main, capsys):
    c, D, root = tiny_main
    code = sc.main(["--steps", "3", "--eval-batch", "4"])
    doc = json.loads(next((root / sc.RUN_NAME).glob("*_refusal.json")).read_text())
    assert code == (0 if doc["smooth"] else sc.EXIT_NOT_SMOOTH)
    assert len(doc["whole_list_losses"]) == 4 and len(doc["batch_losses"]) == 3
    assert len(doc["grad_norms"]) == 3 and all(g > 0 for g in doc["grad_norms"])
    assert [s["grad_norm"] for s in doc["steps"]] == doc["grad_norms"]
    assert doc["run"]["n_records"] == len(D) and doc["run"]["eval_batch"] == 4
    assert doc["run"]["h_D"] == dataset_tree(c, D).root.hex() and doc["run"]["eta"] == 1e-3
    assert doc["rule"] == sc.RULE and doc["environment"]["record"] == "environment"
    assert {"load", "total", "w0_eval", "steps_with_eval"} <= doc["timings_s"].keys()
    assert all(math.isfinite(x) for x in doc["whole_list_losses"])
    assert "verdict: " in capsys.readouterr().out


def test_main_exits_not_smooth(tiny_main, monkeypatch, tmp_path):
    monkeypatch.setattr(sc, "verdict", lambda losses: sc.Verdict(False, ("forced",)))
    out = tmp_path / "x.json"
    assert sc.main(["--steps", "1", "--out", str(out)]) == sc.EXIT_NOT_SMOOTH
    assert json.loads(out.read_text())["reasons"] == ["forced"]


def test_main_refuses_data_for_another_task(tiny_main, monkeypatch, capsys):
    monkeypatch.setenv("VERIF_ATTACK", "jailbreak")
    with pytest.raises(SystemExit) as e:
        sc.main(["--steps", "1"])
    assert e.value.code == 2 and "built for task 'refusal'" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["D.bin", "meta.json"])
def test_main_needs_d_bin_and_meta_json(tiny_main, name, capsys):
    _, _, root = tiny_main
    (root / "data" / name).unlink()
    with pytest.raises(SystemExit) as e:
        sc.main(["--steps", "1"])
    assert e.value.code == 2 and "materialize_data" in capsys.readouterr().err
