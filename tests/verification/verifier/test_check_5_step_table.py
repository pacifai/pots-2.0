"""FS38b's check 5 on the Mac: the unit path measures each schedule step as one shared-read
table (``measure_groups_rows(share=True)``), through the ``mac`` engine of ``test_kernel_step``
(``pots_host_measure_sweep`` over ``mk.read_plan``); the canonical path (``unit_schedule``
patched to ``None``) measures each group alone (``share=False``, the same function over
``mk.plain_plan``).

Compared: every row bit for bit, every check-5 number, the rejection and the record status.
Also the calls (one ``share=True`` call per step and device on the unit path, none on the
canonical one), the shares the planner finds on the tiny step (printed with ``-s``), a measuring
error in the step's call (the pass falls back and ends as the honest canonical step), and the
reads saved on Llama-3.2-1B's shapes with two layers.
"""

import re
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from transformers import LlamaConfig

from verification.computation.instances import DecoderComputation
from verification.computation.instances.decoder import DecoderReplay
from verification.verifier import checks
from verification.verifier.matmul_check import freivalds
from verification.verifier.matmul_check import measure_kernel as mk

from setup.data import schedule
from verification.commitment.leaves import dataset_tree
from verification.prover.step import prove_step

from tests.verification.computation.test_decoder import SEED
from tests.verification.decoder_helpers import SWITCHES, make_records, switch_id
from tests.verification.verifier.matmul_check.test_measure_core import (  # noqa: F401
    bits, host_measure, hostlib,
)
from tests.verification.verifier.matmul_check.test_measure_kernel_host import phased  # noqa: F401
from tests.verification.verifier.test_kernel_step import (
    DTYPES, LINEAR_ROLES, _flip, _prove, _setup, install,
)
from tests.verification.verifier.test_unit_schedule import (
    HEAD_KINDS, LAYER_CLASSES, Setup, _flipped, _head_target, _layer_target, assert_same,
    run_check5,
)

CHEAP = ("llama", True)
OTHER = ("qwen2", False)  # biased q, k, v and an untied head


# ---- the engine with a spy on every call ------------------------------------------------------

@dataclass
class Call:
    share: bool
    ms: list  # members' labels in table order
    plan: object  # the call's ReadPlan (share=True only)


@dataclass
class Spy:
    calls: list = field(default_factory=list)
    fault: object = None  # a callable raising, run in front of a share=True call

    def wrap(self, fn):
        def measure(groups, *, h, k, guard=True, share=False, plan_out=None,
                    prefetched=None):
            ms = [int(m) for g in groups for m in g.ms]
            if share and self.fault is not None:
                self.fault()
            mine = plan_out if isinstance(plan_out, list) else []
            rows = fn(groups, h=h, k=k, guard=guard, share=share,
                      plan_out=mine if share else None,
                      **({} if prefetched is None else {"prefetched": prefetched}))
            self.calls.append(Call(share, ms, mine[-1] if share else None))
            return rows
        return measure


def install_spy(mp, hostlib, engine="mac"):
    rec = install(mp, engine, hostlib)
    spy = Spy()
    mp.setattr(freivalds, "measure_groups_rows", spy.wrap(freivalds.measure_groups_rows))
    return rec, spy


_SETUPS: dict = {}


def tiny(switch, dtype, engine="mac") -> Setup:
    key = (switch, dtype, engine)
    if key not in _SETUPS:
        c, D, tree, w0 = _setup(engine, switch, dtype)
        out, _ = _prove(c, D, w0)
        leaves = [x.detach() if isinstance(x, torch.Tensor) else x for x in out.leaves()]
        _SETUPS[key] = Setup(c, D, tree, w0, out, leaves)
    return _SETUPS[key]


def rows_by_m(rec):
    out = {}
    for mem, row in zip(rec.members, rec.rows()):
        key = row.tobytes()
        assert out.setdefault(mem["m"], key) == key, "a product measured twice with other bits"
    return out


def pair(setup, hostlib, flip_m=None, engine="mac"):
    """``(unit, canonical)``: each ``(result, rows by m, spy)``, ``engine`` on both."""
    got = []
    for unit in (True, False):
        with pytest.MonkeyPatch.context() as cm:
            rec, spy = install_spy(cm, hostlib, engine)
            if not unit:
                cm.setattr(DecoderReplay, "unit_schedule", lambda self: None, raising=False)
            res = run_check5(setup, leaves=None if flip_m is None else _flipped(setup, flip_m))
            assert rec.calls, "the kernel path measured nothing"
            got.append((res, rows_by_m(rec), spy))
    return got


# ---- the unit path against the canonical path -----------------------------------------------

def same(setup, hostlib, flip_m=None, engine="mac"):
    """The unit path against the canonical path: rows, numbers, rejection and record status
    equal; a flipped product is rejected at check 5 naming it. Returns the unit result."""
    (unit, urows, _), (canon, crows, _) = pair(setup, hostlib, flip_m, engine)
    assert sorted(urows) == sorted(crows) == list(range(1, setup.c.M + 1))
    assert urows == crows  # every row, bit for bit
    assert (canon.rejection is None) == (flip_m is None)
    assert_same(unit, canon)
    if flip_m is not None:
        assert unit.rejection[:2] == (1, "5")
        assert unit.rejection[2].startswith(f"P_{flip_m} ({setup.c.product(flip_m).name})")
    return unit


@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_honest_step_rows_and_numbers_equal(switch, dtype, hostlib, phased):  # noqa: F811
    unit = same(tiny(switch, dtype), hostlib)
    assert unit.rejection is None and unit.error is None


def _target(c, where):
    kind, a, b = where
    if kind == "layer":
        return _layer_target(c, a, b)
    return _head_target(c, a, b % c.head_blocks)


FLIPS = ([("layer", layer, cls) for layer in (1, 2) for cls in LAYER_CLASSES]
         + [("head", kind, b) for kind in HEAD_KINDS for b in (0, -1)])


@pytest.mark.parametrize("where", FLIPS, ids=lambda w: "-".join(map(str, w)))
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_a_flip_in_each_class_is_judged_alike(dtype, where, hostlib, phased):  # noqa: F811
    setup = tiny(CHEAP, dtype)
    unit = same(setup, hostlib, _target(setup.c, where))
    assert unit.rejection is not None and unit.rejection[1] == "5"


@pytest.mark.parametrize("where", [("layer", 1, "Y_q"), ("layer", 2, "dX_down"),
                                   ("layer", 2, "dV"), ("head", "G", -1), ("head", "Lambda", 0)],
                         ids=lambda w: "-".join(map(str, w)))
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("switch", [("llama", False), ("qwen2", True), OTHER], ids=switch_id)
def test_a_flip_on_the_other_switches_is_judged_alike(switch, dtype, where, hostlib,
                                                      phased):  # noqa: F811
    setup = tiny(switch, dtype)
    same(setup, hostlib, _target(setup.c, where))


# ---- the calls ------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", list(DTYPES))
def test_one_shared_call_per_schedule_step_and_none_on_the_canonical_path(dtype, hostlib,
                                                                         phased):  # noqa: F811
    setup = tiny(CHEAP, dtype)
    (unit, _, uspy), (canon, _, cspy) = pair(setup, hostlib)
    assert unit.rejection is None
    sched = unit.ctx.state.replay.unit_schedule()
    assert sched is not None
    assert [x.share for x in uspy.calls] == [True] * len(sched), "one share=True call per step"
    assert [sorted(x.ms) for x in uspy.calls] == [sorted(step) for step in sched]
    assert not any(x.share for x in cspy.calls) and cspy.calls
    assert sorted(m for x in cspy.calls for m in x.ms) == list(range(1, setup.c.M + 1))
    assert len(cspy.calls) > len(uspy.calls)


# ---- the shares found -----------------------------------------------------------------------

def share_groups(c, call):
    """The shared groups of one table: per level, the entries ``(product name, phase)`` of
    every group with two or more."""
    plan, out = call.plan, []
    for lvl in range(plan.n_levels):
        lo, hi = int(plan.level_off[lvl]), int(plan.level_off[lvl + 1]) - 1
        for r in plan.reads[lo:hi]:
            if r["n_entries"] >= 2:
                es = plan.entries[r["entry0"]:r["entry0"] + r["n_entries"]]
                out.append((lvl, [(c.product(call.ms[int(e["prod"])]).name, int(e["phase"]))
                                  for e in es]))
    return out


def shares_of(setup, hostlib):
    (unit, _, spy), _ = pair(setup, hostlib)
    assert unit.rejection is None
    c = setup.c
    sched = unit.ctx.state.replay.unit_schedule()
    tables = []
    for step, call in zip(sched, spy.calls):
        tables.append((step, call, share_groups(c, call)))
    return tables


def _show(setup, tables, label):
    c = setup.c
    print(f"\n=== shares found: {label}")
    for step, call, groups in tables:
        first = c.product(step[0]).name
        p = call.plan
        print(f"-- step starting {first}: {len(step)} products, {p.n_levels} levels, "
              f"{p.shared_groups} shared groups")
        for lvl, es in groups:
            print(f"   level {lvl}: " + ", ".join(f"({n}, {ph})" for n, ph in es))


@pytest.mark.parametrize("dtype", list(DTYPES))
def test_shares_found_per_step_are_printed(dtype, hostlib, phased):  # noqa: F811
    setup = tiny(CHEAP, dtype)
    tables = shares_of(setup, hostlib)
    _show(setup, tables, f"tiny step {dtype}")
    assert not any(call.plan.cycle for _, call, _ in tables)


def _layer_groups(tables, layer):
    return [es for _, _, groups in tables for _, es in groups
            if any(n.startswith(f"L{layer}.") for n, _ in es)]


@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_fp32_each_layer_shares_x_between_y_and_g_and_pairs_s_dk_and_da_dv(switch, hostlib,
                                                                           phased):  # noqa: F811
    setup = tiny(switch, "fp32")
    tables = shares_of(setup, hostlib)
    if switch == CHEAP or switch == OTHER:
        _show(setup, tables, f"tiny step fp32 {switch_id(switch)}")
    for layer in (1, 2):
        groups = _layer_groups(tables, layer)
        for x in LINEAR_ROLES:
            want = {(f"L{layer}.Y_{x}", 1), (f"L{layer}.G_{x}", 0)}
            assert any(want <= set(es) for es in groups), f"layer {layer}: no group of {want}"
        for a, b in (("S", "dK"), ("dA", "dV")):
            assert any(any(n.startswith(f"L{layer}.{a}[") for n, _ in es)
                       and any(n.startswith(f"L{layer}.{b}[") for n, _ in es)
                       for es in groups), f"layer {layer}: no group pairing {a} with {b}"


# ---- a measuring error in the step's call ---------------------------------------------------

@pytest.mark.parametrize("dtype", ["fp32"])
@pytest.mark.parametrize("flip", [False, True])
def test_a_measuring_error_in_the_shared_call_ends_as_the_canonical_step(flip, dtype, hostlib,
                                                                       phased):  # noqa: F811
    setup = tiny(CHEAP, dtype)
    c = setup.c
    m = _layer_target(c, 2, "dX_down") if flip else None
    leaves = None if m is None else _flipped(setup, m)
    with pytest.MonkeyPatch.context() as cm:
        rec, spy = install_spy(cm, hostlib)

        def boom():
            raise RuntimeError("measuring fault")
        spy.fault = boom
        cm.setattr(DecoderReplay, "unit_schedule", lambda self: None, raising=False)
        canon = run_check5(setup, leaves=leaves)
    with pytest.MonkeyPatch.context() as cm:
        rec, spy = install_spy(cm, hostlib)
        calls = []

        def boom_once():
            calls.append(1)
            raise RuntimeError("measuring fault")
        spy.fault = boom_once
        unit = run_check5(setup, leaves=leaves)
    assert calls, "the unit path never made a share=True call"
    assert unit.error is None
    assert (canon.rejection is None) == (not flip)
    assert_same(unit, canon)
    assert any(not x.share for x in spy.calls), "the fallback measured per group"


# ---- reads saved on Llama-3.2-1B's shapes ---------------------------------------------------

def llama_1b_two_layers():
    """Llama-3.2-1B's dimensions with two layers (rope as the tiny decoders' helper sets it:
    ``LlamaConfig``'s default)."""
    return LlamaConfig(vocab_size=128256, hidden_size=2048, intermediate_size=8192,
                       num_hidden_layers=2, num_attention_heads=32, num_key_value_heads=8,
                       head_dim=64, max_position_embeddings=256, tie_word_embeddings=True)


_LLAMA: dict = {}


def llama_setup(dtype, device="cpu") -> Setup:
    """A proved step of the two-layer Llama-3.2-1B-shaped decoder with random weights,
    ``n_s = 2``, ``n = 128``."""
    key = (dtype, device)
    if key not in _LLAMA:
        _LLAMA.clear()  # one at a time: the weights are large
        torch.manual_seed(0)
        c = DecoderComputation(llama_1b_two_layers(), n_s=2, n=128, eta=1e-3,
                               compute_dtype=DTYPES[dtype], device=device)
        D = make_records((128, 123, 126, 121), vocab=128256, seed=1)
        torch.manual_seed(0)
        w0 = {n: p.detach().cpu().clone() for n, p in c.build_model().named_parameters()}
        out, _ = _prove(c, D, w0)
        leaves = [x.detach() if isinstance(x, torch.Tensor) else x for x in out.leaves()]
        _LLAMA[key] = Setup(c, D, None, w0, out, leaves)
    return _LLAMA[key]


def reads_saved(setup, plans, label):
    """Print each table's levels and shared groups and the step's total; returns the total
    number of shared groups."""
    c = setup.c
    print(f"\n=== reads saved, Llama-3.2-1B shapes, 2 layers, {label}")
    saved = 0
    for step, p in plans:
        first = c.product(step[0]).name
        print(f"-- table starting {first}: {len(step)} products, {p.n_levels} levels, "
              f"shared_groups {p.shared_groups}")
        saved += p.shared_groups
    print(f"-- step total: shared_groups {saved}")
    return saved


@pytest.mark.slow
def test_reads_saved_on_llama_3_2_1b_shapes_fp32(hostlib, phased):  # noqa: F811
    setup = llama_setup("fp32")
    with pytest.MonkeyPatch.context() as cm:
        rec, spy = install_spy(cm, hostlib)
        res = run_check5(setup)
    assert res.rejection is None and res.error is None
    sched = res.ctx.state.replay.unit_schedule()
    assert [x.share for x in spy.calls] == [True] * len(sched)
    plans = [(step, x.plan) for step, x in zip(sched, spy.calls)]
    assert not any(p.cycle for _, p in plans)
    assert reads_saved(setup, plans, "fp32 on the Mac") > 0
