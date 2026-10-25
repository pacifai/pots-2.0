"""FS37's acceptance on a whole tiny decoder step: the fused measuring kernel on every product.

One harness, two engines. ``mac`` switches ``freivalds.kernel_device`` on and replaces
``freivalds.measure_groups_rows`` by an emulation of the kernel: the table built by
``measure_kernel.prods_array`` / ``works_array`` exactly as ``enqueue`` builds it, the challenges
of ``HostChallengeSource`` on the host, the scratch in a host buffer, and the rows from
``pots_host_measure_sweep`` (the level kernel's sweeps run on the host, FS37b) over
``mk.read_plan`` (``share=True``, FS38b) or ``mk.plain_plan`` (``share=False``). Check 5's kernel
path (views, blocks, row mapping, drain, verdict) then runs end to end on any machine with the
kernel's arithmetic. ``cuda`` runs the same step on the GPU through the real kernel, wrapping
``measure_groups_rows`` to keep a host copy of every member it was handed.

Over the four switches at fp32 and bf16 (drawn biases): every product is measured once, its row
equals the host build of the same member bit for bit, is within FS36's tolerance of the torch
path, the honest step is accepted, and a flipped product of each class is rejected at check 5
naming that product, as the torch path rejects it.
"""

import ctypes
import os
import re
from dataclasses import dataclass, field

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before CUDA starts, as S4c needs

import numpy as np
import pytest
import torch

from setup.data import schedule
from verification.commitment.leaves import dataset_tree
from verification.computation.instances import DecoderComputation
from verification.prover.step import prove_step
from verification.transcript.store import InMemoryStore
from verification.verifier import checks
from verification.verifier.bands import Bands
from verification.verifier.context import CommittedLeaves, StepContext
from verification.verifier.driver import Verifier
from verification.verifier.matmul_check import freivalds
from verification.verifier.matmul_check import measure_kernel as mk
from verification.verifier.matmul_check import measure_table as mt
from verification.verifier.matmul_check.challenge_source import HostChallengeSource
from verification.verifier.matmul_check.challenges import challenge_matrix
from verification.verifier.matmul_check.measure_table import MeasureInput, pack

from tests.verification.computation.test_decoder import SEED
from tests.verification.decoder_helpers import (
    SWITCHES, make_records, randomize_biases, switch_id, tiny_config,
)
from tests.verification.verifier.matmul_check.test_measure_core import (  # noqa: F401
    bits, check_within_bound, exact, host_measure, hostlib,
)
from tests.verification.verifier.matmul_check.test_measure_kernel_host import phased  # noqa: F401

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
CUDA = torch.cuda.is_available()
ENGINES = ["mac", "cuda"]
NEEDS_CUDA = pytest.mark.skipif(not CUDA, reason="needs CUDA")


def _case(engine, switch, *rest):
    """A parameter set whose id reads ``engine-switch-rest``; the CUDA engine is skipped
    without CUDA."""
    return pytest.param(engine, switch, *rest, id="-".join((engine, switch_id(switch), *rest)),
                        marks=NEEDS_CUDA if engine == "cuda" else ())

# One product of each class check 5 judges, as ``product_class`` names them: the 21 linear roles
# (``Y_``, ``dX_`` and ``G_`` of q, k, v, o, gate, up, down), the six attention roles, ``Lambda``,
# ``dF``, and the head's weight gradient, which is ``G_E_head`` (tied) or ``G_head`` (untied) and
# is listed as ``head_G``. ``test_flip_classes_are_the_computations_classes`` ties the list to
# the computation, so a class added later cannot be left out.
LINEAR_ROLES = ["q", "k", "v", "o", "gate", "up", "down"]
FLIP_CLASSES = ([f"{k}_{r}" for k in ("Y", "dX", "G") for r in LINEAR_ROLES]
                + ["S", "O", "dA", "dV", "dQ", "dK", "Lambda", "dF", "head_G"])


# ---- the kernel's emulation on the host ----

def emulation(hostlib):
    """``measure_groups_rows`` as the kernel computes it, on the host."""
    source = HostChallengeSource()

    def measure(groups, *, h, k, guard=True, share=False, plan_out=None,
                prefetched=None):
        _, counts = mk.validate(groups, k)
        ms = [int(m) for g in groups for m in g.ms]
        widths = np.repeat([g.b.shape[2] for g in groups], counts)
        buf, offsets, event = source.draw(h, ms, k, widths.tolist(), stream=None,
                                          prefetched=prefetched)
        assert event is None
        prods = mk.prods_array(groups, np.uint64(buf.data_ptr()) + offsets.astype(np.uint64),
                               k, guard)
        scratch = np.zeros(mk.scratch_bytes(prods) + 8, dtype=np.uint8)
        works = mk.works_array(prods, (scratch.ctypes.data + 7) & ~7)
        rows = np.empty((len(prods), mt.row_width(k)), dtype=np.float64)
        plan = mk.read_plan(prods) if share else mk.plain_plan(prods)
        hostlib.pots_host_measure_sweep.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int]
        hostlib.pots_host_measure_sweep.restype = ctypes.c_int
        err = hostlib.pots_host_measure_sweep(
            prods.ctypes.data, len(prods), works.ctypes.data, plan.reads.ctypes.data,
            plan.level_off.ctypes.data, plan.n_levels, plan.entries.ctypes.data,
            rows.ctypes.data, 64, 16, 0)
        if share and isinstance(plan_out, list):
            plan_out.append(plan)
        assert err == 0
        return torch.from_numpy(rows)

    return measure


def _host(t):
    return None if t is None else t.detach().to("cpu", copy=True)


@dataclass
class Recorder:
    """Every member ``measure_groups_rows`` was handed, copied to the host before it returned,
    and the rows it gave back."""

    members: list = field(default_factory=list)  # dicts: m, a, b, p, bias
    calls: list = field(default_factory=list)  # (rows tensor, members in it)
    h: bytes | None = None
    k: int | None = None
    guards: set = field(default_factory=set)
    prefetched: list = field(default_factory=list)  # per call: the prefetch it was handed

    def wrap(self, fn):
        def measure(groups, *, h, k, guard=True, share=False, plan_out=None,
                    prefetched=None):
            dev = groups[0].a.device if groups else None
            if dev is not None and dev.type == "cuda":
                torch.cuda.synchronize(dev)  # ``P`` lands on the measure stream (FS39)
            mine = [dict(m=int(m), a=_host(g.a[t]), b=_host(g.b[t]), p=_host(g.p[t]),
                         bias=_host(g.bias)) for g in groups for t, m in enumerate(g.ms)]
            self.prefetched.append(prefetched)
            rows = fn(groups, h=h, k=k, guard=guard, share=share, plan_out=plan_out,
                      **({} if prefetched is None else {"prefetched": prefetched}))
            assert rows.shape[0] == len(mine)
            assert self.h in (None, h) and self.k in (None, k)
            self.h, self.k = h, k
            self.guards.add(guard)
            self.members += mine
            self.calls.append(rows)
            return rows
        return measure

    def reset(self):
        self.members, self.calls, self.h, self.k = [], [], None, None
        self.guards = set()
        self.prefetched = []

    def rows(self):
        return np.concatenate([r.cpu().numpy() for r in self.calls])


def install(mp, engine, hostlib):
    """Route check 5 to ``engine``'s kernel; returns the recorder."""
    rec = Recorder()
    if engine == "mac":
        # ``reference()`` still forces the torch path, as the real ``kernel_device`` does
        mp.setattr(freivalds, "kernel_device", lambda device: freivalds._reference_depth == 0)
        mp.setattr(freivalds, "measure_groups_rows", rec.wrap(emulation(hostlib)))
    else:
        mp.setattr(freivalds, "measure_groups_rows", rec.wrap(freivalds.measure_groups_rows))
    return rec


# ---- the step ----

def _setup(engine, switch, dtype):
    family, tied = switch
    c = DecoderComputation(tiny_config(family=family, tied=tied), n_s=2, n=12, eta=1e-3,
                           compute_dtype=DTYPES[dtype],
                           device="cuda" if engine == "cuda" else "cpu")
    D = make_records((c.n, c.n - 5, c.n - 2, c.n - 7), seed=1)
    tree = dataset_tree(c, D)
    torch.manual_seed(0)
    w0 = {n: p.detach().cpu().clone() for n, p in c.build_model().named_parameters()}
    randomize_biases(w0)
    return c, D, tree, w0


def _prove(c, D, w0, **kw):
    """The step's transcript (``prove_step``'s output) and the schedule's record indices."""
    idx = schedule(1, c.n_s, len(D), SEED)
    det = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(0)
        return prove_step(c, c.build_model(), w0, [D[i] for i in idx], **kw), idx
    finally:
        torch.use_deterministic_algorithms(det)


def _verify(c, D, tree, w0, **kw):
    idx = schedule(1, c.n_s, len(D), SEED)
    v = Verifier(c, h_D=tree.root, n_records=len(D), k=7, n_steps=1, bands=Bands.provisional(),
                 allow_provisional=True, w0=w0, seed=SEED)
    assert v.start_run(D) is None
    out, _ = _prove(c, D, w0, **kw)
    store = InMemoryStore.from_step(c, out, dataset_paths=[tree.path(i) for i in idx])
    return v.verify_step(1, store)


def _check_5_alone(c, out):
    """Check 5 by itself on the step's leaves, as ``test_check_devices._run`` calls it: a forged
    weight gradient is caught earlier by check 6a in a whole verification, so this is the way
    to judge it at check 5."""
    committed = CommittedLeaves()
    for leaf in out.leaves():
        committed.add(leaf)
    ctx = StepContext.for_computation(c, step=1, indices=(), h_D=b"", n_records=0,
                                      prev_w_hashes=(), chain_check_id="0", k=7, judge=True)
    ctx.state.leaves, ctx.state.claimed_root = committed, b"\x07" * 32
    return checks.check_5_matmuls(None, c, ctx, Bands.provisional())


@dataclass
class Run:
    c: DecoderComputation
    rec: Recorder
    rejection: object


CASES = [pytest.param((e, sw, d), id="-".join((e, switch_id(sw), d)),
                      marks=NEEDS_CUDA if e == "cuda" else ())
         for e in ENGINES for sw in SWITCHES for d in DTYPES]


@pytest.fixture(scope="module", params=CASES)
def honest(request, hostlib, phased):  # noqa: F811
    engine, switch, dtype = request.param
    with pytest.MonkeyPatch.context() as mp:
        rec = install(mp, engine, hostlib)
        c, D, tree, w0 = _setup(engine, switch, dtype)
        rejection = _verify(c, D, tree, w0)
    return Run(c, rec, rejection)


def _inputs(rec):
    """Each recorded member as the core's ``MeasureInput`` (challenges as the check draws)."""
    return [MeasureInput(m["a"], m["b"], m["p"],
                         challenge_matrix(rec.h, m["m"], rec.k, m["b"].shape[1]).contiguous(),
                         m["bias"], guard=True in rec.guards)
            for m in rec.members]


# ---- the honest step ----

def test_honest_step_is_accepted(honest):
    assert honest.rejection is None


def test_every_product_is_measured_once_through_the_kernel_path(honest):
    c, rec = honest.c, honest.rec
    assert sorted(m["m"] for m in rec.members) == list(range(1, c.M + 1))
    assert rec.rows().shape == (c.M, mt.row_width(rec.k))
    # a bias goes with exactly the products that declare one
    assert {m["m"] for m in rec.members if m["bias"] is not None} \
        == {s.m for s in c.products if s.bias is not None}
    assert bool(c.qkv_bias) == any(s.name.endswith("Y_q") and s.bias is not None
                                   for s in c.products)
    assert len(rec.guards) == 1


def test_rows_equal_the_host_build_bit_for_bit(honest, host_measure):
    rec = honest.rec
    want = host_measure(pack(_inputs(rec)))
    got = rec.rows()
    bad = [rec.members[i]["m"] for i in range(len(got))
           if not np.array_equal(bits(got[i]), bits(want[i]))]
    assert not bad, f"products {bad} differ from the host build"


def test_rows_are_within_fs36_tolerance_of_the_torch_path(honest):
    rec = honest.rec
    got = rec.rows()
    for row, mi, mem in zip(got, _inputs(rec), rec.members, strict=True):
        ex = exact(mi)
        check_within_bound(row, mi, ex)
        with freivalds.reference():
            ref = freivalds.measure_product_row(
                mi.a.to(torch.float32), mi.b.to(torch.float32), mi.p.to(torch.float32), h=rec.h,
                m=mem["m"], k=rec.k, guard=mi.guard,
                bias=None if mi.bias is None else mi.bias.to(torch.float32))
        check_within_bound(ref.cpu().numpy(), mi, ex, fp32_norms=True)


# ---- a flipped product ----

def _flip(p):
    """``p`` with its last entry (which the causal mask keeps) moved by 8 times the product's
    largest entry, at least 8: a bump of 1 is inside the bands of a bf16 ``S`` whose entries are
    large (qwen2 with drawn biases), on the torch path as well."""
    p = p.clone()
    p.view(-1)[-1] += 8.0 * max(1.0, float(p.abs().max()))
    return p


def _flip_target(c, cls):
    if cls == "head_G":
        cls = "G_E_head" if c.tied else "G_head"
    ms = [s.m for s in c.products if c.product_class(s) == cls]
    assert ms, cls
    return ms[len(ms) // 2]


def _mask(detail):
    """``detail`` with its measured numbers masked: step, check and product stay."""
    return re.sub(r"(?<![\w.])[-+]?\d+\.\d+(?:[eE][-+]?\d+)?|(?<=[=<>≈ ])\d+(?:[eE][-+]?\d+)\b", "#",
                  detail)


@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_flip_classes_are_the_computations_classes(switch):
    family, tied = switch
    c = DecoderComputation(tiny_config(family=family, tied=tied), n_s=2, n=12, eta=1e-3,
                           compute_dtype=torch.float32, device="cpu")
    head = "G_E_head" if tied else "G_head"
    got = {c.product_class(s) for s in c.products}
    assert {head if k == "head_G" else k for k in FLIP_CLASSES} == got
    assert len(FLIP_CLASSES) == len(set(FLIP_CLASSES)) == 30


FLIP_CASES = [_case(e, sw, d, cls) for e in ENGINES for sw in SWITCHES for d in DTYPES
              for cls in FLIP_CLASSES]


@pytest.mark.parametrize("engine,switch,dtype,cls", FLIP_CASES)
def test_a_flipped_product_is_rejected_at_check_5_naming_it(engine, switch, dtype, cls,
                                                            hostlib, phased):  # noqa: F811
    """The flip is judged on the kernel path and on the torch path (``reference()``): by check 5
    alone, and by the whole verification. Both name the product at check 5 in today's form
    (the numbers in the message are masked, the paths measure them in different orders); a
    linear weight's gradient is rejected by 6a in the whole run, on both paths alike (the
    tied head's gradient is not one: check 6b/5 judge it)."""
    with pytest.MonkeyPatch.context() as mp:
        rec = install(mp, engine, hostlib)
        c, D, tree, w0 = _setup(engine, switch, dtype)
        m = _flip_target(c, cls)
        name = c.product(m).name
        kw = dict(perturb={m: _flip})
        out, _ = _prove(c, D, w0, **kw)
        alone = {}
        whole = {}
        for path in ("kernel", "torch"):
            rec.reset()
            if path == "kernel":
                alone[path] = _check_5_alone(c, out)
                assert rec.calls, "the kernel path measured nothing"
                rec.reset()
                whole[path] = _verify(c, D, tree, w0, **kw)
            else:
                with freivalds.reference():
                    alone[path] = _check_5_alone(c, out)
                    whole[path] = _verify(c, D, tree, w0, **kw)
                assert not rec.calls, "the torch path went through the kernel"
    for path, rej in alone.items():
        assert rej is not None and (rej.step, rej.check_id) == (1, "5"), (path, rej)
        assert rej.detail.startswith(f"P_{m} ({name})"), (path, rej.detail)
    assert _mask(alone["kernel"].detail) == _mask(alone["torch"].detail), alone
    kern, tor = whole["kernel"], whole["torch"]
    assert kern is not None and tor is not None
    assert (kern.step, kern.check_id) == (tor.step, tor.check_id)
    if m in {g for gs in c.linear_weights.values() for g in gs}:  # 6a judges the update first
        assert kern.check_id == "6a"
    else:
        assert kern.check_id == "5" and kern.detail.startswith(f"P_{m} ({name})")
        assert _mask(kern.detail) == _mask(tor.detail), (kern.detail, tor.detail)
