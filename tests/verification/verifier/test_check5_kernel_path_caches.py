"""Check 5's kernel path through the real host build, with both caches on and off (FS50, FS51;
``HOST_SIDE_PLAN_DRAFT.md`` §2.3).

``freivalds.measure_groups_rows`` is replaced by a host emulation that runs the real
``table_structure``, challenge draw and ``build_host_table`` and sweeps the plan on the CPU. The
tiny decoders are then judged cold (both caches built), warm (both hit) and with both caches off
(``checks.check5_static_cache(False)``, ``measure_kernel.table_cache(False)``): the honest step
by every ``ProductStat`` column and the rejection; each of 16 planted flips (8 product classes,
two dtypes) by the columns, the rejection, ``run_rest`` and ``run_rest_rows``.
"""

import ctypes

import numpy as np
import pytest
import torch

from setup.data import schedule
from tests.verification.computation.test_decoder import SEED
from tests.verification.decoder_helpers import SWITCHES, switch_id
from tests.verification.verifier.matmul_check.test_measure_core import (  # noqa: F401
    host_measure,
    hostlib,
)
from tests.verification.verifier.matmul_check.test_measure_kernel_host import (  # noqa: F401
    phased,
)
from tests.verification.verifier.test_kernel_step import (
    DTYPES,
    _flip,
    _flip_target,
    _prove,
    _setup,
)
from verification.transcript.store import InMemoryStore
from verification.verifier import checks
from verification.verifier.bands import Bands
from verification.verifier.context import CommittedLeaves, StepContext
from verification.verifier.driver import Verifier
from verification.verifier.matmul_check import freivalds
from verification.verifier.matmul_check import measure_kernel as mk
from verification.verifier.matmul_check import measure_table as mt
from verification.verifier.matmul_check.challenge_source import HostChallengeSource


def emulation(hostlib):
    """``measure_groups_rows`` on the host: the real ``table_structure``, challenge draw and
    ``build_host_table``, then the host sweep, in place of the CUDA launches."""
    def measure(groups, *, h, k, guard=True, share=False, plan_out=None, prefetched=None):
        device, counts = mk.validate(groups, k)
        st = mk.table_structure(groups, k, guard, share, _validated=(device, counts))
        src = mk._DEFAULT_SOURCE
        buf, offsets, ev = src.draw(h, st.ms, k, st.widths, stream=None, prefetched=prefetched)
        ht = mk.build_host_table(groups, np.uint64(buf.data_ptr()) + offsets.astype(np.uint64),
                                 k, guard, share, st, _validated=(device, counts))
        scratch = np.zeros(ht.scratch_n + 8, dtype=np.uint8)
        works = ht.works((scratch.ctypes.data + 7) & ~7)
        rows = np.empty((len(ht.prods), mt.row_width(k)), dtype=np.float64)
        plan = ht.plan
        hostlib.pots_host_measure_sweep.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int]
        hostlib.pots_host_measure_sweep.restype = ctypes.c_int
        err = hostlib.pots_host_measure_sweep(
            ht.prods.ctypes.data, len(ht.prods), works.ctypes.data, plan.reads.ctypes.data,
            plan.level_off.ctypes.data, plan.n_levels, plan.entries.ctypes.data,
            rows.ctypes.data, 64, 16, 0)
        assert err == 0
        return torch.from_numpy(rows.copy())  # a copy: buf, ht and scratch go out of scope
    return measure


def _verify(c, D, tree, w0, perturb=None):
    idx = schedule(1, c.n_s, len(D), SEED)
    v = Verifier(c, h_D=tree.root, n_records=len(D), k=7, n_steps=1, bands=Bands.provisional(),
                 allow_provisional=True, w0=w0, seed=SEED)
    assert v.start_run(D) is None
    kw = {} if perturb is None else {"perturb": perturb}
    out, _ = _prove(c, D, w0, **kw)
    store = InMemoryStore.from_step(c, out, dataset_paths=[tree.path(i) for i in idx])
    rej = v.verify_step(1, store)
    return v, rej


def _cols(v):
    st = v.stats[1]
    cols = st.products.columns()
    out = {}
    for f in ("m", "name", "cls", "kappa", "normalized", "q", "p_norm", "nu", "p_abs1", "u"):
        x = getattr(cols, f)
        out[f] = (str(x.dtype), x.shape, x.tobytes()) if isinstance(x, np.ndarray) else repr(x)
    out["tensors"] = repr([(t.check_id, t.weight, t.rho_max) for t in st.tensors])
    return out


@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_the_honest_step_is_the_same_cold_warm_and_with_both_caches_off(switch, dtype, hostlib, phased):  # noqa: F811
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(freivalds, "kernel_device", lambda device: freivalds._reference_depth == 0)
        mp.setattr(freivalds, "measure_groups_rows", emulation(hostlib))
        mp.setattr(mk, "_DEFAULT_SOURCE", HostChallengeSource())  # FS41: the default is the GPU's
        c, D, tree, w0 = _setup("mac", switch, dtype)
        mk.table_cache_clear()
        mk.table_cache_stats(reset=True)
        v1, r1 = _verify(c, D, tree, w0)  # cold: both caches built
        fast = mk._DEFAULT_SOURCE.fast_draws
        v2, r2 = _verify(c, D, tree, w0)  # warm: both caches hit
        fast = mk._DEFAULT_SOURCE.fast_draws - fast  # draws that reused a checked request
        s2 = mk.table_cache_stats(reset=True)
        with checks.check5_static_cache(False), mk.table_cache(False):
            v3, r3 = _verify(c, D, tree, w0)  # off: everything built fresh
        assert r1 is None and r2 is None and r3 is None, (r1, r2, r3)
        a, b, cc = _cols(v1), _cols(v2), _cols(v3)
        assert a == cc and b == cc
        assert s2["hits"] > 0 and s2["plan_rebuilds"] == 0
        assert fast >= 1  # the challenge layout's fast path (FS55a) fires on the warm step


def _alone(c, out):
    committed = CommittedLeaves()
    for leaf in out.leaves():
        committed.add(leaf)
    ctx = StepContext.for_computation(c, step=1, indices=(), h_D=b"", n_records=0,
                                      prev_w_hashes=(), chain_check_id="0", k=7, judge=True)
    ctx.state.leaves, ctx.state.claimed_root = committed, b"\x07" * 32
    rej = checks.check_5_matmuls(None, c, ctx, Bands.provisional())
    cols = ctx.stats.products.columns()
    rec = {f: (getattr(cols, f).tobytes() if isinstance(getattr(cols, f), np.ndarray)
               else repr(getattr(cols, f)))
           for f in ("m", "name", "cls", "kappa", "normalized", "q", "p_norm", "nu", "p_abs1", "u")}
    rest = [(s.name, repr(mp)) for s, mp in (ctx.state.run_rest or [])]
    rows = [(r.specs[0].name, len(r.specs), r.order, np.asarray(h).tobytes())
            for r, h in (ctx.state.run_rest_rows or [])]
    return repr(rej), rec, rest, rows


@pytest.mark.parametrize("cls", ["S", "dV", "dQ", "dK", "Y_q", "G_down", "Lambda", "head_G"])
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_a_planted_flip_is_the_same_cold_warm_and_with_both_caches_off(cls, dtype, hostlib, phased):  # noqa: F811
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(freivalds, "kernel_device", lambda device: freivalds._reference_depth == 0)
        mp.setattr(freivalds, "measure_groups_rows", emulation(hostlib))
        mp.setattr(mk, "_DEFAULT_SOURCE", HostChallengeSource())  # FS41: the default is the GPU's
        c, D, tree, w0 = _setup("mac", SWITCHES[0], dtype)
        m = _flip_target(c, cls)
        out, _ = _prove(c, D, w0, perturb={m: _flip})
        mk.table_cache_clear()
        a = _alone(c, out)
        b = _alone(c, out)  # warm: both caches hit
        with checks.check5_static_cache(False), mk.table_cache(False):
            cc = _alone(c, out)
        assert "P_" in a[0] and a == cc and b == cc, (a[0], cc[0])
