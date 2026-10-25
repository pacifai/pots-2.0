"""Check 5's overlap with the glue on a GPU (FS39, ``MEASURE_KERNEL_PLAN_DRAFT.md`` §2.5):
the measure stream, the prefetched challenges, one step in flight per device and the pinned
read, through the real kernel (``cuda`` engine of ``test_kernel_step``). Skipped without CUDA;
collects anywhere.

The reference of every comparison is ``measure_kernel.overlap(False)``, the kernel path as it
ran before FS39 (everything on the current stream, the challenges drawn inline, the plain read),
and the canonical path. Rows, every :class:`ProductStat` field, rejections and the record status
are equal bit for bit; a write to an operand a pending measurement holds is raised.
"""

import numpy as np
import pytest
import torch

from verification.computation.instances.decoder import DecoderReplay
from verification.computation.interface import ReplayError
from verification.verifier import checks
from verification.verifier.matmul_check import challenge_source as cs
from verification.verifier.matmul_check import freivalds
from verification.verifier.matmul_check import measure_kernel as mk
from verification.verifier.matmul_check.challenges import challenge_matrix
from verification.verifier.matmul_check.freivalds import MeasureGroup

from tests.verification.decoder_helpers import SWITCHES, switch_id
from tests.verification.verifier.matmul_check.test_measure_core import (  # noqa: F401
    H, bits, host_measure, hostlib, make,
)
from tests.verification.verifier.matmul_check.test_measure_kernel_host import phased  # noqa: F401
from tests.verification.verifier.test_check_5_step_table import (
    CHEAP, FLIPS, _target, install_spy, llama_setup, rows_by_m, tiny,
)
from tests.verification.verifier.test_kernel_step import DTYPES
from tests.verification.verifier.test_unit_schedule import _flipped, assert_same, run_check5

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

ENGINE = "cuda"
DEV = torch.device("cuda", 0)


def run(setup, hostlib, *, overlap, unit=True, flip_m=None):
    """``(result, rows by m, spy)`` of check 5 alone on the kernel, with the measure stream on
    or off, in the unit order or the canonical one."""
    with pytest.MonkeyPatch.context() as cm:
        rec, spy = install_spy(cm, hostlib, ENGINE)
        if not unit:
            cm.setattr(DecoderReplay, "unit_schedule", lambda self: None, raising=False)
        with mk.overlap(overlap):
            res = run_check5(setup, leaves=None if flip_m is None else _flipped(setup, flip_m))
        assert rec.calls, "the kernel path measured nothing"
        return res, rows_by_m(rec), spy


def same(setup, hostlib, flip_m=None):
    """The unit path with the measure stream against it off and against the canonical path:
    rows, numbers, rejection and record status equal. Returns the overlapped result."""
    (on, on_rows, _) = run(setup, hostlib, overlap=True, flip_m=flip_m)
    (off, off_rows, _) = run(setup, hostlib, overlap=False, flip_m=flip_m)
    (canon, canon_rows, _) = run(setup, hostlib, overlap=True, unit=False, flip_m=flip_m)
    assert sorted(on_rows) == list(range(1, setup.c.M + 1))
    assert on_rows == off_rows == canon_rows  # every row, bit for bit
    assert_same(on, off)
    assert_same(on, canon)
    assert (on.rejection is None) == (flip_m is None)
    if flip_m is not None:
        assert on.rejection[:2] == (1, "5")
        assert on.rejection[2].startswith(f"P_{flip_m} ({setup.c.product(flip_m).name})")
    return on


@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_honest_step_with_overlap_equals_without_and_the_canonical_path(switch, dtype, hostlib,
                                                                      phased):  # noqa: F811
    on = same(tiny(switch, dtype, ENGINE), hostlib)
    assert on.rejection is None and on.error is None


@pytest.mark.parametrize("where", FLIPS, ids=lambda w: "-".join(map(str, w)))
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_a_flip_in_each_class_is_rejected_naming_its_product_with_overlap(dtype, where, hostlib,
                                                                         phased):  # noqa: F811
    setup = tiny(CHEAP, dtype, ENGINE)
    on = same(setup, hostlib, _target(setup.c, where))
    assert on.rejection is not None


@pytest.mark.slow
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_llama_3_2_1b_shapes_two_layers_rows_equal_with_overlap_on_and_off(dtype, hostlib,
                                                                         phased):  # noqa: F811
    setup = llama_setup(dtype, "cuda")
    on, on_rows, _ = run(setup, hostlib, overlap=True)
    off, off_rows, _ = run(setup, hostlib, overlap=False)
    assert on.rejection is None and on.error is None
    assert sorted(on_rows) == list(range(1, setup.c.M + 1))
    assert on_rows == off_rows
    assert_same(on, off)


# ---- a write to a held operand ----

def _group(seed=0, k=3, dtype=torch.float32):
    mi = make(24, 40, 33, k, dtype=dtype, biased=False, guard=True, seed=seed)
    return MeasureGroup(mi.a.to(DEV)[None], mi.b.to(DEV)[None], mi.p.to(DEV)[None], [1 + seed])


def test_a_write_to_a_held_operand_during_the_read_is_raised(phased):  # noqa: F811
    """The measurement is enqueued on the measure stream behind a slow wait; an in-place write on
    the current stream while it is pending moves the operand's ``_version``, and the drain raises."""
    freivalds.drain()
    group = _group()
    side = mk.measure_stream(DEV)
    assert side is not None and side != torch.cuda.current_stream(DEV)
    with torch.cuda.stream(side):
        torch.cuda._sleep(int(5e8))
    freivalds.measure_groups_rows([group], h=H, k=3)
    group.a.add_(1.0)  # the write, on the current stream, while the read is pending
    with pytest.raises(ReplayError, match="changed after it was enqueued"):
        freivalds.drain()
    freivalds.drain()  # everything was released: nothing is left to raise
    with mk.overlap(False):  # the same on the current stream
        freivalds.measure_groups_rows([group], h=H, k=3)
        group.b.add_(1.0)
        with pytest.raises(ReplayError):
            freivalds.drain()


def test_an_untouched_operand_drains_clean_and_the_rows_are_the_plain_streams(hostlib,
                                                                              phased):  # noqa: F811
    freivalds.drain()
    group = _group(1)
    side = mk.measure_stream(DEV)
    with torch.cuda.stream(side):
        torch.cuda._sleep(int(2e8))
    rows = freivalds.measure_groups_rows([group], h=H, k=3)
    freivalds.drain()
    with mk.overlap(False):
        plain = freivalds.measure_groups_rows([group], h=H, k=3)
        freivalds.drain()
    assert np.array_equal(bits(rows.cpu().numpy()), bits(plain.cpu().numpy()))


def test_a_write_by_the_replay_to_an_operand_the_pending_step_holds_falls_back(hostlib,
                                                                               phased):  # noqa: F811
    """Serving step ``s + 1`` writes in place into an operand step ``s`` holds: the step's drain
    finds the moved ``_version``, the unit pass raises ``_UnitPassError`` and the canonical rerun
    judges the honest step (the CPU test of a serving error does the same for serving)."""
    setup = tiny(CHEAP, "fp32", ENGINE)
    clean, clean_rows, _ = run(setup, hostlib, overlap=True)
    held, wrote = [], []
    real_ops, real_blocks = DecoderReplay.operands, DecoderReplay.operand_blocks

    def touch(spy_calls):
        """Write into an operand of a step already measured, once."""
        if spy_calls and held and not wrote:
            held[0].add_(0.0)
            wrote.append(True)

    with pytest.MonkeyPatch.context() as cm:
        rec, spy = install_spy(cm, hostlib, ENGINE)

        def operands(self, m):
            touch(spy.calls)
            a, b = real_ops(self, m)
            if not spy.calls and not held:
                held.append(a)
            return a, b

        def blocks(self, ms):
            touch(spy.calls)
            out = real_blocks(self, ms)
            if not spy.calls and not held:
                held.append(out[0].a)
            return out

        cm.setattr(DecoderReplay, "operands", operands)
        cm.setattr(DecoderReplay, "operand_blocks", blocks)
        with mk.overlap(True):
            res = run_check5(setup)
    assert wrote, "the patched replay never wrote"
    assert any(not x.share for x in spy.calls), "the canonical rerun did not run"
    assert res.rejection is None and res.error is None
    assert res.stats == clean.stats


# ---- the prefetched challenges ----

@pytest.mark.parametrize("calls", [1, 3])
def test_prefetched_challenges_on_the_device_equal_challenge_matrix_bit_for_bit(calls):
    src = cs.HostChallengeSource()
    stream = torch.cuda.Stream(device=DEV)
    k = 5
    for rep in range(calls):  # successive calls reuse the pool's two slots
        ms = [9 + rep, 3, 7, 3, 4]
        widths = [40, 6, 65, 6, 6]
        pf = src.prefetch(H, ms, k, widths)
        buf, offs, ev = src.draw(H, ms, k, widths, stream=stream, device=DEV, prefetched=pf)
        assert buf.device == DEV and ev is not None
        ev.synchronize()
        for m, w, o in zip(ms, widths, offs):
            got = buf[o // 4:o // 4 + w * k].view(w, k).cpu()
            assert torch.equal(got.view(torch.int32),
                               challenge_matrix(H, m, k, w).view(torch.int32)), (rep, m, w)


def test_a_prefetched_step_gives_the_rows_of_an_inline_draw(hostlib, phased):  # noqa: F811
    group = _group(2)
    pf = mk.prefetch(H, [3], 3, [group.b.shape[2]])
    pre = freivalds.measure_groups_rows([group], h=H, k=3, prefetched=pf)
    inline = freivalds.measure_groups_rows([group], h=H, k=3)
    freivalds.drain()
    assert np.array_equal(bits(pre.cpu().numpy()), bits(inline.cpu().numpy()))


# ---- the pinned read ----

def test_the_pinned_read_equals_the_plain_read():
    groups = [_group(s, dtype=dt) for s, dt in enumerate((torch.float32, torch.bfloat16,
                                                          torch.float32))]
    rows = [freivalds.measure_groups_rows([g], h=H, k=3) for g in groups]
    plain = checks._read_rows(rows)
    pinned = checks._read_rows(rows, pinned=True)
    assert [len(x) for x in pinned] == [len(x) for x in plain]
    for a, b in zip(plain, pinned):
        assert a.dtype == b.dtype == np.float64 and np.array_equal(bits(a), bits(b))
    with mk.overlap(False):  # no measure stream: the plain read
        off = checks._read_rows(rows, pinned=True)
    assert all(np.array_equal(bits(a), bits(b)) for a, b in zip(plain, off))
