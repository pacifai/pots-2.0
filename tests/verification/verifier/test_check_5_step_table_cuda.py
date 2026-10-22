"""FS38b's check 5 on CUDA, through the real kernel (``cuda`` engine of ``test_kernel_step``):
the unit path measures each schedule step as one shared-read table (``share=True``,
``pots_measure_level``), the canonical path each group alone (``share=False``). Skipped without
CUDA; collects anywhere.

Compared as on the Mac (``test_check_5_step_table``): every row bit for bit, every check-5
number, the rejection and the record status. Also the shares found per step on the tiny decoder
(printed with ``-s``) and the reads saved on Llama-3.2-1B's shapes with two layers, at bf16 and
at fp32.
"""

import pytest
import torch


from tests.verification.decoder_helpers import SWITCHES, switch_id
from tests.verification.verifier.matmul_check.test_measure_core import (  # noqa: F401
    bits, host_measure, hostlib,
)
from tests.verification.verifier.matmul_check.test_measure_kernel_host import phased  # noqa: F401
from tests.verification.verifier.test_check_5_step_table import (
    CHEAP, FLIPS, OTHER, _show, _target, llama_setup, pair, reads_saved, same, share_groups,
    tiny,
)
from tests.verification.verifier.test_kernel_step import DTYPES

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

ENGINE = "cuda"


@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_honest_step_rows_and_numbers_equal(switch, dtype, hostlib, phased):  # noqa: F811
    unit = same(tiny(switch, dtype, ENGINE), hostlib, engine=ENGINE)
    assert unit.rejection is None and unit.error is None


@pytest.mark.parametrize("where", FLIPS, ids=lambda w: "-".join(map(str, w)))
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_a_flip_in_each_class_is_rejected_at_check_5_naming_its_product(dtype, where, hostlib,
                                                                       phased):  # noqa: F811
    setup = tiny(CHEAP, dtype, ENGINE)
    unit = same(setup, hostlib, _target(setup.c, where), engine=ENGINE)
    assert unit.rejection is not None


@pytest.mark.parametrize("where", [("layer", 1, "Y_q"), ("layer", 2, "dX_down"),
                                   ("layer", 2, "dV"), ("head", "G", -1), ("head", "Lambda", 0)],
                         ids=lambda w: "-".join(map(str, w)))
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("switch", [("llama", False), ("qwen2", True), OTHER], ids=switch_id)
def test_a_flip_on_the_other_switches_is_rejected_naming_its_product(switch, dtype, where,
                                                                    hostlib,
                                                                    phased):  # noqa: F811
    setup = tiny(switch, dtype, ENGINE)
    same(setup, hostlib, _target(setup.c, where), engine=ENGINE)


@pytest.mark.parametrize("dtype", list(DTYPES))
def test_one_shared_call_per_schedule_step(dtype, hostlib, phased):  # noqa: F811
    setup = tiny(CHEAP, dtype, ENGINE)
    (unit, _, uspy), (_, _, cspy) = pair(setup, hostlib, engine=ENGINE)
    sched = unit.ctx.state.replay.unit_schedule()
    assert [x.share for x in uspy.calls] == [True] * len(sched)
    assert [sorted(x.ms) for x in uspy.calls] == [sorted(step) for step in sched]
    assert cspy.calls and not any(x.share for x in cspy.calls)


@pytest.mark.parametrize("dtype", list(DTYPES))
def test_shares_found_per_step_are_printed(dtype, hostlib, phased):  # noqa: F811
    setup = tiny(CHEAP, dtype, ENGINE)
    (unit, _, spy), _ = pair(setup, hostlib, engine=ENGINE)
    sched = unit.ctx.state.replay.unit_schedule()
    tables = [(step, x, share_groups(setup.c, x)) for step, x in zip(sched, spy.calls)]
    _show(setup, tables, f"tiny step {dtype} on CUDA")
    assert not any(x.plan.cycle for _, x, _ in tables)
    if dtype == "fp32":  # the plan: X shared by Y_x and G_x at fp32 (the Mac test lists them)
        for layer in (1, 2):
            groups = [es for _, _, gs in tables for _, es in gs]
            for x in ("q", "k", "v", "o", "gate", "up", "down"):
                want = {(f"L{layer}.Y_{x}", 1), (f"L{layer}.G_{x}", 0)}
                assert any(want <= set(es) for es in groups), want


@pytest.mark.slow
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_llama_3_2_1b_shapes_two_layers_reads_saved_and_rows_equal(dtype, hostlib,
                                                                   phased):  # noqa: F811
    setup = llama_setup(dtype, "cuda")
    (unit, urows, uspy), (canon, crows, _) = pair(setup, hostlib, engine=ENGINE)
    assert unit.rejection is None and unit.error is None
    assert urows == crows, "the unit path's rows differ from the canonical path's"

    assert sorted(urows) == list(range(1, setup.c.M + 1))
    sched = unit.ctx.state.replay.unit_schedule()
    plans = [(step, x.plan) for step, x in zip(sched, uspy.calls)]
    assert not any(p.cycle for _, p in plans)
    reads_saved(setup, plans, f"{dtype} on CUDA")
