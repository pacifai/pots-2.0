"""The fused measuring kernel on a GPU. Skipped without CUDA; they run on a rented card.

The host decomposition test (``test_measure_kernel_host.py``) holds the shared phase code to the
host build; these hold the compiled kernel to the same numbers."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from tests.verification.verifier.matmul_check.test_measure_core import (  # noqa: E402
    H, SHAPES, bits, check_within_bound, host_measure, hostlib, make,  # noqa: F401
)
from tests.verification.verifier.matmul_check.test_measure_core_spec import (  # noqa: E402
    fs36_cases,
    same_bits,
)
from verification.commitment import gpu_blake3 as gb  # noqa: E402
from verification.computation.interface import ReplayError  # noqa: E402
from verification.verifier.matmul_check import freivalds as fv  # noqa: E402
from verification.verifier.matmul_check import measure_kernel as mk  # noqa: E402
from verification.verifier.matmul_check.challenges import challenge_matrix  # noqa: E402
from verification.verifier.matmul_check.freivalds import MeasureGroup  # noqa: E402
from verification.verifier.matmul_check.measure_table import MeasureInput, pack  # noqa: E402

DEV = "cuda"


def group_of(mi, m=1, device=DEV):
    return MeasureGroup(mi.a.to(device)[None], mi.b.to(device)[None], mi.p.to(device)[None], [m],
                        None if mi.bias is None else mi.bias.to(device))


def kernel_rows(groups, k, guard=True):
    rows = fv.measure_groups_rows(groups, h=H, k=k, guard=guard)
    fv.drain()
    return rows.cpu().numpy()


def host_rows(mis, host_measure):
    return host_measure(pack(mis))


def with_challenge(mi, m, k):
    r = challenge_matrix(H, m, k, mi.b.shape[1]).contiguous()
    return MeasureInput(mi.a, mi.b, mi.p, r, mi.bias, mi.guard)


@pytest.mark.parametrize("k", [1, 3, 9])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("guard", [True, False])
def test_kernel_equals_the_host_build_bit_for_bit(host_measure, k, dtype, biased, guard):
    for s, (n, q, w) in enumerate(SHAPES):
        mi = make(n, q, w, k, dtype=dtype, biased=biased, guard=guard, seed=s)
        want = host_rows([with_challenge(mi, 1, k)], host_measure)
        got = kernel_rows([group_of(mi)], k, guard)
        assert np.array_equal(bits(got), bits(want)), (n, q, w)


def test_strided_and_transposed_views(host_measure):
    mi = make(6, 19, 37, 3, biased=True, seed=4)
    want = host_rows([with_challenge(mi, 1, 3)], host_measure)
    big_a = torch.zeros(6 + 2, 19 * 2 + 1)
    big_a[1:7, 0:38:2] = mi.a
    big_b = torch.zeros(19 + 3, 37 * 2)
    big_b[2:21, 0:74:2] = mi.b
    big_a, big_b = big_a.to(DEV), big_b.to(DEV)  # slice on the device: a view keeps its gaps
    views = MeasureGroup(big_a[1:7, 0:38:2][None], big_b[2:21, 0:74:2][None],
                         mi.p.t().contiguous().to(DEV).transpose(0, 1)[None], [1], mi.bias.to(DEV))
    assert not views.a[0].is_contiguous() and not views.b[0].is_contiguous()
    assert np.array_equal(bits(kernel_rows([views], 3)), bits(want))
    tr = MeasureGroup(mi.a.t().contiguous().to(DEV).transpose(0, 1)[None],
                      mi.b.t().contiguous().to(DEV).transpose(0, 1)[None], views.p, [1],
                      mi.bias.to(DEV))
    assert np.array_equal(bits(kernel_rows([tr], 3)), bits(want))


def test_zero_nan_and_inf_inputs(host_measure):
    cases = []
    for edit in (lambda m: m.a.__setitem__((0, 0), float("nan")),
                 lambda m: m.b.__setitem__((2, 5), float("nan")),
                 lambda m: m.p.__setitem__((1, 3), float("inf")),
                 lambda m: m.a.__setitem__((3, 1), float("-inf")),
                 lambda m: [t.zero_() for t in (m.a, m.b, m.p)],
                 lambda m: [m.a.fill_(2e19), m.b.fill_(1.0), m.p.fill_(2e19)]):
        for guard in (True, False):
            mi = make(4, 6, 40, 2, seed=7, guard=guard)
            mi = MeasureInput(mi.a.clone(), mi.b.clone(), mi.p.clone(), mi.r, mi.bias, guard)
            edit(mi)
            cases.append(mi)
    for mi in cases:
        want = host_rows([with_challenge(mi, 1, 2)], host_measure)
        got = kernel_rows([group_of(mi)], 2, mi.guard)
        # a GPU writes its canonical NaN where the host keeps the input's payload
        assert same_bits(got, want)


def test_kernel_equals_host_on_fs36_inputs(host_measure):
    """Every input FS36 checks bit for bit (the lengths, the order-sensitive cases, NaN, Inf,
    -0.0 and subnormals in A, B, P and the bias, the halving trees), each through the kernel
    and the host build. Subnormals are where a flush-to-zero setting would show."""
    cases = fs36_cases()
    assert len(cases) >= 200
    for label, mi in cases:
        k = mi.r.shape[1]
        mi = with_challenge(mi, 1, k)
        want = host_rows([mi], host_measure)
        got = kernel_rows([group_of(mi)], k, mi.guard)
        assert same_bits(got, want), (label, got, want)


def test_several_groups_of_mixed_shapes_equal_each_group_alone():
    groups = []
    for s, (n, q, w) in enumerate(SHAPES):
        a = torch.randn(3, n, q, generator=torch.Generator().manual_seed(s))
        b = torch.randn(3, q, w, generator=torch.Generator().manual_seed(s + 50))
        groups.append(MeasureGroup(a.to(DEV), b.to(DEV), (a @ b).to(DEV),
                                   [10 * s + 1, 10 * s + 2, 10 * s + 3]))
    one_call = kernel_rows(groups, 4)
    alone = np.concatenate([kernel_rows([g], 4) for g in groups])
    assert np.array_equal(bits(one_call), bits(alone))
    rev = kernel_rows(groups[::-1], 4)
    assert np.array_equal(bits(rev), bits(np.concatenate([kernel_rows([g], 4)
                                                          for g in groups[::-1]])))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("biased", [False, True])
def test_kernel_is_within_tolerance_of_torch_under_reference(dtype, biased):
    for s, (n, q, w) in enumerate(SHAPES):
        mi = make(n, q, w, 3, dtype=dtype, biased=biased, seed=s)
        got = kernel_rows([group_of(mi)], 3)[0]
        check_within_bound(got, with_challenge(mi, 1, 3), fp32_norms=False)
        with fv.reference():
            ref = fv.measure_product_row(mi.a.to(DEV).float(), mi.b.to(DEV).float(),
                                         mi.p.to(DEV).float(), h=H, m=1, k=3,
                                         bias=None if mi.bias is None else mi.bias.to(DEV).float())
        check_within_bound(ref.cpu().numpy(), with_challenge(mi, 1, 3), fp32_norms=True)


def test_measure_product_row_equals_measure_groups_rows_of_the_same_member():
    mi = make(9, 17, 65, 3, biased=True, seed=2)
    g = group_of(mi, m=5)
    row = fv.measure_product_row(g.a[0], g.b[0], g.p[0], h=H, m=5, k=3, bias=g.bias)
    fv.drain()
    want = kernel_rows([g], 3)[0]
    assert np.array_equal(bits(row.cpu().numpy()), bits(want))


def test_second_build_reads_the_cache_and_fmad_is_off(tmp_path):
    nv = gb._nvrtc()
    cc = torch.cuda.get_device_capability()
    first = mk.build(nv, cc, cache=tmp_path, use_cache=True)
    second = mk.build(nv, cc, cache=tmp_path, use_cache=True)
    assert not first.cached and second.cached and first.image == second.image
    assert "--fmad=false" in mk.OPTIONS
    assert any(p.suffix == ".cubin" for p in tmp_path.iterdir())


def test_drain_raises_on_an_input_mutated_after_enqueue():
    mi = make(9, 17, 65, 3, seed=1)
    g = group_of(mi)
    fv.measure_groups_rows([g], h=H, k=3)
    g.a.add_(1.0)
    with pytest.raises(ReplayError):
        fv.drain()
    fv.drain()  # the list is empty afterwards


def test_run_loop_builds_the_measuring_kernel_before_step_1_opens_a_section(monkeypatch):
    """On a tiny decoder on CUDA, ``run_loop`` calls ``measure_kernel.kernels(cuda:0)`` exactly
    once before the first section opens (the loop's pre-build, FS37, which sits before
    ``verifier.start_run``'s ``run:0`` section), so no NVRTC build lands in a timed step. Check
    5's own calls come later and are not counted."""
    from contextlib import contextmanager

    from tests.verification.verifier.test_kernel_step import SWITCHES, _setup
    from verification.runs import loop as loop_mod
    from verification.verifier.bands import Bands
    from verification.verifier.driver import Verifier

    c, D, tree, w0 = _setup("cuda", SWITCHES[0], "fp32")
    log = []
    real = mk.kernels

    def spy(device):
        log.append(f"kernels {torch.device(device)}")
        return real(device)

    monkeypatch.setattr(loop_mod.measure_kernel, "kernels", spy)

    @contextmanager
    def section(name):
        log.append(name)
        yield

    v = Verifier(c, h_D=tree.root, n_records=len(D), k=7, n_steps=1, bands=Bands.provisional(),
                 allow_provisional=True, w0=w0, seed=1)
    loop_mod.run_loop(c, c.build_model(), D, w0, v, final=w0, section=section)
    first_section = next(i for i, x in enumerate(log) if not x.startswith("kernels"))
    before = log[:first_section]
    assert before == [f"kernels {torch.device('cuda', 0)}"], log
    assert len(log) > first_section  # the sections follow
