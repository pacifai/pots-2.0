"""Check 6's fast path against the reference formula, entry cases and decisions.

``checks._update_identity`` must record the same ``ρ_max`` (bit for bit) and return the same
rejection as ``checks._update_identity_reference``, the spec's formula on whole tensors, in
both judging and calibration mode.
"""

import math
import struct

import pytest
import torch

from verification.verifier import checks
from verification.verifier.context import StepContext

ETA = 1e-3
EPS = 2.0 ** -24
TAU = 4.0


def _ctx(judge, eps=EPS):
    return StepContext(step=1, indices=(), h_D=b"\0" * 32, n_records=1, prev_w_hashes=(),
                       chain_check_id="0", k=1, eps_in=eps, eps_acc=eps, eps_w=eps, judge=judge)


def _bits(x):
    return "nan" if math.isnan(x) else struct.pack("<d", x)


def _both(w_t, w_next, g, *, eta=ETA, tau=TAU, judge=True, scratch=None, eps=EPS):
    """Run the dispatching check and the reference; assert they agree; return the result."""
    fast_ctx, ref_ctx = _ctx(judge, eps), _ctx(judge, eps)
    got = checks._update_identity(fast_ctx, "6a", "w", w_t, w_next, g, eta, tau, scratch)
    want = checks._update_identity_reference(ref_ctx, "6a", "w", w_t, w_next, g, eta, tau)
    assert got == want
    (fs,), (rs,) = fast_ctx.stats.tensors, ref_ctx.stats.tensors
    assert (fs.weight, fs.check_id) == (rs.weight, rs.check_id)
    assert _bits(fs.rho_max) == _bits(rs.rho_max), (fs.rho_max, rs.rho_max)
    return got, fs.rho_max


def _honest(shape, seed=0, dtype=torch.float32, eta=ETA):
    """``W_t``, ``G`` and an SGD ``W_{t+1}`` via torch's fused ``add_``, as the prover's."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(shape, generator=gen).to(dtype)
    g = torch.randn(shape, generator=gen).to(dtype)
    w_next = w.clone()
    w_next.add_(g, alpha=-eta)
    return w, w_next, g


def _ulps(x, i, q):
    """``x`` with flat entry ``i`` moved up by ``q`` ulps."""
    x = x.clone()
    for _ in range(q):
        x.view(-1)[i] = torch.nextafter(x.view(-1)[i], torch.tensor(float("inf"), dtype=x.dtype))
    return x


@pytest.fixture(params=[1 << 21, 64], ids=["one-block", "many-blocks"])
def chunk(request, monkeypatch):
    """The real block size, and a tiny one so every tensor spans many blocks."""
    monkeypatch.setattr(checks, "UPDATE_CHUNK", request.param)
    return request.param


@pytest.mark.parametrize("shape", [(48, 40), (7, 3, 33), (300,), (), (0, 5)])
@pytest.mark.parametrize("judge", [True, False])
def test_honest_update_matches_reference(shape, judge, chunk):
    w, w_next, g = _honest(shape)
    rej, _ = _both(w, w_next, g, judge=judge)
    assert rej is None


def test_reference_rounding_gives_zero_rho(chunk):
    """``W_{t+1}`` rounded exactly as the reference rounds it: ``R ≡ 0``, ``ρ_max = 0``."""
    w, _, g = _honest((64, 32))
    rej, rho = _both(w, w - ETA * g, g)
    assert rej is None and rho == 0.0


def test_ties_in_fp32_quotient_keep_the_float64_max(chunk):
    """Many entries off by one ulp share fp32 quotients; the float64 max must still win."""
    w, _, g = _honest((40, 40), seed=3)
    w_next = w - ETA * g
    for i in range(0, w.numel(), 7):
        w_next = _ulps(w_next, i, 1)
    _, rho = _both(w, w_next, g)
    assert 1.0 < rho <= 2.0


@pytest.mark.parametrize("judge", [True, False])
def test_forged_entry_rejects_like_the_reference(judge, chunk):
    w, w_next, g = _honest((50, 30), seed=1)
    forged = _ulps(w_next, 777, 9)
    rej, rho = _both(w, forged, g, judge=judge)
    assert rho > TAU
    if judge:
        assert rej is not None and rej.check_id == "6a" and "entry 777" in rej.detail
    else:
        assert rej is None


def test_rho_exactly_at_tau_accepts_one_below_rejects(chunk):
    w, w_next, g = _honest((20, 20), seed=2)
    forged = _ulps(w_next, 5, 5)
    _, rho = _both(w, forged, g, tau=math.inf)
    assert _both(w, forged, g, tau=rho)[0] is None
    rej, _ = _both(w, forged, g, tau=math.nextafter(rho, 0.0))
    assert rej is not None and "entry 5" in rej.detail


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("judge", [True, False])
def test_non_finite_gradient_rejects_like_the_reference(bad, judge, chunk):
    w, w_next, g = _honest((30, 30), seed=4)
    g = g.clone()
    g.view(-1)[123] = bad
    rej, _ = _both(w, w_next, g, judge=judge)
    assert rej is not None and "non-finite" in rej.detail and "entry 123" in rej.detail


@pytest.mark.parametrize("judge", [True, False])
def test_non_finite_w_next_rejects_like_the_reference(judge, chunk):
    w, w_next, g = _honest((30, 30), seed=5)
    w_next = w_next.clone()
    w_next.view(-1)[17] = float("nan")
    rej, rho = _both(w, w_next, g, judge=judge)
    assert rej is not None and "entry 17" in rej.detail and math.isnan(rho)


@pytest.mark.parametrize("judge", [True, False])
def test_scale_overflow_with_finite_residual_rejects(judge, chunk):
    """``|W_t| + |η·G|`` overflows while ``W_t − η·G`` and ``R`` stay finite."""
    w, w_next, g = _honest((10, 10), seed=6, eta=1.0)
    w, w_next, g = w.clone(), w_next.clone(), g.clone()
    w.view(-1)[9] = g.view(-1)[9] = 3e38  # η = 1: W_t − η·G = 0, |W_t| + |η·G| = inf
    w_next.view(-1)[9] = 0.0
    rej, _ = _both(w, w_next, g, eta=1.0, judge=judge)
    assert rej is not None and "entry 9" in rej.detail


@pytest.mark.parametrize("judge", [True, False])
def test_zero_scale_entries(judge, chunk):
    """``W_t = G = 0``: ``R = 0`` there gives ``ρ = 0`` (0/0 in fp32); ``R ≠ 0`` gives ``inf``."""
    w, w_next, g = _honest((16, 16), seed=7)
    w, w_next, g = w.clone(), w_next.clone(), g.clone()
    for i in (0, 40, 255):
        w.view(-1)[i] = g.view(-1)[i] = w_next.view(-1)[i] = 0.0
    assert _both(w, w_next, g, judge=judge)[0] is None
    w_next.view(-1)[40] = 1e-30
    rej, rho = _both(w, w_next, g, judge=judge)
    assert rho == math.inf
    assert (rej is not None and "entry 40" in rej.detail) if judge else rej is None


def test_ratio_that_underflows_fp32_keeps_its_float64_value(chunk):
    """``max q = 0`` with ``R ≠ 0``: the fp32 quotient underflows, the float64 one doesn't.

    With η = 1, ``W_t = η·G = 1e38`` gives ``W_t − η·G = 0`` and a scale of ``ε·2e38``. A
    ``W_{t+1}`` of one subnormal, ``1.4e-45``, makes ``R/scale ≈ 1e-76``. Every other entry is
    ``W_t = G = W_{t+1} = 0``, a 0/0 in fp32 that the reference counts as ``ρ = 0``.
    """
    w, g, w_next = torch.zeros(4, 4), torch.zeros(4, 4), torch.zeros(4, 4)
    w.view(-1)[9] = g.view(-1)[9] = 1e38
    w_next.view(-1)[9] = 1.4e-45
    r, scale = w_next.view(-1)[9], EPS * (w.view(-1)[9] + g.view(-1)[9])
    assert float(r / scale) == 0.0  # the fp32 quotient underflows
    rej, rho = _both(w, w_next, g, eta=1.0)
    assert rej is None and 0.0 < rho < 1e-70


def test_bf16_weights_match_reference(chunk):
    w, w_next, g = _honest((24, 24), seed=8, dtype=torch.bfloat16)
    _both(w, w_next, g, tau=math.inf)
    _both(w, _ulps(w_next, 3, 3), g, eps=2.0 ** -8)


def test_mixed_dtypes_fall_back_to_reference(monkeypatch):
    w, w_next, g = _honest((8, 8), seed=9)
    calls = []
    ref = checks._update_identity_reference
    monkeypatch.setattr(checks, "_update_identity_reference",
                        lambda *a: calls.append(1) or ref(*a))
    assert checks._update_identity(_ctx(True), "6a", "w", w, w_next, g.double(), ETA, TAU) is None
    assert calls


def test_honest_update_never_calls_the_reference(monkeypatch):
    """The fast path decides an honest acceptance alone (its whole point)."""
    def boom(*a):
        raise AssertionError("reference called")
    monkeypatch.setattr(checks, "_update_identity_reference", boom)
    scratch = checks.UpdateScratch()
    for seed, shape in enumerate([(40, 30), (500,), (7, 9, 11)]):
        w, w_next, g = _honest(shape, seed=seed)
        assert checks._update_identity(_ctx(True), "6a", "w", w, w_next, g, ETA, TAU,
                                       scratch) is None


def test_scratch_reused_across_sizes_and_dtypes(chunk):
    scratch = checks.UpdateScratch()
    for seed, (shape, dtype) in enumerate([((64, 64), torch.float32), ((3, 5), torch.float32),
                                           ((100, 100), torch.float32),
                                           ((10, 10), torch.bfloat16)]):
        w, w_next, g = _honest(shape, seed=seed, dtype=dtype)
        _both(w, _ulps(w_next, 0, 2), g, scratch=scratch, tau=math.inf,
              eps=EPS if dtype == torch.float32 else 2.0 ** -8)
