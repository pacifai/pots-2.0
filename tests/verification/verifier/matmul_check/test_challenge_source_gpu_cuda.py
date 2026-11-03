"""The GPU challenge source on a GPU (FS41c, ``MEASURE_KERNEL_PLAN_DRAFT.md`` §2.9). Skipped
without CUDA; they run on the pod (FS41d). The host source is the reference: every draw is
compared byte for byte (as int32 views, so a NaN-valued float compares by its bits) and at the
same offsets. Check 5's rows and one tiny decoder step are compared under either default
source."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from verification.verifier.matmul_check import challenge_source as cs  # noqa: E402
from verification.verifier.matmul_check import freivalds  # noqa: E402
from verification.verifier.matmul_check import measure_kernel as mk  # noqa: E402
from verification.verifier.matmul_check.freivalds import MeasureGroup  # noqa: E402

from tests.verification.verifier.matmul_check.test_measure_core import H, bits, make  # noqa: E402
from tests.verification.verifier.test_check_5_step_table import CHEAP, tiny  # noqa: E402
from tests.verification.verifier.test_unit_schedule import assert_same, run_check5  # noqa: E402

DEV = torch.device("cuda", 0)
WIDTHS = [1, 15, 16, 17, 33, 128, 896, 4864, 151936]
SMALL_WIDTHS = [1, 15, 16, 17, 33, 128]


@pytest.fixture
def restore_default_source():
    """Puts the default source back after a test that sets it."""
    before = mk.default_source()
    try:
        yield
    finally:
        mk.set_default_source(before)


def assert_draws_equal(src, h, ms, k, widths, stream=None):
    """``src.draw`` on ``stream`` (default: the current stream) equals the host source's draw:
    equal offsets and equal bytes, read on the host after the event."""
    stream = torch.cuda.current_stream(DEV) if stream is None else stream
    buf, offs, event = src.draw(h, ms, k, widths, stream=stream, device=DEV)
    want, want_offs, _ = cs.HostChallengeSource().draw(h, ms, k, widths, stream=None)
    assert buf.device == DEV and buf.dtype == torch.float32 and event is not None
    event.synchronize()
    assert np.array_equal(offs, want_offs)
    assert buf.numel() == want.numel()
    assert torch.equal(buf.cpu().view(torch.int32), want.view(torch.int32))


def random_request(rng, widths, n):
    ms = [int(m) for m in rng.integers(1, 2 ** 32, n)]
    ms[-1] = ms[0]  # a repeated m
    ms[1] = 2 ** 32 - 1
    return ms, [int(w) for w in rng.choice(widths, n)]


@pytest.mark.parametrize("k", [1, 9, 21])
@pytest.mark.parametrize("seed", range(3))
def test_draw_equals_the_host_sources_on_random_requests(k, seed):
    rng = np.random.default_rng(100 * k + seed)
    ms, widths = random_request(rng, WIDTHS, 6)
    widths[seed] = 151936  # every seed draws the widest vocabulary width once
    assert_draws_equal(cs.GpuChallengeSource(), bytes(rng.integers(0, 256, 32, dtype=np.uint8)),
                       ms, k, widths)


@pytest.mark.parametrize("seed", range(2))
def test_draw_equals_the_host_sources_at_k_255_on_small_widths(seed):
    rng = np.random.default_rng(7 + seed)
    ms, widths = random_request(rng, SMALL_WIDTHS, 8)
    assert_draws_equal(cs.GpuChallengeSource(), bytes(rng.integers(0, 256, 32, dtype=np.uint8)),
                       ms, 255, widths)


def test_draw_equals_the_host_sources_on_about_30000_attention_like_members():
    n = 30000
    ms = list(range(1, n + 1))
    widths = [128 if i % 3 else 64 for i in range(n)]
    assert_draws_equal(cs.GpuChallengeSource(), H, ms, 9, widths)


def test_a_repeated_request_with_another_h_hits_the_cache_and_is_still_equal():
    src = cs.GpuChallengeSource()
    ms, widths = (3, 5, 3, 9), (17, 128, 33, 896)  # tuples: the cache's identity hit
    for h in (H, bytes(32), b"\xff" * 32):
        assert_draws_equal(src, h, ms, 9, widths)
    assert_draws_equal(src, H, list(ms), 9, list(widths))  # equal lists: the equality hit


def test_a_draw_on_a_side_stream_is_read_on_the_default_stream_after_its_event():
    src = cs.GpuChallengeSource()
    ms, widths, k = [4, 8, 4], [33, 896, 128], 21
    side = torch.cuda.Stream(device=DEV)
    buf, offs, event = src.draw(H, ms, k, widths, stream=side, device=DEV)
    torch.cuda.current_stream(DEV).wait_event(event)
    got = buf.clone()  # on the default stream, ordered after the draw by the event
    torch.cuda.current_stream(DEV).synchronize()
    want, want_offs, _ = cs.HostChallengeSource().draw(H, ms, k, widths, stream=None)
    assert np.array_equal(offs, want_offs)
    assert torch.equal(got.cpu().view(torch.int32), want.view(torch.int32))


def test_xof_kernels_load_and_a_second_call_returns_the_same_object():
    first = cs.xof_kernels(DEV)
    assert first.fn is not None
    assert cs.xof_kernels(DEV) is first


def groups(k):
    """Three small groups of other shapes and dtypes, labels 1, 2 and 3."""
    out = []
    for seed, dtype in enumerate((torch.float32, torch.bfloat16, torch.float32)):
        mi = make(24 + 8 * seed, 40, 33 + seed, k, dtype=dtype, biased=False, guard=True,
                  seed=seed)
        out.append(MeasureGroup(mi.a.to(DEV)[None], mi.b.to(DEV)[None], mi.p.to(DEV)[None],
                                [1 + seed]))
    return out


@pytest.mark.parametrize("k", [3, 9])
def test_check_5_rows_are_equal_bit_for_bit_under_either_source(k, restore_default_source):
    rows = {}
    for name, src in (("host", cs.HostChallengeSource()), ("gpu", cs.GpuChallengeSource())):
        mk.set_default_source(src)
        got = freivalds.measure_groups_rows(groups(k), h=H, k=k)
        freivalds.drain()
        rows[name] = bits(got.cpu().numpy())
    assert np.array_equal(rows["host"], rows["gpu"])


def test_the_default_source_is_the_gpu_source():
    assert isinstance(mk.default_source(), cs.GpuChallengeSource)


def test_a_tiny_decoder_step_gives_the_same_verdict_and_numbers_under_either_source(
        restore_default_source):
    setup = tiny(CHEAP, "fp32", "cuda")
    results = {}
    for name, src in (("host", cs.HostChallengeSource()), ("gpu", cs.GpuChallengeSource())):
        mk.set_default_source(src)
        results[name] = run_check5(setup)
        assert results[name].rejection is None and results[name].error is None
    assert results["gpu"].stats  # the step has products
    assert_same(results["host"], results["gpu"])


class Comparing:
    """A source that draws on the GPU and on the host, asserts the two are equal byte for byte
    and returns the GPU's draw; ``seen`` counts the requests."""

    def __init__(self):
        self.gpu, self.host, self.seen = cs.GpuChallengeSource(), cs.HostChallengeSource(), 0

    def draw(self, h, ms, k, widths, *, stream, device=None, prefetched=None):
        buf, offs, event = self.gpu.draw(h, ms, k, widths, stream=stream, device=device)
        want, want_offs, _ = self.host.draw(h, ms, k, widths, stream=None)
        event.synchronize()
        assert np.array_equal(offs, want_offs)
        assert torch.equal(buf.cpu().view(torch.int32), want.view(torch.int32))
        self.seen += 1
        return buf, offs, event


def test_every_request_of_a_tiny_steps_check_5_is_equal_on_both_sources(restore_default_source):
    setup = tiny(CHEAP, "fp32", "cuda")
    src = Comparing()
    mk.set_default_source(src)
    res = run_check5(setup)
    assert res.rejection is None and res.error is None
    assert src.seen > 0
