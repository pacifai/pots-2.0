import math
from pathlib import Path

import pytest
import torch

from verification.parameters import LAMBDA, LOG2_G, UNIT_ROUNDOFF, load_protocol_config
from verification.verifier.bands import TAU_W0
from verification.verifier.matmul_check.challenges import SIGMA_R
from verification.verifier.matmul_check.sizing import C_ANTI, F_TARGET, Z


def test_defaults():
    p = load_protocol_config({})
    assert p.k == 9
    assert p.band_file == Path("trainer_output/verification/bands.json")


def test_overrides_are_parsed():
    p = load_protocol_config({"VERIF_K": "9", "VERIF_OUTPUT_DIR": "/tmp/x"})
    assert p.k == 9
    assert p.band_file == Path("/tmp/x/bands.json")


def test_frozen():
    p = load_protocol_config({})
    with pytest.raises(AttributeError):
        p.k = 3  # type: ignore[misc]


@pytest.mark.parametrize("value", ["seven", "0", "-1"])
def test_invalid_k_raises(value):
    with pytest.raises(ValueError, match="VERIF_K"):
        load_protocol_config({"VERIF_K": value})


def test_constants_match_claude_md_table():
    assert (LAMBDA, LOG2_G, Z, F_TARGET, C_ANTI, TAU_W0) == (25, 52, 8, 1.0, math.sqrt(2 / 3), 4.0)
    assert math.isclose(SIGMA_R, 1 / math.sqrt(3))
    assert UNIT_ROUNDOFF == {torch.float32: 2**-24, torch.bfloat16: 2**-8, torch.float16: 2**-11}
