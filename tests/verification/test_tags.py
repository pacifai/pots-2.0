"""The one-byte tags that open hashed bytes are distinct across the modules that own them.

Records (`setup.records`), tensor leaves (`commitment.encoding`) and challenge labels
(`matmul_check.challenges`) each pin their own tag; a shared value would let two kinds of
object encode to the same bytes.
"""

import torch

from setup.records import INT32_CODE, TAG_RECORD
from verification.commitment.encoding import DTYPE_CODES, TENSOR_TAGS
from verification.verifier.matmul_check.challenges import TAG_LABEL


def test_tags_are_distinct():
    tags = [TAG_RECORD, *TENSOR_TAGS, TAG_LABEL]
    assert len(set(tags)) == len(tags)


def test_record_int32_code_is_the_tensor_leaf_code():
    assert INT32_CODE == DTYPE_CODES[torch.int32]
