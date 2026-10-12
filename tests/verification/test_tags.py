"""The one-byte tags that open hashed bytes are distinct across the modules that own them.

Records (`setup.records`), tensor leaves (`commitment.encoding`) and challenge labels
(`matmul_check.challenges`) each pin their own tag; a shared value would let two kinds of
object encode to the same bytes.

The unkeyed BLAKE3 inputs must also open with prefix-free domains: no domain's opening bytes
start another's, so no input of one domain can also parse as an input of another.
"""

import itertools

import torch

from setup import data
from setup.records import INT32_CODE, TAG_RECORD
from verification.commitment import merkle
from verification.commitment.encoding import DTYPE_CODES, TENSOR_TAGS
from verification.verifier.matmul_check.challenges import TAG_LABEL


def test_tags_are_distinct():
    tags = [TAG_RECORD, *TENSOR_TAGS, TAG_LABEL]
    assert len(set(tags)) == len(tags)


def test_record_int32_code_is_the_tensor_leaf_code():
    assert INT32_CODE == DTYPE_CODES[torch.int32]


def test_hash_domains_are_prefix_free():
    tags = {data.TAG_SCHEDULE, data.TAG_POSITION, data.TAG_POISON}
    assert all(t.isascii() and t.endswith(b"/") for t in tags)  # F8c, F8g
    domains = [
        merkle._LEAF_PREFIX, merkle._NODE_PREFIX,  # the Merkle leaf and node hashes
        bytes([TAG_LABEL]),  # a challenge label, the input of a keyed hash
        data._MANIFEST_MAGIC,  # the manifest hash
        data._RECORDS_MAGIC,  # the records file
        b"{",  # JSON: the band file's and the config's hashes
        *tags,
    ]
    for a, b in itertools.permutations(domains, 2):
        assert not b.startswith(a), (a, b)
