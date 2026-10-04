import struct

import pytest
import torch

from setup.records import TAG_RECORD, Record, RecordError, decode_record, encode_record


def i32(xs):
    return torch.tensor(xs, dtype=torch.int32)


def test_tag():
    assert TAG_RECORD == 0x01


def test_record_round_trip_and_layout():
    rec = Record(ids=i32([5, 6, 7]), targets=i32([6, 7, 2]), mask=i32([0, 1, 1]))
    b = encode_record(rec)
    assert b[:6] == bytes([0x01, 0x10]) + struct.pack(">I", 3)
    assert b[6:] == struct.pack("<9i", 5, 6, 7, 6, 7, 2, 0, 1, 1)
    assert decode_record(b) == rec
    assert encode_record(decode_record(b)) == b


def test_record_validation():
    with pytest.raises(RecordError):
        Record(ids=i32([1, 2]), targets=i32([1]), mask=i32([1, 1]))
    with pytest.raises(RecordError):
        Record(ids=torch.tensor([1, 2]), targets=i32([1, 2]), mask=i32([1, 1]))  # int64
    with pytest.raises(RecordError):
        Record(ids=i32([1, 2]), targets=i32([1, 2]), mask=i32([1, 2]))
    with pytest.raises(RecordError):
        Record(ids=i32([]), targets=i32([]), mask=i32([]))


def test_decode_record_rejects_noncanonical():
    b = encode_record(Record(ids=i32([5]), targets=i32([6]), mask=i32([1])))
    for bad in (b[:-1], b + b"\x00", b"\x02" + b[1:], b[:1] + b"\x01" + b[2:]):
        with pytest.raises(RecordError):
            decode_record(bad)
