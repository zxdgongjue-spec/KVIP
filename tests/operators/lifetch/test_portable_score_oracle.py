from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "support"))
import portable_score_oracle as oracle
from workloads import ReplayRound


def example(monkeypatch):
    history = 2054
    ends = np.tile(np.arange(history - 3, history + 1, dtype=np.int32), (16, 1))
    valid = np.arange(history)[None, None, :] < ends[..., None]
    keys = np.ones((16, 4, history), dtype=np.uint16)
    keys[:, :, :2046] = 7
    keys[:, :, -2] = 0
    keys[:, 1, :2] = 0xFF41
    # This row includes large unsigned score keys and signed BF16 bit patterns.
    data = SimpleNamespace(
        geometry={"batch": 16, "history": history, "resident": 2048},
        captured_hot_valid=np.ones((16, 2048), np.uint8),
        captured_lru=np.tile(np.arange(2048, dtype=np.int32), (16, 1)),
        tail_bounds=np.broadcast_to(np.array([2048, 2052], dtype=np.int32), (16, 2)).copy(),
        query_ends=ends,
        captured_layout=np.tile(np.arange(history, dtype=np.int64), (16, 1)),
        logical_score_keys=keys,
        logical_score_valid=valid,
    )
    data.validate_topk = lambda ids: ReplayRound.validate_topk(data, ids)
    base = np.broadcast_to(np.arange(2048, dtype=np.int32), (16, 4, 2048)).copy()
    monkeypatch.setitem(oracle.ARCHIVE, 10, (oracle.domain_digest(data), oracle.score_digest(keys, base)))
    monkeypatch.setitem(oracle.REQUEST_STATE, 10, oracle.request_state_digest(data))
    return data, np.broadcast_to(base, (20, *base.shape)).copy()


def test_equal_cutoff_exchanges_preserve_score_fingerprint(monkeypatch):
    data, outputs = example(monkeypatch)
    outputs[1:, 0, 0, -1] = 2049
    outputs[2] = outputs[2, ..., ::-1]
    data.logical_score_keys[~data.logical_score_valid] = 0xFFFF
    calls = []
    original = data.validate_topk

    def checked(value):
        original(value)
        calls.append(1)

    data.validate_topk = checked
    result = oracle.verify(10, data, outputs)
    assert len(calls) == 20
    assert (result["calls"], result["queries"], result["score_multiset_sha256"]) == (
        20,
        1280,
        oracle.ARCHIVE[10][1],
    )
    ids = result["topk_set_digests"]
    assert ids[0] != ids[1] and ids[1] == ids[2]


@pytest.mark.parametrize(
    "bad,match",
    [
        ("word", "score domain"),
        ("hot", "request state"),
        ("lru", "request state"),
        ("tail", "request state"),
        ("mask", "score domain"),
        ("layout", "score domain"),
        ("tag", "native logical"),
        ("negative", "native logical"),
        ("bounds", "native logical"),
        ("duplicate", "duplicates"),
        ("loser", "ranking"),
        ("shape", "20 measured"),
        ("score_proof", "winner score multisets"),
    ],
)
def test_reject_invalid_outputs_and_sources(monkeypatch, bad, match):
    data, outputs = example(monkeypatch)
    if bad == "word":
        data.logical_score_keys[0, 0, 12] = 8
    elif bad == "hot":
        data.captured_hot_valid[0, 0] = 0
    elif bad == "lru":
        data.captured_lru[0, :2] = [1, 0]
    elif bad == "tail":
        data.tail_bounds[0, 1] -= 1
    elif bad == "mask":
        data.logical_score_valid[0, 0, 8] = False
    elif bad == "layout":
        data.captured_layout[0, :2] = [1, 0]
    elif bad == "tag":
        outputs[0, 0, 0, 0] |= 0x40000000
    elif bad == "negative":
        outputs[0, 0, 0, 0] = -1
    elif bad == "bounds":
        outputs[0, 0, 0, -1] = data.query_ends[0, 0]
    elif bad == "duplicate":
        outputs[0, 0, 0, -1] = 2046
    elif bad == "loser":
        outputs[0, 0, 0, 0] = 2049
    elif bad == "shape":
        outputs = outputs[:1]
    elif bad == "score_proof":
        monkeypatch.setitem(oracle.ARCHIVE, 10, (oracle.ARCHIVE[10][0], "0" * 64))
    with pytest.raises((ValueError, AssertionError), match=match):
        oracle.verify(10, data, outputs)
