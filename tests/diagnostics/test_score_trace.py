# SPDX-License-Identifier: Apache-2.0
"""Capture real byte layouts and validate scores; no device execution claimed."""

import copy
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from test_token_trace import fixture
from kvip.diagnostics.score_trace import (
    REAL_ARRAYS,
    _RETAINED_CAPTURES,
    _expected_thresholds,
    read_score_keys,
    sortable_to_bf16,
    threshold_capacity,
    validate_scores,
)
from kvip.diagnostics.token_trace import TokenIdTrace, TokenIdTraceRecorder, load_token_trace


def score_fixture(*, real=False, threshold_only=False):
    selector, metadata, complete = fixture(2)
    b, h, heads = 2, 8192, 32
    metadata.update(content="scores", capture_real_inputs=real)
    rotating = selector.rotating
    rotating.shape.rolling_threshold = False
    rotating.shape.threshold_only = threshold_only
    rotating.library = NS(v9_li_cores=lambda: 4, v9_threshold_trace_abi=lambda: 1)
    rotating.arguments = [None] * 14
    rotating.arguments[5] = torch.full((b,), 4, dtype=torch.int32)
    rotating.arguments[6] = torch.full((b,), h, dtype=torch.int32)
    rotating.arguments[12] = torch.empty(b * 4 * h * 2, dtype=torch.uint8)
    selector.storage.keys = torch.arange(128, dtype=torch.uint8).repeat(4, h, 1)
    selector.storage.scales = torch.ones((4, h), dtype=torch.float32)
    selector.inputs.query = torch.zeros((b * 4, heads, 128), dtype=torch.float8_e4m3fn)
    selector.inputs.weights = torch.ones((b * 4, heads), dtype=torch.bfloat16)
    selector.inputs.scale = torch.ones((b * 4, heads), dtype=torch.float32)

    def publish(recorder):
        complete()
        keys = np.full((b, 4, h), 0x8000, dtype=np.uint16)
        for row in range(b):
            for q in range(4):
                selected = selector.topk[row, q].numpy()
                keys[row, q, selected] = 0xBF80  # sortable BF16 1
                keys[row, q, selected[0]] = 0xC000  # strictly better BF16 2
        rotating.arguments[12].copy_(torch.from_numpy(keys.view(np.uint8).reshape(-1)))
        for row in range(b):
            for q in range(4):
                # A real native resident log, with only 2046 winners in hot R:
                # its Kth is +0. The final-derived Kth is +1.
                recorder._score_capture.buffer[row, q, :4] = torch.tensor(
                    [4096, 4096, 0x8000, 2 if threshold_only else 1]
                )
        rotating.rows[:, 16] = 2
        for e in range(rotating.rounds):
            rotating.rows[:, 80 + 4 * e : 84 + 4 * e] = 4096 + e * 512
        return keys

    return selector, metadata, publish


@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("threshold_only", [False, True])
def test_schema4_roundtrip_exact_score_bits_and_optional_real_inputs(tmp_path, real, threshold_only):
    selector, metadata, publish = score_fixture(real=real, threshold_only=threshold_only)
    recorder = TokenIdTraceRecorder(torch, tmp_path / "scores", metadata)
    recorder.before(selector)
    assert recorder._score_capture in _RETAINED_CAPTURES
    assert recorder._score_capture.buffer.data_ptr() % 128 == 0
    assert np.any(selector.rotating.context[64:72].numpy())
    if real:
        selector.storage.keys.fill_(99)  # after before: saved input must not alias.
    keys = publish(recorder)
    recorder.after(selector)
    assert recorder.completed
    assert recorder._score_capture not in _RETAINED_CAPTURES
    assert not torch.any(selector.rotating.context[64:72])
    archive = load_token_trace(recorder.directory)
    assert archive.manifest["schema_version"] == 4
    assert archive.manifest["validation"]["kind"] == "score_trace_integrity"
    assert archive.manifest["real_inputs_saved"] is real
    np.testing.assert_array_equal(archive.array("post.score_keys"), keys)
    assert np.all(archive.array("post.threshold_records")[:, :, -1, 2] == 0xBF80)
    assert np.all(archive.array("post.threshold_records")[:, :, -1, 3] == 4)
    if real:
        assert set(archive.arrays) & REAL_ARRAYS == REAL_ARRAYS
        np.testing.assert_array_equal(archive.array("real.keys")[0, 0], np.arange(128, dtype=np.uint8))
        assert np.all(archive.array("real.weights") == 0x3F80)
    else:
        assert not set(archive.arrays) & REAL_ARRAYS


@pytest.mark.parametrize(
    "fault,match",
    [
        ("score", "ranking"),
        ("threshold", "threshold log"),
        ("window", "Window posted"),
        ("nan", "NaN"),
        ("ordinal", "ordinals"),
    ],
)
def test_score_faults_never_publish_and_disarm_on_known_completion(tmp_path, fault, match):
    selector, metadata, publish = score_fixture()
    recorder = TokenIdTraceRecorder(torch, tmp_path / "scores", metadata)
    recorder.before(selector)
    keys = publish(recorder)
    if fault == "score":
        keys[0, 0, 5002] = 0xC080  # unselected item exceeds the selected threshold
        selector.rotating.arguments[12].copy_(torch.from_numpy(keys.view(np.uint8).reshape(-1)))
    elif fault == "threshold":
        recorder._score_capture.buffer[0, 0, 2] += 1
    elif fault == "window":
        selector.rotating.rows[0, 16] += 1
    elif fault == "nan":
        selector.rotating.arguments[12][:2].zero_()  # key0 decodes to NaN, not +0
    else:
        selector.rotating.mapping[0, 5001] = 4096
    with pytest.raises(ValueError, match=match):
        recorder.after(selector)
    assert not recorder.completed and not (recorder.directory / "manifest.json").exists()
    assert recorder._score_capture not in _RETAINED_CAPTURES
    assert not torch.any(selector.rotating.context[64:72])


def test_unknown_completion_retains_native_log_owner(tmp_path):
    selector, metadata, publish = score_fixture()
    recorder = TokenIdTraceRecorder(torch, tmp_path / "scores", metadata)
    recorder.before(selector)
    publish(recorder)
    old = selector.rotating.lease.check_stream

    def fail():
        raise RuntimeError("unknown completion")

    selector.rotating.lease.check_stream = lambda: NS(synchronize=fail)
    with pytest.raises(RuntimeError, match="unknown completion"):
        recorder.after(selector)
    assert recorder._score_capture in _RETAINED_CAPTURES
    assert recorder._score_capture.buffer is not None
    assert not (recorder.directory / "manifest.json").exists()
    # CPU fixture cleanup only after restoring an explicitly successful stream.
    selector.rotating.lease.check_stream = old
    recorder._score_capture.disarm()


def test_old_binary_and_workspace_reuse_are_rejected(tmp_path):
    selector, metadata, _ = score_fixture()
    del selector.rotating.library.v9_threshold_trace_abi
    with pytest.raises(ValueError, match="lacks score-trace ABI"):
        TokenIdTraceRecorder(torch, tmp_path / "old", metadata).before(selector)
    selector.rotating.library.v9_li_cores = lambda: 2
    with pytest.raises(ValueError, match="one complete"):
        read_score_keys(torch, selector.rotating)


def test_read_score_keys_supports_runtime_urma_geometry_and_checks_lengths():
    selector, _, _ = score_fixture()
    rotating = selector.rotating
    rotating.runtime_shape = NS(batch=2, history=8192, splitQ=1, liCores=4)
    values = np.arange(2 * 4 * 8192, dtype=np.uint16).reshape(2, 4, 8192)
    rotating.arguments[12].copy_(torch.from_numpy(values.view(np.uint8).reshape(-1)))
    np.testing.assert_array_equal(read_score_keys(torch, rotating), values)
    rotating.arguments[6][0] -= 128
    with pytest.raises(ValueError, match="fixed full-history"):
        read_score_keys(torch, rotating)


def test_unsigned_sortable_keys_decode_negative_zero_positive():
    np.testing.assert_array_equal(
        sortable_to_bf16(np.array([0x407F, 0x7FFF, 0x8000, 0xBF80, 0xC000], np.uint16)),
        np.array([0xBF80, 0x8000, 0, 0x3F80, 0x4000], np.uint16),
    )


def test_rolling_log_causal_last_partial_and_unused_slots():
    b, h, r = 1, 33024, 16384
    g = dict(
        batch=b, history=h, resident=r, heads=32, rounds=4, split_q=True, rolling_threshold=True, threshold_only=False
    )
    ends = np.array([[33021, 33022, 33023, 33024]], dtype=np.int32)
    scores = np.full((b, 4, h), 0x8000, dtype=np.uint16)
    selected = np.tile(np.arange(30000, 32048, dtype=np.int32), (1, 4, 1))
    scores[:, :, 30000:32048] = 0xBF80
    records = np.zeros((b, 4, threshold_capacity(h), 4), dtype=np.int32)
    for q in range(4):
        for slot, valid, applies, kind in _expected_thresholds(g, int(ends[0, q])):
            threshold = np.partition(scores[0, q, :valid], valid - 2048)[valid - 2048]
            records[0, q, slot] = [valid, applies, threshold, kind]
        records[0, q, -1] = [ends[0, q], h, 0xBF80, 4]
    arrays = {
        "post.score_keys": scores,
        "post.threshold_records": records,
        "pre.query_ends": ends,
        "post.topk": selected,
        "post.window_posted": np.zeros((1, 4), np.int32),
        "post.window_processed_ends": np.zeros((1, 4, 4), np.int32),
        "post.prefetch_ordinals": np.zeros(0, np.int32),
        "post.prefetch_records": np.zeros((0, 3), np.int32),
        "post.read_counts": np.zeros((1, 3), np.int64),
    }
    archive = TokenIdTrace(".", dict(geometry=g, metadata={}, real_inputs_saved=False), arrays)
    validate_scores(archive)
    assert records[0, 0, 2, 0] == 33021 and records[0, 0, 2, 1] == 33024
    archive.arrays = copy.deepcopy(arrays)
    archive.arrays["post.threshold_records"][0, 0, 3, 0] = 1
    with pytest.raises(ValueError, match="threshold log"):
        validate_scores(archive)


@pytest.mark.parametrize("resident,end,kind", [(16384, 32768, 2), (32768, 65536, 1), (16384, 1024, 1)])
def test_threshold_only_log_kind_matches_native_fallback(resident, end, kind):
    modes = dict(resident=resident, history=65536, threshold_only=True, rolling_threshold=False)
    assert _expected_thresholds(modes, end)[0][3] == kind
