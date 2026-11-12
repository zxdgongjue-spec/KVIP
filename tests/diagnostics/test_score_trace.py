# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
import torch
from types import SimpleNamespace

from kvip.diagnostics.score_trace import read_score_keys, validate_scores
from score_archive_fixture import score_archive


@pytest.mark.parametrize(
    "fault,match", [("score", "ranking"), ("threshold", "threshold log"), ("nan", "NaN"), ("ordinal", "ordinals")]
)
def test_exact_causal_score_and_threshold_validation(fault, match):
    trace = score_archive(real=True)
    validate_scores(trace)
    if fault == "score":
        trace.arrays["post.score_keys"][0, 0, 7000] = 0xC100
    elif fault == "threshold":
        trace.arrays["post.threshold_records"][0, 0, 0, 2] += 1
    elif fault == "nan":
        trace.arrays["post.score_keys"][0, 0, 7000] = 0xFFC1
    else:
        trace.arrays["post.prefetch_ordinals"][0] = 7
    with pytest.raises(ValueError, match=match):
        validate_scores(trace)


def test_streaming_readback_preserves_all_score_bits_and_rejects_any_causal_difference():
    from kvip.diagnostics.replay_measure import verify_score_keys

    batch, history = 16, 256
    expected = np.arange(batch * 4 * history, dtype=np.uint16).reshape(batch, 4, history)
    workspace = torch.from_numpy(expected.view(np.uint8).reshape(-1).copy())
    args = [None] * 14
    args[5] = torch.full((batch,), 4, dtype=torch.int32)
    args[6] = torch.full((batch,), history, dtype=torch.int32)
    args[12] = workspace
    rotating = SimpleNamespace(
        runtime_shape=SimpleNamespace(batch=batch, history=history, splitQ=1, liCores=32), arguments=args
    )
    actual = read_score_keys(torch, rotating)
    np.testing.assert_array_equal(actual, expected)
    ends = np.full((batch, 4), 200, dtype=np.int32)
    actual[:, :, 220:] = 0  # Padding/future bytes are deliberately undefined.
    result = verify_score_keys(actual, expected, ends)
    assert result["compared_scores"] == 16 * 4 * 200
    actual[15, 3, 199] ^= np.uint16(1)  # One bit, last causal score: never tolerate.
    with pytest.raises(AssertionError, match="score keys differ"):
        verify_score_keys(actual, expected, ends)
    rotating.runtime_shape.liCores = 16
    with pytest.raises(ValueError, match="reuses a core"):
        read_score_keys(torch, rotating)
