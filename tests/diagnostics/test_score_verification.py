# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from kvip.diagnostics.replay_measure import verify_score_keys


def test_real_score_verification_checks_bits_but_ignores_causal_padding():
    expected = np.arange(32, dtype=np.uint16).reshape(2, 4, 4)
    actual = expected.copy()
    actual[:, :, 3] = 0
    result = verify_score_keys(actual, expected, np.full((2, 4), 3))
    assert result == dict(status="bit_exact", compared_scores=24, included_in_timing=False)
    actual[1, 2, 1] ^= 1
    with pytest.raises(AssertionError, match="row=1 query=2 positions=1"):
        verify_score_keys(actual, expected, np.full((2, 4), 3))


def test_real_score_verification_rejects_wrong_layout_and_invalid_lengths():
    scores = np.zeros((1, 4, 5), dtype=np.uint16)
    with pytest.raises(AssertionError, match="layout"):
        verify_score_keys(scores.astype(np.int32), scores, np.zeros((1, 4)))
    with pytest.raises(AssertionError, match="exceeds"):
        verify_score_keys(scores, scores, np.full((1, 4), 6))
