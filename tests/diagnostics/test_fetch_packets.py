# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from kvip.diagnostics.fetch_packets import packetize, unpack


@pytest.mark.parametrize('density', [0, 1, 7, 16, 17, 64, 128])
def test_packets_and_overflow_preserve_exact_candidates(density):
    rng = np.random.default_rng(808+density)
    values = rng.integers(1, 30, (32, 128), dtype=np.uint16)
    for row in values:
        row[rng.permutation(128)[:density]] += 100
    packets = packetize(values, 50)
    for cutoff in (50, 110, 128):
        scores, ids = unpack(packets, values, cutoff)
        want = np.flatnonzero(values.reshape(-1) > cutoff)
        np.testing.assert_array_equal(ids, want)
        np.testing.assert_array_equal(scores, values.reshape(-1)[want])
