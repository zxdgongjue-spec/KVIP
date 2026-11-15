# SPDX-License-Identifier: Apache-2.0
"""Lazy mapping must retain honest per-invocation traffic accounting."""

import numpy as np
import pytest

from kvip.diagnostics.snapshot import decode_inverse_mapping
from kvip.diagnostics.pipeline_options import admission_for, options_for, q_workers_for, validate_options


def test_stale_entries_are_not_counted_and_abc_is_restored():
    resident, history = 16384, 17024
    raw = np.full((2, history), resident, dtype=np.int32)
    sources = np.full((2, 10240), -1, dtype=np.int32)
    sources[0, :3] = [5, 120, 16500]
    raw[0, sources[0, :3]] = resident + np.arange(3)
    # entries beyond the prefix are final-Fetch events and must not count as
    # prefetch; stale map rows deliberately look like valid virtual slots.
    sources[0, 3] = 6
    bounds = np.array([[16800, 17000], [16384, 17024]], dtype=np.int32)
    actual = decode_inverse_mapping(raw, sources, [3, 0], bounds, resident, resident + 512)
    expected = np.full_like(raw, -1)
    expected[0, sources[0, :3]] = resident + np.arange(3)
    for b, (lo, hi) in enumerate(bounds):
        ids = np.arange(lo, min(hi, lo + 512, history))
        expected[b, ids] = resident + 8192 + ids % 512
    np.testing.assert_array_equal(actual, expected)
    assert np.count_nonzero((actual >= resident) & (actual < resident + 2048)) == 3


@pytest.mark.parametrize("bad", ["forward", "duplicate", "overlap", "count"])
def test_corrupt_current_records_are_rejected(bad):
    raw = np.full((1, 2048), -1, dtype=np.int32)
    sources = np.full((1, 2048), -1, dtype=np.int32)
    sources[0, :2] = [100, 200]
    raw[0, [100, 200]] = [128, 129]
    counts, bounds = [2], None
    if bad == "forward":
        raw[0, 100] = 129
    elif bad == "duplicate":
        sources[0, 1] = 100
    elif bad == "overlap":
        bounds = [[100, 220]]
    else:
        counts = [2049]
    with pytest.raises((AssertionError, ValueError)):
        decode_inverse_mapping(raw, sources, counts, bounds, 128, 512)


def test_inverse_options_preserve_control_budgets_and_disable_early_drain():
    for name, reference in (("inverse-map", "ready-batch4"),
                            ("inverse-map-skip", "saturated-skip"),
                            ("capacity-inverse-map", "capacity")):
        options = options_for(name)
        assert options == options_for(reference) | {"inverse_mapping": True}
        assert admission_for(name) == admission_for(reference)
        assert q_workers_for(name) == 4
        assert not options.get("early_prefetch_drain")
        assert validate_options(options) == options
