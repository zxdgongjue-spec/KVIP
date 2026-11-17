# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from kvip.diagnostics.post_index_profile import decode_post_index
from kvip.diagnostics.native_profile import decode_profile


def data(diagnostic=False):
    raw = np.zeros((64, 16), dtype=np.int64)
    raw[:, 1] = np.arange(64)
    raw[:, 15] = raw[:, 1]
    if diagnostic:
        raw[:, [10, 4, 5, 6, 7, 8, 9]] = [100, 120, 125, 130, 150, 180, 210]
        raw[1, 9] = 220
    return raw


def test_headline_status_checks_all_cores_and_never_interprets_high_stale_state():
    raw = data()
    result = decode_post_index(raw, active=True)
    assert result["submitted_requests"] == sum(range(64)) and not result["fallback"]
    raw[:] = -1
    assert decode_post_index(raw, active=False) == dict(status="not_entered", active=False)


def test_diagnostic_uses_latest_end_and_same_core_durations_not_sum():
    raw = data(True)
    result = decode_post_index(raw, active=True, diagnostic=True)
    assert result["phase_end_from_first_begin_cycles"] == dict(plan=20, pack=25, post_begin=30, post=50, index=80, cq=120)
    assert result["per_core_phase_cycles"]["cq"] == dict(median_cycles=30., max_cycles=40)
    assert result["phase_policy"] == "post_then_index_then_cq"
    raw[:, 2] = 1
    raw[:, 9] = 150  # Synchronous fallback finished its CQ before Index.
    result = decode_post_index(raw, active=True, diagnostic=True)
    assert result["fallback"] and result["fallback_cores"] == 64
    assert result["phase_policy"] == "bounded_sync_get_then_index"
    assert "cq" not in result["per_core_phase_cycles"]


@pytest.mark.parametrize("fault", ["error", "global-error", "incomplete", "bad-fallback", "mixed-fallback", "timed-ticks", "ordering", "missing"])
def test_rejects_errors_or_misleading_phase_data(fault):
    diagnostic = fault in ("ordering", "missing")
    raw = data(diagnostic)
    if fault == "error":
        raw[3, 0] = 1
    elif fault == "global-error":
        raw[4, 3] = 1
    elif fault == "incomplete":
        raw[4, 15] -= 1
    elif fault == "bad-fallback":
        raw[:, 2] = 2
    elif fault == "mixed-fallback":
        raw[0, 2] = 1
    elif fault == "timed-ticks":
        raw[0, 4] = 1
    elif fault == "ordering":
        raw[0, 8] = 99
    elif fault == "missing":
        raw[0, 9] = 0
    with pytest.raises(ValueError):
        decode_post_index(raw, active=True, diagnostic=diagnostic)


def test_new_low_tail_cannot_reinterpret_old_consumer_or_get_timestamps():
    batch, workers = 2, 2
    raw = np.full(batch * 256 + (3 * batch + 4 * workers) * 32, -1, dtype=np.int64)
    producer = raw[:batch * 64].reshape(batch, 64)
    producer[:, 57:64] = [100, 101, 121, 123, 163, 173, 180]
    raw[batch * 64:batch * 128] = 0
    original = raw.copy()
    result = decode_profile(raw, batch=batch, workers=workers, resolver_cores=3, transport="urma",
                            init_route_markers=True, direct_index_markers=True, post_index_low=True)
    assert result["direct_index_counts"] is None
    assert result["metrics"]["index_role_span"]["status"] == "unavailable"
    assert result["metrics"]["fetch_query_ready_wait"]["status"] == "unavailable"
    assert result["metrics"]["init_total"]["samples_cycles"] == [80, 80]
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize("low", [False, True])
def test_small_batch_full_grid_ignores_records_without_old_high_roles(low):
    batch, workers, role_cores = 1, 4, 19
    raw = np.zeros(batch * 256 + 64 * 32, dtype=np.int64)
    raw[batch * 256 + role_cores * 32:] = -1
    result = decode_profile(raw, batch=batch, workers=workers, resolver_cores=3, transport="urma",
                            tail_cores=64, post_index_low=low)
    assert len(result["raw"]["cores"]) == role_cores
    assert result["metrics"]["index_role_span"]["status"] == "unavailable"
    with pytest.raises(ValueError, match="timestamp layout"):
        decode_profile(raw, batch=batch, workers=workers, resolver_cores=3, transport="urma")
