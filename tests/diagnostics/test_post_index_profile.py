# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from kvip.diagnostics.post_index_profile import INDEX_DETAIL_MAGIC, INDEX_DETAIL_SAME_VF_MAGIC, decode_post_index
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


def detail_data(raw):
    detail = np.zeros((len(raw), 32), dtype=np.int64)
    detail[:, 2] = INDEX_DETAIL_MAGIC
    detail[:, 3:8] = [155, 160, 175, 190, 210]
    detail[:, 7] = raw[:, 9]
    detail[:, 9] = sum(raw[:, 1]) // len(raw)
    detail[:sum(raw[:, 1]) % len(raw), 9] += 1
    detail[:, 8] = np.arange(len(raw)) // 2
    detail[:, 10] = detail[:, 8] * 592
    return detail


def test_index_detail_separates_actual_work_from_two_barriers_and_cq():
    raw = data(True)
    detail = detail_data(raw)
    result = decode_post_index(raw, active=True, diagnostic=True, index_detail=detail)["index_work"]
    assert result["assigned_events"]["total"] == sum(range(64))
    assert result["moved_tuples"] == dict(min=0, p50=15.5, max=31, mean=15.5,
                                          total=992, active_cores=62, samples=list(np.arange(64) // 2))
    assert result["assigned_events"]["max"] - result["assigned_events"]["min"] == 1
    assert result["logical_index_rw_bytes"] == 992 * 592
    cycles = result["per_core_cycles"]
    assert cycles["post_barrier"]["p50"] == 5
    assert cycles["index_setup"]["p50"] == 5
    assert cycles["index_work"]["p50"] == 15
    assert cycles["index_publish"]["p50"] == 5
    assert cycles["index_barrier"]["p50"] == 10
    assert cycles["cq_wait"]["p50"] == 20 and cycles["cq_wait"]["max"] == 30
    assert "total" not in cycles["index_work"]
    raw[:, 2] = 1
    raw[:, 9] = raw[:, 7]
    fallback = decode_post_index(raw, active=True, diagnostic=True, index_detail=detail)["index_work"]
    assert "cq_wait" not in fallback["per_core_cycles"]
    assert fallback["per_core_cycles"]["cq_finalize"]["p50"] == 20


def test_index_detail_without_magic_never_interprets_old_role_data():
    raw = data(True)
    for detail in (None, np.full((64, 32), -1, dtype=np.int64)):
        result = decode_post_index(raw, active=True, diagnostic=True, index_detail=detail)
        assert result["index_work"]["status"] == "unavailable"
    assert decode_post_index(raw, active=False, diagnostic=True, index_detail=object()) == dict(status="not_entered", active=False)


@pytest.mark.parametrize("fault", ["shape", "partial", "clock", "post-clock", "index-clock", "cq-clock",
                                   "assigned", "unbalanced", "moved", "bytes"])
def test_index_detail_rejects_incomplete_or_misleading_work_data(fault):
    raw = data(True)
    detail = detail_data(raw)
    if fault == "shape":
        detail = detail[:, :31]
    elif fault == "partial":
        detail[0, 2] = 0
    elif fault == "clock":
        detail[0, 4] = 0
    elif fault == "post-clock":
        detail[0, 3] = raw[0, 7] - 1
    elif fault == "index-clock":
        detail[0, 5:7] = 181
    elif fault == "cq-clock":
        detail[0, 7] += 1
    elif fault == "assigned":
        detail[0, 9] += 1
    elif fault == "unbalanced":
        detail[0, 9] += 2
        detail[1, 9] -= 2
    elif fault == "moved":
        detail[0, 8] = detail[0, 9] + 1
    elif fault == "bytes":
        detail[0, 10] = 592
    with pytest.raises(ValueError):
        decode_post_index(raw, active=True, diagnostic=True, index_detail=detail)


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
        raw[:, 2] = 4
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


def test_wait_first_control_masks_order_bit_and_separates_cq_barrier():
    raw = data(True)
    raw[:, 2] = 2
    raw[:, 9] = 158
    detail = detail_data(raw)
    detail[:, 7] = 158
    result = decode_post_index(raw, active=True, diagnostic=True, index_detail=detail)
    assert result["wait_first"] and not result["fallback"] and result["fallback_cores"] == 0
    assert result["phase_policy"] == "post_then_cq_then_index"
    spans = result["index_work"]["per_core_cycles"]
    assert spans["cq_wait"]["p50"] == 3
    assert spans["cq_barrier"]["p50"] == 2
    assert spans["index_setup"]["p50"] == 0
    assert result["index_work"]["global_span_cycles"]["index_work"] == 15
    detail[0, 7] = 161
    with pytest.raises(ValueError, match="timestamps"):
        decode_post_index(raw, active=True, diagnostic=True, index_detail=detail)


def pix2_data():
    raw = data(True)
    raw[:, 11] = np.arange(64) % 16
    source = np.arange(16*10240, dtype=np.int32).reshape(16, 10240)
    destination = source.copy()
    destination[:, 2048::2] += 1
    detail = detail_data(raw)
    detail[:, 2] = INDEX_DETAIL_SAME_VF_MAGIC
    detail[:, 8] = detail[:, 10] = 0
    total = sum(map(int, raw[:, 1]))
    detail[:, 9] = np.diff([total*core//64 for core in range(65)])
    return raw, detail, (source, destination)


def test_pix2_counts_after_replay_without_instrumenting_index_vf():
    raw, detail, events = pix2_data()
    result = decode_post_index(raw, active=True, diagnostic=True, index_detail=detail, events=events)["index_work"]
    assert result["abi"] == "PIX2" and result["index_implementation"] == "headline_same_VF"
    assert result["count_source"] == "post_replay_CPU_event_readback"
    assert result["moved_tuples"]["total"] == sum(range(64)) // 2
    assert result["logical_index_rw_bytes"] == result["moved_tuples"]["total"]*592
    assert decode_post_index(raw, active=False, diagnostic=True, index_detail=object(), events=object())["status"] == "not_entered"


@pytest.mark.parametrize("fault", ["missing", "dtype", "shape", "row", "capacity", "assigned", "mixed"])
def test_pix2_fails_closed_if_cpu_counts_do_not_match_device_ranges(fault):
    raw, detail, events = pix2_data()
    if fault == "missing":
        events = None
    elif fault == "dtype":
        events = (events[0].astype(np.int64), events[1])
    elif fault == "shape":
        events = (events[0][:, :100], events[1][:, :100])
    elif fault == "row":
        raw[0, 11] = 1
    elif fault == "capacity":
        raw[0, 1] = raw[0, 15] = 8193
    elif fault == "assigned":
        detail[0, 9] += 1
    elif fault == "mixed":
        detail[0, 2] = INDEX_DETAIL_MAGIC
    with pytest.raises(ValueError):
        decode_post_index(raw, active=True, diagnostic=True, index_detail=detail, events=events)


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
