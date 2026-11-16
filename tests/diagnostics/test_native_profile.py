# SPDX-License-Identifier: Apache-2.0
"""Instrumented replay markers must never contaminate headline samples."""

from types import SimpleNamespace

import numpy as np
import pytest

from kvip.diagnostics import native_profile, replay_measure
from kvip.diagnostics.fused_replay import validate_fused_tuning, verify_build_options


def _raw():
    raw = np.zeros(2 * 256 + (3 * 2 + 4 * 2) * 32, dtype=np.int64)
    li = raw[128:256].reshape(8, 16)
    li[:, 12] = 100
    li[:, 13] = 100 + np.arange(8)
    li[:, 14:16] = [200, 211]
    return raw


def test_profile_reports_role_median_max_without_summing_or_clock_conversion(capsys):
    sample = native_profile.decode_profile(
        _raw(), batch=2, workers=2, resolver_cores=3, transport="urma", threshold_markers=True
    )
    assert sample["metrics"]["early_threshold"] == dict(
        status="available", count=8, median_cycles=3.5, max_cycles=7, samples_cycles=list(range(8))
    )
    assert sample["metrics"]["final_topk"]["median_cycles"] == 11
    assert sample["metrics"]["fetch_dependency_wait"]["status"] == "unavailable"
    native_profile.print_profile("urma-recent-kth-only", {"samples": [sample]})
    output = capsys.readouterr().out
    assert "raw_cycles_median/max" in output and "no cycle-to-us conversion" in output
    assert "samples_cycles" not in output and "cores" not in output


def test_old_main_slots_are_unavailable_even_if_they_contain_nonzero_values():
    raw = _raw()
    raw[128 + 13] = -1
    sample = native_profile.decode_profile(raw, batch=2, workers=2, resolver_cores=3, transport="urma")
    assert sample["metrics"]["early_threshold"]["status"] == "unavailable"
    assert sample["metrics"]["final_topk"]["status"] == "unavailable"


def test_output_packing_uses_publication_role_and_remains_separate_from_get_pack():
    raw = _raw()
    cores = raw[512:].reshape(-1, 32)
    for row in (2, 5):
        for query in range(4):
            cores[row, 6 + query * 6] = 100 + query * 100
            cores[row, 7 + query * 6] = 110 + query * 101
    profile = native_profile.decode_profile(raw, batch=2, workers=2, resolver_cores=3, transport="urma")
    assert profile["metrics"]["publication_pack"]["samples_cycles"] == [10, 11, 12, 13] * 2
    assert profile["metrics"]["publication_pack"]["median_cycles"] == 11.5
    assert profile["metrics"]["producer_span"]["status"] == "unavailable"
    assert profile["metrics"]["fetch_query_ready_wait"]["status"] == "unavailable"


def test_init_markers_separate_route_fill_and_tail_without_summing_blocks():
    raw = _raw()
    consumer = raw[:128].reshape(2, 64)[:, 32:]
    consumer[:, 25:32] = [[100, 101, 121, 123, 163, 173, 180],
                         [200, 202, 242, 244, 254, 274, 290]]
    profile = native_profile.decode_profile(
        raw, batch=2, workers=2, resolver_cores=3, transport="urma", init_route_markers=True
    )
    assert profile["metrics"]["init_total"]["samples_cycles"] == [80, 90]
    assert profile["metrics"]["init_route"]["samples_cycles"] == [20, 40]
    assert profile["metrics"]["init_route_publish"]["samples_cycles"] == [20]
    assert profile["metrics"]["init_route_probe"]["samples_cycles"] == [40]
    assert profile["metrics"]["init_fill"]["samples_cycles"] == [40, 10]
    assert profile["metrics"]["init_tail_setup"]["samples_cycles"] == [10, 20]
    assert profile["metrics"]["init_finish"]["samples_cycles"] == [7, 16]
    assert profile["metrics"]["init_total"]["median_cycles"] == 85
    assert profile["metrics"]["init_total"]["max_cycles"] == 90
    assert "Blocks overlap" in profile["scope"]["initialization"]
    distributed = native_profile.decode_profile(
        raw, batch=2, workers=2, resolver_cores=3, transport="urma",
        init_route_markers=True, row_route_publish=True,
    )
    assert distributed["metrics"] == profile["metrics"]
    assert "both intervals may include publishing one pending row" in distributed["scope"]["initialization"]
    assert "Only block 0 publishes" in profile["scope"]["initialization"]
    consumer[:, 25:32] = -1
    old = native_profile.decode_profile(raw, batch=2, workers=2, resolver_cores=3, transport="urma")
    assert old["metrics"]["init_total"]["status"] == "unavailable"
    assert old["metrics"]["init_route_publish"]["status"] == "unavailable"
    assert old["metrics"]["init_route_probe"]["status"] == "unavailable"


@pytest.mark.parametrize("bad", [0, 99])
def test_init_marker_capability_requires_each_block_to_have_valid_markers(bad):
    raw = _raw()
    consumer = raw[:128].reshape(2, 64)[:, 32:]
    consumer[:, 25:32] = [100, 101, 121, 123, 163, 173, 180]
    consumer[1, 27] = bad
    with pytest.raises(ValueError, match="diagnostic timestamps"):
        native_profile.decode_profile(
            raw, batch=2, workers=2, resolver_cores=3, transport="urma", init_route_markers=True
        )


def test_direct_index_counters_are_capability_gated_and_not_latencies():
    raw = _raw()
    consumer = raw[:128].reshape(2, 64)[:, 32:]
    consumer[:, 2:4] = [[1, 19], [0, 0]]
    profile = native_profile.decode_profile(
        raw, batch=2, workers=2, resolver_cores=3, transport="urma", direct_index_markers=True
    )
    assert profile["direct_index_counts"] == dict(
        direct_rows=1, fallback_rows=1, direct_events=19,
        rows=[dict(row=0, direct=True, events=19), dict(row=1, direct=False, events=0)],
    )
    assert "direct_index" not in profile["metrics"]
    consumer[:, 2:4] = -1
    old = native_profile.decode_profile(raw, batch=2, workers=2, resolver_cores=3, transport="urma")
    assert old["direct_index_counts"] is None


@pytest.mark.parametrize("counts", [(2, 1), (1, 8193), (0, 1), (-1, 0)])
def test_invalid_direct_index_counters_are_not_reported_as_saved_work(counts):
    raw = _raw()
    raw[:128].reshape(2, 64)[0, 34:36] = counts
    with pytest.raises(ValueError, match="Invalid direct Index update counters"):
        native_profile.decode_profile(
            raw, batch=2, workers=2, resolver_cores=3, transport="urma", direct_index_markers=True
        )


def test_early_drain_is_separate_from_later_resolver_join():
    raw = _raw()
    consumer = raw[:128].reshape(2, 64)[:, 32:]
    consumer[:, :2] = [100, 180]
    fetch = raw[256:512].reshape(2, 128)
    fetch[:, 96:101] = [200, 260, 270, 280, 210]
    sample = native_profile.decode_profile(
        raw, batch=2, workers=2, resolver_cores=3, transport="urma", early_prefetch_drain=True
    )
    assert sample["metrics"]["early_prefetch_drain"]["samples_cycles"] == [80, 80]
    assert sample["metrics"]["final_prefetch_join"]["samples_cycles"] == [10, 10]
    assert sample["metrics"]["final_index_and_query_join"]["samples_cycles"] == [50, 50]
    consumer[0, 1] = 0
    with pytest.raises(ValueError, match="diagnostic timestamps"):
        native_profile.decode_profile(
            raw, batch=2, workers=2, resolver_cores=3, transport="urma", early_prefetch_drain=True
        )
    old = native_profile.decode_profile(raw, batch=2, workers=2, resolver_cores=3, transport="urma")
    assert "early_prefetch_drain" not in old["metrics"]


def test_index_role_interval_keeps_work_moved_before_init_in_scope():
    raw = _raw()
    consumer = raw[:128].reshape(2, 64)[:, 32:]
    consumer[:, :2] = [100, 180]
    consumer[:, 10] = 200
    consumer[:, 24] = 260
    roles = raw[512:].reshape(-1, 32)
    roles[[1, 4], :2] = [90, 280]
    sample = native_profile.decode_profile(
        raw, batch=2, workers=2, resolver_cores=3, transport="urma", early_prefetch_drain=True
    )
    assert sample["metrics"]["index_role_span"]["samples_cycles"] == [190, 190]
    assert sample["metrics"]["index_consumer_span"]["samples_cycles"] == [60, 60]
    assert sample["metrics"]["early_prefetch_drain"]["samples_cycles"] == [80, 80]
    assert "excludes early drain" in sample["scope"]["index"]
    # The headline role time stays fixed when only the subinterval start moves.
    consumer[:, 10] = 210
    after = native_profile.decode_profile(
        raw, batch=2, workers=2, resolver_cores=3, transport="urma", early_prefetch_drain=True
    )
    assert after["metrics"]["index_role_span"] == sample["metrics"]["index_role_span"]
    assert after["metrics"]["index_consumer_span"]["median_cycles"] == 50


def test_capacity_skip_counter_does_not_become_a_zero_candidate_count(monkeypatch):
    rows = np.zeros((1, 256), dtype=np.int32)
    rows[0, 9] = 2
    events = np.zeros((2, 1, 128), dtype=np.uint8)
    events.view(np.uint32)[:, 0, 7] = [1, 3]
    rotating = SimpleNamespace(
        rows=rows, row_status=np.zeros((1, 32), dtype=np.int32),
        final_posts=np.zeros((1, 4, 16), dtype=np.int32), events=events,
        hot_valid=np.zeros((1, 4), dtype=np.int32),
        arguments=[None] * 9 + [np.array([[0, 1]], dtype=np.int32)],
    )
    monkeypatch.setattr(replay_measure, "cpu_array", lambda torch, value: np.asarray(value))
    monkeypatch.setattr(replay_measure, "mapping_values", lambda *args: np.array([[4, 5, -1, -1]]))
    result = replay_measure.fused_reads(
        None, rotating, SimpleNamespace(status=np.zeros(1)), "urma", 656, skip_exhausted_prefetch=True
    )
    assert result["capacity_skipped_row_windows"] == 4
    assert result["transfers"] == 2 and result["bytes"] == 1312
    assert "not enumerated" in result["suppression_scope"]


@pytest.mark.parametrize("change", ["missing", "reversed", "size"])
def test_new_feature_markers_require_a_complete_valid_contract(change):
    raw = _raw()
    if change == "missing":
        raw[128 + 12] = 0
    elif change == "reversed":
        raw[128 + 13] = 1
    else:
        raw = raw[:-1]
    with pytest.raises(ValueError, match="diagnostic timestamps|timestamp layout"):
        native_profile.decode_profile(
            raw, batch=2, workers=2, resolver_cores=3, transport="urma", threshold_markers=True
        )


def test_threshold_only_flag_is_urma_only_and_native_export_must_match():
    assert validate_fused_tuning({"recent_threshold_only": True}, "urma") == {"recent_threshold_only": True}
    with pytest.raises(ValueError, match="only by URMA"):
        validate_fused_tuning({"recent_threshold_only": True}, "ubmem")
    library = SimpleNamespace(kvip_recent_threshold_only=lambda: 0)
    with pytest.raises(RuntimeError, match="recent_threshold_only=True"):
        verify_build_options(library, {"recent_threshold_only": True})


@pytest.mark.parametrize("fail_profile", [False, True])
def test_diagnostic_pass_follows_headline_events_and_keeps_failure_owners(monkeypatch, fail_profile):
    state = dict(profile=False, events=0, completed=0, replays=0, verifies=0, resets=0, closed=False)
    log, retained = [], {}

    class Times:
        def zero_(self):
            log.append("zero")

    class Forward:
        batch, q_workers, resolver_cores = 2, 2, 3
        times = Times()

        def __call__(self):
            assert not state["profile"]

        def set_options(self, diagnostics=False):
            assert state["completed"] == 3
            state["profile"] = diagnostics
            log.append(("profile", diagnostics))

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self, stream):
            assert not state["profile"]
            state["events"] += 1

        def synchronize(self):
            state["completed"] += 1

        def elapsed_time(self, other):
            return 0.01

    class Graph:
        def capture_begin(self, stream):
            pass

        def capture_end(self):
            pass

        def replay(self, stream):
            state["replays"] += 1
            log.append(("replay", state["profile"]))
            if state["profile"] and fail_profile:
                raise RuntimeError("profile device fault")

        def reset(self):
            state["closed"] = True

    def verify():
        state["verifies"] += 1
        log.append(("verify", state["profile"]))
        return {"checked": True}

    def reset():
        state["resets"] += 1

    bundle = SimpleNamespace(
        reset=reset, verify=verify, read_counts=lambda: {"reads": 1}, forward=Forward(), lease=None
    )
    args = SimpleNamespace(mode="fused", transport="urma", warmup=1, iterations=2, diagnostic_replays=1)
    monkeypatch.setattr(replay_measure, "cpu_array", lambda *args: _raw())

    def call():
        return replay_measure.measure(
            SimpleNamespace(npu=SimpleNamespace(Event=Event)),
            SimpleNamespace(AclGraph=Graph),
            SimpleNamespace(synchronize=lambda: log.append("sync")),
            bundle,
            args,
            retained,
        )

    if fail_profile:
        with pytest.raises(RuntimeError, match="profile device fault"):
            call()
        assert state["profile"] and not state["closed"]
        assert retained["progress"] == {"phase": "diagnostic_replay_and_sync", "step": 0}
        assert retained["graphs"]
    else:
        result = call()
        assert result["timing"]["samples_us"] == [10, 10]
        assert len(result["reads_samples"]) == 2
        assert result["native_diagnostics"]["count"] == 1
        assert result["native_diagnostics"]["included_in_timing"] is False
        assert state["verifies"] == 4 and state["closed"] and not state["profile"]
        assert log.index(("verify", False)) < log.index(("profile", True))
        assert log.index(("verify", True)) < log.index(("profile", False))
    assert state["events"] == 6 and state["replays"] == 5


def test_grouped_urma_post_and_cq_counters_are_deduplicated_by_query_worker():
    batch, workers = 4, 2
    raw = np.zeros(batch * 256 + (3 * batch + 4 * workers) * 32, dtype=np.int64)
    fetch = raw[batch * 128 : batch * 256].reshape(batch, 128)
    queries = fetch[:, :96].reshape(batch, 4, 24)
    for b, q in np.ndindex(batch, 4):
        lane = b % workers
        queries[b, q, [3, 4, 7, 8]] = [100, 120, 200, 300]
        queries[b, q, [12, 13, 14, 15, 16, 17, 20]] = [10 + lane, 1, 20 + q, 2, 1, 1, 3]
    sample = native_profile.decode_profile(raw, batch=batch, workers=workers, resolver_cores=3, transport="urma")
    assert sample["metrics"]["post_active"] == dict(
        status="available", count=8, median_cycles=32, max_cycles=34, samples_cycles=[30, 31, 32, 33, 31, 32, 33, 34]
    )
    assert sample["metrics"]["cq_drain"]["samples_cycles"] == [3] * 8
    assert sample["metrics"]["fetch_conflict_get"]["count"] == 8
    queries[2, 0, 13] += 1
    with pytest.raises(ValueError, match="Replicated shared Q-worker"):
        native_profile.decode_profile(raw, batch=batch, workers=workers, resolver_cores=3, transport="urma")


def test_ready_batch_native_durations_exclude_inter_call_wait_and_deduplicate():
    batch, workers = 4, 2
    raw = np.zeros(batch * 256 + (3 * batch + 4 * workers) * 32, dtype=np.int64)
    q = raw[batch * 128:batch * 256].reshape(batch, 128)[:, :96].reshape(batch, 4, 24)
    for b, query in np.ndindex(batch, 4):
        q[b, query, [3, 4, 12, 13, 18, 19, 20, 21, 22, 23]] = [100, 900, 20, 300, 100, 60, 4, 600, 128, 2]
    result = native_profile.decode_profile(raw, batch=batch, workers=workers, resolver_cores=3, transport='urma')
    assert result['metrics']['fetch_batch_envelope']['median_cycles'] == 800
    assert result['metrics']['fetch_native_get']['median_cycles'] == 600
    assert result['metrics']['fetch_pack']['median_cycles'] == 60
    assert result['metrics']['fetch_safe_submit_or_sync']['status'] == 'unavailable'
    assert result['metrics']['cq_drain']['count'] == 8
    assert result['ready_batch_counts'] == dict(native_calls=16, completed_requests=1024, worker_count=8)
    assert result['metrics']['fetch_dependency_wait']['status'] == 'not_entered'
    assert result['ready_batch_by_query'] == [
        dict(query=i, requests=256, calls=4, native_max_cycles=600, active_workers=2, worker_count=2)
        for i in range(4)
    ]
    q[2, 0, 23] = 9
    with pytest.raises(ValueError, match='Replicated shared'):
        native_profile.decode_profile(raw, batch=batch, workers=workers, resolver_cores=3, transport='urma')


def test_ready_query_load_and_conflict_wait_distinguish_idle_from_missing(capsys):
    batch, workers = 4, 2
    raw = np.zeros(batch * 256 + (3 * batch + 4 * workers) * 32, dtype=np.int64)
    queries = raw[batch * 128:batch * 256].reshape(batch, 128)[:, :96].reshape(batch, 4, 24)
    for b, q in np.ndindex(batch, 4):
        lane = b % workers
        requests = 100 * (q + 1) + lane if q < 3 or lane == 0 else 0
        calls = q + 1 if requests else 0
        cycles = 50 * calls + lane if requests else 0
        queries[b, q, [20, 21, 22, 23]] = [4, cycles, requests, calls]
    sample = native_profile.decode_profile(raw, batch=batch, workers=workers, resolver_cores=3, transport='urma')
    rows = sample['ready_batch_by_query']
    assert rows[0] == dict(query=0, requests=201, calls=2, native_max_cycles=51, active_workers=2, worker_count=2)
    assert rows[3] == dict(query=3, requests=400, calls=4, native_max_cycles=200, active_workers=1, worker_count=2)
    assert sum(row['requests'] for row in rows) == sample['ready_batch_counts']['completed_requests']
    assert sum(row['calls'] for row in rows) == sample['ready_batch_counts']['native_calls']
    native_profile.print_profile('ready', {'samples': [sample]})
    assert 'fetch_dependency_wait=not_entered' in capsys.readouterr().out
    # Per-request dependency timestamps are not replicated worker counters.
    queries[2, 0, [5, 6]] = [100, 140]
    entered = native_profile.decode_profile(raw, batch=batch, workers=workers, resolver_cores=3, transport='urma')
    assert entered['metrics']['fetch_dependency_wait']['status'] == 'available'
    assert entered['metrics']['fetch_dependency_wait']['samples_cycles'] == [40]


@pytest.mark.parametrize('timestamps', [(0, 100), (100, 0), (100, 99)])
def test_ready_conflict_partial_or_reversed_markers_are_not_hidden(timestamps):
    raw = _raw()
    queries = raw[256:512].reshape(2, 128)[:, :96].reshape(2, 4, 24)
    queries[:, :, 20] = 4
    queries[0, 0, [5, 6]] = timestamps
    with pytest.raises(ValueError, match='fetch_dependency_wait'):
        native_profile.decode_profile(raw, batch=2, workers=2, resolver_cores=3, transport='urma')
