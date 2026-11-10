# SPDX-License-Identifier: Apache-2.0
"""Native-profile call ordering, using actual KVAF field names; no NPU timing."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from kvip.diagnostics.replay_measure import measure, profile_urma_get


def core_record(core=0):
    return dict(
        core=core,
        active_requests=178 + core,
        total_posts=6,
        submitted_group_count=1,
        completed_group_count=1,
        completion_count=2,
        post_active_cycles=100 + core,
        drain_wait_cycles=200 + core,
        peak_submitted_not_drained=178 + core,
        kernel_start_cycle=1,
        setup_done_cycle=2,
        compact_done_cycle=3,
        first_doorbell_cycle=4,
        post_done_cycle=5,
        completion_done_cycle=6,
        kernel_finish_cycle=7,
        jetty_completion_wait_cycles=[0] * 64,
        jetty_completion_polls=[0] * 64,
    )


class Runtime:
    def __init__(self, events, fail_snapshot=False, fail_stop=False):
        self.events, self.fail_snapshot, self.fail_stop = events, fail_snapshot, fail_stop

    def start_batch_get_kernel_profile(self):
        self.events.append("start")

    def batch_get_kernel_profile(self):
        self.events.append("snapshot")
        if self.fail_snapshot:
            raise RuntimeError("snapshot failure")
        return [core_record(0), core_record(1)]

    def stop_batch_get_kernel_profile(self):
        self.events.append("stop")
        if self.fail_stop:
            raise RuntimeError("stop failure")
        return [core_record(0), core_record(1)]


def test_profile_restores_syncs_and_reports_native_cycles_without_rescaling():
    events = []
    report = profile_urma_get(
        Runtime(events),
        SimpleNamespace(synchronize=lambda: events.append("sync")),
        lambda: events.append("reset"),
        lambda: events.append("verify") or {"final": 357},
        lambda: events.append("replay"),
    )
    assert events == ["reset", "sync", "start", "replay", "sync", "snapshot", "stop", "verify"]
    assert report["included_in_headline"] is False and report["unit"] == "native_cycles"
    assert report["totals"] == dict(
        active_requests=357,
        total_posts=12,
        submitted_group_count=2,
        completed_group_count=2,
        completion_count=4,
        post_active_cycles=201,
        drain_wait_cycles=401,
    )
    assert report["maxima"] == dict(post_active_cycles=101, drain_wait_cycles=201)
    assert report["active_cores"] == 2 and report["cores"] == [core_record(0), core_record(1)]


@pytest.mark.parametrize(
    "failure,stop_fails", [("replay", False), ("snapshot", False), ("snapshot", True), (None, True)]
)
def test_profile_always_stops_and_keeps_original_error(failure, stop_fails):
    events = []
    runtime = Runtime(events, fail_snapshot=failure == "snapshot", fail_stop=stop_fails)

    def replay():
        events.append("replay")
        if failure == "replay":
            raise RuntimeError("replay failure")

    with pytest.raises(RuntimeError, match=(failure or "stop") + " failure"):
        profile_urma_get(
            runtime,
            SimpleNamespace(synchronize=lambda: events.append("sync")),
            lambda: events.append("reset"),
            lambda: events.append("verify"),
            replay,
        )
    assert events[-1] == "stop" and "verify" not in events


@pytest.mark.parametrize("stage_iterations", (0, 1))
def test_measure_profiles_one_extra_stage_chain_after_all_device_event_samples(stage_iterations):
    events = []

    class Graph:
        def replay(self):
            events.append("graph_replay")

        def reset(self):
            events.append("graph_reset")

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self, stream):
            events.append("event_record")

        def synchronize(self):
            events.append("event_sync")

        def elapsed_time(self, other):
            return 0.1

    torch = SimpleNamespace(npu=SimpleNamespace(NPUGraph=Graph, graph=lambda *a, **k: nullcontext(), Event=Event))
    bundle = SimpleNamespace(
        reset=lambda: events.append("reset"),
        verify=lambda: {"final": 357},
        lease=None,
        forward=lambda: None,
        stage_calls=[("li", lambda: None), ("get", lambda: None)],
    )
    args = SimpleNamespace(transport="urma", mode="serial", warmup=0, iterations=1, stage_iterations=stage_iterations)
    report = measure(
        torch,
        None,
        SimpleNamespace(synchronize=lambda: events.append("sync")),
        bundle,
        args,
        {},
        runtime=Runtime(events),
    )
    assert report["timing"]["count"] == 1
    if not stage_iterations:
        assert "native_get_profile" not in report and "start" not in events
        return
    assert report["stage_diagnostics"]["total"]["count"] == 1
    first, last = events.index("start"), events.index("stop")
    assert "event_record" not in events[first:last]
    assert events[first:last].count("graph_replay") == 2
    assert events[first - 2 : first] == ["reset", "sync"]
    assert events.index("graph_reset") > last
    assert report["native_get_profile"]["totals"]["total_posts"] == 12
