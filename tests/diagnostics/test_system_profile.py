# SPDX-License-Identifier: Apache-2.0
"""System profiling must not include restoration or correctness readbacks."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from kvip.diagnostics import graph_queue_probe, replay, replay_measure, system_profile


def fake_npu(events, *, stop_error=False, mutate_config=False):
    metrics = SimpleNamespace(AiCoreNone=0, PipeUtilization=1, Memory=2)

    class Config:
        def __init__(self, **kwargs):
            self._aic_metrics = 99 if mutate_config else kwargs["aic_metrics"]
            self._profiler_level = kwargs["profiler_level"]
            self._data_simplification = kwargs["data_simplification"]
            self.export_type = [kwargs["export_type"]]

    def handler(directory, **kwargs):
        assert kwargs["analyse_flag"] is True and kwargs["async_mode"] is False

        def export():
            target = Path(directory) / "output"
            target.mkdir()
            (target / "trace_view.json").write_text(json.dumps({"traceEvents": [{"cat": "kernel", "name": "LI"}]}))
            (target / "kernel_details.csv").write_text("Name,aic_mac_ratio,read_bandwidth\nLI,0,0\n")
            (target / "data").mkdir()
            events.append("export")

        return export

    def profile(**kwargs):
        assert kwargs["activities"] == ["cpu", "npu"]
        assert kwargs["record_shapes"] is False and kwargs["profile_memory"] is False

        class Profile:
            def start(self):
                events.append("start")

            def stop(self):
                events.append("stop")
                if stop_error:
                    raise RuntimeError("stop failed")
                kwargs["on_trace_ready"]()

        return Profile()

    return SimpleNamespace(
        __version__="test-npu",
        profiler=SimpleNamespace(
            AiCMetrics=metrics,
            ProfilerLevel=SimpleNamespace(Level0=0, Level1=1),
            ExportType=SimpleNamespace(Text="text"),
            _ExperimentalConfig=Config,
            experimental_config=SimpleNamespace(supported_ai_core_metrics=lambda: [0, 1, 2]),
            ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
            tensorboard_trace_handler=handler,
            profile=profile,
        ),
    )


def collect(tmp_path, events, **overrides):
    kwargs = dict(
        torch=SimpleNamespace(
            __version__="test-torch", profiler=SimpleNamespace(record_function=lambda name: nullcontext())
        ),
        reset=lambda: events.append("reset"),
        replay=lambda: events.append("replay"),
        synchronize=lambda: events.append("sync"),
        verify=lambda: events.append("verify"),
        directory=tmp_path,
        case="urma-fused-ready-batch4",
        metrics=["none", "PipeUtilization", "Memory"],
        metadata={"snapshot_sha256": "frozen", "production_source": {"commit": "production"}},
        retained={},
        checkpoint=lambda *args: None,
        npu=fake_npu(events),
    )
    kwargs.update(overrides)
    return system_profile.collect_system_profiles(**kwargs)


def test_each_metric_profiles_exactly_one_existing_replay_then_verifies(tmp_path):
    events, retained = [], {}
    result = collect(tmp_path, events, retained=retained)
    assert events == ["reset", "sync", "start", "replay", "sync", "stop", "export", "verify"] * 3
    assert result["included_in_headline_timing"] is False
    assert len(retained["system_profilers"]) == 3
    for entry in result["passes"]:
        saved = json.loads((Path(entry["directory"]) / "profile_manifest.json").read_text())
        assert saved["experiment"]["snapshot_sha256"] == "frozen"
        assert saved["versions"]["torch_npu"] == "test-npu"
        assert saved["configuration"]["data_simplification"] is False
        assert saved["graph_replays"] == 1 and saved["correctness"] == "passed"
        assert saved["artifacts"]["status"] == "exports_present_review_required"
        assert saved["artifacts"]["hardware_metrics_validated"] is False
        assert saved["configuration"]["profiler_level"] == ("Level0" if entry["metric"] == "none" else "Level1")
    with pytest.raises(FileExistsError, match="already exists"):
        collect(tmp_path, events)
    assert len(events) == 24


def test_missing_metric_or_silent_fallback_fails_before_first_replay(tmp_path):
    events = []
    with pytest.raises(ValueError, match="does not support"):
        collect(tmp_path, events, metrics=["none", "MemoryAccess"])
    assert not events and not list(tmp_path.iterdir())
    with pytest.raises(RuntimeError, match="changed the requested"):
        collect(tmp_path, events, npu=fake_npu(events, mutate_config=True))
    assert not events


def test_none_does_not_require_membership_in_supported_pmu_list():
    npu = fake_npu([])
    npu.profiler.experimental_config.supported_ai_core_metrics = lambda: [1, 2]
    config = system_profile.profile_configurations(npu, ["none"])[0][1]
    assert config._aic_metrics == 0 and config._profiler_level == 0


def test_text_export_accepts_old_single_value_config_property():
    npu = fake_npu([])
    original = npu.profiler._ExperimentalConfig

    def config(**kwargs):
        result = original(**kwargs)
        result.export_type = "text"
        return result

    npu.profiler._ExperimentalConfig = config
    assert len(system_profile.profile_configurations(npu, ["none", "PipeUtilization", "Memory"])) == 3


def test_replay_failure_attempts_stop_preserves_primary_and_never_verifies(tmp_path):
    events, retained = [], {}

    def fail():
        events.append("replay")
        raise RuntimeError("invalid GM")

    with pytest.raises(RuntimeError, match="invalid GM"):
        collect(
            tmp_path, events, metrics=["none"], replay=fail, retained=retained, npu=fake_npu(events, stop_error=True)
        )
    assert events == ["reset", "sync", "start", "replay", "stop"]
    report = json.loads(Path(retained["system_profile_manifests"][0]).read_text())
    assert report["error"]["message"] == "invalid GM" and report["stop_error"]["message"] == "stop failed"
    assert retained["system_profilers"]  # Unknown completion stays owned until fatal CLI exit.


def test_stop_failure_is_not_retried_and_does_not_verify(tmp_path):
    events = []
    with pytest.raises(RuntimeError, match="stop failed"):
        collect(tmp_path, events, metrics=["none"], npu=fake_npu(events, stop_error=True))
    assert events == ["reset", "sync", "start", "replay", "sync", "stop"]


def test_empty_or_unrecognized_exports_are_not_advertised_as_valid_pmu(tmp_path):
    result = system_profile.inspect_artifacts(tmp_path, "PipeUtilization")
    assert result["status"] == "incomplete_or_unrecognized_exports"
    assert result["hardware_metrics_validated"] is False


def cli(tmp_path, *extra):
    return replay.arguments(
        [
            "--snapshot",
            str(tmp_path / "input"),
            "--build-dir",
            str(tmp_path / "build"),
            "--transport",
            "urma",
            "--mode",
            "fused",
            "--soc",
            "A5",
            "--kvaf-server",
            "127.0.0.1:29097",
            "--output",
            str(tmp_path / "result.json"),
            *extra,
        ]
    )


@pytest.mark.parametrize(
    "extra",
    [
        ["--profile-graph-event"],
        ["--profile-dir", "p", "--profile-graph-event", "--profile-metrics", "Memory"],
        ["--profile-metrics", "Memory"],
        ["--profile-dir", "profiles", "--profile-metrics", "Memory", "Memory"],
        ["--profile-dir", "profiles", "--diagnostic-replays", "1"],
        ["--profile-dir", "profiles", "--topk-probe"],
        ["--profile-dir", "profiles", "--profile-metrics", "Unknown"],
    ],
)
def test_cli_rejects_ambiguous_or_combined_instrumentation(tmp_path, extra):
    with pytest.raises(SystemExit) as error:
        cli(tmp_path, *extra)
    assert error.value.code == 2


def test_cli_defaults_to_timeline_and_accepts_metric_sets(tmp_path):
    assert cli(tmp_path).profile_metrics is None
    assert cli(tmp_path, "--profile-dir", str(tmp_path / "profiles")).profile_metrics == ["none"]
    args = cli(
        tmp_path, "--profile-dir", str(tmp_path / "profiles"), "--profile-metrics", "none", "PipeUtilization", "Memory"
    )
    assert args.profile_metrics == ["none", "PipeUtilization", "Memory"]


def test_auto_timeline_is_one_learned_low_pass(tmp_path):
    base = ["--study", "post-threshold", "--threshold-policy", "recent", "--pipeline-variants", "auto-post-index",
            "--upstream-root", str(tmp_path), "--upstream-revision", "0" * 40, "--profile-dir", str(tmp_path / "p")]
    with pytest.raises(SystemExit):
        cli(tmp_path, *base)
    base += ["--check-auto-route", "--iterations", "2"]
    assert cli(tmp_path, *base).profile_metrics == ["none"]
    for conflict in (["--iterations", "1"], ["--profile-metrics", "Memory"],
                     ["--pipeline-variants", "auto-tail", "auto-post-index"], ["--post-index-wait-first"]):
        with pytest.raises(SystemExit):
            cli(tmp_path, *base, *conflict)


@pytest.mark.parametrize("mismatch", ["high", "fallback", "wait_first", "prefetch", "stale", "no_learning", None])
def test_low_trace_requires_device_route_and_step_completion(mismatch):
    observed = dict(mode="high" if mismatch == "high" else "low", steps=[3 if mismatch == "stale" else 4],
                    learned_owners=[] if mismatch == "no_learning" else [7])
    auto = SimpleNamespace(requests=[[7, 1, 4, 4]], record=lambda reads: observed)
    reads = dict(prefetch=int(mismatch == "prefetch"), final=955,
                 post_index=dict(active=mismatch != "high", fallback=mismatch == "fallback",
                                 wait_first=mismatch == "wait_first"))
    if mismatch:
        with pytest.raises(AssertionError, match="LOW"):
            replay_measure.verified_low_trace(auto, reads, 4)
    else:
        assert replay_measure.verified_low_trace(auto, reads, 4)["final_get"] == 955


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("torch_frontend", [False, True])
def test_measure_profiles_only_after_headline_and_before_graph_reset(tmp_path, monkeypatch, automatic, paired, torch_frontend):
    events = []

    class Graph:
        def capture_begin(self, stream):
            events.append("capture")

        def capture_end(self):
            pass

        def replay(self, stream):
            events.append("replay")

        def reset(self):
            events.append("graph_reset")

    class TorchGraph:
        def replay(self):
            events.append("replay")

        def reset(self):
            events.append("graph_reset")

    stream = SimpleNamespace(synchronize=lambda: events.append("sync"))

    def torch_capture(graph, **kwargs):
        assert torch_frontend and isinstance(graph, TorchGraph) and kwargs == {"stream": stream}
        events.append("capture")
        return nullcontext()

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self, stream):
            events.append("event")

        def synchronize(self):
            events.append("event_sync")

        def elapsed_time(self, other):
            return 0.01

    original = system_profile.collect_system_profiles
    monkeypatch.setattr(
        system_profile, "collect_system_profiles", lambda **kwargs: original(**kwargs, npu=fake_npu(events))
    )
    def queue_probe(torch, stream, begin, end, replay, prepare, verify, first_step, expected_get):
        assert paired and expected_get == 955
        for step in range(first_step, first_step + 4):
            prepare(step)
            replay()
            assert verify(step) == 955
        return dict(samples=[{} for _ in range(4)])

    monkeypatch.setattr(graph_queue_probe, "probe", queue_probe)
    bundle = SimpleNamespace(
        reset=lambda: events.append("reset"),
        verify=lambda: events.append("verify") or {"final": 955},
        forward=lambda: events.append("forward"),
        read_counts=lambda: {"final": 955},
        lease=None,
    )
    if automatic:
        counts = dict(prefetch=0, final=955, post_index=dict(active=True, fallback=False, wait_first=False))
        def begin(*, step, learning):
            assert learning
            events.append(f"begin-{step}")
            bundle.auto_replay.requests = [[7, 1, step, 4]]

        def lifecycle(*args, **kwargs):
            assert args[-1] == (7 if paired else 3) and events[-1] == "verify"
            events.append("lifecycle")
            return {"status": "pass"}

        bundle.auto_replay = SimpleNamespace(
            state=SimpleNamespace(config=[0] * 16), begin=begin,
            record=lambda reads: dict(mode="low", learned_owners=[7], steps=[bundle.auto_replay.requests[0][2]]))
        bundle.read_counts = lambda: counts
        bundle.verify = lambda: events.append("verify") or counts
        monkeypatch.setattr(replay_measure, "cpu_array", lambda torch, data: data)
        monkeypatch.setattr(replay_measure, "check_auto_route", lifecycle)
    iterations = 2 if automatic else 1
    variant = "auto-post-index" if automatic else "ready-batch4"
    result = replay_measure.measure(
        SimpleNamespace(
            __version__="test",
            npu=SimpleNamespace(Event=Event, NPUGraph=TorchGraph, graph=torch_capture),
            profiler=SimpleNamespace(record_function=lambda name: nullcontext()),
        ),
        SimpleNamespace(AclGraph=Graph),
        stream,
        bundle,
        SimpleNamespace(
            mode="fused",
            fused_torch_graph=torch_frontend,
            transport="urma",
            threshold_policy="recent",
            warmup=0,
            iterations=iterations,
            check_auto_route=automatic,
            profile_dir=tmp_path,
            profile_metrics=["none"],
            profile_graph_event=paired,
        ),
        {"progress": {"variant": variant}, "profile_metadata": {}},
    )
    assert result["timing"]["samples_us"] == [10.0] * iterations
    timed_then_profile = ["verify", "reset", *(["begin-3"] if automatic else []),
                          "sync", "start", *(["event"] if paired else []), "replay", *(["event"] if paired else []),
                          "sync", "stop", "export", "verify"]
    if paired:
        for i in range(4):
            timed_then_profile.extend(["reset", *([f"begin-{4+i}"] if automatic else []), "replay", "verify"])
    timed_then_profile.extend([*(["lifecycle"] if automatic else []), "graph_reset"])
    assert events[-len(timed_then_profile):] == timed_then_profile
    assert events.count("event") == 2 * (iterations + int(paired))
    if automatic:
        manifest = json.loads((tmp_path / f"urma-fused-{variant}/none/profile_manifest.json").read_text())
        assert manifest["experiment"]["auto_route"]["mode"] == "low" and manifest["experiment"]["auto_route"]["step"] == 3
        assert result["auto_route"]["low_steps"] == 2


def test_graph_events_bracket_exactly_the_exported_replay(tmp_path):
    events = []
    stream = object()

    def record(s):
        assert s is stream
        events.append("event")

    end = SimpleNamespace(record=record)

    def elapsed(e):
        assert e is end and events[-1] == "verify"
        return .401

    begin = SimpleNamespace(record=record, elapsed_time=elapsed)
    result = collect(tmp_path, events, metrics=["none"], graph_events=(begin, end, stream))
    assert events == ["reset", "sync", "start", "event", "replay", "event", "sync", "stop", "export", "verify"]
    assert result["passes"][0]["graph_event_us"] == 401
