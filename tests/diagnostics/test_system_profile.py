# SPDX-License-Identifier: Apache-2.0
"""System profiling must not include restoration or correctness readbacks."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from kvip.diagnostics import replay, replay_measure, system_profile


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


def test_measure_profiles_only_after_headline_and_before_graph_reset(tmp_path, monkeypatch):
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
    bundle = SimpleNamespace(
        reset=lambda: events.append("reset"),
        verify=lambda: events.append("verify"),
        forward=lambda: events.append("forward"),
        read_counts=lambda: {},
        lease=None,
    )
    result = replay_measure.measure(
        SimpleNamespace(
            __version__="test",
            npu=SimpleNamespace(Event=Event),
            profiler=SimpleNamespace(record_function=lambda name: nullcontext()),
        ),
        SimpleNamespace(AclGraph=Graph),
        SimpleNamespace(synchronize=lambda: events.append("sync")),
        bundle,
        SimpleNamespace(
            mode="fused",
            transport="urma",
            threshold_policy="recent",
            warmup=0,
            iterations=1,
            profile_dir=tmp_path,
            profile_metrics=["none"],
        ),
        {"progress": {"variant": "ready-batch4"}, "profile_metadata": {}},
    )
    assert result["timing"]["samples_us"] == [10.0]
    assert events[-10:] == [
        "verify",
        "reset",
        "sync",
        "start",
        "replay",
        "sync",
        "stop",
        "export",
        "verify",
        "graph_reset",
    ]
    assert events.count("event") == 2
