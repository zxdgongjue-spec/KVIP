# SPDX-License-Identifier: Apache-2.0
"""Skip ablation preserves inputs, native switches and the current serial baseline."""

from copy import deepcopy
import ctypes
import json
from types import SimpleNamespace

import pytest

from kvip.diagnostics import compare, replay, replay_measure
from kvip.diagnostics.fused_replay import validate_fused_tuning, verify_build_options


def _case(mean, tuning=None):
    return dict(
        status="pass",
        snapshot_sha256="same",
        geometry={},
        scope="full",
        soc="A5",
        device=0,
        reads_samples=[
            dict(
                cold_miss_rate=0.1,
                cold_demand=2,
                selected_unique=20,
                prefetch=2,
                final=1,
                unused_prefetch=1,
                unused_prefetch_bytes=656,
                bytes=1968,
                useful_transfer_bytes=1312,
            )
        ],
        timing=dict(mean_us=mean, p50_us=mean),
        runtime_workers=32,
        selection_comparison=dict(selected_set_sha256="ids", ordered_output_sha256="ordered"),
        threshold_comparison=dict(initial_keys=[[40000, 40001]], final_keys=[[40002, 40003]]),
        score_comparison=dict(status="bit_exact", compared_scores=1048576, included_in_timing=False),
        tuning=tuning,
        timing_boundary="complete graph",
    )


def _reports():
    result = {}
    for transport in ("urma", "ubmem"):
        result[f"{transport}-serial"] = _case(120)
        for name, tuning in replay_measure.study_cases(SimpleNamespace(study="skip")).items():
            result[f"{transport}-fused-{name}"] = _case(
                100 if name == "before" else 90, tuning | dict(prefetch_limit=512, window_limit=64)
            )
    return result


def test_skip_plan_keeps_old_pipeline_study_and_threshold_defaults_explicit():
    cases = replay_measure.study_cases(SimpleNamespace(study="skip"))
    assert list(cases) == ["before", "after"]
    assert cases["before"] == dict(
        recent_threshold=True, recent_pipeline=True, recent_unified_topk=True, recent_skip=False
    )
    assert cases["before"] | {"recent_skip": True} == cases["after"]
    pipeline = replay_measure.study_cases(SimpleNamespace(study="pipeline"))
    assert [(v["recent_pipeline"], v["recent_unified_topk"]) for v in pipeline.values()] == [
        (False, False),
        (True, False),
        (True, True),
    ]
    assert all(v["recent_skip"] is False for v in pipeline.values())
    assert all(v["recent_skip"] is True for v in replay_measure.study_cases(SimpleNamespace()).values())


@pytest.mark.parametrize("transport", ["urma", "ubmem"])
def test_skip_native_getter_is_required_and_checked(transport):
    options = dict(recent_pipeline=True, recent_unified_topk=True, recent_skip=True)
    assert validate_fused_tuning(options, transport) == options
    functions = {"kvip_" + name: lambda: 1 for name in options}
    library = SimpleNamespace(**functions)
    verify_build_options(library, options)
    assert library.kvip_recent_skip.argtypes == []
    assert library.kvip_recent_skip.restype is ctypes.c_uint32
    library.kvip_recent_skip = lambda: 0
    with pytest.raises(RuntimeError, match="recent_skip=True"):
        verify_build_options(library, options)
    del library.kvip_recent_skip
    with pytest.raises(AttributeError, match="kvip_recent_skip"):
        verify_build_options(library, options)


@pytest.mark.parametrize("bad", [0, 1, "true", None])
def test_skip_tuning_requires_a_boolean(bad):
    with pytest.raises(ValueError, match="boolean"):
        validate_fused_tuning({"recent_skip": bad}, "ubmem")


def test_skip_reuses_one_runtime_and_publishes_only_after_both_cases(monkeypatch):
    calls, context, archive = [], object(), object()
    args = SimpleNamespace(study="skip", mode="fused", transport="ubmem", threshold_policy="resident")

    def allocate(received, captured, retained):
        assert received is args and captured is archive
        calls.append("allocate")
        return context

    def measure(received, captured, shared, retained):
        assert received is args and captured is archive and shared is context
        calls.append(deepcopy(received.fused_tuning))
        return _case(100, deepcopy(received.fused_tuning))

    monkeypatch.setattr(replay_measure, "allocate_context", allocate)
    monkeypatch.setattr(replay_measure, "measure_case", measure)
    monkeypatch.setattr(replay_measure, "release_context", lambda shared: calls.append(("release", shared)))
    result = replay_measure.run(args, archive, {})
    assert list(result["cases"]) == ["before", "after"]
    assert result["study"] == "skip"
    assert calls == ["allocate", *replay_measure.study_cases(args).values(), ("release", context)]


def test_skip_failure_keeps_context_for_existing_process_exit_path(monkeypatch):
    args = SimpleNamespace(study="skip", mode="fused", transport="urma", threshold_policy="both")
    released = []
    monkeypatch.setattr(replay_measure, "allocate_context", lambda *unused: object())

    def measure(received, *unused):
        result = _case(100)
        if received.fused_tuning["recent_skip"]:
            result["threshold_comparison"]["initial_keys"][0][0] += 1
        return result

    monkeypatch.setattr(replay_measure, "measure_case", measure)
    monkeypatch.setattr(replay_measure, "release_context", released.append)
    with pytest.raises(AssertionError, match="threshold"):
        replay_measure.run(args, object(), {})
    assert released == []


def test_skip_summary_reports_six_paths_and_allows_real_traffic_changes(capsys):
    reports = _reports()
    reports["ubmem-fused-after"]["reads_samples"][0]["prefetch"] = 1
    reports["ubmem-fused-after"]["reads_samples"][0]["final"] = 2
    summary = compare.summarize(reports, "skip")
    assert len(summary["table"]) == 6
    assert summary["effects"]["ubmem"]["skip_reduction_pct"] == pytest.approx(10)
    assert summary["effects"]["ubmem"]["skip_vs_serial_reduction_pct"] == pytest.approx(25)
    assert summary["table"]["ubmem-fused-after"]["reads"]["final"] == 2
    compare.print_summary(summary)
    output = capsys.readouterr().out
    assert "skip=+10.00%/+10.00%" in output and "skip_vs_serial=+25.00%/+25.00%" in output
    assert "final_topk=unchanged_ordered" in output


@pytest.mark.parametrize("field", ["initial_keys", "final_keys"])
def test_skip_rejects_changed_thresholds(field):
    reports = _reports()
    reports["urma-fused-after"]["threshold_comparison"][field][0][0] += 1
    with pytest.raises(ValueError, match="threshold"):
        compare.summarize(reports, "skip")


@pytest.mark.parametrize("field", ["selected_set_sha256", "ordered_output_sha256"])
def test_skip_rejects_changed_topk(field):
    reports = _reports()
    reports["ubmem-fused-after"]["selection_comparison"][field] = "changed"
    with pytest.raises(ValueError, match="selection"):
        compare.summarize(reports, "skip")


@pytest.mark.parametrize("name", ["before", "after"])
@pytest.mark.parametrize("validation", [None, {"status": "unverified"}])
def test_skip_requires_actual_full_score_validation(name, validation):
    reports = _reports()
    reports[f"ubmem-fused-{name}"]["score_comparison"] = validation
    with pytest.raises(ValueError, match="captured valid LI scores"):
        compare.summarize(reports, "skip")


@pytest.mark.parametrize("field,value", [("recent_pipeline", False), ("recent_skip", False), ("window_limit", 32)])
def test_skip_summary_rejects_wrong_binary_settings_or_budget(field, value):
    reports = _reports()
    reports["ubmem-fused-after"]["tuning"][field] = value
    with pytest.raises(ValueError, match="Skip A/B"):
        compare.summarize(reports, "skip")


def test_skip_cli_runs_four_children_and_preserves_serial_commands(tmp_path, monkeypatch, capsys):
    build, snapshot = tmp_path / "build", tmp_path / "snapshot"
    (build / "bin").mkdir(parents=True)
    (build / "bin/kvaf_execute_batch_get_kernels.o").touch()
    snapshot.mkdir()
    (snapshot / "manifest.json").write_text("{}")
    args = compare.arguments(
        [
            "--snapshot",
            str(snapshot),
            "--build-dir",
            str(build),
            "--kvaf-server",
            "127.0.0.1:28999",
            "--study",
            "skip",
            "--transports",
            "both",
            "--result-dir",
            str(tmp_path / "results"),
        ]
    )
    calls, reports = [], _reports()

    def child(command, **kwargs):
        calls.append(command)
        transport, mode = (command[command.index("--" + key) + 1] for key in ("transport", "mode"))
        output = command[command.index("--output") + 1]
        if mode == "serial":
            result = reports[f"{transport}-serial"]
            assert command[command.index("--serial-workers") + 1] == "32"
            assert command[command.index("--threshold-policy") + 1] == "resident"
        else:
            result = {"cases": {name: reports[f"{transport}-fused-{name}"] for name in ("before", "after")}}
            assert "--serial-workers" not in command
        with open(output, "w") as stream:
            json.dump(result, stream)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compare.subprocess, "run", child)
    monkeypatch.setattr(compare, "source_commit", lambda: "current")
    result = compare.run(args)
    assert len(calls) == 4 and len(result["table"]) == 6
    assert all(command[command.index("--study") + 1] == "skip" for command in calls)
    assert len(list(args.result_dir.glob("*.json"))) == 5
    assert "before/after share DDR" in capsys.readouterr().out
    parsed = replay.arguments(
        [
            "--snapshot",
            str(snapshot),
            "--build-dir",
            str(build),
            "--transport",
            "ubmem",
            "--mode",
            "fused",
            "--study",
            "skip",
            "--kvaf-server",
            "127.0.0.1:28999",
            "--soc",
            "A5",
            "--output",
            str(tmp_path / "one.json"),
        ]
    )
    assert parsed.study == "skip"
