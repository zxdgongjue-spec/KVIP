# SPDX-License-Identifier: Apache-2.0
"""Production source/order/measurement contracts of the four-process harness."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kvip.diagnostics import pipeline_compare as compare
from kvip.diagnostics import replay
from kvip.diagnostics import replay_measure
from kvip.diagnostics.pipeline_options import DEFAULT_VARIANTS, options_for, q_workers_for, validate_options


def _arguments(tmp_path, monkeypatch, variants=None):
    monkeypatch.setattr(compare, "validate_checkout", lambda path, sha: Path(path).resolve())
    monkeypatch.setattr(compare, "source_commit", lambda: "4" * 40)
    build, snapshot = tmp_path / "build", tmp_path / "snapshot"
    (build / "bin").mkdir(parents=True, exist_ok=True)
    for name in ("kvaf_execute_batch_get_kernels.o", "kvaf_execute_batch_put_kernels.o"):
        (build / "bin" / name).write_bytes(b"fixture")
    snapshot.mkdir(exist_ok=True)
    (snapshot / "manifest.json").write_text("{}")
    argv = [
        "--snapshot",
        str(snapshot),
        "--build-dir",
        str(build),
        "--kvaf-server",
        "127.0.0.1:28999",
        "--result-dir",
        str(tmp_path / "results"),
        "--iterations",
        "3",
        "--warmup",
        "1",
    ]
    for role, sha in (("main", "1"), ("pr", "2"), ("candidate", "3")):
        argv += [f"--{role}-root", str(tmp_path / role), f"--{role}-sha", sha * 40]
    if variants is not None:
        argv += ["--pipeline-variants", *variants]
    return compare.arguments(argv)


def _case(args, name):
    role = "main" if name.startswith("main-") else "pr" if name == "pr-recent" else "candidate"
    mode = "serial" if name == "main-serial" else "fused"
    mean = 120 if mode == "serial" else 110 if name == "main-resident" else 100 if name == "pr-recent" else 90
    sample = dict(
        cold_miss_rate=0.1,
        cold_demand=4,
        selected_unique=40,
        prefetch=4 if mode == "fused" else 0,
        final=2 if mode == "fused" else 5,
        unused_prefetch=1 if mode == "fused" else 0,
        unused_prefetch_bytes=656 if mode == "fused" else 0,
        bytes=3936 if mode == "fused" else 3280,
        useful_transfer_bytes=3280,
    )
    result = dict(
        status="pass",
        transport="urma",
        mode=mode,
        snapshot_sha256=compare.file_digest(args.snapshot / "manifest.json"),
        geometry=dict(batch=16, token_bytes=656),
        scope="full-graph",
        timing_boundary="complete device chain",
        device=args.device,
        soc=args.soc,
        warmup=args.warmup,
        kvaf_build=str(args.build_dir),
        production_source=dict(
            kind="exact_upstream_checkout", root=str(getattr(args, role + "_root")), commit=getattr(args, role + "_sha")
        ),
        timing=dict(count=3, mean_us=mean, p50_us=mean, min_us=mean, max_us=mean, samples_us=[mean] * 3),
        reads_samples=[deepcopy(sample) for _ in range(3)],
        selection_comparison=dict(status="exact", selected_set_sha256="set", ordered_output_sha256="ordered"),
    )
    if mode == "serial":
        result.update(runtime_workers=32, batch_get_post_size=128, batch_get_max_inflight_total=2048, jettys_per_core=2)
        result["selection_comparison"].update(status="bf16_boundary_ties", selected_set_sha256="serial-set")
    else:
        result.update(
            score_comparison=dict(status="bit_exact"),
            runtime_workers=32,
            tuning=dict(recent_threshold=name != "main-resident", prefetch_limit=512, window_limit=64),
            threshold_comparison=dict(
                status="bit_exact",
                policy="resident" if name == "main-resident" else "recent",
                initial_keys=[[1 if name == "main-resident" else 2]],
                final_keys=[[3]],
            ),
        )
        if role == "candidate":
            result["tuning"].update(options_for(name.removeprefix("candidate-")))
        result["tuning"]["q_workers"] = q_workers_for(name.removeprefix("candidate-")) if role == "candidate" else 2
        result["native_diagnostics"] = dict(
            count=args.diagnostic_replays,
            included_in_timing=False,
            units="raw_device_cycles",
            samples=[
                dict(
                    metrics={"cq_drain": dict(status="available", samples_cycles=[10, 20, 30])}, raw={"fetch": [[123]]}
                )
                for _ in range(args.diagnostic_replays)
            ],
        )
    return result


def _reports(args):
    return {
        name: _case(args, name)
        for name in ["main-serial", "main-resident", "pr-recent", *("candidate-" + v for v in args.pipeline_variants)]
    }


def test_commands_match_actual_replay_parser_and_only_four_processes(tmp_path, monkeypatch):
    args = _arguments(tmp_path, monkeypatch)
    jobs = compare.jobs(args)
    assert [job["name"] for job in jobs] == ["main-serial", "main-resident", "pr-recent", "candidate"]
    for job in jobs:
        cmd = compare.command(args, job)
        parsed = replay.arguments(cmd[4:])
        assert parsed.upstream_root == getattr(args, job["role"] + "_root")
        assert parsed.upstream_revision == getattr(args, job["role"] + "_sha")
        assert parsed.transport == "urma" and parsed.threshold_policy == job["policy"]
        assert parsed.mode == job["mode"] and parsed.diagnostic_replays == int(job["mode"] == "fused")
        if job["mode"] == "serial":
            assert parsed.serial_workers == 32
        else:
            assert parsed.serial_workers is None
        if job["role"] == "candidate":
            assert parsed.study == "post-threshold"
            assert parsed.pipeline_variants == list(DEFAULT_VARIANTS)
        else:
            assert parsed.study == "threshold" and not parsed.pipeline_variants


def test_run_preserves_reverse_candidate_order_and_all_sources(tmp_path, monkeypatch, capsys):
    args = _arguments(tmp_path, monkeypatch, ["combined", "control"])
    calls = []
    reports = _reports(args)

    def child(cmd, **kwargs):
        parsed = replay.arguments(cmd[4:])
        calls.append(parsed)
        assert kwargs["env"]["PYTHONPATH"].split(":")[0] == str(Path(compare.__file__).resolve().parents[2])
        kwargs["stdout"].write("native success chatter\n")
        if parsed.study == "post-threshold":
            result = dict(
                status="pass",
                study="post-threshold",
                execution_order=args.pipeline_variants,
                cases={v: reports["candidate-" + v] for v in args.pipeline_variants},
            )
        else:
            name = (
                "main-serial"
                if parsed.mode == "serial"
                else "main-resident"
                if parsed.threshold_policy == "resident"
                else "pr-recent"
            )
            result = reports[name]
        parsed.output.write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compare.subprocess, "run", child)
    summary = compare.run(args)
    assert len(calls) == 4
    assert summary["candidate_shared_arena_order"] == ["combined", "control"]
    assert summary["execution_order"] == list(reports)
    assert summary["table"]["candidate-combined"]["gains"]["vs_pr"]["p50_reduction_pct"] == pytest.approx(10)
    assert summary["table"]["candidate-combined"]["gains"]["vs_serial"]["mean_reduction_pct"] == pytest.approx(25)
    assert summary["primary_candidate_comparison"] == "vs_control"
    assert summary["table"]["candidate-combined"]["gains"]["vs_control"]["p50_reduction_pct"] == 0
    assert summary["table"]["pr-recent"]["gains"]["vs_control"] is None
    assert summary["serial_fused_selected_sets_identical"] is False
    assert not list(args.result_dir.glob("*.log"))
    assert json.loads((args.result_dir / "invocation.json").read_text())["status"] == "pass"
    assert len(summary["result_files"]) == 4
    assert summary["table"]["candidate-control"]["production_source"]["commit"] == args.candidate_sha
    printed = capsys.readouterr().out
    assert "native success chatter" not in printed and "PIPELINE_COMPARE_PASS" in printed
    assert "candidate-combined -> candidate-control" in printed
    assert "vsControl_p50/mean%" in printed
    assert "TRAFFIC cold_miss%=10.0000 read_MiB=0.0038 unused_bytes=656.0 unused_MiB=0.0006" in printed
    assert "whole_graph_GBps=0.044" in printed and "not link bandwidth/utilization" in printed
    assert "get_drain=20" in printed and "do not sum" in printed
    assert json.loads((args.result_dir / "candidate.json").read_text())["cases"]["control"]["native_diagnostics"][
        "samples"
    ][0]["raw"] == {"fetch": [[123]]}


def test_child_failure_retains_log_and_emits_only_last_30_lines(tmp_path, monkeypatch, capsys):
    args = _arguments(tmp_path, monkeypatch)

    def child(cmd, **kwargs):
        for i in range(60):
            kwargs["stdout"].write(f"NATIVE_LOG_{i:02d}\n")
        return SimpleNamespace(returncode=70)

    monkeypatch.setattr(compare.subprocess, "run", child)
    with pytest.raises(SystemExit) as failure:
        compare.run(args)
    assert failure.value.code == 1
    printed = capsys.readouterr().out
    assert "NATIVE_LOG_29" not in printed and "NATIVE_LOG_30" in printed and "NATIVE_LOG_59" in printed
    assert (args.result_dir / "main-serial.log").is_file()
    invocation = json.loads((args.result_dir / "invocation.json").read_text())
    assert invocation["status"] == "failed" and invocation["failed_job"] == "main-serial"
    assert not (args.result_dir / "summary.json").exists()


@pytest.mark.parametrize(
    "change,error",
    [
        ("source", "exact checkout"),
        ("snapshot", "input or invocation"),
        ("threshold", "early threshold"),
        ("ordered", "ordered TopK"),
        ("score", "bit-exact"),
        ("budget", "prefetch budget"),
        ("option", "build options"),
        ("post", "post=128"),
        ("count", "incomplete or invalid"),
        ("mean", "retained samples"),
        ("scope", "scope differs"),
        ("workers", "q_workers differs"),
        ("profile", "diagnostic contract"),
    ],
)
def test_summary_rejects_unfair_or_invalid_results(tmp_path, monkeypatch, change, error):
    args = _arguments(tmp_path, monkeypatch, ["control", "combined"])
    reports = _reports(args)
    target = reports["candidate-combined"]
    if change == "source":
        target["production_source"]["commit"] = args.pr_sha
    elif change == "snapshot":
        target["snapshot_sha256"] = "different"
    elif change == "threshold":
        target["threshold_comparison"]["initial_keys"] = [[3]]
    elif change == "ordered":
        target["selection_comparison"]["ordered_output_sha256"] = "different"
    elif change == "score":
        target["score_comparison"]["status"] = "close"
    elif change == "budget":
        target["tuning"]["prefetch_limit"] = 1024
    elif change == "option":
        target["tuning"]["early_fetch_ready"] = False
    elif change == "post":
        reports["main-serial"]["batch_get_post_size"] = 32
    elif change == "count":
        target["timing"]["samples_us"].pop()
    elif change == "mean":
        target["timing"]["mean_us"] = 1
    elif change == "scope":
        target["scope"] = "fetch-only"
    elif change == "workers":
        target["tuning"]["q_workers"] = 4
    else:
        target["native_diagnostics"]["included_in_timing"] = True
    with pytest.raises(ValueError, match=error):
        compare.summarize(args, reports, compare.file_digest(args.snapshot / "manifest.json"))


def test_tail_samples_are_preserved_and_mean_p50_gains_differ(tmp_path, monkeypatch):
    args = _arguments(tmp_path, monkeypatch, ["control"])
    reports = _reports(args)
    reports["pr-recent"]["timing"].update(mean_us=400 / 3, p50_us=100, max_us=200, samples_us=[100, 100, 200])
    summary = compare.summarize(args, reports, compare.file_digest(args.snapshot / "manifest.json"))
    pr = summary["table"]["pr-recent"]
    assert pr["timing"]["samples_us"] == [100, 100, 200] and pr["timing_spread"]["flagged"]
    gains = summary["table"]["candidate-control"]["gains"]["vs_pr"]
    assert gains["p50_reduction_pct"] == pytest.approx(10)
    assert gains["mean_reduction_pct"] == pytest.approx(32.5)


def test_control_gain_uses_same_arena_timings_and_bandwidth_uses_complete_graph(tmp_path, monkeypatch):
    args = _arguments(tmp_path, monkeypatch, ["combined", "control"])
    reports = _reports(args)
    reports["candidate-control"]["timing"].update(
        mean_us=110, p50_us=100, min_us=90, max_us=140, samples_us=[90, 100, 140]
    )
    summary = compare.summarize(args, reports, compare.file_digest(args.snapshot / "manifest.json"))
    candidate = summary["table"]["candidate-combined"]
    assert candidate["gains"]["vs_control"]["p50_reduction_pct"] == pytest.approx(10)
    assert candidate["gains"]["vs_control"]["mean_reduction_pct"] == pytest.approx(100 * 20 / 110)
    assert candidate["gains"]["vs_pr"]["mean_reduction_pct"] == pytest.approx(10)
    assert candidate["whole_graph_GBps"] == pytest.approx(3936 / 90 / 1000)
    assert summary["table"]["main-serial"]["whole_graph_GBps"] == pytest.approx(3280 / 120 / 1000)


def test_no_control_does_not_substitute_cross_arena_pr(tmp_path, monkeypatch, capsys):
    args = _arguments(tmp_path, monkeypatch, ["combined"])
    reports = _reports(args)
    summary = compare.summarize(args, reports, compare.file_digest(args.snapshot / "manifest.json"))
    assert summary["primary_candidate_comparison"] is None
    assert all(row["gains"]["vs_control"] is None for row in summary["table"].values())
    compare.print_summary(summary)
    assert "n/a" in capsys.readouterr().out


def test_new_variants_worker_budget_and_control_end_drift(tmp_path, monkeypatch, capsys):
    args = _arguments(tmp_path, monkeypatch)
    reports = _reports(args)
    reports["candidate-control-end"]["timing"].update(
        mean_us=99, p50_us=99, min_us=99, max_us=99, samples_us=[99, 99, 99]
    )
    summary = compare.summarize(args, reports, compare.file_digest(args.snapshot / "manifest.json"))
    assert summary["control_end_drift"]["mean"]["drift_pct"] == pytest.approx(10)
    assert summary["control_end_drift"]["p50"]["delta_us"] == 9
    assert summary["table"]["candidate-workers4"]["gains"]["vs_control"]["mean_reduction_pct"] == 0
    assert summary["table"]["candidate-workers4"]["tuning"]["q_workers"] == 4
    assert summary["table"]["candidate-control"]["tuning"]["q_workers"] == 2
    assert summary["metadata_end_vs_control_end"]["mean"]["reduction_pct"] == pytest.approx(100 / 11)
    compare.print_summary(summary)
    printed = capsys.readouterr().out
    assert "CONTROL_DRIFT mean=+9.000us/+10.00% p50=+9.000us/+10.00% no_correction" in printed
    assert "END_PAIR metadata-end_vs_control-end mean=+9.09% p50=+9.09% no_drift_correction" in printed


def test_ready_batch_dependencies_and_worker_options_are_independent():
    expected = dict(
        final_topk_prune=False,
        selected_metadata=True,
        early_fetch_ready=True,
        compact_init=False,
        ready_batch_fetch=True,
    )
    assert options_for("ready-batch") == options_for("ready-batch4") == expected
    assert q_workers_for("ready-batch") == 2 and q_workers_for("ready-batch4") == 4
    assert options_for("workers4") == options_for("metadata")
    assert options_for("control-end") == options_for("control")
    assert options_for("metadata-end") == options_for("metadata")
    assert validate_options(expected) == expected
    with pytest.raises(ValueError, match="requires"):
        validate_options(expected | {"selected_metadata": False})
    with pytest.raises(ValueError, match="bool"):
        validate_options(expected | {"ready_batch_fetch": 1})


def test_study_delivers_each_worker_count_and_reuses_one_arena(monkeypatch):
    args = SimpleNamespace(
        study="post-threshold",
        mode="fused",
        transport="urma",
        threshold_policy="recent",
        pipeline_variants=list(DEFAULT_VARIANTS),
    )
    context, allocated, calls = object(), [], []
    monkeypatch.setattr(replay_measure, "allocate_context", lambda *args: allocated.append(context) or context)
    monkeypatch.setattr(replay_measure, "release_context", lambda value: allocated.remove(value))

    def measure(args, archive, passed_context, retained):
        assert passed_context is context and len(allocated) == 1
        name = retained["progress"]["variant"]
        calls.append((name, args.pipeline_q_workers, dict(args.pipeline_options)))
        return dict(
            timing={"p50_us": 10},
            selection_comparison={"ordered_output_sha256": "same"},
            threshold_comparison={"initial_keys": [10], "final_keys": [20]},
        )

    monkeypatch.setattr(replay_measure, "measure_case", measure)
    result = replay_measure.run(args, SimpleNamespace(manifest={}), {})
    assert not allocated
    assert result["execution_order"] == list(DEFAULT_VARIANTS)
    assert calls == [(name, q_workers_for(name), options_for(name)) for name in DEFAULT_VARIANTS]
    assert [row[1] for row in calls] == [2, 2, 2, 4, 4, 2, 2]


def test_wrong_candidate_order_or_missing_case_is_rejected():
    job = dict(name="candidate", role="candidate")
    report = dict(
        status="pass",
        study="post-threshold",
        execution_order=["control", "combined"],
        cases={"control": {}, "combined": {}},
    )
    with pytest.raises(ValueError, match="order"):
        compare.unpack_report(job, report, ["combined", "control"])
    report["execution_order"] = ["control"]
    with pytest.raises(ValueError, match="keys/order"):
        compare.unpack_report(job, report, ["control"])
