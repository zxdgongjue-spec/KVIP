# SPDX-License-Identifier: Apache-2.0
"""Reject mismatched real-snapshot comparisons and invalid timing evidence."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kvip.diagnostics.compare import (
    CASES,
    arguments,
    brief_report,
    case_plan,
    command,
    display,
    summarize,
    summarize_saved,
    stage_details,
    stage_payload_rates,
)
from kvip.diagnostics.replay_measure import fused_reads, statistics


def results():
    values = {}
    for transport, mode in CASES:
        latency = 200.0 if mode == "serial" else 100.0
        values[f"{transport}-{mode}"] = dict(
            transport=transport,
            mode=mode,
            status="pass",
            snapshot_sha256="capture_hash",
            geometry=dict(batch=16, token_bytes=2624),
            scope="one_group",
            soc="soc",
            device=0,
            kvip_commit="one_revision",
            runtime_workers=2,
            timing=dict(mean_us=latency, p50_us=latency),
            reads_samples=[dict(bytes=26240)],
        )
    return values


def test_comparison_uses_fixed_archive_and_reports_fusion_reduction():
    value = summarize(results())
    assert value["fusion_vs_serial"]["ubmem"] == dict(
        serial_case="ubmem-serial",
        serial_over_fused=2.0,
        latency_reduction_percent=50.0,
        p50_serial_over_fused=2.0,
        p50_latency_reduction_percent=50.0,
        target_fused_us=130.0,
        target_met=True,
    )
    assert value["table"]["urma-fused"]["mean_read_bytes"] == 26240


def test_stage_rates_use_instrumented_bytes_and_never_urma_cq_tail():
    entry = results()["ubmem-serial"]
    entry["stage_diagnostics"] = dict(
        stages={"batch_get": dict(mean_us=20.0), "resolve_final_fetch": dict(mean_us=40.0)},
        reads_samples=[dict(bytes=100000, final=10), dict(bytes=200000, final=30)],
    )
    assert stage_payload_rates(entry)["payload_GBps"] == 7.5
    entry["mode"] = "fused"
    rate = stage_payload_rates(entry)
    assert rate["mean_bytes"] == 20 * 2624
    assert rate["payload_GBps"] == 20 * 2624 / 40 / 1000
    assert "excludes LI prefetch" in rate["scope"]
    entry["transport"] = "urma"
    entry["stage_diagnostics"]["stages"]["final_fetch_wait"] = dict(mean_us=0.1)
    assert stage_payload_rates(entry) == {}
    entry["mode"] = "serial"
    entry["stage_diagnostics"]["reads_samples"] = []
    assert stage_payload_rates(entry) == {}


def test_compare_fused_variants_reuse_existing_replay_cli(tmp_path):
    value = json.dumps(
        [
            dict(name="before", tuning=dict(compact_reserved=True)),
            dict(name="after", tuning=dict(compact_reserved=True, balanced_index=True)),
        ]
    )
    args = arguments(cli(tmp_path) + ["--ubmem-fused-variants", value])
    argv, _ = command(args, "ubmem", "fused")
    assert argv[argv.index("--fused-variants") + 1] == value
    assert "--fused-tuning" not in argv
    assert "--quiet" in argv
    assert "--fused-variants" not in command(args, "urma", "fused")[0]
    assert "--fused-variants" not in command(args, "ubmem", "serial")[0]
    with pytest.raises(SystemExit):
        arguments(cli(tmp_path) + ["--ubmem-fused-variants", value, "--ubmem-fused-tuning", "{}"])
    with pytest.raises(SystemExit):
        arguments(cli(tmp_path) + ["--ubmem-fused-variants", value, "--urma-serial-workers-candidates", "2", "32"])


@pytest.mark.parametrize("dispatch", ("scalar", "tiled"))
def test_compare_routes_ubmem_dispatch_only_to_ubmem_serial(tmp_path, dispatch):
    args = arguments(cli(tmp_path) + ["--ubmem-serial-dispatch", dispatch])
    for transport, mode in CASES:
        argv, _ = command(args, transport, mode)
        if (transport, mode) == ("ubmem", "serial"):
            assert argv[argv.index("--ubmem-serial-dispatch") + 1] == dispatch
        else:
            assert "--ubmem-serial-dispatch" not in argv
    assert arguments(cli(tmp_path)).ubmem_serial_dispatch == "scalar"
    with pytest.raises(SystemExit):
        arguments(cli(tmp_path) + ["--ubmem-serial-dispatch", "other"])
    with pytest.raises(SystemExit):
        arguments(["--report-only", str(tmp_path / "summary.json"), "--ubmem-serial-dispatch", "tiled"])


def dispatch_options():
    return ["--urma-serial-workers", "32", "--urma-serial-post-size", "32", "--urma-serial-max-inflight", "0"]


def dispatch_results():
    entries = results()
    expanded = {}
    for transport in ("ubmem", "urma"):
        entry = entries.pop(f"{transport}-serial")
        for dispatch, latency in (
            ("scalar", 200.0),
            ("tiled" if transport == "ubmem" else "bulk", 160.0 if transport == "ubmem" else 150.0),
        ):
            result = copy.deepcopy(entry)
            result.update(runtime_workers=32, timing=dict(mean_us=latency, p50_us=latency + 1))
            result[transport + "_batch_get_dispatch"] = dispatch
            result["stage_diagnostics"] = dict(stages={"batch_get": statistics([latency / 2, latency / 2 + 2])})
            if transport == "urma":
                result.update(batch_get_post_size=32, batch_get_max_inflight_total=0, jettys_per_core=2)
            expanded[f"{transport}-serial-dispatch-{dispatch}"] = result
    return {**expanded, **entries}


def test_six_case_dispatch_plan_and_bulk_forwarding(tmp_path):
    args = arguments(cli(tmp_path) + dispatch_options() + ["--serial-dispatch-ablation"])
    plan = list(case_plan(args))
    assert plan == [
        ("ubmem", "serial", None, "scalar"),
        ("ubmem", "serial", None, "tiled"),
        ("urma", "serial", None, "scalar"),
        ("urma", "serial", None, "bulk"),
        ("ubmem", "fused", None, None),
        ("urma", "fused", None, None),
    ]
    paths = []
    for transport, mode, workers, dispatch in plan:
        argv, path = command(args, transport, mode, workers, dispatch)
        paths.append(path)
        if mode == "serial":
            assert argv[argv.index(f"--{transport}-serial-dispatch") + 1] == dispatch
            assert argv[argv.index("--serial-workers") + 1] == "32"
        else:
            assert not any("serial-dispatch" in value for value in argv)
        if transport == "urma" and mode == "serial":
            assert argv[argv.index("--urma-serial-post-size") + 1] == "32"
            assert argv[argv.index("--urma-serial-max-inflight") + 1] == "0"
    assert len(set(paths)) == 6
    args = arguments(cli(tmp_path) + dispatch_options() + ["--urma-serial-dispatch", "bulk"])
    assert len(list(case_plan(args))) == 4
    assert command(args, "urma", "serial")[0][-2:] == ["--urma-serial-dispatch", "bulk"]


@pytest.mark.parametrize(
    "extra",
    [
        ["--urma-serial-workers-candidates", "16", "32"],
        ["--ubmem-fused-variants", "[]"],
        ["--urma-fused-variants", "[]"],
        ["--ubmem-serial-dispatch", "tiled"],
        ["--urma-serial-dispatch", "bulk"],
    ],
)
def test_dispatch_ablation_rejects_ambiguous_options(tmp_path, extra):
    # Omit the singular worker option so the first case exercises the ablation
    # rejection rather than argparse's existing worker mutual-exclusion group.
    base = cli(tmp_path) + dispatch_options()[2:] + ["--serial-dispatch-ablation"]
    with pytest.raises(SystemExit):
        arguments(base + extra)


def test_bulk_and_dispatch_ablation_require_matching_post_and_inflight(tmp_path):
    for dispatch in (["--urma-serial-dispatch", "bulk"], ["--serial-dispatch-ablation"]):
        for incomplete in ([], ["--urma-serial-post-size", "32"], ["--urma-serial-max-inflight", "0"]):
            with pytest.raises(SystemExit):
                arguments(cli(tmp_path) + dispatch + incomplete)
    for extra in (["--serial-dispatch-ablation"], ["--urma-serial-dispatch", "bulk"]):
        with pytest.raises(SystemExit):
            arguments(["--report-only", str(tmp_path / "summary.json")] + extra)


def test_dispatch_summary_keeps_all_candidates_and_selects_each_strong_baseline(capsys):
    entries = dispatch_results()
    for result in entries.values():
        result["candidate_call"] = dict(wall=dict(mean_us=1.0))
    report = summarize(entries)
    assert set(report["table"]) == set(entries)
    assert report["serial_selection"]["ubmem"]["selected"] == "ubmem-serial-dispatch-tiled"
    assert report["serial_selection"]["urma"]["selected"] == "urma-serial-dispatch-bulk"
    assert report["urma_serial_selection"] == report["serial_selection"]["urma"]
    assert report["fusion_vs_serial"]["ubmem"]["target_fused_us"] == 104
    assert report["fusion_vs_serial"]["urma"]["target_fused_us"] == 97.5
    assert report["performance_target"]["target_met"] is False
    assert report["table"]["urma-serial-dispatch-bulk"]["serial_get_config"]["urma_batch_get_dispatch"] == "bulk"
    brief_report(report)
    output = capsys.readouterr().out
    assert output.count("SERIAL_CANDIDATE ") == 4
    assert "SERIAL_SELECTED ubmem=ubmem-serial-dispatch-tiled" in output
    assert "diagnostic_get_mean=81.000us" in output
    # A faster Get diagnostic does not win if its complete production graph is slower.
    entries["ubmem-serial-dispatch-tiled"]["timing"]["mean_us"] = 250
    assert summarize(entries)["serial_selection"]["ubmem"]["selected"] == "ubmem-serial-dispatch-scalar"


@pytest.mark.parametrize(
    "bad", ("snapshot_sha256", "status", "runtime_workers", "batch_get_post_size", "dispatch", "missing")
)
def test_dispatch_summary_rejects_bad_even_nonwinning_candidates(bad):
    entries = dispatch_results()
    candidate = entries["urma-serial-dispatch-scalar"]  # The slower candidate still must pass every check.
    if bad == "missing":
        del entries["urma-serial-dispatch-scalar"]
    elif bad == "dispatch":
        candidate["urma_batch_get_dispatch"] = "bulk"
    elif bad in ("runtime_workers", "batch_get_post_size"):
        candidate[bad] = 16
    else:
        candidate[bad] = "invalid"
    with pytest.raises(ValueError):
        summarize(entries)


def test_dispatch_ablation_main_runs_exactly_six_isolated_validated_cases(tmp_path, monkeypatch, capsys):
    import sys
    from kvip.diagnostics import compare

    entries, calls = dispatch_results(), []

    def child(argv, **kwargs):
        calls.append(argv)
        path = Path(argv[argv.index("--output") + 1])
        path.write_text(json.dumps(entries[path.stem]))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compare.subprocess, "run", child)
    monkeypatch.setattr(
        sys, "argv", ["compare", *cli(tmp_path), *dispatch_options(), "--serial-dispatch-ablation", "--brief"]
    )
    compare.main()
    assert len(calls) == 6
    assert [Path(call[call.index("--output") + 1]).stem for call in calls] == list(entries)
    for key in ("--snapshot", "--device", "--soc", "--warmup", "--iterations", "--stage-iterations"):
        assert len({call[call.index(key) + 1] for call in calls}) == 1
    report = json.loads((tmp_path / "results/summary.json").read_text())
    assert len(report["table"]) == 6
    assert report["fusion_vs_serial"]["urma"]["serial_case"] == "urma-serial-dispatch-bulk"
    output = capsys.readouterr().out
    assert output.count("CASE_BEGIN ") == output.count("CASE_PASS ") == 6
    assert output.count("SERIAL_SELECTED ") == 2


def test_saved_stage_details_use_samples_without_fabricating_missing_evidence():
    assert stage_details(dict(raw_samples=[10.0, 30.0])) == dict(
        p50_us=20.0, mean_us=20.0, min_us=10.0, max_us=30.0, count=2
    )
    assert stage_details(dict(mean_us=12.5)) == dict(p50_us=None, mean_us=12.5, min_us=None, max_us=None, count=None)
    for invalid in (dict(samples_us=[float("nan")]), dict(samples_us=[-1]), dict(samples_us=[1, 2], count=3)):
        with pytest.raises(ValueError):
            stage_details(invalid)


def test_saved_brief_details_include_every_variant_and_native_profile(tmp_path, monkeypatch, capsys):
    import sys
    from kvip.diagnostics import compare

    entries = results()
    paths = []
    for name, entry in entries.items():
        entry["stage_diagnostics"] = dict(
            stages={"batch_get": statistics([10, 30]), "native_li": dict(mean_us=45.0)},
            reads_samples=[dict(bytes=100000), dict(bytes=300000)],
        )
        if name == "urma-serial":
            cores = [
                dict(core=i, active_requests=1, post_active_cycles=100 + i, drain_wait_cycles=1000 + i)
                for i in range(4)
            ]
            entry["native_get_profile"] = dict(
                unit="native_cycles",
                included_in_headline=False,
                active_cores=4,
                totals=dict(
                    active_requests=4,
                    total_posts=4,
                    submitted_group_count=4,
                    completed_group_count=4,
                    completion_count=4,
                    post_active_cycles=406,
                    drain_wait_cycles=4006,
                ),
                maxima=dict(post_active_cycles=103, drain_wait_cycles=1003),
                cores=cores,
            )
        report = entry
        if name.endswith("fused"):
            before = dict(entry, name="before")
            after = copy.deepcopy(dict(entry, name="after"))
            after["stage_diagnostics"] = dict(stages={"init": dict(raw_samples=[12.0, 16.0])})
            report = dict(status="pass", cases=[before, after])
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    summary = summarize_saved(paths)
    path = tmp_path / "summary.json"
    saved = json.dumps(summary)
    path.write_text(saved)

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved stage details must not start device work")

    monkeypatch.setattr(compare.subprocess, "run", forbidden)
    monkeypatch.setattr(sys, "argv", ["compare", "--report-only", str(path), "--brief"])
    compare.main()
    output = capsys.readouterr().out
    for name in (
        "ubmem-serial",
        "urma-serial",
        "ubmem-fused/before",
        "ubmem-fused/after",
        "urma-fused/before",
        "urma-fused/after",
    ):
        assert f"STAGES {name} (diagnostic us)" in output
    rows = [line.split() for line in output.splitlines()]
    assert ["batch_get", "20.000", "20.000", "10.000", "30.000", "2"] in rows
    assert ["init", "14.000", "14.000", "12.000", "16.000", "2"] in rows
    assert ["native_li", "--", "45.000", "--", "--", "--"] in rows
    assert "STAGE_PAYLOAD urma-serial batch_get=20.000us bytes=200000 GB/s=10.000" in output
    assert "post_active_sum/max=406/103 drain_wait_sum/max=4006/1003" in output
    assert "requests=4 logical_posts=4" in output
    assert "bulk logical posts are not doorbell counts" in output
    slow = [line for line in output.splitlines() if "slowest_by_drain" in line]
    assert len(slow) == 3 and "core=3" in slow[0] and "core=1" in slow[2]
    assert "not additive wall latency or us" in output
    assert path.read_text() == saved
    assert "torch_npu" not in sys.modules


@pytest.mark.parametrize("bad_candidate", (None, "kvip_commit", "status", "scope", "mean_us"))
def test_saved_comparison_selects_validated_winner_and_rejects_mixed_nonwinner(tmp_path, bad_candidate):
    entries = results()
    paths = []
    for name in ("ubmem-serial", "urma-serial", "ubmem-fused", "urma-fused"):
        case = entries[name]
        report = case
        if name.endswith("-fused"):
            slow = dict(case, name="baseline", timing=dict(mean_us=110.0, p50_us=109.0))
            fast = dict(case, name="candidate")
            if bad_candidate is not None and name == "urma-fused":
                if bad_candidate == "mean_us":
                    slow["timing"]["mean_us"] = float("nan")
                else:
                    slow[bad_candidate] = "invalid"
            report = dict(status="pass", cases=[slow, fast])
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(report))
        paths.append(path)
    if bad_candidate is not None:
        with pytest.raises(ValueError):
            summarize_saved(paths)
    else:
        summary = summarize_saved(paths)
        assert summary["fusion_vs_serial"]["ubmem"]["serial_over_fused"] == 2.0
        assert summary["saved_result_selection"]["urma-fused"]["selected"] == "candidate"
        assert len(summary["saved_result_selection"]["urma-fused"]["cases"]) == 2


def test_saved_comparison_cli_does_not_launch_or_import_runtime(tmp_path, monkeypatch, capsys):
    import sys
    from kvip.diagnostics import compare

    paths = []
    entries = results()
    for name in ("ubmem-serial", "urma-serial", "ubmem-fused", "urma-fused"):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(entries[name]))
        paths.append(str(path))
    output = tmp_path / "summary.json"

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved summary must not run a device process")

    monkeypatch.setattr(compare.subprocess, "run", forbidden)
    monkeypatch.setattr(sys, "argv", ["compare", "--from-results", *paths, "--output", str(output), "--quiet"])
    compare.main()
    assert "SELECTED ubmem-fused=ubmem-fused" in capsys.readouterr().out
    assert json.loads(output.read_text())["performance_target"]["target_met"] is True
    assert "torch_npu" not in sys.modules
    with pytest.raises(SystemExit):
        arguments(["--from-results", *paths, "--output", str(output)])


def test_comparison_preserves_boundary_tie_and_actual_read_evidence():
    entries = results()
    selection = dict(
        status="bf16_boundary_ties",
        selected_set_sha256="native_selection",
        changed_queries=2,
        total_replacements=3,
        min_overlap_fraction=2046 / 2048,
    )
    entries["ubmem-serial"]["selection_comparison"] = selection
    entries["urma-serial"]["selection_comparison"] = {**selection, "selected_set_sha256": "another_selection"}
    entries["ubmem-serial"]["reads_samples"] = [dict(bytes=30000), dict(bytes=32000)]
    value = summarize(entries)
    assert value["table"]["ubmem-serial"]["selection"] == selection
    assert value["table"]["ubmem-serial"]["mean_read_bytes"] == 31000
    assert value["same_selection_across_transports"] == dict(serial=False)


def test_brief_report_preserves_actual_path_demand_and_score_validation(tmp_path, monkeypatch, capsys):
    import sys
    from kvip.diagnostics import compare

    entries = results()
    for case in entries.values():
        case["geometry"]["token_bytes"] = 656
        fused = case["mode"] == "fused"
        case["reads_samples"] = [
            dict(
                bytes=6560,
                cold_demand=7 if fused else 8,
                selected_unique=100,
                candidate_hot_refetch=2,
                hot_hit=81 if fused else 80,
                abc_hit=10,
                selected_read_demand=9 if fused else 10,
            )
        ]
        if fused:
            case["score_comparison"] = dict(status="bit_exact", compared_scores=100, included_in_timing=False)
    summary = summarize(entries)
    assert set(summary["score_validation"]) == {"ubmem-fused", "urma-fused"}
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved brief report must not run a device process")

    monkeypatch.setattr(compare.subprocess, "run", forbidden)
    monkeypatch.setattr(sys, "argv", ["compare", "--report-only", str(path), "--brief"])
    compare.main()
    output = capsys.readouterr().out
    assert "batch=16 token_bytes=656" in output
    assert "8/100   8.0000" in output
    assert "7/100   7.0000" in output
    assert "ubmem-fused=bit_exact urma-fused=bit_exact" in output
    assert "prefetch" not in output
    assert "torch_npu" not in sys.modules
    with pytest.raises(ValueError):
        brief_report({**summary, "status": "failed"})


def test_comparison_keeps_admission_vectors_and_their_accounting_scope(capsys):
    import numpy as np
    from kvip.diagnostics.prefetch_admission import ADMISSION_ACCOUNTING, prefetch_admission

    entries = results()
    row = np.zeros((1, 256), dtype=np.int32)
    row[0, [8, 9, 16]] = 4
    row[0, [10, 48]] = 3
    row[0, 208:211] = (8192, 8, 2048)
    row[0, 214:216] = (1, 2)
    row[0, 246] = 2
    counts = prefetch_admission(row, window_limit=4, rounds=2)
    for name in ("ubmem-fused", "urma-fused"):
        entries[name]["reads_samples"][0]["prefetch_admission"] = counts
    original = copy.deepcopy(entries)
    report = summarize(entries)
    assert entries == original
    assert report["table"]["ubmem-fused"]["prefetch_admission"]["posted_by_epoch"] == [4.0, 0.0]
    assert report["table"]["urma-serial"]["prefetch_admission"] is None
    assert report["prefetch_admission_accounting"] == ADMISSION_ACCOUNTING
    display(report)
    lines = [line for line in capsys.readouterr().out.splitlines() if "admission/round" in line]
    assert len(lines) == 2 and all("sent=[4,0] suppressed=[3,0]" in line for line in lines)


def test_demand_is_separate_from_observed_transfers_and_rates_name_the_denominator(capsys):
    entries = results()
    width = 2624
    for result in entries.values():
        fused = result["mode"] == "fused"
        result["reads_samples"] = [
            dict(
                prefetch=10 if fused else 0,
                final=5 if fused else 10,
                transfers=15 if fused else 10,
                unused_prefetch=3 if fused else 0,
                bytes=(15 if fused else 10) * width,
                useful_transfers=12 if fused else 10,
                useful_transfer_bytes=(12 if fused else 10) * width,
                selected_unique=100,
                cold_demand=7 if fused else 8,
                candidate_hot_refetch=2,
                hot_hit=81 if fused else 80,
                abc_hit=10,
                selected_read_demand=9 if fused else 10,
                cold_miss_rate=0.07 if fused else 0.08,
            )
        ]
    original = copy.deepcopy(entries)
    report = summarize(entries)
    fused = report["table"]["ubmem-fused"]
    assert fused["demand_per_round"]["cold_demand"] == 7
    assert fused["demand_per_round"]["cold_miss_rate"] == 7 / 100
    assert fused["reads_per_round"]["useful_transfers"] == 12  # Not 7 or 7+2.
    assert fused["mean_useful_transfer_bytes"] == 12 * width
    assert fused["graph_normalized_payload_GBps"] == dict(
        actual_transfers=15 * width / 100000,
        useful_transfers=12 * width / 100000,
        cold_demand=7 * width / 100000,
    )
    assert report["demand_delta_fused_minus_serial"]["ubmem"]["cold_demand"] == -1
    assert entries == original
    display(report)
    output = capsys.readouterr().out
    assert "cold_demand / selected_unique" in output
    assert "runtime_workers" in output
    assert "not Get-stage or hardware bandwidth" in output
    assert "potential in fusion" in output


@pytest.mark.parametrize(
    "key,value",
    [
        ("snapshot_sha256", "other"),
        ("geometry", dict(batch=1, token_bytes=656)),
        ("scope", "fetch_only"),
        ("device", 1),
        ("status", "failed"),
        ("kvip_commit", "other"),
    ],
)
def test_comparison_rejects_mixed_evidence(key, value):
    entries = results()
    entries["urma-fused"][key] = value
    with pytest.raises(ValueError):
        summarize(entries)


def test_command_never_loads_two_transports_in_one_process(tmp_path):
    args = SimpleNamespace(
        snapshot=Path("/snapshot"),
        ubmem_build_dir=Path("/ubmem/build"),
        urma_build_dir=Path("/urma/build"),
        kvaf_server="127.0.0.1:28999",
        device=0,
        soc="soc",
        iterations=10,
        warmup=2,
        stage_iterations=3,
        result_dir=tmp_path,
        ubmem_serial_workers=32,
        urma_serial_workers=2,
        split_index_stages=True,
    )
    for transport, mode in CASES:
        argv, output = command(args, transport, mode)
        assert argv[argv.index("--build-dir") + 1] == f"/{transport}/build"
        assert ("--kvaf-server" in argv) == (transport == "urma")
        assert ("--serial-workers" in argv) == (mode == "serial")
        assert ("--split-index-stages" in argv) == (mode == "fused")
        assert output.name == f"{transport}-{mode}.json"


def cli(tmp_path):
    return [
        "--snapshot",
        "/snapshot",
        "--ubmem-build-dir",
        "/ubmem",
        "--urma-build-dir",
        "/urma",
        "--kvaf-server",
        "127.0.0.1:28880",
        "--soc",
        "soc",
        "--result-dir",
        str(tmp_path / "results"),
    ]


def test_compare_five_cases_preserve_candidates_and_forward_only_backend_tuning(tmp_path):
    args = arguments(
        cli(tmp_path)
        + [
            "--urma-serial-workers-candidates",
            "2",
            "8",
            "--ubmem-fused-tuning",
            '{"grouped_copy":true,"balanced_index":true}',
            "--urma-fused-tuning",
            '{"compact_reserved":true,"prefetch_limit":256,"window_limit":32}',
        ]
    )
    plan = list(case_plan(args))
    assert len(plan) == 5
    assert plan[1:3] == [("urma", "serial", 2, None), ("urma", "serial", 8, None)]
    outputs = []
    for transport, mode, workers, dispatch in plan:
        argv, output = command(args, transport, mode, workers, dispatch)
        outputs.append(output.name)
        if mode == "fused":
            import json

            assert json.loads(argv[argv.index("--fused-tuning") + 1]) == getattr(args, transport + "_fused_tuning")
        else:
            assert "--fused-tuning" not in argv
            assert int(argv[argv.index("--serial-workers") + 1]) == (workers or 32)
    assert outputs[1:3] == ["urma-serial-workers2.json", "urma-serial-workers8.json"]
    assert len(list(case_plan(arguments(cli(tmp_path))))) == 4


@pytest.mark.parametrize(
    "extra",
    [
        ["--urma-serial-workers-candidates", "2", "2"],
        ["--urma-serial-workers-candidates", "2", "49"],
        ["--urma-serial-workers-candidates", "2", "8", "16"],
        ["--urma-serial-workers", "2", "--urma-serial-workers-candidates", "2", "8"],
        ["--urma-fused-tuning", '{"grouped_copy":true}'],
        ["--ubmem-fused-tuning", "[]"],
    ],
)
def test_compare_rejects_unsupported_or_ambiguous_configuration(tmp_path, extra):
    with pytest.raises(SystemExit):
        arguments(cli(tmp_path) + extra)


def test_fastest_correct_serial_sets_a_stricter_target_and_wall_never_enters_ratio(capsys):
    entries = results()
    baseline = entries.pop("urma-serial")
    entries["urma-serial-workers2"] = baseline
    entries["urma-serial-workers8"] = dict(baseline, runtime_workers=8, timing=dict(mean_us=140.0, p50_us=145.0))
    for result in entries.values():
        result["candidate_call"] = dict(wall=dict(mean_us=1.0))
    report = summarize(entries)
    assert report["status"] == "pass"  # Numerical correctness never implies the performance goal.
    assert report["fusion_vs_serial"]["urma"]["serial_case"] == "urma-serial-workers8"
    assert report["fusion_vs_serial"]["urma"]["target_fused_us"] == 91
    assert report["fusion_vs_serial"]["urma"]["target_met"] is False
    assert report["performance_target"]["target_met"] is False
    assert report["urma_serial_selection"]["improvement_over_first_percent"] == pytest.approx(30)
    assert set(report["table"]) == set(entries)
    display(report)
    assert "NOT MET" in capsys.readouterr().out
    entries["urma-serial-workers8"]["status"] = "failed"
    with pytest.raises(ValueError, match="did not pass"):
        summarize(entries)


@pytest.mark.parametrize("fused,met", [(130.0, True), (130.001, False), (200 / 1.5, False)])
def test_both_performance_targets_apply_to_device_mean(fused, met):
    entries = results()
    entries["ubmem-fused"]["timing"] = dict(mean_us=fused, p50_us=1.0)
    assert summarize(entries)["fusion_vs_serial"]["ubmem"]["target_met"] is met


def test_urma_serial_inflight_is_forwarded_and_reported_without_touching_other_paths(tmp_path, capsys):
    args = arguments(cli(tmp_path) + ["--urma-serial-workers", "32", "--urma-serial-max-inflight", "0"])
    for transport, mode, workers, dispatch in case_plan(args):
        argv, _ = command(args, transport, mode, workers, dispatch)
        if (transport, mode) == ("urma", "serial"):
            assert argv[argv.index("--urma-serial-max-inflight") + 1] == "0"
        else:
            assert "--urma-serial-max-inflight" not in argv
    entries = results()
    entries["urma-serial"].update(
        batch_get_max_inflight_total=0, batch_get_max_inflight_total_source="cli", jettys_per_core=2
    )
    summary = summarize(entries)
    assert summary["table"]["urma-serial"]["serial_get_config"] == dict(
        batch_get_max_inflight_total=0,
        batch_get_max_inflight_total_source="cli",
        jettys_per_core=2,
    )
    display(summary)
    assert '"batch_get_max_inflight_total": 0' in capsys.readouterr().out
    assert arguments(cli(tmp_path)).urma_serial_max_inflight is None
    for extra in (
        ["--urma-serial-max-inflight", "-1"],
        ["--urma-serial-max-inflight", str(1 << 32)],
        ["--urma-serial-workers", "32", "--urma-serial-max-inflight", "63"],
        ["--urma-serial-workers-candidates", "16", "32", "--urma-serial-max-inflight", "32"],
    ):
        with pytest.raises(SystemExit):
            arguments(cli(tmp_path) + extra)


def test_post_size_cli_scope_and_transport_specific_forwarding(tmp_path):
    from kvip.diagnostics.replay import arguments as replay_arguments

    args = arguments(cli(tmp_path) + ["--urma-serial-post-size", "32"])
    for transport, mode, workers, dispatch in case_plan(args):
        argv, _ = command(args, transport, mode, workers, dispatch)
        if (transport, mode) == ("urma", "serial"):
            assert argv[argv.index("--urma-serial-post-size") + 1] == "32"
        else:
            assert "--urma-serial-post-size" not in argv
    common = ["--snapshot", "/snapshot", "--build-dir", "/build", "--soc", "soc"]
    assert replay_arguments(common).urma_serial_post_size == 128
    serial = common + ["--transport", "urma", "--mode", "serial", "--kvaf-server", "127.0.0.1:28880"]
    assert replay_arguments(serial + ["--urma-serial-post-size", "32"]).urma_serial_post_size == 32
    for base, size in ((common, "32"), (common, "128"), (serial, "64"), (serial + ["--inspect"], "32")):
        with pytest.raises(SystemExit):
            replay_arguments(base + ["--urma-serial-post-size", size])


@pytest.mark.parametrize("samples", [[], [0], [-1], [float("nan")], [float("inf")], [[1]]])
def test_invalid_device_timing_is_never_a_performance_result(samples):
    with pytest.raises(ValueError):
        statistics(samples)


def test_urma_completion_and_speculative_read_accounting():
    import torch

    rows = torch.zeros((1, 256), dtype=torch.int32)
    rows[0, 9], rows[0, 247] = 2, 1
    completion = torch.zeros((2, 1216), dtype=torch.uint8)
    completion.view(torch.int64)[:, 0] = 0x4B5641464247444F
    mapping = torch.tensor([[4, -1, 5, 4 + 8192]], dtype=torch.int32)
    topk = torch.tensor([[[0, 3]]], dtype=torch.int32)
    rotating = SimpleNamespace(
        rows=rows,
        row_status=torch.zeros((1, 32), dtype=torch.int32),
        final_posts=torch.zeros((1, 4, 32), dtype=torch.int32),
        prefetch_status=completion,
        final_status=completion.clone(),
        mapping=mapping,
        arguments=[None] * 9 + [topk],
        hot_valid=torch.ones((1, 4), dtype=torch.int32),
    )
    update = SimpleNamespace(status=torch.zeros(1, dtype=torch.int32))
    read = fused_reads(torch, rotating, update, "urma", 2624)
    assert read == dict(
        prefetch=2,
        final=1,
        transfers=3,
        bytes=7872,
        unused_prefetch=1,
        unused_prefetch_bytes=2624,
        useful_transfers=2,
        useful_transfer_bytes=5248,
    )
    bad = copy.copy(rotating)
    bad.final_status = completion.clone()
    bad.final_status.view(torch.int64)[0, 2] = 1
    with pytest.raises(AssertionError, match="URMA completion"):
        fused_reads(torch, bad, update, "urma", 2624)
