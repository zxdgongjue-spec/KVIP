# SPDX-License-Identifier: Apache-2.0
import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import fixed_q4_a5_12 as runner
from fixed_q4_a5_12 import plan, summarize_cell, write_summary


@pytest.mark.parametrize("options, kind, group", [((), "real", 2), (("--synthetic",), "synthetic", 0)])
def test_dry_run_needs_no_a5_dependencies(options, kind, group):
    script = Path(__file__).resolve().parent / "support/fixed_q4_a5_12.py"
    result = subprocess.check_output([sys.executable, "-B", script, "--dry-run", *options], text=True)
    data = json.loads(result)
    assert data["batch_sizes"] == [1, 2, 4, 8, 16, 20, 24, 28, 32]
    assert data["group"] == group
    assert data["input_kind"] == kind
    assert data["cohort"] == 0
    assert data["blocks"] == [["serial", "fused"], ["fused", "serial"]]
    assert data["oracles"] == 9
    assert data["measured_graphs"] == 36
    assert data["calls_per_graph"] == 10
    assert data["graph_warmups"] == 4
    assert data["prefetch"] == {"tokens": 1024, "windows_requested": 4, "rank_start": 1024}
    if kind == "synthetic":
        assert data["synthetic_fixture_config"] == {
            "history": 131072, "resident": 16384, "miss_percent": 10, "index_heads": 32, "seed": 43,
        }


def test_synthetic_and_trace_are_exclusive():
    script = Path(__file__).resolve().parent / "support/fixed_q4_a5_12.py"
    result = subprocess.run(
        [sys.executable, "-B", script, "--synthetic", "--trace", "/unused", "--dry-run"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2 and "not allowed with argument" in result.stderr


def test_synthetic_prepares_once_and_uses_fresh_oracles(tmp_path, monkeypatch):
    kvaf = tmp_path / "kvaf"
    cpu = kvaf / "build-cpu/runtime/kvaf_cpu_service"
    cpu.parent.mkdir(parents=True)
    cpu.touch()
    cpu.chmod(0o700)
    out = tmp_path / "out"
    args = SimpleNamespace(kvaf_root=kvaf, cpu_numa=None, cpu_ub_device=None, device=2, physical_npu=2,
                           synthetic=True, trace=None, output=out, window_q1_pair=False, large_window_q1_pair=False,
                           large_window_q_policy=False)

    def git_output(command, **_kwargs):
        if command[3] == "status":
            return ""
        return runner.KVAF_REVISION if command[2] == kvaf else runner.NATIVE

    monkeypatch.setattr(runner.subprocess, "check_output", git_output)
    monkeypatch.setattr(runner, "same_production", lambda *_args: None)
    monkeypatch.setattr(runner.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(runner, "cpu_affinity", lambda _node: ("0", "0"))
    prepared, calls, cells, summaries = [], [], [], []
    fixture_info = {"actual_miss_percent_per_query": [9.96, 10.01, 9.99, 10.04]}

    def prepare(fixture, **config):
        prepared.append((fixture, config))
        return fixture_info

    def arm(_args, directory, batch, route, source_options, group, node, cpus, *, oracle=None, adapter=None):
        assert source_options == ["--synthetic-fixture", str(out / "fixture")]
        assert group == 0 and (node, cpus) == ("0", "0")
        assert directory.is_relative_to(out / f"synthetic/bs{batch}/cohort0")
        assert adapter == ("adapter" if calls else None)
        calls.append((batch, route, directory, oracle))
        return "adapter"

    def cell(_out, directory, batch, group):
        assert group == 0 and directory == out / f"synthetic/bs{batch}/cohort0"
        result = {"batch": batch}
        cells.append(result)
        return result

    monkeypatch.setattr(runner, "prepare_compact_fixture", prepare)
    monkeypatch.setattr(runner, "run_arm", arm)
    monkeypatch.setattr(runner, "summarize_cell", cell)
    monkeypatch.setattr(runner, "report_cell", lambda _row: None)
    monkeypatch.setattr(runner, "write_summary", lambda _out, rows, metadata: summaries.append((len(rows), metadata)))
    runner.run(args)
    assert prepared == [(out / "fixture", runner.SYNTHETIC_FIXTURE)]
    assert calls[:9] == [
        (batch, "serial", out / f"synthetic/bs{batch}/cohort0/oracle/serial", None)
        for batch in runner.FIXED_BATCHES
    ]
    expected = []
    for batch in runner.FIXED_BATCHES:
        case = out / f"synthetic/bs{batch}/cohort0"
        for block, order in enumerate(runner.BLOCKS, 1):
            expected.extend((batch, route, case / f"block{block}" / route, case / "oracle/serial") for route in order)
    assert calls[9:] == expected
    assert len(cells) == 9
    count, metadata = summaries[-1]
    assert count == 9 and metadata["input_kind"] == "synthetic" and metadata["group"] == 0
    assert metadata["synthetic_fixture"] == fixture_info and "trace" not in metadata


def test_one_cohort_summary_reuses_pair_evidence(tmp_path):
    for key, value in [("native", "test-sha"), ("host", "a5-12"), ("physical_npu", "2")]:
        (tmp_path / f"{key}.txt").write_text(value)
    cell = tmp_path / "g2/bs8/cohort0"

    def write_arm(block, arm, graph):
        out = cell / f"block{block}" / arm
        out.mkdir(parents=True)
        (out / "placement_verified").touch()
        (out / "thp.json").write_text(json.dumps(dict(
            status="measured", page_mode="thp", coverage_min_pct=100, ub_device="adapter",
            numa_pages_overlapping_vmas={"0": 100},
        )))
        result = dict(
            identity=dict(group=2, batch=8, host="a5-12", native="test-sha", physical_npu=2,
                          donor_rows=list(range(0, 16, 2)), sequence_sha256="same"),
            arm=arm, calls=10, checked_queries=320, checked_rows=80, correctness="PASS",
            get_workers=2 if arm == "serial" else 40, graph_event_us=graph * 10,
            graph_mean_us=graph, mean_us=dict(LI=0.6 * graph, tail=0.3 * graph),
        )
        if block == 2 and arm == "fused":
            result["identity"]["sequence_sha256"] = "different"
        (out / "packed_result.json").write_text(json.dumps(result))
        return out / "packed_result.json"

    first_serial = write_arm(1, "serial", 100)
    write_arm(1, "fused", 80)
    write_arm(2, "serial", 110)
    last_fused = write_arm(2, "fused", 90)
    with pytest.raises(ValueError, match="inputs, runtime or memory placement"):
        summarize_cell(tmp_path, cell, 8)
    data = json.loads(last_fused.read_text())
    data["identity"] = json.loads(first_serial.read_text())["identity"]
    last_fused.write_text(json.dumps(data))

    row = summarize_cell(tmp_path, cell, 8)
    assert row["input_kind"] == "real"
    assert row["cohorts"] == 1 and row["cohort"] == 0
    assert row["mean_us"]["serial"]["graph"] == 105
    assert row["mean_us"]["fused"]["graph"] == 85
    assert row["graph_gain_pct"] == pytest.approx(100 * 20 / 105)
    write_summary(tmp_path, [row], plan())
    report = json.loads((tmp_path / "packed_summary.json").read_text())
    assert report["cells"][0]["observations"][1]["order"] == ["fused", "serial"]
    with (tmp_path / "summary.csv").open(newline="") as handle:
        data = list(csv.DictReader(handle))[0]
    assert data["fused_graph_us"] == "85" and data["input_kind"] == "real"


def test_q1_dry_run_preserves_nine_baseline_cells():
    script = Path(__file__).resolve().parent / "support/fixed_q4_a5_12.py"
    command = [sys.executable, "-B", script, "--window-q1-pair", "--dry-run"]
    rejected = subprocess.run(command, text=True, capture_output=True)
    assert rejected.returncode == 2 and "requires --synthetic" in rejected.stderr
    data = json.loads(subprocess.check_output([*command, "--synthetic"], text=True))
    assert data["batch_sizes"] == list(runner.FIXED_BATCHES)
    assert data["oracles"] == 9 and data["measured_graphs"] == 56
    assert data["window_q1_pair"]["enabled_batches"] == list(runner.Q1_BATCHES)
    assert data["window_q1_pair"]["unsupported_batches"] == [20, 24, 28, 32]
    assert data["window_q1_pair"]["blocks"] == [["off", "on"], ["on", "off"]]


def test_q1_uses_paired_bridge_and_compares_request_counts(tmp_path, monkeypatch):
    batch = 16
    cell = tmp_path / "synthetic/bs16/cohort0"
    oracle = cell / "oracle/serial"
    for key, value in (("native", "q1-revision"), ("host", "a5-12"), ("physical_npu", "2")):
        (tmp_path / f"{key}.txt").write_text(value)
    enabled_order = []

    def fake_arm(_args, output, actual_batch, arm, _src, group, _node, _cpus,
                 *, oracle=None, adapter=None, window_q1=False):
        assert arm == "fused" and actual_batch == batch and group == 0
        assert oracle == cell / "oracle/serial" and adapter == "adapter"
        enabled_order.append(window_q1)
        output.mkdir(parents=True)
        (output / "placement_verified").touch()
        (output / "thp.json").write_text(json.dumps(dict(
            status="measured", page_mode="thp", coverage_min_pct=100, ub_device="adapter",
            numa_pages_overlapping_vmas={"0": 100},
        )))
        graph = 90 if window_q1 else 100
        prefetch = [2] * batch
        final = [4] * batch
        if window_q1:
            # First request shifts work to Terminal, second increases total GET.
            prefetch[0], final[0], final[1] = 1, 5, 5
        call = dict(prefetch=sum(prefetch), final=sum(final),
                    prefetch_by_request=prefetch, final_by_request=final)
        if window_q1:
            call["window_q1_ready_flags"] = 4 * batch
        (output / "packed_result.json").write_text(json.dumps(dict(
            identity=dict(group=0, batch=batch, host="a5-12", native="q1-revision", physical_npu=2,
                          donor_rows=list(range(batch)), sequence_sha256="same"),
            arm=arm, calls=10, checked_queries=640, checked_rows=160, correctness="PASS", get_workers=40,
            full_KV_Index_checked_calls=10, window_q1_enabled=window_q1, fused_bridge_sha256="a" * 64,
            graph_event_us=10 * graph, graph_mean_us=graph,
            mean_us=dict(LI=60, tail=30, LI1=20, LI2=20, LI3=20), traffic=[call] * 10,
        )))

    monkeypatch.setattr(runner, "run_arm", fake_arm)
    result = runner.q1_cell(None, tmp_path, cell, batch, oracle, [], 0, "0", "0", "adapter")
    assert enabled_order == [False, True, True, False]
    assert result["mean_us"]["on"]["graph"] == 90 and result["graph_gain_pct"] == 10
    assert result["changed_request_records"] == 40 and result["changed_total_records"] == 20
    assert result["traffic_mean"]["off"] == {"prefetch": 32, "final": 64, "total": 96}
    assert result["traffic_mean"]["on"] == {"prefetch": 31, "final": 66, "total": 97}
    assert runner.q1_cell(None, tmp_path, cell, 32, oracle, [], 0, "0", "0", "adapter") == {
        "status": "unsupported",
    }
    path = cell / "q1/block1/on/packed_result.json"
    data = json.loads(path.read_text())
    data["window_q1_enabled"] = False
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="requested Q1 mode"):
        runner.read_q1_pair(cell / "q1/block1", 0, batch, 0, tmp_path)


def test_bs16_q1_profiler_aliases():
    from large_batch_pair import fixed_invocation_intervals, fixed_stage_records

    names = (
        "v9_li_kernel", "v9_li_window_q1_threshold", "v9_li_window_q1_cold", "v9_parallel_terminal",
    )
    events = [dict(name="scope", ph="X", ts=0, dur=20)]
    for index, name in enumerate(names, 1):
        events.append(dict(name=f"{name}_1", ph="X", cat="kernel", ts=4 * index, dur=1))
    recorded = fixed_stage_records(events, "scope", fused=True, batch=16, strict_bounds=True)
    assert fixed_invocation_intervals(recorded, "scope") == {
        "LI1": [(4, 5)], "LI2": [(8, 9)], "LI3": [(12, 13)], "tail": [(16, 17)],
    }


def test_native_large_q1_dry_run_uses_only_four_native_shapes():
    script = Path(__file__).resolve().parent / "support/fixed_q4_a5_12.py"
    command = [sys.executable, "-B", script, "--large-window-q1-pair", "--dry-run"]
    rejected = subprocess.run(command, text=True, capture_output=True)
    assert rejected.returncode == 2 and "requires --synthetic" in rejected.stderr
    data = json.loads(subprocess.check_output([*command, "--synthetic"], text=True))
    assert data["batch_sizes"] == [20, 24, 28, 32]
    assert data["oracles"] == 4 and data["measured_graphs"] == 24
    assert data["blocks"] == [
        ["native", "deferred_pair", "deferred_q1"], ["deferred_q1", "deferred_pair", "native"],
    ]
    assert data["calls_per_graph"] == 10 and data["graph_warmups"] == 4
    assert data["large_window_q1_pair"]["queries_per_worker"] == {"native": None, "deferred_pair": 2, "deferred_q1": 1}
    assert data["prefetch"] == {"tokens": 1024, "windows_requested": 4, "rank_start": 1024}


def test_native_large_q_policy_can_select_arms():
    script = Path(__file__).resolve().parent / "support/fixed_q4_a5_12.py"
    command = [sys.executable, "-B", script, "--large-window-q-policy", "--dry-run"]
    rejected = subprocess.run(command, text=True, capture_output=True)
    assert rejected.returncode == 2 and "requires --synthetic" in rejected.stderr
    data = json.loads(subprocess.check_output([*command, "--synthetic"], text=True))
    assert data["batch_sizes"] == [20, 24, 28, 32]
    assert data["oracles"] == 4 and data["measured_graphs"] == 40
    assert data["blocks"][0] == ["native", "deferred_q1", "deferred_q2", "deferred_q4", "deferred_auto"]
    assert data["blocks"][1] == list(reversed(data["blocks"][0]))
    assert data["large_window_q_policy"]["q_tasks"] == {"20": 80, "24": 96, "28": 112, "32": 128}
    assert data["large_window_q_policy"]["fixed_wave_capacity"] == {"q1": 16, "q2": 32, "q4": 64}

    subset = json.loads(subprocess.check_output(
        [*command, "--synthetic", "--q-policy", "auto", "q2"], text=True,
    ))
    assert subset["measured_graphs"] == 24
    assert subset["large_window_q_policy"]["mode_codes"] == {"native": None, "deferred_q2": 3, "deferred_auto": 0}


@pytest.mark.parametrize("mode, final_symbol", [
    (2, "v9_li_large_final_pair"), (1, "v9_li_large_final_q1"),
    (3, "v9_li_large_final_pair_dense"), (4, "v9_li_large_final_q4"), (0, "v9_li_large_final_auto"),
])
def test_native_large_deferred_window_gets_separate_profiler_stage(mode, final_symbol):
    from large_batch_pair import fixed_invocation_intervals, fixed_kernel_stages, fixed_stage_records

    cold = "v9_li_large_deferred_cold_auto" if mode == 0 else "v9_li_large_deferred_cold"
    names = ("v9_li_large_hot", cold, final_symbol, "v9_parallel_terminal_large")
    events = [dict(name="scope", ph="X", ts=0, dur=30)]
    for index, name in enumerate(names, 1):
        events.append(dict(name=f"{name}_1", ph="X", cat="kernel", ts=5 * index, dur=index))
    recorded = fixed_stage_records(events, "scope", fused=True, batch=32, strict_bounds=True,
                                   large_window_queries_per_worker=mode)
    times = fixed_kernel_stages(fixed_invocation_intervals(recorded, "scope"), fused=True)
    assert times["LI"] == 6 and times["last_window"] == 3 and times["tail"] == 4
    with pytest.raises(ValueError, match="Unattributed device task"):
        fixed_stage_records(events, "scope", fused=True, batch=32, strict_bounds=True,
                            large_window_queries_per_worker=(mode + 1) % 5)


@pytest.mark.parametrize("policy", [False, True])
def test_native_large_comparison_verifies_receipts_and_traffic(tmp_path, monkeypatch, policy):
    batch = 32
    cell = tmp_path / "synthetic/bs32/cohort0"
    oracle = cell / "oracle/serial"
    for key, value in (("native", "native-q1-revision"), ("host", "a5-12"), ("physical_npu", "2")):
        (tmp_path / f"{key}.txt").write_text(value)
    modes = []

    def fake_arm(_args, output, actual_batch, arm, _src, group, _node, _cpus, *,
                 oracle=None, adapter=None, large_window_queries_per_worker=None):
        assert arm == "fused" and actual_batch == batch and group == 0
        assert oracle == cell / "oracle/serial" and adapter == "adapter"
        mode = large_window_queries_per_worker
        modes.append(mode)
        output.mkdir(parents=True)
        (output / "placement_verified").touch()
        (output / "thp.json").write_text(json.dumps(dict(
            status="measured", page_mode="thp", coverage_min_pct=100, ub_device="adapter",
            numa_pages_overlapping_vmas={"0": 100},
        )))
        graph, last_window = {None: (100, 0), 2: (120, 27), 1: (108, 15), 3: (103, 10),
                              4: (110, 17), 0: (104, 11)}[mode]
        prefetch = [2] * batch
        final = [4] * batch
        if mode == 1:
            # First request changes partition only, while second changes total.
            prefetch[0], final[0], final[1] = 1, 5, 5
        call = dict(prefetch=sum(prefetch), final=sum(final), prefetch_by_request=prefetch, final_by_request=final,
                    window_q1_ready_flags=4 * batch if mode is not None else 0)
        if policy and mode is not None:
            if mode == 0:
                # Two light Q4 rows free the two lanes needed to upgrade one heavy row to Q1.
                call["window_q_rows_by_width"] = {"1": 1, "2": 29, "4": 2}
            else:
                call["window_q_rows_by_width"] = {
                    str(width): batch if width == {1: 1, 3: 2, 4: 4}[mode] else 0 for width in (1, 2, 4)
                }
        (output / "packed_result.json").write_text(json.dumps(dict(
            identity=dict(group=0, batch=batch, host="a5-12", native="native-q1-revision", physical_npu=2,
                          sequence_sha256="same"),
            arm=arm, calls=10, checked_queries=40 * batch, checked_rows=10 * batch, correctness="PASS",
            get_workers=32, full_KV_Index_checked_calls=10, large_window_queries_per_worker=mode,
            fused_bridge_sha256="a" * 64, graph_event_us=10 * graph, graph_mean_us=graph,
            mean_us=dict(LI=60 + last_window, tail=30, LI1=20, LI23=40, last_window=last_window),
            traffic=[call] * 10,
        )))

    monkeypatch.setattr(runner, "run_arm", fake_arm)
    policy_modes = runner.large_policy_modes(runner.Q_POLICY_CHOICES) if policy else None
    row = runner.large_q1_cell(None, tmp_path, cell, batch, oracle, [], 0, "0", "0", "adapter",
                               policy_modes=policy_modes)
    if policy:
        assert modes == [None, 1, 3, 4, 0, 0, 4, 3, 1, None]
        other = "deferred_q2"
        assert row["window_q_rows_by_width_mean"]["deferred_auto"] == {"1": 1, "2": 29, "4": 2}
    else:
        assert modes == [None, 2, 1, 1, 2, None]
        other = "deferred_pair"
    pair_gain = row["comparisons"][f"deferred_q1_vs_{other}"]
    assert pair_gain["graph_gain_pct"] == pytest.approx((-500 / 103) if policy else 10)
    assert pair_gain["changed_request_records"] == 40 and pair_gain["changed_total_records"] == 20
    assert row["comparisons"]["deferred_q1_vs_native"]["graph_gain_pct"] == -8
    q_policies = runner.Q_POLICY_CHOICES if policy else None
    runner.write_large_q1_summary(tmp_path, [row], plan(synthetic=True, large_window_q1_pair=not policy,
                                                      q_policies=q_policies))
    with (tmp_path / "summary.csv").open(newline="") as handle:
        summary = list(csv.DictReader(handle))[0]
    assert summary["deferred_q1_vs_native_graph_gain_pct"] == "-8.0"
    assert summary["deferred_q1_last_window_us"] == "15"
    if policy:
        assert summary["deferred_auto_mean_rows_q4"] == "2"
    path = cell / "large_window/block1/deferred_q1/packed_result.json"
    data = json.loads(path.read_text())
    data["traffic"][0]["window_q1_ready_flags"] = 0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="completion|receipts"):
        extra = {"modes": policy_modes, "require_widths": True} if policy else {}
        runner.read_large_q1_block(cell / "large_window/block1", 0, batch, 0, tmp_path, **extra)


def test_last_window_width_receipts_validate_one_wave_budget():
    from fixed_q4_packed_worker import window_q_rows_by_width

    widths = np.repeat(np.array([1] + [2] * 29 + [4] * 2)[:, None], 4, axis=1)
    assert window_q_rows_by_width(widths, 0) == {"1": 1, "2": 29, "4": 2}
    with pytest.raises(ValueError, match="requested fixed Q width"):
        window_q_rows_by_width(widths, 3)
    widths[-1, :] = 2
    with pytest.raises(ValueError, match="one-wave AIV budget"):
        window_q_rows_by_width(widths, 0)
    widths[0, 0] = 2
    with pytest.raises(ValueError, match="request's Q width"):
        window_q_rows_by_width(widths, 0)
