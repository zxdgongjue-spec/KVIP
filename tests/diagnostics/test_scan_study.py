"""Keep scan/admission changes independent and account the full device chain."""

import json
import hashlib
from types import SimpleNamespace
import numpy as np
import pytest

from kvip.diagnostics import replay, replay_measure, scan_report, scan_options
from kvip.diagnostics.scan_options import DEFAULT_VARIANTS, ROLLING_VARIANTS, BATCHED_VARIANTS, WINDOWED_VARIANTS, LEAN_VARIANTS, budgets_for, options_for, validate_options


@pytest.mark.parametrize("variants", [DEFAULT_VARIANTS, ROLLING_VARIANTS, BATCHED_VARIANTS, WINDOWED_VARIANTS, LEAN_VARIANTS])
def test_scan_study_reuses_one_arena_and_delivers_independent_options(monkeypatch, variants):
    args = SimpleNamespace(study="scan-order", mode="fused", transport="urma",
                           threshold_policy="recent", scan_variants=list(variants))
    context, allocated, calls = object(), [], []
    monkeypatch.setattr(scan_options, "prebuild", lambda *a: None)
    monkeypatch.setattr(replay_measure, "allocate_context", lambda *a: allocated.append(context) or context)
    monkeypatch.setattr(replay_measure, "release_context", lambda c: allocated.remove(c))

    def measure(args, archive, passed_context, retained):
        assert passed_context is context and allocated == [context]
        name = retained["progress"]["variant"]
        assert not getattr(args, "pipeline_options", None)
        calls.append((name, args.scan_options, args.scan_admission))
        return dict(timing={"p50_us": 10}, selection_comparison={"ordered_output_sha256": "same"},
                    threshold_comparison={"initial_keys": [10], "final_keys": [20]})

    monkeypatch.setattr(replay_measure, "measure_case", measure)
    result = replay_measure.run(args, SimpleNamespace(manifest={}), {})
    assert not allocated
    assert result["execution_order"] == list(variants)
    assert calls == [(name, options_for(name), budgets_for(name)) for name in variants]


@pytest.mark.parametrize("variants", [DEFAULT_VARIANTS, ROLLING_VARIANTS, BATCHED_VARIANTS, WINDOWED_VARIANTS, LEAN_VARIANTS])
def test_actual_command_line_accepts_scan_variants(tmp_path, variants):
    args = replay.arguments([
        "--snapshot", str(tmp_path / "input"), "--build-dir", str(tmp_path / "build"),
        "--transport", "urma", "--mode", "fused", "--threshold-policy", "recent",
        "--study", "scan-order", "--scan-variants", *variants,
        "--upstream-root", str(tmp_path / "source"), "--upstream-revision", "1" * 40,
        "--kvaf-server", "127.0.0.1:29089", "--device", "0", "--soc", "Ascend950DT_9582",
        "--warmup", "2", "--iterations", "10", "--diagnostic-replays", "0", "--quiet",
        "--output", str(tmp_path / "result.json"),
    ])
    assert args.scan_variants == list(variants)
    assert args.pipeline_variants is None


def test_report_uses_full_chain_and_keeps_mean_and_median_separate(tmp_path, capsys):
    cases = {}
    for name in DEFAULT_VARIANTS:
        cases[name] = dict(
            timing=dict(mean_us=200 if name == "reverse" else 100, p50_us=90),
            geometry={"token_bytes": 656},
            reads_samples=[dict(unused_prefetch_bytes=656, final=2, bytes=3280,
                                useful_transfer_bytes=2624, cold_demand=3, selected_unique=30)],
        )
    path = tmp_path / "result.json"
    path.write_text(json.dumps(dict(status="pass", execution_order=list(DEFAULT_VARIANTS), cases=cases)))
    scan_report.report(path)
    output = capsys.readouterr().out
    assert "mean_gain=-100.00% p50_gain=+0.00%" in output
    assert "cold_miss=10.0000%" in output
    assert "not link utilization or coldEarly/lead" in output


def test_rolling_experiment_does_not_change_budgets_or_enable_pipeline_options():
    assert {budgets_for(name) for name in ROLLING_VARIANTS} == {(512, 64)}
    assert options_for("reverse-refilter") == dict(reverse_cold=True, rolling_threshold=True, window_refilter=True, rolling_batched=False, rolling_windowed=False, rolling_compact=False, rolling_first_window=False)
    with pytest.raises(ValueError, match="requires rolling"):
        validate_options(dict(reverse_cold=True, rolling_threshold=False, window_refilter=True, rolling_batched=False, rolling_windowed=False, rolling_compact=False, rolling_first_window=False))


def test_rolling_reference_uses_unique_seed_and_only_scanned_causal_positions():
    from kvip.diagnostics.rolling_check import reference

    scores = np.zeros((1, 4, 16384), dtype=np.uint16)
    scores[:, :, :2048] = 100
    scores[:, :, 12288:14336] = 110
    scores[:, :, 14336:] = 65000  # Unscanned and outside causal end.
    ends = np.full((1, 4), 14336)
    bounds = np.array([[12288, 13312]])
    progress = np.zeros((1, 4, 32), dtype=np.uint32)
    progress[:, :, 0] = 4096  # First cold window in reverse visits [12288,14336).
    initial = np.full((1, 4), 100, dtype=np.uint16)
    actual, exact = reference(scores, ends, 2048, bounds, progress, initial, 8, True)
    assert np.all(actual == 110) and np.all(exact == 110)
    actual, exact = reference(scores, ends, 2048, bounds, progress, initial, 8, False)
    assert np.all(actual == 100) and np.all(exact == 100)


def rolling_report(tmp_path):
    cases = {}
    for name in ROLLING_VARIANTS:
        cases[name] = dict(
            timing=dict(mean_us=100, p50_us=90),
            geometry={"token_bytes": 656},
            reads_samples=[dict(unused_prefetch_bytes=6560, final=20, bytes=30000,
                                useful_transfer_bytes=23440, cold_demand=3, selected_unique=30)],
            production_source={"commit": "1" * 40},
            diagnostics_commit="2" * 40,
            rolling_threshold_check=dict(status="bit_exact", queries=64, tightened_queries=60,
                                         capped_queries=0, mean_final_gap_keys=1),
        )
    return dict(status="pass", execution_order=list(ROLLING_VARIANTS), cases=cases)


def test_report_rejects_old_or_wrong_source_instead_of_calling_it_rolling(tmp_path):
    data = rolling_report(tmp_path)
    path = tmp_path / "result.json"
    data["execution_order"] = list(DEFAULT_VARIANTS)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="were not measured"):
        scan_report.load_report(path, require_rolling=True)
    data["execution_order"] = list(ROLLING_VARIANTS)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="revision does not match"):
        scan_report.load_report(path, require_rolling=True, expected_native="3" * 40)
    data["cases"]["reverse-refilter"].pop("rolling_threshold_check")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="oracle has not passed"):
        scan_report.load_report(path, require_rolling=True, expected_native="1" * 40)


def test_report_checks_snapshot_before_reusing_saved_result(tmp_path):
    data = rolling_report(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"group": 7}')
    for case in data["cases"].values():
        case["snapshot_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    path = tmp_path / "result.json"
    path.write_text(json.dumps(data))
    scan_report.report(path, require_rolling=True, expected_native="1" * 40,
                       snapshot=tmp_path, check_only=True)
    manifest.write_text('{"group": 2}')
    with pytest.raises(ValueError, match="manifest does not match"):
        scan_report.report(path, require_rolling=True, snapshot=tmp_path, check_only=True)


def test_report_separates_saved_waste_from_added_repair_and_both_drifts(tmp_path, capsys):
    data = rolling_report(tmp_path)
    cases = data["cases"]
    cases["reverse-roll"]["reads_samples"][0].update(unused_prefetch_bytes=3280, final=25, bytes=30000)
    cases["reverse-roll"]["timing"].update(mean_us=95, p50_us=99)
    cases["forward-end"]["timing"].update(mean_us=102, p50_us=89.1)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(data))
    scan_report.report(path, require_rolling=True)
    output = capsys.readouterr().out
    assert "reverse->reverse-roll mean_gain=+5.00% p50_gain=-10.00%" in output
    assert "waste_save_MiB=+0.0031 waste_save=+50.00% repair_delta_MiB=+0.0031 total_save_MiB=+0.0000" in output
    assert "baseline_mean_drift=+2.00% baseline_p50_drift=-1.00%" in output


def test_scan_compiles_unique_variants_before_allocating_arena(tmp_path, monkeypatch):
    from kvip.runtime import build

    args = SimpleNamespace(study="scan-order", mode="fused", transport="urma", threshold_policy="recent",
                           scan_variants=list(ROLLING_VARIANTS), build_dir=tmp_path, soc="A5")
    calls = []
    monkeypatch.setattr(build, "kvaf_build_paths", lambda path: (path / "cann", path / "headers"))
    monkeypatch.setattr(build, "build_libraries", lambda **kwargs: calls.append(kwargs))

    def stop_before_allocation(*unused):
        assert len(calls) == 5  # forward-end reuses forward's binary.
        assert [tuple(call[key] for key in scan_options.OPTIONS) for call in calls] == [
            tuple(options_for(name).values()) for name in ROLLING_VARIANTS[:-1]]
        raise RuntimeError("stop before NPU allocation")

    monkeypatch.setattr(replay_measure, "allocate_context", stop_before_allocation)
    with pytest.raises(RuntimeError, match="stop before NPU allocation"):
        replay_measure.run(args, SimpleNamespace(manifest={}), {})

    def compiler_failure(**kwargs):
        if kwargs["rolling_threshold"]:
            raise RuntimeError("compiler diagnostic")

    monkeypatch.setattr(build, "build_libraries", compiler_failure)
    retained = {}
    with pytest.raises(RuntimeError, match="compiler diagnostic"):
        replay_measure.run(args, SimpleNamespace(manifest={}), retained)
    assert retained["progress"] == dict(variant="reverse-roll", phase="compile_preflight", step=None)


def test_batch_options_keep_old_control_and_require_new_oracle(tmp_path, capsys):
    assert not options_for("reverse-refilter")["rolling_batched"]
    assert options_for("reverse-batch-ref")["rolling_batched"]
    assert not options_for("reverse-batch")["window_refilter"]
    assert {budgets_for(name) for name in BATCHED_VARIANTS} == {(512, 64)}
    data = rolling_report(tmp_path)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Batched histogram variants were not measured"):
        scan_report.load_report(path, require_batched=True)
    case = data["cases"]["reverse-refilter"]
    data["execution_order"] = list(BATCHED_VARIANTS)
    data["cases"] = {name: dict(case) for name in BATCHED_VARIANTS}
    path.write_text(json.dumps(data))
    scan_report.report(path, require_batched=True)
    assert "CHANGE reverse-refilter->reverse-batch-ref" in capsys.readouterr().out
    data["cases"]["reverse-batch"].pop("rolling_threshold_check")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="oracle has not passed"):
        scan_report.load_report(path, require_batched=True)


def test_window_control_keeps_old_binaries_and_requires_a_real_latency_gain(tmp_path, capsys):
    assert {budgets_for(name) for name in WINDOWED_VARIANTS} == {(512, 64)}
    assert not options_for("reverse-batch-ref")["rolling_windowed"]
    assert options_for("reverse-window-ref")["rolling_windowed"]
    assert not options_for("reverse-window-ref")["rolling_batched"]
    data = rolling_report(tmp_path)
    case = data["cases"]["reverse-refilter"]
    cases = {name: json.loads(json.dumps(case)) for name in WINDOWED_VARIANTS}
    cases["forward-end"]["timing"].update(mean_us=102, p50_us=89)
    # Mean alone is insufficient; lower bytes alone are also insufficient.
    cases["reverse-window"]["timing"].update(mean_us=95, p50_us=91)
    cases["reverse-window-ref"]["timing"].update(mean_us=94, p50_us=84)
    scan_report.check_acceptance(cases)
    output = capsys.readouterr().out
    assert "GATE reverse-window KEEP_BASELINE" in output
    assert "GATE reverse-window-ref CANDIDATE" in output
    assert "GATE forward-window-ref KEEP_BASELINE" in output


def test_lean_report_does_not_call_a_frozen_threshold_full_prefix_exact(tmp_path, capsys):
    assert options_for("reverse-first-ref")["rolling_first_window"]
    assert not options_for("reverse-compact-ref")["rolling_first_window"]
    assert not options_for("reverse-window-ref")["rolling_compact"]
    assert {budgets_for(name) for name in LEAN_VARIANTS} == {(512,64)}
    data = rolling_report(tmp_path)
    case = data["cases"]["reverse-refilter"]
    data["execution_order"] = list(LEAN_VARIANTS)
    data["cases"] = {name: json.loads(json.dumps(case)) for name in LEAN_VARIANTS}
    data["cases"]["reverse-first-ref"]["rolling_threshold_check"]["scope"] = "first_cold_window_frozen"
    path = tmp_path / "result.json"
    path.write_text(json.dumps(data))
    scan_report.report(path, require_lean=True)
    output = capsys.readouterr().out
    assert "first_cold_window_frozen" in output
    assert "CHANGE reverse-window-ref->reverse-compact-ref" in output
    assert "CHANGE reverse-compact-ref->reverse-first-ref" in output
    assert "GATE reverse-first-ref KEEP_BASELINE" in output


def test_frozen_reference_does_not_include_unobserved_future_scores():
    from kvip.diagnostics.rolling_check import reference
    scores = np.zeros((1, 4, 32768), dtype=np.uint16)
    scores[:, :, :16384] = 100
    scores[:, :, 16384:18688] = 110
    scores[:, :, 20000:] = 130
    ends = np.full((1, 4), 32768)
    bounds = np.array([[16384,16384]])
    progress = np.zeros((1,4,32), dtype=np.uint32)
    progress[:,:,0] = 32768
    initial = np.full((1,4),100,dtype=np.uint16)
    frozen, _ = reference(scores, ends, 16384, bounds, progress, initial, 8, False, first_window=True)
    full, _ = reference(scores, ends, 16384, bounds, progress, initial, 8, False)
    assert np.all(frozen==110) and np.all(full==130)


def test_lean_gate_requires_beating_reverse_not_only_forward(capsys):
    cases = {name: dict(timing=dict(mean_us=100, p50_us=100)) for name in LEAN_VARIANTS}
    cases["reverse"]["timing"] = dict(mean_us=90, p50_us=90)
    cases["reverse-compact-ref"]["timing"] = dict(mean_us=95, p50_us=95)
    cases["reverse-first-ref"]["timing"] = dict(mean_us=85, p50_us=85)
    scan_report.check_acceptance(cases, ("reverse", "reverse-compact-ref", "reverse-first-ref"))
    text = capsys.readouterr().out
    assert "GATE reverse CANDIDATE" in text
    assert "GATE reverse-compact-ref KEEP_BASELINE" in text
    assert "GATE reverse-first-ref CANDIDATE" in text
