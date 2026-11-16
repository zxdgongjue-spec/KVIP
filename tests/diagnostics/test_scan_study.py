"""Keep scan/admission changes independent and account the full device chain."""

import json
from types import SimpleNamespace

from kvip.diagnostics import replay, replay_measure, scan_report
from kvip.diagnostics.scan_options import DEFAULT_VARIANTS, budgets_for, options_for


def test_scan_study_reuses_one_arena_and_delivers_independent_options(monkeypatch):
    args = SimpleNamespace(study="scan-order", mode="fused", transport="urma",
                           threshold_policy="recent", scan_variants=list(DEFAULT_VARIANTS))
    context, allocated, calls = object(), [], []
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
    assert result["execution_order"] == list(DEFAULT_VARIANTS)
    assert calls == [(name, options_for(name), budgets_for(name)) for name in DEFAULT_VARIANTS]


def test_actual_command_line_accepts_scan_variants(tmp_path):
    args = replay.arguments([
        "--snapshot", str(tmp_path / "input"), "--build-dir", str(tmp_path / "build"),
        "--transport", "urma", "--mode", "fused", "--threshold-policy", "recent",
        "--study", "scan-order", "--scan-variants", *DEFAULT_VARIANTS,
        "--upstream-root", str(tmp_path / "source"), "--upstream-revision", "1" * 40,
        "--kvaf-server", "127.0.0.1:29089", "--device", "0", "--soc", "Ascend950DT_9582",
        "--warmup", "2", "--iterations", "10", "--diagnostic-replays", "0", "--quiet",
        "--output", str(tmp_path / "result.json"),
    ])
    assert args.scan_variants == list(DEFAULT_VARIANTS)
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
