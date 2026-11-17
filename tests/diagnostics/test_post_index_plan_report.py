# SPDX-License-Identifier: Apache-2.0
import copy
import json

import pytest

from kvip.diagnostics.post_index_plan_report import ROLES, summarize
from kvip.diagnostics.post_index_profile import decode_post_index
from test_post_index_order_report import reports
from test_post_index_profile import finalization_data
from test_read_through_report import case


def inputs():
    fused = reports()["overlap"]
    fused["native_diagnostics"]["samples"] *= 2
    result = {}
    for role in ROLES:
        row = result[role] = case("serial") if role.startswith("serial") else copy.deepcopy(fused)
        if role.startswith("after"):
            row["kvip_commit"] = row["production_source"]["commit"] = "c" * 40
            row["library_sha256"] = "candidate"
            row["timing"].update(samples_us=[360., 300., 310.], mean_us=970/3, p50_us=310.)
    return result


def write(root, values):
    for role, row in values.items():
        report = row if role.startswith("serial") else dict(status="pass", cases={"auto-post-index": row})
        (root / f"g2-{role}.json").write_text(json.dumps(report))


def test_full_graph_brackets_and_unmixed_diagnostics(tmp_path, capsys):
    write(tmp_path, inputs())
    summarize(tmp_path, 2)
    output = capsys.readouterr().out
    assert "STEADY_mean/p50=305.000/305.000" in output
    assert "vs_before=+18.67%/+18.67%" in output
    assert "DRIFT after" in output and "DRIFT before" in output and "DRIFT serial" in output
    assert "all_brackets_mean_and_p50_parity=true reduction_ge35pct=false" in output
    assert "PHASE_END after" in output and "phase_ends_not_additive=true" in output
    assert "FINALIZE after sample=1 unavailable=true" in output


def test_report_prints_existing_finalizer_ticks_without_adding_them_to_headline(tmp_path, capsys):
    rows = inputs()
    raw, detail = finalization_data()
    decoded = decode_post_index(raw, active=True, diagnostic=True, finalization_detail=detail)
    for role in ("before", "after"):
        for sample in rows[role]["native_diagnostics"]["samples"]:
            sample["reads"]["post_index"]["finalization"] = decoded["finalization"]
    write(tmp_path, rows)
    summarize(tmp_path, 2)
    output = capsys.readouterr().out
    assert "FINALIZE after sample=1 cycles finalize_end=200 post_cq_tail=80" in output
    assert "scope=join_reduction_publication_feedback headline_excluded=true" in output
    assert "STEADY_mean/p50=305.000/305.000" in output


def test_existing_init_li_metrics_are_reported_without_new_measurements(tmp_path, capsys):
    rows = inputs()
    metrics = dict(
        init_total=dict(status="available", median_cycles=19000., max_cycles=21000),
        init_route=dict(status="available", median_cycles=10000., max_cycles=11500),
        init_fill=dict(status="unavailable", median_cycles=None, max_cycles=None),
        li_final_classification=dict(status="available", median_cycles=9000.5, max_cycles=11000),
    )
    for sample in rows["after"]["native_diagnostics"]["samples"]:
        sample["metrics"] = metrics
    write(tmp_path, rows)
    summarize(tmp_path, 2)
    output = capsys.readouterr().out
    line = next(line for line in output.splitlines() if line.startswith("INIT_LI after sample=1"))
    assert "init_total=19000/21000" in line and "init_route=10000/11500" in line
    assert "li_final_classification=9000.5/11000" in line and "init_fill=" not in line
    assert "overlapping_not_additive=true headline_excluded=true" in line
    assert "STEADY_mean/p50=305.000/305.000" in output
    assert "identical_low_traffic=PASS" in output


def test_all_high_does_not_claim_empty_low_traffic_pass(tmp_path, capsys):
    rows = inputs()
    for role, item in rows.items():
        if role.startswith("serial"):
            continue
        item["auto_route"].update(low_steps=0, high_steps=3)
        for observed, reads in zip(item["auto_route"]["samples"], item["reads_samples"]):
            observed["mode"] = "high"
            reads["post_index"] = dict(status="not_entered", active=False)
        for sample in item["native_diagnostics"]["samples"]:
            sample["reads"]["post_index"] = dict(status="not_entered", active=False)
    write(tmp_path, rows)
    summarize(tmp_path, 2)
    output = capsys.readouterr().out
    assert "identical_low_traffic=not_applicable" in output
    assert "identical_low_traffic=PASS" not in output
    assert "correctness=PASS lifecycle=PASS prediction_recovery=PASS" in output
    # HIGH-only measurement must still enforce the device lifecycle contract.
    rows["after"]["auto_route"]["lifecycle"]["graph_captures"] = 2
    write(tmp_path, rows)
    with pytest.raises(ValueError):
        summarize(tmp_path, 2)


@pytest.mark.parametrize("fault", ["source", "binary", "missing-binary", "traffic", "lifecycle", "weak-serial",
                                  "ordered", "wait-first", "missing-diagnostics", "clock"])
def test_refuse_false_success(tmp_path, fault):
    rows = inputs()
    after = rows["after-end"]
    if fault == "source":
        after["kvip_commit"] = after["production_source"]["commit"] = "d" * 40
    elif fault == "binary":
        after["library_sha256"] = "wrong"
    elif fault == "missing-binary":
        after["library_sha256"] = rows["after"]["library_sha256"] = None
    elif fault == "traffic":
        after["reads_samples"][1]["bytes"] += 656
    elif fault == "lifecycle":
        after["auto_route"]["lifecycle"]["graph_captures"] = 2
    elif fault == "weak-serial":
        rows["serial"]["runtime_workers"] = 2
    elif fault == "ordered":
        after["selection_comparison"]["ordered_output_sha256"] = "different"
    elif fault == "missing-diagnostics":
        rows["before"].pop("native_diagnostics")
    elif fault == "clock":
        rows["after"]["native_diagnostics"]["counter_calibration"]["cycles_per_us"] = 1
    else:
        after["post_index_wait_first"] = True
    write(tmp_path, rows)
    with pytest.raises(ValueError):
        summarize(tmp_path, 2)


def test_high_schedule_noise_permitted_but_route_change_is_not(tmp_path, capsys):
    rows = inputs()
    after = rows["after"]
    after["reads_samples"][0].update(prefetch=20, final=945, transfers=965, bytes=965*656,
                                    unused_prefetch=10, useful_transfers=955)
    after["auto_route"]["samples"][0].update(observed_prefetch=20, observed_final=945)
    write(tmp_path, rows)
    summarize(tmp_path, 2)
    assert "CHECK correctness=PASS" in capsys.readouterr().out
    after["auto_route"]["samples"][1]["mode"] = "high"
    after["auto_route"].update(high_steps=2, low_steps=1)
    after["reads_samples"][1]["post_index"] = dict(status="not_entered", active=False)
    write(tmp_path, rows)
    with pytest.raises(ValueError, match="route decisions"):
        summarize(tmp_path, 2)
