# SPDX-License-Identifier: Apache-2.0
import copy
import json
from types import SimpleNamespace

import pytest

from kvip.diagnostics import replay
from kvip.diagnostics.counter_clock import calibration_summary
from kvip.diagnostics.pipeline_options import options_for
from kvip.diagnostics.post_index_order_report import ROLES, summarize
from kvip.diagnostics.post_index_profile import decode_post_index
from test_post_index_profile import pix2_data
from test_read_through_report import case


def reports():
    values = {}
    for role in ROLES:
        item = values[role] = case("before")
        item["tuning"] = options_for("auto-post-index")
        item["library_sha256"] = "libdigest"
        item["post_index_wait_first"] = wait = role.startswith("wait-first")
        for observed, reads in zip(item["auto_route"]["samples"], item["reads_samples"]):
            reads["post_index"] = (dict(status="pass", active=True, wait_first=wait, fallback=False,
                                      submitted_requests=reads["final"], completed_requests=reads["final"])
                                   if observed["mode"] == "low" else dict(status="not_entered", active=False))
        raw, detail, events = pix2_data()
        if wait:
            raw[:, 2] = 2
            raw[:, 9] = detail[:, 7] = 158
        phase = decode_post_index(raw, active=True, diagnostic=True, index_detail=detail, events=events)
        item["native_diagnostics"] = dict(included_in_timing=False,
            counter_calibration=calibration_summary([dict(cycles=20_000_000, event_us=20_000)]*3),
            samples=[dict(reads=dict(post_index=phase))])
    return values


def write(root, values):
    for role, value in values.items():
        (root/f"g2-{role}.json").write_text(json.dumps(dict(status="pass", cases={"auto-post-index": value})))


def test_same_binary_complete_order_control_and_calibrated_cycles(tmp_path, capsys):
    write(tmp_path, reports())
    summarize(SimpleNamespace(directory=tmp_path, group=2))
    output = capsys.readouterr().out
    assert "same_binary=true all_Index_and_Get_work=retained" in output
    assert "ORDER_GAIN overlap_vs_wait-first STEADY mean/p50=+0.00%/+0.00%" in output
    assert "DRIFT wait-first" in output and "DRIFT overlap" in output
    assert "index_span_us_approx=0.015" in output
    assert "index_scope=current_impl_isolated" in output
    assert "async_order_exercised=true" in output


@pytest.mark.parametrize("fault", ["order", "binary", "source", "traffic", "route", "lifecycle", "clock", "old-vf"])
def test_order_report_rejects_invalid_or_unfair_comparisons(tmp_path, fault):
    values = reports()
    item = values["overlap"]
    if fault == "order":
        item["reads_samples"][1]["post_index"]["wait_first"] = True
    elif fault == "binary":
        item["library_sha256"] = "other"
    elif fault == "source":
        item["kvip_commit"] = item["production_source"]["commit"] = "d"*40
    elif fault == "traffic":
        item["reads_samples"][1]["bytes"] += 656
    elif fault == "route":
        item["auto_route"]["samples"][1]["mode"] = "high"
    elif fault == "lifecycle":
        item["auto_route"].pop("lifecycle")
    elif fault == "clock":
        item["native_diagnostics"]["counter_calibration"]["cycles_per_us"] *= 2
    elif fault == "old-vf":
        item["native_diagnostics"]["samples"][0]["reads"]["post_index"]["index_work"]["abi"] = "PIX1"
    write(tmp_path, copy.deepcopy(values))
    with pytest.raises(ValueError):
        summarize(SimpleNamespace(directory=tmp_path, group=2))


def test_order_control_requires_exact_variant_and_calibration_is_diagnostic_only(tmp_path):
    base = ["--snapshot", "/snapshot", "--build-dir", "/build", "--transport", "urma", "--mode", "fused",
            "--soc", "Ascend950DT_9582", "--kvaf-server", "127.0.0.1:29084", "--output", str(tmp_path/"out.json"),
            "--study", "post-threshold", "--threshold-policy", "recent", "--upstream-root", "/source",
            "--upstream-revision", "a"*40]
    args = replay.arguments(base + ["--pipeline-variants", "auto-post-index", "--post-index-wait-first",
                                   "--diagnostic-replays", "2", "--calibrate-counter"])
    assert args.post_index_wait_first and args.calibrate_counter
    for suffix in (["--pipeline-variants", "auto-tail", "--post-index-wait-first"],
                   ["--pipeline-variants", "auto-post-index", "--calibrate-counter"]):
        with pytest.raises(SystemExit):
            replay.arguments(base + suffix)


def test_high_prefetch_schedule_variation_does_not_reject_identical_low_work(tmp_path, capsys):
    values = reports()
    item = values["overlap"]
    high = item["reads_samples"][0]
    high.update(prefetch=20, final=945, transfers=965, bytes=965*656,
                unused_prefetch=10, useful_transfers=955)
    item["auto_route"]["samples"][0].update(observed_prefetch=20, observed_final=945)
    write(tmp_path, values)
    summarize(SimpleNamespace(directory=tmp_path, group=2))
    output = capsys.readouterr().out
    gains = [line for line in output.splitlines() if line.startswith("ORDER_GAIN ")]
    assert len(gains) == 4 and all(" STEADY " in line for line in gains)
    assert not any(" ALL " in line for line in gains)
    assert "identical_low_traffic=PASS" in output
    assert "identical_traffic=PASS" not in output


def test_consistent_but_different_low_traffic_is_rejected(tmp_path):
    values = reports()
    item = values["overlap"]
    low = item["reads_samples"][1]
    low.update(final=956, transfers=956, bytes=956*656, useful_transfers=956)
    low["post_index"].update(submitted_requests=956, completed_requests=956)
    item["auto_route"]["samples"][1].update(observed_final=956, observed_useful_get=956, necessary_get=956)
    write(tmp_path, values)
    with pytest.raises(ValueError, match="LOW Get traffic changed"):
        summarize(SimpleNamespace(directory=tmp_path, group=2))


@pytest.mark.parametrize("reason", ["steady_contains_high_route", "steady_blocking_fallback"])
def test_order_gain_requires_all_steady_samples_low_and_asynchronous(tmp_path, capsys, reason):
    values = reports()
    for item in values.values():
        if reason == "steady_contains_high_route":
            item["auto_route"].update(low_steps=1, high_steps=2)
            item["auto_route"]["samples"][1]["mode"] = "high"
            item["reads_samples"][1]["post_index"] = dict(status="not_entered", active=False)
        else:
            item["reads_samples"][1]["post_index"]["fallback"] = True
    write(tmp_path, values)
    summarize(SimpleNamespace(directory=tmp_path, group=2))
    output = capsys.readouterr().out
    assert not any(line.startswith("ORDER_GAIN ") for line in output.splitlines())
    assert f"ORDER_COMPARISON unavailable reason={reason}" in output
    assert "order_gain_valid=false" in output


def references(root, *, fault=None):
    before, end = case("before"), case("before")
    end["timing"].update(samples_us=[410., 390., 400.], mean_us=400., p50_us=400.)
    # The last observed route is not a compile/runtime policy.
    before["tuning"]["auto_low_active"] = False
    end["tuning"]["auto_low_active"] = True
    if fault == "source":
        end["kvip_commit"] = end["production_source"]["commit"] = "d"*40
    elif fault == "tuning":
        end["tuning"]["prefetch_rank"] = 1024
    elif fault == "selection":
        end["selection_comparison"]["ordered_output_sha256"] = "changed"
    elif fault == "route":
        end["auto_route"].update(low_steps=1, high_steps=2)
        end["auto_route"]["samples"][1]["mode"] = "high"
    elif fault == "low-traffic":
        end["reads_samples"][1].update(final=956, transfers=956, bytes=956*656, useful_transfers=956)
        end["auto_route"]["samples"][1].update(observed_final=956, observed_useful_get=956, necessary_get=956)
    for name, item in (("before", before), ("before-end", end)):
        (root/f"{name}.json").write_text(json.dumps(dict(status="pass", cases={name: item})))
    return dict(before=root/"before.json", before_end=root/"before-end.json")


def test_before_end_reports_both_controls_and_unadjusted_drift(tmp_path, capsys):
    write(tmp_path, reports())
    summarize(SimpleNamespace(directory=tmp_path, group=2, **references(tmp_path)))
    output = capsys.readouterr().out
    assert "GAIN_vs_before overlap STEADY_mean/p50=+0.00%/+0.00%" in output
    assert "GAIN_vs_before-end overlap STEADY_mean/p50=+5.06%/+5.06%" in output
    assert "DRIFT before end/first STEADY_mean/p50=+5.33%/+5.33% unadjusted=true" in output
    assert "REFERENCE_SCOPE before=first_end_bracketed" in output
    assert "single_point_not_bracketed" not in output


@pytest.mark.parametrize("fault", ["source", "tuning", "selection", "route", "low-traffic"])
def test_before_end_rejects_changed_source_policy_or_output(tmp_path, capsys, fault):
    write(tmp_path, reports())
    with pytest.raises(ValueError):
        summarize(SimpleNamespace(directory=tmp_path, group=2, **references(tmp_path, fault=fault)))
    assert "GAIN_vs_before" not in capsys.readouterr().out


def test_before_end_cannot_be_used_without_before(tmp_path):
    with pytest.raises(ValueError, match="--before-end requires --before"):
        summarize(SimpleNamespace(directory=tmp_path, group=2, before_end=tmp_path/"end.json"))
