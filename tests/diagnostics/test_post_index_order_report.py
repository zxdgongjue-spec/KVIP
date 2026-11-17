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
