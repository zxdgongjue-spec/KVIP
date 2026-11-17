# SPDX-License-Identifier: Apache-2.0
import copy
import json
from types import SimpleNamespace

import pytest

from kvip.diagnostics.pipeline_options import options_for
from kvip.diagnostics.post_index_report import ROLES, summarize
from test_read_through_report import case


def reports():
    values = {}
    for role in ROLES:
        values[role] = case(role if role.startswith("serial") else "before")
        if not role.startswith("serial"):
            name = "auto-post-index" if role == "post-index" else "auto-tail-row-direct"
            values[role]["tuning"] = options_for(name)
        if role == "post-index":
            values[role]["kvip_commit"] = values[role]["production_source"]["commit"] = "d" * 40
            for observed, reads in zip(values[role]["auto_route"]["samples"], values[role]["reads_samples"]):
                reads["post_index"] = (dict(status="pass", active=True, submitted_requests=reads["final"],
                                           completed_requests=reads["final"], fallback=False)
                                       if observed["mode"] == "low" else dict(status="not_entered", active=False))
    return values


def write(directory, values):
    for role, value in values.items():
        report = value if role.startswith("serial") else dict(status="pass", cases={role: value})
        (directory / f"g2-{role}.json").write_text(json.dumps(report))


def test_post_index_report_preserves_mean_median_and_same_run_controls(tmp_path, capsys):
    write(tmp_path, reports())
    summarize(SimpleNamespace(directory=tmp_path, group=2))
    output = capsys.readouterr().out
    assert "RESULT_BEGIN G2" in output and "RESULT_END G2" in output
    assert "GAIN_ALL" in output and "GAIN_STEADY" in output
    assert "DRIFT serial" in output and "DRIFT before" in output
    assert "normal_hot_promotion=PASS" in output and "scratch=NONE" in output
    assert "POST_INDEX_ORDER low_samples=2 fallback_samples=0 diagnostic_fallback_samples=0 fulfilled=true" in output
    assert "steady_mean_and_p50_parity=false" in output


def test_capacity_fallback_is_correct_but_does_not_fulfil_async_order(tmp_path, capsys):
    values = reports()
    values["post-index"]["reads_samples"][1]["post_index"]["fallback"] = True
    write(tmp_path, values)
    summarize(SimpleNamespace(directory=tmp_path, group=2))
    output = capsys.readouterr().out
    assert "correctness=PASS" in output
    assert "POST_INDEX_ORDER low_samples=2 fallback_samples=1 diagnostic_fallback_samples=0 fulfilled=false" in output


@pytest.mark.parametrize("fault", ["missing-flag", "read-through", "compact", "weak-serial", "different-input",
                                  "missing-lifecycle", "different-selection", "changed-control", "candidate-control"])
def test_post_index_rejects_invalid_or_unfair_results(tmp_path, fault):
    values = reports()
    candidate = values["post-index"]
    if fault == "missing-flag":
        candidate["tuning"].pop("low_miss_post_index")
    elif fault == "read-through":
        candidate["read_through"] = {}
    elif fault == "compact":
        candidate["tuning"]["low_miss_compact_plan"] = True
    elif fault == "weak-serial":
        values["serial"]["runtime_workers"] = 2
    elif fault == "different-input":
        candidate["snapshot_sha256"] = "other"
    elif fault == "missing-lifecycle":
        candidate["auto_route"].pop("lifecycle")
    elif fault == "different-selection":
        candidate["selection_comparison"]["ordered_output_sha256"] = "changed"
    elif fault == "changed-control":
        values["before-end"]["tuning"]["low_miss_direct_index"] = False
    elif fault == "candidate-control":
        values["before"]["tuning"]["low_miss_post_index"] = True
        values["before-end"]["tuning"]["low_miss_post_index"] = True
    write(tmp_path, copy.deepcopy(values))
    with pytest.raises(ValueError):
        summarize(SimpleNamespace(directory=tmp_path, group=2))
