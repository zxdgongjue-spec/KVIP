# SPDX-License-Identifier: Apache-2.0
import copy
import json
from statistics import mean, median
from types import SimpleNamespace

import pytest

from kvip.diagnostics.read_through_report import LIFECYCLE, ROLES, summarize


def case(role):
    serial, candidate = role.startswith("serial"), role == "read-through"
    samples = [340., 320., 330.] if serial else [390., 370., 380.]
    if candidate:
        samples = [400., 310., 320.]
    geometry = dict(batch=16, token_count=64, token_bytes=656, output_stride=16896, owners=17)
    reads = dict(prefetch=0, final=955, transfers=955, bytes=955*656, unused_prefetch=0, useful_transfers=955)
    sha = ("a" if serial else "c" if candidate else "b") * 40
    result = dict(
        status="pass", transport="urma", mode="serial" if serial else "fused",
        timing=dict(count=len(samples), samples_us=samples, mean_us=mean(samples), p50_us=median(samples)),
        reads_samples=[dict(reads) for _ in samples], geometry=geometry,
        snapshot_sha256="snapshot", kvaf_build="/build-npu", device=0, soc="Ascend950DT_9582", warmup=2,
        kvip_commit=sha, production_source=dict(kind="exact_upstream_checkout", commit=sha),
        selection_comparison=dict(selected_set_sha256="selection", ordered_output_sha256="ordered"))
    if serial:
        result.update(runtime_workers=32, batch_get_post_size=128,
                      batch_get_max_inflight_total=2048, jettys_per_core=2)
        return result
    route_samples = [dict(mode="high" if i == 0 else "low", observed_prefetch=0,
                          observed_final=955, observed_useful_get=955, necessary_get=955)
                     for i in range(len(samples))]
    lifecycle = dict(
        status="pass", graph_captures=1, captured_addresses_unchanged=True,
        feedback_reset=False, real_packed_row_reorder=True, prediction_fault_injection=True,
        sequence=[dict(name=name, learned_owners=[], mode="low" if name == "prediction_fault_injection_low" else "high")
                  for name in LIFECYCLE])
    result.update(
        tuning=dict(auto_route=True), score_comparison=dict(status="bit_exact"),
        auto_route=dict(status="pass", capture_and_warmup_learning=False,
                        decision_and_feedback_in_timing=True, snapshot_restored_each_step=True,
                        samples=route_samples, low_steps=2, high_steps=1, lifecycle=lifecycle))
    if candidate:
        result["tuning"]["low_miss_read_through"] = True
        result["read_through"] = dict(
            original_hot_abc_stride=16896, output_stride=16896+8192, scratch_slots_per_owner=8192,
            scratch_bytes=17*8192*656,
            low_persistent_check=dict(status="pass", persistent_index_unchanged=True,
                                      hot_abc_bytes_unchanged=True, scratch_slots=8192))
        lifecycle["persistent_carry"] = dict(
            status="pass", graph_captures=1, snapshot_restores=1,
            persistent_reset_between_steps=False, captured_addresses_unchanged=True,
            sequence=[dict(name=name, mode=mode, persistent_state_verified=True)
                      for name, mode in (("low", "low"), ("low-repacked", "low"), ("high-after-low", "high"))])
    return result


def write_cases(directory, cases):
    for role, value in cases.items():
        report = value if role.startswith("serial") else dict(status="pass", cases={role: value})
        (directory / f"g2-{role}.json").write_text(json.dumps(report))


def test_same_run_report_distinguishes_all_steady_and_drift(tmp_path, capsys):
    cases = {role: case(role) for role in ROLES}
    write_cases(tmp_path, cases)
    summarize(SimpleNamespace(directory=tmp_path, group=2))
    output = capsys.readouterr().out
    assert "RESULT_BEGIN G2" in output and "RESULT_END G2" in output
    assert "343.333/ 320.000" in output
    assert "315.000/ 315.000" in output
    assert "GAIN_ALL" in output and "GAIN_STEADY" in output
    assert "DRIFT serial" in output and "DRIFT before" in output
    assert "one_graph_carry=PASS" in output
    assert "steady_mean_and_p50_parity=true" in output
    assert "latency_reduction_ge35pct=false" in output


@pytest.mark.parametrize("fault", ["no-carry", "persistent-reset", "persistent-corruption", "low-off",
                                  "traffic", "missing-lifecycle", "weak-serial", "different-input",
                                  "different-end-source", "wrong-mean"])
def test_rejects_incomplete_or_unfair_reports(tmp_path, fault):
    cases = {role: case(role) for role in ROLES}
    candidate = cases["read-through"]
    if fault == "no-carry":
        del candidate["auto_route"]["lifecycle"]["persistent_carry"]
    elif fault == "persistent-reset":
        candidate["auto_route"]["lifecycle"]["persistent_carry"]["persistent_reset_between_steps"] = True
    elif fault == "persistent-corruption":
        candidate["read_through"]["low_persistent_check"]["hot_abc_bytes_unchanged"] = False
    elif fault == "low-off":
        candidate["tuning"]["low_miss_read_through"] = False
    elif fault == "traffic":
        candidate["reads_samples"][1]["bytes"] += 656
    elif fault == "missing-lifecycle":
        del candidate["auto_route"]["lifecycle"]
    elif fault == "weak-serial":
        cases["serial"]["runtime_workers"] = 2
    elif fault == "different-input":
        candidate["snapshot_sha256"] = "other"
    elif fault == "different-end-source":
        cases["before-end"]["kvip_commit"] = "d" * 40
        cases["before-end"]["production_source"]["commit"] = "d" * 40
    elif fault == "wrong-mean":
        candidate["timing"]["mean_us"] = 1
    write_cases(tmp_path, copy.deepcopy(cases))
    with pytest.raises(ValueError):
        summarize(SimpleNamespace(directory=tmp_path, group=2))
