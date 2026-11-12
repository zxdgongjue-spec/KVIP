# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
from types import SimpleNamespace

import pytest

from kvip.diagnostics.compare import print_summary, summarize
from kvip.diagnostics.fused_replay import validate_fused_tuning
from kvip.diagnostics.replay_measure import study_cases, verify_case_equivalence
from kvip.runtime.build import recent_build_options


def _case(mean):
    return dict(
        status="pass",
        snapshot_sha256="same",
        geometry={},
        scope="full",
        soc="A5",
        device=0,
        reads_samples=[
            dict(
                cold_miss_rate=0.1,
                cold_demand=2,
                selected_unique=20,
                prefetch=2,
                final=1,
                unused_prefetch=1,
                unused_prefetch_bytes=656,
                bytes=1968,
                useful_transfer_bytes=1312,
            )
        ],
        timing=dict(mean_us=mean, p50_us=mean),
        runtime_workers=32,
        selection_comparison=dict(selected_set_sha256="ids", ordered_output_sha256="ordered"),
        threshold_comparison=dict(initial_keys=[[40000, 40001]], final_keys=[[40002, 40003]]),
        timing_boundary="complete graph",
    )


def _reports():
    return {
        f"{transport}-{mode}": _case(mean)
        for transport in ("urma", "ubmem")
        for mode, mean in (("serial", 120), ("fused-before", 100), ("fused-pipeline", 90), ("fused-unified", 81))
    }


def test_study_changes_only_seed_execution_and_reports_each_increment(capsys):
    cases = study_cases(SimpleNamespace(study="pipeline"))
    assert list(cases) == ["before", "pipeline", "unified"]
    assert all(value["recent_threshold"] for value in cases.values())
    assert cases["before"] | {"recent_pipeline": True} == cases["pipeline"]
    assert cases["pipeline"] | {"recent_unified_topk": True} == cases["unified"]
    reports = _reports()
    verify_case_equivalence(reports, same_threshold=True)
    result = summarize(reports, "pipeline")
    effect = result["effects"]["ubmem"]
    assert effect["pipeline_reduction_pct"] == pytest.approx(10)
    assert effect["unified_topk_incremental_reduction_pct"] == pytest.approx(10)
    assert effect["combined_reduction_pct"] == pytest.approx(19)
    assert effect["combined_vs_serial_reduction_pct"] == pytest.approx(32.5)
    print_summary(result)
    assert "combined=19.00%" in capsys.readouterr().out


@pytest.mark.parametrize("field", ["initial_keys", "final_keys"])
def test_study_rejects_a_changed_threshold_even_when_final_topk_matches(field):
    reports = _reports()
    reports["ubmem-fused-unified"]["threshold_comparison"][field][0][0] += 1
    with pytest.raises(AssertionError, match="threshold"):
        verify_case_equivalence(reports, same_threshold=True)
    with pytest.raises(ValueError, match="threshold"):
        summarize(reports, "pipeline")


def test_study_rejects_reordered_topk_and_different_inputs():
    reports = _reports()
    changed = deepcopy(reports)
    changed["urma-fused-pipeline"]["selection_comparison"]["ordered_output_sha256"] = "changed"
    with pytest.raises(AssertionError, match="ordered TopK"):
        verify_case_equivalence(changed, same_threshold=True)
    with pytest.raises(ValueError, match="selection"):
        summarize(changed, "pipeline")
    reports["ubmem-fused-unified"]["snapshot_sha256"] = "different"
    with pytest.raises(ValueError, match="identical frozen"):
        summarize(reports, "pipeline")


@pytest.mark.parametrize("bad", [0, 1, "false", None])
def test_compile_switches_reject_non_boolean_values(bad):
    with pytest.raises(ValueError, match="boolean"):
        recent_build_options(pipeline=bad)
    with pytest.raises(ValueError, match="boolean"):
        recent_build_options(unified_topk=bad)
    with pytest.raises(ValueError, match="boolean"):
        validate_fused_tuning({"recent_pipeline": bad}, "ubmem")
