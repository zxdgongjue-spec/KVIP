"""CPU tests for 128-byte Rotation Event diagnostics (no NPU or KVAF needed)."""

import sys
from pathlib import Path

import numpy as np
import pytest

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))

from epoch_audit import audit_rotation_events, format_rotation_table  # noqa: E402


def synthetic_events(*, profiling=False):
    # B4, splitQ=1, 8 epochs, 2 AIV lanes, one AIC per epoch.
    data = np.zeros((8, 2, 128), dtype=np.uint8)
    words = data.view(np.uint32).reshape(8, 2, 32)
    words64 = data.view(np.uint64).reshape(8, 2, 16)
    skipped_epochs = {2, 3, 5, 6, 7}
    for epoch in range(8):
        core = (epoch % 4) * 2 + epoch // 4
        for lane in range(2):
            words[epoch, lane, 4] = core
            words[epoch, lane, 5] = epoch
            words[epoch, lane, 6] = 1
            words[epoch, lane, 7] = int(epoch in skipped_epochs)
            if profiling:
                words64[epoch, lane, 0] = 200
                words64[epoch, lane, 1] = 210 if epoch in skipped_epochs else 250
                words64[epoch, lane, 4] = 175
                words[epoch, lane, 22] = 3 if epoch in skipped_epochs else 2
                if epoch not in skipped_epochs:
                    words64[epoch, lane, 5] = 10
                    words64[epoch, lane, 6] = 11
                    words64[epoch, lane, 7] = 12
                    words64[epoch, lane, 8] = 1024
                    words64[epoch, lane, 9] = 7
                    words64[epoch, lane, 10] = 2
    return data


def audit(data, *, profiling=False):
    return audit_rotation_events(
        data,
        batch=4,
        split_q=1,
        li_cores=8,
        rounds=8,
        service_aics=1,
        service_lanes=2,
        history=8192,
        resident=2048,
        profiling_enabled=profiling,
    )


def test_epoch_audit_checks_ownership_and_skips():
    result = audit(synthetic_events())
    assert result["errors"] == []
    assert result["summary"]["validation"] == "PASS"
    assert result["summary"]["total_lanes"] == 16
    assert result["summary"]["complete_lanes"] == 16
    assert result["summary"]["skipped_lanes"] == 10
    assert result["summary"]["served_lanes"] == 6
    assert result["summary"]["candidate_count_observed"] is None
    assert result["summary"]["selected_for_post_count_observed"] is None
    assert [item["recorded_core"] for item in result["events"][::2]] == [
        0, 2, 4, 6, 1, 3, 5, 7
    ]
    assert result["events"][0]["trigger_token"] == 2048
    assert result["events"][-1]["trigger_token"] == 8192
    assert result["events"][4]["skip_reason"] == "trigger_not_reached_or_no_li_work"


def test_epoch_audit_profiled_counters_only_on_served_epochs():
    result = audit(synthetic_events(profiling=True), profiling=True)
    assert result["summary"]["candidate_count_observed"] == 6 * 7
    assert result["summary"]["selected_for_post_count_observed"] == 6 * 2
    assert result["summary"]["confirmed_dma_completed"] is None
    served = [x for x in result["events"] if not x["skipped"]]
    skipped = [x for x in result["events"] if x["skipped"]]
    assert all(x["candidate_count"] == 7 for x in served)
    assert all(x["selected_for_post_count"] == 2 for x in served)
    assert all(x["wait_cycles"] == 25 and x["service_cycles"] == 50 for x in served)
    assert all(x["wait_polls"] == 2 for x in served)
    assert all(x["candidate_count"] is None for x in skipped)
    assert all(x["selected_for_post_count"] is None for x in skipped)
    assert all(x["posted_count"] == 0 for x in skipped)
    assert all(x["wait_cycles"] == 25 for x in skipped)
    assert all(x["closure_pre_publish_cycles"] == 10 for x in skipped)
    assert all(x["wait_polls"] == 3 for x in skipped)
    assert result["summary"]["skipped_closure_pre_publish_cycles_total"] == 100
    assert result["summary"]["skipped_wait_polls_total"] == 30
    assert result["summary"]["empty_served_lanes"] == 0
    assert "ClosurePrePublishCycles" in format_rotation_table(result)


def test_epoch_audit_does_not_mislabel_disabled_profiling_as_zero():
    data = synthetic_events(profiling=True)
    result = audit(data, profiling=False)
    assert result["summary"]["candidate_count_observed"] is None
    assert all(e["candidate_count"] is None for e in result["events"])


@pytest.mark.parametrize("field,value,error", [
    (4, 99, "recorded core"),
    (5, 99, "recorded epoch"),
    (6, 0, "complete"),
    (7, 2, "skipped"),
])
def test_epoch_audit_rejects_invalid_event_fields(field, value, error):
    data = synthetic_events()
    words = data.view(np.uint32).reshape(8, 2, 32)
    words[1, 0, field] = value
    result = audit(data)
    assert result["summary"]["validation"] == "FAIL"
    assert any(error in message for message in result["errors"])


def test_epoch_audit_rejects_posted_greater_than_candidate():
    data = synthetic_events(profiling=True)
    words64 = data.view(np.uint64).reshape(8, 2, 16)
    words64[0, 0, 10] = 100
    result = audit(data, profiling=True)
    assert result["summary"]["validation"] == "FAIL"
    assert any("posted" in error for error in result["errors"])


def test_epoch_audit_rejects_wrong_abi_shape():
    with pytest.raises(ValueError, match="requires uint8"):
        audit(np.zeros((8, 2, 127), dtype=np.uint8))
