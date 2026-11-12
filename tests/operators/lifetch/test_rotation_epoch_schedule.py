"""CPU-only topology checks for V9 variable-length Rotation Epoch closure.

These tests verify ownership and control-flow wiring. They do not substitute
for the Ascend 950DT nonzero-prefetch device test.
"""

from pathlib import Path

import pytest

from kvip.operators.lifetch.geometry import li_geometry, trigger_tokens


ROOT = Path(__file__).resolve().parents[3]


def service_rank(batch, split_query, core):
    return ((core % 2) * batch + core // 2) if split_query else core


def service_core(batch, split_query, rank):
    return ((rank % batch) * 2 + rank // batch) if split_query else rank


def event_lane(batch, split_query, group, core, sub):
    return (service_rank(batch, split_query, core) % group) * 2 + sub


@pytest.mark.parametrize("batch,windows,aic", [
    (1, 2, 32), (2, 4, 32), (4, 8, 32), (8, 8, 32),
    (16, 8, 32), (4, 4, 8),
])
def test_epoch_owner_is_unique_per_epoch_lane(batch, windows, aic):
    geometry = li_geometry(batch, "auto", windows, aic)
    split = int(geometry["cores_per_batch"] == 2)
    group = geometry["service_aics_per_window"]
    rounds = geometry["intermediate_fetches"]
    lanes = geometry["service_aivs_per_window"]
    # Every valid event has precisely one physical AIV producer, even when
    # actual Q/K lengths make that producer unable to reach its Trigger.
    producers = {}
    for epoch in range(rounds):
        for group_id in range(group):
            rank = epoch * group + group_id
            core = service_core(batch, split, rank)
            assert service_rank(batch, split, core) == rank
            for sub in (0, 1):
                lane = event_lane(batch, split, group, core, sub)
                assert 0 <= lane < lanes
                key = (epoch, lane)
                assert key not in producers
                producers[key] = (core, sub)
    assert len(producers) == rounds * lanes
    assert [len(g) for g in geometry["service_groups"]] == [group] * rounds


@pytest.mark.parametrize("reachable", [
    (),
    (0,),
    (0, 1, 4),
    (2, 5, 7),
    tuple(range(8)),
])
def test_missing_epoch_closure_preserves_per_lane_order(reachable):
    """Model the contract: missing epochs wait on previous completion."""
    batch = 4
    geometry = li_geometry(batch, "auto", 8, 32)
    assert geometry["service_groups"] == [[0], [2], [4], [6], [1], [3], [5], [7]]
    assert trigger_tokens(batch, 8192, 2048, "auto", 8, 32)[0] == 2048

    complete = {}
    skipped = {}
    for epoch, group_cores in enumerate(geometry["service_groups"]):
        core = group_cores[0]
        for lane in range(2):
            if epoch:
                assert complete[epoch - 1, lane]
            skipped[epoch, lane] = core not in reachable
            complete[epoch, lane] = True

    assert len(complete) == 16
    assert all(complete.values())
    assert sum(skipped.values()) == (8 - len(reachable)) * 2


def test_device_source_closes_zero_work_and_completed_work_paths():
    kernel = (ROOT / "csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/indexer.h").read_text()
    prefetch = (ROOT / "csrc/operators/lightning_indexer_with_fetch/op_kernel/prefetch.h").read_text()
    vector = (ROOT / "csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/indexer_vector.h").read_text()
    assert "ProcessInvalid();\n        CloseUnreachedRotation();" in kernel
    assert "ProcessMain();\n    CloseUnreachedRotation();" in kernel
    assert "rotationEnabled_ && !rotationServed_" in kernel
    assert "vectorService.CompleteUnreachedRotation();" in kernel
    assert "rotating::CompleteUnreachedEpoch(" in vector
    assert "event->skipped = 1;" in prefetch
    assert "event->complete = 1;" in prefetch
    assert "} while (!previous->complete);" in prefetch
