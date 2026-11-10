# SPDX-License-Identifier: Apache-2.0
"""Initial-residency demand is neither query-entry count nor transfer count."""

from types import SimpleNamespace

import numpy as np
import pytest

from kvip.diagnostics.demand import SnapshotDemand, snapshot_demand


def fixture():
    metadata = np.ones((5, 8), dtype=np.int64)
    metadata[2] = [4, 0, 5, 1, 6, 2, 7, 3]  # Interleaved TND requests.
    metadata[4] = 0
    metadata[4].view(np.int32)[:2] = [1, 2]
    ids = np.tile(np.arange(12), (3, 1))
    ids[1] = [4, 1, 2, 5, 0, 3, 6, 7, 8, 9, 10, 11]
    hot = np.ones((3, 4), dtype=np.uint8)
    hot[0], hot[1, 2] = 0, 0
    packed = np.tile([[0, 1, 2], [3, 4, 5], [6, 0, 1], [2, 3, 4]], (2, 1))
    arrays = {
        "pre.metadata": metadata,
        "pre.logical_ids": ids,
        "pre.hot_valid": hot,
        "pre.tail_bounds": np.array([[3, 5], [2, 4]], dtype=np.int32),
        "pre.lru_slots": np.array([[0, 1], [0, 1]], dtype=np.int32),
        "post.logical_output": packed[metadata[2]],
    }
    return SimpleNamespace(
        manifest={"geometry": dict(batch=2, history=12, resident=4, output_stride=8)},
        array=arrays.__getitem__,
    )


def test_q4_union_owner_identity_permuted_hot_and_abc_precedence():
    archive = fixture()
    actual = snapshot_demand(archive)
    assert actual == dict(
        selected_unique=14,
        cold_demand=6,
        candidate_hot_refetch=3,
        hot_hit=1,
        abc_hit=4,
        selected_read_demand=9,
        cold_miss_rate=6 / 14,
    )
    # Counting query entries would report 24; deduplicating across owners would
    # report 7. Hot physical slot 0 holds logical 4, and ABC overrides it.
    assert sum(actual[key] for key in ("cold_demand", "candidate_hot_refetch", "hot_hit", "abc_hit")) == 14


def test_actual_changed_selection_changes_cold_count_without_assuming_tie_probability():
    archive = fixture()
    classifier = SnapshotDemand(archive)
    selected = archive.array("post.logical_output").copy()
    selected[selected == 6] = 5
    changed = classifier.classify(selected)
    assert changed["selected_unique"] == 12
    assert changed["cold_demand"] == 4
    assert changed["candidate_hot_refetch"] == 3
    # Precomputed state is reusable and the source archive was not mutated.
    assert classifier.classify(archive.array("post.logical_output"))["cold_demand"] == 6


def test_inactive_guard_rows_do_not_contribute_or_merge_into_live_request():
    archive = fixture()
    metadata = archive.array("pre.metadata")
    inactive = metadata[2] >= 4
    metadata[3, inactive], metadata[2, inactive] = 0, 0
    metadata[4].view(np.int32)[1] = 0
    archive.array("post.logical_output")[inactive] = -1
    assert snapshot_demand(archive) == dict(
        selected_unique=7,
        cold_demand=3,
        candidate_hot_refetch=1,
        hot_hit=1,
        abc_hit=2,
        selected_read_demand=4,
        cold_miss_rate=3 / 7,
    )
    archive.array("post.logical_output")[inactive, 0] = 0
    with pytest.raises(ValueError, match="padding"):
        snapshot_demand(archive)


@pytest.mark.parametrize(
    "fault,match", [("unpack", "packed"), ("owner", "owners"), ("lru", "LRU"), ("logical", "history")]
)
def test_rejects_ambiguous_owner_or_invalid_selection(fault, match):
    archive = fixture()
    if fault == "unpack":
        archive.array("pre.metadata")[2, 0] = 100
    elif fault == "owner":
        archive.array("pre.metadata")[4].view(np.int32)[1] = 1
    elif fault == "lru":
        archive.array("pre.lru_slots")[0] = [0, 0]
    else:
        archive.array("post.logical_output")[0, 0] = 12
    with pytest.raises(ValueError, match=match):
        snapshot_demand(archive)
