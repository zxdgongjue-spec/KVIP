# SPDX-License-Identifier: Apache-2.0
"""Reserved-hot reuse must change measured demand without relaxing the KV oracle."""

from types import SimpleNamespace

import numpy as np
import pytest

from kvip.diagnostics.demand import SnapshotDemand
from kvip.diagnostics.replay_measure import fused_demand, supports_low_miss_hot_hits


def selection():
    metadata = np.zeros((5, 4), np.int64)
    metadata[2] = np.arange(4)
    metadata[3] = 1
    metadata[4].view(np.int32)[0] = 1
    hot = np.ones((2, 8), np.uint32)
    hot[1, 1] = 0  # An invalid reserved hot slot must still fetch from DDR.
    arrays = {
        "pre.logical_ids": np.tile(np.arange(16, dtype=np.int64), (2, 1)),
        "pre.hot_valid": hot,
        "pre.metadata": metadata,
        "pre.tail_bounds": np.array([[6, 10]], np.int32),
        "pre.lru_slots": np.array([[0, 1, 6]], np.int32),
    }
    archive = SimpleNamespace(
        manifest={"geometry": dict(batch=1, history=16, resident=8, output_stride=12)},
        array=arrays.__getitem__,
    )
    # Q4 repeats IDs. ABC 6..9 takes precedence over hot and its reservation.
    logical = np.array([[0, 1, 2, 6 + q, 10] for q in range(4)], np.int32)
    return SnapshotDemand(archive), logical


@pytest.mark.parametrize("capable,low", [(False, True), (True, False), (True, True)])
def test_only_capable_low_route_reuses_reserved_hot(capable, low):
    demand, logical = selection()
    baseline = demand.classify(logical)
    assert baseline == dict(
        selected_unique=8, cold_demand=2, candidate_hot_refetch=1, hot_hit=1, abc_hit=4,
        selected_read_demand=3, cold_miss_rate=0.25,
    )
    bundle = SimpleNamespace(tuning=dict(
        low_miss_hot_hits=capable, low_miss_skip=True, auto_route=True, auto_low_active=low,
    ))
    count = 2 if capable and low else 3
    reads = dict(prefetch=0, unused_prefetch=0, final=count, useful_transfers=count)
    actual = fused_demand(bundle, demand, logical, reads)
    assert actual["cold_demand"] == 2 and actual["cold_miss_rate"] == 0.25
    assert actual["initial_candidate_hot_selected"] == 1
    assert actual["abc_hit"] == 4
    if capable and low:
        assert actual["candidate_hot_refetch"] == 0 and actual["reserved_hot_reused"] == 1
        assert actual["hot_hit"] == 2 and actual["selected_read_demand"] == 2
    else:
        assert actual["candidate_hot_refetch"] == 1 and actual["reserved_hot_reused"] == 0
        assert actual["hot_hit"] == 1 and actual["selected_read_demand"] == 3
    # Snapshot and strong serial retain their original cold+candidate semantics.
    assert demand.classify(logical) == baseline


@pytest.mark.parametrize("changed", [dict(final=1), dict(final=3), dict(useful_transfers=1),
                                    dict(prefetch=1), dict(unused_prefetch=1)])
def test_low_route_rejects_missing_duplicate_or_speculative_gets(changed):
    demand, logical = selection()
    bundle = SimpleNamespace(tuning=dict(
        low_miss_hot_hits=True, low_miss_skip=True, adaptive_admission=True, admission_controls=[0, 64, 0, 0],
    ))
    reads = dict(prefetch=0, unused_prefetch=0, final=2, useful_transfers=2) | changed
    with pytest.raises(AssertionError, match="cold/invalid-hot demand"):
        fused_demand(bundle, demand, logical, reads)


def test_capability_comes_from_loaded_binary_and_excludes_ubmem():
    old = SimpleNamespace(library=SimpleNamespace())
    off = SimpleNamespace(library=SimpleNamespace(kvip_low_miss_hot_hits=lambda: 0))
    new = SimpleNamespace(library=SimpleNamespace(kvip_low_miss_hot_hits=lambda: 1))
    assert not supports_low_miss_hot_hits(old, "urma")
    assert not supports_low_miss_hot_hits(off, "urma")
    assert supports_low_miss_hot_hits(new, "urma")
    assert not supports_low_miss_hot_hits(new, "ubmem")


@pytest.mark.parametrize("transport,capability", [("urma", None), ("urma", 0), ("ubmem", 1)])
@pytest.mark.parametrize("option,label", [
    ("auto_route_fast_probe", "Fast route probe"),
    ("auto_route_row_publish", "Row route publication"),
    ("low_miss_direct_index", "Direct Index update"),
])
def test_route_variants_require_the_loaded_kernel_feature(transport, capability, option, label):
    from kvip.diagnostics.replay_measure import prepare_fused_state

    library = SimpleNamespace()
    if capability is not None:
        setattr(library, "kvip_" + option, lambda: capability)
    bundle = SimpleNamespace(selector=None, update=None,
                             rotating=SimpleNamespace(library=library),
                             tuning={option: True})
    with pytest.raises(ValueError, match=label + " requires a capable loaded URMA build"):
        prepare_fused_state(bundle, None, None, None, None, transport)


@pytest.mark.parametrize("prefetch", [0, 1])
def test_sparse_low_uses_actual_get_counters_and_never_reads_stale_mapping(monkeypatch, prefetch):
    from kvip.diagnostics import replay_measure

    demand, _ = selection()
    rows = np.zeros((1, 256), dtype=np.int32)
    rows[0, 9], rows[0, 247] = prefetch, 2
    rotating = SimpleNamespace(rows=rows, row_status=np.zeros((1, 32), dtype=np.int32),
                               final_posts=np.zeros((1, 4, 16), dtype=np.int32))
    update = SimpleNamespace(status=np.zeros(1))
    monkeypatch.setattr(replay_measure, "cpu_array", lambda torch, value: np.asarray(value))
    monkeypatch.setattr(replay_measure, "mapping_values", lambda *args: pytest.fail("Read stale sparse map"))
    if prefetch:
        with pytest.raises(AssertionError, match="zero actual speculative"):
            replay_measure.fused_reads(None, rotating, update, "urma", 656, demand, sparse_low=True)
    else:
        result = replay_measure.fused_reads(None, rotating, update, "urma", 656, demand, sparse_low=True)
        assert result["final"] == result["useful_transfers"] == 2
        assert result["bytes"] == 1312
        assert result["prefetch"] == result["unused_prefetch"] == result["prefetch_classified"] == 0
        assert result["prefetch_unclassified"] == 0
