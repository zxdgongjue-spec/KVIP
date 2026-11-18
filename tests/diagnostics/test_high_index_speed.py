# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from kvip.diagnostics.high_index_speed import ARMS, BASE, DIAGNOSTICS, command, one_to_one, qualified_targets
from kvip.diagnostics.replay import arguments


def test_pinned_binaries_and_work_orders_use_real_parser(tmp_path):
    args = SimpleNamespace(cache=tmp_path, source=tmp_path, revision="a" * 40,
                           server="127.0.0.1:29084", result_dir=tmp_path)
    outputs = set()
    for arm in ARMS:
        parsed = arguments(command(args, 7, arm)[4:])
        assert parsed.upstream_revision == (args.revision if arm in ("candidate", "wait") else BASE)
        assert parsed.high_post_index == (arm != "original")
        assert parsed.post_index_wait_first == (arm == "wait")
        assert parsed.diagnostic_replays == (DIAGNOSTICS if arm in ("old", "wait", "candidate") else 0)
        assert parsed.calibrate_counter == bool(parsed.diagnostic_replays)
        assert parsed.check_auto_route and parsed.device == 0 and parsed.transport == "urma"
        assert parsed.output not in outputs
        outputs.add(parsed.output)


def test_exposure_excludes_fallback_and_reused_or_different_tasks():
    left = [("a", 1), ("a", 2), (None, 3), ("b", 4)]
    right = [("a", 5), (None, 6), ("c", 7), ("a", 8), ("a", 9)]
    assert one_to_one(left, right) == [(1, 5), (2, 8)]


def test_target_requires_measured_units_and_never_scales_nearby_timings(capsys):
    qualified_targets(7, [dict(events=10000, moves=14000, index_span=48.5,
                               preplan=dict(index_union_including_early=59, first_preplan_to_last_index=68)),
                          dict(events=6999, moves=9999, index_span=42.0,
                               preplan=dict(index_union_including_early=47, first_preplan_to_last_index=60))])
    lines = capsys.readouterr().out.splitlines()
    assert "EVENTS exact_n=1" in lines[0] and "post_exact_us=48.500/48.500" in lines[0]
    assert "MOVES exact_n=0" in lines[1] and "post_exact_us=unmeasured" in lines[1]
    assert "nearest_moves=9999" in lines[1] and "with_early_ge10k_under50_n=0" in lines[1]


def test_preplan_reports_measurements_without_summing_concurrent_requests():
    import numpy as np
    import pytest
    from kvip.diagnostics.high_index_profile import _preplan_detail, validate_move_counts

    marks = np.zeros((64, 32), dtype=np.uint64)
    marks[:, 3:6] = [90, 100, 125]
    planners = marks[32:48]
    planners[:, 14:16] = [10, 35]
    planners[:, 21] = 11
    planners[0, 14:16] = [8, 25]
    planners[:2, 20] = [2, 3]
    planners[0, 22] = 1
    # Row 0 is one swap. Row 1 is a three-cycle. Across the 64 AIVs the five
    # logical positions are split by global cuts; the pair has one owner.
    owner = lambda pos: ((pos+1)*64+4)//5-1
    marks[owner(0), 16] = 1
    marks[[owner(i) for i in (2,3,4)], 17] = 1
    marks[[16,48], 19] = 1
    moved = 2*marks[:,16]+marks[:,17]
    early = _preplan_detail(marks, moved, {"event_count": 3})
    assert early["early_union"] == 27 and early["early_active_max"] == 11
    assert early["index_union_including_early"] == 52 and early["first_preplan_to_last_index"] == 117
    state = np.tile(np.arange(8), (16,1))
    after = state.copy()
    after[0,:2] = [1,0]
    after[1,:3] = [2,0,1]
    profile = dict(moved=moved.tolist(), preplan=early)
    kwargs = dict(two_cycle_moves=[2]+[0]*15, invalidations=[2]+[0]*15)
    validate_move_counts(profile, state, after, np.arange(16), **kwargs)
    assert profile["persistent_moves_checked"]
    profile["preplan"]["row_pairs"][1] = 1
    with pytest.raises(ValueError, match="persistent Index permutation"):
        validate_move_counts(profile, state, after, np.arange(16), **kwargs)
    marks[0, 14] = 1
    with pytest.raises(ValueError, match="misplaced"):
        _preplan_detail(marks, moved, {"event_count": 3})
