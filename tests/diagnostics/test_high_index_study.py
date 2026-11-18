# SPDX-License-Identifier: Apache-2.0
"""The two work orders must differ only in execution order inside the captured graph."""

from types import SimpleNamespace

import numpy as np
import pytest

from kvip.diagnostics.high_index_profile import (
    decode_high_index, descriptor_fingerprint, event_work, validate_move_counts, verify_permutation,
)
from kvip.diagnostics.high_index_study import GROUPS, ROLES, equivalent_savings, replay_command
from kvip.diagnostics.replay import arguments


def test_every_high_study_invocation_uses_the_real_parser(tmp_path):
    args = SimpleNamespace(cache=tmp_path, source=tmp_path, revision="a"*40,
                           server="127.0.0.1:29084", result_dir=tmp_path)
    outputs = set()
    for group in GROUPS:
        for role in ROLES:
            argv = replay_command(args, group, role)
            parsed = arguments(argv[4:])
            assert parsed.high_post_index == (not role.startswith("original"))
            assert parsed.post_index_wait_first == role.startswith("wait-first")
            assert parsed.diagnostic_replays == (2 if role in ("wait-first", "overlap") else 0)
            assert parsed.calibrate_counter == bool(parsed.diagnostic_replays)
            assert parsed.pipeline_variants == ["auto-post-index"] and parsed.check_auto_route
            assert parsed.iterations == 10 and parsed.warmup == 2
            assert parsed.output not in outputs and not parsed.profile_dir
            outputs.add(parsed.output)


@pytest.mark.parametrize("wait_first", [False, True])
def test_high_diagnostic_checks_work_order_and_per_core_fallback(wait_first):
    records = np.zeros((64,16), np.int64)
    marks = np.zeros((64,32), np.int64)
    records[:,14] = 2
    records[:,2] = 2 * int(wait_first)
    records[1,2] += 1  # One core alone enters blocking Get.
    records[:,7] = 1
    marks[:,2] = 0x50494831
    order = [3,7,4,5,6] if wait_first else [3,4,5,6,7]
    for tick, field in enumerate(order, 2):
        marks[:,field] = tick
    phase = decode_high_index(records, marks, wait_first=wait_first, final_get=0, work={})
    assert phase["fallback"] and phase["fallback_cores"] == 1 and phase["active_aiv"] == 0
    assert phase["index_span"] == 1 and "joined_span" not in phase
    marks[:,13] = 9
    phase = decode_high_index(records, marks, wait_first=wait_first, final_get=0, work={})
    assert phase["joined_span"] == 7
    marks[2,13] = 0
    with pytest.raises(ValueError, match="collective"):
        decode_high_index(records, marks, wait_first=wait_first, final_get=0, work={})
    marks[:,13] = 9
    marks[63, order[-1]] = 1
    with pytest.raises(ValueError, match="Unordered"):
        decode_high_index(records, marks, wait_first=wait_first, final_get=0, work={})
    records[0,14] = 0
    with pytest.raises(ValueError, match="Incomplete HIGH"):
        decode_high_index(records, None, wait_first=wait_first, final_get=0, work={})


def test_index_work_ignores_unused_event_slots_preserves_order():
    flags = np.zeros((1,16,32), dtype=np.int32)
    flags[:,[0,12,13,14,15],0] = 1
    flags[:,0,1] = 2
    src = np.zeros((1,10240), dtype=np.int32)
    dst = src.copy()
    src[0,:2] = [12,13]
    dst[0,:2] = [0,1]
    def fingerprint():
        return event_work(flags, src, dst, history=32)["event_sha256"]
    ref = fingerprint()
    src[0,2222] = -1
    assert fingerprint() == ref
    src[0,:2] = [13,12]
    assert fingerprint() != ref


def test_get_sha_is_relocation_invariant_and_checks_live_addresses():
    posts = np.zeros((1,4,32), np.int32)
    posts[0,0,0] = 1
    indices = np.zeros(8192, np.int32)
    a = np.full((1,8192), 140000, np.int64)
    idx = (indices, indices, indices)
    ref = descriptor_fingerprint(posts, idx, a, final_get=1, arena_base=120000)
    assert descriptor_fingerprint(posts, idx, a+4000, final_get=1, arena_base=124000) == ref
    a[0,0] += 65536
    assert descriptor_fingerprint(posts, idx, a, final_get=1, arena_base=120000) != ref


def test_verified_physical_row_counts_include_displaced_rows_and_row_reordering():
    ids = np.tile(np.arange(12), (17,1))
    after = ids.copy()
    after[9,:5] = [1,2,3,4,0]
    owners = [9, *range(9), *range(10,16)]
    moved = [0]*64
    moved[0::16] = [1,1,1,2]
    record = dict(moved=moved)
    validate_move_counts(record, ids, after, owners)
    assert record["persistent_moves_checked"]
    record["moved"][0] += 1
    with pytest.raises(ValueError, match="verified pre/post"):
        validate_move_counts(record, ids, after, owners)


def test_exact_work_matches_exclude_fallback_and_differing_descriptors():
    def case(name, count, elapsed, fallback):
        rows = [dict(high_post_index=dict(event_sha256="e", descriptor_sha256=name,
                                          final_get_by_core=[1]*64, fallback=fallback),
                     prefetch=100, final=64, unused_prefetch=12)] * (count+1)
        return dict(_samples=[500]+[elapsed]*count, reads_samples=rows)
    cases = {"wait-first": case("A", 3, 500, False), "overlap": case("A", 1, 480, False),
             "wait-first-end": case("B", 1, 450, False), "overlap-end": case("B", 1, 440, True)}
    assert equivalent_savings(cases) == (1, 20)



def test_high_permutation_composes_parallel_chains_then_resolves_next_phase_source():
    flags = np.zeros((1,16,32), np.int32)
    flags[0,0,1] = 2
    flags[0,12,1] = 1
    src = np.zeros((1,10240), np.int32)
    dst = src.copy()
    src[0,:2] = [0,1]
    dst[0,:2] = [1,2]      # closure 2->0 makes the first phase a 3-cycle
    src[0,2048] = 0        # original 0 is physically at 1 after that cycle
    dst[0,2048] = 3
    before = np.array([[5,9,11,18]], np.int64)
    after = np.array([[11,18,9,5]], np.int64)
    verify_permutation(flags, src, dst, before, after, [0])
    with pytest.raises(ValueError, match="permutation differs"):
        verify_permutation(flags, src, dst, before, before, [0])
