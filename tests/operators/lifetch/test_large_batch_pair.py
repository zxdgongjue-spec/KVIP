# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the large-batch input identity and profiler accounting."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from large_batch_input import BATCHES, FrozenRound, expand, input_digest
from large_batch_pair import check_scores, large_account, li_intervals, stage_sample, tail_team_intervals, verify_abc, verify_assignments
from workloads import ReplayPagePlan, ReplayRound


def fixture():
    rows = np.arange(16)
    values = rows[:, None]
    identities = tuple((i // 2, str(i), i % 2) for i in rows.tolist())
    pages = tuple(((i // 2, 0), (i // 2, i + 1)) for i in rows.tolist())
    return ReplayRound(
        1,
        101,
        identities,
        dict(batch=16, history=256, token_bytes=656),
        values,
        values,
        values,
        values,
        values,
        values,
        values,
        values,
        pages,
        values,
        values,
        values,
        ("example",),
        values,
        values,
    )


class LargeBatchReplay(unittest.TestCase):
    def test_tail_teams_two_waves_with_one_slow_query(self):
        stamps = np.zeros((20, 4, 32), dtype=np.int64)
        stamps[:16, :, 0], stamps[:16, :, 10] = 100, 200
        stamps[16:, :, 0], stamps[16:, :, 10] = 200, 300
        stamps[19, 2, 10] = 320
        summary = tail_team_intervals(stamps, 2)
        self.assertAlmostEqual(summary["mean_assigned_teams"], 2020 / 220)
        self.assertEqual((summary["row_max_us"], summary["last_row"], summary["last_q"]), (60, 19, 3))
        stamps[4, 1, 10] = 0
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            tail_team_intervals(stamps, 2)

    def test_rows_aliases_and_pair_identity(self):
        base = fixture()
        for batch in BATCHES:
            data, donors = expand(base, batch)
            np.testing.assert_array_equal(data.logical_keys.ravel(), donors)
            self.assertEqual(len(set(data.identities)), batch)
            pages = ReplayPagePlan(FrozenRound(data, []))
            np.testing.assert_array_equal(pages.page_indices[:16:2, 0], pages.page_indices[1:16:2, 0])
            self.assertFalse(np.intersect1d(pages.page_indices[:16], pages.page_indices[16:]).size)
            clone, _ = expand(base, batch)
            self.assertEqual(input_digest(data, pages, donors), input_digest(clone, pages, donors))
            clone.captured_hot_valid[0] += 1
            self.assertNotEqual(input_digest(data, pages, donors), input_digest(clone, pages, donors))

    def test_merged_li_scope_and_median_inputs(self):
        scope = dict(name="measured", ph="X", ts=100, dur=28)
        events = [scope]
        for name, a, d in [
            ("v9_li_large_hot_0", 101, 4),
            ("v9_li_large_threshold_cold", 107, 9),
            ("v9_parallel_terminal_large_1", 119, 5),
            ("v9_parallel_terminal_large", 7, 9),
        ]:
            events.append(dict(name=name, ph="X", cat="kernel", ts=a, dur=d))
        got = stage_sample(large_account(events, "measured", 25), True)
        self.assertEqual(
            (got["graph"], got["LI"], got["tail"], got["span"], got["inter_kernel_gap"]), (25, 13, 5, 23, 5)
        )
        with self.assertRaises(ValueError):
            stage_sample(large_account(events[:-2], "measured", 25), True)
        native = dict(event_us=30, intervals_us=dict(li=[(0, 11)], lidu_prepare=[(12, 17)], fetch=[(19, 28)]))
        self.assertEqual(stage_sample(native, False)["tail"], 14)

    def test_cann_mixed_task_names_and_unknown_compute(self):
        def device(name, start, width, typ, task):
            return dict(
                name=name,
                ph="X",
                pid=1,
                tid=27,
                ts=start,
                dur=width,
                args={"Task Type": typ, "Physic Stream Id": 27, "Task Id": task},
            )

        markers = [
            dict(name="sample", ph="X", ts=0, dur=50),
            dict(name="process_name", ph="M", pid=1, args={"name": "Ascend Hardware"}),
        ]
        events = markers + [
            device("MODEL_EXECUTE", 1, 1, "MODEL_EXECUTE", 21),
            device("v9_li_large_hot_mix_aic_0", 10, 4, "KERNEL_MIX_AIC", 22),
            device("v9_li_large_hot_mix_aiv_0", 11, 4, "KERNEL_MIX_AIV", 22),
            device("v9_li_large_threshold_cold_0", 16, 8, "MIX_AIC", 23),
            device("v9_parallel_terminal_large_12", 25, 7, "KERNEL_AIVEC", 24),
        ]
        self.assertEqual(stage_sample(large_account(events, "sample", 40), True)["LI1"], 5)
        with self.assertRaisesRegex(ValueError, "Unattributed"):
            large_account(events + [device("unexpected", 33, 1, "AI_VECTOR_CORE", 25)], "sample", 40)
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            large_account(
                [*events[:4], device("v9_li_large_hot_mix_aiv", 11, 4, "KERNEL_MIX_AIV", 199), *events[5:]],
                "sample",
                40,
            )
        native = markers + [
            device("QuantLightningIndexer", 2, 8, "MIX_AIC", 3),
            device("A5LiduFusedBatchGetPrepareMtp", 10, 3, "AI_VECTOR_CORE", 4),
            device("KvafExecuteBatchGetAivKernelV2", 14, 4, "AI_VECTOR_CORE", 5),
        ]
        self.assertEqual(stage_sample(large_account(native, "sample", 30, fused=False), False)["tail"], 7)
        with self.assertRaisesRegex(ValueError, "Unattributed"):
            large_account(
                native + [device("aclnnAdd_CastAiCore_Cast", 19, 4, "AI_VECTOR_CORE", 6)], "sample", 30, fused=False
            )

    def test_only_cutoff_ties_allowed(self):
        keys = np.broadcast_to(np.array([3, 2, 2, 1]), (1, 4, 4))
        inputs = SimpleNamespace(geometry=dict(batch=1), logical_score_keys=keys)

        def four(a):
            return np.broadcast_to(a, (1, 4, len(a)))

        self.assertEqual(check_scores(inputs, four([0, 2]), four([0, 1])), 0)
        with self.assertRaises(ValueError):
            check_scores(inputs, four([0, 3]), four([0, 1]))

    def test_ordered_index_displaced_ids_and_validity(self):
        initial = np.arange(6)
        before_valid = np.array([1, 1, 0])
        # Admitting src[3,0] to dst[1,2] retires slot 0; the released
        # sources receive the former destination IDs in reservation order.
        layout = np.array([2, 3, 0, 1, 4, 5])
        valid = np.array([0, 1, 1])
        verify_assignments(initial, layout, before_valid, valid, [3, 0], [1, 2])
        with self.assertRaises(ValueError):
            verify_assignments(initial, layout, before_valid, valid, [0, 3], [1, 2])
        with self.assertRaises(ValueError):
            verify_assignments(initial, layout, before_valid, np.ones(3), [3, 0], [1, 2])

    def test_abc_bounds_and_circular_slots(self):
        ids = np.array([[[7, 9, 10]]])
        slots = np.array([[[0, 17, 18]]])
        bounds = np.array([[9, 12]])
        verify_abc(slots, ids, bounds, 16, 20)
        with self.assertRaises(ValueError):
            verify_abc(slots, ids, np.array([[10, 12]]), 16, 20)
        with self.assertRaises(ValueError):
            verify_abc(slots, ids + 1, bounds, 16, 20)

    def test_overlapping_query_stamps(self):
        words = np.zeros(128, dtype=np.uint64)
        for q in range(4):
            words[64 + 16 * q : 68 + 16 * q] = [10 + q, 20 + q, 21 + q, 40 + q]
        actual = li_intervals(words, 1, 1)["spans"]
        self.assertEqual(actual["LI2_active_us"], 13)
        self.assertEqual(actual["LI3_active_us"], 22)
        self.assertEqual(actual["overlap_us"], 2)
        with self.assertRaises(ValueError):
            li_intervals(np.zeros_like(words), 1, 1)


if __name__ == "__main__":
    unittest.main()
