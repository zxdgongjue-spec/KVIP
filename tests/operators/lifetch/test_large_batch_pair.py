# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the large-batch input identity and profiler accounting."""

from pathlib import Path
from types import SimpleNamespace
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from large_batch_input import BATCHES, FrozenRound, expand, input_digest, load, save_source
from fixed_bs16_report import BATCHES as FIXED_BATCHES, PUBLISHED_EVENT_PER_CALL, combined_pass
from large_batch_pair import check_scores, large_account, last_row_windows, li_intervals, stage_sample, tail_team_intervals, verify_abc, verify_assignments
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
    def test_packed_and_single_event_gates_are_independent(self):
        cells = [dict(group=g, batch=b, serial_graph_us=100, fused_graph_us=80)
                 for g in PUBLISHED_EVENT_PER_CALL for b in FIXED_BATCHES]
        packed = [dict(group=g, context_matched=True, new_packed_us=[110, 90], published_packed_us=[99, 101])
                  for g in PUBLISHED_EVENT_PER_CALL]
        self.assertEqual(combined_pass(cells, packed), (45, 5))
        cells[0]['fused_graph_us'] = 80.01
        packed[0]['context_matched'] = False
        packed[-1]['new_packed_us'] = [115, 90]
        self.assertEqual(combined_pass(cells, packed), (44, 3))
        with self.assertRaises(ValueError):
            combined_pass(cells[:-1] + cells[:1], packed)
        with self.assertRaises(ValueError):
            combined_pass(cells, packed[:-1] + packed[:1])
        cells[0]['serial_graph_us'] = float('nan')
        with self.assertRaises(ValueError):
            combined_pass(cells, packed)

    def test_last_row_window_clock_partition(self):
        stamps = np.zeros((20, 4, 4), dtype=np.uint64)
        stamps[:, :, 2:] = (100, 200)
        stamps[7, :, 2:] = (100, 210)
        li = dict(per_query_cycle_stamps=stamps, clock_mhz=2)
        events = np.zeros((4, 20, 16), dtype=np.uint64)
        events[:, 7, 2] = np.arange(4, dtype=np.uint64) << 32
        events[:, 7, 3] = 1
        events[0, 7, 1] = 110
        for epoch in (1, 2, 3):
            start = 110 + 30 * (epoch - 1)
            events[epoch, 7, [4, 13, 14, 1]] = [start + 12, start + 15, start + 24, start + 30]
        result = last_row_windows(events, li)
        self.assertEqual((result["row"], result["span_us"]), (7, 55))
        self.assertEqual(result["segments_us"], dict(initial_window=5, cold_to_window=18, predecessor_wait=4.5,
                                                     rank_and_join=13.5, service_and_post=9, retire=5))
        events[1, 7, 13] = 0
        self.assertEqual(last_row_windows(events, li)["reason"], "cold_window_clock")

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

    def test_fixed_small_batches_cover_all_donors_once(self):
        for batch in (1, 2, 4, 8, 16):
            cohorts = 16 // batch
            observed = []
            for i in range(cohorts):
                data, donors = expand(fixture(), batch, cohort=i)
                self.assertEqual(len(set(donors)), batch)
                np.testing.assert_array_equal(data.logical_keys.ravel(), donors)
                self.assertEqual(ReplayPagePlan(FrozenRound(data, [])).geometry["batch"], batch)
                observed.extend(donors)
            self.assertEqual(sorted(observed), list(range(16)))
            with self.assertRaises(ValueError):
                expand(fixture(), batch, cohort=cohorts)

    def test_prepared_bank_preserves_operand_and_page_fingerprint(self):
        from workloads import CaptureReference
        source = fixture()
        ref = CaptureReference(0, 1, 101, 10, Path("example"), (("a", 1), ("b", 2)))
        with TemporaryDirectory() as d:
            bank = Path(d) / "donors"
            trace = Path(d) / "source"
            save_source(bank, trace, 10, 1, source, [ref])
            for b, cohort in ((1, 14), (8, 1), (16, 0), (32, 0)):
                stream, donors = load(trace, 10, 1, b, cohort, bank)
                raw, expected_rows = expand(source, b, cohort)
                self.assertEqual(donors, expected_rows)
                self.assertEqual(stream.rounds, ((ref,),))
                self.assertEqual(input_digest(raw, ReplayPagePlan(FrozenRound(raw, [ref])), donors),
                                 input_digest(stream.inputs, ReplayPagePlan(stream), donors))
            with self.assertRaises(ValueError):
                load(trace, 2, 1, 8, 0, bank)

    def test_small_batch_kernel_accounting(self):
        marker = dict(name="sample", ph="X", ts=90, dur=60)
        for bs in (1, 2, 4, 8, 16):
            names = ("v9_li_kernel", "v9_li_kernel_1", "v9_li_kernel_2") if bs == 16 else (
                "v9_li_fixed_hot", "v9_li_kernel_1", "v9_li_fixed_cold")
            events = [marker]
            for i, name in enumerate((*names, "v9_parallel_terminal")):
                events.append(dict(name=name + "_1", ph="X", cat="kernel", ts=100 + i * 7, dur=i + 1))
            stages = stage_sample(large_account(events, "sample", 30, batch=bs), True)
            self.assertEqual((stages["LI1"], stages["LI2"], stages["LI3"], stages["LI"], stages["tail"]),
                             (1, 2, 3, 6, 4))
            if bs == 16:
                for missing in (2, 3):
                    with self.assertRaisesRegex(ValueError, "expected three"):
                        large_account(events[:missing] + events[missing+1:], "sample", 30, batch=bs)
                invalid = list(events)
                invalid[2] = dict(invalid[2], name="v9_li_kernel_16_1_1")
                with self.assertRaisesRegex(ValueError, "wrong PR30 LI2"):
                    large_account(invalid, "sample", 30, batch=bs)
            else:
                events[1] = dict(events[1], name="v9_li_kernel")
                with self.assertRaisesRegex(ValueError, "Unattributed"):
                    large_account(events, "sample", 30, batch=bs)

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
