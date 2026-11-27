# SPDX-License-Identifier: Apache-2.0
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).with_name("support")))
from four_rank_prefetch_ready import decode_visibility


class VisibilityTest(unittest.TestCase):
    def setUp(self):
        self.stamps, self.detail, self.probes = ([[[0]*32 for _ in range(4)]] for _ in range(3))
        t = self.stamps[0][1]
        t[16], t[24], t[17] = 100, 200, 220
        self.detail[0][1][0] = 2
        self.detail[0][1][11], self.detail[0][1][15] = 120, 150
        r = self.probes[0][1]
        r[:15] = [101, 103, 7, 4, 1, 122, 126, 7, 4, 2, 153, 158, 7, 4, 2]
        r[25:30] = [209, 215, 11, 0, 3]

    def test_partial_then_all_visible_only_original_drain_consumes(self):
        row, = decode_visibility(self.stamps, self.detail, self.probes)
        self.assertEqual([s["all_visible"] for s in row["observations"]], [False, True, True])
        self.assertEqual(row["cursor_after"], 11)

    def test_no_prefetch_and_no_final_reads(self):
        self.detail[0][1] = [0]*32
        p = [0]*32
        p[:5], p[25:30] = [101,103,5,0,3], [210,215,5,0,3]
        self.probes[0][1] = p
        row, = decode_visibility(self.stamps, self.detail, self.probes)
        self.assertEqual(len(row["observations"]), 1)
        self.assertEqual(row["cursor_after"], 5)

    def test_rejects_corrupt_lease_cursor_status_and_causality(self):
        for index, value in [(4, 4), (7, 8), (8, 3), (14, 1), (27, 10), (19, 2), (25, 180), (5,111)]:
            with self.subTest(index=index):
                raw = copy.deepcopy(self.probes)
                raw[0][1][index] = value
                with self.assertRaises(ValueError):
                    decode_visibility(self.stamps, self.detail, raw)
        self.probes[0][0][0] = 1
        with self.assertRaisesRegex(ValueError, "non-owner"):
            decode_visibility(self.stamps, self.detail, self.probes)
