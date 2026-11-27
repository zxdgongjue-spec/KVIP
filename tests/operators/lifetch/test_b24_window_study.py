# SPDX-License-Identifier: Apache-2.0
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))
from b24_window_policy import MODES, policy, stage, trace_lanes


class WindowStudy(unittest.TestCase):
    def test_ranks_and_budget(self):
        for mode in MODES:
            ranks, grants = policy(mode)
            x = int(mode.split("-")[1][1:])
            self.assertEqual((len(ranks), len(grants), sum(grants)), (1+x, 1+x, 1024))
            self.assertEqual((ranks[0], ranks[-1], grants[0]), (1024, 2048, 256))
            self.assertTrue(all(a < b for a, b in zip(ranks, ranks[1:])))
            self.assertEqual(grants[1:], (768//x,) * x)
        self.assertEqual(policy("window-x3-early")[0], (1024, 1568, 1792, 2048))
        self.assertEqual(policy("window-x3-linear")[0], (1024, 1365, 1707, 2048))

    def test_trace_words_do_not_overwrite_ranks_or_neighbour(self):
        words = np.zeros((32, 4, 32), dtype=np.uint64)
        ranks = policy("window-x8-early")[0]
        words[:, :, :9] = ranks
        for e in range(8):
            for lane in range(2):
                q = 2*lane + e//4
                slot = 16 + 4*(e%4)
                words[:, q, slot:slot+4] = 100*lane + 4*e + np.arange(4)
        for x in (2,3,4,8):
            clocks = trace_lanes(words, x)
            for lane in range(2):
                np.testing.assert_array_equal(clocks[31, lane], (100*lane+np.arange(4*x)).reshape(x,4))
        self.assertTrue((words[:, :, :9] == ranks).all())

    def test_image_key_changes_for_every_policy(self):
        digests = {}
        for mode in MODES:
            with TemporaryDirectory() as td:
                experiment = stage(SUPPORT/"large_two_cold_native", Path(td), mode)
                digests[mode] = (experiment/"policy.sha256").read_text()
        self.assertEqual(digests["window-x4-earned-medium"], digests["window-x4-earned"])
        self.assertEqual(len(set(digests.values())), len(MODES)-1)
