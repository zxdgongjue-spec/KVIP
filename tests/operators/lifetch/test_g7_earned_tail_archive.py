# SPDX-License-Identifier: Apache-2.0
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from g7_earned_tail_archive import chunks, inspect_call


class TailIssueCounts(unittest.TestCase):
    def test_pacing_boundary_and_separate_private_queues(self):
        call = dict(row_query_final=[[252, 2, 1, 0], [252, 2, 1, 1]],
                    row_window_prefetch=[[0]*5, [0]*5], prefetch=0, final=511, final_cqes=11)
        result = inspect_call(call, 2)
        self.assertEqual(result["long_rows"], 1)
        self.assertEqual(result["parts"][0], 255 + 16 + 1 + 1 + 1)
        self.assertEqual(sum(result["parts"]), 511)
        self.assertEqual(result["cq"], 11)
        call["final_cqes"] = 10
        with self.assertRaises(ValueError):
            inspect_call(call, 2)

    def test_all_supported_counts_match_the_receipt_rule(self):
        for n in range(2049):
            for paced in (True, False):
                result = chunks(n, paced)
                self.assertEqual(sum(result), n)
                self.assertEqual(len(result), min(n,4) if paced else int(n > 0))


if __name__ == "__main__":
    unittest.main()
