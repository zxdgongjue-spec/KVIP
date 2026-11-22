"""Check candidate ownership and duplicate-bit safety at warp boundaries."""
import itertools
import unittest


class WarpPartitionTest(unittest.TestCase):
    def test_every_candidate_once_and_no_mixed_q_warp(self):
        for counts in itertools.product((0, 1, 11, 31, 32, 33, 94, 255, 256, 2048), repeat=3):
            padded = [(n + 31) // 32 * 32 for n in counts]
            boundaries = (padded[0], padded[0] + padded[1])
            seen = []
            for begin in range(0, sum(padded), 32):
                owners = set()
                for j in range(begin, begin + 32):
                    q = 0 if j < boundaries[0] else 1 if j < boundaries[1] else 2
                    offset = 0 if q == 0 else boundaries[q - 1]
                    owners.add(q)
                    if j - offset < counts[q]:
                        seen.append((q, j - offset))
                self.assertEqual(len(owners), 1, (counts, begin))
            self.assertEqual(seen, [(q, i) for q, n in enumerate(counts) for i in range(n)])

    def test_duplicate_token_bits_must_not_be_added(self):
        bit = 1 << 12
        self.assertNotEqual(bit + bit, bit | bit)
        # Each Q's unique-token mask is reduced separately, then atomically ORed.
        per_q = [sum(1 << token for token in q) for q in ((12, 13), (12, 14))]
        self.assertEqual(per_q[0] | per_q[1], sum(1 << token for token in (12, 13, 14)))


if __name__ == "__main__":
    unittest.main()
