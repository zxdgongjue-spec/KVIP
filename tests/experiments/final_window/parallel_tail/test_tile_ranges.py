"""CPU regression for the bounded bitmap-exclusion loop used by the model."""
import random
import unittest


class TileRangesTest(unittest.TestCase):
    def test_excluded_intervals_match(self):
        rng = random.Random(19)
        for _ in range(10000):
            base = rng.randrange(4096) * 128
            count = rng.randrange(129) * 128
            begin = rng.randrange(base + count + 1024)
            end = begin + rng.randrange(2048)
            original = [
                (max(base + i, begin), min(base + i + 128, end))
                for i in range(0, count, 128)
                if max(base + i, begin) < min(base + i + 128, end)
            ]
            bounded = []
            if begin < end and begin < base + count and end > base:
                first = max(0, begin - base) // 128 * 128
                stop = min(count, end - base)
                bounded = [
                    (max(base + i, begin), min(base + i + 128, end))
                    for i in range(first, stop, 128)
                ]
            self.assertEqual(original, bounded)


if __name__ == '__main__':
    unittest.main()
