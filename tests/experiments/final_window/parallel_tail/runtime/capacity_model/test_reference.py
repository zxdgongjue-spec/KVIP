import itertools
import random
import unittest

from reference import (commit_partitions, extend_destinations,
                       gather_partitions, parallel_permutation)


class CapacityTests(unittest.TestCase):
    def test_boundary_preserves_existing_destinations_and_later_q_hits(self):
        lru = list(range(4096, 6144))
        protected = set(range(0, 32)) | set(range(64, 128))
        for count in (2047, 2048, 2049, 2050, 2075, 2100, 4000):
            destinations = extend_destinations(lru, count, 16384, protected)
            self.assertEqual(len(destinations), count)
            self.assertEqual(len(set(destinations)), count)
            self.assertEqual(destinations[:2048], lru[:min(count, 2048)])
            self.assertFalse(set(destinations[2048:]) & protected)

    def test_exhausted_capacity_does_not_reuse_selected_hot_slot(self):
        with self.assertRaises(ValueError):
            extend_destinations([0, 1], 3, 4, {2, 3})

    def verify_parallel(self, sources, destinations, history=32768):
        values = list(range(history))
        partitions = parallel_permutation(sources, destinations)
        snapshots = gather_partitions(values, partitions)
        expected = None
        for order in itertools.permutations(range(4)):
            actual = commit_partitions(values, snapshots, order)
            self.assertEqual([actual[d] for d in destinations], sources)
            self.assertEqual(sorted(actual), values)
            if expected is None:
                expected = actual
            self.assertEqual(actual, expected)

    def test_cross_worker_chain_cycle_and_independent_tasks(self):
        self.verify_parallel([20000, 4820, 5732, 6100, 21000],
                             [4820, 5732, 6100, 6200, 6300])
        self.verify_parallel([4820, 5732, 6100, 20000],
                             [5732, 6100, 4820, 6200])

    def test_mixed_updates_across_capacity_boundary(self):
        rng = random.Random(43)
        for count in (230, 2048, 2049, 2100, 4000):
            self.verify_parallel(rng.sample(range(32768), count),
                                 rng.sample(range(16384), count))


if __name__ == '__main__':
    unittest.main()
