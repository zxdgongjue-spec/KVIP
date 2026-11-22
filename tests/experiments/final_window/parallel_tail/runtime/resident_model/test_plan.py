import importlib.util
from pathlib import Path
import random
import unittest

spec = importlib.util.spec_from_file_location('resident_plan', Path(__file__).with_name('plan.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PermutationTests(unittest.TestCase):
    def verify(self, sources, destinations, history=32768, resident=16384):
        moves = module.plan(sources, destinations, history, resident)
        original = list(range(history))
        actual = module.apply(original, moves)
        self.assertEqual(sorted(actual), original)
        self.assertEqual([actual[d] for d in destinations], sources)
        for p in set(range(history)) - set(sources) - set(destinations):
            self.assertEqual(actual[p], p)

    def test_reported_resident_source(self):
        # Old independent-cold-swap precondition rejects precisely this source.
        self.assertLess(5732, 16384)
        self.verify([20000, 5732], [4100, 4820])

    def test_chain_across_four_partitions(self):
        self.verify([20000, 4820, 5732, 6100], [4820, 5732, 6100, 6200])

    def test_cycle_and_identity(self):
        self.verify([4820, 5732, 6100, 6200], [5732, 6100, 4820, 6200])

    def test_random_mixed_assignments(self):
        rng = random.Random(43)
        for count in (1, 57, 230, 512, 2048):
            sources = rng.sample(range(32768), count)
            destinations = rng.sample(range(16384), count)
            self.verify(sources, destinations)

    def test_invalid_inputs_remain_rejected(self):
        for sources, destinations in (([1, 1], [2, 3]), ([1, 2], [3, 3]),
                                     ([32768], [1]), ([1], [16384])):
            with self.assertRaises(ValueError):
                module.plan(sources, destinations, 32768, 16384)
