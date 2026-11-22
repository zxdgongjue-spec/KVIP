"""Exhaust the votes and arrival orders of the request-local dependency join."""
import itertools
import unittest


class DependencyJoinTest(unittest.TestCase):
    def test_no_early_release_and_no_vote_leaks_between_rounds(self):
        for votes in itertools.product((False, True), repeat=4):
            for order in itertools.permutations(range(4)):
                counter = 0
                for completed, q in enumerate(order, 1):
                    counter += 1 + (256 if votes[q] else 0)
                    self.assertEqual((counter & 255) == 4, completed == 4)
                self.assertEqual(bool(counter >> 8), any(votes))
                # Reset is permitted only after every participant has advanced
                # to the later completion join, not merely after arrival four.
                completion = set()
                for q in reversed(order):
                    completion.add(q)
                    if len(completion) == 4:
                        counter = 0
                for _ in range(4):
                    counter += 1
                self.assertEqual(counter, 4)
                self.assertEqual(counter >> 8, 0)
