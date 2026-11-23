"""CPU regression checks for the parallel terminal's out-of-band Index receipt."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'support'))
from main_five_run import verify_assignments


class TerminalIndexTest(unittest.TestCase):
    def test_cross_query_displacement_in_reservation_order(self):
        # 0 -> 1 -> 2 is a resident chain. 6 -> 0 and 5 -> 3 bring cold
        # rows in; their displaced victims must return to 6 then 5.
        start = np.array([21, 17, 5, 2, 99, 31, 87])
        src, dst = [0, 1, 6, 5], [1, 2, 0, 3]
        final = start[[6, 0, 1, 5, 4, 3, 2]]
        verify_assignments(start, final, np.zeros(4), np.ones(4), src, dst)
        wrong = final.copy()
        wrong[[5, 6]] = wrong[[6, 5]]
        with self.assertRaisesRegex(ValueError, 'event replay'):
            verify_assignments(start, wrong, np.zeros(4), np.ones(4), src, dst)

    def test_unfilled_resident_source_and_empty_plan(self):
        start = np.arange(7)
        before = np.array([1, 0, 1, 1])
        verify_assignments(start, start, before, before, [], [])
        verify_assignments(start, start[[1, 0, 2, 3, 4, 5, 6]], before, [0, 1, 1, 1], [0], [1])
        for src, dst in (([2, 2], [0, 1]), ([2, 3], [1, 1]), ([-1], [0]), ([7], [1]), ([3], [4])):
            with self.assertRaisesRegex(ValueError, 'assignments'):
                verify_assignments(start, start, before, before, src, dst)


if __name__ == '__main__':
    unittest.main()
