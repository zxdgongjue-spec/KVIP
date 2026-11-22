"""Integration contract model, not a real URMA queue test."""
import unittest


def queue_assignment(batch):
    prefetch = {2 * row for row in range(batch)}
    final = {(row, q): (2 * row + 1 if q == 0 else
                        2 * batch + 3 * row + q - 1)
             for row in range(batch) for q in range(4)}
    return prefetch, final


class QueueOwnershipTest(unittest.TestCase):
    def test_exclusive_views_and_capacity(self):
        for batch in (1, 2, 4, 8, 16):
            prefetch, final = queue_assignment(batch)
            self.assertEqual(len(set(final.values())), 4 * batch)
            self.assertFalse(prefetch.intersection(final.values()))
            capacity = max(5 * batch, 2 * batch + 16)
            self.assertLess(max(prefetch | set(final.values())), capacity)

    def test_legacy_spare_reset_conflicts_with_q1(self):
        for batch in (1, 2, 4, 8, 16):
            _, final = queue_assignment(batch)
            old_reset_targets = {2 * row + 1 for row in range(batch)}
            self.assertEqual(old_reset_targets,
                             {final[row, 0] for row in range(batch)})

    def test_prefetch_only_reset_preserves_all_final_statuses(self):
        for batch in (1, 2, 4, 8, 16):
            prefetch, final = queue_assignment(batch)
            status = {view: (row, q, "completed")
                      for (row, q), view in final.items()}
            expected = dict(status)
            for view in prefetch:
                status[view] = "prefetch completed"
            self.assertEqual({view: status[view] for view in final.values()}, expected)


if __name__ == "__main__":
    unittest.main()
