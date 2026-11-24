"""CPU checks for changing the workspace limit without changing the 128K workload."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'support'))
from cap_pair_input import bind_capacity
from native_li_reference import digest
from workloads import ReplayPagePlan, ReplayRound, ReplaySequence


class CapacityInputTest(unittest.TestCase):
    def setUp(self):
        h, b = 2048, 2
        ids = np.arange(h, dtype=np.int32)
        self.source = ReplayRound(
            1, 51, ((0, 'a', 1), (0, 'b', 2)),
            dict(batch=b, history=h, token_bytes=656),
            np.broadcast_to(np.arange(128, dtype=np.uint8), (b, h, 128)),
            np.ones((b, h), np.float32) * 2,
            np.zeros((b, 4, 32, 128), np.uint8), np.ones((b, 4, 32), np.uint16),
            np.ones((b, 4, 32), np.float32), np.tile([2037, 2038, 2039, 2040], (b, 1)),
            np.zeros((b, 2), dtype=np.int64), np.ones((b, h), np.uint8),
            tuple(((0, 9), None) + tuple([(row, p) for p in range(2, h // 128)]) for row in range(b)),
            np.stack([ids, ids[::-1]]), np.zeros((b, 512), np.uint8), np.ones((b, 512), np.int32),
            ('capture',), np.ones((b, 4, h), np.uint16),
            np.broadcast_to(ids[None, None, :] < np.arange(2037, 2041)[None, :, None], (b, 4, h)))
        self.sequence = ReplaySequence.__new__(ReplaySequence)
        self.sequence.rounds = (('capture',),)
        self.sequence.cached_ranked_round = self.source
        self.pages = ReplayPagePlan(self.sequence)

    def test_padding_preserves_active_storage_and_shared_private_ddr(self):
        old = self.source.geometry['history']
        sequence, wide, pages, proof = bind_capacity(self.sequence, self.source, self.pages, 2 * old, digest)
        self.assertIs(next(iter(sequence)), wide)
        self.assertIs(next(iter(self.sequence)), self.source)
        self.assertEqual(wide.geometry['history'], 2 * old)
        self.assertTrue(np.array_equal(wide.query_ends, self.source.query_ends))
        for axis, names in [(1, ('logical_keys', 'logical_scales', 'captured_layout', 'logical_address_valid')),
                            (2, ('logical_score_keys', 'logical_score_valid'))]:
            for name in names:
                prefix = getattr(wide, name).take(np.arange(old), axis=axis)
                np.testing.assert_array_equal(prefix, getattr(self.source, name))
        self.assertFalse(wide.logical_keys[:, old:].any())
        self.assertTrue((wide.logical_scales[:, old:] == 1).all())
        self.assertFalse(wide.logical_address_valid[:, old:].any())
        self.assertFalse(wide.logical_score_valid[:, :, old:].any())
        np.testing.assert_array_equal(wide.captured_layout[:, old:], np.tile(np.arange(old, old*2), (2, 1)))
        self.assertEqual(proof['captured_page_count'], proof['bound_page_count'])
        self.assertEqual(pages.page_count, self.pages.page_count)
        np.testing.assert_array_equal(pages.page_indices[:, :old//128], self.pages.page_indices)
        self.assertFalse(pages.page_indices[:, old//128:].any())
        self.assertEqual(pages.page_indices[0, 0], pages.page_indices[1, 0])
        self.assertNotEqual(pages.page_indices[0, 1], pages.page_indices[1, 1])
        for row in (0, 1):
            ids = np.array([0, 127, 128, 767, old-1])
            np.testing.assert_array_equal(pages.token_payload(row, ids), self.pages.token_payload(row, ids))
        _, short, original, baseline = bind_capacity(self.sequence, self.source, self.pages, old, digest)
        self.assertIs(short, self.source)
        self.assertIs(original, self.pages)
        for key in proof.keys() - {'bound_history'}:
            self.assertEqual(proof[key], baseline[key])

    def test_rejects_truncating_source_or_bad_active_bounds(self):
        with self.assertRaises(ValueError):
            bind_capacity(self.sequence, self.source, self.pages, 1024, digest)
        self.source.query_ends[0, 2] -= 1
        with self.assertRaises(ValueError):
            bind_capacity(self.sequence, self.source, self.pages, 4096, digest)


if __name__ == '__main__':
    unittest.main()
