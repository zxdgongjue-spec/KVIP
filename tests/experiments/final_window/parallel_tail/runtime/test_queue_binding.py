import struct
import unittest

from queue_binding import bind_queues, final_queue, wqe_template


def view(index):
    words = [0] * 48
    words[6:8] = [0xABCDEF, 0x98765432]
    words[12] = 0xDEADBEEF
    words[14:18] = [0x100000 + index * 0x1000, 0x12, 0x200000 + index * 128, 0x3F]
    words[19:22] = [8192, 0xF1234567, 64]
    words[22:26] = [1, 2, 3, 4]
    words[26:30] = [0x300000 + index * 0x1000, 0x3F, 0x400000 + index * 128, 0x3F]
    return dict(entity_address=0x120000100000 + index * 192,
                queue_address=0x120000200000 + index * 64,
                output_address=0x123456789ABC,
                entity_bytes=struct.pack('<48I', *words))


class QueueBindingTest(unittest.TestCase):
    def test_disjoint_queues_for_all_supported_batches(self):
        for batch in (1, 2, 4, 8, 16):
            with self.subTest(batch=batch):
                bindings = bind_queues(batch, [view(i) for i in range(batch * 5)], 656)
                used = {b.view for b in bindings}
                prefetch = set(range(0, 2 * batch, 2))
                self.assertEqual(len(used), batch * 4)
                self.assertFalse(used & prefetch)
                self.assertEqual(used | prefetch, set(range(batch * 5)))
                self.assertEqual([(b.row, b.query) for b in bindings],
                                 [(r, q) for r in range(batch) for q in range(4)])

    def test_template_matches_production_wqe_fields(self):
        binding = view(0)
        words = wqe_template(binding['entity_bytes'], binding['output_address'], 656)
        self.assertEqual(words[1:10], (1536, 0x1234567, 0xBCDEF, 1, 2, 3, 4, 0x98765432, 0))
        self.assertEqual(words[12:18], (656, 0xDEADBEEF, 0x56789ABC, 0x1234, 0, 8192))
        self.assertEqual(words[18:20], (0, 0))  # Live producer is never snapshotted.
        self.assertEqual(words[24:28], (0x100000, 0x12, 0x200000, 0x3F))

    def test_shared_prefetch_or_final_queue_is_rejected(self):
        for field in ('queue_address', 'sq', 'cq', 'sq_db', 'cq_db'):
            with self.subTest(field=field):
                views = [view(i) for i in range(5)]
                if field == 'queue_address':
                    views[1][field] = views[0][field]
                else:
                    slot = dict(sq=7, sq_db=8, cq=13, cq_db=14)[field]
                    words = list(struct.unpack('<24Q', views[1]['entity_bytes']))
                    words[slot] = struct.unpack('<24Q', views[0]['entity_bytes'])[slot]
                    views[1]['entity_bytes'] = struct.pack('<24Q', *words)
                with self.assertRaisesRegex(ValueError, 'exclusive'):
                    bind_queues(1, views, 656)

    def test_output_rebinding_changes_wqe_destination(self):
        entity = view(0)['entity_bytes']
        a = wqe_template(entity, 0x120000000000, 656)
        b = wqe_template(entity, 0x130012345678, 656)
        self.assertNotEqual(a[14:16], b[14:16])
        self.assertEqual(a[:14] + a[16:], b[:14] + b[16:])

    def test_invalid_geometry(self):
        for args in ((0, 0, 0), (17, 0, 0), (1, 1, 0), (1, 0, 4)):
            with self.assertRaises(ValueError):
                final_queue(*args)


if __name__ == '__main__':
    unittest.main()
