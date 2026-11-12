import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from kvip.operators.lifetch.index_update import DirectIndexUpdate

ROOT = Path(__file__).resolve().parents[4]

ROOT = Path(__file__).resolve().parents[4]

class IndexUpdateTest(unittest.TestCase):
    def setUp(self):
        self.rotating = SimpleNamespace(
            lru_slots=torch.zeros((2, 2048), dtype=torch.int32),
            change_slots=torch.zeros((2, 10240), dtype=torch.int32),
            change_count=torch.full((2,), 2048, dtype=torch.int32),
            changes=torch.full((2, 10240), -1, dtype=torch.int32),
            rows=torch.zeros((2, 256), dtype=torch.int32),
            hot_valid=torch.zeros((5, 128), dtype=torch.int32),
            row_status=torch.zeros((2, 32), dtype=torch.int32),
            final_posts=torch.zeros((2, 2, 32), dtype=torch.int32),
            prefetch_status=torch.zeros((4, 1216), dtype=torch.uint8),
            final_status=torch.zeros((4, 1216), dtype=torch.uint8),
            final_workers=2,
            owner=SimpleNamespace(abi_sizes={"status": 1216}),
        )
        self.inputs = dict(
            keys=torch.zeros((5, 256, 128), dtype=torch.int8),
            scales=torch.ones((5, 256)),
            addresses=torch.zeros((5, 256), dtype=torch.int64),
            logical_ids=torch.zeros((5, 256), dtype=torch.int64),
            row_owners=torch.tensor([4, 1], dtype=torch.int32),
            rotating=self.rotating,
        )
        self.torch = SimpleNamespace(
            zeros=torch.zeros,
            empty=torch.empty,
            tensor=torch.tensor,
            int32=torch.int32,
            int64=torch.int64,
            float32=torch.float32,
            npu=SimpleNamespace(current_stream=lambda device: SimpleNamespace(npu_stream=123)),
        )

    def make(self):
        return DirectIndexUpdate(self.torch, **self.inputs)

    def test_stable_owner_rows_and_replay_without_allocation(self):
        update = self.make()
        self.assertEqual(
            update.bindings.tolist(),
            [self.inputs[name].data_ptr() for name in ("keys", "scales", "addresses", "logical_ids")]
            + [
                self.rotating.change_slots.data_ptr(),
                self.rotating.changes.data_ptr(),
                update.status.data_ptr(),
                256,
                self.rotating.rows.data_ptr(),
                self.inputs["row_owners"].data_ptr(),
                self.rotating.row_status.data_ptr(),
                self.rotating.final_posts.data_ptr(),
                self.rotating.prefetch_status.data_ptr(),
                self.rotating.final_status.data_ptr(),
                2,
                1216,
                self.rotating.hot_valid.data_ptr(),
                0,
            ]
            + [0] * 6
            + [256, 128, self.rotating.change_count.data_ptr(), 0],
        )
        before = update.bindings.clone()
        self.inputs["row_owners"].copy_(torch.tensor([1, 3], dtype=torch.int32))
        self.assertTrue(torch.equal(before, update.bindings))
        self.assertFalse(callable(update))

    def test_service_output_binding_carries_runtime_geometry(self):
        query_inputs = SimpleNamespace(unpack_indices=torch.arange(8), output_valid=torch.ones(8, dtype=torch.int64))
        self.rotating.arguments = [torch.zeros(1)] * 9 + [torch.zeros(2, 4, 2048, dtype=torch.int32)]
        self.rotating.locations = torch.zeros(2, 4, 2048, dtype=torch.int32)
        self.inputs["query_inputs"] = query_inputs
        outputs = torch.zeros(2, 8, 1, 2048, dtype=torch.int32)
        self.inputs["outputs"] = outputs
        update = self.make()
        self.assertEqual(update.bindings.numel(), 28)
        self.assertEqual(update.bindings[22], outputs[0].data_ptr())
        self.assertEqual(update.bindings[23], outputs[1].data_ptr())
        self.assertFalse(callable(update))

    def test_reject_wrong_producer_batch(self):
        self.rotating.changes = self.rotating.changes[:1]
        with self.assertRaisesRegex(ValueError, "producer changes"):
            self.make()

    def test_fixture_reuses_production_binding_with_padded_address_rows(self):
        from index_update import DirectCacheUpdate

        self.rotating.hot_valid = self.rotating.hot_valid[:2]
        self.rotating.arguments = [torch.zeros(1)] * 9 + [torch.zeros(2, 4, 2048, dtype=torch.int32)]
        self.rotating.locations = torch.zeros(2, 4, 2048, dtype=torch.int32)
        addresses = torch.zeros(2, 300, dtype=torch.int64)[:, :256]
        update = DirectCacheUpdate(
            torch,
            self.inputs["keys"][:2],
            self.inputs["scales"][:2],
            addresses,
            None,
            2,
            256,
            128,
            topk=self.rotating.arguments[9],
            locations=self.rotating.locations,
            rotating=self.rotating,
        )
        self.assertIsInstance(update, DirectIndexUpdate)
        self.assertEqual(update.bindings[2], addresses.data_ptr())
        self.assertEqual(update.bindings[7], 300)
        self.assertEqual(update.bindings[22], update.logical_output.data_ptr())
        self.assertEqual(update.bindings[23], update.slot_output.data_ptr())
        self.assertEqual(update.bindings[24:26].tolist(), [256, 128])
