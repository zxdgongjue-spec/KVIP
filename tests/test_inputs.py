import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from kvip.fused_inputs import FusedQueryInputs
from kvip.fused_workspace import FusedWorkspace
from kvip.index_storage import IndexCacheStorage


class FusedInputsTest(unittest.TestCase):
    def test_fp8_pack_uses_byte_gather_preserving_all_bit_patterns(self):
        native_select = torch.index_select

        def supported_select(source, dim, index, *, out=None):
            self.assertNotEqual(source.dtype, torch.float8_e4m3fn)
            return native_select(source, dim, index, out=out)

        for heads in (32, 64):
            inputs = FusedQueryInputs(torch, 2, torch.device("cpu"), pin_memory=False, index_heads=heads)
            inputs.prepare([1, 2])
            raw = torch.arange(3 * heads * 128).remainder(256).to(torch.uint8).reshape(3, heads, 128)
            query = raw.view(torch.float8_e4m3fn)
            pointer = inputs.query.data_ptr()
            with patch.object(torch, "index_select", side_effect=supported_select):
                packed, _, _ = inputs.pack(query, torch.ones(3, heads, dtype=torch.bfloat16), torch.ones(3, heads))
            self.assertEqual(packed.data_ptr(), pointer)
            self.assertEqual(packed.dtype, query.dtype)
            torch.testing.assert_close(packed.view(torch.uint8), raw[[0, 0, 0, 0, 1, 2, 0, 0]])

    def test_mixed_query_counts_and_padding(self):
        inputs = FusedQueryInputs(torch, 3, torch.device("cpu"), pin_memory=False)
        inputs.prepare([1, 4])
        ends = inputs.bind_positions(torch.tensor([57, 10000, 10001, 10002, 10003]))
        self.assertEqual(ends.tolist(), [[58, 0, 0, 0], [10001, 10002, 10003, 10004], [0, 0, 0, 0]])
        query = torch.arange(5).float()[:, None, None].expand(-1, 64, 128).to(torch.float8_e4m3fn)
        weights = torch.arange(5).bfloat16()[:, None].expand(-1, 64)
        scale = torch.arange(5).float()[:, None].expand(-1, 64)
        q, w, s = inputs.pack(query, weights, scale)
        self.assertEqual(q[:, 0, 0].float().tolist(), [0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0])
        self.assertTrue(torch.equal(w.float(), s))
        locations = torch.arange(12 * 2).reshape(3, 4, 2)
        self.assertTrue(torch.equal(inputs.unpack(locations, 5), locations.reshape(12, 2)[[0, 4, 5, 6, 7]]))
        padded = inputs.unpack(locations, 8)
        self.assertTrue(torch.all(padded[5:] == -1))

    def test_replay_metadata_and_mtp_rollback_keep_addresses(self):
        inputs = FusedQueryInputs(torch, 2, torch.device("cpu"), pin_memory=False)
        pointers = (inputs.metadata.data_ptr(), inputs.query_ends.data_ptr(), inputs.query.data_ptr())
        inputs.prepare([4, 4])
        inputs.bind_positions(torch.tensor([300, 301, 302, 303, 500, 501, 502, 503]))
        inputs.prepare([2, 1])
        result = inputs.bind_positions(torch.tensor([301, 302, 501]))
        self.assertEqual(result.tolist(), [[302, 303, 0, 0], [502, 0, 0, 0]])
        self.assertEqual(pointers, (inputs.metadata.data_ptr(), inputs.query_ends.data_ptr(), inputs.query.data_ptr()))
        inputs.prepare([0, 0])
        self.assertEqual(inputs.bind_positions(torch.tensor([0])).count_nonzero().item(), 0)

    def test_reject_incompatible_shapes_before_launch(self):
        inputs = FusedQueryInputs(torch, 2, torch.device("cpu"), pin_memory=False)
        for counts in ([5], [1, 1, 1], [-1], [True]):
            with self.assertRaises(ValueError):
                inputs.prepare(counts)
        with self.assertRaises(ValueError):
            inputs.pack(torch.zeros(4, 32, 128), torch.zeros(4, 64), None)

    def test_32_head_service_quantizer_flattened_heads_preserve_output_storage(self):
        inputs = FusedQueryInputs(torch, 2, torch.device("cpu"), pin_memory=False, index_heads=32)
        inputs.prepare([1, 2])
        query = torch.arange(3 * 32).remainder(16).float()[:, None].expand(-1, 128).to(torch.float8_e4m3fn)
        scales = torch.arange(3 * 32).float()
        weights = torch.ones((3, 32), dtype=torch.bfloat16)
        pointers = tuple(t.data_ptr() for t in (inputs.query, inputs.weights, inputs.scale))
        q, _, s = inputs.pack(query, weights, scales)
        expected = torch.tensor([0, 0, 0, 0, 1, 2, 0, 0])
        torch.testing.assert_close(q.float(), query.view(3, 32, 128).float()[expected])
        torch.testing.assert_close(s, scales.view(3, 32)[expected])
        self.assertEqual(pointers, tuple(t.data_ptr() for t in (inputs.query, inputs.weights, inputs.scale)))
        with self.assertRaisesRegex(ValueError, "one scale per token/head"):
            inputs.pack(query, weights, torch.ones(3))

    def test_service_quantizer_flattened_heads_preserve_output_storage(self):
        inputs = FusedQueryInputs(torch, 2, torch.device("cpu"), pin_memory=False)
        inputs.prepare([1, 2])
        query = torch.arange(3 * 64).remainder(16).float()[:, None].expand(-1, 128).to(torch.float8_e4m3fn)
        scales = torch.arange(3 * 64).float()
        weights = torch.ones((3, 64), dtype=torch.bfloat16)
        pointers = tuple(t.data_ptr() for t in (inputs.query, inputs.weights, inputs.scale))
        q, _, s = inputs.pack(query, weights, scales)
        expected = torch.tensor([0, 0, 0, 0, 1, 2, 0, 0])
        torch.testing.assert_close(q.float(), query.view(3, 64, 128).float()[expected])
        torch.testing.assert_close(s, scales.view(3, 64)[expected])
        self.assertEqual(pointers, tuple(t.data_ptr() for t in (inputs.query, inputs.weights, inputs.scale)))
        with self.assertRaisesRegex(ValueError, "one scale per token/head"):
            inputs.pack(query, weights, torch.ones(3))

    def test_owner_mapping_shares_metadata_upload_and_clears_padding(self):
        inputs = FusedQueryInputs(torch, 3, torch.device("cpu"), pin_memory=False)
        address = inputs.row_owners.data_ptr()
        inputs.prepare([1, 2], query_starts=[10, 20], query_owners=[7, 3])
        self.assertEqual(inputs.row_owners.tolist(), [7, 3, 0])
        self.assertEqual(inputs.gather_indices.tolist(), [10, 0, 0, 0, 20, 21, 0, 0, 0, 0, 0, 0])
        inputs.prepare([1], query_owners=[2])
        self.assertEqual(inputs.row_owners.tolist(), [2, 0, 0])
        self.assertEqual(inputs.row_owners.data_ptr(), address)


class WorkspaceTest(unittest.TestCase):
    def test_arguments_alias_owner_storage_and_shared_workspace(self):
        workspace = FusedWorkspace(
            torch, batch=2, history=4096, aics=32, windows=8, device=torch.device("cpu"), pin_memory=False
        )
        first = IndexCacheStorage(torch, 4, 4096, 2048, "cpu")
        second = IndexCacheStorage(torch, 4, 4096, 2048, "cpu")
        a, b = (workspace.arguments(first), workspace.arguments(second))
        self.assertEqual(len(a), 14)
        self.assertEqual(a[1].data_ptr(), first.keys.data_ptr())
        self.assertEqual(b[1].data_ptr(), second.keys.data_ptr())
        for index in (0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13):
            self.assertEqual(a[index].data_ptr(), b[index].data_ptr())
        self.assertEqual(workspace.query_lengths.tolist(), [4, 4])
        self.assertEqual(workspace.key_lengths.tolist(), [4096, 4096])
        address = workspace.inputs.block_table.data_ptr()
        workspace.inputs.prepare([1, 4], query_owners=[3, 1])
        self.assertEqual(workspace.inputs.block_table[:, 0].tolist(), [96, 32])
        self.assertEqual(workspace.inputs.block_table[:, -1].tolist(), [127, 63])
        workspace.inputs.prepare([2], query_owners=[2])
        self.assertEqual(workspace.inputs.block_table[:, 0].tolist(), [64, 0])
        self.assertEqual(workspace.inputs.block_table.data_ptr(), address)
        self.assertGreater(workspace.nbytes, 16 * 1024 * 1024)
        previous = workspace.nbytes
        workspace.allocate_rotation(torch, batch=2, history=4096, resident=2048, aics=32, windows=8)
        self.assertEqual(tuple(workspace.scratch["mapping"].shape), (2, 4096))
        self.assertEqual(tuple(workspace.scratch["lru_slots"].shape), (2, 2048))
        self.assertEqual(
            workspace.nbytes - previous, sum(t.numel() * t.element_size() for t in workspace.scratch.values())
        )
        with self.assertRaisesRegex(RuntimeError, "already allocated"):
            workspace.allocate_rotation(torch, batch=2, history=4096, resident=2048, aics=32, windows=8)


class IndexStorageTest(unittest.TestCase):
    def setUp(self):
        self.storage = IndexCacheStorage(torch, 3, 32, 8, torch.device("cpu"))
        self.cache = (
            torch.arange(8, dtype=torch.uint8)[:, None].expand(-1, 128).clone().view(4, 2, 1, 128),
            torch.arange(8).float().view(4, 2, 1),
        )

    def test_prefix_import_once_and_recycled_owner_clears_only_metadata(self):
        plan = NS(owner=1, request="a", start=2, begin=0, stop=2)
        metadata = NS(block_size=2, block_table=torch.tensor([[2, 0]]), slot_mapping=torch.tensor([0, 1]))
        self.storage.prepare([plan])
        self.storage.ingest_prefill(self.cache, metadata, [plan], torch.tensor([2, 3]))
        self.assertEqual(self.storage.keys[1, :4, 0].tolist(), [4, 5, 0, 1])
        self.assertEqual(self.storage.scales[1, :4].tolist(), [4, 5, 0, 1])
        self.storage.keys[1, 0].fill_(123)
        self.storage.hot_valid[1, 0] = 1
        self.storage.prepare([plan])
        self.storage.ingest_prefill(self.cache, metadata, [plan], torch.tensor([2, 3]))
        self.assertEqual(int(self.storage.keys[1, 0, 0]), 123)
        self.assertEqual(int(self.storage.hot_valid[1, 0]), 1)
        self.storage.logical_ids[1, 0] = 17
        self.storage.addresses[1].fill_(1000)
        self.storage.prepare([NS(owner=1, request="b")])
        self.assertNotIn(1, self.storage.imported)
        self.assertEqual(int(self.storage.hot_valid[1].sum()), 0)
        self.assertEqual(int(self.storage.addresses[1].sum()), 0)
        torch.testing.assert_close(self.storage.logical_ids[1], self.storage.identity)

    def test_committed_address_append_preserves_permutation_and_group_stride(self):
        self.storage.prepare([NS(owner=1, request="a")])
        self.storage.publish(1, [10000, 20000], block_size=2, token_bytes=2624)
        self.assertEqual(self.storage.addresses[1, :4].tolist(), [10000, 12624, 20000, 22624])
        self.storage.addresses[1, [0, 2]] = self.storage.addresses[1, [2, 0]]
        saved = self.storage.addresses[1, :4].clone()
        self.storage.publish(1, [10000, 20000, 30000], block_size=2, token_bytes=2624)
        torch.testing.assert_close(self.storage.addresses[1, :4], saved)
        self.assertEqual(self.storage.addresses[1, 4:6].tolist(), [30000, 32624])

    def test_decode_dynamic_owners_and_corrected_positions_do_not_copy_history(self):
        inputs = FusedQueryInputs(torch, 2, torch.device("cpu"), pin_memory=False)
        inputs.prepare([1, 1])
        inputs.bind_positions(torch.tensor([4, 5]))
        owners = torch.tensor([1, 2], dtype=torch.int32)
        metadata = NS(slot_mapping=torch.tensor([6, 7]))
        self.storage.keys[1:].fill_(99)
        self.storage.ingest_decode(self.cache, metadata, inputs, owners)
        self.assertEqual(int(self.storage.keys[1, 4, 0]), 6)
        self.assertEqual(int(self.storage.keys[2, 5, 0]), 7)
        self.assertTrue(torch.all(self.storage.keys[1:, :4] == 99))
        owners.copy_(torch.tensor([2, 1], dtype=torch.int32))
        inputs.bind_positions(torch.tensor([3, 4]))
        self.storage.ingest_decode(self.cache, metadata, inputs, owners)
        self.assertEqual(int(self.storage.keys[2, 3, 0]), 6)
        self.assertEqual(int(self.storage.keys[1, 4, 0]), 7)
        self.assertTrue(torch.all(self.storage.keys[1:, :3] == 99))
