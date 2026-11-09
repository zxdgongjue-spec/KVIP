# SPDX-License-Identifier: Apache-2.0
"""CPU fake-Runtime checks for page coverage, bounds and fixed-slot replay."""

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from kvip.diagnostics.ddr import export_pages, import_pages, rebase_addresses, validate_pages


class CpuTorch:
    """Use real CPU tensors while retaining the production NPU API calls."""

    def __getattr__(self, name):
        return getattr(torch, name)

    def device(self, name):
        if name != "npu:0":
            raise AssertionError(name)
        return torch.device("cpu")


class Runtime:
    transport = "ubmem"
    device_id = 0
    batch_get_token_bytes = 656
    batch_get_tokens_per_block = 2

    def __init__(self, base=0x100000, slots=12):
        self.remote_region = dict(base=base, length=slots * 4096, slot_size=4096)
        self.pages = {}
        self.handles = {}
        self.get_calls = []
        self.put_calls = []
        self.bad_address = self.bad_length = self.invalid = self.fail_get = False

    def seed(self, keys):
        for key in keys:
            self.pages[key] = (np.arange(1312, dtype=np.int64) + key * 17).astype(np.uint8)

    def prepare_access_table(self, keys):
        if any(key not in self.pages for key in keys):
            raise ValueError("missing key")
        handle = max(self.handles, default=0) + 1
        self.handles[handle] = keys
        return handle

    def access_table_view(self, handle, device="cpu"):
        if device == "npu":
            return handle
        keys = self.handles[handle]
        addresses = np.asarray([self.remote_region["base"] + key * 4096 for key in keys], dtype=np.uint64)
        lengths = np.full(len(keys), 1312, dtype=np.uint64)
        valid = np.ones(len(keys), dtype=np.uint8)
        if self.bad_address:
            addresses[0] += 4096
        if self.bad_length:
            lengths[0] -= 1
        if self.invalid:
            valid[0] = 0
        # Signed high-bit addresses must retain their exact bit pattern.
        return dict(addresses=addresses.view(np.int64), lengths=lengths, valid=valid)

    def release_access_table(self, handle):
        del self.handles[handle]

    def batch_get(self, table, indices, offsets, count, destinations, output, *, synchronize):
        self.assert_pinned = bool(self.handles)
        if not synchronize or not self.assert_pinned:
            raise AssertionError("export must synchronously read pinned pages")
        self.get_calls.append((indices.tolist(), offsets.tolist(), output.numel()))
        if self.fail_get:
            raise RuntimeError("completed Get failed")
        for i in range(int(count[0])):
            key = self.handles[table][int(indices[i])]
            offset = int(offsets[i])
            output[int(destinations[i])].copy_(torch.from_numpy(self.pages[key][offset : offset + 656]))

    def batch_put(self, keys, source, offsets, lengths):
        self.put_calls.append((keys, source.numel()))
        for key, offset, length in zip(keys, offsets, lengths):
            if key in self.pages:
                raise ValueError("key already exists")
            self.pages[key] = source[offset : offset + length].numpy().copy()


class DDRTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.runtime = Runtime()
        self.runtime.seed([1, 4, 9])

    def export(self, **kwargs):
        return export_pages(
            self.runtime,
            [4, 1, 4, 9],
            self.path,
            page_mode="thp",
            generation="before-7",
            chunk_bytes=2624,
            torch_module=CpuTorch(),
            **kwargs,
        )

    def test_full_pages_dedup_chunk_pins_and_relocation(self):
        manifest = self.export()
        self.assertEqual(manifest["record_count"], 3)
        self.assertEqual(manifest["generation"], "before-7")
        self.assertEqual([call[2] for call in self.runtime.get_calls], [2624, 1312])
        self.assertEqual(self.runtime.get_calls[0][:2], ([0, 0, 1, 1], [0, 656, 0, 656]))
        self.assertEqual(self.runtime.handles, {})
        self.assertEqual(
            (self.path / "payload.bin").read_bytes(), b"".join(self.runtime.pages[key].tobytes() for key in [4, 1, 9])
        )
        replay = Runtime(base=0x900000)
        result = import_pages(replay, self.path, manifest, chunk_bytes=2624, torch_module=CpuTorch())
        self.assertEqual([call[1] for call in replay.put_calls], [2624, 1312])
        self.assertEqual(list(replay.pages), [4, 1, 9])
        for key in replay.pages:
            np.testing.assert_array_equal(replay.pages[key], self.runtime.pages[key])
        self.assertIn(result["access_table"], replay.handles)
        expected = np.array([0, 0x900000 + 4 * 4096 + 656], dtype=np.int64)
        source = np.array([0, 0x100000 + 4 * 4096 + 656], dtype=np.int64)
        np.testing.assert_array_equal(
            rebase_addresses(source, self.path, manifest, result["new_base"], width=656), expected
        )
        replay.release_access_table(result["access_table"])

    def test_high_address_signed_bits(self):
        self.runtime.remote_region["base"] = (1 << 63) + 0x100000
        manifest = self.export()
        source = np.array([manifest["region_base"] + 4096 + 656, 0], dtype=np.uint64).view(np.int64)
        target = rebase_addresses(source, self.path, manifest, (1 << 63) + 0x800000, width=656)
        self.assertEqual(target.dtype, np.dtype("int64"))
        self.assertEqual(int(target.view(np.uint64)[0]), (1 << 63) + 0x800000 + 4096 + 656)
        self.assertEqual(int(target[1]), 0)

    def test_missing_keys_rejected_before_payload(self):
        self.runtime.pages.pop(9)
        with self.assertRaisesRegex(ValueError, "missing"):
            self.export()
        self.assertEqual(self.runtime.handles, {})
        self.assertFalse((self.path / "payload.bin").exists())

    def test_invalid_or_wrong_geometry_records_release_pin(self):
        for fault in ("bad_address", "bad_length", "invalid"):
            with self.subTest(fault=fault):
                setattr(self.runtime, fault, True)
                with self.assertRaises(ValueError):
                    self.export()
                self.assertEqual(self.runtime.handles, {})
                setattr(self.runtime, fault, False)

    def test_get_failure_releases_handle(self):
        self.runtime.fail_get = True
        with self.assertRaisesRegex(RuntimeError, "Get failed"):
            self.export()
        self.assertEqual(self.runtime.handles, {})

    def test_rebase_rejects_holes_padding_cross_boundary_and_outside(self):
        manifest = self.export()
        base = manifest["region_base"]
        for address, width in (
            (base + 2 * 4096, 656),
            (base + 4096 + 1312, 1),
            (base + 4096 + 656, 657),
            (base - 1, 1),
            (base + manifest["region_bytes"], 1),
        ):
            with self.subTest(address=address, width=width), self.assertRaises(ValueError):
                rebase_addresses(np.array([address], dtype=np.uint64), self.path, manifest, 0x800000, width=width)

    def test_checksum_and_truncation_fail_before_put(self):
        manifest = self.export()
        path = self.path / "payload.bin"
        original = path.read_bytes()
        path.write_bytes(original[:-1])
        replay = Runtime(base=0x900000)
        with self.assertRaisesRegex(ValueError, "checksum"):
            import_pages(replay, self.path, manifest, torch_module=CpuTorch())
        self.assertEqual(replay.put_calls, [])
        forged = copy.deepcopy(manifest)
        forged["files"]["payload"]["sha256"] = hashlib.sha256(original[:-1]).hexdigest()
        with self.assertRaisesRegex(ValueError, "truncated"):
            import_pages(replay, self.path, forged, torch_module=CpuTorch())
        self.assertEqual(replay.put_calls, [])

    def test_restore_geometry_must_match(self):
        manifest = self.export()
        replay = Runtime(slots=11)
        with self.assertRaisesRegex(ValueError, "geometry"):
            import_pages(replay, self.path, manifest, torch_module=CpuTorch())
        self.assertEqual(replay.put_calls, [])

    def test_array_checksum_and_path_validation(self):
        manifest = self.export()
        forged = copy.deepcopy(manifest)
        forged["files"]["keys"]["path"] = "../keys.npy"
        with self.assertRaisesRegex(ValueError, "path"):
            import_pages(Runtime(), self.path, forged, torch_module=CpuTorch())
        with (self.path / "keys.npy").open("ab") as target:
            target.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            import_pages(Runtime(), self.path, manifest, torch_module=CpuTorch())

    def test_rejects_overwrite_and_too_small_chunk(self):
        self.export()
        with self.assertRaises(FileExistsError):
            self.export()
        with self.assertRaisesRegex(ValueError, "chunk"):
            export_pages(
                self.runtime,
                [1],
                self.path / "small",
                page_mode="base",
                generation=1,
                chunk_bytes=656,
                torch_module=CpuTorch(),
            )

    def test_empty_capture(self):
        manifest = export_pages(self.runtime, [], self.path, page_mode="base", generation=0, torch_module=CpuTorch())
        self.assertEqual(manifest["payload_bytes"], 0)
        replay = Runtime(base=0x900000)
        result = import_pages(replay, self.path, manifest, torch_module=CpuTorch())
        self.assertIsNone(result["access_table"])
        self.assertEqual(replay.get_calls + replay.put_calls, [])
        np.testing.assert_array_equal(
            rebase_addresses(np.zeros((2, 3), dtype=np.int64), self.path, manifest, result["new_base"]),
            np.zeros((2, 3), dtype=np.int64),
        )

    def test_public_validation_has_no_address_and_rejects_bad_policy(self):
        manifest = self.export()
        summary = validate_pages(self.path, manifest)
        self.assertTrue(summary["checksums_validated"])
        self.assertEqual(summary["record_count"], 3)
        self.assertNotIn("region_base", summary)
        forged = dict(manifest, page_mode="base", numa_node=0)
        with self.assertRaisesRegex(ValueError, "NUMA"):
            validate_pages(self.path, forged)

    def test_key_outside_arena_fails_before_pin(self):
        self.runtime.pages[12] = self.runtime.pages[1]
        with self.assertRaisesRegex(ValueError, "arena"):
            export_pages(self.runtime, [12], self.path, page_mode="thp", generation=1, torch_module=CpuTorch())
        self.assertEqual(self.runtime.handles, {})

    def test_vectorized_rebase_unsorted_slots_and_uint64_width_boundary(self):
        manifest = self.export()
        rng = np.random.default_rng(19)
        keys = rng.choice([1, 4, 9], size=(20, 50)).astype(np.uint64)
        within = rng.choice([0, 656], size=keys.shape).astype(np.uint64)
        addresses = np.uint64(manifest["region_base"]) + keys * np.uint64(4096) + within
        addresses[::3, ::5] = 0
        new_base = ((1 << 64) - 1 - manifest["region_bytes"]) // 32 * 32
        expected = np.where(addresses != 0, np.uint64(new_base) + keys * np.uint64(4096) + within, 0)
        actual = rebase_addresses(addresses, self.path, manifest, new_base, width=656)
        np.testing.assert_array_equal(actual, expected)
        with self.assertRaises(ValueError):
            rebase_addresses(addresses, self.path, manifest, new_base, width=(1 << 64) - 1)


if __name__ == "__main__":
    unittest.main()
