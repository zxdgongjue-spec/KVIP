# SPDX-License-Identifier: Apache-2.0
"""Real packed files, checksums and bit-exact incremental tensor snapshots."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from kvip.diagnostics.archive import TensorArchive, load_snapshot
from kvip.diagnostics.storage import PackedObjectStore, TENSOR_CHUNK_BYTES


def test_object_reuse_flush_reopen_and_range_reads(tmp_path):
    directory = tmp_path / "dp0" / "objects"
    snapshot = tmp_path / "dp0" / "round0001" / "group000"
    snapshot.mkdir(parents=True)
    value = bytes(range(256)) * 33
    with PackedObjectStore(directory, create=True) as writer:
        first = writer.put(value)
        assert writer.put(memoryview(value)) == first
        assert (directory / "packs" / first["pack"]).stat().st_size == len(value)
        locator = writer.describe_for(snapshot)
        # Oracle readers must see exact bytes before writer.close()/DB commit.
        with PackedObjectStore.from_description(snapshot, locator) as reader:
            assert reader.read(first, offset=250, length=19) == value[250:269]
            assert reader.read(first) == value
        writer.flush()
    with PackedObjectStore(directory, create=True) as writer:
        assert writer.put(value) == first
        assert (directory / "packs" / first["pack"]).stat().st_size == len(value)
        second = writer.put(value + b"x")
        assert second["offset"] == len(value)
        assert writer.read(second) == value + b"x"


def test_writer_lock_and_unrelated_directory_are_rejected(tmp_path):
    directory = tmp_path / "objects"
    with PackedObjectStore(directory, create=True):
        with pytest.raises(BlockingIOError):
            PackedObjectStore(directory, create=True)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep").write_text("data")
    with pytest.raises(FileExistsError):
        PackedObjectStore(unrelated, create=True)
    assert (unrelated / "keep").read_text() == "data"


def test_reference_bounds_identity_corruption_and_path_confinement(tmp_path):
    snapshot = tmp_path / "round0001" / "group000"
    snapshot.mkdir(parents=True)
    with PackedObjectStore(tmp_path / "objects", create=True) as store:
        reference = store.put(b"known bytes")
        locator = store.describe_for(snapshot)
        with pytest.raises(ValueError, match="identity"):
            PackedObjectStore.from_description(snapshot, dict(locator, store_id="0" * 32))
        with pytest.raises(ValueError, match="locator"):
            PackedObjectStore.from_description(snapshot, dict(locator, relative_path=str(store.directory)))
        for change in ({"pack": "../escape"}, {"offset": -1}, {"offset": True}, {"length": 99}, {"sha256": "bad"}):
            with pytest.raises(ValueError):
                store.read(dict(reference, **change))
        with pytest.raises(ValueError, match="exceeds"):
            store.read(reference, offset=3, length=999)
        path = store.directory / "packs" / reference["pack"]
        with path.open("r+b") as output:
            output.write(b"X")
        with pytest.raises(ValueError, match="checksum"):
            store.read(reference)
        outside = tmp_path / "outside"
        outside.write_bytes(b"known bytes")
        path.unlink()
        path.symlink_to(outside)
        with pytest.raises(ValueError, match="escapes"):
            store.read(reference)


def test_packs_rotate_and_reader_handles_are_bounded(tmp_path, monkeypatch):
    from kvip.diagnostics import storage

    monkeypatch.setattr(storage, "PACK_BYTES", 64)
    with PackedObjectStore(tmp_path / "objects", create=True) as store:
        values = [bytes([i]) * 48 for i in range(12)]
        refs = [store.put(value) for value in values]
        assert len(list((store.directory / "packs").glob("*.bin"))) == 12
        for ref, value in zip(refs, values):
            assert store.read(ref) == value
        assert len(store._readers) == 8
        assert all(path.stat().st_size <= 64 for path in (store.directory / "packs").glob("*.bin"))
        with pytest.raises(ValueError, match="bounded"):
            store.put(bytes(65))


def test_tensor_chunks_reuse_unchanged_data_and_preserve_every_byte(tmp_path, monkeypatch):
    first_directory, second_directory = (tmp_path / f"round{i:04d}" / "group000" for i in (1, 2))
    first_directory.mkdir(parents=True)
    second_directory.mkdir(parents=True)
    data = np.arange(TENSOR_CHUNK_BYTES * 2 + 20, dtype=np.uint32).view(np.uint8)
    first_manifest = dict(schema_version=2, status="complete", tensors={})
    second_manifest = copy.deepcopy(first_manifest)
    with PackedObjectStore(tmp_path / "objects", create=True) as store:
        first = TensorArchive(first_directory, first_manifest, object_store=store)
        first.save("bits", data)
        first.save("empty", np.empty((2, 0, 4), dtype=np.float32))
        changed = data.copy()
        changed[TENSOR_CHUNK_BYTES + 5] ^= 1
        second = TensorArchive(second_directory, second_manifest, object_store=store)
        second.save("bits", changed)
        before_refs, after_refs = (
            first_manifest["tensors"]["bits"]["chunks"],
            second_manifest["tensors"]["bits"]["chunks"],
        )
        assert len(before_refs) == len(after_refs)
        assert [i for i, (a, b) in enumerate(zip(before_refs, after_refs)) if a != b] == [1]
        np.testing.assert_array_equal(first.array("bits", verify=True), data)
        np.testing.assert_array_equal(second.array("bits", verify=True), changed)
        assert first.array("empty", verify=True).shape == (2, 0, 4)
        assert not (first_directory / "tensors").exists()
        with pytest.raises(FileExistsError):
            first.save("bits", data)
        store.flush()
        for directory, manifest in ((first_directory, first_manifest), (second_directory, second_manifest)):
            (directory / "manifest.json").write_text(json.dumps(manifest))
    loaded = load_snapshot(first_directory)
    try:
        from kvip.diagnostics import archive as archive_module

        monkeypatch.setattr(archive_module, "TENSOR_CHUNK_BYTES", 1 << 10)
        # Reads use the captured layout, not a future writer's chunk setting.
        loaded.verify()
        np.testing.assert_array_equal(loaded.array("bits"), data)
    finally:
        loaded.close()


def test_packed_torch_unusual_dtypes_keep_raw_bits(tmp_path):
    directory = tmp_path / "round0001" / "group000"
    directory.mkdir(parents=True)
    with PackedObjectStore(tmp_path / "objects", create=True) as store:
        archive = TensorArchive(directory, {"tensors": {}}, object_store=store)
        for name, source in (
            ("bf16", torch.tensor([0, -1, 32767, -32768], dtype=torch.int16).view(torch.bfloat16)),
            ("fp8", torch.arange(256, dtype=torch.int16).to(torch.uint8).view(torch.float8_e4m3fn)),
            ("address", torch.tensor([0, -1, -(1 << 63)], dtype=torch.int64)),
        ):
            archive.save_tensor(name, torch, source)
            restored = archive.tensor(name, torch, "cpu")
            assert restored.dtype == source.dtype
            assert torch.equal(restored.view(torch.uint8), source.view(torch.uint8))


def test_manifest_chunk_and_aggregate_checksums_are_enforced(tmp_path):
    directory = tmp_path / "round0001" / "group000"
    directory.mkdir(parents=True)
    with PackedObjectStore(tmp_path / "objects", create=True) as store:
        manifest = {"tensors": {}}
        archive = TensorArchive(directory, manifest, object_store=store)
        archive.save("data", np.arange(48, dtype=np.uint8))
        description = manifest["tensors"]["data"]
        original = copy.deepcopy(description)
        description["chunks"][0]["length"] -= 1
        with pytest.raises(ValueError, match="chunk sizes"):
            archive.verify()
        manifest["tensors"]["data"] = copy.deepcopy(original)
        manifest["tensors"]["data"]["sha256"] = hashlib.sha256(b"wrong").hexdigest()
        with pytest.raises(ValueError, match="checksum"):
            archive.verify()
        with pytest.raises(ValueError, match="checksum"):
            archive.array("data", verify=True)
