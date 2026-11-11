# SPDX-License-Identifier: Apache-2.0
"""Actual packed-store DDR exports remain reread, bounded and replayable."""

import copy
import hashlib

import numpy as np
import pytest

from kvip.diagnostics.ddr import (
    PayloadReader,
    export_pages,
    import_pages,
    materialize_pages,
    rebase_addresses,
    validate_pages,
)
from kvip.diagnostics.oracle import DecodeOracle
from kvip.diagnostics.storage import PackedObjectStore
from test_ddr import CpuTorch, OriginalUrmaRuntime, Runtime


@pytest.fixture
def pages(tmp_path):
    runtime = Runtime()
    runtime.seed([1, 4, 9])
    with PackedObjectStore(tmp_path / "objects", create=True) as store:
        yield runtime, store


def export(runtime, store, directory, keys=(4, 1, 9)):
    return export_pages(
        runtime,
        keys,
        directory,
        page_mode="thp",
        generation=directory.name,
        chunk_bytes=2624,
        torch_module=CpuTorch(),
        object_store=store,
    )


def stored_bytes(store):
    return sum(path.stat().st_size for path in (store.directory / "packs").glob("*.bin"))


def test_content_dedup_rereads_same_key_and_preserves_previous_round(tmp_path, pages):
    runtime, store = pages
    runtime.pages[4] = runtime.pages[1].copy()
    first = export(runtime, store, tmp_path / "round1" / "ddr", keys=(4, 1, 4, 9))
    assert first["schema_version"] == 2 and first["record_count"] == 3
    assert "payload" not in first["files"]
    assert not (tmp_path / "round1/ddr/payload.bin").exists()
    assert first["payload"]["blocks"][0] == first["payload"]["blocks"][1]
    assert stored_bytes(store) == 2 * 1312
    second = export(runtime, store, tmp_path / "round2" / "ddr")
    assert second["payload"]["blocks"] == first["payload"]["blocks"]
    assert stored_bytes(store) == 2 * 1312
    assert len(runtime.get_calls) == 4  # Two real BatchGet chunks per round.
    previous = runtime.pages[4].copy()
    runtime.pages[4][0] ^= 1  # Same key and native address now contain different bytes.
    third = export(runtime, store, tmp_path / "round3" / "ddr")
    assert third["payload"]["blocks"][0] != first["payload"]["blocks"][0]
    assert third["payload"]["blocks"][1:] == first["payload"]["blocks"][1:]
    assert stored_bytes(store) == 3 * 1312
    assert len(runtime.get_calls) == 6 and not runtime.handles
    with PayloadReader(tmp_path / "round1/ddr", first) as reader:
        np.testing.assert_array_equal(reader[0, :], previous)
    assert validate_pages(tmp_path / "round3/ddr", third)["checksums_validated"]


@pytest.mark.parametrize("transport", ("ubmem", "urma"))
def test_import_preserves_original_keys_gaps_chunking_and_high_bit_relocation(tmp_path, pages, transport):
    runtime, store = pages
    runtime.remote_region["base"] = (1 << 63) + 0x100000
    directory = tmp_path / "round" / "ddr"
    manifest = export(runtime, store, directory)
    replay = (Runtime if transport == "ubmem" else OriginalUrmaRuntime)(base=(1 << 63) + 0x900000)
    result = import_pages(
        replay, directory, manifest, target_transport=transport, chunk_bytes=2624, torch_module=CpuTorch()
    )
    assert list(replay.pages) == [4, 1, 9]
    assert [count for _, count in replay.put_calls] == [2624, 1312]
    for key in replay.pages:
        np.testing.assert_array_equal(replay.pages[key], runtime.pages[key])
    address = np.asarray([runtime.remote_region["base"] + 4 * 4096 + 656, 0], dtype=np.uint64).view(np.int64)
    relocated = rebase_addresses(address, directory, manifest, result["new_base"], width=656)
    assert int(relocated.view(np.uint64)[0]) == result["new_base"] + 4 * 4096 + 656
    assert relocated[1] == 0
    replay.release_access_table(result["access_table"])


def test_materialization_has_no_dependency_on_object_store(tmp_path, pages):
    runtime, store = pages
    directory = tmp_path / "round" / "ddr"
    manifest = export(runtime, store, directory)
    independent = tmp_path / "independent"
    legacy = materialize_pages(directory, manifest, independent)
    assert legacy["schema_version"] == 1 and "payload" not in legacy
    assert manifest["schema_version"] == 2 and "payload" in manifest
    data = b"".join(runtime.pages[key].tobytes() for key in (4, 1, 9))
    assert (independent / "payload.bin").read_bytes() == data
    assert legacy["files"]["payload"]["sha256"] == hashlib.sha256(data).hexdigest()
    store.close()
    store.directory.rename(tmp_path / "unavailable-objects")
    assert validate_pages(independent, legacy)["checksums_validated"]
    replay = Runtime(base=0x900000)
    result = import_pages(replay, independent, legacy, torch_module=CpuTorch())
    replay.release_access_table(result["access_table"])
    with pytest.raises(FileExistsError):
        materialize_pages(independent, legacy, independent)


def test_payload_reader_bounds_and_lru_avoid_repeated_hashes(tmp_path, pages, monkeypatch):
    runtime, store = pages
    directory = tmp_path / "round" / "ddr"
    manifest = export(runtime, store, directory)
    with PayloadReader(directory, manifest, cache_bytes=1312) as reader:
        calls = []
        original = reader._store.read

        def read(reference, **kwargs):
            calls.append(kwargs.get("verify", True))
            return original(reference, **kwargs)

        monkeypatch.setattr(reader._store, "read", read)
        np.testing.assert_array_equal(reader[0, 656:1312], runtime.pages[4][656:])
        np.testing.assert_array_equal(reader[0, :32], runtime.pages[4][:32])
        assert calls == [True]
        reader[1, :]
        reader[0, :]
        assert calls == [True, True, False]  # Evicted bytes are reread, not rehashed per token.
        assert reader._cache_bytes == 1312 and len(reader._cache) == 1
        assert reader.read_records(0, 2) == b"".join(runtime.pages[key].tobytes() for key in (4, 1))
        assert reader.read_records(1, 1) == b""
        for record, within in ((-1, slice(None)), (3, slice(None)), (0, slice(0, 1313)), (0, slice(None, None, 2))):
            with pytest.raises(ValueError):
                reader[record, within]
        with pytest.raises(ValueError):
            reader.read_records(2, 1)


@pytest.mark.parametrize("fault", ("bytes", "length", "reference_count", "digest", "path", "offset", "identity"))
def test_bad_cas_is_rejected_before_any_native_put(tmp_path, pages, fault):
    runtime, store = pages
    directory = tmp_path / "round" / "ddr"
    manifest = export(runtime, store, directory)
    bad = copy.deepcopy(manifest)
    ref = bad["payload"]["blocks"][0]
    if fault == "bytes":
        path = store.directory / "packs" / ref["pack"]
        with path.open("r+b") as stream:
            stream.seek(ref["offset"])
            stream.write(bytes([int(runtime.pages[4][0]) ^ 1]))
    elif fault == "length":
        ref["length"] -= 1
    elif fault == "reference_count":
        bad["payload"]["blocks"].pop()
    elif fault == "digest":
        bad["payload"]["sha256"] = "0" * 64
    elif fault == "path":
        ref["pack"] = "../store.json"
    elif fault == "offset":
        ref["offset"] = 1 << 40
    else:
        bad["payload"]["store"]["store_id"] = "0" * 32
    replay = Runtime(base=0x900000)
    with pytest.raises(ValueError):
        import_pages(replay, directory, bad, torch_module=CpuTorch())
    assert not replay.put_calls and not replay.handles


def test_empty_series_and_failed_get_keep_pin_contract(tmp_path, pages):
    runtime, store = pages
    directory = tmp_path / "empty" / "ddr"
    manifest = export(runtime, store, directory, keys=())
    assert manifest["payload"]["blocks"] == []
    assert validate_pages(directory, manifest)["payload_bytes"] == 0
    assert PayloadReader(directory, manifest).read_records(0, 0) == b""
    runtime.fail_get = True
    with pytest.raises(RuntimeError, match="Get failed"):
        export(runtime, store, tmp_path / "failed" / "ddr")
    assert not runtime.handles


def test_actual_oracle_reads_cas_with_legacy_identical_results(tmp_path):
    from test_snapshot import captured

    # The actual shared oracle fixture has resident/ABC/DDR-selected tokens.
    archive, state = captured.__wrapped__(tmp_path)
    expected = DecodeOracle(archive).verify(state)
    manifest = archive.manifest["ddr"]
    directory = archive.directory / "ddr"
    keys = np.load(directory / "keys.npy", allow_pickle=False)
    width = archive.manifest["geometry"]["token_bytes"] * manifest["tokens_per_block"]
    payload = (directory / "payload.bin").read_bytes()
    with PackedObjectStore(tmp_path / "objects", create=True) as store:
        blocks = [store.put(payload[start : start + width]) for start in range(0, len(payload), width)]
        store.flush()
        manifest["schema_version"] = 2
        manifest["record_count"] = len(keys)
        manifest["payload"] = dict(
            storage="cas_blocks",
            store=store.describe_for(directory),
            blocks=blocks,
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        manifest["files"].pop("payload")
        actual = DecodeOracle(archive).verify(state)
        for name in ("selected_tags", "selected_payload", "selected_references"):
            np.testing.assert_array_equal(actual[name], expected[name])
        assert actual["index_hashes"] == expected["index_hashes"]
