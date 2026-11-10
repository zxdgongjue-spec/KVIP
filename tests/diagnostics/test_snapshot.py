# SPDX-License-Identifier: Apache-2.0
"""Host tests exercise real snapshot bytes/oracles, not simulated device timing."""

import copy
import json
import shutil
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from kvip.diagnostics.archive import TensorArchive, array_digest, file_digest, load_snapshot
from kvip.diagnostics.oracle import DecodeOracle
from kvip.diagnostics.replay import arguments
from kvip.diagnostics.snapshot import (
    DecodeSnapshotRecorder,
    check_completion,
    decode_tagged_mapping,
    validate_capture_inputs,
)


def test_tensor_archive_preserves_fp8_bfloat16_and_unsigned_bits(tmp_path):
    manifest = dict(schema_version=1, status="complete", tensors={})
    archive = TensorArchive(tmp_path, manifest)
    for name, source in (
        ("fp8", torch.arange(256, dtype=torch.int16).to(torch.uint8).view(torch.float8_e4m3fn)),
        ("bf16", torch.tensor([0, -1, 32767, -32768], dtype=torch.int16).view(torch.bfloat16)),
        ("address", torch.tensor([0, -1, -(1 << 63)], dtype=torch.int64)),
    ):
        archive.save_tensor(name, torch, source)
        restored = archive.tensor(name, torch, "cpu")
        assert restored.dtype == source.dtype
        assert torch.equal(restored.view(torch.uint8), source.view(torch.uint8))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    load_snapshot(tmp_path)
    with (tmp_path / manifest["tensors"]["fp8"]["file"]).open("ab") as output:
        output.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_snapshot(tmp_path)


def test_manifest_rejects_pickle_escape_and_partial_publication(tmp_path):
    manifest = dict(schema_version=1, status="incomplete", tensors={})
    archive = TensorArchive(tmp_path, manifest)
    with pytest.raises(ValueError, match="Object"):
        archive.save("bad", np.asarray([object()], dtype=object))
    with pytest.raises(ValueError, match="escapes"):
        archive.save("../../outside", np.zeros(1, dtype=np.uint8))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        load_snapshot(tmp_path)


def _payload(owner, ids, width):
    ids = np.asarray(ids, dtype=np.int64)
    return ((ids[:, None] + np.arange(width)[None, :] + owner * 23) % 251).astype(np.uint8)


@pytest.fixture
def captured(tmp_path):
    b, h, r, width, tail = 2, 4096, 2048, 656, 384
    owners = 3
    ids = np.tile(np.arange(h, dtype=np.int64), (owners, 1))
    ids[1:, [0, 2048]] = ids[1:, [2048, 0]]
    keys = ((ids[:, :, None] + np.arange(128)) % 251).astype(np.uint8)
    scales = (ids / 100).astype(np.float32)
    valid = np.ones((owners, r), dtype=np.int32)
    valid[0] = 0
    metadata = np.zeros((5, b * 4), dtype=np.int64)
    metadata[2, :5] = [0, 4, 5, 6, 7]
    metadata[3, :5] = 1
    metadata[4].view(np.int32)[:b] = [1, 2]
    bounds = np.tile([3968, h], (b, 1)).astype(np.int32)
    output = np.zeros((owners, r + tail, width), dtype=np.uint8)
    base, slot_bytes = 0x10000000, 86016
    addresses = np.zeros((owners, h), dtype=np.int64)
    ddr_keys, pages = [], []
    for owner in (1, 2):
        output[owner, :r] = _payload(owner, ids[owner, :r], width)
        tail_ids = np.arange(3968, h)
        output[owner, r + tail_ids % tail] = _payload(owner, tail_ids, width)
        committed = ids[owner] < 3968
        tokens = ids[owner, committed]
        addresses[owner, committed] = base + ((owner - 1) * 32 + tokens // 128) * slot_bytes + tokens % 128 * width
        for block in range(31):
            ddr_keys.append((owner - 1) * 32 + block)
            pages.append(_payload(owner, np.arange(block * 128, (block + 1) * 128), width))
    directory = tmp_path / "ddr"
    directory.mkdir()
    np.save(directory / "keys.npy", np.asarray(ddr_keys, dtype=np.int64), allow_pickle=False)
    np.save(directory / "valid_lengths.npy", np.full(len(ddr_keys), 128 * width, dtype=np.uint64), allow_pickle=False)
    (directory / "payload.bin").write_bytes(np.asarray(pages).tobytes())
    manifest = dict(
        geometry=dict(batch=b, history=h, resident=r, token_bytes=width, output_stride=r + tail), tensors={}
    )
    manifest["ddr"] = dict(
        region_base=base,
        slot_bytes=slot_bytes,
        tokens_per_block=128,
        files={
            name: dict(path=filename)
            for name, filename in (
                ("keys", "keys.npy"),
                ("valid_lengths", "valid_lengths.npy"),
                ("payload", "payload.bin"),
            )
        },
    )
    archive = TensorArchive(tmp_path, manifest)
    pre = dict(
        keys=keys,
        key_scales=scales,
        addresses=addresses,
        logical_ids=ids,
        hot_valid=valid,
        output=output,
        metadata=metadata,
        tail_bounds=bounds,
        query_ends=np.full((b, 4), h, dtype=np.int32),
    )
    for name, value in pre.items():
        archive.save("pre." + name, value)
    state = {name: value.copy() for name, value in pre.items() if name not in ("metadata", "tail_bounds", "query_ends")}
    logical = np.full((b * 4, 2048), -1, dtype=np.int32)
    slots = np.full_like(logical, -1)
    for row in range(5):
        owner = 1 if row == 0 else 2
        selection = ids[owner, :r].copy()
        selection[-2:] = [3000, 4095]  # A real cold record and ABC-only record.
        logical[row] = selection
        physical = np.arange(r)
        physical[-2:] = [r - 2, r + 4095 % tail]
        slots[row] = physical
        state["output"][owner, r - 2] = _payload(owner, [3000], width)[0]
    for owner in (1, 2):
        for name in ("keys", "key_scales", "addresses", "logical_ids"):
            state[name][owner, [r - 2, 3000]] = state[name][owner, [3000, r - 2]]
    state["logical_output"], state["slot_output"] = logical, slots
    return archive, state


def test_oracle_covers_true_cold_hot_abc_and_cross_packed_tnd(captured):
    archive, state = captured
    oracle = DecodeOracle(archive)
    result = oracle.verify(state, expected_topk=state["logical_output"])
    assert len(result["selected_tags"]) == 4096
    assert np.all(result["selected_references"][5:] == -1)
    assert result["selected_payload"].shape == (4096, 656)


@pytest.mark.parametrize("fault", ["selected_byte", "hot_byte", "key", "scale", "address", "ids", "padding"])
def test_oracle_rejects_payload_and_index_corruption(captured, fault):
    archive, original = captured
    state = {name: value.copy() for name, value in original.items()}
    if fault == "selected_byte":
        state["output"][1, 2046, 33] ^= 1
    elif fault == "hot_byte":
        state["output"][1, 22, 45] ^= 1
    elif fault == "key":
        state["keys"][1, 3000, 3] ^= 1
    elif fault == "scale":
        state["key_scales"][1, 3000] += 1
    elif fault == "address":
        state["addresses"][1, 3000] += 656
    elif fault == "ids":
        state["logical_ids"][1, 0] = 1
    else:
        state["slot_output"][7, 0] = 0
    with pytest.raises(AssertionError):
        DecodeOracle(archive).verify(state)


def test_oracle_allows_consistent_index_permutation_and_address_relocation(captured):
    archive, original = captured
    state = {name: value.copy() for name, value in original.items()}
    for name in ("keys", "key_scales", "addresses", "logical_ids"):
        state[name][1, [3000, 3001]] = state[name][1, [3001, 3000]]
    delta = 0x20000000
    state["addresses"][state["addresses"] != 0] += delta
    DecodeOracle(archive).verify(state, address_delta=delta, expected_topk=original["logical_output"])


def test_missing_future_prefetch_page_is_not_fabricated(captured):
    archive, _ = captured
    oracle = DecodeOracle(archive)
    assert np.array_equal(oracle.expected(1, [3500]), _payload(1, [3500], 656))
    oracle.key_to_record.pop(3500 // 128)
    with pytest.raises(AssertionError, match="absent"):
        oracle.expected(1, [3500])


def test_oracle_rebases_high_bit_addresses_across_signed_boundary(captured):
    archive, state = captured
    oracle = DecodeOracle(archive)
    values = oracle.pre["addresses"].view(np.uint64).copy()
    values[values != 0] += np.uint64(1 << 63)
    oracle.pre["addresses"] = values.view(np.int64)
    oracle.ddr = dict(oracle.ddr, region_base=oracle.ddr["region_base"] + (1 << 63))
    assert np.array_equal(oracle.expected(1, [3500]), _payload(1, [3500], 656))
    oracle.verify(state, address_delta=-(1 << 63), expected_topk=state["logical_output"])


def test_oracle_rejects_future_token_using_cross_packed_source(captured):
    archive, state = captured
    oracle = DecodeOracle(archive)
    oracle.pre["query_ends"] = oracle.pre["query_ends"].copy()
    oracle.pre["query_ends"].reshape(-1)[4] = 4095
    with pytest.raises(AssertionError, match="causal"):
        oracle.verify(state)


@pytest.mark.parametrize("compact_reserved", (False, True))
def test_complete_recorder_publishes_only_verified_files(captured, tmp_path, compact_reserved):
    archive, state = captured
    b, h, r, width = 2, 4096, 2048, 656
    pre = {
        name: torch.from_numpy(archive.array("pre." + name).copy())
        for name in ("keys", "key_scales", "addresses", "logical_ids", "hot_valid", "output", "metadata", "tail_bounds")
    }
    metadata = pre["metadata"]
    metadata[0] = torch.arange(8)
    metadata[1].fill_(1)
    metadata[2] = torch.arange(8)
    metadata[3].fill_(1)
    inputs = NS(
        metadata=metadata,
        query=torch.zeros((8, 32, 128), dtype=torch.uint8).view(torch.float8_e4m3fn),
        weights=torch.ones((8, 32), dtype=torch.bfloat16),
        scale=torch.ones((8, 32)),
        query_ends=torch.tensor([[4093, 4094, 4095, 4096]] * b, dtype=torch.int32),
        block_table=torch.zeros((b, h // 128), dtype=torch.int32),
        token_count=8,
    )
    context = torch.zeros(256, dtype=torch.uint8)
    context[120:128].view(torch.int32).copy_(torch.tensor([512, 512], dtype=torch.int32))
    rotating = NS(
        lease=NS(runtime=NS(transport="ubmem"), check_stream=lambda: NS(synchronize=lambda: None)),
        shape=NS(
            batch=b,
            history=h,
            resident=r,
            heads=32,
            token_bytes=width,
            aic_count=32,
            intermediate_fetches=8,
            soc="cpu_contract_fixture",
            compact_reserved=compact_reserved,
        ),
        context=context,
        rounds=8,
        service_lanes=2,
        final_workers=4,
        output_stride=r + 384,
        hot_valid=pre["hot_valid"],
        output=pre["output"],
        tail_bounds=pre["tail_bounds"],
        arguments=[None] * 5 + [torch.full((b,), 4, dtype=torch.int32), torch.full((b,), h, dtype=torch.int32)],
        lru_slots=torch.arange(2048, dtype=torch.int32).repeat(b, 1),
        reserved_hot=torch.full((b, r), 91 if compact_reserved else -1, dtype=torch.int32),
        rows=torch.zeros((b, 256), dtype=torch.int32),
        row_status=torch.zeros((b, 32), dtype=torch.int32),
        final_posts=torch.zeros((b, 4, 32), dtype=torch.int32),
        ubmem_status=torch.zeros((b, 16), dtype=torch.int64),
        mapping=torch.full((b, h), -1, dtype=torch.int32),
        locations=torch.zeros((b, 4, 2048), dtype=torch.int32),
        fallback_addresses=torch.zeros((b, 8192), dtype=torch.int64),
        destinations=torch.zeros(b * 8192, dtype=torch.int32),
        changes=torch.zeros((b, 10240), dtype=torch.int32),
        change_slots=torch.zeros((b, 10240), dtype=torch.int32),
        change_count=torch.zeros(b, dtype=torch.int32),
    )
    storage = NS(
        keys=pre["keys"],
        scales=pre["key_scales"],
        addresses=pre["addresses"],
        logical_ids=pre["logical_ids"],
        hot_valid=pre["hot_valid"],
    )
    logical = np.repeat(state["logical_output"][[0, 1]], 4, axis=0)
    slots = np.repeat(state["slot_output"][[0, 1]], 4, axis=0)
    logical[:, -1], slots[:, -1] = 4091, r + 4091 % 384
    update = NS(
        status=torch.zeros(b, dtype=torch.int32),
        logical_output=torch.from_numpy(logical),
        slot_output=torch.from_numpy(slots),
    )
    selector = NS(
        rotating=rotating,
        storage=storage,
        inputs=inputs,
        update=update,
        topk=torch.zeros((b, 4, 2048), dtype=torch.int32),
    )
    ddr = dict(
        archive.manifest["ddr"],
        schema_version=1,
        transport="ubmem",
        region_bytes=64 * 86016,
        token_bytes=width,
        page_mode="base",
        numa_node=-1,
        generation=1,
        record_count=62,
        payload_bytes=62 * 128 * width,
        device_id=0,
    )
    old_folder = archive.directory / "ddr"
    keys = np.load(old_folder / "keys.npy", allow_pickle=False)
    for name, data in (
        ("slot_offsets", keys.astype(np.uint64) * 86016),
        ("addresses", keys.astype(np.uint64) * 86016 + ddr["region_base"]),
    ):
        np.save(old_folder / (name + ".npy"), data, allow_pickle=False)
    ddr["files"] = {
        name: dict(path=path, sha256=file_digest(old_folder / path))
        for name, path in (
            ("keys", "keys.npy"),
            ("addresses", "addresses.npy"),
            ("slot_offsets", "slot_offsets.npy"),
            ("valid_lengths", "valid_lengths.npy"),
            ("payload", "payload.bin"),
        )
    }

    def exporter(_, destination):
        shutil.copytree(old_folder, destination)
        return ddr

    source = dict(
        capture_mode="eager_diagnostic",
        geometry=dict(batch=b),
        context=dict(min_context=4092, max_context=4096),
        requests=[dict(owner=i + 1, query_begin=4 * i, query_stop=4 * i + 4) for i in range(b)],
    )
    recorder = DecodeSnapshotRecorder(torch, tmp_path / "recorded", source, exporter)
    recorder.before(selector)
    assert not recorder.completed and (recorder.directory / "INCOMPLETE").is_file()
    if compact_reserved:
        assert torch.all(rotating.reserved_hot == 91)  # Recording did not execute init early.
        assert "pre.reserved_hot" in recorder.manifest["input_derivations"]
        np.testing.assert_array_equal(recorder.archive.array("pre.reserved_hot"), np.tile(np.arange(r), (b, 1)))
        with pytest.raises(AssertionError, match="reserved-hot reconstruction"):
            recorder.after(selector)
        assert not (recorder.directory / "manifest.json").exists()
        rotating.reserved_hot.copy_(rotating.lru_slots)  # CPU fixture substitutes the validated native init.
    for name, value in state.items():
        if name in ("logical_output", "slot_output"):
            continue
        target = rotating.output if name == "output" else getattr(storage, "scales" if name == "key_scales" else name)
        target.copy_(torch.from_numpy(value))
    recorder.after(selector)
    loaded = load_snapshot(recorder.directory)
    assert recorder.completed and not (recorder.directory / "INCOMPLETE").exists()
    assert loaded.manifest["oracle"]["passed"] is True
    assert loaded.array("post.selected_payload").shape == (4096, width)
    assert loaded.array("post.final_records").shape == (0, 4)
    validate_capture_inputs(loaded)
    loaded.manifest["metadata"]["context"]["min_context"] = 4093
    with pytest.raises(ValueError, match="Actual NPU"):
        validate_capture_inputs(loaded)


def test_recorder_does_not_publish_on_bad_mode_or_existing_directory(tmp_path):
    with pytest.raises(ValueError, match="eager"):
        DecodeSnapshotRecorder(torch, tmp_path, {}, None).before(None)
    (tmp_path / "retained.txt").write_text("keep")
    selector = NS(rotating=NS(lease=NS(runtime=NS(transport="ubmem")), tail_bounds=object()))
    recorder = DecodeSnapshotRecorder(torch, tmp_path, dict(capture_mode="eager_diagnostic"), None)
    with pytest.raises(FileExistsError):
        recorder.before(selector)
    assert not recorder.completed and not (tmp_path / "manifest.json").exists()


def test_completion_failure_never_becomes_pass():
    rotating = NS(
        rows=torch.zeros(2, 256, dtype=torch.int32),
        row_status=torch.zeros(2, 32, dtype=torch.int32),
        final_posts=torch.zeros(2, 4, 32, dtype=torch.int32),
        ubmem_status=torch.zeros(2, 16, dtype=torch.int64),
    )
    update = NS(status=torch.zeros(2, dtype=torch.int32))
    assert check_completion(torch, rotating, update)["prefetch"] == 0
    rotating.ubmem_status[0, 1] = 1
    with pytest.raises(AssertionError, match="completed"):
        check_completion(torch, rotating, update)
    rotating.ubmem_status.zero_()
    update.status[1] = 4
    with pytest.raises(AssertionError, match="Index"):
        check_completion(torch, rotating, update)


def test_cli_inspect_needs_no_build_or_device_and_urma_requires_server():
    assert arguments(["--snapshot", "/tmp/snapshot", "--inspect"]).inspect
    with pytest.raises(SystemExit):
        arguments(["--snapshot", "/tmp/snapshot"])
    with pytest.raises(SystemExit):
        arguments(["--snapshot", "/tmp/snapshot", "--transport", "urma", "--build-dir", "/tmp/build", "--soc", "soc"])
    assert arguments(["--snapshot", "/tmp/snapshot", "--transport", "urma", "--inspect"]).inspect


def test_failed_replay_retains_owners_without_unsafe_cleanup(monkeypatch):
    from kvip.diagnostics import replay

    owner = object()

    def fail(args, archive, retained):
        retained["pending_device_owner"] = owner
        raise RuntimeError("completion unknown")

    monkeypatch.setattr(replay, "_run", fail)
    before = len(replay._RETAINED_RUNS)
    try:
        with pytest.raises(RuntimeError, match="unknown"):
            replay.run(None, None)
        assert len(replay._RETAINED_RUNS) == before + 1
        assert replay._RETAINED_RUNS[-1]["pending_device_owner"] is owner
        monkeypatch.setattr(replay, "_run", lambda *_: {"status": "pass"})
        assert replay.run(None, None) == {"status": "pass"}
        assert len(replay._RETAINED_RUNS) == before + 1
    finally:
        del replay._RETAINED_RUNS[before:]


def test_array_hash_includes_every_byte():
    left = np.zeros((8, 656), dtype=np.uint8)
    right = copy.deepcopy(left)
    right[-1, -1] = 1
    assert array_digest(left) != array_digest(right)


def test_tagged_mapping_diagnostics_keep_only_current_epoch_and_preserve_slots():
    rows = np.zeros((2, 256), dtype=np.uint32)
    rows[:, 0] = [1, 16383]
    rows[:, 1] = 0x4D415031
    raw = np.array(
        [
            [(1 << 18) | 16384, (2 << 18) | 16385, 0xFFFFFFFF],
            [(16383 << 18) | 147839, (16382 << 18) | 1, 0xFFFFFFFF],
        ],
        dtype=np.uint32,
    )
    before = raw.copy()
    assert np.array_equal(decode_tagged_mapping(raw, rows), [[16384, -1, -1], [147839, -1, -1]])
    assert np.array_equal(raw, before)
    rows[0, 1] = 0
    with pytest.raises(AssertionError, match="epoch state"):
        decode_tagged_mapping(raw, rows)
    rows[0, 1] = 0x4D415031
    rows[0, 0] = 0
    with pytest.raises(AssertionError, match="epoch state"):
        decode_tagged_mapping(raw, rows)
