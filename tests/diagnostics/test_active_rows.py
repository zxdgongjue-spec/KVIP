# SPDX-License-Identifier: Apache-2.0
"""Mask real-capacity Q4 fixtures without renumbering owner storage or DDR."""

import copy
import json

import numpy as np
import pytest
import torch

from kvip.diagnostics.active_rows import mask_active_rows
from kvip.diagnostics.archive import TensorArchive, file_digest
from kvip.diagnostics.oracle import DecodeOracle


@pytest.fixture
def full_capture(tmp_path):
    batch, history, resident, width = 16, 4096, 2048, 656
    owner_count, stride = batch + 1, resident + 384
    owners = np.roll(np.arange(1, owner_count, dtype=np.int32), 3)
    metadata = np.zeros((5, batch * 4), dtype=np.int64)
    # TND row order deliberately differs from packed order; queries within
    # each request are also permuted to catch output-prefix slicing mistakes.
    unpack = (np.roll(np.arange(batch), 5)[:, None] * 4 + [2, 0, 3, 1]).reshape(-1)
    metadata[0] = np.argsort(unpack)
    metadata[1] = metadata[3] = 1
    metadata[2] = unpack
    metadata[4].view(np.int32)[:batch] = owners
    ids = np.tile(np.arange(history, dtype=np.int64), (owner_count, 1))
    keys = ((ids[..., None] + np.arange(128)) % 251).astype(np.uint8)
    valid = np.ones((owner_count, resident), dtype=np.int32)
    valid[0] = 0
    output = np.zeros((owner_count, stride, width), dtype=np.uint8)
    for owner in range(owner_count):
        output[owner] = (np.arange(stride)[:, None] + owner + np.arange(width)) % 251
    lru = np.tile(np.arange(resident - 1, -1, -1, dtype=np.int32), (batch, 1))
    reserved = np.empty_like(lru)
    for row in range(batch):
        reserved[row, lru[row]] = np.arange(resident)
    pre = dict(
        metadata=metadata,
        query_ends=np.tile(np.arange(history - 3, history + 1, dtype=np.int32), batch),
        keys=keys,
        key_scales=(ids / 100).astype(np.float32),
        logical_ids=ids,
        addresses=np.zeros((owner_count, history), dtype=np.int64),
        hot_valid=valid,
        output=output,
        query=np.zeros((batch * 4, 32, 128), dtype=np.uint8),
        weights=np.full((batch * 4, 32), 0x3F80, dtype=np.uint16),
        query_scale=np.ones((batch * 4, 32), dtype=np.float32),
        query_lengths=np.full(batch, 4, dtype=np.int32),
        key_lengths=np.full(batch, history, dtype=np.int32),
        block_table=owners[:, None] * (history // 128) + np.arange(history // 128, dtype=np.int32),
        lru_slots=lru,
        reserved_hot=reserved,
        tail_bounds=np.tile([history - 128, history], (batch, 1)).astype(np.int32),
    )
    manifest = dict(
        schema_version=1,
        status="complete",
        tensors={},
        geometry=dict(
            batch=batch,
            history=history,
            resident=resident,
            token_bytes=width,
            output_stride=stride,
            token_count=batch * 4,
        ),
        oracle=dict(passed=True),
    )
    directory = tmp_path / "ddr"
    directory.mkdir()
    np.save(directory / "keys.npy", np.asarray([0], dtype=np.int64))
    np.save(directory / "valid_lengths.npy", np.asarray([128 * width], dtype=np.uint64))
    (directory / "payload.bin").write_bytes(bytes(128 * width))
    manifest["ddr"] = dict(
        region_base=0x100000,
        region_bytes=16 << 30,
        slot_bytes=86016,
        tokens_per_block=128,
        files={
            name: {"path": path}
            for name, path in (("keys", "keys.npy"), ("valid_lengths", "valid_lengths.npy"), ("payload", "payload.bin"))
        },
    )
    archive = TensorArchive(tmp_path, manifest)
    for name, value in pre.items():
        archive.save("pre." + name, value, dtype={"query": "float8_e4m3fn", "weights": "bfloat16"}.get(name))
    state = {
        name: pre[name].copy() for name in ("keys", "key_scales", "logical_ids", "addresses", "hot_valid", "output")
    }
    state["logical_output"] = np.tile(np.arange(2048, dtype=np.int32), (batch * 4, 1))
    state["slot_output"] = state["logical_output"].copy()
    oracle = DecodeOracle(archive)
    selected = oracle.verify(state)
    for name in ("logical_output", "slot_output", "logical_ids", "hot_valid"):
        archive.save("post." + name, state[name])
    for name in ("selected_tags", "selected_payload", "selected_references"):
        archive.save("post." + name, selected[name])
    archive.save("post.prefetch_records", np.array([[0, 10, 10], [5, 11, 11]], dtype=np.int64))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return archive, state, owners


@pytest.mark.parametrize("count", [1, 2, 4, 16])
def test_masks_packed_rows_without_reordering_tnd_or_touching_residents(full_capture, count):
    archive, state, owners = full_capture
    original_manifest = copy.deepcopy(archive.manifest)
    digest = file_digest(archive.directory / "manifest.json")
    view = mask_active_rows(archive, count)
    assert view.manifest["geometry"] == archive.manifest["geometry"]
    assert view.manifest["ddr"] == archive.manifest["ddr"]
    assert view.manifest["replay_selection"]["active_rows"] == list(range(count))
    assert view.manifest["replay_selection"]["owners"] == owners[:count].tolist()
    before, after = archive.array("pre.metadata"), view.array("pre.metadata")
    expected_active = before[2] // 4 < count
    assert np.array_equal(after[3] != 0, expected_active)
    assert np.array_equal(after[2, expected_active], before[2, expected_active])
    assert np.all(after[2, ~expected_active] == 0)
    assert np.array_equal(after[4].view(np.int32)[:count], owners[:count])
    assert np.all(after[4].view(np.int32)[count:16] == 0)
    for name in (
        "keys",
        "key_scales",
        "addresses",
        "logical_ids",
        "hot_valid",
        "output",
        "query",
        "weights",
        "query_scale",
        "query_lengths",
        "key_lengths",
    ):
        assert np.array_equal(view.array("pre." + name), archive.array("pre." + name))
    assert np.all(view.array("pre.query_ends").reshape(16, 4)[count:] == 0)
    assert np.all(view.array("pre.tail_bounds")[count:] == 0)
    for row in range(count, 16):
        candidates = view.array("pre.lru_slots")[row]
        assert len(np.unique(candidates)) == 2048
        assert np.array_equal(view.array("pre.reserved_hot")[row, candidates], np.arange(2048))
        assert np.array_equal(view.array("pre.block_table")[row], np.arange(32))
    filtered_state = dict(state)
    for name in ("logical_output", "slot_output"):
        filtered_state[name] = view.array("post." + name)
    verified = DecodeOracle(view).verify(filtered_state, expected_topk=view.array("post.logical_output"))
    assert np.array_equal(verified["selected_tags"], view.array("post.selected_tags"))
    assert np.array_equal(verified["selected_payload"], view.array("post.selected_payload"))
    assert np.array_equal(verified["selected_references"], view.array("post.selected_references"))
    assert view.array("post.selected_payload").shape == (count * 2048, 656)
    assert archive.manifest == original_manifest
    assert file_digest(archive.directory / "manifest.json") == digest
    assert not view.array("pre.metadata").flags.writeable
    with pytest.raises(RuntimeError, match="read-only"):
        view.save("unexpected", np.zeros(1))


def test_tensor_views_keep_fp8_bf16_and_overlay_output_shapes(full_capture):
    archive, _, _ = full_capture
    view = mask_active_rows(archive, 1)
    for name, dtype in (("query", torch.float8_e4m3fn), ("weights", torch.bfloat16)):
        actual = view.tensor("pre." + name, torch, "cpu")
        assert actual.dtype == dtype
        assert torch.equal(actual.view(torch.uint8), archive.tensor("pre." + name, torch, "cpu").view(torch.uint8))
    assert view.tensor("post.selected_payload", torch, "cpu").shape == (2048, 656)


def test_rejects_unsafe_guard_nested_views_and_invalid_counts(full_capture):
    archive, _, _ = full_capture
    for count in (0, 17, True, 1.0):
        with pytest.raises(ValueError, match="count"):
            mask_active_rows(archive, count)
    with pytest.raises(ValueError, match="original"):
        mask_active_rows(mask_active_rows(archive, 2), 1)
    source_array = archive.array

    def unsafe(name, **kwargs):
        value = source_array(name, **kwargs)
        if name == "pre.hot_valid":
            value = value.copy()
            value[0, 0] = 1
        return value

    archive.array = unsafe
    with pytest.raises(ValueError, match="guard owner"):
        mask_active_rows(archive, 1)
