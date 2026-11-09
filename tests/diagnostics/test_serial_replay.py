# SPDX-License-Identifier: Apache-2.0
"""Host contract checks; native LI/LIDU execution still requires the NPU."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kvip.diagnostics.archive import TensorArchive
from kvip.diagnostics.serial_replay import BITMAP_WORDS, create_serial, serial_inputs


def payload(owner, tokens, width):
    return ((np.asarray(tokens)[:, None] * 13 + np.arange(width)[None, :] * 7 + owner * 19) % 251).astype(np.uint8)


def fixture(directory, batch=1, width=656):
    history, resident, stride, heads = 4096, 2048, 2432, 32
    owners = batch + 1
    ids = np.tile(np.arange(history, dtype=np.int64), (owners, 1))
    ids[1:, [0, 2048]] = ids[1:, [2048, 0]]
    keys = ((ids[:, :, None] + np.arange(128)) % 121).astype(np.uint8)
    scales = (1 + ids / 8192).astype(np.float32)
    metadata = np.zeros((5, batch * 4), dtype=np.int64)
    metadata[0] = metadata[2] = np.arange(batch * 4)
    metadata[1] = metadata[3] = 1
    metadata[4].view(np.int32)[:batch] = np.arange(1, owners)
    ends = np.tile(np.arange(4093, 4097, dtype=np.int32), (batch, 1))
    bounds = np.tile([3968, 4096], (batch, 1)).astype(np.int32)
    valid = np.ones((owners, resident), dtype=np.int32)
    valid[0] = 0
    output = np.zeros((owners, stride, width), dtype=np.uint8)
    addresses = np.zeros_like(ids)
    base = 0x10000000
    slot = (width * 128 + 4095) // 4096 * 4096
    ddr_keys, pages = [], []
    for owner in range(1, owners):
        output[owner, :resident] = payload(owner, ids[owner, :resident], width)
        tail = np.arange(3968, history)
        output[owner, resident + tail % (stride - resident)] = payload(owner, tail, width)
        committed = ids[owner] < 3968
        tokens = ids[owner, committed]
        addresses[owner, committed] = base + ((owner - 1) * 32 + tokens // 128) * slot + tokens % 128 * width
        for page in range(31):
            ddr_keys.append((owner - 1) * 32 + page)
            pages.append(payload(owner, np.arange(page * 128, (page + 1) * 128), width))
    ddr = directory / "ddr"
    ddr.mkdir()
    np.save(ddr / "keys.npy", np.asarray(ddr_keys, dtype=np.int64), allow_pickle=False)
    np.save(ddr / "valid_lengths.npy", np.full(len(ddr_keys), width * 128, dtype=np.uint64), allow_pickle=False)
    (ddr / "payload.bin").write_bytes(np.asarray(pages).tobytes())
    archive = TensorArchive(
        directory,
        dict(
            geometry=dict(
                batch=batch,
                history=history,
                resident=resident,
                heads=heads,
                owners=owners,
                token_bytes=width,
                output_stride=stride,
            ),
            ddr=dict(
                region_base=base,
                region_bytes=owners * 32 * slot,
                slot_bytes=slot,
                tokens_per_block=128,
                files={
                    name: dict(path=path)
                    for name, path in (
                        ("keys", "keys.npy"),
                        ("valid_lengths", "valid_lengths.npy"),
                        ("payload", "payload.bin"),
                    )
                },
            ),
            tensors={},
        ),
    )
    for name, value in dict(
        keys=keys,
        key_scales=scales,
        logical_ids=ids,
        addresses=addresses,
        hot_valid=valid,
        output=output,
        metadata=metadata,
        query_ends=ends,
        tail_bounds=bounds,
        lru_slots=np.tile(np.r_[2046, 2047, np.arange(2046)], (batch, 1)).astype(np.int32),
    ).items():
        archive.save("pre." + name, value)
    archive.save(
        "pre.query",
        np.arange(batch * 4 * heads * 128, dtype=np.uint8).reshape(batch * 4, heads, 128),
        dtype="float8_e4m3fn",
    )
    archive.save_tensor("pre.weights", torch, torch.ones(batch * 4, heads, dtype=torch.bfloat16))
    archive.save("pre.query_scale", np.ones((batch * 4, heads), dtype=np.float32))
    selected = np.stack(
        [np.r_[np.arange(2046), 3000, 4092 + query] for _ in range(batch) for query in range(4)]
    ).astype(np.int32)
    archive.save("post.logical_output", selected.reshape(batch * 4, 1, 2048))
    return archive


@pytest.mark.parametrize("batch,width", [(1, 656), (1, 2624), (2, 656), (2, 2624)])
def test_restores_logical_index_hot_abc_and_published_pages(tmp_path, batch, width):
    archive = fixture(tmp_path, batch, width)
    result = serial_inputs(archive, 0x70000000)
    assert BITMAP_WORDS == 4176
    for row, owner in enumerate(result["owners"]):
        np.testing.assert_array_equal(result["keys"][owner, :, 0], np.arange(4096) % 121)
        np.testing.assert_array_equal(result["scales"][owner], 1 + np.arange(4096) / 8192)
        assert result["forward"][owner, 2048] == 0
        assert result["forward"][owner, 0] == -1
        assert result["forward"][owner, 4095] == 2048 + 4095 % 384
        assert result["page_valid"][owner].sum() == 31
        assert (
            result["page_addresses"][owner, 0] == 0x70000000 + (owner - 1) * 32 * archive.manifest["ddr"]["slot_bytes"]
        )
        assert result["bitmap"][row, :64].tolist() == [-1] * 64
        assert np.all(result["bitmap"][row, 64:] == 0)


@pytest.mark.parametrize(
    "fault,match",
    [
        ("permutation", "permutations"),
        ("causal", "consecutive"),
        ("visible_prefix", "causal prefixes"),
        ("page", "contiguous"),
        ("lru", "distinct"),
        ("unpack", "TND"),
    ],
)
def test_rejects_unrecoverable_snapshot_contract(tmp_path, fault, match):
    archive = fixture(tmp_path)
    original = archive.array
    arrays = {name: original(name).copy() for name in archive.manifest["tensors"]}
    if fault == "permutation":
        arrays["pre.logical_ids"][1, 2] = 1
    elif fault == "causal":
        arrays["pre.query_ends"][0, 1] -= 1
    elif fault == "visible_prefix":
        arrays["pre.logical_ids"][1, [1, 4095]] = arrays["pre.logical_ids"][1, [4095, 1]]
    elif fault == "page":
        arrays["pre.addresses"][1, 1] += 1
    elif fault == "lru":
        arrays["pre.lru_slots"][0, 1] = arrays["pre.lru_slots"][0, 0]
    else:
        arrays["pre.metadata"][2, :2] = [1, 0]
    archive.array = arrays.__getitem__
    with pytest.raises(ValueError, match=match):
        serial_inputs(archive, 0x70000000)


class FakeRuntime:
    transport = "ubmem"

    def __init__(self, archive, new_base):
        self.archive, self.base, self.calls = archive, new_base, []
        ddr = archive.manifest["ddr"]
        self.width, self.slot = archive.manifest["geometry"]["token_bytes"], ddr["slot_bytes"]
        keys = np.load(archive.directory / "ddr/keys.npy")
        data = np.fromfile(archive.directory / "ddr/payload.bin", dtype=np.uint8).reshape(len(keys), -1)
        self.pages = dict(zip(map(int, keys), data))

    def batch_get(self, addresses, valid, indices, offsets, total, destinations, output, stream, synchronize):
        assert synchronize is False
        self.calls.append(int(total[0]))
        for index in range(int(total[0])):
            table_index, offset, destination = int(indices[index]), int(offsets[index]), int(destinations[index])
            assert valid[table_index] == 1
            key = (int(addresses[table_index]) - self.base) // self.slot
            output[destination].copy_(torch.from_numpy(self.pages[key][offset : offset + self.width]))


@pytest.mark.parametrize("batch,width,transport", [(1, 656, "ubmem"), (2, 2624, "ubmem"), (1, 2624, "urma")])
def test_bundle_runs_real_chain_contract_and_resets_all_mutable_state(tmp_path, monkeypatch, batch, width, transport):
    archive = fixture(tmp_path, batch, width)
    calls = []
    expected = torch.from_numpy(archive.array("post.logical_output").copy())

    def li(**kwargs):
        calls.append("li")
        assert kwargs["layout_query"] == "TND" and kwargs["layout_key"] == "PA_BSND"
        assert kwargs["sparse_mode"] == 3 and kwargs["sparse_count"] == 2048
        assert kwargs["query"].shape == (batch * 4, 32, 128)
        assert kwargs["key"].shape == ((batch + 1) * 32, 128, 1, 128)
        assert kwargs["actual_seq_lengths_query"].tolist() == list(range(4, batch * 4 + 1, 4))
        assert kwargs["actual_seq_lengths_key"].tolist() == [4096] * batch
        assert kwargs["key"].view(torch.uint8).reshape(batch + 1, 4096, 128)[1, 0, 0] == 0
        return expected.clone()

    def lidu(
        selected,
        cumulative,
        entries,
        forward,
        reverse,
        cursors,
        budgets,
        lengths,
        preferred,
        bitmap,
        blocks,
        logical_lengths,
        profile,
        destinations,
        counts,
        indices,
        offsets,
        output_indices,
        resident_lengths,
        profiling,
        layer,
        groups,
        owners,
        qcap,
    ):
        calls.append("lidu")
        assert (layer, groups, owners, qcap) == (0, 1, batch + 1, 32)
        assert bitmap.shape == (batch, 4176) and profile.item() == 0
        descriptor = 0
        for row, owner_tensor in enumerate(entries):
            owner = int(owner_tensor)
            logical = selected[row * 4 : (row + 1) * 4].reshape(4, 2048)
            unique = np.unique(logical.numpy())
            reserved = set(map(int, preferred[row]))
            misses = [
                int(token) for token in unique if forward[owner, token] < 0 or int(forward[owner, token]) in reserved
            ]
            counts[row] = len(misses)
            # This fixture exercises <=2048 union misses. Real native LIDU
            # preserves candidate order, clears all victims, then installs.
            victims = list(map(int, preferred[row, : len(misses)]))
            for slot in victims:
                previous = int(reverse[owner, slot])
                if previous >= 0:
                    forward[owner, previous] = -1
            for token, slot in zip(misses, victims):
                forward[owner, token], reverse[owner, slot] = slot, token
                indices[descriptor] = owner * qcap + token // 128
                offsets[descriptor] = token % 128 * 656
                output_indices[descriptor] = int(blocks[row, slot // 128]) * 128 + slot % 128
                descriptor += 1
            destinations[row * 4 : (row + 1) * 4].copy_(forward[owner, logical.long()].view(4, 1, 2048))
        counts[-1] = descriptor

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_quant_lightning_indexer=li))
    monkeypatch.setattr("kvip.operators.lidu.lidu", lidu)
    runtime = FakeRuntime(archive, 0x70000000)
    runtime.transport = transport
    bundle = create_serial(
        torch,
        runtime,
        archive,
        dict(access_table=1, new_base=runtime.base),
        object(),
        tmp_path,
        "test",
        "cpu",
        object(),
        tmp_path,
        32,
        transport=transport,
    )
    assert bundle.lease is None
    assert [name for name, _ in bundle.stage_calls] == ["native_li", "lidu_prepare", "batch_get", "slot_mapping"]
    for _ in range(2):
        bundle.reset()
        bundle.forward()
        reads = bundle.verify()
        assert reads == dict(
            prefetch=0,
            final=batch * 2047,
            bytes=batch * 2047 * width,
            final_per_request=[2047] * batch,
            cold_demand=batch * 2,
            candidate_hot_refetch=batch * 2045,
        )
    assert calls == ["li", "lidu", "li", "lidu"]
    assert runtime.calls == [batch * 2047, batch * 2047]
    assert len(bundle.retained["generated_outputs"]) == 4
    assert bundle.retained["generated_outputs"][0] is not bundle.retained["generated_outputs"][2]
    bundle.retained["output"][2432 + 2046, 3] ^= 1
    with pytest.raises(AssertionError, match="payload"):
        bundle.verify()
