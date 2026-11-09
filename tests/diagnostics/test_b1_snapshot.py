# SPDX-License-Identifier: Apache-2.0
"""B1/Q4 archive contract with production CPU tensors; no device timing claims."""

from types import SimpleNamespace

import numpy as np
import torch

from kvip.cache.index_storage import IndexCacheStorage
from kvip.diagnostics.archive import file_digest, load_snapshot
from kvip.diagnostics.oracle import DecodeOracle
from kvip.diagnostics.replay import inspect_snapshot
from kvip.diagnostics.snapshot import (
    DecodeSnapshotRecorder,
    input_tensors,
    persistent_tensors,
    post_state,
    relocate_snapshot_addresses,
)
from kvip.operators.lifetch.geometry import final_aivs, li_geometry
from kvip.operators.lifetch_ubmem.specialization import Specialization
from kvip.operators.lifetch_ubmem.workspace import UbmemWorkspace


def _payload(ids, width):
    return ((np.asarray(ids)[:, None] + np.arange(width)[None, :]) % 251).astype(np.uint8)


def test_b1_recorder_oracle_archive_and_replay_contract(tmp_path):
    # Use a valid native specialization, with a short history to keep this host
    # regression small. The requested 128K/R16K shape is validated separately.
    shape = Specialization(1, 8704, 8192, 32, 2624, 32, 8, "Ascend950DT_9582")
    Specialization(1, 131328, 16384, 32, 2624, 32, 8, "Ascend950DT_9582")
    b, h, r, width = shape.batch, shape.history, shape.resident, shape.token_bytes
    geometry = li_geometry(b, "auto", 8, 32)
    assert (geometry["rotating_aic"], geometry["intermediate_fetches"], geometry["service_aivs_per_window"]) == (
        2,
        2,
        2,
    )
    assert final_aivs(b, "auto", 32) == 4
    workspace = UbmemWorkspace(
        torch,
        batch=b,
        history=h,
        resident=r,
        aics=32,
        windows=8,
        device=torch.device("cpu"),
        index_heads=32,
        pin_memory=False,
    )
    storage = IndexCacheStorage(torch, 2, h, r, torch.device("cpu"))
    inputs = workspace.inputs
    inputs.prepare([4], query_owners=[1])
    inputs.query.zero_()
    inputs.weights.fill_(1)
    inputs.scale.fill_(1)
    inputs.query_ends.copy_(torch.arange(h - 3, h + 1, dtype=torch.int32))
    assert tuple(inputs.metadata.shape) == (5, 4)
    assert inputs.row_owners.tolist() == [1]
    assert inputs.metadata[4].view(torch.int32)[1:].eq(0).all()
    prefix = h - 128
    workspace.tail_bounds[0] = torch.tensor([prefix, h])
    stride = r + 384
    output = torch.zeros((2, stride, width), dtype=torch.uint8)
    for begin in range(0, r, 256):
        output[1, begin : begin + 256] = torch.from_numpy(_payload(np.arange(begin, begin + 256), width))
    tail_ids = np.arange(prefix, h)
    output[1, r + torch.from_numpy(tail_ids % 384)] = torch.from_numpy(_payload(tail_ids, width))
    storage.hot_valid[1].fill_(1)
    storage.keys[1].copy_(torch.from_numpy(_payload(np.arange(h), 128)))
    storage.scales[1].copy_(torch.arange(h, dtype=torch.float32) / 100)
    base, slot_bytes = 0x10000000, 128 * width
    keys = np.arange(prefix // 128, dtype=np.int64) * 2 + 3
    region_bytes = int(keys[-1] + 3) * slot_bytes
    logical = np.arange(prefix)
    addresses = base + keys[logical // 128] * slot_bytes + logical % 128 * width
    storage.addresses[1, :prefix].copy_(torch.from_numpy(addresses))
    context = torch.zeros(256, dtype=torch.uint8)
    context[120:128].view(torch.int32).copy_(torch.tensor([512, 64], dtype=torch.int32))
    final_workers = final_aivs(b, "auto", 32)
    rotating = SimpleNamespace(
        **workspace.scratch,
        shape=shape,
        context=context,
        output=output,
        hot_valid=storage.hot_valid,
        output_stride=stride,
        tail_bounds=workspace.tail_bounds,
        arguments=workspace.arguments(storage),
        rounds=geometry["intermediate_fetches"],
        service_lanes=geometry["service_aivs_per_window"],
        final_workers=final_workers,
        final_posts=workspace.scratch["owners"][b * 32 + b * final_workers * 288 :].reshape(b, final_workers, 32),
        lease=SimpleNamespace(
            runtime=SimpleNamespace(transport="ubmem"), check_stream=lambda: SimpleNamespace(synchronize=lambda: None)
        ),
    )
    outputs = torch.full((2, 4, 1, 2048), -1, dtype=torch.int32)
    update = SimpleNamespace(
        status=torch.zeros(b, dtype=torch.int32), logical_output=outputs[0], slot_output=outputs[1]
    )
    selector = SimpleNamespace(rotating=rotating, storage=storage, inputs=inputs, update=update, topk=workspace.topk)

    def exporter(_, directory):
        directory.mkdir()
        arrays = dict(
            keys=keys,
            addresses=(base + keys * slot_bytes).astype(np.uint64),
            slot_offsets=(keys * slot_bytes).astype(np.uint64),
            valid_lengths=np.full(len(keys), slot_bytes, dtype=np.uint64),
        )
        files = {}
        for name, values in arrays.items():
            path = directory / (name + ".npy")
            np.save(path, values, allow_pickle=False)
            files[name] = dict(path=path.name, sha256=file_digest(path))
        path = directory / "payload.bin"
        with path.open("wb") as destination:
            for page in range(len(keys)):
                destination.write(_payload(np.arange(page * 128, (page + 1) * 128), width).tobytes())
        files["payload"] = dict(path=path.name, sha256=file_digest(path))
        return dict(
            schema_version=1,
            transport="ubmem",
            region_base=base,
            region_bytes=region_bytes,
            slot_bytes=slot_bytes,
            token_bytes=width,
            tokens_per_block=128,
            page_mode="thp",
            numa_node=0,
            generation=1,
            record_count=len(keys),
            payload_bytes=len(keys) * slot_bytes,
            device_id=0,
            files=files,
        )

    metadata = dict(
        capture_mode="eager_diagnostic",
        geometry=dict(batch=1),
        context=dict(min_context=h - 4, max_context=h),
        requests=[dict(owner=1, query_begin=0, query_stop=4)],
    )
    recorder = DecodeSnapshotRecorder(torch, tmp_path / "b1", metadata, exporter)
    recorder.before(selector)
    # Supply a legal completed state on CPU: one cold DDR token replaces hot
    # slot 2046, while the last selection uses the pre-existing ABC tail.
    cold, tail = r + 100, h - 5
    selected = torch.arange(2048, dtype=torch.int32)
    selected[-2:] = torch.tensor([cold, tail], dtype=torch.int32)
    slots = torch.arange(2048, dtype=torch.int32)
    slots[-1] = r + tail % 384
    update.logical_output.copy_(selected.reshape(1, 1, -1).expand(4, 1, -1))
    update.slot_output.copy_(slots.reshape(1, 1, -1).expand(4, 1, -1))
    for tensor in (storage.keys, storage.scales, storage.addresses, storage.logical_ids):
        tensor[1, [2046, cold]] = tensor[1, [cold, 2046]]
    output[1, 2046] = torch.from_numpy(_payload([cold], width)[0])
    rotating.rows[0, 247] = 1
    rotating.fallback_addresses[0, 0] = int(addresses[cold])
    rotating.destinations[0] = stride + 2046
    rotating.ubmem_status[0, 3:6] = torch.tensor([1, 1, width])
    recorder.after(selector)
    archive = load_snapshot(recorder.directory)
    summary = inspect_snapshot(archive)
    assert summary["geometry"]["batch"] == summary["active_requests"] == 1
    assert summary["query_ends"] == [[h - 3, h - 2, h - 1, h]]
    assert archive.array("post.selected_payload").shape == (2048, width)
    assert archive.array("post.final_records")[0].tolist() == [0, cold, int(addresses[cold]) - base, stride + 2046]
    # Exercise the same replay data restoration contract, including metadata
    # aliases and a different arena base, without pretending to run a kernel.
    # CPU .cpu() views alias their source; preserve the completed oracle before
    # exercising restore. Real NPU readbacks already own independent CPU data.
    actual = {name: value.copy() for name, value in post_state(torch, selector).items()}
    rebased = relocate_snapshot_addresses(archive, base + region_bytes)
    for name, target in (persistent_tensors(selector) | input_tensors(selector)).items():
        seed = torch.from_numpy(rebased.copy()) if name == "addresses" else archive.tensor("pre." + name, torch, "cpu")
        assert seed.shape == target.shape and seed.dtype == target.dtype
        target.copy_(seed)
        assert torch.equal(target.reshape(-1).view(torch.uint8), seed.reshape(-1).view(torch.uint8))
    assert inputs.row_owners.tolist() == [1]
    assert inputs.unpack_indices.tolist() == [0, 1, 2, 3]
    assert storage.addresses[0].eq(0).all()
    actual["addresses"][1, actual["addresses"][1] != 0] += region_bytes
    checked = DecodeOracle(archive).verify(
        actual, address_delta=region_bytes, expected_topk=archive.array("post.logical_output").reshape(4, 2048)
    )
    np.testing.assert_array_equal(checked["selected_payload"], archive.array("post.selected_payload"))
