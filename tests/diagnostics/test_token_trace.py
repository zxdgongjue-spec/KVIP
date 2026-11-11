# SPDX-License-Identifier: Apache-2.0
"""Real recorder/ID validation with CPU tensors; no mock NPU performance."""

import copy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from kvip.diagnostics.archive import file_digest
from kvip.diagnostics.token_trace import TokenIdTraceRecorder, load_token_trace


def fixture(batch=2, high_addresses=False, tagged=False):
    history, resident, width, stride = 8192, 4096, 2624, 4096 + 8192 + 384
    owners = np.array([3, 1][:batch], dtype=np.int32)
    ids = torch.arange(history, dtype=torch.int64).repeat(4, 1)
    ids[:, [5, 6]] = ids[:, [6, 5]]
    valid = torch.ones((4, resident), dtype=torch.int32)
    valid[0] = 0
    base = (1 << 63) + 0x10000000 if high_addresses else 0x10000000
    slot_bytes = ((width * 128 + 4095) // 4096) * 4096
    addresses = np.zeros((4, history), dtype=np.uint64)
    for owner in owners:
        logical = ids[int(owner)].numpy()
        published = logical < 8064
        keys = int(owner) * 64 + logical[published] // 128
        addresses[owner, published] = (
            base + keys.astype(np.uint64) * slot_bytes + (logical[published] % 128).astype(np.uint64) * width
        )
    metadata = torch.zeros((5, batch * 4), dtype=torch.int64)
    metadata[0] = torch.arange(batch * 4)
    metadata[1].fill_(1)
    metadata[2] = torch.arange(batch * 4).reshape(batch, 4).flip(0).reshape(-1)
    metadata[3].fill_(1)
    metadata[4].view(torch.int32)[:batch] = torch.from_numpy(owners)
    context = torch.zeros(256, dtype=torch.uint8)
    context[120:128].view(torch.int32).copy_(torch.tensor([256, 32], dtype=torch.int32))
    storage = NS(logical_ids=ids, hot_valid=valid, addresses=torch.from_numpy(addresses.view(np.int64)))
    rotating = NS(
        lease=NS(
            runtime=NS(transport="ubmem"),
            context=dict(region_base=base, region_bytes=256 * slot_bytes, slot_bytes=slot_bytes, tokens_per_block=128),
            check_stream=lambda: NS(synchronize=lambda: None),
        ),
        shape=NS(
            batch=batch,
            history=history,
            resident=resident,
            heads=32,
            token_bytes=width,
            aic_count=32,
            intermediate_fetches=8,
            soc="cpu_fixture",
            tagged_map=tagged,
        ),
        context=context,
        rounds=8,
        service_lanes=2,
        final_workers=4,
        output_stride=stride,
        tail_bounds=torch.tensor([[8064, history]] * batch, dtype=torch.int32),
        lru_slots=torch.arange(2048, 4096, dtype=torch.int32).repeat(batch, 1),
        rows=torch.zeros(batch, 256, dtype=torch.int32),
        row_status=torch.zeros(batch, 32, dtype=torch.int32),
        final_posts=torch.zeros(batch, 4, 32, dtype=torch.int32),
        ubmem_status=torch.zeros(batch, 16, dtype=torch.int64),
        mapping=torch.full((batch, history), -1, dtype=torch.int32),
        fallback_addresses=torch.zeros(batch, 8192, dtype=torch.int64),
        destinations=torch.zeros(batch * 8192, dtype=torch.int32),
    )
    inputs = NS(
        metadata=metadata,
        query_ends=torch.tensor([[8189, 8190, 8191, 8192]] * batch, dtype=torch.int32),
        token_count=batch * 4,
    )
    update = NS(
        status=torch.zeros(batch, dtype=torch.int32),
        logical_output=torch.empty(batch * 4, 2048, dtype=torch.int32),
        slot_output=torch.empty(batch * 4, 2048, dtype=torch.int32),
    )
    selector = NS(
        storage=storage,
        rotating=rotating,
        inputs=inputs,
        update=update,
        topk=torch.empty(batch, 4, 2048, dtype=torch.int32),
    )
    source = dict(
        capture_mode="eager_diagnostic",
        geometry=dict(batch=batch),
        context=dict(min_context=8188, max_context=8192),
        requests=[
            dict(request_id=f"request{i}", owner=int(owner), query_begin=i * 4, query_stop=i * 4 + 4)
            for i, owner in enumerate(owners)
        ],
        published_pages=[dict(addresses=[base], keys=[1])],
        provenance=dict(page_mode="thp", numa_node=2),
    )

    def complete():
        selected = np.arange(2048, dtype=np.int32)
        selected[-3:] = [2048, 5000, 8187]
        for row, owner in enumerate(owners):
            before = storage.logical_ids[int(owner)].numpy().copy()
            inverse = np.argsort(before)
            selector.topk[row] = torch.from_numpy(inverse[selected].astype(np.int32)).repeat(4, 1)
            # One selected candidate hot prefetch, one unused cold prefetch,
            # and one final cold transfer. Only ID/slot metadata is simulated.
            rotating.mapping[row, inverse[[2048, 5001]]] = torch.tensor([resident, resident + 1], dtype=torch.int32)
            rotating.fallback_addresses[row, 0] = storage.addresses[int(owner), inverse[5000]]
            rotating.destinations[row * 8192] = int(owner) * stride + 2050
            rotating.rows[row, 247] = 1
            rotating.ubmem_status[row, [1, 2, 3, 4, 5]] = torch.tensor([2, 2, 1, 1, 3 * width])
            for destination, token in ((2049, 5001), (2050, 5000)):
                physical = int(np.flatnonzero(storage.logical_ids[int(owner)].numpy() == token)[0])
                storage.logical_ids[int(owner), [destination, physical]] = storage.logical_ids[
                    int(owner), [physical, destination]
                ]
            slots = inverse[selected].astype(np.int32)
            slots[-3:] = [2048, 2050, resident + 8187 % (stride - resident)]
            for tnd in torch.nonzero(metadata[2] // 4 == row).reshape(-1):
                update.logical_output[tnd] = torch.from_numpy(selected)
                update.slot_output[tnd] = torch.from_numpy(slots)
        if tagged:
            raw = rotating.mapping.numpy().view(np.uint32)
            mask = raw != np.uint32(0xFFFFFFFF)
            raw[mask] |= np.uint32(7 << 18)
            rotating.rows[:, 0] = 7
            rotating.rows[:, 1] = 0x4D415031

    return selector, source, complete


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("high_addresses,tagged", [(False, False), (True, True)])
def test_actual_recorder_roundtrip_id_counts_no_payload_or_addresses(tmp_path, batch, high_addresses, tagged):
    selector, source, complete = fixture(batch, high_addresses, tagged)
    recorder = TokenIdTraceRecorder(torch, tmp_path / "capture", source)
    recorder.before(selector)
    assert not recorder.completed and not (recorder.directory / "manifest.json").exists()
    complete()
    recorder.after(selector)
    assert recorder.completed and recorder._addresses is None
    assert not (recorder.directory / "INCOMPLETE").exists()
    archive = load_token_trace(recorder.directory)
    assert archive.manifest["validation"]["passed"]
    assert "published_pages" not in archive.manifest["metadata"]
    assert "oracle" not in archive.manifest
    assert sorted(p.name for p in recorder.directory.iterdir()) == ["manifest.json", "token_ids.npz"]
    assert not any(
        name.endswith(("keys", "scales", "query", "output", "addresses", "payload"))
        for name in archive.arrays
        if not name.startswith("post.")
    )
    assert archive.array("pre.logical_ids").dtype == np.int32
    assert archive.array("pre.logical_ids").shape == (batch, 8192)
    assert archive.array("pre.ddr_page_slots")[0, 0] == 3 * 64
    assert archive.array("pre.ddr_page_slots")[0, -1] == -1
    summary = archive.manifest["summary"]
    assert summary["demand"]["cold_demand"] == batch
    assert summary["demand"]["candidate_hot_refetch"] == batch
    assert summary["demand"]["abc_hit"] == batch
    assert summary["demand"]["selected_unique"] == batch * 2048
    assert summary["reads"] == dict(
        prefetch=2 * batch,
        final=batch,
        unused_prefetch=batch,
        total=3 * batch,
        bytes=3 * batch * 2624,
        unused_bytes=batch * 2624,
    )
    assert summary["transfer_classes"]["prefetch_candidate_hot"] == batch
    assert summary["transfer_classes"]["unused_prefetch_cold"] == batch
    assert summary["transfer_classes"]["final_cold"] == batch
    assert all(record[2] == 2050 for record in archive.array("post.final_records"))


@pytest.mark.parametrize(
    "fault,match",
    [
        ("causal", "causal"),
        ("positions", "positions"),
        ("permutation", "permutation"),
        ("slot", "hot slots"),
        ("counter", "counters"),
        ("completion", "pending"),
        ("missing_source", "no pre-call"),
    ],
)
def test_recorder_failure_never_publishes_complete(tmp_path, fault, match):
    selector, source, complete = fixture()
    recorder = TokenIdTraceRecorder(torch, tmp_path / "capture", source)
    recorder.before(selector)
    complete()
    if fault == "causal":
        selector.update.logical_output[0, -1] = 8191
        selector.topk[1, 0, -1] = 8191
    elif fault == "positions":
        selector.topk[0, 0, 0] = 4000
    elif fault == "permutation":
        selector.storage.logical_ids[3, 100] = 101
    elif fault == "slot":
        selector.update.slot_output[0, 0] = 50
    elif fault == "counter":
        selector.rotating.ubmem_status[:, 1:3] += 1
        selector.rotating.ubmem_status[:, 5] += 2624
    elif fault == "completion":
        selector.rotating.rows[0, 248] = 1
    else:
        selector.rotating.fallback_addresses[0, 0] = 12345
    with pytest.raises((ValueError, AssertionError), match=match):
        recorder.after(selector)
    assert not recorder.completed and recorder._addresses is None
    assert not (recorder.directory / "manifest.json").exists()


@pytest.mark.parametrize(
    "fault,match",
    [
        ("partial_page", "complete logical pages"),
        ("noncontiguous", "contiguous"),
        ("owner", "owners"),
        ("q3", "four real"),
        ("permutation", "permutation"),
    ],
)
def test_before_rejects_invalid_metadata_and_published_domain(tmp_path, fault, match):
    selector, source, _ = fixture()
    if fault == "partial_page":
        selector.storage.addresses[3, 100] = 0
    elif fault == "noncontiguous":
        selector.storage.addresses[3, 100] += 32
    elif fault == "owner":
        selector.inputs.metadata[4].view(torch.int32)[0] = 1
    elif fault == "q3":
        selector.inputs.metadata[1, 0] = 0
    else:
        selector.storage.logical_ids[3, 100] = 101
    with pytest.raises(ValueError, match=match):
        TokenIdTraceRecorder(torch, tmp_path / "capture", source).before(selector)
    assert not (tmp_path / "capture" / "manifest.json").exists()


def test_loader_detects_tamper_and_mislabeled_full_validation(tmp_path):
    selector, source, complete = fixture(1)
    recorder = TokenIdTraceRecorder(torch, tmp_path / "capture", source)
    recorder.before(selector)
    complete()
    recorder.after(selector)
    path = recorder.directory / "token_ids.npz"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        load_token_trace(recorder.directory)
    manifest = copy.deepcopy(recorder.manifest)
    manifest["archive"]["sha256"] = file_digest(path)
    manifest["summary"]["demand"]["cold_demand"] = 0
    (recorder.directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="summary"):
        load_token_trace(recorder.directory)


def test_recorder_rejects_reuse_and_overwrite(tmp_path):
    selector, source, _ = fixture(1)
    recorder = TokenIdTraceRecorder(torch, tmp_path / "capture", source)
    recorder.before(selector)
    with pytest.raises(RuntimeError, match="exactly one"):
        recorder.before(selector)
    with pytest.raises(FileExistsError):
        TokenIdTraceRecorder(torch, recorder.directory, source).before(selector)


def test_shared_published_page_retains_original_slot_identity(tmp_path):
    selector, source, complete = fixture(2, high_addresses=True)
    selector.storage.addresses[1, :128].copy_(selector.storage.addresses[3, :128])
    recorder = TokenIdTraceRecorder(torch, tmp_path / "capture", source)
    recorder.before(selector)
    complete()
    recorder.after(selector)
    archive = load_token_trace(recorder.directory)
    slots = archive.array("pre.ddr_page_slots")
    assert slots[0, 0] == slots[1, 0] == 192
    assert slots[0, 1] != slots[1, 1]
    assert "region_base" not in archive.manifest["geometry"]
    assert archive.manifest["geometry"]["numa_node"] == 2


def test_loader_requires_id_only_validation_marker(tmp_path):
    selector, source, complete = fixture(1)
    recorder = TokenIdTraceRecorder(torch, tmp_path / "capture", source)
    recorder.before(selector)
    complete()
    recorder.after(selector)
    manifest = copy.deepcopy(recorder.manifest)
    manifest["validation"]["kind"] = "full_payload_oracle"
    (recorder.directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="ID-integrity"):
        load_token_trace(recorder.directory)


def test_all_hot_call_has_zero_observed_gets(tmp_path):
    selector, source, _ = fixture(1)
    recorder = TokenIdTraceRecorder(torch, tmp_path / "capture", source)
    recorder.before(selector)
    selector.topk[0] = torch.arange(2048, dtype=torch.int32).repeat(4, 1)
    selector.update.logical_output[:] = selector.storage.logical_ids[3, :2048].to(torch.int32)
    selector.update.slot_output[:] = torch.arange(2048, dtype=torch.int32)
    recorder.after(selector)
    archive = load_token_trace(recorder.directory)
    assert archive.array("post.final_records").shape == (0, 3)
    assert archive.array("post.prefetch_records").shape == (0, 3)
    assert archive.manifest["summary"]["reads"]["total"] == 0
    assert archive.manifest["summary"]["demand"]["hot_hit"] == 2048
