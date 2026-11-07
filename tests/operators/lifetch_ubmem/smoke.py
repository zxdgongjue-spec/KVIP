#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real KVAF Put -> UBmem LI/TopK/Fetch -> Index update, with no model weights."""

import argparse
from pathlib import Path

import numpy as np

from kvip.adapters.ubmem import UbmemLease
from kvip.cache.index_storage import IndexCacheStorage
from kvip.operators.lifetch_ubmem.binding import UbmemBinding
from kvip.operators.lifetch_ubmem.forward import UbmemForward
from kvip.operators.lifetch_ubmem.index_update import UbmemIndexUpdate
from kvip.operators.lifetch_ubmem.specialization import Specialization, build_specialization, kvaf_cann
from kvip.operators.lifetch_ubmem.workspace import UbmemWorkspace
from kvip.runtime.device import query_device_resources


def payload(owner, tokens, token_bytes):
    tokens = np.asarray(tokens, dtype=np.uint64).reshape(-1)
    columns = np.arange(token_bytes, dtype=np.uint16)
    values = ((tokens[:, None] % 251 + columns[None, :] + owner * 17) % 251).astype(np.uint8)
    values[:, :8] = (tokens + np.uint64(owner << 32)).astype("<u8").view(np.uint8).reshape(-1, 8)
    return values


def check_shape(kvaf, torch, build, device_id, shape, library):
    batch, history, resident, token_bytes = shape.batch, shape.history, shape.resident, shape.token_bytes
    device = torch.device(f"npu:{device_id}")
    block_bytes = 128 * token_bytes
    slot_bytes = (block_bytes + 4095) // 4096 * 4096
    blocks = history // 128
    keys = list(range(batch * blocks))
    source_cpu = np.concatenate([payload(owner, np.arange(history), token_bytes) for owner in range(1, batch + 1)])
    source = torch.from_numpy(source_cpu).to(device)
    runtime = kvaf.NpuRuntime(
        str(build / "bin/kvaf_ubmem_batch_get.o"),
        token_bytes,
        128,
        device_id=device_id,
        npu_core_count=2,
        batch_put_kernel_object_path=str(build / "bin/kvaf_ubmem_batch_put.o"),
        transport="ubmem",
    )
    table = lease = graph = None
    stream = torch.npu.current_stream(device)
    try:
        runtime.initialize((len(keys) + 2) * slot_bytes, slot_bytes)
        runtime.batch_put(keys, source, [i * block_bytes for i in keys], [block_bytes] * len(keys))
        table = runtime.prepare_access_table(keys)
        device_table = runtime.access_table_view(table, device="npu")
        storage = IndexCacheStorage(torch, batch + 1, history, resident, device)
        storage.keys.fill_(0x38)  # FP8 1.0; cold scales dominate the small key variation.
        key_columns = np.where(np.arange(history) % 2, 0x40, 0x38).astype(np.uint8)
        storage.keys[1:, :, 0].copy_(torch.from_numpy(key_columns).to(device))
        storage.scales[:, -2048:].fill_(8)
        storage.hot_valid[1:].fill_(1)
        offsets = torch.arange(128, dtype=torch.int64, device=device) * token_bytes
        storage.addresses[1:].copy_((device_table.addresses.view(batch, blocks, 1) + offsets).reshape(batch, history))
        original_addresses = storage.addresses.cpu().numpy()
        stride = resident
        output = torch.zeros((batch + 1, stride, token_bytes), dtype=torch.uint8, device=device)
        output[1:, :resident].copy_(source.view(batch, history, token_bytes)[:, :resident])
        # Match vLLM policy.bytes_outputs: KVAF registers one token per row.
        # Keep the owner-shaped alias only for payload verification below.
        output_rows = output.view(-1, token_bytes)
        workspace = UbmemWorkspace(
            torch,
            batch=batch,
            history=history,
            resident=resident,
            aics=shape.aic_count,
            windows=shape.intermediate_fetches,
            device=device,
            index_heads=shape.heads,
        )
        workspace.inputs.prepare([4] * batch, query_owners=list(range(1, batch + 1)))
        workspace.inputs.query.view(torch.uint8).fill_(0x38)
        workspace.inputs.weights.fill_(1)
        workspace.inputs.scale.fill_(1)
        workspace.inputs.query_ends.fill_(history)
        lease = UbmemLease(torch, runtime, output_rows, stream)
        lease.register_table(device_table)
        rotating = UbmemBinding(
            torch,
            library,
            workspace.arguments(storage),
            lease,
            output_rows,
            shape=shape,
            query_ends=workspace.inputs.query_ends.view(batch, 4),
            hot_valid=storage.hot_valid,
            row_owners=workspace.inputs.row_owners,
            token_addresses=storage.addresses,
            output_stride=stride,
            scratch=workspace.scratch,
        )
        # Match KVCachePolicy.prepare: candidates and their inverse reservation
        # map are one input. Binding initializes reservations to -1, so publish
        # both after construction. On later rounds prefetch may overwrite a
        # candidate that LI still selects; it must be refetched, not read as a hit.
        ordinals = torch.arange(2048, dtype=torch.int32, device=device)
        rotating.lru_slots.copy_(ordinals)
        rotating.reserved_hot[:, :2048].copy_(ordinals)
        update = UbmemIndexUpdate(
            torch,
            keys=storage.keys,
            scales=storage.scales,
            addresses=storage.addresses,
            logical_ids=storage.logical_ids,
            row_owners=workspace.inputs.row_owners,
            rotating=rotating,
            status=torch.zeros(batch, dtype=torch.int32, device=device),
            query_inputs=workspace.inputs,
            outputs=torch.empty((2, batch * 4, 1, 2048), dtype=torch.int32, device=device),
            plan_buffer=workspace.index_plan,
            extended_work=workspace.index_work,
        )
        forward = UbmemForward(rotating, update)

        def verify(owners, *, phase, counts=None, padding=False):
            stream.synchronize()
            assert not np.any(update.status.cpu().numpy()), "Index update rejected UBmem completion"
            counters = rotating.ubmem_status.cpu().numpy()
            assert not np.any(counters[:, 0]), "UBmem copy failed"
            assert np.array_equal(counters[:, 1], counters[:, 2]), "UBmem prefetch incomplete"
            assert np.array_equal(counters[:, 3], counters[:, 4]), "UBmem final fetch incomplete"
            logical = update.logical_output.cpu().numpy().reshape(batch * 4, 2048)
            slots = update.slot_output.cpu().numpy().reshape(batch * 4, 2048)
            if padding:
                assert np.all(logical == -1) and np.all(slots == -1), "Padding inherited live outputs"
                return
            expected_ids = np.arange(history - 2048, history)
            ids = storage.logical_ids.cpu().numpy()
            addresses = storage.addresses.cpu().numpy()
            scales = storage.scales.cpu().numpy()
            key_values = storage.keys[:, :, 0].contiguous().cpu().numpy()
            counts = [4] * len(owners) if counts is None else counts
            output_row = 0
            for owner, count in zip(owners, counts):
                assert np.array_equal(np.sort(ids[owner]), np.arange(history)), "Index ownership is not a permutation"
                assert np.array_equal(addresses[owner], original_addresses[owner, ids[owner]]), (
                    "Index addresses detached"
                )
                assert np.array_equal(scales[owner], np.where(ids[owner] >= history - 2048, 8, 1)), (
                    "Index scales detached"
                )
                assert np.array_equal(key_values[owner], key_columns[ids[owner]]), "Index keys detached"
                # Tags check the persistent hot-cache invariant, including
                # prefetched entries absent from this round's final TopK.
                hot_valid = storage.hot_valid[owner].cpu().numpy() != 0
                hot_tags = output[owner, :resident, :8].contiguous().cpu().numpy().view("<u8").reshape(-1)
                expected_tags = ids[owner, :resident].astype(np.uint64) + np.uint64(owner << 32)
                bad_hot = np.flatnonzero(hot_valid & (hot_tags != expected_tags))
                if bad_hot.size:
                    slot = int(bad_hot[0])
                    raise AssertionError(
                        f"Hot slot/Index mismatch: phase={phase} device={device_id} batch={batch} "
                        f"token_bytes={token_bytes} owner={owner} slot={slot} "
                        f"expected_tag={int(expected_tags[slot])} actual_tag={int(hot_tags[slot])}"
                    )
                for _ in range(count):
                    assert np.array_equal(np.sort(logical[output_row]), expected_ids), "LI selected the wrong TopK set"
                    selected_slots = torch.from_numpy(slots[output_row].astype(np.int64)).to(device)
                    actual = torch.index_select(output[owner], 0, selected_slots).cpu().numpy()
                    expected = payload(owner, logical[output_row], token_bytes)
                    mismatch = np.flatnonzero(np.any(actual != expected, axis=1))
                    if mismatch.size:
                        k = int(mismatch[0])
                        byte = int(np.flatnonzero(actual[k] != expected[k])[0])
                        raise AssertionError(
                            f"Fused Fetch bytes differ from KVAF Put: phase={phase} device={device_id} "
                            f"batch={batch} token_bytes={token_bytes} owner={owner} query={output_row} "
                            f"token={int(logical[output_row, k])} slot={int(slots[output_row, k])} "
                            f"byte={byte} expected={int(expected[k, byte])} actual={int(actual[k, byte])}"
                        )
                    output_row += 1
            assert np.all(logical[output_row:] == -1) and np.all(slots[output_row:] == -1), "TND padding is stale"

        forward()
        verify(list(range(1, batch + 1)), phase="eager")
        first_counters = rotating.ubmem_status.cpu().numpy()
        assert np.any(first_counters[:, 1] + first_counters[:, 3]), "Fixture did not exercise a cold DDR fetch"
        graph = kvaf.AclGraph()
        with lease.capture_scope():
            graph.capture_begin(stream)
            forward()
            graph.capture_end()
        lease.retain_graph(graph)
        # Replay after changing active row ownership. Index rows and output HBM
        # belong to stable owners, not capture-time packed row positions.
        owners = list(range(batch, 0, -1))
        workspace.inputs.prepare([4] * batch, query_owners=owners)
        workspace.inputs.query_ends.fill_(history)
        graph.replay(stream)
        verify(owners, phase="graph_replay")
        if batch == 2:
            # Compact TND output row 1..4 belongs to the second packed request;
            # output block number is not its producer/status block number.
            counts = [1, 4]
            workspace.inputs.prepare(counts, query_owners=owners)
            workspace.inputs.query_ends.view(batch, 4).copy_(
                torch.tensor([[history, 0, 0, 0], [history] * 4], dtype=torch.int32, device=device)
            )
            graph.replay(stream)
            verify(owners, phase="packed_queries", counts=counts)
        # All-padding warmup is legal: owner zero can repeat in inactive rows.
        workspace.inputs.prepare([0] * batch, query_owners=[0] * batch)
        workspace.inputs.query_ends.zero_()
        before = storage.logical_ids.clone()
        graph.replay(stream)
        verify([], phase="padding", padding=True)
        assert torch.equal(storage.logical_ids, before), "Padding changed Index ownership"
        graph.reset()
        graph = None
        lease.close()
        lease = None
        runtime.release_access_table(table)
        table = None
        runtime.batch_free(keys)
        runtime.shutdown()
    finally:
        # A failed fence deliberately leaves the lease and its buffers retained
        # by KVAF. Never continue freeing storage after uncertain completion.
        if graph is not None:
            if graph.capturing:
                graph.capture_end()
            if graph.ready:
                graph.synchronize(stream)
            graph.reset()
        if lease is not None:
            lease.close()
        if table is not None:
            runtime.release_access_table(table)
        if runtime.initialized:
            runtime.shutdown()
    print(
        f"UBMEM_LIFETCH_PASS device={device_id} batch={batch} token_bytes={token_bytes} "
        "put=topk=fetch=index_update=graph=packed_queries=padding"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--soc", required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--token-bytes", type=int, nargs="+", default=[656, 2624])
    parser.add_argument("--history", type=int, default=33024)
    parser.add_argument("--resident", type=int, default=8192)
    args = parser.parse_args()
    import kvaf
    import torch
    import torch_npu  # noqa: F401

    build = args.build_dir.resolve()
    if not Path(kvaf.__file__).resolve().is_relative_to(build / "python/kvaf"):
        raise RuntimeError(f"Wrong KVAF module {kvaf.__file__}; source the selected KVIP scripts/env.sh")
    if not torch.npu.is_available():
        raise RuntimeError("UBmem fusion smoke requires a real NPU; it cannot be skipped")
    torch.npu.set_device(args.device)
    cann = kvaf_cann(build)
    aics = query_device_resources(cann, args.device)["effective_aic"]
    for batch in args.batches:
        for token_bytes in args.token_bytes:
            shape = Specialization(batch, args.history, args.resident, 64, token_bytes, aics, 8, args.soc)
            library = build_specialization(cache_dir=build / "lifetch_ubmem", cann=cann, shape=shape)
            check_shape(kvaf, torch, build, args.device, shape, library)


if __name__ == "__main__":
    main()
