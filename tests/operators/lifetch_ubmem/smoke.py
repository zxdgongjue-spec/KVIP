#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real KVAF Put -> UBmem LI/TopK/Fetch -> Index update, with no model weights."""

import argparse
import ctypes
import hashlib
import sys
import tempfile
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
    """Same packed C8 records used by the URMA numerical oracle."""
    from contract_payload import payload as layer_payload

    return np.concatenate(
        [layer_payload(owner - 1 + layer * 1024, tokens) for layer in range(token_bytes // 656)], axis=1
    )


def controlled_fixture(history, resident, heads, miss_percent):
    """Use precisely the same deterministic Q4 fixture as URMA's A/B test."""
    support = Path(__file__).resolve().parents[1] / "lifetch/support"
    if str(support) not in sys.path:
        sys.path.insert(0, str(support))
    from real_fetch_inputs import prepare_full_chain_fixture

    with tempfile.TemporaryDirectory(prefix="kvip-ubmem-fixture-") as tmp:
        root = Path(tmp) / "fixture"
        info = prepare_full_chain_fixture(root, history, resident, miss_percent, index_heads=heads)
        data = {
            "info": info,
            "order": np.load(root / "physical_to_logical.npy").astype(np.int64),
            "keys": np.fromfile(root / "arg1.bin", np.uint8).reshape(history, 128),
            "scales": np.fromfile(root / "arg4.bin", np.float32),
            "query": np.fromfile(root / "arg0.bin", np.uint8).reshape(4, heads, 128),
            "weights": np.fromfile(root / "arg2.bin", np.float32).reshape(4, heads),
            "query_scale": np.fromfile(root / "arg3.bin", np.float32).reshape(4, heads),
            "scores": np.load(root / "logical_scores.npy"),
        }
    order = data["order"]
    assert len(order) == history and np.array_equal(np.sort(order), np.arange(history))
    # URMA's single-operator harness identifies tokens by physical fixture row.
    # Its source payload and initial hot slots use these same row IDs.
    physical_scores = data["scores"][:, order]
    data["expected"] = np.sort(np.argpartition(physical_scores, -2048, axis=1)[:, -2048:], axis=1)
    all_cold = np.unique(data["expected"][data["expected"] >= resident])
    data["native_miss_union"] = len(all_cold)
    return data


def mixed_query_counts(batch):
    """Exercise Q1-Q4 and one inactive row in the same graph replay."""
    counts = [1 + row % 4 for row in range(batch)]
    if batch >= 5:
        counts[-1] = 0
    return counts


def check_shape(
    kvaf,
    torch,
    build,
    device_id,
    shape,
    library,
    prefetch_tokens=512,
    args_server="127.0.0.1:28880",
    miss_percent=100,
    warmup=0,
    iterations=0,
):
    batch, history, resident, token_bytes = shape.batch, shape.history, shape.resident, shape.token_bytes
    fixture = controlled_fixture(history, resident, shape.heads, miss_percent)
    expected_ids = fixture["expected"]
    cold_count = fixture["native_miss_union"]
    logical_scales = fixture["scales"]
    logical_key_columns = fixture["keys"][:, 0]
    device = torch.device(f"npu:{device_id}")
    block_bytes = 128 * token_bytes
    slot_bytes = (block_bytes + 4095) // 4096 * 4096
    blocks = history // 128
    keys = list(range(batch * blocks))
    source_cpu = np.concatenate([payload(owner, np.arange(history), token_bytes) for owner in range(1, batch + 1)])
    source = torch.from_numpy(source_cpu).to(device)
    runtime = kvaf.NpuRuntime(
        str(build / "bin/kvaf_execute_batch_get_kernels.o"),
        token_bytes,
        128,
        device_id=device_id,
        npu_core_count=2,
        batch_put_kernel_object_path=str(build / "bin/kvaf_execute_batch_put_kernels.o"),
        transport="ubmem",
    )
    table = lease = graph = None
    stream = torch.npu.current_stream(device)
    try:
        runtime.link_to(args_server)
        runtime.initialize((len(keys) + 2) * slot_bytes, slot_bytes)
        runtime.batch_put(keys, source, [i * block_bytes for i in keys], [block_bytes] * len(keys))
        table = runtime.prepare_access_table(keys)
        device_table = runtime.access_table_view(table, device="npu")
        storage = IndexCacheStorage(torch, batch + 1, history, resident, device)
        key_tensor = torch.from_numpy(fixture["keys"]).to(device)
        scale_tensor = torch.from_numpy(fixture["scales"]).to(device)
        storage.keys[1:].copy_(key_tensor.expand(batch, -1, -1))
        storage.scales[1:].copy_(scale_tensor.expand(batch, -1))
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
        workspace.inputs.query.view(torch.uint8).copy_(
            torch.from_numpy(fixture["query"]).to(device).repeat(batch, 1, 1)
        )
        workspace.inputs.weights.copy_(
            torch.from_numpy(fixture["weights"]).to(device, dtype=torch.bfloat16).repeat(batch, 1)
        )
        workspace.inputs.scale.copy_(torch.from_numpy(fixture["query_scale"]).to(device).repeat(batch, 1))
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
            prefetch_limit=prefetch_tokens,
        )
        # Check the compiled UBmem schedule against the URMA runtime contract,
        # including epoch zero at the end of the hot region.
        from kvip.operators.lifetch.geometry import trigger_tokens

        trigger = rotating.library.v9_window_trigger_tile
        trigger.argtypes, trigger.restype = [ctypes.c_uint32, ctypes.c_void_p], ctypes.c_uint32
        actual_triggers = [128 * trigger(epoch, ctypes.byref(rotating.runtime_shape)) for epoch in range(rotating.rounds)]
        expected_triggers = trigger_tokens(
            batch, history, resident, intermediate_fetches=shape.intermediate_fetches, aic_count=shape.aic_count
        )
        if actual_triggers != expected_triggers:
            raise AssertionError(f"UBmem/URMA prefetch window mismatch: {actual_triggers} != {expected_triggers}")
        print(f"UBMEM_WINDOW_TRIGGERS tokens={actual_triggers}", flush=True)
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
            index_status = update.status.cpu().tolist()
            if any(index_status):
                rows = rotating.rows.cpu().tolist()
                diagnostics = dict(
                    phase=phase,
                    prefetch_tokens=prefetch_tokens,
                    index_status=index_status,
                    ubmem_status=rotating.ubmem_status.cpu().tolist(),
                    row_status=rotating.row_status.cpu().tolist(),
                    final_posts=rotating.final_posts[:, :, :8].cpu().tolist(),
                    change_count=rotating.change_count.cpu().tolist(),
                    row_fields=[{i: row[i] for i in (8, 9, 11, 208, 209, 210, 211, 247, 248)} for row in rows],
                )
                raise AssertionError(f"Index update rejected UBmem completion: {diagnostics}")
            counters = rotating.ubmem_status.cpu().numpy()
            assert not np.any(counters[:, 0]), "UBmem copy failed"
            assert np.array_equal(counters[:, 1], counters[:, 2]), "UBmem prefetch incomplete"
            assert np.array_equal(counters[:, 3], counters[:, 4]), "UBmem final fetch incomplete"
            assert np.all(counters[:, 1] <= prefetch_tokens), "Prefetch exceeded its total budget"
            epochs = rotating.rows.cpu().numpy()[:, 16 : 16 + rotating.rounds]
            assert np.all(epochs <= max(1, prefetch_tokens // shape.intermediate_fetches)), (
                "Prefetch exceeded its per-window budget"
            )
            logical = update.logical_output.cpu().numpy().reshape(batch * 4, 2048)
            slots = update.slot_output.cpu().numpy().reshape(batch * 4, 2048)
            if padding:
                assert np.all(logical == -1) and np.all(slots == -1), "Padding inherited live outputs"
                return
            ids = storage.logical_ids.cpu().numpy()
            addresses = storage.addresses.cpu().numpy()
            scales = storage.scales.cpu().numpy()
            key_values = storage.keys[:, :, 0].contiguous().cpu().numpy()
            counts = [4] * len(owners) if counts is None else counts
            output_row = 0
            for owner, count in zip(owners, counts):
                if not count:
                    continue
                assert np.array_equal(np.sort(ids[owner]), np.arange(history)), "Index ownership is not a permutation"
                assert np.array_equal(addresses[owner], original_addresses[owner, ids[owner]]), (
                    "Index addresses detached"
                )
                assert np.array_equal(scales[owner], logical_scales[ids[owner]]), "Index scales detached"
                assert np.array_equal(key_values[owner], logical_key_columns[ids[owner]]), "Index keys detached"
                # Verify every valid persistent hot entry, not just selected rows.
                hot_valid = storage.hot_valid[owner].cpu().numpy() != 0
                hot_slots = np.flatnonzero(hot_valid)
                actual_hot = output[owner, :resident].cpu().numpy()[hot_slots]
                expected_hot = payload(owner, ids[owner, hot_slots], token_bytes)
                if not np.array_equal(actual_hot, expected_hot):
                    difference = np.argwhere(actual_hot != expected_hot)[0]
                    raise AssertionError(
                        f"Hot slot/Index mismatch: phase={phase} owner={owner} "
                        f"slot={hot_slots[difference[0]]} byte={difference[1]}"
                    )
                for query in range(count):
                    assert np.array_equal(np.sort(logical[output_row]), expected_ids[query]), (
                        "LI selected the wrong TopK set"
                    )
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

        # Restore the exact initial mapping/cache outside the event pair.
        measured_buffers = (
            storage.keys,
            storage.scales,
            storage.addresses,
            storage.logical_ids,
            storage.hot_valid,
            output,
        )
        initial_state = tuple(t.clone() for t in measured_buffers) if iterations else ()

        # Compare LI selection against the independent torch_npu native LI on
        # exactly the same query, Index cache, scales and block table. This is
        # outside graph timing and happens before Index ownership is updated.
        import torch_npu

        native = torch_npu.npu_quant_lightning_indexer(
            query=workspace.inputs.query.reshape(batch, 4, shape.heads, 128),
            key=storage.keys.view(torch.float8_e4m3fn).reshape((batch + 1) * blocks, 128, 1, 128),
            weights=workspace.inputs.weights.reshape(batch, 4, shape.heads),
            query_dequant_scale=workspace.inputs.scale.reshape(batch, 4, shape.heads),
            key_dequant_scale=storage.scales.reshape((batch + 1) * blocks, 128, 1),
            actual_seq_lengths_query=workspace.query_lengths,
            actual_seq_lengths_key=workspace.key_lengths,
            block_table=workspace.inputs.block_table,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="BSND",
            layout_key="PA_BSND",
            sparse_count=2048,
            sparse_mode=0,
        )
        native_ids = (native[0] if isinstance(native, (tuple, list)) else native).cpu().numpy().reshape(batch, 4, 2048)
        if not np.all(np.sort(native_ids, axis=-1) == expected_ids[None, :, :]):
            raise AssertionError("Native LI differs from the independent controlled TopK oracle")
        # The URMA test commits this same native TopK to the service LRU before
        # choosing victims. Match the exact ordered candidates and inverse map.
        from kvip.cache.lru import SlotLRU

        candidates = np.empty((batch, 2048), dtype=np.int32)
        reserved = np.full((batch, resident), -1, dtype=np.int32)
        for row in range(batch):
            policy = SlotLRU()
            policy.resize(resident, resident)
            hits = native_ids[row].reshape(-1)
            policy.commit(hits[hits < resident])
            slot = int(policy.head[0])
            for i in range(2048):
                candidates[row, i] = slot
                slot = int(policy.next[0, slot])
            reserved[row, candidates[row]] = np.arange(2048, dtype=np.int32)
        rotating.lru_slots.copy_(torch.from_numpy(candidates).to(device))
        rotating.reserved_hot.copy_(torch.from_numpy(reserved).to(device))
        del native, native_ids

        forward()
        verify(list(range(1, batch + 1)), phase="eager")
        # Actual service SFA with the URMA fixture's packed C8 payload, using
        # this fused path's final token-to-slot mapping. All outside timing.
        from types import SimpleNamespace
        from attention import check_sfa

        selected_ids = update.logical_output.cpu().numpy().reshape(batch * 4, 1, 2048)
        selected_slots = update.slot_output.cpu().numpy().reshape(batch * 4, 1, 2048)
        physical_ids = storage.logical_ids.cpu().numpy()
        forward_map = np.full((batch, history), -1, np.int32)
        for row in range(batch):
            forward_map[row, physical_ids[row + 1, :resident]] = np.arange(resident)
            forward_map[row, selected_ids[row * 4 : (row + 1) * 4].reshape(-1)] = selected_slots[
                row * 4 : (row + 1) * 4
            ].reshape(-1)
        block_table = torch.arange(batch * resident // 128, device=device, dtype=torch.int32).reshape(batch, -1)
        geometry = SimpleNamespace(
            batch=batch, queries=4, num_layers=1, layer_id=0, resident_blocks=batch * resident // 128
        )
        before = {"pool_rows": np.arange(batch), "budgets": np.full(batch, resident)}
        after = {"topk": selected_ids, "forward": forward_map}
        tensors = dict(
            query=workspace.inputs.query,
            resident_output=output[1:, :, :656].contiguous(),
            forward=torch.from_numpy(forward_map).to(device),
            pool_rows=torch.arange(batch, dtype=torch.int32, device=device),
            topk=torch.from_numpy(selected_ids).to(device),
            resident_table=block_table,
            cumulative_query=torch.arange(1, batch + 1, dtype=torch.int32, device=device) * 4,
            logical_lens=torch.full((batch,), history, dtype=torch.int32, device=device),
        )
        sfa = check_sfa(torch, tensors, geometry, before, after)
        print(f"UBMEM_SFA_PASS max_abs={sfa['max_abs']:.8f}", flush=True)
        first_counters = rotating.ubmem_status.cpu().numpy()
        print(
            "UBMEM_FETCH_COUNTS "
            f"prefetch_posted={int(first_counters[:, 1].sum())} "
            f"prefetch_completed={int(first_counters[:, 2].sum())} "
            f"final_posted={int(first_counters[:, 3].sum())} "
            f"final_completed={int(first_counters[:, 4].sum())} "
            f"native_required={batch * cold_count} "
            f"requested_miss_pct={miss_percent:g} actual_misses_per_query={fixture['info']['final_misses_per_query']} "
            f"Q4_union_per_request={cold_count} fixture=shared_urma_seed20261104",
            flush=True,
        )
        if cold_count:
            assert np.any(first_counters[:, 1] + first_counters[:, 3]), "Fixture did not exercise a cold DDR fetch"
        assert np.all(first_counters[:, 3] >= max(0, cold_count - prefetch_tokens)), (
            "Final repair must fetch all missing TopK rows even beyond the speculative budget"
        )
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
        # Compact TND outputs no longer have the fixed four-row stride of
        # their source request. Exercise all Q1..Q4 lengths at the requested
        # batch, including an inactive/padded row when the batch permits it.
        counts = mixed_query_counts(batch)
        packed_owners = [owner if count else 0 for owner, count in zip(owners, counts)]
        workspace.inputs.prepare(counts, query_owners=packed_owners)
        ends = torch.tensor(
            [[history if q < count else 0 for q in range(4)] for count in counts],
            dtype=torch.int32, device=device,
        )
        workspace.inputs.query_ends.view(batch, 4).copy_(ends)
        graph.replay(stream)
        verify(packed_owners, phase="packed_queries", counts=counts)
        # All-padding warmup is legal: owner zero can repeat in inactive rows.
        workspace.inputs.prepare([0] * batch, query_owners=[0] * batch)
        workspace.inputs.query_ends.zero_()
        before = storage.logical_ids.clone()
        graph.replay(stream)
        verify([], phase="padding", padding=True)
        assert torch.equal(storage.logical_ids, before), "Padding changed Index ownership"
        if iterations:
            workspace.inputs.prepare([4] * batch, query_owners=list(range(1, batch + 1)))
            workspace.inputs.query_ends.fill_(history)
            elapsed = []
            for turn in range(warmup + iterations):
                for tensor, saved in zip(measured_buffers, initial_state):
                    tensor.copy_(saved)
                begin = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                begin.record(stream)
                graph.replay(stream)
                end.record(stream)
                end.synchronize()
                if turn >= warmup:
                    elapsed.append(begin.elapsed_time(end) * 1000)
            verify(list(range(1, batch + 1)), phase="timed_replay")
            print(
                f"UBMEM_GRAPH_EVENT_US p50={np.percentile(elapsed, 50):.3f} "
                f"p99={np.percentile(elapsed, 99):.3f} iterations={iterations} "
                "includes=init,LI,prefetch,resolve,final_fetch,index_update "
                "excludes=fixture_restore,native_li,verification,CPU_LRU,SFA",
                flush=True,
            )
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
        f"UBMEM_LIFETCH_PASS device={device_id} batch={batch} heads={shape.heads} "
        f"history={history} resident={resident} token_bytes={token_bytes} "
        f"requested_miss_pct={miss_percent:g} Q4_unique_misses={cold_count} "
        f"prefetch_tokens={prefetch_tokens} "
        "put=native_li=topk=fetch=sfa=index_update=graph=packed_queries=padding"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--kvaf-server", required=True, help="KVAF CPU service host:port")
    parser.add_argument("--soc", required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--token-bytes", type=int, nargs="+", default=[656, 2624])
    parser.add_argument("--index-heads", type=int, choices=(32, 64), default=32)
    parser.add_argument("--history", type=int, default=33024)
    parser.add_argument("--resident", type=int, default=8192)
    parser.add_argument("--prefetch-tokens", type=int, nargs="+", default=[512])
    parser.add_argument(
        "--miss-percent", type=float, default=100, help="Controlled per-query miss; fixture matches the URMA benchmark"
    )
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--runtime-shapes", action="store_true", help="Also check a second history/hot shape with the same binary")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 0:
        parser.error("warmup/iterations must be non-negative")
    if not 0 <= args.miss_percent <= 100:
        parser.error("miss-percent must be in 0..100")
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
    selected_library = None
    cases = [(batch, args.history, args.resident, width) for batch in args.batches for width in args.token_bytes]
    if args.runtime_shapes and (33024, 8192) != (args.history, args.resident):
        cases.append((args.batches[0], 33024, 8192, args.token_bytes[0]))
    for batch, history, resident, token_bytes in cases:
        shape = Specialization(batch, history, resident, args.index_heads, token_bytes, aics, 8, args.soc)
        library = build_specialization(cache_dir=build / "lifetch_ubmem", cann=cann, shape=shape)
        if selected_library is None:
            selected_library = library.resolve()
            print(f"UBMEM_RUNTIME_BINARY path={selected_library} sha256={hashlib.sha256(library.read_bytes()).hexdigest()}",
                  flush=True)
        elif library.resolve() != selected_library:
            raise AssertionError("Runtime shape change unexpectedly rebuilt UBmem LIFetch")
        for prefetch_tokens in args.prefetch_tokens:
            check_shape(
                kvaf,
                torch,
                build,
                args.device,
                shape,
                library,
                prefetch_tokens,
                args.kvaf_server,
                args.miss_percent,
                args.warmup,
                args.iterations,
            )



if __name__ == "__main__":
    main()
