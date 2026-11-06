#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fixed-snapshot Q4 LI/TopK/Fetch/Index-update graph benchmark, one transport per process.

Reuse the controlled LI fixture and production bindings. Setup, real KVAF Put,
snapshot restoration and correctness readbacks are outside the device events.
This measures one operator/group, not model TPOT or a growing-context decode.
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "python"), str(Path(__file__).parent / "support")]
from benchmark_reads import fixed_read_workload  # noqa: E402 — standalone test bootstrap


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transport", required=True, choices=("urma", "ubmem"))
    p.add_argument("--build-dir", type=Path, required=True, help="Exact KVAF build with bin/ and python/kvaf/")
    p.add_argument("--kvaf-server", help="URMA server address; not used by UBmem")
    p.add_argument("--soc", required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--history", type=int, default=131072, help="Allocated/scored Index capacity, aligned to 256")
    p.add_argument("--sequence-length", type=int, help="Valid shared Q4 prefix; defaults to --history")
    p.add_argument("--resident", type=int, default=8192)
    p.add_argument("--index-heads", type=int, choices=(32, 64), default=32)
    p.add_argument("--token-bytes", type=int, choices=(656, 2624), required=True)
    p.add_argument("--intermediate-fetches", type=int, default=8)
    p.add_argument("--miss-percent", type=float, default=10)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--output", type=Path, required=True, help="New result JSON; generated fixtures are temporary")
    a = p.parse_args(argv)
    a.sequence_length = a.history if a.sequence_length is None else a.sequence_length
    if not 1 <= a.batch <= 16 or not 0 <= a.device:
        p.error("batch must be 1..16 and device nonnegative")
    if not 2048 <= a.resident < a.sequence_length <= a.history <= 210944:
        p.error("require 2048 <= resident < sequence-length <= history <= 210944")
    if any(n % 256 for n in (a.history, a.sequence_length, a.resident)):
        p.error("history, sequence-length and resident must align to 256")
    if not 0 <= a.miss_percent <= 100 or not 1 <= a.intermediate_fetches <= 32:
        p.error("invalid miss percent or intermediate-fetches")
    if a.warmup < 0 or a.iterations < 1:
        p.error("warmup must be nonnegative and iterations positive")
    if (a.transport == "urma") != bool(a.kvaf_server):
        p.error("--kvaf-server is required only for URMA")
    a.build_dir, a.output = a.build_dir.resolve(), a.output.resolve()
    if a.output.exists():
        p.error("--output already exists; use a new result path")
    return a


def payload(owner, tokens, token_bytes):
    """Exact token/owner tags and nonconstant bytes; bounded uint16 temporaries."""
    tokens = np.asarray(tokens, dtype=np.uint64).reshape(-1)
    columns = np.arange(token_bytes, dtype=np.uint16)
    residues = (tokens % 251).astype(np.uint16)
    values = ((residues[:, None] + columns[None, :] + owner * 17) % 251).astype(np.uint8)
    values[:, :8] = (tokens + np.uint64(owner << 32)).astype("<u8").view(np.uint8).reshape(-1, 8)
    return values


def statistics(samples):
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("device event times must be finite, positive microseconds")
    return dict(
        samples_us=values.tolist(),
        count=int(values.size),
        min_us=float(values.min()),
        mean_us=float(values.mean()),
        p50_us=float(np.median(values)),
        p90_us=float(np.percentile(values, 90)),
        max_us=float(values.max()),
    )


def prepare_fixture(args, directory):
    from real_fetch_inputs import prepare_full_chain_fixture
    from kvip.cache.lru import SlotLRU

    info = prepare_full_chain_fixture(
        directory, args.sequence_length, args.resident, args.miss_percent, index_heads=args.index_heads
    )
    order = np.load(directory / "physical_to_logical.npy")
    scores = np.load(directory / "logical_scores.npy")[:, order]
    expected = []
    for row in scores:
        ids = np.flatnonzero(row >= np.partition(row, -2048)[-2048]).astype(np.int32)
        if len(ids) != 2048:
            raise AssertionError("controlled fixture needs an unambiguous TopK boundary")
        expected.append(ids)
    expected = np.stack(expected)
    # Same LRU preparation as test_lifetch.py: move the selected hot prefix
    # to MRU, then take 2048 existing oldest slots. No operator chooses victims.
    lru = SlotLRU()
    lru.resize(args.resident, args.resident)
    lru.commit(expected[expected < args.resident])
    candidates = lru.slots[0].copy()
    reserved = np.full(args.resident, -1, dtype=np.int32)
    reserved[candidates] = np.arange(2048, dtype=np.int32)
    np.save(directory / "expected_ids.npy", expected)
    np.save(directory / "lru_candidates.npy", candidates)
    digest = hashlib.sha256()
    for name in ("arg0.bin", "arg1.bin", "arg2.bin", "arg3.bin", "arg4.bin", "expected_ids.npy", "lru_candidates.npy"):
        digest.update(name.encode())
        digest.update((directory / name).read_bytes())
    return info, expected, candidates, reserved, digest.hexdigest()


def progress(message):
    print(f"[lifetch-benchmark] {message}", file=sys.stderr, flush=True)


def run(args, work):
    # Both runtimes export a module named kvaf. Select exactly one before import;
    # do not infer it from dependencies.json or switch SDKs within this process.
    sys.path.insert(0, str(args.build_dir / "python"))
    import kvaf
    import torch
    import torch_npu  # noqa: F401

    from kvip.cache.index_storage import IndexCacheStorage
    from kvip.operators.lifetch.geometry import final_aivs, li_geometry
    from kvip.runtime.device import query_device_resources
    from prefetch_audit import actual_reads

    if not Path(kvaf.__file__).resolve().is_relative_to(args.build_dir / "python/kvaf"):
        raise RuntimeError(f"Wrong KVAF module: {kvaf.__file__}")
    if not torch.npu.is_available():
        raise RuntimeError("A real NPU is required; no simulated performance results")
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    stream = torch.npu.current_stream(device)
    progress("prepare temporary shared fixture")
    fixture_info, expected, candidates, reserved, fixture_sha = prepare_fixture(args, work / "fixture")
    b, h, r, width = args.batch, args.history, args.resident, args.token_bytes
    if args.transport == "ubmem":
        from kvip.operators.lifetch_ubmem.specialization import Specialization, build_specialization, kvaf_cann
        from kvip.operators.lifetch_ubmem.workspace import UbmemWorkspace

        cann = kvaf_cann(args.build_dir)
    else:
        from kvip.runtime.build import build_libraries, kvaf_build_paths
        from kvip.operators.lifetch.workspace import LIFetchWorkspace

        cann, headers = kvaf_build_paths(args.build_dir)
    aics = query_device_resources(cann, args.device)["effective_aic"]
    if b > aics:
        raise ValueError("batch exceeds concurrently resident LI core budget")
    geometry = li_geometry(b, "auto", args.intermediate_fetches, aics)
    final_workers = final_aivs(b, "auto", aics)
    workers = (b * (final_workers + 1) + 1) // 2 if args.transport == "urma" else 2
    progress(
        f"build/load {args.transport}: B={b} H={h} valid={args.sequence_length} width={width} heads={args.index_heads}"
    )
    if args.transport == "ubmem":
        shape = Specialization(b, h, r, args.index_heads, width, aics, args.intermediate_fetches, args.soc)
        library = build_specialization(cache_dir=args.build_dir / "lifetch_ubmem", cann=cann, shape=shape)
        workspace = UbmemWorkspace(
            torch,
            batch=b,
            history=h,
            resident=r,
            aics=aics,
            windows=args.intermediate_fetches,
            device=device,
            index_heads=args.index_heads,
        )
        kernel_names = ("kvaf_ubmem_batch_get.o", "kvaf_ubmem_batch_put.o")
    else:
        library, update_library = build_libraries(
            cache_dir=args.build_dir / "lifetch", cann=cann, headers=headers, soc=args.soc
        )
        workspace = LIFetchWorkspace(
            torch,
            batch=b,
            history=h,
            aics=aics,
            windows=args.intermediate_fetches,
            device=device,
            index_heads=args.index_heads,
        )
        kernel_names = ("kvaf_execute_batch_get_kernels.o", "kvaf_execute_batch_put_kernels.o")
    get_kernel, put_kernel = [args.build_dir / "bin" / name for name in kernel_names]
    for path in (get_kernel, put_kernel):
        if not path.is_file():
            raise FileNotFoundError(path)
    transport_options = {"transport": "ubmem"} if args.transport == "ubmem" else {"jettys_per_core": 2}
    runtime = kvaf.NpuRuntime(
        str(get_kernel),
        width,
        128,
        device_id=args.device,
        npu_core_count=workers,
        batch_put_kernel_object_path=str(put_kernel),
        **transport_options,
    )
    if args.transport == "urma":
        runtime.link_to(args.kvaf_server)
    blocks = h // 128
    block_bytes = 128 * width
    # Identical padded block geometry on both transports, including 656 bytes.
    slot_bytes = (block_bytes + 4095) // 4096 * 4096
    keys = list(range(b * blocks))
    runtime.initialize((len(keys) + 2) * slot_bytes, slot_bytes)
    progress(f"real KVAF Put: {b * h * width} bytes; excluded from timing")
    for owner in range(1, b + 1):
        for first in range(0, blocks, 128):
            count = min(128, blocks - first)
            source = torch.from_numpy(payload(owner, np.arange(first * 128, (first + count) * 128), width)).to(device)
            chunk_keys = keys[(owner - 1) * blocks + first : (owner - 1) * blocks + first + count]
            runtime.batch_put(chunk_keys, source, [i * block_bytes for i in range(count)], [block_bytes] * count)
            stream.synchronize()
    table = runtime.prepare_access_table(keys)
    device_table = runtime.access_table_view(table, device="npu")
    if not np.all(device_table.valid.cpu().numpy()):
        raise AssertionError("KVAF table contains invalid blocks")
    storage = IndexCacheStorage(torch, b + 1, h, r, device)
    f = work / "fixture"
    key_seed = torch.from_numpy(np.fromfile(f / "arg1.bin", np.uint8).reshape(args.sequence_length, 128)).to(device)
    scale_seed = torch.from_numpy(np.fromfile(f / "arg4.bin", np.float32)).to(device)
    storage.keys[1:, : args.sequence_length].copy_(key_seed)
    storage.scales.zero_()
    storage.scales[1:, : args.sequence_length].copy_(scale_seed)
    offsets = torch.arange(128, dtype=torch.int64, device=device) * width
    storage.addresses[1:].copy_((device_table.addresses.view(torch.int64).view(b, blocks, 1) + offsets).reshape(b, h))
    storage.hot_valid[1:].fill_(1)
    stride = r + 8192
    initial_output = torch.full((b + 1, stride, width), 0xA5, dtype=torch.uint8, device=device)
    for owner in range(1, b + 1):
        initial_output[owner, :r].copy_(torch.from_numpy(payload(owner, np.arange(r), width)).to(device))
    output = initial_output.clone()
    output_rows = output.view(-1, width)
    workspace.inputs.prepare([4] * b, query_owners=list(range(1, b + 1)))
    workspace.inputs.query.view(torch.uint8).view(b, 4, args.index_heads, 128).copy_(
        torch.from_numpy(np.fromfile(f / "arg0.bin", np.uint8).reshape(4, args.index_heads, 128)).to(device)
    )
    # Match the existing URMA fixture's exact BF16 truncation of binary weights.
    weight_bits = (np.fromfile(f / "arg2.bin", np.float32).view(np.uint32) >> 16).astype(np.uint16)
    workspace.inputs.weights.view(torch.int16).view(b, 4, args.index_heads).copy_(
        torch.from_numpy(weight_bits.view(np.int16).reshape(4, args.index_heads)).to(device)
    )
    workspace.inputs.scale.view(b, 4, args.index_heads).copy_(
        torch.from_numpy(np.fromfile(f / "arg3.bin", np.float32).reshape(4, args.index_heads)).to(device)
    )
    workspace.inputs.query_ends.fill_(args.sequence_length)
    common = dict(
        query_ends=workspace.inputs.query_ends.view(b, 4),
        hot_valid=storage.hot_valid,
        row_owners=workspace.inputs.row_owners,
        token_addresses=storage.addresses,
        output_stride=stride,
    )
    if args.transport == "ubmem":
        from kvip.adapters.ubmem import UbmemLease
        from kvip.operators.lifetch_ubmem.binding import UbmemBinding
        from kvip.operators.lifetch_ubmem.forward import UbmemForward
        from kvip.operators.lifetch_ubmem.index_update import UbmemIndexUpdate

        lease = UbmemLease(torch, runtime, output_rows, stream)
        lease.register_table(device_table)
        rotating = UbmemBinding(
            torch,
            library,
            workspace.arguments(storage),
            lease,
            output_rows,
            shape=shape,
            scratch=workspace.scratch,
            **common,
        )
        update_type, forward_type, update_args = UbmemIndexUpdate, UbmemForward, ()
    else:
        from kvip.adapters.kvaf import KvafGetLease, KvafQueueView
        from kvip.operators.lifetch.binding import LIFetchBinding
        from kvip.operators.lifetch.forward import LIFetchForward
        from kvip.operators.lifetch.index_update import DirectIndexUpdate

        lease = KvafGetLease(torch, runtime, library, output_rows, stream)
        owner_view = KvafQueueView(lease, output_rows, device_table.addresses, device_table.valid)
        rotating = LIFetchBinding(
            torch,
            library,
            workspace.arguments(storage),
            owner_view,
            batch=b,
            history=h,
            resident=r,
            aic_count=aics,
            intermediate_fetches=args.intermediate_fetches,
            token_bytes=width,
            **common,
        )
        update_type, forward_type, update_args = DirectIndexUpdate, LIFetchForward, (update_library,)
    update = update_type(
        torch,
        *update_args,
        keys=storage.keys,
        scales=storage.scales,
        addresses=storage.addresses,
        logical_ids=storage.logical_ids,
        row_owners=workspace.inputs.row_owners,
        rotating=rotating,
        status=torch.zeros(b, dtype=torch.int32, device=device),
        query_inputs=workspace.inputs,
        outputs=torch.empty((2, b * 4, 1, 2048), dtype=torch.int32, device=device),
    )
    forward = forward_type(rotating, update)
    candidates_device = torch.from_numpy(candidates).to(device)
    reserved_device = torch.from_numpy(reserved).to(device)
    snapshots = [
        (tensor, tensor.clone()) for tensor in (storage.keys, storage.scales, storage.addresses, storage.logical_ids)
    ]
    initial_addresses = storage.addresses.cpu().numpy()
    initial_keys = storage.keys[1].cpu().numpy()
    initial_scales = storage.scales[1].cpu().numpy()

    def reset():
        for tensor, seed in snapshots:
            tensor.copy_(seed)
        storage.hot_valid.zero_()
        storage.hot_valid[1:].fill_(1)
        rotating.lru_slots.copy_(candidates_device)
        rotating.reserved_hot.copy_(reserved_device)
        output.copy_(initial_output)
        # Per-round scratch resets are inside forward. Never reset URMA queue cursors.

    def check(*, full=False):
        rows = rotating.rows.cpu().numpy()
        if np.any(rows[:, [11, 211, 248]]) or np.any(rotating.row_status.cpu().numpy()[:, 0]):
            raise AssertionError("fused transport/resolver completion failed")
        if np.any(rotating.final_posts.cpu().numpy()[:, :, 1:4]):
            raise AssertionError("final fetch has pending/error/deferred work")
        if args.transport == "ubmem":
            counters = rotating.ubmem_status.cpu().numpy()
            if np.any(counters[:, 0]) or not np.array_equal(counters[:, 1:4:2], counters[:, 2:5:2]):
                raise AssertionError("UBmem submitted/completed copies differ")
        else:
            for tensor in (rotating.prefetch_status, rotating.final_status):
                words = tensor.cpu().view(torch.uint64).numpy()
                if np.any(words[:, 0] != 5428598170857194575) or np.any(words[:, 2]):
                    raise AssertionError("URMA completion status failed")
        if np.any(update.status.cpu().numpy()):
            raise AssertionError("Index update rejected the completed transfer")
        logical = update.logical_output.cpu().numpy().reshape(b, 4, 2048)
        slots = update.slot_output.cpu().numpy().reshape(b, 4, 2048)
        if not np.array_equal(np.sort(logical, axis=-1), np.broadcast_to(expected, logical.shape)):
            raise AssertionError("logical TopK differs from the independent score oracle")
        if np.any(slots < 0) or np.any(slots >= stride):
            raise AssertionError("selected slot is outside the registered output")
        if full:
            ids = storage.logical_ids.cpu().numpy()
            addresses = storage.addresses.cpu().numpy()
            scales = storage.scales.cpu().numpy()
            valid = storage.hot_valid.cpu().numpy()
            for owner in range(1, b + 1):
                if not np.array_equal(np.sort(ids[owner]), np.arange(h)):
                    raise AssertionError("Index IDs are not a permutation")
                if not np.array_equal(addresses[owner], initial_addresses[owner, ids[owner]]):
                    raise AssertionError("Index addresses detached from IDs")
                if not np.array_equal(scales[owner], initial_scales[ids[owner]]):
                    raise AssertionError("Index scales detached from IDs")
                if not np.array_equal(storage.keys[owner].cpu().numpy(), initial_keys[ids[owner]]):
                    raise AssertionError("Index keys detached from IDs")
                selected = torch.from_numpy(slots[owner - 1].reshape(-1).astype(np.int64)).to(device)
                actual = torch.index_select(output[owner], 0, selected).cpu().numpy()
                if not np.array_equal(actual, payload(owner, logical[owner - 1].reshape(-1), width)):
                    raise AssertionError("selected payload differs from real KVAF Put")
                hot_tags = output[owner, :r, :8].contiguous().cpu().numpy().view("<u8").reshape(-1)
                wanted = ids[owner, :r].astype(np.uint64) + np.uint64(owner << 32)
                if np.any((valid[owner] != 0) & (hot_tags != wanted)):
                    raise AssertionError("valid hot payload differs from Index ownership")
        return actual_reads(rotating.mapping.cpu().numpy(), workspace.topk.cpu().numpy(), rows, r)

    progress("verify full forward, then capture its exact production submission")
    reset()
    forward()
    stream.synchronize()
    check(full=True)
    reset()
    stream.synchronize()
    graph = kvaf.AclGraph()
    with lease.capture_scope():
        graph.capture_begin(stream)
        forward()
        graph.capture_end()
    if args.transport == "ubmem":
        lease.retain_graph(graph)
    reset()
    graph.replay(stream)
    stream.synchronize()
    baseline_reads = check(full=True)
    baseline_workload = fixed_read_workload(baseline_reads)
    samples, read_samples = [], []
    begin, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
    progress(f"time graph replay: warmup={args.warmup} iterations={args.iterations}; reset/check excluded")
    for step in range(args.warmup + args.iterations):
        reset()
        stream.synchronize()
        begin.record(stream)
        graph.replay(stream)
        end.record(stream)
        end.synchronize()
        elapsed = begin.elapsed_time(end) * 1000
        reads = check(full=step == args.warmup + args.iterations - 1)
        workload = fixed_read_workload(reads)
        if workload != baseline_workload:
            raise AssertionError(f"fixed-snapshot Fetch workload changed: {workload} != {baseline_workload}")
        if step >= args.warmup:
            samples.append(elapsed)
            read_samples.append(reads)
    graph.reset()
    lease.close()
    runtime.release_access_table(table)
    runtime.batch_free(keys)
    runtime.shutdown()
    contract = dict(
        batch=b,
        queries_per_request=4,
        mtp_speculative_tokens=3,
        history_capacity=h,
        valid_sequence_length=args.sequence_length,
        resident=r,
        token_bytes=width,
        index_heads=args.index_heads,
        index_head_dim=128,
        topk=2048,
        output_stride=stride,
        requested_windows=args.intermediate_fetches,
        prefetch_limit=2048,
        window_limit=480,
        lru_order="existing_SlotLRU_after_oracle_hot_commit",
        fixture_sha256=fixture_sha,
        requested_miss_percent=args.miss_percent,
        scope="LI+TopK+Fetch+IndexUpdate+TND_outputs",
        visibility="Q4_same_valid_prefix_no_ABC_growth",
        snapshot="restore_before_every_replay",
    )
    return dict(
        schema_version=1,
        status="pass",
        transport=args.transport,
        device=args.device,
        soc=args.soc,
        host=socket.gethostname(),
        device_visibility=os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
        contract=contract,
        comparison_key=hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
        timing=statistics(samples),
        warmup=args.warmup,
        geometry=geometry,
        final_workers=final_workers,
        fixture=fixture_info,
        final_cold_per_query=[int(np.count_nonzero(ids >= r)) for ids in expected],
        distinct_final_cold_per_request=int(np.unique(expected[expected >= r]).size),
        reads_per_replay=baseline_reads,
        reads_samples=read_samples,
        kvaf_build=str(args.build_dir),
        kvaf_module=str(Path(kvaf.__file__).resolve()),
        cann=str(cann),
        runtime_workers=workers,
        kvaf_server=args.kvaf_server,
        ddr_slot_bytes=slot_bytes,
        library=str(library),
        library_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest(),
        kvip_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        note="Synthetic fixed-load operator latency in microseconds; not end-to-end TPOT.",
    )


def main(argv=None):
    args = arguments(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="kvip-lifetch-benchmark-") as work:
            result = run(args, Path(work))
    except BaseException as error:
        # Retain traceback locals (runtime, lease, graph, tensors) and bypass
        # destructors on failure: stream completion alone cannot prove URMA CQ
        # drain. Process exit quarantines resources; never reuse failed queues.
        traceback.print_exc()
        result = dict(status="failed", transport=args.transport, error=f"{type(error).__name__}: {error}")
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        sys.stderr.flush()
        os._exit(70)
    result["wall_seconds_including_setup"] = time.monotonic() - started
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(dict(status="pass", transport=args.transport, result=str(args.output), timing=result["timing"])),
        flush=True,
    )


if __name__ == "__main__":
    main()
