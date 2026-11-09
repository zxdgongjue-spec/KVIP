# SPDX-License-Identifier: Apache-2.0
"""Replay one real decode group without model weights; setup/reset are untimed."""

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from kvip.diagnostics.archive import file_digest, load_snapshot, source_commit
from kvip.diagnostics.oracle import DecodeOracle
from kvip.diagnostics.snapshot import (
    check_completion,
    input_tensors,
    persistent_tensors,
    post_state,
    relocate_snapshot_addresses,
    validate_capture_inputs,
)


# If completion becomes unknown, Python must not collect tensors or destroy a
# captured graph while kernels may still reference them. CLI failure retains
# the complete owner bundle until process exit, with no unsafe cleanup retry.
_RETAINED_RUNS = []


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--inspect", action="store_true", help="Verify hashes and address coverage without Torch/NPU")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--soc")
    parser.add_argument("--transport", choices=("ubmem",), default="ubmem")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, help="New result JSON; stdout also contains the result")
    args = parser.parse_args(argv)
    if args.device < 0 or args.warmup < 0 or args.iterations < 1:
        parser.error("device/warmup must be nonnegative and iterations positive")
    if not args.inspect and (args.build_dir is None or not args.soc):
        parser.error("device replay requires --build-dir and --soc")
    if args.output is not None and args.output.exists():
        parser.error("--output exists; refusing to overwrite")
    return args


def inspect_snapshot(archive):
    from kvip.diagnostics.ddr import validate_pages

    ddr = archive.manifest["ddr"]
    summary = validate_pages(archive.directory / "ddr", ddr)
    relocate_snapshot_addresses(archive, ddr["region_base"])
    validate_capture_inputs(archive)
    if archive.manifest.get("oracle", {}).get("passed") is not True:
        raise ValueError("Captured service call has no verified oracle")
    geometry = archive.manifest["geometry"]
    metadata = archive.array("pre.metadata")
    owners = metadata[4].view(np.int32)[: geometry["batch"]]
    ends = archive.array("pre.query_ends").reshape(geometry["batch"], 4)
    metadata_source = archive.manifest["metadata"]
    source = {
        name: metadata_source.get(name) for name in ("capture_mode", "dp_rank", "dp_size", "group", "layer_names")
    }
    source["request_count"] = len(metadata_source.get("requests", []))
    source["vllm_commit"] = metadata_source.get("provenance", {}).get("vllm_commit")
    source["kvip_commit"] = archive.manifest.get("provenance", {}).get("kvip_commit", "unknown")
    return dict(
        status="verified_snapshot",
        geometry=geometry,
        query_ends=ends.tolist(),
        row_owners=owners.tolist(),
        active_requests=int(np.count_nonzero(np.any(ends > 0, axis=1))),
        ddr=summary,
        source=source,
        scope=archive.manifest["scope"],
        fidelity=archive.manifest["fidelity"],
    )


def run(args, archive):
    retained = {}
    _RETAINED_RUNS.append(retained)
    result = _run(args, archive, retained)
    _RETAINED_RUNS[:] = [owner for owner in _RETAINED_RUNS if owner is not retained]
    return result


def _run(args, archive, retained):
    # Import exactly the selected KVAF build before creating any device state.
    build = args.build_dir.resolve()
    sys.path.insert(0, str(build / "python"))
    import kvaf
    import torch
    import torch_npu  # noqa: F401

    from kvip.adapters.ubmem import UbmemLease
    from kvip.cache.index_storage import IndexCacheStorage
    from kvip.diagnostics.ddr import import_pages
    from kvip.operators.lifetch_ubmem.binding import UbmemBinding
    from kvip.operators.lifetch_ubmem.forward import UbmemForward
    from kvip.operators.lifetch_ubmem.index_update import UbmemIndexUpdate
    from kvip.operators.lifetch_ubmem.specialization import Specialization, build_specialization, kvaf_cann
    from kvip.operators.lifetch_ubmem.workspace import UbmemWorkspace
    from kvip.runtime.device import query_device_resources

    if not Path(kvaf.__file__).resolve().is_relative_to(build / "python/kvaf"):
        raise RuntimeError(f"Wrong KVAF module: {kvaf.__file__}")
    if not torch.npu.is_available():
        raise RuntimeError("Replay timing requires a real NPU; no CPU/simulator timing fallback")
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    stream = torch.npu.current_stream(device)
    geometry, ddr = archive.manifest["geometry"], archive.manifest["ddr"]
    cann = kvaf_cann(build)
    available = query_device_resources(cann, args.device)["effective_aic"]
    if available < geometry["aic_count"]:
        raise ValueError("Replay device has fewer AICs than the captured geometry")
    shape = Specialization(
        geometry["batch"],
        geometry["history"],
        geometry["resident"],
        geometry["heads"],
        geometry["token_bytes"],
        geometry["aic_count"],
        geometry["requested_windows"],
        args.soc,
    )
    library = build_specialization(cache_dir=build / "lifetch_ubmem", cann=cann, shape=shape)
    runtime = kvaf.NpuRuntime(
        str(build / "bin/kvaf_ubmem_batch_get.o"),
        shape.token_bytes,
        ddr["tokens_per_block"],
        device_id=args.device,
        npu_core_count=2,
        batch_put_kernel_object_path=str(build / "bin/kvaf_ubmem_batch_put.o"),
        transport="ubmem",
        ubmem_page_mode=ddr["page_mode"],
        ubmem_numa_node=ddr["numa_node"],
    )
    retained["runtime"] = runtime
    runtime.initialize(ddr["region_bytes"], ddr["slot_bytes"])
    restored = import_pages(runtime, archive.directory / "ddr", ddr, torch_module=torch)
    retained["restored"] = restored
    if restored["access_table"] is None:
        raise ValueError("A real-history replay requires published DDR pages")
    table = runtime.access_table_view(restored["access_table"], device="npu")
    retained["table"] = table
    workspace = UbmemWorkspace(
        torch,
        batch=shape.batch,
        history=shape.history,
        resident=shape.resident,
        aics=shape.aic_count,
        windows=shape.intermediate_fetches,
        device=device,
        index_heads=shape.heads,
    )
    retained["workspace"] = workspace
    storage = IndexCacheStorage(torch, geometry["owners"], shape.history, shape.resident, device)
    retained["storage"] = storage
    output = archive.tensor("pre.output", torch, device)
    retained["output"] = output
    lease = UbmemLease(torch, runtime, output, stream)
    retained["lease"] = lease
    lease.register_table(table)
    rotating = UbmemBinding(
        torch,
        library,
        workspace.arguments(storage),
        lease,
        output,
        shape=shape,
        query_ends=workspace.inputs.query_ends.view(shape.batch, 4),
        hot_valid=storage.hot_valid,
        row_owners=workspace.inputs.row_owners,
        token_addresses=storage.addresses,
        tail_bounds=workspace.tail_bounds,
        output_stride=geometry["output_stride"],
        scratch=workspace.scratch,
        prefetch_limit=geometry["prefetch_limit"],
        window_limit=geometry["window_limit"],
    )
    retained["rotating"] = rotating
    update = UbmemIndexUpdate(
        torch,
        keys=storage.keys,
        scales=storage.scales,
        addresses=storage.addresses,
        logical_ids=storage.logical_ids,
        row_owners=workspace.inputs.row_owners,
        rotating=rotating,
        status=torch.zeros(shape.batch, dtype=torch.int32, device=device),
        query_inputs=workspace.inputs,
        outputs=torch.empty((2, shape.batch * 4, 1, 2048), dtype=torch.int32, device=device),
        plan_buffer=workspace.index_plan,
        extended_work=workspace.index_work,
    )
    retained["update"] = update
    forward = UbmemForward(rotating, update)
    selector = SimpleNamespace(rotating=rotating, storage=storage, inputs=workspace.inputs, update=update)
    seeds = []
    retained["seeds"] = seeds
    for name, target in (persistent_tensors(selector) | input_tensors(selector)).items():
        if name == "addresses":
            array = relocate_snapshot_addresses(archive, restored["new_base"])
            seed = torch.from_numpy(array.copy()).to(device)
        else:
            seed = archive.tensor("pre." + name, torch, device)
        seeds.append((target, seed))
        if target.shape != seed.shape or target.dtype != seed.dtype:
            raise ValueError(f"Replay binding differs from captured tensor: {name}")
    oracle = DecodeOracle(archive)
    expected = archive.array("post.logical_output").reshape(shape.batch * 4, 2048)
    delta = restored["new_base"] - restored["old_base"]

    def reset():
        for target, seed in seeds:
            target.copy_(seed)
        # Round-local scratch is initialized by the production forward.
        # Native lease/queue cursors are never deserialized or reset.

    def verify():
        counters = check_completion(torch, rotating, update)
        checked = oracle.verify(post_state(torch, selector), address_delta=delta, expected_topk=expected)
        # The baseline stores actual selected bytes. Compare by logical tag,
        # because transport scheduling may choose another legal physical slot.
        saved_tags = archive.array("post.selected_tags")
        saved_payload = archive.array("post.selected_payload")
        positions = {tuple(tag): i for i, tag in enumerate(checked["selected_tags"])}
        if len(positions) != len(saved_tags):
            raise AssertionError("Replay selected a different logical payload set")
        for i, tag in enumerate(saved_tags):
            position = positions.get(tuple(tag))
            if position is None or not np.array_equal(saved_payload[i], checked["selected_payload"][position]):
                raise AssertionError("Replay KV bytes differ from the captured service output")
        return counters

    reset()
    forward()
    stream.synchronize()
    verify()
    reset()
    stream.synchronize()
    graph = kvaf.AclGraph()
    retained["graph"] = graph
    lease.retain_graph(graph)
    with lease.capture_scope():
        graph.capture_begin(stream)
        forward()
        graph.capture_end()
    reset()  # Capture can execute the in-place update; always restore again.
    graph.replay(stream)
    stream.synchronize()
    verify()
    begin, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
    retained["events"] = begin, end
    samples, reads = [], []
    for step in range(args.warmup + args.iterations):
        reset()
        stream.synchronize()
        begin.record(stream)
        graph.replay(stream)
        end.record(stream)
        end.synchronize()
        elapsed = begin.elapsed_time(end) * 1000
        if not np.isfinite(elapsed) or elapsed <= 0:
            raise AssertionError("Invalid NPU device-event timing")
        counters = verify()
        if step >= args.warmup:
            samples.append(float(elapsed))
            reads.append(counters)
    graph.reset()
    lease.close()
    runtime.release_access_table(restored["access_table"])
    runtime.shutdown()
    return dict(
        schema_version=1,
        status="pass",
        transport="ubmem",
        snapshot=str(archive.directory),
        scope=archive.manifest["scope"],
        geometry=geometry,
        host=socket.gethostname(),
        device=args.device,
        device_visibility=os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
        soc=args.soc,
        kvaf_module=str(Path(kvaf.__file__).resolve()),
        library=str(library),
        library_sha256=file_digest(library),
        kvip_commit=source_commit(),
        capture_kvip_commit=archive.manifest.get("provenance", {}).get("kvip_commit", "unknown"),
        warmup=args.warmup,
        timing=dict(
            samples_us=samples,
            count=len(samples),
            p50_us=float(np.median(samples)),
            mean_us=float(np.mean(samples)),
            min_us=min(samples),
            max_us=max(samples),
        ),
        reads_samples=reads,
        address_relocation=restored["new_base"] - restored["old_base"],
        fidelity=archive.manifest["fidelity"],
        exclusions="Model, input preparation/import/packing, DDR Put, state restoration and correctness readbacks",
        interpretation="Isolated group graph latency; not DP service latency or TPOT",
    )


def main():
    args = arguments()
    archive = load_snapshot(args.snapshot)
    inspected = inspect_snapshot(archive)
    result = inspected if args.inspect else run(args, archive)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            output.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
