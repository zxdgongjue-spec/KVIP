# SPDX-License-Identifier: Apache-2.0
"""Startup binding of service storage to the original LIFetch launcher."""

import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from kvip.runtime.build import build_libraries
from kvip.operators.lifetch.binding import LIFetchBinding
from kvip.adapters.vllm.selector import bind_group_selector, group_scratch, index_update_arguments
from kvip.operators.lifetch.index_update import DirectIndexUpdate
from kvip.adapters.kvaf import KvafGetLease, KvafQueueView, LeasedRuntime, UbmemLease, UbmemLeasedRuntime
from kvip.operators.lifetch_ubmem.binding import UbmemBinding
from kvip.operators.lifetch.index_update import UbmemIndexUpdate
from kvip.operators.lifetch_ubmem.build import UbmemGeometry, build_ubmem_library, kvaf_cann
from kvip.operators.lifetch.workspace import LIFetchWorkspace, UbmemWorkspace
from kvip.runtime.device import query_device_resources


def bind_selectors(torch, policy, config, *, batch=None):
    """Bind one graph geometry, sharing the backing KV and Index storage."""
    with torch.inference_mode():
        setup, workspace = policy.fused_setup, policy.fused_workspace
        if setup is None or workspace is None or policy.writer is None:
            raise RuntimeError("LIFetch requires profiled workspace and bound service caches")
        if batch is not None and batch != setup.batch:
            setup = SimpleNamespace(**{**vars(setup), "batch": batch})
            shape_args = dict(
                batch=batch, history=workspace.block_history, aics=setup.aics,
                windows=setup.windows, device=policy.pool.device,
                index_heads=workspace.inputs.index_heads, compute=workspace.compute,
            )
            if config.transport == "urma":
                workspace = LIFetchWorkspace(torch, **shape_args)
                workspace.allocate_rotation(torch, batch=batch, history=workspace.block_history,
                                            resident=policy.history_capacity, aics=setup.aics, windows=setup.windows)
            else:
                workspace = UbmemWorkspace(torch, resident=policy.history_capacity, **shape_args)
                setup.shapes = {width: replace(shape, batch=batch) for width, shape in setup.shapes.items()}
        if config.transport == "ubmem":
            selectors = _bind_ubmem_selectors(torch, policy, config, setup, workspace)
        elif config.transport == "urma":
            selectors = _bind_selectors(torch, policy, config, setup, workspace)
        else:
            raise ValueError(f"Unsupported LIFetch transport: {config.transport}")
        if getattr(config, "lifetch_snapshot_dir", ""):
            from kvip.adapters.vllm.snapshot import LIFetchSnapshot

            group = config.lifetch_snapshot_group
            if group not in selectors:
                raise ValueError(f"LIFetch snapshot group {group} is out of range")
            selectors[group].snapshot = LIFetchSnapshot(selectors[group], config)
        if batch is None:
            policy.indexer_selectors = selectors
        return selectors


def _bind_selectors(torch, policy, config, setup, workspace):
    writer = policy.writer
    launch_type = LIFetchBinding
    libraries = {}
    selectors = {}
    for group, layer in enumerate(writer.groups):
        storage = policy.fused_index_storage[group]
        token_bytes = layer.geometry[0]
        if token_bytes not in libraries:
            libraries[token_bytes] = build_libraries(
                cache_dir=Path(config.build_dir) / "lifetch",
                cann=setup.cann,
                headers=setup.headers,
                soc=setup.soc,
            )
        (library,) = libraries[token_bytes]
        runtime = writer.runtimes[layer.geometry]
        if not isinstance(runtime, LeasedRuntime):
            lease = KvafGetLease(
                torch, runtime, library, policy.bytes_outputs[group], torch.npu.current_stream(policy.pool.device)
            )
            runtime = LeasedRuntime(lease)
            # Replace immediately so adapter shutdown owns partial startup too.
            writer.runtimes[layer.geometry] = runtime
        owner = KvafQueueView(
            runtime.lease, policy.bytes_outputs[group], policy.addresses.view(-1), policy.valid.view(-1)
        )
        scratch = group_scratch(policy, workspace, setup.batch, group)
        rotating = launch_type(
            torch,
            library,
            workspace.arguments(storage),
            owner,
            batch=setup.batch,
            history=storage.history,
            resident=policy.history_capacity,
            intermediate_fetches=setup.windows,
            prefetch_limit=config.prefetch_tokens,
            prefetch_rank_start=1024,
            window_limit=max(1, config.prefetch_tokens // setup.windows),
            aic_count=setup.aics,
            token_bytes=token_bytes,
            query_ends=workspace.inputs.query_ends.view(setup.batch, 4),
            hot_valid=storage.hot_valid,
            row_owners=workspace.inputs.row_owners,
            token_addresses=storage.addresses,
            tail_bounds=workspace.tail_bounds,
            output_stride=policy.slot_capacity,
            scratch=scratch,
        )
        update = DirectIndexUpdate(
            torch,
            **index_update_arguments(policy, workspace, storage, rotating, group),
        )
        selectors[group] = bind_group_selector(torch, policy, workspace, setup, group, storage, rotating, update)
    logging.getLogger("vllm").info(
        "KVAF decode backend=li_fetch groups=%s batch_capacity=%s history_capacity=%s hot_tokens=%s "
        "requested_windows=%s; shared Runtime queues and original LRU candidates",
        len(selectors),
        setup.batch,
        policy.fused_index_storage[0].history,
        policy.history_capacity,
        setup.windows,
    )
    return selectors


def prepare_workspace(torch, policy, config, device_id, soc, index_heads=64):
    if config.transport != "ubmem" or not policy.fused_index_storage or policy.fused_workspace is not None:
        raise RuntimeError("UBmem workspace requires initialized Index storage and exactly one preparation")
    cann = kvaf_cann(config.build_dir)
    aics = query_device_resources(cann, device_id)["effective_aic"]
    batch, history = policy.pool.max_requests - 1, policy.fused_index_storage[0].history
    shapes, libraries = {}, {}
    for width in sorted(set(policy.group_sizes)):
        token_bytes = 656 * width
        shape = UbmemGeometry(
            batch,
            history,
            policy.history_capacity,
            index_heads,
            token_bytes,
            aics,
            config.intermediate_fetches,
            soc,
        )
        shapes[token_bytes] = shape
        libraries[token_bytes] = build_ubmem_library(
            cache_dir=Path(config.build_dir) / "lifetch_ubmem",
            cann=cann,
            shape=shape,
        )
    policy.fused_setup = SimpleNamespace(
        cann=cann,
        soc=soc,
        aics=aics,
        batch=batch,
        windows=config.intermediate_fetches,
        workers=config.npu_core_count,
        shapes=shapes,
        libraries=libraries,
    )
    policy.fused_workspace = UbmemWorkspace(
        torch,
        batch=batch,
        history=history,
        resident=policy.history_capacity,
        aics=aics,
        windows=config.intermediate_fetches,
        device=policy.pool.device,
        index_heads=index_heads,
    )


def _bind_ubmem_selectors(torch, policy, config, setup, workspace):
    writer = policy.writer
    selectors = {}
    for group, layer in enumerate(writer.groups):
        storage, output = policy.fused_index_storage[group], policy.bytes_outputs[group]
        token_bytes = layer.geometry[0]
        runtime = writer.runtimes[layer.geometry]
        if not isinstance(runtime, UbmemLeasedRuntime):
            lease = UbmemLease(torch, runtime, output, torch.npu.current_stream(policy.pool.device))
            runtime = UbmemLeasedRuntime(lease)
            writer.runtimes[layer.geometry] = runtime
        scratch = group_scratch(policy, workspace, setup.batch, group)
        rotating = UbmemBinding(
            torch,
            setup.libraries[token_bytes],
            workspace.arguments(storage),
            runtime.lease,
            output,
            shape=setup.shapes[token_bytes],
            query_ends=workspace.inputs.query_ends.view(setup.batch, 4),
            hot_valid=storage.hot_valid,
            row_owners=workspace.inputs.row_owners,
            token_addresses=storage.addresses,
            tail_bounds=workspace.tail_bounds,
            output_stride=policy.slot_capacity,
            prefetch_limit=config.prefetch_tokens,
            window_limit=max(1, config.prefetch_tokens // setup.windows),
            scratch=scratch,
        )
        update = UbmemIndexUpdate(
            torch,
            **index_update_arguments(policy, workspace, storage, rotating, group),
        )
        selectors[group] = bind_group_selector(
            torch, policy, workspace, setup, group, storage, rotating, update, ubmem=True
        )
    logging.getLogger("vllm").info(
        "KVAF transport=ubmem decode_backend=li_fetch groups=%s batch_capacity=%s history_capacity=%s "
        "hot_tokens=%s requested_windows=%s device=%s; CPU-service Host DDR mapping",
        len(selectors),
        setup.batch,
        policy.fused_index_storage[0].history,
        policy.history_capacity,
        setup.windows,
        policy.pool.device,
    )
    return selectors
