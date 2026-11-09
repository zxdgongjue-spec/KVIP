# SPDX-License-Identifier: Apache-2.0
"""Bind vLLM storage directly to the independent same-host UBmem pipeline."""

import logging
from pathlib import Path
from types import SimpleNamespace

from kvip.adapters.ubmem import UbmemLease, UbmemLeasedRuntime
from kvip.adapters.vllm.selector import bind_group_selector, group_scratch, index_update_arguments
from kvip.operators.lifetch_ubmem.binding import UbmemBinding
from kvip.operators.lifetch.index_update import UbmemIndexUpdate
from kvip.operators.lifetch_ubmem.build import UbmemGeometry, build_ubmem_library, kvaf_cann
from kvip.operators.lifetch.workspace import UbmemWorkspace
from kvip.runtime.device import query_device_resources


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


def bind_selectors(torch, policy, config):
    with torch.inference_mode():
        _bind_selectors(torch, policy, config)


def _bind_selectors(torch, policy, config):
    setup, workspace, writer = policy.fused_setup, policy.fused_workspace, policy.writer
    if config.transport != "ubmem" or setup is None or workspace is None or writer is None:
        raise RuntimeError("UBmem binding requires a profiled workspace and service caches")
    if policy.indexer_selectors:
        raise RuntimeError("UBmem selectors are already bound")
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
            plan_buffer=workspace.index_plan,
            extended_work=workspace.index_work,
        )
        selectors[group] = bind_group_selector(
            torch, policy, workspace, setup, group, storage, rotating, update, ubmem=True
        )
    policy.indexer_selectors = selectors
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
