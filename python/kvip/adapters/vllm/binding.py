# SPDX-License-Identifier: Apache-2.0
"""Startup binding of service storage to the original LIFetch launcher."""

import logging
from pathlib import Path

from kvip.runtime.build import build_libraries
from kvip.operators.lifetch.binding import LIFetchBinding
from kvip.adapters.vllm.selector import bind_group_selector, group_scratch
from kvip.operators.lifetch.index_update import DirectIndexUpdate
from kvip.adapters.kvaf import KvafGetLease, KvafQueueView, LeasedRuntime


def bind_selectors(torch, policy, config):
    # Profiling allocates inference tensors; startup must initialize their
    # shared views under the same mode, without cloning graph-bound storage.
    with torch.inference_mode():
        transport = getattr(config, "transport", "urma")
        if transport == "ubmem":
            from kvip.adapters.vllm.ubmem_binding import _bind_selectors as bind_ubmem

            bind_ubmem(torch, policy, config)
        elif transport == "urma":
            _bind_selectors(torch, policy, config)
        else:
            raise ValueError(f"Unsupported LIFetch transport: {transport}")


def _bind_selectors(torch, policy, config):
    setup, workspace, writer = policy.fused_setup, policy.fused_workspace, policy.writer
    if setup is None or workspace is None or writer is None:
        raise RuntimeError("LIFetch requires profiled workspace and bound service caches")
    if policy.indexer_selectors:
        raise RuntimeError("LIFetch selectors are already bound")
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
        library, update_library = libraries[token_bytes]
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
            update_library,
            keys=storage.keys,
            scales=storage.scales,
            addresses=storage.addresses,
            logical_ids=storage.logical_ids,
            row_owners=workspace.inputs.row_owners,
            rotating=rotating,
            status=policy.fused_errors[group],
            query_inputs=workspace.inputs,
            outputs=policy.fused_outputs[group],
        )
        selectors[group] = bind_group_selector(torch, policy, workspace, setup, group, storage, rotating, update)
    policy.indexer_selectors = selectors
    logging.getLogger("vllm").info(
        "KVAF decode backend=li_fetch groups=%s batch_capacity=%s history_capacity=%s hot_tokens=%s "
        "requested_windows=%s; shared Runtime queues and original LRU candidates",
        len(selectors),
        setup.batch,
        policy.fused_index_storage[0].history,
        policy.history_capacity,
        setup.windows,
    )
