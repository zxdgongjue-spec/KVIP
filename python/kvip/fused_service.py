# SPDX-License-Identifier: Apache-2.0
"""Startup binding of service storage to the original LIFetch launcher."""

import logging
from pathlib import Path

from .fused_build import build_libraries, load_original
from .fused_selector import FusedIndexSelector
from .index_update import DirectIndexUpdate
from .transport import FusedGetLease, FusedQueueView, LeasedRuntime


def bind_selectors(torch, policy, config):
    # Profiling allocates inference tensors; startup must initialize their
    # shared views under the same mode, without cloning graph-bound storage.
    with torch.inference_mode():
        _bind_selectors(torch, policy, config)


def _bind_selectors(torch, policy, config):
    setup, workspace, writer = policy.fused_setup, policy.fused_workspace, policy.writer
    if setup is None or workspace is None or writer is None:
        raise RuntimeError("LIFetch requires profiled workspace and bound service caches")
    if policy.indexer_selectors:
        raise RuntimeError("LIFetch selectors are already bound")
    launch_type = load_original("launch").RotatingLaunch
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
            lease = FusedGetLease(
                torch, runtime, library, policy.bytes_outputs[group], torch.npu.current_stream(policy.pool.device)
            )
            runtime = LeasedRuntime(lease)
            # Replace immediately so adapter shutdown owns partial startup too.
            writer.runtimes[layer.geometry] = runtime
        owner = FusedQueueView(
            runtime.lease, policy.bytes_outputs[group], policy.addresses.view(-1), policy.valid.view(-1)
        )
        scratch = dict(workspace.scratch)
        scratch["rows"] = policy.fused_fetch_rows[group]
        # Bind existing candidates directly: no extra selection or per-layer copy.
        scratch["lru_slots"] = policy.spare_slots[group, : setup.batch]
        scratch["reserved_hot"] = policy.fused_reserved_hot[group, : setup.batch]
        rotating = launch_type(
            torch,
            library,
            workspace.arguments(storage),
            owner,
            batch=setup.batch,
            history=storage.history,
            resident=policy.history_capacity,
            intermediate_fetches=setup.windows,
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
        selector = FusedIndexSelector(
            torch,
            inputs=workspace.inputs,
            rotating=rotating,
            update=update,
            logical_ids=storage.logical_ids,
            row_owners=workspace.inputs.row_owners,
            prefix_workspace=workspace.prefix_reset,
            storage=storage,
        )
        selector.tail_source = policy.abc_descriptors[group, : setup.batch, 2:4]
        selectors[group] = selector
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
