# SPDX-License-Identifier: Apache-2.0
"""Indexer selector using service-owned LIFetch inputs and Index updates.

The caller binds cache rows, LRU candidates and query positions before forward.
This object neither allocates a Runtime nor selects replacement candidates.
"""

from copy import copy

from kvip.operators.lifetch.forward import LIFetchForward


def group_scratch(policy, workspace, batch, group):
    """Bind shared scratch and the existing per-group candidate buffers."""
    scratch = dict(workspace.scratch)
    scratch["rows"] = policy.fused_fetch_rows[group]
    scratch["lru_slots"] = policy.spare_slots[group, :batch]
    scratch["reserved_hot"] = policy.fused_reserved_hot[group, :batch]
    return scratch


def index_update_arguments(policy, workspace, storage, rotating, group):
    """Bind persistent Index and output buffers shared by both transports."""
    return dict(
        keys=storage.keys, scales=storage.scales, addresses=storage.addresses,
        logical_ids=storage.logical_ids, row_owners=workspace.inputs.row_owners,
        rotating=rotating, status=policy.fused_errors[group], query_inputs=workspace.inputs,
        outputs=policy.fused_outputs[group],
    )


def bind_group_selector(torch, policy, workspace, setup, group, storage, rotating, update, *, ubmem=False):
    """Bind the shared selector to one group's persistent storage."""
    selector = VllmIndexSelector(
        torch, inputs=workspace.inputs, rotating=rotating, update=update,
        logical_ids=storage.logical_ids, row_owners=workspace.inputs.row_owners,
        prefix_workspace=workspace.prefix_reset, storage=storage, unique_active_owners=ubmem,
    )
    selector.tail_source = policy.abc_descriptors[group, : setup.batch, 2:4]
    return selector


class VllmIndexSelector:
    def __init__(
        self,
        torch,
        *,
        inputs,
        rotating,
        update,
        logical_ids,
        row_owners,
        prefix_workspace,
        storage=None,
        forward_factory=LIFetchForward,
        unique_active_owners=False,
    ):
        self.torch = torch
        self.inputs, self.rotating, self.update = inputs, rotating, update
        self.forward = forward_factory(rotating, update)
        self.unique_active_owners = unique_active_owners
        self.logical_ids, self.row_owners = logical_ids, row_owners
        self.prefix_workspace = prefix_workspace
        self.storage, self.positions = storage, None
        self.topk = rotating.arguments[9].view(torch.int32).view(inputs.batch, 4, 2048)
        self.history = logical_ids.shape[1]
        if logical_ids.ndim != 2 or not logical_ids.is_contiguous() or logical_ids.dtype != torch.int64:
            raise ValueError("LIFetch logical IDs require contiguous int64 owner/history rows")
        if row_owners.shape != (inputs.batch,) or row_owners.dtype != torch.int32:
            raise ValueError("LIFetch row owners must match the bound batch")
        if rotating.locations.shape != self.topk.shape:
            raise ValueError("LIFetch locations and TopK must share the bound shape")
        for actual, expected in (
            (rotating.arguments[0], inputs.query),
            (rotating.arguments[2], inputs.weights),
            (rotating.arguments[3], inputs.scale),
        ):
            if actual.data_ptr() != expected.data_ptr():
                raise ValueError("LIFetch kernel must bind the persistent packed query buffers")
        self.attention_indices = None
        self.plans = None
        self.tail_source = None

    def prepare(self, plans, *, prepare_inputs=True):
        """CPU request layout, once per forward; no device readback."""
        self.plans = tuple(plans)
        self.decodes = tuple(p for p in self.plans if p.decode)
        if self.unique_active_owners:
            owners = [p.owner for p in self.decodes if p.stop > p.begin]
            if len(owners) != len(set(owners)):
                raise ValueError("UBmem active decode rows must have distinct stable owners")
        if self.storage is not None:
            self.storage.prepare(self.plans)
        if prepare_inputs:
            self.inputs.prepare(
                [int(p.stop - p.begin) for p in self.decodes],
                query_starts=[int(p.begin) for p in self.decodes],
                query_owners=[int(p.owner) for p in self.decodes],
            )
        if self.row_owners.data_ptr() != self.inputs.row_owners.data_ptr():
            self.row_owners.copy_(self.inputs.row_owners)

    def __call__(self, query, weights, indexer, metadata, query_scale):
        if self.storage is not None:
            if self.plans and len(self.decodes) != len(self.plans):
                self.storage.ingest_prefill(indexer.k_cache.kv_cache, metadata, self.plans, self.positions)
            else:
                self.storage.ingest_decode(indexer.k_cache.kv_cache, metadata, self.inputs, self.row_owners)
        if self.plans and not self.decodes:
            return indexer.select_topk_native(query, weights, metadata, query_scale)
        self.inputs.pack(query, weights, query_scale)
        if self.tail_source is not None:
            self.rotating.tail_bounds.copy_(self.tail_source)
        # v9_init owns round-local initialization inside the captured launch.
        self.forward()

        mixed = self.plans is not None and len(self.decodes) != len(self.plans)
        token_count = self.inputs.token_count if mixed or weights.shape[0] > self.inputs.batch * 4 else weights.shape[0]
        # The update's prepare kernel publishes TND outputs before Apply moves
        # Index rows. No external gather/cast/mask or Q4 unpack is required.
        logical = self.update.logical_output[:token_count]
        self.attention_indices = self.update.slot_output[:token_count]
        if not mixed:
            return logical

        # Prefill keeps the original paged Index-cache and native LI path. Only
        # its query ranges execute native selection; decode is not recomputed.
        result = self.torch.full((weights.shape[0], 1, 2048), -1, dtype=self.torch.int32, device=weights.device)
        query_rows = query.view(weights.shape[0], self.inputs.index_heads, 128)
        scale_rows = query_scale.view(weights.shape[0], self.inputs.index_heads)
        offset = 0
        for row, plan in enumerate(self.plans):
            count = plan.stop - plan.begin
            if plan.decode:
                result[plan.begin : plan.stop].copy_(logical[offset : offset + count])
                offset += count
                continue
            prefill = copy(metadata)
            prefill.topk_selector = None
            prefill.num_actual_tokens = count
            prefill.block_table = metadata.block_table[row : row + 1]
            prefill.actual_seq_lengths_query = prefill.cum_query_lens = plan.query_end
            prefill.actual_seq_lengths_key = prefill.seq_lens = plan.key_end
            selected = indexer.select_topk_native(
                query_rows[plan.begin : plan.stop],
                weights[plan.begin : plan.stop],
                prefill,
                scale_rows[plan.begin : plan.stop],
            )
            result[plan.begin : plan.stop].copy_(selected.view(count, 1, 2048))
        return result

    def slots(self, token_count):
        if self.attention_indices is None:
            raise RuntimeError("LIFetch selection must execute before SFA")
        if token_count > self.attention_indices.shape[0]:
            raise ValueError("SFA query count exceeds the fused selection")
        return self.attention_indices[:token_count]
