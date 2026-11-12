# SPDX-License-Identifier: Apache-2.0
"""Opt-in eager capture immediately around one real service fusion call."""

import numpy as np

from kvip.diagnostics.archive import cpu_array


def decode_tagged_mapping(mapping, rows):
    """Normalize the opt-in epoch14/slot18 ABI for untimed diagnostics."""
    raw = np.asarray(mapping).view(np.uint32)
    control = np.asarray(rows).view(np.uint32)
    if raw.ndim != 2 or control.ndim != 2 or len(raw) != len(control) or control.shape[1] < 2:
        raise ValueError("Tagged mapping requires matching mapping and control rows")
    epochs = control[:, 0]
    if np.any(control[:, 1] != 0x4D415031) or np.any((epochs == 0) | (epochs > 16383)):
        raise AssertionError("Tagged mapping has invalid epoch state")
    valid = (raw != np.uint32(0xFFFFFFFF)) & ((raw >> 18) == epochs[:, None])
    return np.where(valid, raw & np.uint32(0x3FFFF), np.uint32(0xFFFFFFFF)).view(np.int32)


def mapping_values(torch, rotating, rows=None):
    raw = cpu_array(torch, rotating.mapping)
    if not getattr(getattr(rotating, "shape", None), "tagged_map", False):
        return raw
    return decode_tagged_mapping(raw, cpu_array(torch, rotating.rows) if rows is None else rows)


def persistent_tensors(selector):
    rotating, storage = selector.rotating, selector.storage
    if storage is None:
        raise ValueError("A decode snapshot requires service-owned Index storage")
    return dict(
        keys=storage.keys,
        key_scales=storage.scales,
        addresses=storage.addresses,
        logical_ids=storage.logical_ids,
        hot_valid=storage.hot_valid,
        output=rotating.output,
    )


def input_tensors(selector):
    rotating, inputs = selector.rotating, selector.inputs
    return dict(
        metadata=inputs.metadata,
        query=inputs.query,
        weights=inputs.weights,
        query_scale=inputs.scale,
        query_ends=inputs.query_ends,
        query_lengths=rotating.arguments[5],
        key_lengths=rotating.arguments[6],
        block_table=inputs.block_table,
        lru_slots=rotating.lru_slots,
        reserved_hot=rotating.reserved_hot,
        tail_bounds=rotating.tail_bounds,
    )


def post_state(torch, selector):
    state = {name: cpu_array(torch, value) for name, value in persistent_tensors(selector).items()}
    state.update(
        logical_output=cpu_array(torch, selector.update.logical_output),
        slot_output=cpu_array(torch, selector.update.slot_output),
    )
    return state


def check_completion(torch, rotating, update):
    rows = cpu_array(torch, rotating.rows)
    status = cpu_array(torch, rotating.row_status)
    posts = cpu_array(torch, rotating.final_posts)
    completed = cpu_array(torch, rotating.ubmem_status)
    if np.any(rows[:, [11, 211, 248]]) or np.any(status[:, 0]) or np.any(posts[:, :, 1:4]):
        raise AssertionError("Fusion transport/resolver has error or pending work")
    if np.any(completed[:, 0]) or not np.array_equal(completed[:, 1:4:2], completed[:, 2:5:2]):
        raise AssertionError("UBmem submitted/completed counts disagree")
    if np.any(cpu_array(torch, update.status)):
        raise AssertionError("Index update rejected the completed fusion")
    return dict(prefetch=int(completed[:, 1].sum()), final=int(completed[:, 3].sum()), bytes=int(completed[:, 5].sum()))


def relocate_snapshot_addresses(archive, new_base):
    """Only packed-owner rows are executable; retired owners retain inert bits."""
    from kvip.diagnostics.ddr import rebase_addresses

    values = archive.array("pre.addresses").copy()
    batch = archive.manifest["geometry"]["batch"]
    owners = np.unique(archive.array("pre.metadata")[4].view(np.int32)[:batch])
    if np.any(owners < 0) or np.any(owners >= len(values)):
        raise ValueError("Snapshot row owner is outside its Index storage")
    values[owners] = rebase_addresses(
        values[owners],
        archive.directory / "ddr",
        archive.manifest["ddr"],
        new_base,
        width=archive.manifest["geometry"]["token_bytes"],
    )
    return values


def validate_capture_inputs(archive):
    """Check actual packed device inputs, independent of lagging CPU MTP lengths."""
    geometry, source = archive.manifest["geometry"], archive.manifest["metadata"]
    batch = geometry["batch"]
    metadata = archive.array("pre.metadata")
    ends = archive.array("pre.query_ends").reshape(batch, 4)
    owners = metadata[4].view(np.int32)[:batch]
    requests = sorted(source.get("requests", []), key=lambda request: request["query_begin"])
    if batch != source.get("geometry", {}).get("batch", 16):
        raise ValueError("Actual fusion batch differs from the requested capture batch")
    if (
        geometry["token_count"] != batch * 4
        or len(requests) != batch
        or np.any(metadata[1] != 1)
        or np.any(metadata[3] != 1)
        or any(request["query_stop"] - request["query_begin"] != 4 for request in requests)
    ):
        raise ValueError("Snapshot requires every packed batch row to contain four real MTP queries")
    expected_owners = [request["owner"] for request in requests]
    if np.any(owners <= 0) or len(np.unique(owners)) != batch or not np.array_equal(owners, expected_owners):
        raise ValueError("Actual packed owners differ from the selected real requests")
    if np.any(np.diff(ends, axis=1) != 1) or np.any(ends <= 0) or np.any(ends > geometry["history"]):
        raise ValueError("Actual packed query positions are not four consecutive causal positions")
    context = source["context"]
    if np.any(ends[:, 0] - 1 < context["min_context"]) or np.any(ends[:, -1] > context["max_context"]):
        raise ValueError("Actual NPU query positions are outside the requested capture context")
