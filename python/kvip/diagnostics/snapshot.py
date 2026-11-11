# SPDX-License-Identifier: Apache-2.0
"""Opt-in eager capture immediately around one real service fusion call."""

import json
from pathlib import Path

import numpy as np

from kvip.diagnostics.archive import TensorArchive, cpu_array, source_commit
from kvip.diagnostics.oracle import DecodeOracle
from kvip.adapters.vllm.inputs import reserved_hot_reference


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


class DecodeSnapshotRecorder:
    """One diagnostic eager round; normal graph capture/replay must not call this.

    The controller attaches this object only after selecting a genuine decode
    round. Before/after synchronize and copy data: capture timing is deliberately
    not a service performance measurement. The exporter pins all published
    pages during the before copy, including pages outside TopK. The controller
    preserves serving ownership through after().
    """

    def __init__(self, torch, directory, metadata, page_exporter, *, object_store=None):
        self.torch, self.directory = torch, Path(directory).resolve()
        self.metadata, self.page_exporter = dict(metadata), page_exporter
        self.object_store = object_store
        self.completed, self.result_path, self.started = False, None, False

    def before(self, selector):
        if self.started or self.completed:
            raise RuntimeError("A decode recorder may capture exactly one invocation")
        if self.metadata.get("capture_mode") != "eager_diagnostic":
            raise ValueError("Real decode snapshots require an explicitly selected eager diagnostic round")
        rotating = selector.rotating
        if not hasattr(rotating, "lease") or rotating.lease.runtime.transport != "ubmem":
            raise ValueError("Service capture currently supports the UBmem fusion binding")
        if rotating.tail_bounds is None:
            raise ValueError("Service snapshot requires explicit ABC bounds")
        if self.directory.exists() and any(self.directory.iterdir()):
            raise FileExistsError(f"Refusing to overwrite snapshot directory: {self.directory}")
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "INCOMPLETE").write_text("A manifest is published only after full verification.\n")
        self.started = True
        rotating.lease.check_stream().synchronize()
        shape = rotating.shape
        raw_context = cpu_array(self.torch, rotating.context).reshape(-1)
        limits = raw_context[120:128].view("<u4")
        geometry = dict(
            batch=shape.batch,
            history=shape.history,
            resident=shape.resident,
            heads=shape.heads,
            token_bytes=shape.token_bytes,
            aic_count=shape.aic_count,
            requested_windows=shape.intermediate_fetches,
            rounds=rotating.rounds,
            service_lanes=rotating.service_lanes,
            final_workers=rotating.final_workers,
            output_stride=rotating.output_stride,
            owners=rotating.hot_valid.shape[0],
            prefetch_limit=int(limits[0]),
            window_limit=int(limits[1]),
            soc=shape.soc,
            token_count=selector.inputs.token_count,
        )
        self.manifest = dict(
            schema_version=2 if self.object_store is not None else 1,
            status="incomplete",
            source_transport="ubmem",
            metadata=self.metadata,
            provenance=dict(kvip_commit=source_commit()),
            geometry=geometry,
            tensors={},
            scope="packed_Q4_Index_ready_to_LI_TopK_Fetch_IndexUpdate_TND_outputs",
            fidelity="Exact tensor bytes and arena-relative VA offsets; physical pages and THP coverage are not restored.",
            aliases={
                "row_owners": "metadata[4].view(int32)[:batch]",
                "unpack_indices": "metadata[2]",
                "output_valid": "metadata[3]",
                "selected": "topk",
                "final_posts": "owners scratch tail",
                "prefix_reset": "prefix_storage byte prefix",
                "hot_payload": "registered output owner rows",
                "policy": "Rebuild workspace/bindings/leases; never restore native pointers or queue cursors.",
            },
        )
        self.archive = TensorArchive(self.directory, self.manifest, object_store=self.object_store)
        compact_reserved = getattr(shape, "compact_reserved", False)
        for name, tensor in (persistent_tensors(selector) | input_tensors(selector)).items():
            if name == "reserved_hot" and compact_reserved:
                # This output of init still contains the preceding round at
                # the capture boundary. Save its exact deterministic expected
                # value from the captured compact input, without running init
                # early or changing any live device buffer/cache decision.
                expected = reserved_hot_reference(
                    self.archive.array("pre.lru_slots"), self.archive.array("pre.query_ends"), shape.resident
                )
                self.archive.save("pre.reserved_hot", expected)
                self.manifest["input_derivations"] = {
                    "pre.reserved_hot": "Init output reconstructed from pre.lru_slots and pre.query_ends; inactive rows are canonical -1."
                }
                self.manifest["fidelity"] += (
                    " reserved_hot is derived from compact LRU input, not a pre-forward device copy."
                )
            else:
                self.archive.save_tensor("pre." + name, self.torch, tensor)
        validate_capture_inputs(self.archive)
        self.manifest["ddr"] = self.page_exporter(selector, self.directory / "ddr")
        # Rebase to the same base only to prove every nonzero address has a
        # complete saved record in every executable owner row. Unreferenced
        # retired owners may still hold inert metadata for freed DDR blocks.
        relocate_snapshot_addresses(self.archive, self.manifest["ddr"]["region_base"])

    def after(self, selector):
        if not self.started or self.completed:
            raise RuntimeError("Snapshot after() requires one unfinished before()")
        selector.rotating.lease.check_stream().synchronize()
        reads = check_completion(self.torch, selector.rotating, selector.update)
        if "input_derivations" in self.manifest:
            actual = cpu_array(self.torch, selector.rotating.reserved_hot)
            expected = self.archive.array("pre.reserved_hot")
            active = np.any(self.archive.array("pre.query_ends").reshape(len(expected), 4) > 0, axis=1)
            if not np.array_equal(actual[active], expected[active]):
                raise AssertionError("Init reserved-hot reconstruction differs from captured compact LRU candidates")
            self.archive.save("post.reserved_hot", actual)
        state = post_state(self.torch, selector)
        oracle = DecodeOracle(self.archive)
        result = oracle.verify(state)
        for name in ("logical_output", "slot_output", "logical_ids", "hot_valid"):
            self.archive.save("post." + name, state[name])
        for name in ("selected_tags", "selected_payload", "selected_references"):
            self.archive.save("post." + name, result[name])
        for name, tensor in dict(
            topk=selector.topk,
            locations=selector.rotating.locations,
            rows=selector.rotating.rows,
            row_status=selector.rotating.row_status,
            final_posts=selector.rotating.final_posts,
            ubmem_status=selector.rotating.ubmem_status,
            final_addresses=selector.rotating.fallback_addresses,
            destinations=selector.rotating.destinations,
            changes=selector.rotating.changes,
            change_slots=selector.rotating.change_slots,
            change_count=selector.rotating.change_count,
            index_status=selector.update.status,
        ).items():
            self.archive.save_tensor("post." + name, self.torch, tensor)
        if getattr(getattr(selector.rotating, "shape", None), "tagged_map", False):
            self.archive.save_tensor("post.mapping_raw", self.torch, selector.rotating.mapping)
            self.manifest["post_mapping_encoding"] = "post.mapping_raw is epoch14/slot18; post.mapping is decoded slots"
        self.archive.save("post.mapping", mapping_values(self.torch, selector.rotating))
        mapping = self.archive.array("post.mapping")
        pre_ids = self.archive.array("pre.logical_ids")
        row_owners = self.archive.array("pre.metadata")[4].view(np.int32)[: len(mapping)]
        prefetched = []
        resident = self.manifest["geometry"]["resident"]
        for row, values in enumerate(mapping):
            # ABC mappings use resident+8192+tail_offset and are not DDR reads.
            positions = np.flatnonzero((values >= resident) & (values < resident + 2048))
            prefetched.extend((row, int(position), int(pre_ids[row_owners[row], position])) for position in positions)
        self.archive.save("post.prefetch_records", np.asarray(prefetched, dtype=np.int64).reshape(-1, 3))
        final_records = []
        rows = self.archive.array("post.rows")
        addresses = self.archive.array("post.final_addresses").view(np.uint64)
        destinations = self.archive.array("post.destinations")
        pre_addresses = self.archive.array("pre.addresses").view(np.uint64)
        base = self.manifest["ddr"]["region_base"]
        for row in range(len(rows)):
            count = int(rows[row, 247])
            if not 0 <= count <= 8192:
                raise AssertionError("Final Fetch count exceeds descriptor capacity")
            source_addresses = pre_addresses[row_owners[row]]
            order = np.argsort(source_addresses)
            sorted_addresses = source_addresses[order]
            for i in range(count):
                address = addresses[row, i]
                offset = int(address) - base
                if not 0 <= offset < self.manifest["ddr"]["region_bytes"]:
                    raise AssertionError("Final Fetch descriptor is outside its source arena")
                position = int(np.searchsorted(sorted_addresses, address))
                if position == len(order) or sorted_addresses[position] != address:
                    raise AssertionError("Final Fetch descriptor has no pre-round logical token")
                logical_id = int(pre_ids[row_owners[row], order[position]])
                final_records.append((row, logical_id, offset, int(destinations[row * 8192 + i])))
        self.archive.save("post.final_records", np.asarray(final_records, dtype=np.int64).reshape(-1, 4))
        self.manifest["oracle"] = dict(
            passed=True,
            logical_topk="same logical set per valid query; arbitrary physical slots allowed",
            index="complete logical-ID permutation with key/scale/address identity and valid-hot payload",
            selected_payload="deduplicated actual output bytes indexed by selected_references",
            index_hashes=result["index_hashes"],
            reads=reads,
            read_records={
                "prefetch_records": ["packed_row", "pre_index_position", "logical_id"],
                "final_records": ["packed_row", "logical_id", "arena_byte_offset", "destination_row"],
            },
        )
        self.manifest["status"] = "complete"
        # Every shared extent must be durable before this invocation becomes
        # independently replayable. No whole-array cache survives after().
        if self.object_store is not None:
            self.object_store.flush()
        pending = self.directory / "manifest.json.tmp"
        pending.write_text(json.dumps(self.manifest, indent=2, sort_keys=True) + "\n")
        pending.replace(self.directory / "manifest.json")
        (self.directory / "INCOMPLETE").unlink()
        self.completed = True
        self.result_path = str(self.directory / "manifest.json")
