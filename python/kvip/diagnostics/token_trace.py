# SPDX-License-Identifier: Apache-2.0
"""Small, ID-only observations of one real Q4 fusion call.

This is not a numerical/payload snapshot. It can drive separately labelled
synthetic operator validation, but cannot reproduce the real LI scores or
measurements. Device addresses exist only temporarily while translating Fetch
records; no native pointer, query, key, scale or KV bytes are serialized.
"""

import copy
import json
import os
from pathlib import Path

import numpy as np

from kvip.diagnostics.archive import checked_path, cpu_array, file_digest, source_commit
from kvip.diagnostics.demand import DEMAND_ACCOUNTING, classify_initial_slots, sum_demand
from kvip.diagnostics.snapshot import check_completion, mapping_values, validate_capture_inputs

KIND = "kvaf_decode_token_trace"
SCHEMA_VERSION = 3
TOPK = 2048
QUERIES = 4
BLOCK = 128
_ARRAYS = {
    "pre.row_owners",
    "pre.logical_ids",
    "pre.hot_valid",
    "pre.metadata",
    "pre.query_ends",
    "pre.tail_bounds",
    "pre.lru_slots",
    "pre.address_valid",
    "pre.ddr_page_slots",
    "post.logical_ids",
    "post.hot_valid",
    "post.logical_output",
    "post.slot_output",
    "post.topk",
    "post.prefetch_records",
    "post.final_records",
    "post.read_counts",
}


def _i32(value):
    value = np.asarray(value)
    if value.dtype.kind not in "iu" or np.any(value < -(1 << 31)) or np.any(value > (1 << 31) - 1):
        raise ValueError("Token IDs/slots must fit signed int32")
    return np.ascontiguousarray(value, dtype=np.int32)


def _metadata(value):
    """The serving controller's full-snapshot metadata can contain pointers."""
    if isinstance(value, dict):
        return {
            key: _metadata(item)
            for key, item in value.items()
            if key not in {"published_pages", "addresses", "region_base", "host_base", "device_base"}
        }
    if isinstance(value, (list, tuple)):
        return [_metadata(item) for item in value]
    return copy.deepcopy(value)


class TokenIdTrace:
    """In-memory compressed archive reader; compact owner rows follow packed Q4."""

    def __init__(self, directory, manifest, arrays):
        self.directory, self.manifest, self.arrays = Path(directory), manifest, arrays

    def array(self, name, *, verify=False):
        return self.arrays[name]

    def close(self):
        self.arrays.clear()


def _geometry(archive):
    g = archive.manifest["geometry"]
    b, h, r, stride = (g[name] for name in ("batch", "history", "resident", "output_stride"))
    if (
        not all(type(n) is int for n in (b, h, r, stride))
        or not 1 <= b <= 16
        or h % BLOCK
        or h < r
        or r < TOPK
        or r % BLOCK
        or stride <= r
        or g["token_bytes"] not in (656, 2624)
        or g["tokens_per_block"] != BLOCK
        or type(g["slot_bytes"]) is not int
        or g["slot_bytes"] < BLOCK * g["token_bytes"]
        or type(g["region_bytes"]) is not int
        or g["region_bytes"] <= 0
        or g["region_bytes"] % g["slot_bytes"]
    ):
        raise ValueError("Token trace geometry is outside the Q4 service contract")
    return b, h, r, stride


def _validate_pre(archive, *, page_slots=True):
    b, h, r, stride = _geometry(archive)
    a = archive.arrays
    shapes = {
        "pre.row_owners": (b,),
        "pre.logical_ids": (b, h),
        "pre.hot_valid": (b, r),
        "pre.metadata": (5, b * QUERIES),
        "pre.query_ends": (b, QUERIES),
        "pre.tail_bounds": (b, 2),
        "pre.lru_slots": (b, TOPK),
        "pre.address_valid": (b, h),
        "pre.ddr_page_slots": (b, h // BLOCK),
    }
    for name, shape in shapes.items():
        if name == "pre.ddr_page_slots" and not page_slots:
            continue
        dtype = (
            np.int64
            if name == "pre.metadata"
            else np.uint8
            if name.endswith(("hot_valid", "address_valid"))
            else np.int32
        )
        if a[name].shape != shape or a[name].dtype != dtype:
            raise ValueError(f"Token trace dtype/shape differs: {name}")
    validate_capture_inputs(archive)
    owners = a["pre.row_owners"]
    if not np.array_equal(owners, a["pre.metadata"][4].view(np.int32)[:b]) or np.any(
        owners >= archive.manifest["geometry"]["owners"]
    ):
        raise ValueError("Compact rows differ from original packed owners")
    unpack = a["pre.metadata"][2]
    if not np.array_equal(np.sort(unpack), np.arange(b * QUERIES)):
        raise ValueError("TND mapping must cover every packed query exactly once")
    for row in range(b):
        ids, hot, valid = (a[name][row] for name in ("pre.logical_ids", "pre.hot_valid", "pre.address_valid"))
        if not np.array_equal(np.sort(ids), np.arange(h)):
            raise ValueError("Pre Index logical IDs are not a complete permutation")
        if np.any(hot > 1) or np.any(valid > 1):
            raise ValueError("Hot/DDR validity must contain only zero or one")
        lru = a["pre.lru_slots"][row]
        if np.any(lru < 0) or np.any(lru >= r) or len(np.unique(lru)) != TOPK:
            raise ValueError("LRU candidate slots must be distinct resident slots")
        begin, end = map(int, a["pre.tail_bounds"][row])
        ends = a["pre.query_ends"][row]
        if (
            not (0 <= begin <= int(ends[0]) - 1 < int(ends[-1]) <= end <= h)
            or begin % BLOCK
            or end - begin > stride - r
        ):
            raise ValueError("ABC bounds do not contain the captured causal queries")
        ordered = np.empty(h, dtype=np.uint8)
        ordered[ids] = valid
        pages = ordered.reshape(-1, BLOCK)
        if page_slots:
            slots = a["pre.ddr_page_slots"][row]
            g = archive.manifest["geometry"]
            if (
                np.any(slots < -1)
                or np.any(slots >= g["region_bytes"] // g["slot_bytes"])
                or not np.array_equal(slots >= 0, pages[:, 0] != 0)
            ):
                raise ValueError("Published page slots differ from DDR availability or arena capacity")
        if np.any(pages != pages[:, :1]) or np.any(ordered[:begin] == 0):
            raise ValueError("Published availability requires complete logical pages and committed prefix")


def _validate_post(archive):
    b, h, r, stride = _geometry(archive)
    a = archive.arrays
    shapes = {
        "post.logical_ids": (b, h),
        "post.hot_valid": (b, r),
        "post.logical_output": (b * QUERIES, TOPK),
        "post.slot_output": (b * QUERIES, TOPK),
        "post.topk": (b, QUERIES, TOPK),
        "post.read_counts": (b, 3),
    }
    for name, shape in shapes.items():
        dtype = np.uint8 if name.endswith("hot_valid") else np.int64 if name.endswith("read_counts") else np.int32
        if a[name].shape != shape or a[name].dtype != dtype:
            raise ValueError(f"Token trace dtype/shape differs: {name}")
    for name in ("post.prefetch_records", "post.final_records"):
        if a[name].ndim != 2 or a[name].shape[1] != 3 or a[name].dtype != np.int32:
            raise ValueError(f"Invalid token Fetch records: {name}")
    for row in range(b):
        if not np.array_equal(np.sort(a["post.logical_ids"][row]), np.arange(h)):
            raise ValueError("Post Index logical IDs are not a complete permutation")
    if np.any(a["post.hot_valid"] > 1):
        raise ValueError("Post hot validity must contain only zero or one")
    topk = a["post.topk"].reshape(b * QUERIES, TOPK)
    for tnd, source in enumerate(a["pre.metadata"][2]):
        row, query = divmod(int(source), QUERIES)
        positions, logical, slots = topk[source], a["post.logical_output"][tnd], a["post.slot_output"][tnd]
        if (
            np.any(positions < 0)
            or np.any(positions >= h)
            or not np.array_equal(a["pre.logical_ids"][row, positions], logical)
        ):
            raise ValueError("Actual TopK positions and published logical IDs disagree")
        if len(np.unique(logical)) != TOPK or np.any(logical < 0) or np.any(logical >= a["pre.query_ends"][row, query]):
            raise ValueError("Actual TopK violates unique causal selection")
        if np.any(slots < 0) or np.any(slots >= stride):
            raise ValueError("Selected output slot is outside the owner payload pool")
        hot = slots < r
        if np.any(a["post.hot_valid"][row, slots[hot]] == 0) or not np.array_equal(
            a["post.logical_ids"][row, slots[hot]], logical[hot]
        ):
            raise ValueError("Published hot slots disagree with post Index IDs")
        begin, end = a["pre.tail_bounds"][row]
        if np.any((logical[~hot] < begin) | (logical[~hot] >= end)) or not np.array_equal(
            slots[~hot], r + logical[~hot] % (stride - r)
        ):
            raise ValueError("Published ABC slots disagree with causal tail IDs")
    counts = a["post.read_counts"]
    if np.any(counts < 0) or not np.array_equal(
        counts[:, 2], counts[:, :2].sum(axis=1) * archive.manifest["geometry"]["token_bytes"]
    ):
        raise ValueError("Fetch counts and inferred byte counts disagree")
    for column, name in enumerate(("post.prefetch_records", "post.final_records")):
        records = a[name]
        if np.any(records[:, 0] < 0) or np.any(records[:, 0] >= b):
            raise ValueError("Fetch record packed row is outside batch")
        if not np.array_equal(np.bincount(records[:, 0], minlength=b), counts[:, column]):
            raise ValueError("Fetch token records do not match completed transport counters")
        for row in range(b):
            records_row = records[records[:, 0] == row]
            if column == 0:
                positions, ids = records_row[:, 1], records_row[:, 2]
                if (
                    np.any(positions < 0)
                    or np.any(positions >= h)
                    or not np.array_equal(a["pre.logical_ids"][row, positions], ids)
                ):
                    raise ValueError("Prefetch IDs differ from original Index positions")
            else:
                ids, destinations = records_row[:, 1], records_row[:, 2]
                if np.any(ids < 0) or np.any(ids >= h) or np.any(destinations < 0) or np.any(destinations >= r):
                    raise ValueError("Final Fetch ID/destination is outside the resident pool")
                if np.any(a["post.hot_valid"][row, destinations] == 0) or not np.array_equal(
                    a["post.logical_ids"][row, destinations], ids
                ):
                    raise ValueError("Final Fetch destinations disagree with published Index IDs")
                inverse = np.empty(h, dtype=np.int32)
                inverse[a["pre.logical_ids"][row]] = np.arange(h)
                positions = inverse[ids]
            if np.any(a["pre.address_valid"][row, positions] == 0):
                raise ValueError("Fetch token was absent from pre-call published DDR availability")


def trace_summary(archive):
    """Selection demand and actual I/O are deliberately separate quantities."""
    b, h, r, stride = _geometry(archive)
    a, requests = archive.arrays, []
    categories = ("cold", "candidate_hot", "other_hot", "abc")
    totals = {prefix + category: 0 for prefix in ("prefetch_", "unused_prefetch_", "final_") for category in categories}
    for row in range(b):
        ids = a["pre.logical_ids"][row]
        forward = np.full(h, -1, dtype=np.int32)
        hot = np.flatnonzero(a["pre.hot_valid"][row])
        forward[ids[hot]] = hot
        abc = np.arange(*a["pre.tail_bounds"][row], dtype=np.int32)
        forward[abc] = r + abc % (stride - r)
        tnd = np.flatnonzero(a["pre.metadata"][2] // QUERIES == row)
        selected = np.unique(a["post.logical_output"][tnd])
        lru = a["pre.lru_slots"][row]
        counts = classify_initial_slots(forward[selected], lru, r)
        observed = {"prefetch": 0, "final": 0, "unused_prefetch": 0}
        breakdown = {key: 0 for key in totals}
        for name, column in (("prefetch", 2), ("final", 1)):
            records = a["post." + name + "_records"]
            tokens = records[records[:, 0] == row, column]
            slots = forward[tokens]
            cold, tail = slots < 0, slots >= r
            candidate = ~cold & ~tail & np.isin(slots, lru)
            masks = (cold, candidate, ~cold & ~tail & ~candidate, tail)
            unused = ~np.isin(tokens, selected)
            observed[name] = len(tokens)
            if name == "final" and np.any(unused):
                raise ValueError("Final Fetch contains a token absent from the actual TopK union")
            for category, mask in zip(categories, masks):
                breakdown[name + "_" + category] = int(np.count_nonzero(mask))
                if name == "prefetch":
                    breakdown["unused_prefetch_" + category] = int(np.count_nonzero(mask & unused))
            if name == "prefetch":
                observed["unused_prefetch"] = int(np.count_nonzero(unused))
        for key, value in breakdown.items():
            totals[key] += value
        observed["total"] = observed["prefetch"] + observed["final"]
        observed["bytes"] = observed["total"] * archive.manifest["geometry"]["token_bytes"]
        observed["unused_bytes"] = observed["unused_prefetch"] * archive.manifest["geometry"]["token_bytes"]
        requests.append(
            dict(packed_row=row, owner=int(a["pre.row_owners"][row]), **counts, reads=observed, **breakdown)
        )
    reads = {key: sum(item["reads"][key] for item in requests) for key in requests[0]["reads"]}
    return dict(demand=sum_demand(requests), reads=reads, transfer_classes=totals, requests=requests)


def load_token_trace(directory, *, verify=True):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != KIND
        or manifest.get("content") != "token_ids"
        or manifest.get("status") != "complete"
    ):
        raise ValueError("A complete schema-3 token-ID trace is required")
    if (
        manifest.get("validation", {}).get("passed") is not True
        or manifest["validation"].get("kind") != "token_id_integrity"
    ):
        raise ValueError("Token trace requires its distinct ID-integrity validation marker")
    if _metadata(manifest["metadata"]) != manifest["metadata"]:
        raise ValueError("Token trace metadata contains forbidden native-address fields")
    description = manifest["archive"]
    path = checked_path(directory, description["file"])
    if verify and file_digest(path) != description["sha256"]:
        raise ValueError("Token trace archive checksum differs")
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != _ARRAYS or set(manifest["arrays"]) != _ARRAYS:
            raise ValueError("Token trace contains missing or forbidden arrays")
        arrays = {name: source[name] for name in source.files}
    for name, value in arrays.items():
        spec = manifest["arrays"][name]
        if value.dtype.hasobject or value.dtype.name != spec["dtype"] or list(value.shape) != spec["shape"]:
            raise ValueError(f"Token trace array differs from its manifest: {name}")
    archive = TokenIdTrace(directory, manifest, arrays)
    _validate_pre(archive)
    _validate_post(archive)
    if trace_summary(archive) != manifest["summary"]:
        raise ValueError("Token trace summary differs from its actual IDs")
    return archive


class TokenIdTraceRecorder:
    """Record real IDs and transport counts, with no DDR export or payload read."""

    def __init__(self, torch, directory, metadata):
        self.torch, self.directory = torch, Path(directory).resolve()
        self.metadata = _metadata(metadata)
        self.started, self.completed, self.result_path = False, False, None
        self._addresses = None

    def before(self, selector):
        try:
            self._capture_before(selector)
        except BaseException:
            self._addresses = None
            raise

    def _capture_before(self, selector):
        if self.started or self.completed:
            raise RuntimeError("A token-ID recorder may capture exactly one invocation")
        if self.metadata.get("capture_mode") != "eager_diagnostic":
            raise ValueError("Token trace requires an explicitly selected eager diagnostic round")
        rotating, storage, inputs = selector.rotating, selector.storage, selector.inputs
        if rotating.lease.runtime.transport != "ubmem" or storage is None or rotating.tail_bounds is None:
            raise ValueError("Token capture requires the service UBmem binding and explicit ABC bounds")
        if self.directory.exists() and any(self.directory.iterdir()):
            raise FileExistsError(f"Refusing to overwrite token trace: {self.directory}")
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "INCOMPLETE").write_text("IDs become complete only after transport and ID checks.\n")
        self.started = True
        rotating.lease.check_stream().synchronize()
        shape = rotating.shape
        limits = cpu_array(self.torch, rotating.context).reshape(-1)[120:128].view("<u4")
        region = rotating.lease.context
        provenance = self.metadata.get("provenance", {})
        configuration = provenance.get("kvaf_config", {})
        geometry = dict(
            batch=shape.batch,
            history=shape.history,
            resident=shape.resident,
            heads=shape.heads,
            token_bytes=shape.token_bytes,
            tokens_per_block=region["tokens_per_block"],
            slot_bytes=int(region["slot_bytes"]),
            region_bytes=int(region["region_bytes"]),
            page_mode=provenance.get("page_mode", configuration.get("ubmem_page_mode", "unknown")),
            numa_node=provenance.get("numa_node"),
            aic_count=shape.aic_count,
            requested_windows=shape.intermediate_fetches,
            rounds=rotating.rounds,
            service_lanes=rotating.service_lanes,
            final_workers=rotating.final_workers,
            output_stride=rotating.output_stride,
            owners=int(storage.logical_ids.shape[0]),
            prefetch_limit=int(limits[0]),
            window_limit=int(limits[1]),
            soc=shape.soc,
            token_count=inputs.token_count,
        )
        metadata = cpu_array(self.torch, inputs.metadata).copy()
        owners = metadata[4].view(np.int32)[: shape.batch].copy()
        if np.any(owners <= 0) or np.any(owners >= geometry["owners"]):
            raise ValueError("Token capture packed owner is outside Index storage")
        self._owners = owners
        arrays = {
            "pre.row_owners": owners,
            "pre.metadata": metadata,
            "pre.query_ends": _i32(cpu_array(self.torch, inputs.query_ends)).reshape(shape.batch, QUERIES),
            "pre.tail_bounds": _i32(cpu_array(self.torch, rotating.tail_bounds)),
            "pre.lru_slots": _i32(cpu_array(self.torch, rotating.lru_slots)),
        }
        # Read only executable owner rows. These small ID/address copies never
        # touch Index keys, scales, query tensors or the registered KV output.
        for name in ("logical_ids", "hot_valid"):
            values = np.stack([cpu_array(self.torch, getattr(storage, name)[int(owner)]) for owner in owners])
            arrays["pre." + name] = _i32(values) if name == "logical_ids" else (values != 0).astype(np.uint8)
        self._addresses = np.stack([cpu_array(self.torch, storage.addresses[int(owner)]) for owner in owners]).view(
            np.uint64
        )
        arrays["pre.address_valid"] = (self._addresses != 0).astype(np.uint8)
        self.manifest = dict(
            schema_version=SCHEMA_VERSION,
            kind=KIND,
            content="token_ids",
            status="incomplete",
            source_transport="ubmem",
            metadata=self.metadata,
            provenance=dict(kvip_commit=source_commit()),
            geometry=geometry,
            owner_layout="Arrays with B rows follow pre.row_owners in packed-Q4 order.",
            validation_scope="Transport completion, causal unique TopK IDs, Index ID permutation, slot IDs and observed Fetch records; no key/scale/score/KV-byte oracle.",
            replay_scope="Supports separately labelled synthetic operator validation; real LI numerical inputs, real payloads and physical DDR placement are not saved.",
            demand_accounting=DEMAND_ACCOUNTING,
            record_columns={
                "post.prefetch_records": ["packed_row", "pre_index_position", "logical_id"],
                "post.final_records": ["packed_row", "logical_id", "owner_relative_destination_slot"],
                "post.read_counts": ["prefetch", "final", "bytes"],
            },
        )
        self.archive = TokenIdTrace(self.directory, self.manifest, arrays)
        _validate_pre(self.archive, page_slots=False)
        # Verify the available bits really refer to complete, contiguous logical
        # pages. Synthetic replay may de-share/rebase them, never invent holes.
        page_slots = np.full((shape.batch, shape.history // BLOCK), -1, dtype=np.int32)
        base, region_bytes, slot_bytes = (int(region[name]) for name in ("region_base", "region_bytes", "slot_bytes"))
        if (
            not 0 < base < (1 << 64)
            or region_bytes <= 0
            or slot_bytes < BLOCK * shape.token_bytes
            or region_bytes % slot_bytes
            or base + region_bytes > (1 << 64)
        ):
            raise ValueError("Invalid native DDR region geometry")
        for row, ids in enumerate(arrays["pre.logical_ids"]):
            ordered = np.empty(shape.history, dtype=np.uint64)
            ordered[ids] = self._addresses[row]
            pages = ordered.reshape(-1, BLOCK)
            present = pages[:, 0] != 0
            expected = pages[present, :1] + np.arange(BLOCK, dtype=np.uint64) * shape.token_bytes
            if not np.array_equal(pages[present], expected):
                raise ValueError("Published logical pages do not have contiguous token addresses")
            starts = pages[present, 0]
            if np.any(starts < base) or np.any(starts >= base + region_bytes):
                raise ValueError("Published page lies outside its actual DDR arena")
            offsets = starts - np.uint64(base)
            if np.any(offsets % slot_bytes):
                raise ValueError("Published logical page does not begin at a KVAF slot")
            page_slots[row, present] = _i32(offsets // slot_bytes)
        arrays["pre.ddr_page_slots"] = page_slots
        _validate_pre(self.archive)

    def after(self, selector):
        if not self.started or self.completed:
            raise RuntimeError("Token trace after() requires one unfinished before()")
        rotating = selector.rotating
        try:
            rotating.lease.check_stream().synchronize()
            check_completion(self.torch, rotating, selector.update)
            a = self.archive.arrays
            for name in ("logical_ids", "hot_valid"):
                values = np.stack(
                    [cpu_array(self.torch, getattr(selector.storage, name)[int(owner)]) for owner in self._owners]
                )
                a["post." + name] = _i32(values) if name == "logical_ids" else (values != 0).astype(np.uint8)
            for name in ("logical_output", "slot_output"):
                a["post." + name] = _i32(cpu_array(self.torch, getattr(selector.update, name))).reshape(-1, TOPK)
            a["post.topk"] = _i32(cpu_array(self.torch, selector.topk))
            rows = cpu_array(self.torch, rotating.rows)
            mapping = mapping_values(self.torch, rotating, rows)
            prefetched, final = [], []
            r, stride = self.manifest["geometry"]["resident"], self.manifest["geometry"]["output_stride"]
            addresses = cpu_array(self.torch, rotating.fallback_addresses).view(np.uint64)
            destinations = cpu_array(self.torch, rotating.destinations).reshape(len(self._owners), -1)
            for row, values in enumerate(mapping):
                positions = np.flatnonzero((values >= r) & (values < r + TOPK))
                prefetched.extend(
                    (row, int(position), int(a["pre.logical_ids"][row, position])) for position in positions
                )
                count = int(rows[row, 247])
                if not 0 <= count <= QUERIES * TOPK:
                    raise ValueError("Final Fetch count exceeds descriptor capacity")
                valid = np.flatnonzero(self._addresses[row])
                order = valid[np.argsort(self._addresses[row, valid])]
                source = self._addresses[row, order]
                wanted = addresses[row, :count]
                found = np.searchsorted(source, wanted)
                if np.any(found >= len(source)) or (count and not np.array_equal(source[found], wanted)):
                    raise ValueError("Final Fetch address has no pre-call logical token")
                if count and np.any(np.searchsorted(source, wanted, side="right") - found != 1):
                    raise ValueError("Final Fetch source ambiguously names multiple logical tokens")
                for i, position in enumerate(order[found]):
                    final.append(
                        (
                            row,
                            int(a["pre.logical_ids"][row, position]),
                            int(destinations[row, i]) - int(self._owners[row]) * stride,
                        )
                    )
            a["post.prefetch_records"] = np.asarray(prefetched, dtype=np.int32).reshape(-1, 3)
            a["post.final_records"] = np.asarray(final, dtype=np.int32).reshape(-1, 3)
            counts = cpu_array(self.torch, rotating.ubmem_status)
            a["post.read_counts"] = np.ascontiguousarray(counts[:, [1, 3, 5]], dtype=np.int64)
            _validate_post(self.archive)
            self.manifest["summary"] = trace_summary(self.archive)
            self.manifest["validation"] = dict(
                passed=True, kind="token_id_integrity", scope=self.manifest["validation_scope"]
            )
            self.manifest["arrays"] = {
                name: dict(dtype=value.dtype.name, shape=list(value.shape)) for name, value in a.items()
            }
            path = self.directory / "token_ids.npz"
            with path.open("xb") as output:
                np.savez_compressed(output, **a)
                output.flush()
                os.fsync(output.fileno())
            self.manifest["archive"] = dict(file=path.name, sha256=file_digest(path), bytes=path.stat().st_size)
            self.manifest["status"] = "complete"
            pending = self.directory / "manifest.json.tmp"
            with pending.open("x") as output:
                json.dump(self.manifest, output, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            pending.replace(self.directory / "manifest.json")
            (self.directory / "INCOMPLETE").unlink()
            self.completed, self.result_path = True, str(self.directory / "manifest.json")
            self.archive.close()
        finally:
            # No address table survives a completed or failed after() call.
            self._addresses = None
