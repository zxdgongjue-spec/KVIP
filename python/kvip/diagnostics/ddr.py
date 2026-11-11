# SPDX-License-Identifier: Apache-2.0
"""Copy frozen UBmem pages and restore their fixed-slot layout on either transport.

The caller stops submission, drains background Put/feedback, and synchronizes
the device *before* export. A fused lease alone does not freeze serving state.
Only complete, published main-cache blocks are supported. No native addresses
are dereferenced from Python, and no queue/lease state is serialized.
"""

from __future__ import annotations

import hashlib
import copy
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
import shutil

import numpy as np

CHUNK_BYTES = 16 << 20
_U64_MAX = (1 << 64) - 1
_FILES = {
    "keys": "keys.npy",
    "addresses": "addresses.npy",
    "slot_offsets": "slot_offsets.npy",
    "valid_lengths": "valid_lengths.npy",
    "payload": "payload.bin",
}


def _integer(value, name, minimum=0, maximum=_U64_MAX):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} is out of range")
    return value


def _geometry(base, size, slot, token, tokens):
    values = [
        _integer(x, name, 1)
        for x, name in zip(
            (base, size, slot, token, tokens),
            ("region_base", "region_bytes", "slot_bytes", "token_bytes", "tokens_per_block"),
        )
    ]
    base, size, slot, token, tokens = values
    if base % 32 or base + size > _U64_MAX or size % slot or token not in (656, 2624) or token * tokens > slot:
        raise ValueError("inconsistent UBmem arena/block geometry")
    return dict(zip(("region_base", "region_bytes", "slot_bytes", "token_bytes", "tokens_per_block"), values))


def _runtime_geometry(runtime, target_transport="ubmem"):
    if target_transport not in ("ubmem", "urma"):
        raise ValueError("target_transport must be ubmem or urma")
    reported = getattr(runtime, "transport", None)
    # The original URMA binding has no transport property. Only an explicit
    # target_transport permits that ABI; absence must never infer a backend.
    if target_transport == "ubmem" and reported != "ubmem":
        raise ValueError("UBmem DDR snapshots require an explicit UBmem Runtime")
    if target_transport == "urma" and reported not in (None, "urma"):
        raise ValueError("target_transport differs from the Runtime transport")
    region = runtime.remote_region
    return _geometry(
        region["base"],
        region["length"],
        region["slot_size"],
        runtime.batch_get_token_bytes,
        runtime.batch_get_tokens_per_block,
    )


def _addresses(values):
    """Preserve uint64 address bits stored in signed torch/numpy int64 arrays."""
    values = np.asarray(values)
    if values.dtype.kind not in "iu" or values.dtype.itemsize != 8:
        raise ValueError("addresses must use int64 or uint64 storage")
    return values.astype(np.int64, copy=False).view(np.uint64) if values.dtype.kind == "i" else values


def _keys(values):
    # Shared prefix records may occur in more than one request; store each once.
    return np.asarray(list(dict.fromkeys(_integer(x, "key", maximum=(1 << 63) - 1) for x in values)), dtype=np.int64)


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_records(info, keys, addresses, offsets, lengths):
    count = len(keys)
    if any(np.asarray(x).ndim != 1 or len(x) != count for x in (keys, addresses, offsets, lengths)):
        raise ValueError("DDR record arrays must be equally sized vectors")
    if keys.dtype != np.dtype("int64") or any(x.dtype != np.dtype("uint64") for x in (addresses, offsets, lengths)):
        raise ValueError("DDR record array dtype mismatch")
    if len(set(map(int, keys))) != count or np.any(keys < 0):
        raise ValueError("DDR keys must be unique nonnegative slot IDs")
    block = info["token_bytes"] * info["tokens_per_block"]
    for key, address, offset, length in zip(keys, addresses, offsets, lengths):
        key, address, offset, length = map(int, (key, address, offset, length))
        if key >= info["region_bytes"] // info["slot_bytes"] or offset != key * info["slot_bytes"]:
            raise ValueError("DDR key/slot offset is outside the arena or inconsistent")
        if address != info["region_base"] + offset or length != block or offset + length > info["region_bytes"]:
            raise ValueError("DDR address/valid length differs from its complete fixed slot")


def _read_table(runtime, handle, keys, info):
    view = runtime.access_table_view(handle)
    addresses = _addresses(view["addresses"]).copy()
    lengths = np.asarray(view["lengths"], dtype=np.uint64).copy()
    valid = np.asarray(view["valid"])
    if valid.ndim != 1 or len(valid) != len(keys) or not np.all(valid == 1):
        raise ValueError("DDR access table contains missing or unpublished records")
    offsets = np.asarray([int(key) * info["slot_bytes"] for key in keys], dtype=np.uint64)
    _validate_records(info, keys, addresses, offsets, lengths)
    return addresses, offsets, lengths


def _chunk_records(info, chunk_bytes):
    chunk_bytes = _integer(chunk_bytes, "chunk_bytes", 1, 32 << 20)
    block = info["token_bytes"] * info["tokens_per_block"]
    if block > chunk_bytes:
        raise ValueError("one complete DDR block exceeds the chunk byte limit")
    return min(chunk_bytes // block, ((1 << 31) - 1) // info["tokens_per_block"])


def _torch(module):
    if module is None:
        import torch

        return torch
    return module


def _page_policy(page_mode, numa_node, generation):
    if page_mode not in ("base", "thp"):
        raise ValueError("snapshot page_mode must be base or thp")
    numa_node = _integer(numa_node, "numa_node", -1, 4095)
    if page_mode == "base" and numa_node != -1:
        raise ValueError("base page mode does not support explicit NUMA binding")
    if isinstance(generation, bool) or not isinstance(generation, (str, int)) or generation == "":
        raise ValueError("generation must identify the frozen capture")
    return dict(page_mode=page_mode, numa_node=numa_node, generation=generation)


def export_pages(
    runtime,
    keys,
    directory,
    *,
    page_mode,
    generation,
    numa_node=-1,
    chunk_bytes=CHUNK_BYTES,
    torch_module=None,
    object_store=None,
):
    """Export ALL supplied published blocks at the frozen before boundary.

    ``keys`` must cover the selected group's entire valid input address domain,
    not merely this launch's final misses. Caller-provided generation identifies
    that frozen capture; it is not a native slot generation counter. With an
    object store, every block is STILL read through BatchGet on every export;
    only the bytes written to disk are deduplicated by content.
    """
    info = _runtime_geometry(runtime)
    policy = _page_policy(page_mode, numa_node, generation)
    keys = _keys(keys)
    if len(keys) > (1 << 31) - 1 or np.any(keys >= info["region_bytes"] // info["slot_bytes"]):
        raise ValueError("DDR keys exceed the arena or int32 table index range")
    per_chunk = _chunk_records(info, chunk_bytes)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in _FILES.values()):
        raise FileExistsError("DDR snapshot files already exist")
    handle = None
    torch = _torch(torch_module)
    device = torch.device(f"npu:{runtime.device_id}")
    digest = hashlib.sha256()
    blocks = []
    try:
        if len(keys):
            handle = runtime.prepare_access_table(keys.tolist())
            addresses, offsets, lengths = _read_table(runtime, handle, keys, info)
            device_table = runtime.access_table_view(handle, device="npu")
        else:
            addresses = offsets = lengths = np.empty(0, dtype=np.uint64)
        output_file = (directory / _FILES["payload"]).open("xb") if object_store is None else nullcontext(None)
        with output_file as target:
            for first in range(0, len(keys), per_chunk):
                stop = min(first + per_chunk, len(keys))
                tokens, token_bytes = info["tokens_per_block"], info["token_bytes"]
                count = (stop - first) * tokens
                indices = torch.from_numpy(np.repeat(np.arange(first, stop, dtype=np.int32), tokens)).to(device)
                within = torch.from_numpy(np.tile(np.arange(tokens, dtype=np.int32) * token_bytes, stop - first)).to(
                    device
                )
                output_indices = torch.arange(count, dtype=torch.int32, device=device)
                miss_count = torch.tensor([count], dtype=torch.int32, device=device)
                output = torch.empty((count, token_bytes), dtype=torch.uint8, device=device)
                runtime.batch_get(device_table, indices, within, miss_count, output_indices, output, synchronize=True)
                data = output.cpu().numpy().tobytes()
                if target is not None:
                    target.write(data)
                else:
                    block = tokens * token_bytes
                    blocks.extend(object_store.put(data[start : start + block]) for start in range(0, len(data), block))
                digest.update(data)
        arrays = dict(keys=keys, addresses=addresses, slot_offsets=offsets, valid_lengths=lengths)
        files = {}
        for name, values in arrays.items():
            path = directory / _FILES[name]
            with path.open("xb") as target:
                np.save(target, values, allow_pickle=False)
            files[name] = {"path": path.name, "sha256": _hash(path)}
        payload = {}
        if object_store is None:
            files["payload"] = {"path": _FILES["payload"], "sha256": digest.hexdigest()}
        else:
            object_store.flush()
            payload["payload"] = dict(
                storage="cas_blocks",
                store=object_store.describe_for(directory),
                blocks=blocks,
                sha256=digest.hexdigest(),
            )
        return dict(
            info,
            schema_version=1 if object_store is None else 2,
            transport="ubmem",
            device_id=_integer(runtime.device_id, "device_id", 0, (1 << 31) - 1),
            **policy,
            record_count=len(keys),
            payload_bytes=len(keys) * info["token_bytes"] * info["tokens_per_block"],
            files=files,
            **payload,
        )
    finally:
        if handle is not None:
            runtime.release_access_table(handle)


class PayloadReader:
    """Bounded record access for legacy files and content-addressed page blocks.

    ``record_count``/``block_bytes`` also support callers that already validated
    geometry (the oracle). No entire DDR payload is materialized in memory.
    CAS objects are immutable: their complete bytes are checked before the
    first read, then cached with a bounded LRU for per-token oracle access.
    """

    def __init__(self, directory, manifest, *, record_count=None, block_bytes=None, cache_bytes=32 << 20):
        from kvip.diagnostics.archive import checked_path

        self.manifest = manifest
        self.schema = manifest.get("schema_version", 1)
        self.record_count = _integer(manifest["record_count"] if record_count is None else record_count, "record_count")
        self.block_bytes = _integer(
            manifest["token_bytes"] * manifest["tokens_per_block"] if block_bytes is None else block_bytes,
            "block_bytes",
            1,
        )
        self.shape = (self.record_count, self.block_bytes)
        self.nbytes = self.record_count * self.block_bytes
        self._cache_limit = _integer(cache_bytes, "cache_bytes", 0, 32 << 20)
        self._cache, self._cache_bytes, self._verified = OrderedDict(), 0, set()
        self._legacy = self._store = None
        if self.schema == 1:
            entry = manifest["files"]["payload"]
            if entry["path"] != _FILES["payload"]:
                raise ValueError("unexpected DDR snapshot file path")
            self.path = checked_path(directory, entry["path"])
            if self.path.stat().st_size != self.nbytes:
                raise ValueError("DDR snapshot payload is truncated or has trailing bytes")
            self.digest = entry.get("sha256")
            if self.nbytes:
                self._legacy = np.memmap(self.path, mode="r", dtype=np.uint8).reshape(self.shape)
        elif self.schema == 2:
            from kvip.diagnostics.storage import PackedObjectStore

            payload = manifest["payload"]
            if payload.get("storage") != "cas_blocks" or "payload" in manifest["files"]:
                raise ValueError("unsupported DDR payload storage")
            self._blocks = payload["blocks"]
            if not isinstance(self._blocks, list) or len(self._blocks) != self.record_count:
                raise ValueError("DDR payload record reference count mismatch")
            self.digest = payload["sha256"]
            self._identities = []
            for ref in self._blocks:
                if not isinstance(ref, dict) or not isinstance(ref.get("pack"), str):
                    raise ValueError("invalid DDR payload object reference")
                digest = ref.get("sha256")
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("invalid DDR payload object checksum")
                offset = _integer(ref["offset"], "payload object offset")
                if _integer(ref["length"], "payload object length", 1) != self.block_bytes:
                    raise ValueError("DDR payload object does not contain one complete block")
                self._identities.append((digest, ref["pack"], offset, self.block_bytes))
            self._store = PackedObjectStore.from_description(directory, payload["store"])
        else:
            raise ValueError("unsupported DDR payload schema")

    def _record(self, index):
        index = _integer(index, "record index", 0, self.record_count - 1)
        if self.schema == 1:
            return self._legacy[index]
        identity = self._identities[index]
        if identity in self._cache:
            self._cache.move_to_end(identity)
            return self._cache[identity]
        raw = self._store.read(self._blocks[index], verify=identity not in self._verified)
        if len(raw) != self.block_bytes:
            raise ValueError("DDR object changed or was truncated during read")
        self._verified.add(identity)
        result = np.frombuffer(raw, dtype=np.uint8)
        if result.nbytes <= self._cache_limit:
            while self._cache_bytes + result.nbytes > self._cache_limit:
                _, old = self._cache.popitem(last=False)
                self._cache_bytes -= old.nbytes
            self._cache[identity] = result
            self._cache_bytes += result.nbytes
        return result

    def __getitem__(self, item):
        """Read ``reader[record, byte_slice]`` without allocating other records."""
        if isinstance(item, tuple):
            if len(item) != 2:
                raise IndexError("DDR payload expects a record and byte slice")
            record, within = item
        else:
            record, within = item, slice(None)
        if isinstance(within, slice):
            start = 0 if within.start is None else _integer(within.start, "byte slice start", 0, self.block_bytes)
            stop = (
                self.block_bytes
                if within.stop is None
                else _integer(within.stop, "byte slice stop", 0, self.block_bytes)
            )
            if within.step not in (None, 1) or stop < start:
                raise ValueError("DDR payload byte slices must be contiguous and ordered")
            return self._record(record)[start:stop]
        index = _integer(within, "byte index", 0, self.block_bytes - 1)
        return self._record(record)[index]

    def read_records(self, first, stop):
        """Return a bounded consecutive-record chunk as bytes."""
        first = _integer(first, "first record", 0, self.record_count)
        stop = _integer(stop, "stop record", first, self.record_count)
        if (stop - first) * self.block_bytes > 32 << 20:
            raise ValueError("DDR payload read exceeds the 32 MiB chunk limit")
        if self.schema == 1:
            count = (stop - first) * self.block_bytes
            with self.path.open("rb") as source:
                source.seek(first * self.block_bytes)
                data = source.read(count)
            if len(data) != count:
                raise ValueError("DDR payload changed during restore")
            return data
        return b"".join(self._record(index).tobytes() for index in range(first, stop))

    def verify(self):
        """Check every referenced byte and its original ordered payload digest."""
        if self.schema == 1:
            actual = _hash(self.path)
        else:
            digest = hashlib.sha256()
            for index in range(self.record_count):
                digest.update(self._record(index))
            actual = digest.hexdigest()
        if actual != self.digest:
            raise ValueError("DDR snapshot checksum mismatch: payload")
        return self

    def close(self):
        store = getattr(self, "_store", None)
        if store is not None:
            store.close()
            self._store = None
        # Existing returned NumPy views keep their backing bytes/mmap alive.
        self._legacy = None
        if hasattr(self, "_cache"):
            self._cache.clear()
            self._cache_bytes = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()


def _load(directory, manifest, *, verify_payload):
    if manifest.get("schema_version") not in (1, 2) or manifest.get("transport") != "ubmem":
        raise ValueError("unsupported DDR snapshot schema/transport")
    _page_policy(manifest["page_mode"], manifest["numa_node"], manifest["generation"])
    info = _geometry(
        *(manifest[name] for name in ("region_base", "region_bytes", "slot_bytes", "token_bytes", "tokens_per_block"))
    )
    paths = {}
    from kvip.diagnostics.archive import checked_path

    for name, filename in _FILES.items():
        if name == "payload":
            continue
        entry = manifest["files"][name]
        if entry["path"] != filename:
            raise ValueError("unexpected DDR snapshot file path")
        path = checked_path(directory, filename)
        if _hash(path) != entry["sha256"]:
            raise ValueError(f"DDR snapshot checksum mismatch: {name}")
        paths[name] = path
    arrays = {name: np.load(paths[name], allow_pickle=False) for name in _FILES if name != "payload"}
    _validate_records(info, arrays["keys"], arrays["addresses"], arrays["slot_offsets"], arrays["valid_lengths"])
    expected = len(arrays["keys"]) * info["token_bytes"] * info["tokens_per_block"]
    if manifest["record_count"] != len(arrays["keys"]) or manifest["payload_bytes"] != expected:
        raise ValueError("DDR snapshot record/payload count mismatch")
    if verify_payload and manifest["schema_version"] == 1:
        entry = manifest["files"]["payload"]
        if entry["path"] != _FILES["payload"]:
            raise ValueError("unexpected DDR snapshot file path")
        if _hash(checked_path(directory, entry["path"])) != entry["sha256"]:
            raise ValueError("DDR snapshot checksum mismatch: payload")
    reader = PayloadReader(directory, manifest)
    if verify_payload and manifest["schema_version"] == 2:
        try:
            reader.verify()
        except BaseException:
            reader.close()
            raise
    return info, arrays, reader


def validate_pages(directory, manifest):
    """Verify all files and return a printable summary without native addresses."""
    info, arrays, reader = _load(directory, manifest, verify_payload=True)
    reader.close()
    return {
        **{name: value for name, value in info.items() if name != "region_base"},
        **_page_policy(manifest["page_mode"], manifest["numa_node"], manifest["generation"]),
        "record_count": len(arrays["keys"]),
        "payload_bytes": manifest["payload_bytes"],
        "checksums_validated": True,
    }


def import_pages(runtime, directory, manifest, *, target_transport="ubmem", chunk_bytes=CHUNK_BYTES, torch_module=None):
    """Restore into an already initialized, fresh Runtime. Caller releases the returned handle.

    The source manifest remains UBmem. Explicit ``target_transport="urma"``
    allows restoring it through the original URMA binding, whose Runtime has
    no transport property. All files are verified before Put. On failure discard
    the replay Runtime; never free/reuse slots after unknown transfer completion.
    """
    old, arrays, payload = _load(directory, manifest, verify_payload=True)
    current = _runtime_geometry(runtime, target_transport)
    if any(old[name] != current[name] for name in old if name != "region_base"):
        raise ValueError("replay Runtime must preserve arena size, slot size and block geometry")
    per_chunk = _chunk_records(old, chunk_bytes)
    block = old["token_bytes"] * old["tokens_per_block"]
    keys = arrays["keys"]
    torch = _torch(torch_module)
    device = torch.device(f"npu:{runtime.device_id}")
    with payload:
        for first in range(0, len(keys), per_chunk):
            stop = min(first + per_chunk, len(keys))
            data = payload.read_records(first, stop)
            tensor = torch.from_numpy(np.frombuffer(data, dtype=np.uint8).copy()).to(device)
            runtime.batch_put(
                keys[first:stop].tolist(),
                tensor,
                [i * block for i in range(stop - first)],
                arrays["valid_lengths"][first:stop].tolist(),
            )
    handle = None
    addresses = np.empty(0, dtype=np.uint64)
    try:
        if len(keys):
            handle = runtime.prepare_access_table(keys.tolist())
            addresses, offsets, _ = _read_table(runtime, handle, keys, current)
            if not np.array_equal(offsets, arrays["slot_offsets"]):
                raise ValueError("restored DDR offsets differ from the snapshot")
        return {
            "old_base": old["region_base"],
            "new_base": current["region_base"],
            "access_table": handle,
            "addresses": addresses,
        }
    except BaseException:
        if handle is not None:
            runtime.release_access_table(handle)
        raise


def materialize_pages(source_directory, manifest, destination):
    """Verify and write an independent legacy DDR payload for one series item."""
    info, _, reader = _load(source_directory, manifest, verify_payload=True)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError("materialized DDR directory must be empty")
    per_chunk = _chunk_records(info, CHUNK_BYTES)
    digest = hashlib.sha256()
    with reader, (destination / _FILES["payload"]).open("xb") as output:
        for first in range(0, reader.record_count, per_chunk):
            data = reader.read_records(first, min(first + per_chunk, reader.record_count))
            output.write(data)
            digest.update(data)
    if digest.hexdigest() != reader.digest:
        raise ValueError("DDR payload changed during materialization")
    result = copy.deepcopy(manifest)
    result["schema_version"] = 1
    result.pop("payload", None)
    result["files"]["payload"] = {"path": _FILES["payload"], "sha256": digest.hexdigest()}
    for name, filename in _FILES.items():
        if name != "payload":
            shutil.copyfile(Path(source_directory) / filename, destination / filename)
            if _hash(destination / filename) != result["files"][name]["sha256"]:
                raise ValueError("DDR metadata changed during materialization")
    return result


def rebase_addresses(addresses, directory, manifest, new_base, *, width=1):
    """Rebase only valid captured addresses, preserving zeros, shape and int64 bits.

    ``import_pages`` verifies the large payload once. This function verifies
    metadata checksums and ranges; it does not reread payload for every tensor.
    """
    info, arrays, reader = _load(directory, manifest, verify_payload=False)
    reader.close()
    new_base = _integer(new_base, "new_base", 1)
    width = _integer(width, "width", 1)
    if new_base % 32 or new_base + info["region_bytes"] > _U64_MAX:
        raise ValueError("relocated arena address range is invalid")
    original = np.asarray(addresses)
    values = _addresses(original)
    result = np.zeros(values.shape, dtype=np.uint64)
    live = values.ravel() != 0
    selected = values.ravel()[live]
    if selected.size:
        base = np.uint64(info["region_base"])
        end = np.uint64(info["region_base"] + info["region_bytes"])
        if np.any(selected < base) or np.any(selected >= end):
            raise ValueError("captured address is outside the original arena")
        # Subtract only after checking the unsigned address range. Signed int64
        # comparison would corrupt addresses with their high bit set.
        offsets = selected - base
        slots, within = np.divmod(offsets, np.uint64(info["slot_bytes"]))
        order = np.argsort(arrays["keys"])
        keys = arrays["keys"][order].astype(np.uint64)
        positions = np.searchsorted(keys, slots)
        if not len(keys) or np.any(positions >= len(keys)):
            raise ValueError("captured address refers to a missing DDR slot")
        lengths = arrays["valid_lengths"][order][positions]
        if np.any(keys[positions] != slots) or np.any(lengths < np.uint64(width)):
            raise ValueError("captured address is missing or exceeds its valid payload")
        if np.any(within > lengths - np.uint64(width)):
            raise ValueError("captured address is padding or exceeds its valid payload")
        result.ravel()[live] = np.uint64(new_base) + offsets
    return result.view(np.int64) if original.dtype.kind == "i" else result
