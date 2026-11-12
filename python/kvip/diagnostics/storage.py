# SPDX-License-Identifier: Apache-2.0
"""Append-only, lossless object storage for one DP rank's diagnostic series.

Payloads live in bounded pack files, not in the SQLite index or a RAM cache.
The single writer publishes immutable extents; snapshots contain their hashes
and offsets so readers do not need the writable deduplication index.
"""

from __future__ import annotations

from collections import OrderedDict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import uuid


PACK_BYTES = 256 << 20
TENSOR_CHUNK_BYTES = 64 << 10
_PACK = re.compile(r"pack-[0-9]{8}\.bin\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _number(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"Object {name} must be a nonnegative integer")
    return value


class PackedObjectStore:
    def __init__(self, directory, *, create=False):
        self.directory = Path(directory).resolve()
        self._writer = bool(create)
        self._index = self._lock = self._output = None
        self._readers = OrderedDict()
        self._recent = OrderedDict()
        self._pack_number = 0
        self._closed = False
        marker = self.directory / "store.json"
        if create:
            self.directory.mkdir(parents=True, exist_ok=True)
            if not marker.exists() and any(self.directory.iterdir()):
                raise FileExistsError("Object store directory contains unrelated or incomplete files")
            if not marker.exists():
                with marker.open("x") as target:
                    json.dump({"schema_version": 1, "store_id": uuid.uuid4().hex}, target, sort_keys=True)
                    target.write("\n")
                    target.flush()
                    os.fsync(target.fileno())
            (self.directory / "packs").mkdir(exist_ok=True)
        info = json.loads(marker.read_text())
        if info.get("schema_version") != 1 or not re.fullmatch(r"[0-9a-f]{32}", info.get("store_id", "")):
            raise ValueError("Unsupported or invalid diagnostic object store")
        self.store_id = info["store_id"]
        packs_directory = self.directory / "packs"
        if packs_directory.is_symlink() or not packs_directory.is_dir() or packs_directory.resolve() != packs_directory:
            raise ValueError("Object packs must remain inside their store")
        if create:
            try:
                if any((self.directory / name).is_symlink() for name in ("writer.lock", "index.sqlite3")):
                    raise ValueError("Object store writer files must not be symlinks")
                self._lock = (self.directory / "writer.lock").open("a+b")
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._index = sqlite3.connect(self.directory / "index.sqlite3")
                self._index.execute("PRAGMA journal_mode=WAL")
                self._index.execute("PRAGMA synchronous=FULL")
                self._index.execute("PRAGMA cache_size=-8192")
                self._index.execute(
                    "CREATE TABLE IF NOT EXISTS objects "
                    "(sha256 TEXT PRIMARY KEY, pack TEXT NOT NULL, offset INTEGER NOT NULL, length INTEGER NOT NULL)"
                )
                self._index.commit()
                packs = sorted((self.directory / "packs").glob("pack-*.bin"))
                if packs:
                    if not all(_PACK.fullmatch(path.name) for path in packs):
                        raise ValueError("Invalid object pack filename")
                    self._pack_number = int(packs[-1].stem.removeprefix("pack-"))
            except BaseException:
                self.close()
                raise

    def describe_for(self, snapshot_directory):
        root = Path(snapshot_directory).resolve()
        if self.directory.parent not in root.parents or self.directory in root.parents:
            raise ValueError("Object store must be beside an ancestor of the snapshot, not outside its series")
        return {
            "schema_version": 1,
            "relative_path": os.path.relpath(self.directory, root),
            "store_id": self.store_id,
        }

    @classmethod
    def from_description(cls, snapshot_directory, description):
        relative = description.get("relative_path")
        if description.get("schema_version") != 1 or not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("Invalid object store locator")
        root = Path(snapshot_directory).resolve()
        store = cls(root / relative)
        try:
            expected = store.describe_for(root)
            if description != expected:
                raise ValueError("Object store locator or identity differs from the snapshot")
        except BaseException:
            store.close()
            raise
        return store

    def _reference(self, reference):
        if self._closed:
            raise RuntimeError("Object store is closed")
        if not isinstance(reference, dict) or set(reference) != {"sha256", "pack", "offset", "length"}:
            raise ValueError("Invalid object reference fields")
        digest, pack = reference["sha256"], reference["pack"]
        if not isinstance(digest, str) or not _HASH.fullmatch(digest):
            raise ValueError("Invalid object checksum")
        if not isinstance(pack, str) or not _PACK.fullmatch(pack):
            raise ValueError("Invalid object pack path")
        offset, length = (_number(reference[key], key) for key in ("offset", "length"))
        path = self.directory / "packs" / pack
        if path.resolve().parent != self.directory / "packs" or path.is_symlink():
            raise ValueError("Object pack escapes its store")
        size = path.stat().st_size
        if offset > size or length > size - offset:
            raise ValueError("Object pack is truncated or reference is out of bounds")
        return path, offset, length

    def _remember(self, digest, reference):
        self._recent[digest] = reference
        self._recent.move_to_end(digest)
        if len(self._recent) > 8192:
            self._recent.popitem(last=False)

    def _open_output(self, length):
        if self._output is None:
            path = self.directory / "packs" / f"pack-{self._pack_number:08d}.bin"
            if path.is_symlink():
                raise ValueError("Object pack must not be a symlink")
            if path.exists() and path.stat().st_size and path.stat().st_size + length > PACK_BYTES:
                self._pack_number += 1
                path = self.directory / "packs" / f"pack-{self._pack_number:08d}.bin"
                if path.is_symlink():
                    raise ValueError("Object pack must not be a symlink")
            self._output = path.open("ab", buffering=0)
        if self._output.tell() and self._output.tell() + length > PACK_BYTES:
            os.fsync(self._output.fileno())
            self._output.close()
            self._output = None
            self._pack_number += 1
            self._open_output(length)

    def put(self, data):
        if not self._writer or self._closed:
            raise RuntimeError("Object store is not an open writer")
        raw = memoryview(data).cast("B")
        if raw.nbytes > PACK_BYTES:
            raise ValueError("Object exceeds the bounded pack size")
        digest = hashlib.sha256(raw).hexdigest()
        reference = self._recent.get(digest)
        if reference is None:
            row = self._index.execute("SELECT pack, offset, length FROM objects WHERE sha256=?", (digest,)).fetchone()
            if row:
                reference = dict(sha256=digest, pack=row[0], offset=row[1], length=row[2])
        if reference is not None:
            self._reference(reference)
            if reference["length"] != raw.nbytes:
                raise ValueError("Deduplication index has inconsistent object length")
            self._remember(digest, reference)
            return dict(reference)
        self._open_output(raw.nbytes)
        reference = dict(
            sha256=digest, pack=Path(self._output.name).name, offset=self._output.tell(), length=raw.nbytes
        )
        written = self._output.write(raw)
        if written != raw.nbytes:
            raise OSError("Incomplete diagnostic object write")
        self._index.execute(
            "INSERT INTO objects(sha256, pack, offset, length) VALUES(?,?,?,?)",
            (digest, reference["pack"], reference["offset"], reference["length"]),
        )
        self._remember(digest, reference)
        return dict(reference)

    def _reader(self, path):
        name = path.name
        if name not in self._readers:
            self._readers[name] = path.open("rb", buffering=0)
            if len(self._readers) > 8:
                _, old = self._readers.popitem(last=False)
                old.close()
        self._readers.move_to_end(name)
        return self._readers[name]

    def read(self, reference, *, offset=0, length=None, verify=True):
        path, start, size = self._reference(reference)
        offset = _number(offset, "read offset")
        length = size - offset if length is None else _number(length, "read length")
        if offset > size or length < 0 or length > size - offset:
            raise ValueError("Object read exceeds its immutable extent")
        source = self._reader(path)
        if verify:
            digest = hashlib.sha256()
            source.seek(start)
            remaining = size
            while remaining:
                block = source.read(min(1 << 20, remaining))
                if not block:
                    raise ValueError("Object pack changed or was truncated during verification")
                digest.update(block)
                remaining -= len(block)
            if digest.hexdigest() != reference["sha256"]:
                raise ValueError("Object checksum mismatch")
        source.seek(start + offset)
        data = source.read(length)
        if len(data) != length:
            raise ValueError("Object pack changed or was truncated during read")
        return data

    def flush(self):
        """Make object extents durable before publishing a complete manifest."""
        if self._closed:
            raise RuntimeError("Object store is closed")
        if self._output is not None:
            os.fsync(self._output.fileno())
        if self._index is not None:
            self._index.commit()

    def close(self):
        if self._closed:
            return
        try:
            self.flush()
        finally:
            if self._output is not None:
                self._output.close()
            for source in self._readers.values():
                source.close()
            if self._index is not None:
                self._index.close()
            if self._lock is not None:
                self._lock.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
