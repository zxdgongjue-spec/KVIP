# SPDX-License-Identifier: Apache-2.0
"""Bit-exact tensor files without pickle or serialized device pointers."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from kvip.diagnostics.storage import PackedObjectStore, TENSOR_CHUNK_BYTES


NUMPY_DTYPES = {
    "uint8": "u1",
    "int8": "i1",
    "bool": "?",
    "int16": "<i2",
    "uint16": "<u2",
    "int32": "<i4",
    "uint32": "<u4",
    "int64": "<i8",
    "uint64": "<u8",
    "float16": "<f2",
    "float32": "<f4",
    "float64": "<f8",
    "bfloat16": "<u2",
    "float8_e4m3fn": "u1",
}


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def source_commit():
    """Diagnostic provenance only; installed wheels may have no Git checkout."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def checked_path(directory, name):
    root = Path(directory).resolve()
    path = (root / name).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Snapshot path escapes its directory: {name}")
    return path


def cpu_array(torch, tensor):
    dtype = str(tensor.dtype).removeprefix("torch.")
    if dtype not in NUMPY_DTYPES:
        raise ValueError(f"Unsupported snapshot tensor dtype: {dtype}")
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
    return raw.view(NUMPY_DTYPES[dtype]).reshape(tuple(tensor.shape))


class TensorArchive:
    def __init__(self, directory, manifest, *, object_store=None):
        self.directory, self.manifest = Path(directory), manifest
        self.object_store = object_store
        self._owns_store = False
        if object_store is not None:
            description = object_store.describe_for(self.directory)
            if manifest.get("object_store", description) != description:
                raise ValueError("Tensor archive object store differs from its manifest")
            self.manifest["object_store"] = description
        elif "object_store" in manifest:
            self.object_store = PackedObjectStore.from_description(self.directory, manifest["object_store"])
            self._owns_store = True

    def save(self, name, value, *, dtype=None):
        if name in self.manifest.setdefault("tensors", {}):
            raise FileExistsError(f"Snapshot tensor already exists: {name}")
        relative = f"tensors/{name}.npy"
        path = checked_path(self.directory, relative)
        value = np.ascontiguousarray(value)
        if value.dtype.hasobject:
            raise ValueError("Object arrays are forbidden in snapshots")
        dtype = dtype or value.dtype.name
        if dtype not in NUMPY_DTYPES:
            raise ValueError(f"Unsupported snapshot dtype: {dtype}")
        if value.dtype.itemsize != np.dtype(NUMPY_DTYPES[dtype]).itemsize:
            raise ValueError("Snapshot dtype width differs from its data")
        if self.object_store is not None:
            raw = memoryview(value.view(np.uint8).reshape(-1))
            chunks = [
                self.object_store.put(raw[start : start + TENSOR_CHUNK_BYTES])
                for start in range(0, value.nbytes, TENSOR_CHUNK_BYTES)
            ]
            self.manifest["tensors"][name] = dict(
                storage="cas_chunks",
                dtype=dtype,
                shape=list(value.shape),
                nbytes=value.nbytes,
                sha256=hashlib.sha256(raw).hexdigest(),
                chunk_bytes=TENSOR_CHUNK_BYTES,
                chunks=chunks,
            )
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            np.save(output, value.view(np.uint8).reshape(-1), allow_pickle=False)
        self.manifest["tensors"][name] = dict(
            file=relative, dtype=dtype, shape=list(value.shape), nbytes=value.nbytes, sha256=file_digest(path)
        )

    def save_tensor(self, name, torch, tensor):
        self.save(name, cpu_array(torch, tensor), dtype=str(tensor.dtype).removeprefix("torch."))

    def _layout(self, name):
        description = self.manifest["tensors"][name]
        dtype = np.dtype(NUMPY_DTYPES[description["dtype"]])
        shape = tuple(description["shape"])
        if any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"Invalid snapshot shape: {name}")
        size = math.prod(shape) * dtype.itemsize
        if type(description["nbytes"]) is not int or size != description["nbytes"]:
            raise ValueError(f"Snapshot size/dtype mismatch: {name}")
        return description, dtype, shape, size

    def _chunks(self, name, description, size):
        chunks = description.get("chunks")
        if description.get("storage") != "cas_chunks" or self.object_store is None or not isinstance(chunks, list):
            raise ValueError(f"Invalid tensor object storage: {name}")
        chunk_bytes = description.get("chunk_bytes")
        if type(chunk_bytes) is not int or not 0 < chunk_bytes <= (32 << 20):
            raise ValueError(f"Invalid tensor chunk size: {name}")
        total = 0
        for i, reference in enumerate(chunks):
            length = reference.get("length") if isinstance(reference, dict) else None
            if type(length) is not int or length <= 0 or length > chunk_bytes:
                raise ValueError(f"Invalid tensor chunk length: {name}")
            if i < len(chunks) - 1 and length != chunk_bytes:
                raise ValueError(f"Invalid nonfinal tensor chunk length: {name}")
            total += length
        if total != size:
            raise ValueError(f"Snapshot chunk sizes differ from its tensor: {name}")
        return chunks

    def array(self, name, *, verify=False):
        description, dtype, shape, size = self._layout(name)
        if "storage" in description:
            chunks = self._chunks(name, description, size)
            raw = np.empty(size, dtype=np.uint8)
            start = 0
            digest = hashlib.sha256() if verify else None
            for reference in chunks:
                data = self.object_store.read(reference, verify=verify)
                raw[start : start + len(data)] = np.frombuffer(data, dtype=np.uint8)
                start += len(data)
                if digest is not None:
                    digest.update(data)
            if digest is not None and digest.hexdigest() != description["sha256"]:
                raise ValueError(f"Snapshot checksum mismatch: {name}")
            return raw.view(dtype).reshape(shape)
        path = checked_path(self.directory, description["file"])
        if verify and file_digest(path) != description["sha256"]:
            raise ValueError(f"Snapshot checksum mismatch: {name}")
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        if raw.dtype != np.uint8 or raw.ndim != 1 or raw.nbytes != size:
            raise ValueError(f"Snapshot size/dtype mismatch: {name}")
        return raw.view(dtype).reshape(shape)

    def tensor(self, name, torch, device):
        description = self.manifest["tensors"][name]
        raw = np.asarray(self.array(name)).view(np.uint8).reshape(-1).copy()
        return torch.from_numpy(raw).to(device).view(getattr(torch, description["dtype"])).reshape(description["shape"])

    def verify(self):
        for name in self.manifest["tensors"]:
            description, _, _, size = self._layout(name)
            if "storage" not in description:
                self.array(name, verify=True)
                continue
            digest = hashlib.sha256()
            for reference in self._chunks(name, description, size):
                digest.update(self.object_store.read(reference, verify=True))
            if digest.hexdigest() != description["sha256"]:
                raise ValueError(f"Snapshot checksum mismatch: {name}")

    def close(self):
        if self._owns_store:
            self.object_store.close()
            self._owns_store = False


def load_snapshot(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") not in (1, 2) or manifest.get("status") != "complete":
        raise ValueError("Snapshot is incomplete or has an unsupported schema")
    archive = TensorArchive(directory, manifest)
    try:
        archive.verify()
    except BaseException:
        archive.close()
        raise
    return archive
