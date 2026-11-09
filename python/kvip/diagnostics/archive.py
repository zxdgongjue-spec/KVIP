# SPDX-License-Identifier: Apache-2.0
"""Bit-exact tensor files without pickle or serialized device pointers."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


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
    def __init__(self, directory, manifest):
        self.directory, self.manifest = Path(directory), manifest

    def save(self, name, value, *, dtype=None):
        value = np.ascontiguousarray(value)
        if value.dtype.hasobject:
            raise ValueError("Object arrays are forbidden in snapshots")
        dtype = dtype or value.dtype.name
        if dtype not in NUMPY_DTYPES:
            raise ValueError(f"Unsupported snapshot dtype: {dtype}")
        if value.dtype.itemsize != np.dtype(NUMPY_DTYPES[dtype]).itemsize:
            raise ValueError("Snapshot dtype width differs from its data")
        relative = f"tensors/{name}.npy"
        path = checked_path(self.directory, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            np.save(output, value.view(np.uint8).reshape(-1), allow_pickle=False)
        self.manifest.setdefault("tensors", {})[name] = dict(
            file=relative, dtype=dtype, shape=list(value.shape), nbytes=value.nbytes, sha256=file_digest(path)
        )

    def save_tensor(self, name, torch, tensor):
        self.save(name, cpu_array(torch, tensor), dtype=str(tensor.dtype).removeprefix("torch."))

    def array(self, name, *, verify=False):
        description = self.manifest["tensors"][name]
        path = checked_path(self.directory, description["file"])
        if verify and file_digest(path) != description["sha256"]:
            raise ValueError(f"Snapshot checksum mismatch: {name}")
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        dtype = np.dtype(NUMPY_DTYPES[description["dtype"]])
        shape = tuple(description["shape"])
        if any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"Invalid snapshot shape: {name}")
        size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if raw.dtype != np.uint8 or raw.ndim != 1 or raw.nbytes != size or size != description["nbytes"]:
            raise ValueError(f"Snapshot size/dtype mismatch: {name}")
        return raw.view(dtype).reshape(shape)

    def tensor(self, name, torch, device):
        description = self.manifest["tensors"][name]
        raw = np.asarray(self.array(name)).view(np.uint8).reshape(-1).copy()
        return torch.from_numpy(raw).to(device).view(getattr(torch, description["dtype"])).reshape(description["shape"])

    def verify(self):
        for name in self.manifest["tensors"]:
            self.array(name, verify=True)


def load_snapshot(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Snapshot is incomplete or has an unsupported schema")
    archive = TensorArchive(directory, manifest)
    archive.verify()
    return archive
