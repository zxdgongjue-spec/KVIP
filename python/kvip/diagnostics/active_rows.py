# SPDX-License-Identifier: Apache-2.0
"""Read-only replay views with fewer active Q4 requests at the original capacity.

Validate/inspect the original service archive before deriving this view. It is
an offline input transformation, not another service capture. DDR files, arena
geometry and every resident owner's pre-state remain owned by the source.
"""

import copy

import numpy as np

from kvip.diagnostics.archive import TensorArchive, file_digest


class ActiveRowsArchive(TensorArchive):
    """An in-memory overlay; source file checksums still verify source files."""

    def __init__(self, source, arrays, selection):
        manifest = copy.deepcopy(source.manifest)
        manifest["replay_selection"] = selection
        super().__init__(source.directory, manifest)
        self.source = source
        self._arrays = {}
        for name, value in arrays.items():
            value = np.ascontiguousarray(value)
            value.flags.writeable = False
            self._arrays[name] = value
            # Keep the source filename/hash as provenance, not a claim that
            # filtered tensors have been written back to those files.
            description = self.manifest["tensors"][name]
            description.update(shape=list(value.shape), nbytes=value.nbytes, replay_overlay=True)

    def array(self, name, *, verify=False):
        if name not in self._arrays:
            return self.source.array(name, verify=verify)
        if verify:
            self.source.array(name, verify=True)
        return self._arrays[name]

    def save(self, *args, **kwargs):
        raise RuntimeError("ActiveRowsArchive is a read-only replay view")


def mask_active_rows(archive, n):
    """Keep the first n real packed requests; preserve B-capacity and TND order.

    The caller first inspects the original archive. Query lengths remain four
    and key lengths remain H: only query validity/ends mask inactive LI work.
    This lets baseline and skip-enabled kernels consume exactly the same view.
    """
    if "replay_selection" in archive.manifest:
        raise ValueError("Select active rows from the original service snapshot")
    geometry = archive.manifest["geometry"]
    batch, history, resident = (geometry[name] for name in ("batch", "history", "resident"))
    if type(n) is not int or not 1 <= n <= batch:
        raise ValueError("active row count must be an integer in 1..batch capacity")
    if geometry["token_count"] != batch * 4:
        raise ValueError("Active-row replay requires the original fixed Q4 capacity")
    metadata = archive.array("pre.metadata").copy()
    ends = archive.array("pre.query_ends").copy()
    if metadata.dtype != np.int64 or metadata.shape != (5, batch * 4) or ends.size != batch * 4:
        raise ValueError("Snapshot packed Q4 metadata geometry differs")
    if not np.all(metadata[1] == 1) or not np.all(metadata[3] == 1) or np.any(ends <= 0):
        raise ValueError("Derive active rows only from a fully active real capture")
    owners = metadata[4].view(np.int32)[:batch].copy()
    hot = archive.array("pre.hot_valid")
    addresses = archive.array("pre.addresses")
    if (
        np.any(owners <= 0)
        or np.any(owners >= len(hot))
        or len(np.unique(owners)) != batch
        or np.any(hot[0])
        or np.any(addresses[0])
    ):
        raise ValueError("Padding requires unused guard owner 0 with no valid hot or DDR entries")
    unpack = metadata[2].copy()
    if not np.array_equal(np.sort(unpack), np.arange(batch * 4)):
        raise ValueError("Full-capture TND unpack must name every packed query exactly once")
    real_rows = np.flatnonzero(np.any(ends.reshape(batch, 4) > 0, axis=1))
    active_rows = real_rows[:n]
    packed_active = np.zeros(batch, dtype=bool)
    packed_active[active_rows] = True
    output_active = packed_active[unpack // 4]
    query_active = np.repeat(packed_active, 4)
    metadata[0, ~query_active] = 0
    metadata[1, ~query_active] = 0
    metadata[2, ~output_active] = 0
    metadata[3, ~output_active] = 0
    metadata[4].view(np.int32)[:batch][~packed_active] = 0
    ends.reshape(batch, 4)[~packed_active] = 0
    arrays = {"pre.metadata": metadata, "pre.query_ends": ends}

    blocks = archive.array("pre.block_table").copy()
    if history % 128 or blocks.shape != (batch, history // 128):
        raise ValueError("Index block table differs from captured owner/history geometry")
    offsets = np.arange(history // 128, dtype=blocks.dtype)
    if not np.array_equal(blocks, owners[:, None] * len(offsets) + offsets):
        raise ValueError("Index block table does not address the captured physical owners")
    blocks[~packed_active] = offsets
    arrays["pre.block_table"] = blocks

    candidates = archive.array("pre.lru_slots").copy()
    reserved = archive.array("pre.reserved_hot").copy()
    bounds = archive.array("pre.tail_bounds").copy()
    if candidates.shape != (batch, 2048) or reserved.shape != (batch, resident) or bounds.shape != (batch, 2):
        raise ValueError("Snapshot LRU/reserved/ABC metadata geometry differs")
    if resident < candidates.shape[1]:
        raise ValueError("Padding needs 2048 distinct legal LRU candidates")
    ordinals = np.arange(candidates.shape[1], dtype=candidates.dtype)
    candidates[~packed_active] = ordinals
    reserved[~packed_active] = -1
    reserved[np.flatnonzero(~packed_active), : len(ordinals)] = ordinals
    bounds[~packed_active] = 0
    arrays.update({"pre.lru_slots": candidates, "pre.reserved_hot": reserved, "pre.tail_bounds": bounds})
    # Packed query/weights/scale bytes remain untouched. They are already
    # gathered inputs, and query_ends=0 makes inactive bytes semantically inert.
    for name in ("logical_output", "slot_output"):
        values = archive.array("post." + name).copy()
        values.reshape(batch * 4, 2048)[~output_active] = -1
        arrays["post." + name] = values

    references = archive.array("post.selected_references").copy()
    if references.shape != (batch * 4, 2048):
        raise ValueError("Selected payload references differ from TND geometry")
    tags, payload = (archive.array("post." + name) for name in ("selected_tags", "selected_payload"))
    retained = np.unique(references[output_active])
    if tags.shape != (len(payload), 2) or np.any(retained < 0) or np.any(retained >= len(tags)):
        raise ValueError("Selected payload references are outside captured records")
    if not np.all(np.isin(tags[retained, 0], owners[active_rows])):
        raise ValueError("Active outputs refer to another physical owner's payload")
    remap = np.full(len(tags), -1, dtype=np.int64)
    remap[retained] = np.arange(len(retained))
    references[output_active] = remap[references[output_active]]
    references[~output_active] = -1
    arrays.update(
        {
            "post.selected_tags": tags[retained].copy(),
            "post.selected_payload": payload[retained].copy(),
            "post.selected_references": references,
        }
    )
    for name in ("logical_ids", "hot_valid"):
        key = "post." + name
        if key in archive.manifest["tensors"]:
            values = archive.array(key).copy()
            values[owners[~packed_active]] = archive.array("pre." + name)[owners[~packed_active]]
            arrays[key] = values
    for name in ("prefetch_records", "final_records"):
        key = "post." + name
        if key in archive.manifest["tensors"]:
            values = archive.array(key)
            arrays[key] = values[np.isin(values[:, 0], active_rows)].copy()
    selection = dict(
        kind="masked_real_snapshot",
        batch_capacity=batch,
        active_count=n,
        active_rows=active_rows.tolist(),
        owners=owners[active_rows].tolist(),
        guard_owner=0,
        source={
            "directory": str(archive.directory),
            "manifest_sha256": file_digest(archive.directory / "manifest.json"),
        },
        policy="Original B/H/owner storage and DDR; unchanged active packed inputs; inactive Q4 and TND outputs masked.",
        evidence="Original oracle/counters describe the source capture; derived replay must pass its own oracle.",
    )
    return ActiveRowsArchive(archive, arrays, selection)
