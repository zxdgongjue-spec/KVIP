# SPDX-License-Identifier: Apache-2.0
"""Validate logical selection, every selected KV byte and persistent Index state."""

import numpy as np

from kvip.diagnostics.archive import array_digest, checked_path


class DecodeOracle:
    def __init__(self, archive):
        self.archive = archive
        self.geometry = archive.manifest["geometry"]
        self.pre = {
            name: archive.array("pre." + name)
            for name in (
                "keys",
                "key_scales",
                "addresses",
                "logical_ids",
                "hot_valid",
                "output",
                "metadata",
                "tail_bounds",
                "query_ends",
            )
        }
        self.batch = self.geometry["batch"]
        self.history = self.geometry["history"]
        self.resident = self.geometry["resident"]
        self.width = self.geometry["token_bytes"]
        self.stride = self.geometry["output_stride"]
        self.owner_count = self.pre["logical_ids"].shape[0]
        self.owners = self.pre["metadata"][4].view(np.int32)[: self.batch]
        self.output = (
            self.pre["output"]
            .reshape(-1, self.width)[: self.owner_count * self.stride]
            .reshape(self.owner_count, self.stride, self.width)
        )
        self.inverse = np.empty_like(self.pre["logical_ids"])
        identity = np.arange(self.history)
        for owner, ids in enumerate(self.pre["logical_ids"]):
            if not np.array_equal(np.sort(ids), identity):
                raise ValueError(f"Pre Index is not a complete permutation: owner {owner}")
            self.inverse[owner, ids] = identity
        self.ddr = archive.manifest["ddr"]
        folder = archive.directory / "ddr"
        files = self.ddr["files"]
        self.ddr_keys = np.load(checked_path(folder, files["keys"]["path"]), allow_pickle=False)
        self.lengths = np.load(checked_path(folder, files["valid_lengths"]["path"]), allow_pickle=False)
        self.key_to_record = {int(key): row for row, key in enumerate(self.ddr_keys)}
        self.block_bytes = self.width * self.ddr["tokens_per_block"]
        self.payload = np.memmap(checked_path(folder, files["payload"]["path"]), mode="r", dtype=np.uint8).reshape(
            len(self.ddr_keys), self.block_bytes
        )

    def expected(self, owner, logical):
        """Use pre-round hot/ABC bytes or the complete committed DDR snapshot."""
        logical = np.asarray(logical, dtype=np.int64).reshape(-1)
        if np.any(logical < 0) or np.any(logical >= self.history):
            raise AssertionError("Logical output is outside Index history")
        positions = self.inverse[owner, logical]
        result = np.empty((len(logical), self.width), dtype=np.uint8)
        remaining = np.ones(len(logical), dtype=bool)
        for packed in np.flatnonzero(self.owners == owner):
            begin, end = self.pre["tail_bounds"][packed]
            tail = remaining & (logical >= begin) & (logical < end)
            tail_tokens = self.stride - self.resident
            if np.any(tail):
                if tail_tokens < 384:
                    raise AssertionError("ABC output has no valid tail capacity")
                result[tail] = self.output[owner, self.resident + logical[tail] % tail_tokens]
                remaining[tail] = False
        hot = remaining & (positions < self.resident)
        hot &= self.pre["hot_valid"][owner, np.minimum(positions, self.resident - 1)] != 0
        result[hot] = self.output[owner, positions[hot]]
        remaining[hot] = False
        for index in np.flatnonzero(remaining):
            address = int(self.pre["addresses"].view(np.uint64)[owner, positions[index]])
            offset = address - self.ddr["region_base"]
            if not address or offset < 0:
                raise AssertionError(f"Selected token has no pre hot/ABC/DDR bytes: owner={owner}, id={logical[index]}")
            key, within = divmod(offset, self.ddr["slot_bytes"])
            record = self.key_to_record.get(key)
            if record is None or within + self.width > int(self.lengths[record]):
                raise AssertionError("Selected DDR range is absent from the exported valid pages")
            result[index] = self.payload[record, within : within + self.width]
        return result

    def verify(self, state, *, address_delta=0, expected_topk=None):
        logical = state["logical_output"].reshape(self.batch * 4, 2048)
        slots = state["slot_output"].reshape(self.batch * 4, 2048)
        ids = state["logical_ids"]
        output = (
            state["output"]
            .reshape(-1, self.width)[: self.owner_count * self.stride]
            .reshape(self.owner_count, self.stride, self.width)
        )
        identity = np.arange(self.history)
        for owner in range(self.owner_count):
            if not np.array_equal(np.sort(ids[owner]), identity):
                raise AssertionError(f"Post Index lost its permutation: owner={owner}")
            source = self.inverse[owner, ids[owner]]
            for name in ("keys", "key_scales"):
                wanted = np.ascontiguousarray(self.pre[name][owner, source])
                if not np.array_equal(np.ascontiguousarray(state[name][owner]).view(np.uint8), wanted.view(np.uint8)):
                    raise AssertionError(f"Index {name} detached from logical IDs: owner={owner}")
            wanted = self.pre["addresses"][owner, source].view(np.uint64).copy()
            if owner in self.owners:
                mask = wanted != 0
                old_base = self.ddr["region_base"]
                new_base = old_base + address_delta
                if not 0 < new_base < (1 << 64):
                    raise ValueError("Relocated arena base is outside uint64")
                wanted[mask] = wanted[mask] - np.uint64(old_base) + np.uint64(new_base)
            wanted = wanted.view(self.pre["addresses"].dtype)
            if not np.array_equal(state["addresses"][owner], wanted):
                raise AssertionError(f"Index address detached from logical IDs: owner={owner}")
            hot = np.flatnonzero(state["hot_valid"][owner] != 0)
            for start in range(0, len(hot), 256):
                selected = hot[start : start + 256]
                if not np.array_equal(output[owner, selected], self.expected(owner, ids[owner, selected])):
                    raise AssertionError(f"Valid hot KV bytes disagree with Index: owner={owner}")
        metadata = self.pre["metadata"]
        active = metadata[3] != 0
        if np.any(logical[~active] != -1) or np.any(slots[~active] != -1):
            raise AssertionError("Padding outputs are not -1")
        if expected_topk is not None and not np.array_equal(np.sort(logical, axis=-1), np.sort(expected_topk, axis=-1)):
            raise AssertionError("Logical TopK differs from the captured service selection")
        records, payloads, references = {}, [], np.full(logical.shape, -1, dtype=np.int64)
        tags = []
        for row in np.flatnonzero(active):
            source = int(metadata[2, row])
            if not 0 <= source < self.batch * 4:
                raise AssertionError("TND source row is outside packed Q4")
            owner = int(self.owners[source // 4])
            query_end = int(self.pre["query_ends"].reshape(-1)[source])
            if not 0 < query_end <= self.history or np.any(logical[row] < 0) or np.any(logical[row] >= query_end):
                raise AssertionError("Logical TopK violates its packed query causal boundary")
            if len(np.unique(logical[row])) != 2048:
                raise AssertionError("Valid query does not contain 2048 unique logical selections")
            if np.any(slots[row] < 0) or np.any(slots[row] >= self.stride):
                raise AssertionError("Selected slot is outside the registered owner output")
            hot_slots = slots[row] < self.resident
            if np.any(state["hot_valid"][owner, slots[row, hot_slots]] == 0) or not np.array_equal(
                ids[owner, slots[row, hot_slots]], logical[row, hot_slots]
            ):
                raise AssertionError("Selected hot slot does not publish the selected logical token")
            expected = self.expected(owner, logical[row])
            actual = output[owner, slots[row]]
            if not np.array_equal(actual, expected):
                raise AssertionError(f"Selected KV bytes mismatch: TND row={row}, owner={owner}")
            for k, token in enumerate(logical[row]):
                key = owner, int(token)
                if key not in records:
                    records[key] = len(payloads)
                    payloads.append(actual[k].copy())
                    tags.append(key)
                references[row, k] = records[key]
        return {
            "selected_tags": np.asarray(tags, dtype=np.int64).reshape(-1, 2),
            "selected_payload": np.asarray(payloads, dtype=np.uint8).reshape(-1, self.width),
            "selected_references": references,
            "index_hashes": {
                name: array_digest(state[name])
                for name in ("keys", "key_scales", "addresses", "logical_ids", "hot_valid")
            },
        }
