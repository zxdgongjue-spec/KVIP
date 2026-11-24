# SPDX-License-Identifier: Apache-2.0
"""Opt-in graph-safe capture of the packed operands of one LIFetch group."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np


class LIFetchSnapshot:
    def __init__(self, selector, config):
        self.selector = selector
        self.windows = config.intermediate_fetches
        self.transport = config.transport
        self.buffers = {}

    def capture(self):
        selector = self.selector
        torch = selector.torch
        rotating, inputs = selector.rotating, selector.inputs
        values = {
            "query": inputs.query.view(torch.uint8),
            "weights": inputs.weights.view(torch.int16),
            "query_scale": inputs.scale,
            "query_ends": rotating.query_ends,
            "tail_bounds": rotating.tail_bounds,
            "lru_slots": rotating.lru_slots,
            "reserved_hot": rotating.reserved_hot,
            "owners": inputs.row_owners,
        }
        owner_values = {
            "keys": selector.storage.keys.view(torch.uint8),
            "scales": selector.storage.scales,
            "addresses": selector.storage.addresses,
            "logical_ids": selector.storage.logical_ids,
            "hot_valid": selector.storage.hot_valid,
        }
        owners = inputs.row_owners.to(torch.int64)
        for name, value in values.items():
            if name not in self.buffers:
                self.buffers[name] = torch.empty_like(value)
            self.buffers[name].copy_(value)
        for name, value in owner_values.items():
            if name not in self.buffers:
                self.buffers[name] = torch.empty(
                    (inputs.batch, *value.shape[1:]), device=value.device, dtype=value.dtype
                )
            torch.index_select(value, 0, owners, out=self.buffers[name])

    def save(self, directory, *, group, active_batch):
        if not self.buffers:
            raise RuntimeError("No LIFetch operands captured for the selected group")
        selector = self.selector
        rotating = selector.rotating
        shape = getattr(rotating, "runtime_shape", None)
        if shape is None:
            batch = rotating.shape.batch
            token_bytes = rotating.shape.token_bytes
            aics = rotating.shape.aic_count
            windows = rotating.shape.intermediate_fetches
            resident = rotating.shape.resident
        else:
            batch, token_bytes, aics = shape.batch, shape.tokenBytes, shape.aicCount
            windows, resident = shape.rounds, shape.resident
        geometry = dict(
            batch=batch, active_batch=active_batch, history=selector.history,
            resident=resident, heads=selector.inputs.index_heads, token_bytes=token_bytes,
            output_stride=rotating.output_stride, windows=self.windows, actual_windows=windows,
            aics=aics, group=group, transport=self.transport,
            prefetch_tokens=rotating.prefetch_limit,
            prefetch_rank_start=getattr(rotating, "prefetch_rank_start", 1024),
        )
        directory = Path(directory)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, prefix=f".lifetch-{os.getpid()}-",
                                         suffix=".npz", delete=False) as output:
            path = Path(output.name)
            try:
                arrays = {name: tensor.cpu().numpy() for name, tensor in self.buffers.items()}
                # Store relative DDR page identities; service virtual addresses
                # are neither needed nor valid in the replay Runtime.
                address = arrays.pop("addresses").view(np.uint64)
                logical = arrays["logical_ids"].astype(np.int64, copy=False)
                valid = address != 0
                pages = np.zeros_like(address)
                pages[valid] = address[valid] - (logical[valid] % 128).astype(np.uint64) * token_bytes
                _, inverse = np.unique(pages[valid], return_inverse=True)
                aliases = np.full(address.shape, -1, dtype=np.int32)
                aliases[valid] = inverse
                arrays["page_aliases"] = aliases
                arrays["completion"] = selector.update.status.cpu().numpy()
                np.savez(output, schema_version=1, geometry=json.dumps(geometry), **arrays)
            except BaseException:
                path.unlink(missing_ok=True)
                raise
        result = path.with_name(path.name.removeprefix("."))
        path.rename(result)
        return result
