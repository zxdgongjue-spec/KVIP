# SPDX-License-Identifier: Apache-2.0
"""Bind static UBmem kernels to a retained KVAF mapping and service storage."""

import ctypes

import numpy as np

from kvip.operators.lifetch.geometry import final_aivs, li_geometry
from kvip.operators.lifetch_ubmem.specialization import query
from kvip.operators.lifetch_ubmem.workspace import device_words, rotation_spec


class UbmemBinding:
    def __init__(
        self,
        torch,
        library,
        arguments,
        lease,
        output,
        *,
        shape,
        query_ends,
        hot_valid,
        row_owners,
        token_addresses,
        tail_bounds=None,
        output_stride=None,
        scratch=None,
        prefetch_limit=2048,
        window_limit=480,
    ):
        self.torch, self.lease, self.shape = torch, lease, shape
        self.library = ctypes.CDLL(str(library))
        self.device = arguments[0].device
        if len(arguments) != 14 or self.device.type != "npu" or self.device != lease.device:
            raise ValueError("UBmem requires 14 input buffers on the leased NPU")
        if any(t.device != self.device or not t.is_contiguous() for t in arguments):
            raise ValueError("UBmem inputs must be contiguous on one NPU")
        for name, expected in (
            ("batch", shape.batch),
            ("history", shape.history),
            ("resident", shape.resident),
            ("token_bytes", shape.token_bytes),
            ("aic_count", shape.aic_count),
            ("index_heads", shape.heads),
            ("context_bytes", 240),
            ("row_bytes", 1024),
        ):
            if query(self.library, name) != expected:
                raise ValueError(f"UBmem static binary contract mismatch: {name}")
        version = self.library.kvip_ubmem_abi_version
        version.argtypes, version.restype = [], ctypes.c_uint32
        if version() != 1:
            raise ValueError("UBmem fusion ABI 1 is required")
        if lease.context["token_bytes"] != shape.token_bytes or lease.context["tokens_per_block"] != 128:
            raise ValueError("UBmem kernel and KVAF slot geometry differ")
        if not 0 <= prefetch_limit <= 8192 or not 1 <= window_limit <= 480:
            raise ValueError("Invalid UBmem admission budget")
        geometry = li_geometry(shape.batch, "auto", shape.intermediate_fetches, shape.aic_count)
        self.rounds, self.service_lanes = geometry["intermediate_fetches"], geometry["service_aivs_per_window"]
        self.final_workers = final_aivs(shape.batch, "auto", shape.aic_count)
        for name, expected in (
            ("rounds", self.rounds),
            ("service_lanes", self.service_lanes),
            ("final_workers", self.final_workers),
            ("li_cores", geometry["rotating_aic"]),
        ):
            if query(self.library, name) != expected:
                raise ValueError(f"UBmem static scheduling mismatch: {name}")
        for name, expected in (("context_bytes", 240), ("region_bytes", 128), ("status_bytes", 128)):
            function = getattr(self.library, "kvip_ubmem_" + name)
            function.argtypes, function.restype = [], ctypes.c_uint32
            if function() != expected:
                raise ValueError(f"UBmem descriptor ABI mismatch: {name}")
        for name, (size, dtype) in rotation_spec(
            shape.batch, shape.history, shape.resident, shape.aic_count, shape.intermediate_fetches
        ).items():
            value = (
                torch.zeros(size, dtype=getattr(torch, dtype), device=self.device) if scratch is None else scratch[name]
            )
            if value.device != self.device or tuple(value.shape) != size or value.dtype != getattr(torch, dtype):
                raise ValueError(f"UBmem scratch mismatch: {name}")
            if not value.is_contiguous():
                raise ValueError(f"UBmem scratch must be contiguous: {name}")
            setattr(self, name, value)
        self.final_posts = self.owners[shape.batch * 32 + shape.batch * self.final_workers * 288 :].reshape(
            shape.batch, self.final_workers, 32
        )
        self.region = lease.region(output)
        self.output = output
        self.query_ends, self.hot_valid, self.row_owners = query_ends, hot_valid, row_owners
        self.token_addresses, self.tail_bounds = token_addresses, tail_bounds
        for name, tensor, expected in (
            ("query_ends", query_ends, (shape.batch, 4)),
            ("row_owners", row_owners, (shape.batch,)),
            ("hot_valid", hot_valid, (hot_valid.shape[0], shape.resident)),
        ):
            if (
                tensor.device != self.device
                or tensor.dtype != torch.int32
                or tuple(tensor.shape) != expected
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"UBmem {name} must be contiguous int32 with shape {expected}")
        if (
            token_addresses.device != self.device
            or token_addresses.dtype not in (torch.int64, torch.uint64)
            or tuple(token_addresses.shape) != (hot_valid.shape[0], shape.history)
            or not token_addresses.is_contiguous()
        ):
            raise ValueError("UBmem token addresses must match the stable-owner Index rows")
        self.output_stride = shape.resident + 8192 if output_stride is None else output_stride
        if self.output_stride not in (shape.resident + 8192, shape.resident + 8192 + 384):
            raise ValueError("UBmem output stride must preserve hot/fallback/optional ABC slots")
        if self.output_stride * shape.token_bytes % 32 or output.data_ptr() % 32:
            raise ValueError("UBmem owner rows must start on distinct aligned 32-byte sectors")
        if tail_bounds is not None and (
            tail_bounds.device != self.device
            or tail_bounds.dtype != torch.int32
            or tuple(tail_bounds.shape) != (shape.batch, 2)
            or not tail_bounds.is_contiguous()
            or self.output_stride != shape.resident + 8192 + 384
        ):
            raise ValueError("UBmem ABC bounds require contiguous int32 [B,2] and 384 output slots")
        if output.numel() * output.element_size() < hot_valid.shape[0] * self.output_stride * shape.token_bytes:
            raise ValueError("UBmem output does not cover every stable owner")
        pointers = [
            self.progress.data_ptr(),
            self.rows.data_ptr(),
            self.mapping.data_ptr(),
            self.locations.data_ptr(),
            self.events.data_ptr(),
            arguments[13].data_ptr() + shape.batch * (4 * 2048 * 6 + 1024),
            arguments[9].data_ptr(),
            self.owners.data_ptr(),
            0,
            self.blocks.data_ptr(),
            self.offsets.data_ptr(),
            self.destinations.data_ptr(),
            self.count.data_ptr(),
            self.row_status.data_ptr(),
            0,
        ]
        extra = [
            self.region.data_ptr(),
            self.ubmem_status.data_ptr(),
            self.lru_slots.data_ptr(),
            self.reserved_hot.data_ptr(),
            self.changes.data_ptr(),
            query_ends.data_ptr(),
            self.fallback_addresses.data_ptr(),
            0,
            hot_valid.data_ptr(),
            row_owners.data_ptr(),
            token_addresses.data_ptr(),
            token_addresses.stride(0),
            self.output_stride,
            0 if tail_bounds is None else tail_bounds.data_ptr(),
        ]
        blob = np.concatenate(
            (
                np.asarray(pointers, dtype="<u8").view("u1"),
                np.asarray([prefetch_limit, window_limit], dtype="<u4").view("u1"),
                np.asarray(extra, dtype="<u8").view("u1"),
            )
        )
        if blob.size != 240:
            raise AssertionError("UBmem Context ABI changed")
        self.context = torch.from_numpy(blob.copy()).to(self.device)
        self.arguments = tuple(arguments) + (self.context,)
        self.bindings = device_words(torch, [t.data_ptr() for t in self.arguments], self.device)
        self.reserved_hot.fill_(-1)
        lease.bindings.append(self)
