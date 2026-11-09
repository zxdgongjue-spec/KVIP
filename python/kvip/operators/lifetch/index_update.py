# SPDX-License-Identifier: Apache-2.0
"""Bind producer change records to stable service-owned Index rows."""

import ctypes

from kvip.operators.lifetch.workspace import WORK_WORDS, device_words


def _update_pointers(keys, scales, addresses, logical_ids, row_owners, rotating, status, plan,
                     prefetch_status, final_status, status_size, work, query_inputs, logical, slots,
                     history, resident):
    # Shared 28-word device ABI. Transport-specific completion descriptors are
    # supplied by the caller; their semantics and lifetime remain unchanged.
    values = [keys.data_ptr(), scales.data_ptr(), addresses.data_ptr(), logical_ids.data_ptr(),
              rotating.change_slots.data_ptr(), rotating.changes.data_ptr(), status.data_ptr(),
              addresses.stride(0), rotating.rows.data_ptr(), row_owners.data_ptr(),
              rotating.row_status.data_ptr(), rotating.final_posts.data_ptr(),
              prefetch_status.data_ptr(), final_status.data_ptr(), rotating.final_workers, status_size,
              rotating.hot_valid.data_ptr(), plan.data_ptr()]
    if query_inputs is not None:
        values.extend([rotating.arguments[9].data_ptr(), rotating.locations.data_ptr(),
                       query_inputs.unpack_indices.data_ptr(), query_inputs.output_valid.data_ptr(),
                       logical.data_ptr(), slots.data_ptr()])
    values.extend([0] * (24 - len(values)))
    values.extend([history, resident, rotating.change_count.data_ptr(), work.data_ptr()])
    return values


class DirectIndexUpdate:
    def __init__(
        self,
        torch,
        library,
        *,
        keys,
        scales,
        addresses,
        logical_ids,
        row_owners,
        rotating,
        status=None,
        query_inputs=None,
        outputs=None,
    ):
        self.torch, self.device = torch, keys.device
        self.batch = row_owners.numel()
        if not 1 <= self.batch <= 32:
            raise ValueError("LIFetch update requires 1..32 active rows")
        self.storage = keys, scales, addresses, logical_ids, row_owners, rotating
        if any(t.device != self.device for t in self.storage[:-1]) or any(
            not t.is_contiguous() for t in (keys, scales, logical_ids, row_owners)
        ):
            raise ValueError("LIFetch Index storage must be contiguous on one device")
        if row_owners.dtype != torch.int32 or row_owners.ndim != 1:
            raise ValueError("Request owners must be a flat int32 vector")
        if keys.ndim != 3 or keys.shape[-1] != 128 or keys.element_size() != 1:
            raise ValueError("LIFetch Index keys require [owner,history,128] byte storage")
        if any(t.shape != keys.shape[:2] for t in (scales, addresses, logical_ids)):
            raise ValueError("Index key, scale, address and logical-ID rows must match")
        if addresses.stride(1) != 1 or addresses.stride(0) < keys.shape[1]:
            raise ValueError("Index addresses require contiguous, nonoverlapping rows")
        if scales.dtype != torch.float32 or addresses.dtype != torch.int64 or logical_ids.dtype != torch.int64:
            raise ValueError("Index scales require float32; address and logical IDs require int64")
        for name, shape in (
            ("change_slots", (self.batch, 10240)),
            ("changes", (self.batch, 10240)),
            ("change_count", (self.batch,)),
            ("rows", (self.batch, 256)),
        ):
            tensor = getattr(rotating, name)
            if (
                tensor.device != self.device
                or not tensor.is_contiguous()
                or tensor.dtype != torch.int32
                or tuple(tensor.shape) != shape
            ):
                raise ValueError(f"LIFetch producer {name} must be contiguous int32 with shape {shape}")
        self.library = ctypes.CDLL(str(library))
        version = self.library.kvop_direct_update_abi_version
        version.argtypes, version.restype = [], ctypes.c_uint32
        if version() != 3:
            raise ValueError("Index update requires hot-only ABI 3")
        self.status = torch.zeros(self.batch, dtype=torch.int32, device=self.device) if status is None else status
        if (
            self.status.device != self.device
            or self.status.dtype != torch.int32
            or self.status.shape != (self.batch,)
            or not self.status.is_contiguous()
        ):
            raise ValueError("Index update status must be contiguous int32 [batch] on the cache device")
        plan_words = self.library.kvop_direct_update_plan_words
        plan_words.argtypes, plan_words.restype = [], ctypes.c_uint32
        self.plan_buffer = torch.zeros((self.batch, plan_words()), dtype=torch.int32, device=self.device)
        work_words = self.library.kvop_direct_update_work_words
        work_words.argtypes, work_words.restype = [], ctypes.c_uint32
        self.work_buffer = torch.empty((self.batch, work_words()), dtype=torch.int32, device=self.device)
        self.logical_output = self.slot_output = None
        if query_inputs is not None:
            if outputs is None:
                outputs = torch.zeros((2, self.batch * 4, 1, 2048), dtype=torch.int32, device=self.device)
            if (
                tuple(outputs.shape) != (2, self.batch * 4, 1, 2048)
                or outputs.dtype != torch.int32
                or outputs.device != self.device
                or not outputs.is_contiguous()
            ):
                raise ValueError("Service outputs must be contiguous int32 [2, B*4, 1, 2048]")
            self.logical_output, self.slot_output = outputs.unbind(0)
        pointers = _update_pointers(
            keys, scales, addresses, logical_ids, row_owners, rotating, self.status, self.plan_buffer,
            rotating.prefetch_status, rotating.final_status, rotating.owner.abi_sizes["status"],
            self.work_buffer, query_inputs, self.logical_output, self.slot_output,
            keys.shape[1], rotating.hot_valid.shape[1],
        )
        self.bindings = torch.tensor(pointers, dtype=torch.int64, device=self.device)
        self.launch = (
            self.library.kvop_direct_update_owned if query_inputs is None else self.library.kvop_direct_update_outputs
        )
        self.launch.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]
        self.launch.restype = ctypes.c_uint32

    def __call__(self):
        result = self.launch(
            self.batch, self.bindings.data_ptr(), self.torch.npu.current_stream(self.device).npu_stream
        )
        if result:
            raise RuntimeError(f"LIFetch Index update launch failed: {result}")

    def _stage(self, name):
        function = getattr(self.library, name)
        function.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]
        function.restype = ctypes.c_uint32
        result = function(self.batch, self.bindings.data_ptr(), self.torch.npu.current_stream(self.device).npu_stream)
        if result:
            raise RuntimeError(f"Index update {name} failed: {result}")

    def plan(self):
        self._stage("kvop_direct_plan_outputs")

    def apply(self):
        self._stage("kvop_direct_apply_owned")


class UbmemIndexUpdate:
    def __init__(
        self,
        torch,
        *,
        keys,
        scales,
        addresses,
        logical_ids,
        row_owners,
        rotating,
        status,
        query_inputs,
        outputs,
        plan_buffer=None,
        extended_work=None,
    ):
        self.torch, self.device, self.library = torch, keys.device, rotating.library
        batch, history = rotating.shape.batch, rotating.shape.history
        self.storage = keys, scales, addresses, logical_ids, row_owners, rotating, query_inputs, outputs
        for name, tensor, shape, dtype in (
            ("scales", scales, keys.shape[:2], torch.float32),
            ("addresses", addresses, keys.shape[:2], torch.int64),
            ("logical_ids", logical_ids, keys.shape[:2], torch.int64),
            ("row_owners", row_owners, (batch,), torch.int32),
            ("status", status, (batch,), torch.int32),
            ("outputs", outputs, (2, batch * 4, 1, 2048), torch.int32),
        ):
            if (
                tensor.device != self.device
                or tuple(tensor.shape) != tuple(shape)
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"UBmem Index {name} must be contiguous {dtype} with shape {shape}")
        if keys.ndim != 3 or keys.shape[1:] != (history, 128) or keys.element_size() != 1 or not keys.is_contiguous():
            raise ValueError("UBmem Index keys must be contiguous bytes [owner, history, 128]")
        words = self.library.kvip_ubmem_plan_words
        words.argtypes, words.restype = [], ctypes.c_uint32
        self.plan_buffer = (
            torch.zeros((batch, words()), dtype=torch.int32, device=self.device) if plan_buffer is None else plan_buffer
        )
        self.extended_work = (
            torch.empty((batch, WORK_WORDS), dtype=torch.int32, device=self.device)
            if extended_work is None
            else extended_work
        )
        for name, tensor, shape in (
            ("plan_buffer", self.plan_buffer, (batch, words())),
            ("extended_work", self.extended_work, (batch, WORK_WORDS)),
        ):
            if (
                tensor.device != self.device
                or tensor.dtype != torch.int32
                or tuple(tensor.shape) != shape
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"UBmem Index {name} must be contiguous int32 with shape {shape}")
        self.status = status
        self.logical_output, self.slot_output = outputs.unbind(0)
        pointers = _update_pointers(
            keys, scales, addresses, logical_ids, row_owners, rotating, status, self.plan_buffer,
            rotating.ubmem_status, rotating.region, 128, self.extended_work, query_inputs,
            self.logical_output, self.slot_output, history, rotating.shape.resident,
        )
        self.bindings = device_words(torch, pointers, self.device)
        rotating.lease.bindings.append(self)
