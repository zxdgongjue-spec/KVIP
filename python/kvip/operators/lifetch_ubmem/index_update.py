# SPDX-License-Identifier: Apache-2.0
"""Publish LI outputs and update Index ownership after real UBmem completion."""

import ctypes

from kvip.operators.lifetch_ubmem.workspace import WORK_WORDS, device_words
from kvip.operators.output_views import validate_slot_output


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
        slot_output=None,
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
        if slot_output is not None:
            self.slot_output = validate_slot_output(
                torch,
                slot_output,
                batch=batch,
                device=self.device,
                protected=(
                    keys,
                    scales,
                    addresses,
                    logical_ids,
                    row_owners,
                    status,
                    self.plan_buffer,
                    self.extended_work,
                    self.logical_output,
                    rotating.change_slots,
                    rotating.changes,
                    rotating.change_count,
                    rotating.rows,
                    rotating.hot_valid,
                    rotating.row_status,
                    rotating.final_posts,
                    rotating.ubmem_status,
                    rotating.region,
                    rotating.output,
                    *rotating.arguments,
                    rotating.locations,
                    query_inputs.unpack_indices,
                    query_inputs.output_valid,
                ),
            )
        pointers = [
            keys.data_ptr(),
            scales.data_ptr(),
            addresses.data_ptr(),
            logical_ids.data_ptr(),
            rotating.change_slots.data_ptr(),
            rotating.changes.data_ptr(),
            status.data_ptr(),
            addresses.stride(0),
            rotating.rows.data_ptr(),
            row_owners.data_ptr(),
            rotating.row_status.data_ptr(),
            rotating.final_posts.data_ptr(),
            rotating.ubmem_status.data_ptr(),
            rotating.region.data_ptr(),
            rotating.final_workers,
            128,
            rotating.hot_valid.data_ptr(),
            self.plan_buffer.data_ptr(),
            rotating.arguments[9].data_ptr(),
            rotating.locations.data_ptr(),
            query_inputs.unpack_indices.data_ptr(),
            query_inputs.output_valid.data_ptr(),
            self.logical_output.data_ptr(),
            self.slot_output.data_ptr(),
            history,
            rotating.shape.resident,
            rotating.change_count.data_ptr(),
            self.extended_work.data_ptr(),
        ]
        self.bindings = device_words(torch, pointers, self.device)
        rotating.lease.bindings.append(self)
