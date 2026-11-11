# SPDX-License-Identifier: Apache-2.0
"""Validate caller-owned output views before binding native pointers."""


def validate_slot_output(torch, output, *, batch, device, protected):
    if (
        output.device != device
        or output.dtype != torch.int32
        or tuple(output.shape) != (batch * 4, 1, 2048)
        or not output.is_contiguous()
    ):
        raise ValueError("Slot output must be contiguous int32 [B*4, 1, 2048] on the Index device")
    begin = output.data_ptr()
    end = begin + output.numel() * output.element_size()
    for tensor in protected:
        if not tensor.numel():
            continue
        # Address rows may have padding. Reject overlap with their entire
        # strided span, rather than accepting a view into a later padded row.
        span = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
        source = tensor.data_ptr()
        if begin < source + span * tensor.element_size() and source < end:
            raise ValueError("Slot output must not overlap Index inputs or logical output")
    return output
