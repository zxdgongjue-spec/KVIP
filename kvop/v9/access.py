"""Immutable access rules and byte-range checks, used only during preparation."""

from types import MappingProxyType

OUTPUT_FIELDS = (
    "topk",
    "destination",
    "miss_counts",
    "resident_lengths",
    "transfers",
    "status",
    "extra_prefetch_slots",
    "extra_prefetch_counts",
)
MUTABLE_INPUT_FIELDS = frozenset(
    {"key", "key_scale", "forward", "reverse", "cursors", "resident_output"}
)
FIELD_ACCESS = MappingProxyType(
    {
        **dict.fromkeys(OUTPUT_FIELDS, "write"),
        **dict.fromkeys(MUTABLE_INPUT_FIELDS, "read_write"),
    }
)


def tensor_range(name, tensor):
    begin = int(tensor.data_ptr())
    size = int(tensor.numel()) * int(tensor.element_size())
    if begin <= 0 or size <= 0:
        raise ValueError(f"{name}: buffer must have nonempty addressable storage")
    return name, begin, begin + size


def check_writable_ranges(ranges, writable):
    """Reject every overlap involving a writer; adjacent ranges remain valid."""
    for i, (name, begin, end) in enumerate(ranges):
        for other, lo, hi in ranges[:i]:
            if begin < hi and lo < end and (name in writable or other in writable):
                raise ValueError(f"overlapping contract buffers: {other}, {name}")
