"""Reference for publishing staged resolver misses to the existing URMA fetch.

This models descriptor reordering only; it performs no device I/O or KV copies.
Slots are physical hot slots and source addresses are snapshots from before
Index metadata is committed.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Descriptor:
    token: int
    slot: int
    source_address: int


@dataclass(frozen=True)
class FinalPost:
    descriptor: int
    count: int
    nonconflicting: int
    deferred: bool


def publish(records, reserved, used, batch, owner, output_stride, workers, max_slots=8192):
    if workers < 1 or not 0 <= used <= 2048 or len(records) > max_slots:
        raise ValueError('invalid worker count, prefetch count, or descriptor count')
    if batch < 0 or owner < 0 or output_stride < 1:
        raise ValueError('invalid batch, owner, or output stride')
    if len({entry.token for entry in records}) != len(records):
        raise ValueError('resolver output contains duplicate tokens')
    if len({entry.slot for entry in records}) != len(records):
        raise ValueError('resolver output contains duplicate destinations')
    for entry in records:
        if entry.token < 0 or not 0 <= entry.slot < len(reserved) or entry.source_address < 0:
            raise ValueError('invalid resolver record')
    count = len(records)
    packed, posts = [], []
    for worker in range(workers):
        begin, end = count * worker // workers, count * (worker + 1) // workers
        safe, conflict = [], []
        for entry in records[begin:end]:
            ordinal = reserved[entry.slot]
            (conflict if 0 <= ordinal < used else safe).append(entry)
        own = safe + conflict
        posts.append(FinalPost(batch * max_slots + begin, len(own), len(safe), bool(own)))
        for entry in own:
            packed.append((entry.token, entry.source_address, owner * output_stride + entry.slot))
    return packed, posts
