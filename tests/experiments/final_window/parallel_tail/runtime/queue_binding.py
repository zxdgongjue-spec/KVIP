"""Capture-time bindings for the four-Q terminal path.

This module does not run in the measured graph. Entity snapshots must come from
the live output lease, never from a saved failure snapshot. Queue heads remain
device-side and are deliberately absent from the immutable template.
"""

from dataclasses import dataclass
import struct


QUERIES = 4
ENTITY_BYTES = 192
TEMPLATE_WORDS = 32
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1


def final_queue(batch: int, row: int, query: int) -> int:
    """Match rotating::FinalQueue(batch, 4, row, query)."""
    if not 1 <= batch <= 16 or not 0 <= row < batch or not 0 <= query < QUERIES:
        raise ValueError("Invalid terminal queue geometry")
    return row * 2 + 1 if query == 0 else batch * 2 + row * 3 + query - 1


@dataclass(frozen=True)
class QueueBinding:
    row: int
    query: int
    view: int
    entity_address: int
    queue_address: int
    output_address: int
    template: tuple[int, ...]


def wqe_template(entity: bytes, output: int, token_bytes: int) -> tuple[int, ...]:
    """Mirror SubmitWqesAt's lease-constant fields, plus SQ/doorbell addresses."""
    if len(entity) != ENTITY_BYTES:
        raise ValueError("KVAF entity ABI must be 192 bytes")
    if not 0 < output <= UINT64_MAX or not 0 < token_bytes <= UINT32_MAX:
        raise ValueError("Invalid output address or token size")
    fields = struct.unpack("<48I", entity)
    depth = fields[19]
    if depth < 2 or depth & (depth - 1):
        raise ValueError("SQ depth must be a power of two")
    if not (fields[14] | fields[15]) or not (fields[16] | fields[17]):
        raise ValueError("Live SQ and doorbell addresses are required")
    words = [0] * TEMPLATE_WORDS
    words[1] = 6 << 8
    words[2] = (fields[20] & 0xFFFFFF) | 0x1000000
    words[3] = fields[6] & 0xFFFFF
    words[4:8] = fields[22:26]
    words[8] = fields[7]
    words[12:16] = [token_bytes, fields[12], output & UINT32_MAX, output >> 32]
    words[17] = depth
    words[24:28] = fields[14:18]
    return tuple(words)


def bind_queues(batch: int, views: list[dict], token_bytes: int) -> list[QueueBinding]:
    """Resolve real lease views once; reject aliases before graph capture.

    Each view contains entity_address, queue_address, output_address and the
    corresponding entity_bytes. All prefetch and terminal queues must be unique.
    No index, owner, active-row or progress value is snapshotted here.
    """
    if not 1 <= batch <= 16 or len(views) < batch * 5:
        raise ValueError("Four final queues plus one prefetch queue per row required")
    queues = set()
    sqs = set()
    doorbells = set()
    cqs = set()
    cq_doorbells = set()
    for view in views[:batch * 5]:
        queue = view["queue_address"]
        entity = view["entity_bytes"]
        if not 0 < queue <= UINT64_MAX or len(entity) != ENTITY_BYTES:
            raise ValueError("Invalid live queue binding")
        fields = struct.unpack("<24Q", entity)
        sq, doorbell = fields[7:9]
        cq, cq_doorbell = fields[13:15]
        if not all((sq, doorbell, cq, cq_doorbell)):
            raise ValueError("Live SQ/CQ and doorbell addresses are required")
        if (queue in queues or sq in sqs or doorbell in doorbells
                or cq in cqs or cq_doorbell in cq_doorbells):
            raise ValueError("Prefetch/final queues must have exclusive SQ/CQ ownership")
        queues.add(queue)
        sqs.add(sq)
        doorbells.add(doorbell)
        cqs.add(cq)
        cq_doorbells.add(cq_doorbell)
    bindings = []
    for row in range(batch):
        for query in range(QUERIES):
            index = final_queue(batch, row, query)
            view = views[index]
            if not 0 < view["entity_address"] <= UINT64_MAX:
                raise ValueError("Invalid live entity address")
            bindings.append(QueueBinding(
                row, query, index, view["entity_address"], view["queue_address"],
                view["output_address"], wqe_template(
                    view["entity_bytes"], view["output_address"], token_bytes)))
    return bindings
