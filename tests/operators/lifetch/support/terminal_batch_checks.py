"""Verify successful private SQ/CQ batches from live KVAF receipts after replay."""
import numpy as np


STATUS_MAGIC = 0x4B5641464247444F


def check_receipts(counts, pending_error, status):
    if counts.shape != (16, 4) or pending_error.shape != (16, 4, 2) or status.shape != (16, 4, 4):
        raise ValueError('Expected one terminal Post and KVAF receipt per row and query')
    if (np.any(counts < 0) or np.any(counts > 2048) or np.any(pending_error)
            or np.any(status[:, :, 0] != STATUS_MAGIC) or np.any(status[:, :, 2])
            or np.any(status[:, :, 1] != counts)):
        raise ValueError('Invalid per-query KVAF completion, descriptor count or pending Post')
    histogram = [0] * 5
    for row, descriptors in enumerate(counts):
        total = int(descriptors.sum())
        for query, own in enumerate(descriptors):
            own = int(own)
            batches = int(status[row, query, 3])
            ends = {own} if own else set()
            if total >= 256 and own > 1:
                ends.update((own * n + 15) // 16 for n in (1, 6, 11))
            if batches != len(ends):
                raise ValueError(f'Row {row} Q{query+1} expected {len(ends)} CQEs for {own} of {total} '
                                 f'final descriptors, observed {batches}')
            histogram[batches] += 1
    return histogram


def check_status_binding(terminal, live_ranges):
    """Inspect captured argument pointers once, before the hardware graph is created."""
    status = terminal.owner.core_status
    size = terminal.owner.abi_sizes['status']
    base = status.data_ptr()
    end = base + 80*size
    views = np.asarray(terminal.queue_views)
    rows, qs = np.indices((16, 4))
    expected_views = np.where(qs == 0, 2*rows+1, 32+3*rows+qs-1).reshape(-1)
    args = terminal.args.cpu().numpy()
    if (tuple(status.shape) != (80, size) or size < 32 or status.element_size() != 1
            or not status.is_contiguous() or args.shape != (16, 4, 112) or args.dtype != np.uint8
            or not np.array_equal(views, expected_views)):
        raise ValueError('Invalid private terminal status binding')
    # In the pinned KVAF ABI, core_status is the eighth u64 in each argument.
    actual = args.reshape(64, 112).view('<u8')[:, 7]
    if not np.array_equal(actual, base + views * size):
        raise ValueError('Captured terminal writes to a different status allocation')
    if any(a < end and base < b for a, b in live_ranges):
        raise ValueError('Live replay states share a terminal status allocation')
    live_ranges.append((base, end))


def audit_queues(terminal, rotating):
    # The row-summary status and the live private KVAF queue status are distinct.
    # Each replay state owns its status allocation even when its lease is reused.
    views = np.asarray(terminal.queue_views).reshape(16, 4)
    raw = terminal.owner.core_status.cpu().numpy()
    if (raw.dtype != np.uint8 or raw.ndim != 2 or raw.shape[1] < 32
            or views.min() < 0 or views.max() >= len(raw) or np.unique(views).size != views.size):
        raise ValueError('Invalid private terminal queue views or KVAF status allocation')
    words = raw[:, :32].copy().view('<u8')
    posts = rotating.final_posts.cpu().numpy()
    return check_receipts(posts[:, :, 0], posts[:, :, 1:3], words[views])


def totals(calls):
    counts = np.asarray(calls)
    if (counts.shape != (20, 5) or counts.dtype.kind not in 'iu' or (counts < 0).any()
            or not (counts.sum(axis=1) == 64).all()):
        raise ValueError('Missing private-CQ observations from 20 calls with 64 queries')
    return counts.sum(axis=0).tolist()
