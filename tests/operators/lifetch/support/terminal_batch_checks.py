"""Verify successful private SQ/CQ batches from live KVAF receipts after replay."""
import numpy as np


STATUS_MAGIC = 0x4B5641464247444F


def check_receipts(counts, pending_error, status, *, policy="paced"):
    if policy not in ("paced", "single"):
        raise ValueError(f"Unknown terminal batch policy: {policy}")
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
            if policy == "paced" and total >= 256 and own > 1:
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


def _queue_receipts(terminal, rotating):
    # The row-summary status and the live private KVAF queue status are distinct.
    # Each replay state owns its status allocation even when its lease is reused.
    views = np.asarray(terminal.queue_views).reshape(16, 4)
    raw = terminal.owner.core_status.cpu().numpy()
    if (raw.dtype != np.uint8 or raw.ndim != 2 or raw.shape[1] < 32
            or views.min() < 0 or views.max() >= len(raw) or np.unique(views).size != views.size):
        raise ValueError('Invalid private terminal queue views or KVAF status allocation')
    words = raw[:, :32].copy().view('<u8')
    posts = rotating.final_posts.cpu().numpy()
    return posts[:, :, 0], posts[:, :, 1:3], words[views]


def audit_queues(terminal, rotating, *, policy="paced"):
    return check_receipts(*_queue_receipts(terminal, rotating), policy=policy)


def ablation_receipt(state, policy, *, diagnostic):
    counts, pending_error, status = _queue_receipts(state.forward.terminal, state.rotating)
    check_receipts(counts, pending_error, status, policy=policy)
    prefetch = state.rotating.rows[:, 9].cpu().numpy()
    if prefetch.shape != (16,) or np.any(prefetch < 0) or np.any(prefetch > 2048):
        raise ValueError('Invalid measured per-request LI prefetch count')
    cqe = status[:, :, 3]
    receipt = dict(prefetch=prefetch.tolist(), final=counts.tolist(), cqe=cqe.tolist())
    if diagnostic:
        stamps = state.forward.terminal.times.cpu().numpy()
        if stamps.shape != (16, 4, 32) or not np.array_equal(stamps[:, :, 20], counts):
            raise ValueError('Terminal diagnostic counts and live request receipts disagree')
        receipt.update(submission_pressure(counts, cqe, stamps, state.forward.counter_clock_mhz))
    return receipt


def submission_pressure(counts, cqe, stamps, clock_mhz):
    """Describe initial submissions across queues during the separate diagnostic replay."""
    if not np.isfinite(clock_mhz) or clock_mhz <= 0:
        raise ValueError('Expected a calibrated terminal clock')
    first_posts, pending = [], []
    for row, q in zip(*np.nonzero(counts)):
        start, stop = map(int, stamps[row, q, [4, 24]])
        if not 0 < start < stop:
            raise ValueError('Invalid first publication or final CQ observation timestamp')
        own = int(counts[row, q])
        first = (own + 15) // 16 if cqe[row, q] > 1 else own
        first_posts.append((start, first))
        pending.extend(((start, 1), (stop, -1)))
    # A CQ observation can follow the actual hardware completion. This peak
    # counts overlapping software observation windows, not hardware queue depth.
    active = peak_pending = 0
    for _, delta in sorted(pending):
        active += delta
        peak_pending = max(peak_pending, active)
    first_posts.sort()
    left = volume = peak_volume = 0
    for start, entries in first_posts:
        volume += entries
        while start - first_posts[left][0] >= 4 * clock_mhz:
            volume -= first_posts[left][1]
            left += 1
        peak_volume = max(peak_volume, volume)
    return dict(first_post_peak_4us_tokens=peak_volume, peak_pending_queries=peak_pending)


def diagnostic_pressure(sample, policy):
    """Read the same terminal stages from an archived production diagnostic."""
    records = sample['streaming_cycles']
    row_counts = sorted((r for r in records if 'final_counts' in r), key=lambda r: r['batch'])
    if [r['batch'] for r in row_counts] != list(range(16)):
        raise ValueError('Missing per-request terminal diagnostics')
    counts = np.asarray([[q['total'] for q in r['final_counts']] for r in row_counts])
    if counts.shape != (16, 4) or counts.dtype.kind not in 'iu' or np.any(counts < 0) or np.any(counts > 2048):
        raise ValueError('Invalid archived per-query GET counts')
    timeline, = [r['terminal_timeline'] for r in records if 'terminal_timeline' in r]
    cores = sorted(timeline['cores'], key=lambda c: c['block'])
    if [c['block'] for c in cores] != list(range(64)):
        raise ValueError('Missing terminal worker timeline')
    stamps = np.zeros((16, 4, 32), dtype=np.int64)
    for core in cores:
        b, q = divmod(core['block'], 4)
        stages = {p['name']: p['end_cycles'] for p in core['stages']}
        stamps[b, q, [4, 24]] = [stages['WQE_and_publish'], stages['final_CQ_join']]
    if policy != 'single':
        raise ValueError('Paced diagnostic pressure requires observed per-Q CQ receipts')
    return submission_pressure(counts, counts != 0, stamps, timeline['clock_mhz'])


def totals(calls):
    counts = np.asarray(calls)
    if (counts.shape != (20, 5) or counts.dtype.kind not in 'iu' or (counts < 0).any()
            or not (counts.sum(axis=1) == 64).all()):
        raise ValueError('Missing private-CQ observations from 20 calls with 64 queries')
    return counts.sum(axis=0).tolist()
