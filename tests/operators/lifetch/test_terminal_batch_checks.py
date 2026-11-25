from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent/'support'))
from terminal_batch_checks import STATUS_MAGIC, audit_queues, check_receipts, totals


def test_below_cutoff_and_exact_cutoff_share_one_rule():
    counts = np.zeros((16, 4), dtype=np.int64)
    counts[0] = (255, 0, 0, 0)
    counts[1] = (250, 3, 2, 1)
    status = np.zeros((16, 4, 4), dtype=np.uint64)
    status[:, :, 0] = STATUS_MAGIC
    status[:, :, 1] = counts
    status[0, :, 3] = (1, 0, 0, 0)
    status[1, :, 3] = (4, 3, 2, 1)
    pending = np.zeros((16, 4, 2), dtype=np.int32)
    histogram = check_receipts(counts, pending, status)
    assert histogram == [59, 2, 1, 1, 1]
    assert totals([histogram]*20) == [1180, 40, 20, 20, 20]
    status[1, 0, 3] -= 1
    with pytest.raises(ValueError, match='expected 4 CQEs'):
        check_receipts(counts, pending, status)
    status[1, 0, 3] += 1
    pending[0, 0, 0] = 1
    with pytest.raises(ValueError, match='pending Post'):
        check_receipts(counts, pending, status)


def test_read_private_queue_views():
    from types import SimpleNamespace as Object

    def tensor(array):
        return Object(cpu=lambda: Object(numpy=lambda: array))
    views = [row*2 + 1 if q == 0 else 32+row*3+q-1 for row in range(16) for q in range(4)]
    raw = np.zeros((80, 256), dtype=np.uint64)
    raw[views, 0] = STATUS_MAGIC
    raw[views[0], 1:4] = (300, 0, 4)
    posts = np.zeros((16, 4, 32), dtype=np.int32)
    posts[0, 0, 0] = 300
    rotating = Object(final_posts=tensor(posts))
    terminal = Object(queue_views=views, owner=Object(core_status=tensor(raw.view(np.uint8))))
    assert audit_queues(terminal, rotating) == [63, 0, 0, 0, 1]
    views[-1] = views[0]
    with pytest.raises(ValueError, match='queue views'):
        audit_queues(terminal, rotating)


def test_captured_status_must_be_private_to_each_replay_state():
    from types import SimpleNamespace as Object
    from terminal_batch_checks import check_status_binding

    base, size = 0x100000, 1216
    views = [r*2 + 1 if q == 0 else 32+r*3+q-1 for r in range(16) for q in range(4)]
    args = np.zeros((16, 4, 14), dtype='<u8')
    args[:, :, 7] = base + np.array(views).reshape(16, 4)*size
    status = Object(shape=(80, size), element_size=lambda: 1, is_contiguous=lambda: True, data_ptr=lambda: base)
    terminal = Object(owner=Object(core_status=status, abi_sizes={'status': size}), queue_views=views,
                      args=Object(cpu=lambda: Object(numpy=lambda: args.view('u1'))))
    live = []
    check_status_binding(terminal, live)
    with pytest.raises(ValueError, match='share a terminal status'):
        check_status_binding(terminal, live)
    args[0, 0, 7] += size
    with pytest.raises(ValueError, match='different status allocation'):
        check_status_binding(terminal, [])


def test_single_policy_and_pressure_for_partial_and_overlapping_batches():
    from terminal_batch_checks import diagnostic_pressure, submission_pressure

    counts = np.zeros((16, 4), np.int32)
    counts[0] = [64, 64, 64, 64]
    pending = np.zeros((16, 4, 2), np.int32)
    status = np.zeros((16, 4, 4), np.uint64)
    status[:, :, 0] = STATUS_MAGIC
    status[:, :, 1] = counts
    status[:, :, 3] = counts != 0
    assert check_receipts(counts, pending, status, policy='single') == [60, 4, 0, 0, 0]
    with pytest.raises(ValueError, match='expected 4 CQEs'):
        check_receipts(counts, pending, status, policy='paced')
    stamps = np.zeros((16, 4, 32), np.int64)
    stamps[0, :, 4] = [100, 200, 400, 500]
    stamps[0, :, 24] = [300, 400, 700, 600]
    single = submission_pressure(counts, status[:, :, 3], stamps, 100.)
    assert single == dict(first_post_peak_4us_tokens=192, peak_pending_queries=2)
    status[0, :, 3] = 4
    assert check_receipts(counts, pending, status, policy='paced') == [60, 0, 0, 0, 4]
    assert submission_pressure(counts, status[:, :, 3], stamps, 100.) == dict(
        first_post_peak_4us_tokens=12, peak_pending_queries=2)
    with pytest.raises(ValueError, match='expected 1 CQEs'):
        check_receipts(counts, pending, status, policy='single')
    rows = [dict(batch=b, final_counts=[{'total': int(n)} for n in counts[b]]) for b in range(16)]
    cores = [dict(block=b*4+q, stages=[dict(name=name, end_cycles=int(stamps[b, q, index]))
                                       for name, index in (('WQE_and_publish', 4), ('final_CQ_join', 24))])
             for b, q in np.ndindex(16, 4)]
    rows[0]['terminal_timeline'] = dict(cores=cores, clock_mhz=100.)
    assert diagnostic_pressure(dict(streaming_cycles=rows), 'single') == single
