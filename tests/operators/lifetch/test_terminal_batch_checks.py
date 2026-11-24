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
