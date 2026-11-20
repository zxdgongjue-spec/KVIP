# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from ranked_five_report import JOINT_MAGIC
from ranked_joint_report import drift, joint_endpoints


def sample():
    ticks = np.zeros((1, 6656), dtype=np.int64)
    core = ticks[:, 4096:6144].reshape(1, 64, 32)
    core[..., 0] = 100
    core[0, 48, 16:26] = [180, 200, 270, 210, 220, 210, 220, 3, 10, 30]
    ticks[:, :1024].reshape(1, 16, 64)[..., 56] = 260
    fetch = ticks[:, 2048:4096].reshape(1, 16, 128)
    fetch[..., 99] = 301
    fetch[0, 0, 16] = 1
    stamp = ticks[:, 6144:].reshape(1, 64, 8)
    for i in range(64):
        r, f = divmod(i, 3) if i < 48 else (i-48, 3)
        stamp[0, i] = [JOINT_MAGIC, r, i << 32 | f, 30, 150, 240, 160, 0]
    return {'times': ticks, 'cycles_per_us': 2}, core, fetch, stamp


def test_split_prefix_can_finish_on_another_block():
    raw, core, _, _ = sample()
    ends = joint_endpoints(raw)
    assert (ends['wins'], ends['remote_wins'], ends['precq_moves'], ends['unprocessed_by_prefix_moves']) == (1,1,10,20)
    assert ends['last_index_minus_last_cq']['p50'] == -5
    core[0, 48, 17] = 225
    with pytest.raises(ValueError, match='Invalid final'):
        joint_endpoints(raw)


def test_empty_get_and_no_help():
    raw, core, fetch, _ = sample()
    core[0, 48, 16:28] = 0
    fetch[0, 0, 16] = 0
    ends = joint_endpoints(raw)
    assert ends['first_doorbell']['p50'] is None
    assert ends['attempts'] == 0
    assert ends['last_index']['p50'] == 80


def test_capture_drift_separates_mean_and_median():
    assert drift([1, 1, 1, 1, 1, 2, 2, 2, 2, 12]) == (3,1)


def test_idle_row_core_may_finish_its_assigned_remote_row():
    raw, _, _, stamp = sample()
    # Block 0 can finish remote row 1 after its local duties have completed.
    stamp[0, 49, 2] = 3
    assert joint_endpoints(raw)['cross_row_completions'] == 1
    stamp[0, 50, 2] = 3
    with pytest.raises(ValueError, match='physical owner'):
        joint_endpoints(raw)
