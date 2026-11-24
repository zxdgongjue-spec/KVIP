# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest

from near512_worker import CAPACITY, audit_journal, geometry_check


def test_require_all_rows_long_with_published_high_addresses():
    ends = 524280 - 17 * np.arange(16)[:, None] + np.arange(4)
    g = dict(batch=16, heads=32, history=CAPACITY, resident=16384, token_bytes=656)
    inputs = SimpleNamespace(geometry=g, query_ends=ends, logical_address_valid=np.ones((16, CAPACITY), np.uint8))
    geometry_check(inputs)
    ends[13] = [100, 101, 102, 103]
    with pytest.raises(ValueError, match='510K'):
        geometry_check(inputs)
    ends[13] += 524100
    inputs.logical_address_valid[12, 262144:] = 0
    with pytest.raises(ValueError, match='Row 12'):
        geometry_check(inputs)


class DeviceView:
    def __init__(self, data):
        self.value = np.array(data)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def test_diagnostic_checks_exact_order_and_admission_count():
    start = np.array([[0, 1, 2, 3]])
    state = SimpleNamespace(input_layout=start, rotating=SimpleNamespace(change_count=DeviceView([2])),
                            forward=SimpleNamespace(events_src=DeviceView([[2, 3]]), events_dst=DeviceView([[1, 0]])))
    valid = np.ones((1, 2))
    inputs = SimpleNamespace(captured_hot_valid=valid)
    fetch = np.array([1])
    final = np.array([[0, 0, 1, 0]])
    updated = np.array([[3, 2, 1, 0]])
    assert audit_journal(state, inputs, updated, valid, fetch, final)['ordered_rows'] == 1
    with pytest.raises(ValueError, match='event replay'):
        audit_journal(state, inputs, np.array([[2, 3, 0, 1]]), valid, fetch, final)
    with pytest.raises(ValueError, match='admission counts'):
        audit_journal(state, inputs, updated, valid, fetch, final+1)
