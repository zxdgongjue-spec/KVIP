# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest

from near512_worker import CAPACITY, audit_journal, geometry_check, worker


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


@pytest.mark.parametrize('actual,donor,kind', [(7, 7, 'stitched'), (17, 7, 'stitched'), (17, 17, 'capture')])
def test_worker_rejects_wrong_stitch_before_device(tmp_path, monkeypatch, actual, donor, kind):
    sample = object()

    class Sequence:
        rounds = ((1,),)

        def __iter__(self):
            yield sample

    args = SimpleNamespace(main=tmp_path, output=tmp_path/'out', trace=None, group=17,
                           revision='x'*40, work=tmp_path, stitch_source=True, preserve_workload=False,
                           source_proof=tmp_path/'g17'/'mainline_checks.json')

    def make_input(work, group, trace, output, *, stitch_source, source_proof, preserve_workload):
        assert not preserve_workload
        assert (group, stitch_source, source_proof) == (17, True, args.source_proof)
        return Sequence(), sample, dict(actual_group=actual, kind=kind, donor=dict(group=donor))

    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES', '4')
    monkeypatch.setattr('near512_input.make_input', make_input)
    monkeypatch.setattr('near512_worker.subprocess.check_output',
                        lambda command, **_: args.revision if 'rev-parse' in command else '')
    with pytest.raises(ValueError, match='G17'):
        worker(args)
