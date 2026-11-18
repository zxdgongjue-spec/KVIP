# SPDX-License-Identifier: Apache-2.0
import pytest

from kvip.diagnostics.original_native_window import original_native_window


def original_trace():
    base = 2**54
    cores = [[base+1, base+80, 0, 0, 0, 0] + [0]*26 for _ in range(64)]
    cores[1][1] += 5
    for r in cores[48:]:
        r[2], r[5] = base+3, base+70
    for i, (n, start, end) in enumerate([(3, 20, 60), (5, 9, 69)]):
        r = cores[48+i]
        r[3], r[4], r[9], r[10] = base+start, base+end, 2, n
    return dict(raw=dict(batch=16, resolver_cores=3, workers_per_query=4, cores=cores),
                reads=dict(final=8), auto_route=dict(mode="high"))


def test_endpoints_count_only_nonempty_workers():
    assert original_native_window(original_trace()) == dict(first_native_call=8, last_native_return=68,
        last_consumer_exit=84, last_role_exit=84, final_Get=8, active_workers=2, native_calls=4)


def test_reject_misdecoded_worker_records():
    sample = original_trace()
    sample["reads"]["final"] = 10
    with pytest.raises(ValueError, match="request count"):
        original_native_window(sample)
