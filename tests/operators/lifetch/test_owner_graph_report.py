"""Validate same-invocation matching and local-clock bounds without a device."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent/'support'))
from owner_tail_graph import CALLS, FIELDS, PASSES, counter_digest, summarize, traced


def test_graph_bands_owner_skew_and_missing_plan_controls():
    rows = np.zeros((64, 848), dtype=np.int64)
    requests = []
    for block, row in enumerate(rows):
        b, q = divmod(block, 4)
        # Independent clock epochs deliberately disagree by much more than the entire graph.
        base = (block+1)*10_000
        marks = dict(tail_start=1, resolve_start=11, resolve_end=18, adapt_start=21, adapt_end=23,
                     index_start=32, index_end=38, done=40, barrier_start=86, barrier_end=100+block%4)
        if q:
            marks.update(raw_wait=12, raw_ready=14, index_wait=24, index_ready=30)
        if q == 3:
            marks.update(prefetch_wait=3, prefetch_ready=4, prefetch_end=9, finalize_start=42, finalize_end=57)
            row[:12] = np.arange(base+43, base+55)
            row[FIELDS['prefetch_drain_cycles']] = 3
        if block == 3:
            marks.update(barrier_start=99, barrier_end=100)
        for field, value in marks.items():
            row[FIELDS[field]] = base+value
        row[FIELDS['block']] = block+1
        row[FIELDS['queue']] = 2*b+1 if q == 0 else 32+3*b+q-1
        if q == 0:
            requests.append(dict(request=b, prefetch=0, final=0, window_posted=[0]*4, queries=[]))
        requests[-1]['queries'].append(dict(query=q+1, final_posted=0, final_safe=0))
    ticks = np.repeat(rows[None, :, :], CALLS, axis=0)
    data = dict(schema_version=1, group=7, checked_calls=80, ticks_per_us=1,
                ticks_sha256=counter_digest(ticks), passes=[])
    archive = []
    for pass_id, name in enumerate(PASSES):
        cells = []
        for i in range(CALLS):
            tail = 151+i if i < 10 else 250+i
            plan = [[f'{pass_id*100+i:064x}',f'{pass_id*100+i+1:064x}']] * 16
            accounting = dict(invocation=i, stages_us={'resolve_update':tail},
                              intervals_us=dict(li=[[1,2],[2,3],[3,4],[4,5]],resolve_update=[[6,6+tail]]))
            cells.append(dict(call=i+1, traced=traced(name, i+1), requests=requests, layout_receipts=plan,
                              query_owners=dict(owners=64, queues=64, processed=0), accounting=accounting))
            if pass_id == 0:
                archive.append(dict(tail_us=tail, layout_receipts=plan))
        data['passes'].append(dict(name=name, calls=cells, graph_event_us=5000))
    text = '\n'.join(summarize(data, ticks, archive))
    assert 'lower/upper/outside=10/10/0' in text
    assert text.count('near_last=R0Q4 lag_bound_us=1.00') == 20
    assert 'crossover_same_slot_exact_plans=0/20' in text
    assert 'observed_exact_plan_on_minus_off_min/P50/P95/max_us=NA' in text
    ticks[0, 0, 0] += 1
    with pytest.raises(ValueError, match='tick payload'):
        summarize(data, ticks, archive)
