import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
from joined_tail_archive import describe, matched_halves, workload


def example(invocation, plan):
    requests = []
    for row in range(16):
        counts = (1, 5, 0, 1) if row == 3 else (0, 0, 0, 0)
        queries = [dict(query=q+1, unique=2048, new_unique=2048, cross_q_duplicate=0,
                        final_posted=count, final_safe=min(1, count)) for q, count in enumerate(counts)]
        requests.append(dict(request=row, final=sum(counts), prefetch=4, selected_unique=8192, queries=queries))
    sample = dict(accounting=dict(invocation=invocation, stages_us=dict(resolve_update=20+invocation)),
                  algorithm=dict(requests=requests, totals=dict(prefetch=64, final=7, selected_unique=131072)))
    return sample, (plan,) * 16


def test_fast_and_slow_workload_and_full_plan_are_distinct(capsys):
    rows = [workload(7, *example(i, 0 if i < 11 else 1)) for i in range(20)]
    first = rows[0]
    assert (first['active'], first['safe'], first['unsafe'], first['max_row'], first['max_owner'],
            first['owner']) == (3, 3, 4, 7, 5, 'R3Q2')
    assert matched_halves(rows[:10], rows[10:]) == dict(full_plan=1, row_plans=16, workload=10)
    describe(7, rows[::-1])
    output = capsys.readouterr().out
    assert 'FASTEST_10 mean/P50 tail=24.50/24.50' in output
    assert 'SLOWEST_10 mean/P50 tail=34.50/34.50' in output
    assert 'row_plan_matches=16/160 owner_count_vector_matches=10/10' in output


def test_inconsistent_safe_count_is_rejected():
    sample, plan = example(0, 0)
    sample['algorithm']['requests'][0]['queries'][0]['final_safe'] = 2
    with pytest.raises(ValueError, match='invalid query counts'):
        workload(2, sample, plan)
