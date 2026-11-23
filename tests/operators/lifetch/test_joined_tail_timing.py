from contextlib import redirect_stdout
from io import StringIO

import pytest

from support.joined_tail_timing import PASSES, check_calls, compare_slots, local_intervals


def owner(q):
    t = dict(entry=100, topk_end=120, class_published=122, resolve_begin=140,
             resolve_end=180, adapt_begin=270 if q == 0 else 210, adapt_end=278 if q == 0 else 218,
             index_begin=236, index_release=262, barrier_enter=400, barrier_exit=407+q, retire_issued=418,
             first_doorbell=200, last_post=203, completion_done=312, get_begin=198, callback_begin=206,
             callback_end=282, get_end=315, post_active_cycles=5, cq_drain_cycles=22,
             final_posts=1, final_cq_events=1, bucket_words=512, miss_count=1, safe_count=1,
             resolver_restore_bytes=128, resolver_save_bytes=256, index_restore_bytes=0,
             index_save_bytes=0, q4_prefetch_polls=0, q4_prefetch_wait_cycles=0)
    for a, b in [('raw_wait', 'raw_ready'), ('bitmap_wait', 'bitmap_ready'),
                 ('resolver_restore_begin', 'resolver_restore_end'), ('save_begin', 'save_end'),
                 ('conflict_wait', 'conflict_ready'), ('descriptor_sync_begin', 'descriptor_sync_end'),
                 ('classification_wait', 'classification_ready'), ('index_wait', 'index_ready'),
                 ('index_restore_begin', 'index_restore_end'), ('index_save_begin', 'index_save_end'),
                 ('apply_begin', 'apply_end'), ('q1_read_wait', 'q1_read_ready'), ('commit_begin', 'commit_end'),
                 ('final_output_wait', 'final_output_ready')]:
        t[a] = t[b] = 0
    for i in range(6, 12):
        t[f'final_flag{i}_wait'] = t[f'final_flag{i}_ready'] = 0
    t.update(apply_begin=242, apply_end=252,
             prefetch_ready=126 if q == 3 else 0, prefetch_end=137 if q == 3 else 0,
             finalize_begin=345 if q == 3 else 0, finalize_end=375 if q == 3 else 0)
    t.update({f'unused{i}': 0 for i in range(96 - len(t))})
    return t


def test_resolve_and_index_not_double_counted():
    t = owner(3)
    t.update(final_flag6_wait=347, final_flag6_ready=352, raw_wait=145, raw_ready=149,
             resolver_restore_begin=150, resolver_restore_end=158, save_begin=165, save_end=170)
    q = dict(query=4, final_posted=1, final_safe=1)
    v = local_intervals(list(t), list(t.values()), 3, q, 1)
    assert (v['parts']['resolve_work'], v['parts']['row_close']) == (23, 25)
    assert sum(v['parts'].values()) == v['work'] == 300
    assert v['parts']['final_cq'] == 22 and v['barrier'] == 10
    t['raw_ready'] = 290
    with pytest.raises(ValueError, match='overlap'):
        local_intervals(list(t), list(t.values()), 3, q, 1)


def test_slots_and_receipts_bound_before_clock_report():
    owners = [owner(i % 4) for i in range(64)]
    ticks = [list(x.values()) for x in owners]
    rows = [dict(request=b, window_posted=[1]*4, prefetch=4, final=4, queries=[
        dict(query=q, final_posted=1, final_safe=1) for q in range(1, 5)]) for b in range(16)]
    plan = [(f'{i:064x}', 'f'*64) for i in range(16)]
    data = dict(group=7, warmup=4, independent_states=20, source_round=1, checked_calls=80,
                ticks_per_us=10, trace_fields=list(owners[0]), passes=[])
    for name in PASSES:
        calls = []
        for i in range(20):
            accounting = dict(invocation=i, owner_topk_kernel='v9_owner_topk_resolve_update',
                              intervals_us={'li': [(0, 4), (4, 9), (12, 18)], 'resolve_update': [(18, 63+i)]},
                              stages_us={'resolve_update': 45+i})
            on = name == ('odd_on' if i%2 == 0 else 'even_on')
            calls.append(dict(call=i+1, traced=on, score_check=dict(calls=1, queries=64, exact_queries=0),
                              query_owners=dict(owners=64, queues=64, processed=64), layout_receipts=plan,
                              topk_set_digest='a'*64, accounting=accounting, tail_us=45+i, requests=rows,
                              **({'ticks': ticks} if on else {})))
        data['passes'].append(dict(name=name, calls=calls, graph_event_us=1000))
    calls, archive = check_calls(data, {'layout_receipts': [plan]*20}, 7)
    with redirect_stdout(StringIO()) as output:
        compare_slots('G7', data, calls, archive, [45+i for i in range(20)])
    assert 'exact_on_off_work=20/20' in output.getvalue()
    assert 'NEAR_LAST band=lower n=10' in output.getvalue()
    assert 'NEAR_LAST band=upper n=10' in output.getvalue()
    assert 'archive_event_rows=320/320' in output.getvalue()
    data['passes'][1]['calls'][0]['traced'] = False
    with pytest.raises(ValueError, match='Invalid slot'):
        check_calls(data, {'layout_receipts': [plan]*20}, 7)
