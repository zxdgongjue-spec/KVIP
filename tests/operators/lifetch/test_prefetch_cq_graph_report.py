from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "support"))
import prefetch_cq_graph_report as report


def make_graph():
    # The AIV offsets deliberately differ by orders of magnitude. Only their
    # local elapsed times can be compared, never raw timestamps across AIVs.
    ticks = np.zeros(report.SHAPE, np.int64)
    req = []
    for b in range(16):
        req.append(
            dict(
                request=b,
                prefetch=12,
                final=8,
                queries=[dict(query=q + 1, final_safe=2, final_posted=2) for q in range(4)],
            )
        )
    for call, rows in enumerate(ticks):
        for block, r in enumerate(rows):
            b, q = divmod(block, 4)
            base = 2000 + 100000 * block
            last = block == (3 if call % 2 else 4)
            ts = dict(
                tail_start=1,
                resolve_start=2,
                resolve_end=6,
                adapt_start=42,
                adapt_end=44,
                get_start=9,
                callback_begin=42,
                callback_end=150,
                get_end=179,
                barrier_start=195,
                barrier_end=196 if last else 216,
                index_end=115 if q else 58,
                output_published=100 if q else 80,
            )
            if q:
                ts.update(index_wait=57, index_ready=64)
            if not q:
                ts.update(
                    prefetch_wait=83,
                    prefetch_observed=133,
                    prefetch_return=135,
                    prefetch_ready=137,
                    prefetch_end=140,
                )
            if q == 3:
                ts.update(
                    old_begin=80, old_end=85, gate_begin=87, gate_end=98, finalize_start=180, finalize_end=194
                )
                r[:12] = base + np.arange(181, 193)
                r[12:14] = base + np.array([100, 105])
            for key, n in ts.items():
                r[report.F[key]] = base + n
            for key, n in dict(
                block=block + 1, queue=2 * b + 1 if not q else 32 + 3 * b + q - 1, count=2, safe=2, processed=2
            ).items():
                r[report.F[key]] = n
            if not q:
                for key, n in dict(
                    prefetch_pending=3, prefetch_completed=3, prefetch_polls=41, prefetch_drain_cycles=47
                ).items():
                    r[report.F[key]] = n
            r[64:73] = [2, 2, 0, 4, 12, 2, 2, 2, 2]
            r[80:84] = [base + 10, 0, base + 40, 1]
            r[336:344] = [base + 20, base + 24, 0, 1, base + 152, base + 160, 1, 2]
    arcs = [dict(tail_us=270 if i < 8 else 390, layout_receipts=[("a" * 64, "b" * 64)] * 16) for i in range(20)]
    data = dict(
        schema_version=1,
        group=7,
        checked_calls=80,
        ticks_sha256=report.counter_digest(ticks),
        ticks_per_us=1,
        placement="matched",
        passes=[],
    )
    for name in report.PASSES:
        calls = []
        for i in range(1, 21):
            tail = 290 if i % 2 else 390
            accounting = dict(
                invocation=i - 1,
                stages_us=dict(resolve_update=tail),
                intervals_us=dict(li=[(n, n + 1) for n in range(4)], resolve_update=[(4, 4 + tail)]),
            )
            calls.append(
                dict(
                    call=i,
                    traced=report.traced(name, i),
                    layout_receipts=arcs[i - 1]["layout_receipts"],
                    requests=req,
                    query_owners=dict(owners=64, queues=64, processed=128),
                    accounting=accounting,
                )
            )
        data["passes"].append(dict(name=name, graph_event_us=8000, calls=calls))
    return data, ticks, arcs


def test_private_cq_on_both_sides_of_callback_and_independent_clocks():
    data, ticks, archive = make_graph()
    sample = data["passes"][1]["calls"][0]
    near, q1, q4, capped = report.row_trace(sample, ticks[0], 1)
    assert (report.owner_name(near), near["barrier"], q1["index_release"], q1["pref_begin"]) == (
        "R1Q1",
        1,
        57,
        82,
    )
    assert (q1["drain"], q4["gate"], q4["old"], q4["early"], q4["late"], capped) == (47, 11, 5, 4, 8, 0)
    assert q4["observed_lag"] == 120 and q4["retry_cqs"] == 1
    assert q4["finalize"] == 14 and q4["joins"] == [1] * 4
    lines = report.summarize(data, ticks, archive)
    assert len([s for s in lines if s.startswith("G7 调用")]) == 20
    assert any("超过300微秒的有打点调用=10/20 近末核=R0Q4×10" in s for s in lines)
    assert any("与同位置关闭计划相同=20/20" in s for s in lines)
    # A CQ event during the callback cannot be real on the same scalar core.
    ticks[0, 0, 340] = ticks[0, 0, report.F["callback_end"]] - 1
    with pytest.raises(ValueError, match="while executing"):
        report.row_trace(sample, ticks[0], 1)


def test_conflict_path_and_capped_events():
    data, ticks, _ = make_graph()
    r = ticks[0, 0]
    base = int(r[report.F["tail_start"]] - 1)
    query = dict(query=1, final_safe=0, final_posted=2)
    r[report.F["safe"]] = 0
    for key, time in dict(
        conflict_wait=1,
        prefetch_wait=2,
        prefetch_observed=3,
        prefetch_return=4,
        prefetch_ready=5,
        prefetch_end=6,
        conflict_ready=7,
    ).items():
        r[report.F[key]] = base + time
    assert report.owner(r, 0, query, 1)["conflict"] == 6
    r[66] = 1
    assert report.owner(r, 0, query, 1)["late"] is None
    r[report.F["prefetch_completed"]] = 0
    with pytest.raises(ValueError, match="completion count"):
        report.owner(r, 0, query, 1)
