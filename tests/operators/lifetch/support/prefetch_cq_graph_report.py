"""Report the G7 final-Get and prefetch waits using only same-AIV clock differences."""

from collections import Counter
from math import isfinite
from statistics import median

import numpy as np

from owner_tail_graph import CALLS, PASSES, SHAPE, check_kernel, counter_digest, traced
from owner_tail_report import FIELDS, percent, span
from query_owner_report import checked_layout_plan


# Additional scalar fields in the "Q1 drains prefetch" trace. 64..72 remain
# the KVAF trace header shared with the original diagnostic.
F = dict(
    FIELDS,
    old_begin=62,
    old_end=63,
    gate_begin=73,
    gate_end=74,
    output_published=75,
    callback_end=76,
    callback_begin=77,
)


def read_queue(raw, v):
    """Separate actual scalar polling before/after the callback from its wall time."""
    doors, cqs, overflow = raw[64:67]
    total_doors, total_cqs, groups, completed = raw[69:73]
    if (
        not 0 <= doors <= 128
        or not 0 <= cqs <= 128
        or groups != completed
        or not 0 <= groups <= total_doors
        or not 0 <= groups <= total_cqs
    ):
        raise ValueError("Private SQ/CQ trace has inconsistent counts")
    if not v["count"]:
        if any((doors, cqs, total_doors, total_cqs, groups, overflow, v["get_start"], v["get_end"])):
            raise ValueError("Empty final Get contains SQ/CQ events")
        return dict(
            first=None, last=None, early=0, late=0, groups=0, capped=False, observed_lag=None, retry_cqs=0
        )
    if not (
        doors
        and cqs
        and groups
        and 0 < v["get_start"] <= v["callback_begin"] <= v["callback_end"] <= v["get_end"]
    ):
        raise ValueError("Missing final Get or callback on a nonempty owner")
    if doors != min(total_doors, 128) or cqs != min(total_cqs, 128):
        raise ValueError("Incomplete private queue event table")
    door_ticks = raw[80 : 80 + 2 * doors : 2]
    if (
        door_ticks != sorted(door_ticks)
        or door_ticks[0] < v["get_start"]
        or door_ticks[-1] > v["callback_begin"]
    ):
        raise ValueError("Doorbell outside the private posting interval")
    door_by_group = dict(zip(raw[81 : 81 + 2 * doors : 2], door_ticks))
    before, after, last_observation = 0, 0, v["get_start"]
    delays, retries_after = [], 0
    for j in range(cqs):
        begin, seen, group, polls = raw[336 + 4 * j : 340 + 4 * j]
        if polls <= 0:
            raise ValueError("Observed a private CQ entry without a probe")
        if not last_observation <= begin <= seen <= v["get_end"]:
            raise ValueError("Invalid private CQ observation interval")
        if not overflow and (group not in door_by_group or seen < door_by_group[group]):
            raise ValueError("CQ observation without its submitted group")
        if not overflow:
            delays.append(seen - door_by_group[group])
        if seen <= v["callback_begin"]:
            before += seen - begin
        elif begin >= v["callback_end"]:
            after += seen - begin
            retries_after += int(polls > 1)
        else:
            raise ValueError("One AIV cannot poll its private CQ while executing its callback")
        last_observation = seen
    return dict(
        first=door_ticks[0] - v["tail_start"],
        last=None if overflow else door_ticks[-1] - v["tail_start"],
        early=None if overflow else before,
        late=None if overflow else after,
        groups=groups,
        capped=bool(overflow),
        observed_lag=None if overflow else max(delays),
        retry_cqs=None if overflow else retries_after,
    )


def owner(raw, block, query, mhz):
    raw = raw.tolist()
    v = {k: raw[i] for k, i in F.items()}
    q = block % 4
    if (
        len(raw) != SHAPE[2]
        or v["block"] != block + 1
        or query["query"] != q + 1
        or v["queue"] != (2 * (block // 4) + 1 if q == 0 else 32 + 3 * (block // 4) + q - 1)
        or not 0 <= v["safe"] <= v["count"] <= 2048
        or v["count"] != v["processed"]
        or (v["safe"], v["count"]) != (query["final_safe"], query["final_posted"])
    ):
        raise ValueError(f"Final Get receipt and trace disagree on owner {block}")

    def us(a, b, optional=False):
        return span(v[a], v[b], optional) / mhz

    def relative(key):
        return us("tail_start", key)

    private = read_queue(raw, v)
    result = {
        k: (val / mhz if val is not None else None)
        for k, val in private.items()
        if k in ("first", "last", "early", "late", "observed_lag")
    }
    result.update(
        resolve=us("resolve_start", "resolve_end"),
        adapt=us("adapt_start", "adapt_end"),
        apply=span(raw[58], raw[59], True) / mhz,
        groups=private["groups"],
        retry_cqs=private["retry_cqs"],
        capped=private["capped"],
        block=block,
        barrier=us("barrier_start", "barrier_end"),
        work=us("tail_start", "barrier_start"),
        callback=us("callback_begin", "callback_end"),
        index_wait=us("index_wait", "index_ready", q == 0),
        conflict=us("conflict_wait", "conflict_ready", v["safe"] == v["count"]),
        old=us("old_begin", "old_end", True),
        gate=us("gate_begin", "gate_end", True),
        commit=span(raw[12], raw[13], True) / mhz,
    )
    if q == 0:
        if (
            v["prefetch_pending"] != v["prefetch_completed"]
            or v["prefetch_polls"] < v["prefetch_completed"]
            or not (v["prefetch_wait"] <= v["prefetch_return"] <= v["prefetch_ready"] <= v["prefetch_end"])
        ):
            raise ValueError("Q1 prefetch completion count or timestamps disagree")
        if v["prefetch_pending"] and not v["prefetch_wait"] <= v["prefetch_observed"] <= v["prefetch_return"]:
            raise ValueError("Missing final prefetch CQ observation")
        if v["safe"] == v["count"]:
            if (
                not v["index_end"]
                <= v["output_published"]
                <= v["prefetch_wait"]
                <= v["prefetch_end"]
                <= v["callback_end"]
            ):
                raise ValueError("Safe Q1 lost its early Index or output publication")
        elif (
            not v["conflict_wait"]
            <= v["prefetch_wait"]
            <= v["prefetch_end"]
            <= v["conflict_ready"]
            <= v["get_start"]
        ):
            raise ValueError("Q1 posted conflicting writes before prefetch completion")
        result.update(
            pref_begin=relative("prefetch_wait"),
            pref_publish=relative("prefetch_end"),
            drain=v["prefetch_drain_cycles"] / mhz,
            polls=v["prefetch_polls"],
            pending=v["prefetch_pending"],
            index_release=relative("index_end"),
            output=relative("output_published"),
        )
    elif any(
        v[k] for k in ("prefetch_wait", "prefetch_ready", "prefetch_end", "prefetch_pending", "prefetch_polls")
    ):
        raise ValueError("Only Q1 consumes prefetch CQ")
    if q == 3:
        if v["count"] == v["safe"]:
            if not (v["callback_begin"] <= v["gate_begin"] <= v["gate_end"] <= v["callback_end"]):
                raise ValueError("Missing safe Q4 prefetch gate in the callback")
        elif v["gate_begin"] or v["gate_end"] or v["conflict_ready"] > v["get_start"]:
            raise ValueError("Unexpected Q4 prefetch gate for conflicting slots")
        result.update(
            joins=[span(raw[2 * i], raw[2 * i + 1]) / mhz for i in range(2, 6)],
            finalize=us("finalize_start", "finalize_end"),
        )
    return result


def row_trace(call, ticks, mhz):
    requests = call["requests"]
    if len(requests) != 16:
        raise ValueError("Incomplete batch")
    owners = []
    for b, row in enumerate(requests):
        if row["request"] != b or len(row["queries"]) != 4:
            raise ValueError("Incorrect row/Q mapping")
        for q, query in enumerate(row["queries"]):
            block = 4 * b + q
            owners.append(owner(ticks[block], block, query, mhz))
    # SyncAll cannot finish before the last arrival. The shortest observed
    # barrier duration gives an upper bound on the selected block's arrival lag.
    near = min(owners, key=lambda item: item["barrier"])
    b = near["block"] // 4
    q1, q4 = owners[4 * b], owners[4 * b + 3]
    return near, q1, q4, sum(x["capped"] for x in owners)


def fmt(value):
    return "NA" if value is None else f"{value:.2f}"


def joined(*values):
    return "/".join(map(fmt, values))


def owner_name(o):
    b, q = divmod(o["block"], 4)
    return f"R{b}Q{q + 1}"


def summarize(data, ticks, archived_calls):
    if (
        data["schema_version"] != 1
        or data["group"] != 7
        or data["checked_calls"] != 80
        or len(archived_calls) != CALLS
        or [p["name"] for p in data["passes"]] != list(PASSES)
        or ticks.shape != SHAPE
        or ticks.dtype != np.int64
        or counter_digest(ticks) != data["ticks_sha256"]
        or not isfinite(data["ticks_per_us"])
        or data["ticks_per_us"] <= 0
    ):
        raise ValueError("Incomplete G7 CQ diagnostic or invalid clock calibration")
    state = {}
    archived = {checked_layout_plan(c["layout_receipts"], state, "archive") for c in archived_calls}
    original = [c["tail_us"] for c in archived_calls]
    stages = {}
    lines = [
        f"G7 Q1接手预取回执版 现有归档尾段P50/P95微秒={joined(median(original), percent(original, 0.95))} "
        f"超过300微秒={sum(t > 300 for t in original)}/20",
        "G7 本次只测Q1接手预取回执。以下时间单位为微秒，门铃和发布时刻均从本核进入尾段开始计算。",
    ]
    names = dict(off_before="先全关", odd_on="奇数调用开启", even_on="偶数调用开启", off_after="后全关")
    for p in data["passes"]:
        mode = p["name"]
        cells = p["calls"]
        if len(cells) != CALLS or not (isfinite(p["graph_event_us"]) and p["graph_event_us"] > 0):
            raise ValueError(f"Incomplete G7 pass {mode}")
        samples = []
        for i, c in enumerate(cells, 1):
            if (
                c["call"] != i
                or c["traced"] != traced(mode, i)
                or c["query_owners"]
                != dict(owners=64, queues=64, processed=sum(r["final"] for r in c["requests"]))
            ):
                raise ValueError(f"Wrong trace slot or private CQ receipts in pass {mode} call {i}")
            samples.append(
                (check_kernel(c["accounting"], i), checked_layout_plan(c["layout_receipts"], state, mode), c)
            )
        stages[mode] = samples
        tail = [s[0] for s in samples]
        lines.append(
            f"G7 {names[mode]} 整图均值={p['graph_event_us'] / 20:.2f} 尾段P50/P95={joined(median(tail), percent(tail, 0.95))} "
            f"超过300微秒={sum(t > 300 for t in tail)}/20"
        )
    cohorts = {False: [], True: []}
    bracket, matched, deltas, all_deltas, archived_matches = [], 0, [], [], 0
    for i in range(CALLS):
        on, off = ("odd_on", "even_on") if i % 2 == 0 else ("even_on", "odd_on")
        tail, plan, call = stages[on][i]
        other, other_plan, _ = stages[off][i]
        all_deltas.append(tail - other)
        if plan == other_plan:
            deltas.append(tail - other)
        pair = [stages[k][i] for k in ("off_before", "off_after")]
        bracket.append(pair[1][0] - pair[0][0])
        matched += int(all(s[1] == plan for s in pair))
        archived_matches += int(plan in archived)
        near, q1, q4, capped = row_trace(call, ticks[i], data["ticks_per_us"])
        cohorts[tail > 300].append((near, q1, q4))
        traffic = call["requests"]
        row = traffic[near["block"] // 4]
        lines.append(
            f"G7 调用{i + 1} 尾段开/同位置关={joined(tail, other)} 搬移计划相同={int(plan == other_plan)} "
            f"近末核={owner_name(near)} 到达滞后上界={near['barrier']:.2f} 近末核工作={near['work']:.2f} "
            f"整批预取/补取数={sum(r['prefetch'] for r in traffic)}/{sum(r['final'] for r in traffic)} "
            f"该行Q1/Q4补取数={row['queries'][0]['final_posted']}/{row['queries'][3]['final_posted']} "
            f"近末核分配/输出适配/等Index/应用Index交换={joined(near['resolve'], near['adapt'], near['index_wait'], near['apply'])} "
            f"近末核最终回执早批/回调后={joined(near['early'], near['late'])} "
            f"该行Q1交接/读完旧映射/开始收预取/发布预取={joined(q1['index_release'], q1['output'], q1['pref_begin'], q1['pref_publish'])} "
            f"Q1排预取={q1['drain']:.2f} 预取回执数/探测次数={q1['pending']}/{q1['polls']} "
            f"Q1最终批数/回调耗时/回调后收自己回执={q1['groups']}/{joined(q1['callback'], q1['late'])} "
            f"Q4首次/末次最终门铃={joined(q4['first'], q4['last'])} Q4最终批数={q4['groups']} "
            f"Q4冲突等待/等Index/等旧映射/提交前等预取/写映射={joined(q4['conflict'], q4['index_wait'], q4['old'], q4['gate'], q4['commit'])} "
            f"Q4最终回执早批/回调后={joined(q4['early'], q4['late'])} Q1/Q4回调后首次探测未就绪的最终回执数="
            f"{q1['retry_cqs']}/{q4['retry_cqs']} Q4最慢批末门铃至软件见回执={fmt(q4['observed_lag'])} Q4收尾={q4['finalize']:.2f} "
            f"收尾内等Q1到Q4={joined(*q4['joins'])} 事件截断核数={capped}"
        )
    for slow, group in cohorts.items():
        if not group:
            continue
        leader = Counter(owner_name(n) for n, _, _ in group)
        counts = ",".join(f"{k}×{n}" for k, n in leader.most_common())

        def typical(index, key):
            values = [item[index][key] for item in group if item[index][key] is not None]
            return median(values) if values else None

        lines.append(
            f"G7 {'超过' if slow else '不超过'}300微秒的有打点调用={len(group)}/20 近末核={counts} "
            f"同一行中位数 Q1排预取={typical(1, 'drain'):.2f} Q4等Index/等旧映射/提交前等预取="
            f"{joined(typical(2, 'index_wait'), typical(2, 'old'), typical(2, 'gate'))} "
            f"Q1/Q4回调后收最终回执={joined(typical(1, 'late'), typical(2, 'late'))} "
            f"Q4收尾内等Q1={median(x[2]['joins'][0] for x in group):.2f}"
        )
    lines.append(
        f"G7 条件核对 内存放置={data.get('placement', '未验证')} 开启调用与生产归档计划相同={archived_matches}/20 "
        f"与同位置关闭计划相同={len(deltas)}/20 与前后全关计划都相同={matched}/20 "
        f"同位置开减关P50={median(all_deltas):+.2f} 同计划开减关P50={fmt(median(deltas)) if deltas else 'NA'} "
        f"两次全关同位置之差最小/中位/最大={joined(min(bracket), median(bracket), max(bracket))}"
    )
    lines.append("G7 回执记录的是软件看到完成的时间。Q4收尾已包含等各个Q的时间。各核并行的等待不能相加。")
    return lines
