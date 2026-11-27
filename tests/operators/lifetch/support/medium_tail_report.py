# SPDX-License-Identifier: Apache-2.0
"""Compare completed same-host natural ten-call graphs without launching a control."""
from statistics import mean

from b24_window_policy import ADVANCE, MEDIUM
from b24_window_report import stats, values
from g7_x4_local import MODE, check_earned_limits, choose, placement, valid_local


def plan(own, total, medium):
    if not own:
        return ()
    if total >= 256:
        ends = sorted({(own*i+15)//16 for i in (1,6,11)} | {own})
    elif medium and total >= 192 and own >= 64:
        ends = [(own+1)//2, own]
    else:
        ends = [own]
    return tuple(r-l for l,r in zip((0,*ends[:-1]),ends))


def traffic(record, medium):
    receipts = record.get("measured_private_final_CQEs", ())
    if medium and len(receipts) != 10:
        raise ValueError("Missing actual private CQ receipts for measured calls")
    runs = []
    for i, call in enumerate(record["traffic"]):
        rows = call["row_query_final"]
        if len(rows) != record["identity"]["batch"] or any(len(row)!=4 for row in rows):
            raise ValueError("Missing tail row/query counters")
        plans = [[plan(q, sum(row), medium) for q in row] for row in rows]
        counts = [[len(q) for q in row] for row in plans]
        if medium and counts != receipts[i]:
            raise ValueError("Native CQ receipts disagree with planned medium-row splits")
        qs = [b for r in plans for b in r if b]
        if sum(sum(b) for b in qs)!=call["final"]:
            raise ValueError("Missing tail token in batch reconstruction")
        runs.append((call["final"], sum(b[0] for b in qs), max((b[0] for b in qs),default=0),
                     sum(map(sum,counts))))
    return [mean(xs) for xs in zip(*runs)]


def reference(area, batch, reader, mode):
    state, ref, arm = choose(area, batch, reader, mode)
    if state == "paired-cache":
        r, _, _ = values(arm, ref, 7, batch, mode)
    elif state == "local-cache":
        r = valid_local(arm,batch,mode)
    else:
        return None
    if mode in (ADVANCE, MEDIUM):
        check_earned_limits(r,batch)
    return r, placement(arm/"thp.json")


def compare(area,batch,reader):
    baseline = reference(area,batch,reader,MEDIUM)
    if baseline is None:
        raise ValueError("Medium-tail result missing after completion")
    new, numa = baseline
    if not new.get("medium_tail") or not new.get("original_tail_text") or new["original_tail_text"] == new["tail_text"]:
        raise ValueError("Missing experimental tail image identity")
    fmt = lambda xs: "/".join(f"{x:.2f}" for x in xs)
    print(f"G7 BS{batch} medium_tail final/first_all/max_Q_first/actual_final_CQEs={fmt(traffic(new,True))}")
    for mode in (ADVANCE, MODE):
        tag=f"G7 BS{batch} medium_tail_vs_{mode}"
        oldref=reference(area,batch,reader,mode)
        if oldref is None:
            print(f"{tag} UNAVAILABLE missing_same_host_NPU4_archive")
            continue
        old, old_numa = oldref
        if (numa != old_numa or old["identity"] != new["identity"] or old["tail_text"] != new["original_tail_text"]
                or old["original_library"] != new["original_library"]):
            print(f"{tag} UNAVAILABLE input_HOT_or_ordinary_tail_mismatch")
            continue
        if mode == ADVANCE and old["cold_build_fingerprint"] != new["cold_build_fingerprint"]:
            print(f"{tag} UNAVAILABLE measured_LI_fingerprint_mismatch")
            continue
        same=sum((a["row_query_final"],a["row_window_prefetch"]) == (b["row_query_final"],b["row_window_prefetch"])
                 for a,b in zip(old["traffic"],new["traffic"],strict=True))
        gain=[a-b for a,b in zip(stats(old),stats(new))]
        print(f"{tag} archived_same_card graph/LI/LI1/LI23/tail_saved_mean_us={fmt(gain)} "
              f"matched_row_Q_counts={same}/10 identical_LI_build={int(old['cold_build_fingerprint']==new['cold_build_fingerprint'])} "
              f"old_final/first_all/max_Q_first/planned_final_CQEs={fmt(traffic(old,False))}")
