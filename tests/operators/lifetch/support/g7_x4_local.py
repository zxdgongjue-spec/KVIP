# SPDX-License-Identifier: Apache-2.0
"""Resolve same-host NPU4 evidence independently for each G7 batch."""
import argparse
import json
import os
from pathlib import Path

from b24_window_policy import ADVANCE, CREDIT_CEILINGS, MODERATE, policy
from b24_window_report import cached, stats
from large_batch_placement import field, placement

BASE = "d0fc5963881bd38fa97eff0f6ef40c757a6e8713"
ORIGINAL = "e602ff4a688a733664aaabe84aa30732b9e68062"
SDK = "813f1750a50b7e6c92d5f5397f2fbc05befef6da"
TREE = "1de18e8315137d7a0227c3ae4c82afd45b5fe743"
MODE = "window-x4-early"

def content(path):
    return path.read_text().strip() if path.is_file() else ""

def here(root):
    if any(field(root,key) != expected for key,expected in
           (("host",os.uname().nodename),("physical_npu","4"),("native",BASE),("kvaf",SDK),("kvaf_tree",TREE))):
        return False
    for local, key in ((Path("/etc/machine-id"),"machine_id"),(Path("/proc/sys/kernel/random/boot_id"),"boot_id")):
        known, recorded = content(local), field(root,key)
        if known and recorded:
            return known == recorded
    return False

def read(p):
    return json.loads(p.read_bytes())

def valid_local(arm,batch,mode=MODE):
    if not (arm/"local_placement_measured").is_file() or not here(arm.parents[3]):
        raise ValueError(f"Local host/card receipt missing: {arm}")
    placement(arm/"thp.json")
    data=read(arm/"packed_result.json")
    ranks, grants=policy(mode)
    identity=data["identity"]
    if (read(arm/"window-study.json") != dict(mode=mode,reference=None)
            or any(identity[k]!=v for k,v in (("group",7),("batch",batch),("native",BASE),
                  ("physical_npu",4),("calls",10),("host",os.uname().nodename)))
            or len(identity["sequence_sha256"]) != 64 or not data.get("tail_text") or not data.get("original_library")
            or data["correctness"] != "PASS" or data["calls"] != 10 or data["get_workers"] != 32
            or data["full_KV_Index_checked_calls"] != 10 or data["checked_rows"] != 10*batch
            or data["checked_queries"] != 40*batch or data["checked_rank_queries"] != 4*batch
            or data["actual_ranks"] != list(ranks) or data["window_grants_per_row"] != list(grants)):
        raise ValueError(f"Invalid local G7 BS{batch} evidence: {arm}")
    if mode in CREDIT_CEILINGS:
        check_earned_limits(data,batch,mode)
    return data

def check_earned_limits(result,batch,mode=ADVANCE):
    ceilings=CREDIT_CEILINGS[mode]
    per_window=tuple(policy(ADVANCE)[1])
    if len(result["traffic"])!=10:
        raise ValueError("Expected all ten window journals")
    for call in result["traffic"]:
        rows=call["row_window_prefetch"]
        if len(rows)!=batch:
            raise ValueError("Missing measured row credit")
        for row in rows:
            used=earned=0
            if len(row)!=5:
                raise ValueError("Expected five epochs")
            for count,grant,cap in zip(row,per_window,ceilings,strict=True):
                earned+=grant
                used+=count
                if not 0<=count<=cap or used>earned:
                    raise ValueError("Measured grant exceeds its cumulative or local ceiling")

def known_readers(reader,mode):
    if mode == MODERATE:
        return (reader,)
    if mode == ADVANCE:
        return ("fe37fdc78a3292eef34b71334437dd768380f583",reader)
    return (ORIGINAL, "a3970e5fc8b9e29bdde38ffc8b047746653db00c",
            "24e402be219f4b16f17b67bb5a56208939574e82", reader)

def choose(area,batch,reader,mode=MODE):
    roots=sorted(area.glob("fixed-q4-four-ranks-one-block.*"),key=lambda p:p.stat().st_mtime_ns,reverse=True)
    origins=[]
    for root in roots:
        base=root/f"g7/bs{batch}/block1/fused"
        if (here(root) and all((base/k).is_file() for k in ("thp.json","packed_result.json","placement_verified"))):
            origins.append(root)
            for checkout in known_readers(reader,mode):
                hit=cached(checkout,root,7,batch,mode)
                if hit:
                    return "paired-cache",root,hit
    if origins:
        return "paired-run",origins[0],"-"
    for root in sorted(area.glob(f"fixed-q4-audit-{mode}.*"),key=lambda p:p.stat().st_mtime_ns,reverse=True):
        if not here(root) or field(root,"audit_reader") not in known_readers(reader,mode):
            continue
        arm=root/f"g7/bs{batch}/block1/{mode}"
        if (arm/"local_placement_measured").is_file():
            valid_local(arm,batch,mode)
            return "local-cache","-",arm
    return "local-run","-","-"

def show_local(arm,batch,mode=MODE):
    data=valid_local(arm,batch,mode)
    print(f"G7 BS{batch} {mode} CHECK TopK_score_vs_captured_scores/KV/Index/private_CQ=PASS "
          f"calls=10/10 queries={40*batch}/{40*batch} rows={10*batch}/{10*batch}")
    print(f"G7 BS{batch} {mode} graph/LI/LI1/LI23/tail_mean_us="+"/".join(f"{t:.2f}" for t in stats(data))+
          " vs_four_window=UNAVAILABLE no_same_card_four_window_archive vs_serial=UNVERIFIED")

def actual_traffic(r):
    rows=[call["posted_per_window"] for call in r["traffic"]]
    return [sum(v)/len(rows) for v in zip(*rows)], sum(c["final"] for c in r["traffic"])/len(rows)

def compare_new(area,batch,reader):
    pieces=[]
    for mode in (MODE,ADVANCE):
        state, ref, arm=choose(area,batch,reader,mode)
        if state not in ("local-cache","paired-cache"):
            pieces.append(None)
            continue
        if state=="paired-cache":
            from b24_window_report import values
            r,_,_=values(arm,ref,7,batch,mode)
        else:
            r=valid_local(arm,batch,mode)
        if mode==ADVANCE:
            check_earned_limits(r,batch)
        posts, final=actual_traffic(r)
        peak=max(sum(v) for c in r["traffic"] for v in c["row_window_prefetch"])
        print(f"G7 BS{batch} {mode} measured_prefetch_by_window_mean="
              +"/".join(f"{v:.1f}" for v in posts)+f" final_GET_mean={final:.1f} max_row_credit={peak}/1024")
        pieces.append((r,placement(arm/"thp.json")))
    if any(item is None for item in pieces):
        print(f"G7 BS{batch} vs_five_original=UNAVAILABLE missing_checked_reference")
        return
    (old, old_placement), (new, new_placement)=pieces
    if (old_placement!=new_placement or old["identity"]!=new["identity"]
            or old["tail_text"]!=new["tail_text"] or old["original_library"]!=new["original_library"]):
        print(f"G7 BS{batch} vs_five_original=UNAVAILABLE mismatched_input_placement_or_tail")
        return
    print(f"G7 BS{batch} vs_five_original historical_same_host graph/LI/LI1/LI23/tail_saved_mean_us="
          +"/".join(f"{a-b:+.2f}" for a,b in zip(stats(old),stats(new))))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("choose","show-local","compare"))
    p.add_argument("path",type=Path)
    p.add_argument("batch",type=int,choices=(28,32))
    p.add_argument("reader",nargs="?")
    p.add_argument("--mode",choices=(MODE,*CREDIT_CEILINGS),default=MODE)
    a=p.parse_args()
    if a.action=="choose":
        print(*choose(a.path,a.batch,a.reader,a.mode),sep="\t")
    elif a.action=="show-local":
        show_local(a.path,a.batch,a.mode)
    else:
        compare_new(a.path,a.batch,a.reader)
