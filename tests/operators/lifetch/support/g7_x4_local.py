# SPDX-License-Identifier: Apache-2.0
"""Resolve same-host NPU4 evidence independently for each G7 batch."""
import argparse
import json
import os
from pathlib import Path

from b24_window_policy import policy
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

def valid_local(arm,batch):
    if not (arm/"local_placement_measured").is_file() or not here(arm.parents[3]):
        raise ValueError(f"Local host/card receipt missing: {arm}")
    placement(arm/"thp.json")
    data=read(arm/"packed_result.json")
    ranks, grants=policy(MODE)
    identity=data["identity"]
    if (read(arm/"window-study.json") != dict(mode=MODE,reference=None)
            or any(identity[k]!=v for k,v in (("group",7),("batch",batch),("native",BASE),
                  ("physical_npu",4),("calls",10),("host",os.uname().nodename)))
            or len(identity["sequence_sha256"]) != 64 or not data.get("tail_text") or not data.get("original_library")
            or data["correctness"] != "PASS" or data["calls"] != 10 or data["get_workers"] != 32
            or data["full_KV_Index_checked_calls"] != 10 or data["checked_rows"] != 10*batch
            or data["checked_queries"] != 40*batch or data["checked_rank_queries"] != 4*batch
            or data["actual_ranks"] != list(ranks) or data["window_grants_per_row"] != list(grants)):
        raise ValueError(f"Invalid local G7 BS{batch} evidence: {arm}")
    return data

def choose(area,batch,reader):
    roots=sorted(area.glob("fixed-q4-four-ranks-one-block.*"),key=lambda p:p.stat().st_mtime_ns,reverse=True)
    origins=[]
    for root in roots:
        base=root/f"g7/bs{batch}/block1/fused"
        if (here(root) and all((base/k).is_file() for k in ("thp.json","packed_result.json","placement_verified"))):
            origins.append(root)
            for checkout in (ORIGINAL,reader):
                hit=cached(checkout,root,7,batch,MODE)
                if hit:
                    return "paired-cache",root,hit
    if origins:
        return "paired-run",origins[0],"-"
    for root in sorted(area.glob(f"fixed-q4-audit-{MODE}.*"),key=lambda p:p.stat().st_mtime_ns,reverse=True):
        if not here(root) or field(root,"audit_reader") != reader:
            continue
        arm=root/f"g7/bs{batch}/block1/{MODE}"
        if (arm/"local_placement_measured").is_file():
            valid_local(arm,batch)
            return "local-cache","-",arm
    return "local-run","-","-"

def show_local(arm,batch):
    data=valid_local(arm,batch)
    print(f"G7 BS{batch} {MODE} CHECK TopK_score_vs_captured_scores/KV/Index/private_CQ=PASS "
          f"calls=10/10 queries={40*batch}/{40*batch} rows={10*batch}/{10*batch}")
    print(f"G7 BS{batch} {MODE} graph/LI/LI1/LI23/tail_mean_us="+"/".join(f"{t:.2f}" for t in stats(data))+
          " vs_four_window/serial=UNAVAILABLE no_same_card_archive")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("choose","show-local"))
    p.add_argument("path",type=Path)
    p.add_argument("batch",type=int,choices=(28,32))
    p.add_argument("reader",nargs="?")
    a=p.parse_args()
    if a.action=="choose":
        print(*choose(a.path,a.batch,a.reader),sep="\t")
    else:
        show_local(a.path,a.batch)
