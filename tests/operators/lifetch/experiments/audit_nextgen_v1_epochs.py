#!/usr/bin/env python3
"""Offline audit of the frozen Q4 noncausal Generic LI rotation schedule."""
import json,re,sys,runpy
from pathlib import Path

def audit(geometry, trigger_tokens, k, h, r, windows, aics):
    b=len(k)
    assert all(r<=n<=h and n%128==0 for n in k)
    # Frozen SplitCore: noncausal, isLDOpen=false, S1=4/2 when splitQ.
    # Every nonempty Q group owns its entire S2; scope excludes inactive Q.
    g=geometry(b,"auto",windows,aics)
    per=g["cores_per_batch"]
    assert g["logical_aic_tasks"]==b*per
    assert g["rotating_aic"]==b*per
    triggers=trigger_tokens(b,h,r,"auto",windows,aics)
    rows=[]
    for e,(cores,trigger) in enumerate(zip(g["service_groups"],triggers)):
        for c in cores:
            req=c//per
            rows.append(dict(epoch=e,core=c,request=req,query_group=c%per,
                             actual_k=k[req],last_tile=k[req]//128,trigger_tile=trigger//128,
                             reachable=trigger<=k[req]))
    return g,rows

def main():
    root,op,out=map(Path,sys.argv[1:])
    case,=root.glob("basic_bs4/*/cases/basic_bs4/case0-b4-prototype-graph")
    config=json.loads((case.parent/"config.json").read_text())
    assert config["varlen_probe"] and config["queries"]==4 and config["batches"]==[4]
    assert config["history"]==131072 and config["resident"]==16384
    assert config["intermediate_fetches"]==8
    log=(case/"native.log").read_text()
    matches=re.findall(r"V1_ACTUAL_LENGTHS (\[[^\n]+\]) Q4 sparse_mode=0",log)
    assert len(matches)==1,matches
    k=json.loads(matches[0]);assert k==[32768,65536,98304,131072]
    assert "NPU7_MAPPING_PASS" in (root/"run.log").read_text()
    assert "'effective_aic': 32" in (root/"run.log").read_text()
    trace=(case/"error.log").read_text()
    assert "line 1924, in measure" in trace and "507014" in trace
    # Import the exact frozen pure-Python geometry module, not installed kvip.
    funcs=runpy.run_path(str(op/"python/kvip/operators/lifetch/geometry.py"))
    g,rows=audit(funcs["li_geometry"],funcs["trigger_tokens"],k,config["history"],config["resident"],config["intermediate_fetches"],32)
    missing=[x["epoch"] for x in rows if not x["reachable"]]
    assert missing==[4,5,6],missing
    assert [x["core"] for x in rows]==[0,2,4,6,1,3,5,7]
    assert g["service_aics_per_window"]==1 and g["service_aivs_per_window"]==2
    # Negative control: uniform history makes all original triggers reachable.
    _,uniform=audit(funcs["li_geometry"],funcs["trigger_tokens"],[config["history"]]*4,config["history"],config["resident"],8,32)
    assert all(x["reachable"] for x in uniform)
    for row in rows:print("V1_EPOCH",json.dumps(row),flush=True)
    report=dict(scope="static Q4 noncausal Generic SplitCore reconstruction; not device trace",
                geometry=g,rows=rows,missing_epochs=missing,
                missing_event_records=2*len(missing),uniform_control_all_reachable=True)
    (out/"epoch_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print("V1_EPOCH_REACHABILITY_REPRODUCED missing=4,5,6 missing_event_records=6")
    print("V1_UNIFORM_REACHABILITY_CONTROL_PASS")
if __name__=="__main__":main()
