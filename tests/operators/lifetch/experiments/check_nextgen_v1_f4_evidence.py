#!/usr/bin/env python3
"""Verify successful F4 raw evidence before F5; CPU-only."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
root,lib,out=map(Path,sys.argv[1:])
assert (root/"test_commit.txt").read_text().strip()=="2afd1a84b304adeb87c406242174ab0278132cb7"
assert (root/"physical_device.txt").read_text().strip()=="7"
assert "NEXTGEN_V1_F4_GRAPH_PASS" in (root/"run.log").read_text().splitlines()
summary,=(root/"basic_bs4").glob("*/summary.json")
row,=json.loads(summary.read_text())["cases"]
assert row["case"]=="basic_bs4/f4_graph" and row["accepted"] is True and row["correctness"]=="PASS"
worker=summary.parent/"cases/basic_bs4"
exitfile=worker/"worker-result.json"
assert json.loads(exitfile.read_text())["exit_code"]==0
report,=worker.glob("case0-b4-prototype-graph/report.json")
d=report.parent
r=json.loads(report.read_text())
assert r["correctness"]=="PASS" and r["first_round_verified"]==4 and r["timing"]=={}
assert r["actual_k"]==[32768,65536,98304,131072] and r["actual_q"]==[4]*4
assert r["exact_same_topk"] is True and r["production_lease_type"]=="KvafGetLease"
assert r["explicit_generic_streaming"] is True and r["parallel_terminal"] is False
assert (worker/"universal-build/libv9_bridge.so").resolve()==lib.resolve()
assert hashlib.sha256(lib.read_bytes()).hexdigest()=="fc56dc00b505a552cf71356188fc93790e9082cbcf32f7d6e64fab5265c52261"
files=[root/"run.log",root/"test_commit.txt",root/"physical_device.txt",summary,exitfile,report,d/"native.log"]
assert len(r["checks"])==4
for i,stem in enumerate(("f3","f4_replay_01","f4_replay_02","f4_replay_03")):
    c=r["checks"][i]
    assert c=={"reads":{"prefetch":559,"final":494},"completion":{"final_reads":494,"nonconflicting_reads":494,"conflicting_reads":0,"native_deferred_rows":0,"completed_checkpoints":8+4*i}},c
    evpath=d/(stem+"_event_evidence.json")
    rawpath=d/(stem+"_event_evidence.npz")
    topk=d/(stem+"_topk.npz")
    ev=json.loads(evpath.read_text())
    assert len(ev)==16
    with np.load(rawpath) as z:
        w=z["events"].view("<u4").reshape(8,2,32)
        for j,x in enumerate(ev):
            epoch,lane=divmod(j,2)
            assert x==dict(epoch=epoch,lane=lane,core=(0,2,4,6,1,3,5,7)[epoch],event_epoch=epoch,complete=1,skipped=int(epoch in (4,5,6)))
            assert w[epoch,lane,4:8].tolist()==[x["core"],epoch,1,x["skipped"]]
    with np.load(topk) as z:
        assert z["native"].shape==z["fused"].shape==(4,4,2048)
        assert np.array_equal(np.sort(z["native"],axis=-1),np.sort(z["fused"],axis=-1))
    files += [evpath,rawpath,topk]
out.write_text(json.dumps({str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in files},indent=2)+"\n")
print("PREVIOUS_V1_F4_RAW_EVIDENCE_PASS",flush=True)
