#!/usr/bin/env python3
"""Read-only F3 raw evidence gate for F4."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
root,lib,out=map(Path,sys.argv[1:])
assert (root/"test_commit.txt").read_text().strip()=="5244e924a13d7f05e550d8559361fd56c9a441fe"
assert (root/"physical_device.txt").read_text().strip()=="7"
assert "NEXTGEN_V1_F3_EAGER_PASS" in (root/"run.log").read_text().splitlines()
summary,=(root/"basic_bs4").glob("*/summary.json")
row,=json.loads(summary.read_text())["cases"]
assert row["case"]=="basic_bs4/f3_eager" and row["accepted"] is True and row["correctness"]=="PASS"
worker=summary.parent/"cases/basic_bs4"
exitfile=worker/"worker-result.json"
assert json.loads(exitfile.read_text())["exit_code"]==0
report,=worker.glob("case0-b4-prototype-graph/report.json")
d=report.parent
r=json.loads(report.read_text())
assert r["correctness"]=="PASS" and r["first_round_verified"]==1 and r["timing"]=={}
assert r["actual_k"]==[32768,65536,98304,131072] and r["actual_q"]==[4]*4
assert r["exact_same_topk"] is True and r["production_lease_type"]=="KvafGetLease"
assert r["checks"]==[{"reads":{"prefetch":559,"final":494},"completion":{"final_reads":494,"nonconflicting_reads":494,"conflicting_reads":0,"native_deferred_rows":0,"completed_checkpoints":8}}]
assert (worker/"universal-build/libv9_bridge.so").resolve()==lib.resolve()
assert hashlib.sha256(lib.read_bytes()).hexdigest()=="fc56dc00b505a552cf71356188fc93790e9082cbcf32f7d6e64fab5265c52261"
ev=json.loads((d/"f3_event_evidence.json").read_text())
assert len(ev)==16
for i,x in enumerate(ev):
    epoch,lane=divmod(i,2)
    assert x==dict(epoch=epoch,lane=lane,core=(0,2,4,6,1,3,5,7)[epoch],event_epoch=epoch,complete=1,skipped=int(epoch in (4,5,6))),x
with np.load(d/"f3_event_evidence.npz") as z:
    w=z["events"].view("<u4").reshape(8,2,32)
    for x in ev:
        assert w[x["epoch"],x["lane"],4:8].tolist()==[x["core"],x["event_epoch"],x["complete"],x["skipped"]]
with np.load(d/"v1_topk_evidence.npz") as z:
    assert z["native"].shape==z["fused"].shape==(4,4,2048)
    assert np.array_equal(np.sort(z["native"],axis=-1),np.sort(z["fused"],axis=-1))
files=[root/"run.log",root/"test_commit.txt",root/"physical_device.txt",summary,exitfile,report,d/"f3_event_evidence.json",d/"f3_event_evidence.npz",d/"v1_topk_evidence.npz"]
out.write_text(json.dumps({str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in files},indent=2)+"\n")
print("PREVIOUS_V1_F3_RAW_EVIDENCE_PASS",flush=True)
