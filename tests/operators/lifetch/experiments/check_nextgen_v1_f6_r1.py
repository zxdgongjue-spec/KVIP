#!/usr/bin/env python3
"""F6-R1 BS4 regression: compare the new F6 candidate to historical F4 raw evidence."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
root,lib,old=map(Path,sys.argv[1:])
out=root/"f6-r1-evidence-sha256.json"
assert (root/"physical_device.txt").read_text().strip()=="7"
assert (old/"test_commit.txt").read_text().strip()=="2afd1a84b304adeb87c406242174ab0278132cb7"
old_report,=(old/"basic_bs4").glob("*/cases/basic_bs4/case0-b4-prototype-graph/report.json")
old_dir=old_report.parent
old_r=json.loads(old_report.read_text())
summary,=(root/"basic_bs4").glob("*/summary.json")
row,=json.loads(summary.read_text())["cases"]
assert row["case"]=="basic_bs4/f6_r1" and row["accepted"] is True and row["correctness"]=="PASS" and row["execution"]=="COMPLETED"
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
assert hashlib.sha256(lib.read_bytes()).hexdigest()=="cea9c320043a074074808bcb7595edebd0fc1ebbaae1766955d858d8c38b13b4"
files=[root/"test_commit.txt",root/"physical_device.txt",summary,exitfile,report,d/"native.log",d/"sampling_protocol.json"]
assert r["stateful_rounds"]==0 and r["sfa"] is None
assert r["actual_k"]==old_r["actual_k"] and r["actual_q"]==old_r["actual_q"]
assert len(r["checks"])==4
for i,stem in enumerate(("f3","f4_replay_01","f4_replay_02","f4_replay_03")):
    c=r["checks"][i]
    assert c["reads"]["final"]==494 and c["reads"]["prefetch"]>=0,c
    assert c["completion"]=={"final_reads":494,"nonconflicting_reads":494,"conflicting_reads":0,"native_deferred_rows":0,"completed_checkpoints":8+4*i},c
    evpath=d/(stem+"_event_evidence.json")
    rawpath=d/(stem+"_event_evidence.npz")
    topk=d/(stem+"_topk.npz")
    ev=json.loads(evpath.read_text())
    assert len(ev)==16
    with np.load(rawpath,allow_pickle=False) as z:
        w=z["events"].view("<u4").reshape(8,2,32)
        for j,x in enumerate(ev):
            epoch,lane=divmod(j,2)
            assert x==dict(epoch=epoch,lane=lane,core=(0,2,4,6,1,3,5,7)[epoch],event_epoch=epoch,complete=1,skipped=int(epoch in (4,5,6)))
            assert w[epoch,lane,4:8].tolist()==[x["core"],epoch,1,x["skipped"]]
        rows=z["rows"]
        for ep in (4,5,6):
            assert not rows[:,16+ep].any() and not rows[:,214+ep].any()
        with np.load(old_dir/(stem+"_event_evidence.npz"),allow_pickle=False) as prev:
            assert np.array_equal(w[:,:,4:8],prev["events"].view("<u4").reshape(8,2,32)[:,:,4:8])
    with np.load(topk,allow_pickle=False) as z, np.load(old_dir/(stem+"_topk.npz"),allow_pickle=False) as prev:
        assert z["native"].shape==z["fused"].shape==(4,4,2048)
        assert np.array_equal(np.sort(z["native"],axis=-1),np.sort(z["fused"],axis=-1))
        for key in ("native","fused"):
            assert np.array_equal(np.sort(z[key],axis=-1),np.sort(prev[key],axis=-1)),(stem,key)
    print("V1_F6_R1_SAMPLE_PASS",stem,"queries=16 events=16 skipped=6",json.dumps(c),flush=True)
    files += [evpath,rawpath,topk]
log=(d/"native.log").read_text()
assert [line.split()[1] for line in log.splitlines() if line.startswith("V1_F4_SAMPLE_PASS ")]==["f3","f4_replay_01","f4_replay_02","f4_replay_03"]
assert log.count("V1_F4_CAPTURE_PASS")==1 and log.count("V1_F4_GRAPH_RESET_PASS")==1
assert "V1_F4_LEASE_CLOSED_PASS" in (worker/"worker.log").read_text()
out.write_text(json.dumps({str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in files},indent=2)+"\n")
print("V1_F6_R1_RAW_PASS samples=4",flush=True)
