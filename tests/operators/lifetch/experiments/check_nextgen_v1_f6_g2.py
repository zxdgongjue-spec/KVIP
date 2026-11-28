#!/usr/bin/env python3
"""F6-G2: check all 101 independent raw snapshots (CPU only)."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
root,lib=map(Path,sys.argv[1:])
summary,=(root/"basic_bs16").glob("*/summary.json")
row,=json.loads(summary.read_text())["cases"]
assert row["case"]=="basic_bs16/f6_graph" and row["accepted"] is True and row["correctness"]=="PASS" and row["execution"]=="COMPLETED",row
worker=summary.parent/"cases/basic_bs16"
assert json.loads((worker/"worker-result.json").read_text())["exit_code"]==0
report,=worker.glob("case0-b16-prototype-graph/report.json")
d=report.parent
r=json.loads(report.read_text())
assert r["correctness"]=="PASS" and r["first_round_verified"]==101 and r["timing"]=={}
assert r["actual_k"]==[131072]*8+[32768]*8 and r["actual_q"]==[4]*16
assert r["explicit_generic_streaming"] is True and r["parallel_terminal"] is False
assert r["production_lease_type"]=="KvafGetLease" and r["stateful_rounds"]==0 and r["sfa"] is None
assert r["exact_same_topk"] is True
assert (worker/"universal-build/libv9_bridge.so").resolve()==lib.resolve()
assert hashlib.sha256(lib.read_bytes()).hexdigest()=="cea9c320043a074074808bcb7595edebd0fc1ebbaae1766955d858d8c38b13b4"
assert len(r["checks"])==101
files=[summary,worker/"worker-result.json",report,d/"native.log"]
stems=["f3"]+[f"f4_replay_{i:02d}" for i in range(1,101)]
assert {x.parent.name for x in d.glob("*/f5_topk_diagnostic.json")}==set(stems)
assert {x.stem.removesuffix("_topk") for x in d.glob("f4_replay_*_topk.npz")}|{"f3"}==set(stems)
for i,stem in enumerate(stems):
    jp=d/stem/"f5_topk_diagnostic.json";raw=jp.with_suffix(".npz")
    j=json.loads(jp.read_text())
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==j["npz_sha256"]
    assert j["failing_queries"]==0 and len(j["queries"])==64
    H,R=131072,16384
    with np.load(raw,allow_pickle=False) as z:
        lengths=z["actual_k"]
        assert lengths.tolist()==r["actual_k"]
        score=z["raw_score_u16"][:64*H].reshape(16,4,H)
        oracle=z["oracle_bf16"]^np.uint16(0x8000)
        native,fused=z["native"],z["fused"]
        assert oracle.shape==(4,H) and native.shape==fused.shape==(16,4,2048)
        assert np.array_equal(fused,z["service_logical_output"])
        assert np.array_equal(fused,z["physical_selected"])
        for b,n in enumerate(lengths):
            n=int(n)
            for q in range(4):
                assert np.array_equal(score[b,q,:n],oracle[q,:n]),("score",b,q)
                assert np.array_equal(np.sort(native[b,q]),np.sort(fused[b,q])),("TopK",b,q)
                assert int(z["score_offsets_u16"][b,q])==(4*b+q)*H,("source",b,q)
                assert int(z["incremental_words"][b,q,1])==(n|(n<<32)),("cursor",b,q)
                assert int(z["incremental_words"][b,q,2])!=0,("cutoff",b,q)
        assert int(np.count_nonzero(fused[12,3]>=R))==26
        assert int(np.intersect1d(native[12,3],fused[12,3]).size)==2048
        w=z["events"].view("<u4").reshape(8,8,32)
        owners=[[2*b+g for b in range(i,i+4)] for g in (0,1) for i in (0,4,8,12)]
        for ep,group in enumerate(owners):
            assert w[ep,:,4].tolist()==[c for c in group for _ in (0,1)]
            assert np.all(w[ep,:,5]==ep) and np.all(w[ep,:,6]==1)
            assert np.all(w[ep,:,7]==int(ep in (2,3,6,7)))
        rows=z["row_words_8_onward"]
        for ep in (2,3,6,7):
            assert not rows[:,16+ep-8].any() and not rows[:,214+ep-8].any()
        assert int(lengths.sum())*4==5242880
    assert r["checks"][i]["completion"]==j["completion"] and r["checks"][i]["reads"]==j["reads"]
    with np.load(d/(stem+"_topk.npz"),allow_pickle=False) as top, np.load(raw,allow_pickle=False) as snap:
        assert np.array_equal(top["native"],snap["native"]) and np.array_equal(top["fused"],snap["fused"])
    completion=r["checks"][i]["completion"]
    assert completion["final_reads"]==completion["nonconflicting_reads"]==2512
    assert completion["conflicting_reads"]==completion["native_deferred_rows"]==0
    assert r["checks"][i]["reads"]["final"]==2512
    print("V1_F6_G2_SAMPLE_PASS",stem,"score=5242880 source=64 cursor=64 cold=26 overlap=2048 events=64 skipped=32")
    print("V1_F6_G2_SAMPLE_FETCH",stem,json.dumps(r["checks"][i]))
    files += [jp,raw,d/(stem+"_event_evidence.json"),d/(stem+"_event_evidence.npz"),d/(stem+"_topk.npz")]
log=(d/"native.log").read_text()
assert "V1_F4_CAPTURE_PASS" in log and "V1_F4_GRAPH_RESET_PASS" in log
assert [line.split()[1] for line in log.splitlines() if line.startswith("V1_F4_SAMPLE_PASS ")]==stems
assert log.count("V1_F4_CAPTURE_PASS")==1 and log.count("V1_F4_GRAPH_RESET_PASS")==1
assert "V1_F4_LEASE_CLOSED_PASS" not in log or log.count("V1_F4_LEASE_CLOSED_PASS")==1
print("V1_F6_G2_RAW_GRAPH_PASS samples=101")
(root/"f6-g2-evidence-sha256.json").write_text(json.dumps({str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in files},indent=2)+"\n")
