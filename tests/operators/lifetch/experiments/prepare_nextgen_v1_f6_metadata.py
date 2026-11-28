#!/usr/bin/env python3
"""Offline F5 exact-score audit, race model, and isolated F6 candidate patch."""
import difflib, hashlib, itertools, json, subprocess, sys
from pathlib import Path
import numpy as np

EXPECTED_NPZ="6add2e0e89ef1b857ecdad44c4862ab6f3fa48320ccebc80bd3135f84cef59d7"
F2_SHA="fc56dc00b505a552cf71356188fc93790e9082cbcf32f7d6e64fab5265c52261"
SOURCE_PATH="csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/incremental_methods.h"
OLD="        for(uint32_t b=lane,i=0;b<runtimeShape_.batch;b+=runtimeShape_.serviceLanes,i+=64)\n            DataCopy(metadata[b*64],incrementalMetadata_[i],64);"
NEW="        for(uint32_t b=lane,i=0;b<runtimeShape_.batch;b+=runtimeShape_.serviceLanes,i+=64) {\n            if (half) {\n                // Preserve the established uniform sequence-split path.\n                DataCopy(metadata[b*64],incrementalMetadata_[i],64);\n            } else {\n                // A generic producer can publish its resident header AFTER\n                // this window's UB snapshot. Never overwrite an unready row.\n                // Progress is captured before metadata and published after\n                // SaveResident has stored both the header and prefix payload.\n                for(uint32_t q=0;q<4;++q) {\n                    uint32_t mi=i+q*16;\n                    uint64_t ready=incrementalMetadata_.GetValue(mi+4);\n                    uint64_t source=incrementalMetadata_.GetValue(mi);\n                    if(uint32_t(ready)<runtimeShape_.resident || !source)continue;\n                    DataCopy(metadata[b*64+q*16],incrementalMetadata_[mi],16);\n                }\n            }\n        }"
H,R,K=131072,16384,2048

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def topk(x):
    return np.lexsort((np.arange(len(x)), -x.astype(np.int32)))[:K]

def audit(case,out):
    jp=case/"f5_topk_diagnostic.json"
    npz=case/"f5_topk_diagnostic.npz"
    d=json.loads(jp.read_text())
    assert d["npz_sha256"]==EXPECTED_NPZ==sha(npz)
    with np.load(npz,allow_pickle=False) as z:
        lengths=z["actual_k"]
        assert lengths.tolist()==[H]*8+[32768]*8
        score=z["raw_score_u16"][:64*H].reshape(16,4,H)
        oracle=z["oracle_bf16"]^np.uint16(0x8000)
        native,fused=z["native"],z["fused"]
        assert oracle.shape==(4,H) and native.shape==fused.shape==(16,4,K)
        assert np.array_equal(fused,z["service_logical_output"])
        assert np.array_equal(fused,z["physical_selected"])
        records=[]
        for b,n in enumerate(lengths):
            for q in range(4):
                x=score[b,q,:n]
                assert np.array_equal(x,oracle[q,:n]),(b,q)
                missing=np.setdiff1d(native[b,q],fused[b,q])
                extra=np.setdiff1d(fused[b,q],native[b,q])
                records.append(dict(b=b,q=q,n=int(n),missing=missing.tolist(),extra=extra.tolist()))
        bad=[r for r in records if r["missing"] or r["extra"]]
        assert len(bad)==1 and (bad[0]["b"],bad[0]["q"])==(12,3)
        missing=np.asarray(bad[0]["missing"])
        extra=np.asarray(bad[0]["extra"])
        assert len(missing)==len(extra)==26 and (missing>=R).all() and (extra<R).all()
        x=score[12,3,:32768]
        assert np.array_equal(np.sort(topk(x[:R])),np.sort(fused[12,3]))
        assert np.array_equal(np.sort(topk(x)),np.sort(native[12,3]))
        assert np.array_equal(np.argwhere(z["score_offsets_u16"]<0),[[12,3]])
        meta=z["incremental_words"]
        assert meta[12,3,1:4].tolist()==[0,0,0]
        for b,n in enumerate(lengths):
            for q in range(4):
                if (b,q)==(12,3):continue
                assert z["score_offsets_u16"][b,q]==(4*b+q)*H
                assert int(meta[b,q,1])==int(n)|(int(n)<<32)
        w=z["events"].view("<u4").reshape(8,8,32)
        assert np.all(w[:,:,6]==1)
        owners=[[2*b+g for b in range(i,i+4)] for g in (0,1) for i in (0,4,8,12)]
        for ep,group in enumerate(owners):
            assert w[ep,:,4].tolist()==[c for c in group for _ in (0,1)]
            assert np.all(w[ep,:,5]==ep)
            assert np.all(w[ep,:,7]==int(ep in (2,3,6,7)))
        rows=z["row_words_8_onward"]
        for ep in (2,3,6,7):
            assert not rows[:,16+ep-8].any() and not rows[:,214+ep-8].any()
        assert int(d["completion"]["final_reads"])==2517
        result=dict(npz_sha256=EXPECTED_NPZ,score_elements=int(lengths.sum())*4,score_mismatches=0,
                    bad_queries=bad,resident_only_exact=True,source_missing_only=[[12,3]],
                    missing_score_bf16=(x[missing]^np.uint16(0x8000)).tolist(),
                    event_complete=64,event_skipped=32,skipped_admission_zero=True,
                    scope="post-forward snapshot, not temporal trace")
        (out/"offline_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print("F5_OFFLINE_SCORE_PASS elements=5242880 mismatches=0")
    print("F5_RESIDENT_ONLY_REPRODUCED b=12 q=3 missing_cold=26 extra_resident=26")
    print("F5_ARCHIVED_METADATA_LOSS_REPRODUCED b=12 q=3")
    print("F5_EVENT_OWNER_AND_ADMISSION_PASS records=64 skipped=32")

def race_model(out):
    # Producer initializes header, then publishes readiness. Service captures
    # readiness, captures metadata, and eventually writes its UB snapshot back.
    # Final consumer joins both. Model ordering only, not device cache proof.
    events=("init","publish","snapshot_ready","snapshot_meta","writeback")
    cases=[]
    for order in itertools.permutations(events):
        if not(order.index("init")<order.index("publish") and
               order.index("snapshot_ready")<order.index("snapshot_meta")<order.index("writeback")):
            continue
        def run(fixed):
            header=0;ready=0;captured_ready=0;captured_header=0
            for e in order:
                if e=="init":header=1
                elif e=="publish":ready=1
                elif e=="snapshot_ready":captured_ready=ready
                elif e=="snapshot_meta":captured_header=header
                elif e=="writeback":
                    if not fixed or (captured_ready and captured_header):
                        header=captured_header
            return header
        cases.append(dict(order=order,old_final_source=run(False),fixed_final_source=run(True)))
    assert len(cases)==10 and any(not x["old_final_source"] for x in cases)
    assert all(x["fixed_final_source"] for x in cases)
    (out/"race_model.json").write_text(json.dumps(cases,indent=2)+"\n")
    print("F6_METADATA_RACE_MODEL_PASS",len(cases),"old_lost",sum(not x["old_final_source"] for x in cases),"fixed_lost",sum(not x["fixed_final_source"] for x in cases))

def prepare(f2,out):
    assert (f2/"test_commit.txt").read_text().strip()=="af85cc3b1de72b0a4fc3d92a335f12c2a971c80c"
    assert "V1_F2_COMPILE_ONLY_PASS" in (f2/"run.log").read_text().splitlines()
    lib=Path((f2/"candidate_library.txt").read_text().strip()).resolve(strict=True)
    assert sha(lib)==F2_SHA
    src=f2/"source"
    diff=subprocess.check_output(["git","-C",str(src),"diff","--binary","HEAD"])
    assert diff==(f2/"applied-source.diff").read_bytes()
    hashes=json.loads((f2/"patched-source-sha256.json").read_text())
    for name,value in hashes.items():assert sha(src/name)==value,name
    old=(src/SOURCE_PATH).read_text()
    assert old.count(OLD)==1
    assert "uint32_t half=actualSequenceSplit_?" in old
    new=old.replace(OLD,NEW)
    patch=out/"f6-metadata-writeback.patch"
    patch.write_text("".join(difflib.unified_diff(old.splitlines(keepends=True),new.splitlines(keepends=True),
                         fromfile="a/"+SOURCE_PATH,tofile="b/"+SOURCE_PATH)))
    subprocess.run(["git","-C",str(src),"apply","--check",str(patch)],check=True)
    assert sha(lib)==F2_SHA
    assert subprocess.check_output(["git","-C",str(src),"diff","--binary","HEAD"])==diff
    print("F6_PATCH_APPLY_CHECK_PASS",sha(patch))
    print("F6_ORIGINAL_F2_UNCHANGED")

def main():
    case,f2,out=map(lambda x:Path(x).resolve(strict=True),sys.argv[1:])
    audit(case,out);race_model(out);prepare(f2,out)
    print("NEXTGEN_F6_HOST_PREPARED",out)
if __name__=="__main__":main()
