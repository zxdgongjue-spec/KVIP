#!/usr/bin/env python3
"""Read-only V3b evidence contract; no NPU or torch imports."""
import hashlib,json,math,statistics,sys
from pathlib import Path

def main():
    root,dest=map(Path,sys.argv[1:])
    cases=("smoke","smoke_q2","v05","v07","v08","v09","v10")
    commits="dd3f154b2d7d44d3f9a86f960452f7f17a61c00a"
    assert (root/"test_commit.txt").read_text().strip()==commits
    log=(root/"run.log").read_text()
    assert "V3B_TOPK_MATRIX_PASS" in log.splitlines()
    summary=json.loads((root/"summary.json").read_text())
    assert [r["case"] for r in summary]==list(cases)
    files=[root/"test_commit.txt",root/"run.log",root/"matrix-device.log",root/"summary.json"]
    sha=None
    for case,s in zip(cases,summary):
        path=root/(case+".json");files.append(path)
        r=json.loads(path.read_text())
        assert r["status"]=="PASS" and r["case"]==case and r["physical_npu"]==7
        assert len(r["actual_k"])==len(r["actual_q"])==16
        assert set(r["order_exact"])=={"count_direct","count_merge","adaptive_merge"}
        assert all(v=={"local_order":True,"final_order":True} for v in r["order_exact"].values())
        assert r["order_exact"]==s["order_exact"]
        libs=r["library_sha256"]
        assert set(libs)=={"score","local","global"}
        assert libs["score"]=="0f3696d6872c6f1a05ec7c97e2900fabc28227c25149aa3f193ebdf0f79261c5"
        assert libs["global"]=="41fe1ad28ab31d3e562a72f86ef88b49804931e1ca8db990fc7eadb8c28983a9"
        if sha is None:sha=libs
        assert libs==sha
        for policy,scopes in (("count",{"score","three_stage","direct"}),("adaptive",{"score","three_stage"})):
            v=r["policies"][policy]
            assert v["eager_exact_membership"] is True and v["graph_exact_replays"]==3
            assert set(v["timing"])==scopes
            for t in v["timing"].values():
                raw=t["raw_us"]
                assert len(raw)==50 and all(math.isfinite(x) and x>0 for x in raw)
                assert math.isclose(statistics.median(raw),t["p50_us"],rel_tol=1e-12)
        c,a=(r["policies"][k]["timing"] for k in ("count","adaptive"))
        for key,t in (("score_count",c["score"]),("score_adaptive",a["score"]),("count_direct",c["direct"]),("count_three_stage",c["three_stage"]),("adaptive_three_stage",a["three_stage"])):
            assert math.isclose(t["p50_us"],s[key],rel_tol=1e-12)
        assert math.isclose(100*(1-s["adaptive_three_stage"]/s["count_direct"]),s["reduction_vs_count_direct_percent"],rel_tol=1e-12)
        print("V3B_EVIDENCE_CASE_PASS",case,flush=True)
    dest.write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+"\n")
    print("PREVIOUS_V3B_EVIDENCE_PASS",flush=True)
if __name__=="__main__":main()
