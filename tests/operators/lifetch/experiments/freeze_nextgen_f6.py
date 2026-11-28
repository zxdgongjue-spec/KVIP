#!/usr/bin/env python3
"""CPU-only F6 evidence freeze. No device, service, installation or source mutation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

work, out = (Path(x).resolve() for x in sys.argv[1:])
assert not out.exists(), out
def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
def git(p, *args):
    return subprocess.check_output(["git", "-C", str(p), *args])
def require(p, value):
    assert p.read_text().strip() == value, p
def marker(p, value):
    assert value in p.read_text().splitlines(), (p, value)
def hash_gate(p, want):
    assert digest(p) == want, p
frozen_base = "79c6a22557ed63a53da687e58f1f5bc1401fd8ed"
f6 = work / "v1-f6-build.b0xfjt"
require(f6/"test_commit.txt","225f77f03b798827c2b7e7aa5ddd0318c0538427")
marker(f6/"run.log","V1_F6_COMPILE_ONLY_PASS")
src = f6/"source"
assert git(src,"rev-parse","HEAD").decode().strip() == frozen_base
assert git(src,"diff","--binary","HEAD") == (f6/"applied-source.diff").read_bytes()
hash_gate(f6/"f6-metadata-writeback.patch","b83173331923462c25e6fa206c2bbea122c35a5d1a52dd31544531c2e0ecf40e")
candidate = Path((f6/"candidate_library.txt").read_text().strip()).resolve(strict=True)
assert candidate == f6/"build-cache/054e5eb4d7609d29d5f0/libv9_bridge.so"
hash_gate(candidate,"cea9c320043a074074808bcb7595edebd0fc1ebbaae1766955d858d8c38b13b4")
source_hashes = json.loads((f6/"patched-source-sha256.json").read_text())
prefix = "csrc/operators/lightning_indexer_with_fetch/op_kernel/"
want = {prefix+x for x in ("indexer.cpp","indexer/indexer.h","indexer/indexer_vector.h","indexer/incremental_methods.h")}
assert set(source_hashes) == want
assert set(git(src,"diff","--name-only","HEAD").decode().splitlines()) == want
for name, sha in source_hashes.items():
    hash_gate(src/name, sha)
subprocess.run(["sha256sum","--strict","-c",str(f6/"frozen-sha256.txt")],check=True)
print("F6_FREEZE_SOURCE_AND_BINARY_PASS",flush=True)
stages = [
 ("E","v1-f6-eager.aPHgy1","d5b5f18116533d5a4d2f7eabc6bd5d28400f05ba","NEXTGEN_V1_F6_EAGER_PASS","f6-evidence-sha256.json",8),
 ("G1","v1-f6-graph.SyLzb1","dab0fd5e7912df0e9c2583fbea7518a4b2e5dde0","NEXTGEN_V1_F6_GRAPH_PASS","f6-graph-evidence-sha256.json",24),
 ("G2","v1-f6-g2.8smo45","a712f28b7f7d0546be258e9138d8fb3609d4504b","NEXTGEN_V1_F6_G2_PASS","f6-g2-evidence-sha256.json",509),
 ("R1","v1-f6-r1.TzmrIV","e094ae393f72d0cb12631ba64042b4d6cd7bdf41","NEXTGEN_V1_F6_R1_PASS","f6-r1-evidence-sha256.json",19),
 ("R2","v1-f6-r2.eIDKFe","aa98ff71e30ac0833aad318c69a6b0512a129efa","NEXTGEN_V1_F6_R2_PASS","f6-r2-evidence-sha256.json",24),
 ("R3","v1-f6-r3.r62ZGx","5462f9cb6a2e3cede7ec291120074d9285234fd8","NEXTGEN_V1_F6_R3_PASS","f6-r3-evidence-sha256.json",24),
]
records={}
for label, dirname, commit, passed, name, count in stages:
    root=work/dirname
    require(root/"test_commit.txt",commit)
    require(root/"physical_device.txt","7")
    marker(root/"run.log",passed)
    manifest=root/name
    entries=json.loads(manifest.read_text())
    assert len(entries)==count,(label,len(entries))
    for filename, sha in entries.items():
        path=Path(filename).resolve(strict=True)
        assert path.is_relative_to(root) and path.is_file(),path
        hash_gate(path,sha)
    records[label] = dict(root=str(root),commit=commit,files=count,
                          manifest_sha256=digest(manifest),log_sha256=digest(root/"run.log"))
    print("F6_FREEZE_EVIDENCE_PASS",label,"files="+str(count),flush=True)
# A freeze record binds existing evidence, not a new execution or source merge.
result=dict(schema=1,scope="Generic LI Q4 noncausal independent first-round correctness <=128K; no timing/stateful/production claim",
    operator_base=frozen_base,kvaf_base="87e261ae4d07d4868cfdbb306d15f3f137edfc3d",
    candidate=str(candidate),candidate_sha256=digest(candidate),
    patch_sha256=digest(f6/"f6-metadata-writeback.patch"),
    applied_diff_sha256=digest(f6/"applied-source.diff"),patched_source_sha256=source_hashes,
    evidence=records,total_evidence_files=sum(x["files"] for x in records.values()))
out.mkdir(parents=False,exist_ok=False)
(out/"f6-freeze.json").write_text(json.dumps(result,indent=2)+"\n")
for name in ("applied-source.diff","f6-metadata-writeback.patch","patched-source-sha256.json"):
    (out/name).write_bytes((f6/name).read_bytes())
print("NEXTGEN_F6_BASELINE_FREEZE_PASS",result["total_evidence_files"],flush=True)
print("OUTPUT_DIR="+str(out),flush=True)
