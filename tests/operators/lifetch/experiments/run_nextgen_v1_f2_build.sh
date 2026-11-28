#!/usr/bin/env bash
# F2 compile only: exact F1 candidate in a new source worktree and build cache.
# No torch import, NPU context, KVAF service, installation or original SO writes.
set -Eeuo pipefail
umask 077
[[ $# == 3 ]] || { echo "Usage: $0 URMA_MANIFEST KVAF_PATCH_EVIDENCE F1_HOST_EVIDENCE"; exit 2; }
MANIFEST=$(realpath "$1")
KVAF_EVIDENCE=$(realpath "$2")
F1=$(realpath "$3")
source "$MANIFEST"
DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$DIR" rev-parse --show-toplevel)
[[ "$KVIP_SHA" == 79c6a22557ed63a53da687e58f1f5bc1401fd8ed ]]
[[ "$KVAF_SHA" == 87e261ae4d07d4868cfdbb306d15f3f137edfc3d ]]
[[ "$SOC" == Ascend950DT_9582 && "$TRANSPORT" == urma ]]
[[ $(git -C "$NEW_KVIP" rev-parse HEAD) == "$KVIP_SHA" ]]
[[ $(git -C "$NEW_KVAF" rev-parse HEAD) == "$KVAF_SHA" ]]
git -C "$NEW_KVIP" diff --quiet HEAD
git -C "$REPO" diff --quiet HEAD
[[ $(git -C "$NEW_KVAF" diff --binary HEAD | sha256sum | awk '{print $1}') == 8132c738cb7e7a79fe1dc14f173954cc0110be322e14b25274debed15a8a9052 ]]
[[ $(cat "$F1/test_commit.txt") == 94b702681bf9c43aeab00013d8bd676e6d02ee3e ]]
grep -Fxq 'V1_F1_CLOSURE_MODEL_PASS 2524 matrix scenarios; plus 3 explicit controls' "$F1/run.log"
grep -Fxq V1_F1_HOST_PREPARED "$F1/run.log"
PATCH_SHA=a74723dcfa0c6d9a530c8caaccdd27a79a5c136fc5ea89c018cdd9bf6fe6b0e2
sha256sum --strict -c <<EOF
$PATCH_SHA  $F1/v1-f1-candidate.patch
EOF
OUT=$(mktemp -d "$(dirname "$VENV")/v1-f2-build.XXXXXX")
exec > >(tee "$OUT/run.log") 2>&1
trap 'rc=$?; if (( rc != 0 )); then echo "V1_F2_BUILD_FAILED rc=$rc OUT=$OUT"; fi' EXIT
echo "OUTPUT_DIR=$OUT"
git -C "$REPO" rev-parse HEAD | tee "$OUT/test_commit.txt"
cp "$MANIFEST" "$OUT/urma_manifest.env"
cp "$KVAF_EVIDENCE" "$OUT/kvaf_new_files.json"
printf '%s\n' "$F1" > "$OUT/f1_evidence_path.txt"
cat > "$OUT/frozen-sha256.txt" <<EOF
85dd545b92d494d6c0c112133f5626781bd086cd80ba31e274ad5999f402663e  $CPU_SERVICE
724b3e2dd86d86e74232566d7469f5ebbdd674037e3cd18638c6ccb7dab26276  $NPU_BUILD/bin/kvaf_execute_batch_get_kernels.o
02e84f41800ed9ffaca5abe3871601946c45be814bd71f6686af11a81b3f8386  $NPU_BUILD/bin/kvaf_execute_batch_put_kernels.o
8c0c71a6806f00eb9539fe0bfeaabd6e194a0c252c88999d922f8578131fb2b7  $LIFETCH_LIBRARY
EOF
sha256sum --strict -c "$OUT/frozen-sha256.txt"
"$VPY" - "$NEW_KVAF" "$KVAF_EVIDENCE" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,evidence=map(Path,sys.argv[1:])
records=json.loads(evidence.read_text())
expected={"access_operator/algorithms/get/direct/urma_aiv/include/kvaf_batch_get_device.h",
          "access_operator/include/ubmem/fused_batch_get_device.h"}
assert len(records)==2 and {r["path"] for r in records}==expected
for r in records:
    assert hashlib.sha256((root/r["path"]).read_bytes()).hexdigest()==r["sha256"]
print("V1_F2_KVAF_HEADER_GATE_PASS")
PY
echo '===== 1. REGENERATE AND ISOLATE EXACT F1 PATCH ====='
"$VPY" -u "$DIR/prepare_nextgen_v1_f1_patch.py" "$NEW_KVIP" "$OUT"
cmp "$F1/v1-f1-candidate.patch" "$OUT/v1-f1-candidate.patch"
SRC="$OUT/source"
git -C "$NEW_KVIP" worktree add --detach "$SRC" "$KVIP_SHA"
git -C "$SRC" apply --check "$OUT/v1-f1-candidate.patch"
git -C "$SRC" apply "$OUT/v1-f1-candidate.patch"
git -C "$SRC" diff --check
git -C "$SRC" diff --binary HEAD > "$OUT/applied-source.diff"
"$VPY" - "$SRC" "$OUT" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
src,out=map(Path,sys.argv[1:])
prefix="csrc/operators/lightning_indexer_with_fetch/op_kernel/"
want={prefix+p for p in ("indexer.cpp","indexer/indexer.h","indexer/indexer_vector.h","indexer/incremental_methods.h")}
changed=set(subprocess.check_output(["git","-C",str(src),"diff","--name-only","HEAD"],text=True).splitlines())
assert changed==want,changed
(out/"patched-source-sha256.json").write_text(json.dumps({p:hashlib.sha256((src/p).read_bytes()).hexdigest() for p in sorted(want)},indent=2)+"\n")
print("V1_F2_ISOLATED_SOURCE_PASS",src)
PY
echo '===== 2. BUILD VIA FROZEN STARTUP BUILDER IN NEW CACHE ====='
set +u
source "$CANN/set_env.sh"
set -Eeuo pipefail
export USER="${USER:-$(id -un)}"
export PATH="$VENV/bin:$PATH" PYTHON="$VPY" PYTHONNOUSERSITE=1
export ASCEND_HOME_PATH="$CANN" ASCEND_CANN_PACKAGE_PATH="$CANN" SOC_VERSION="$SOC"
# Use runpy rather than importing installed kvip/torch; the builder derives
# its operator root from the isolated build.py's own __file__.
"$VPY" -u - "$SRC" "$OUT" "$NPU_BUILD" "$CANN" "$SOC" "$NEW_KVAF" "$LIFETCH_LIBRARY" <<'PY'
import json,runpy,sys
from pathlib import Path
src,out,npu,cann=map(Path,sys.argv[1:5])
soc=sys.argv[5]
kvaf,original=map(Path,sys.argv[6:])
f=runpy.run_path(str(src/"python/kvip/runtime/build.py"))
actual_cann,headers=f["kvaf_build_paths"](npu)
assert actual_cann.resolve()==cann.resolve()
assert headers.resolve()==(kvaf/"access_operator/algorithms/get/direct/urma_aiv/include").resolve()
# Verify exact baseline configure options instead of inheriting a guessed SoC.
baseline={}
for line in (original.parent/"CMakeCache.txt").read_text().splitlines():
    if line.startswith(("#","//")) or "=" not in line:continue
    key,val=line.split("=",1);baseline[key.split(":",1)[0]]=val
assert baseline["RUN_MODE"]=="npu" and baseline["SOC_VERSION"]==soc
assert Path(baseline["ASCEND_CANN_PACKAGE_PATH"]).resolve()==cann.resolve()
assert Path(baseline["KVAF_INCLUDE"]).resolve()==headers.resolve()
print("V1_F2_BUILD_OPTIONS_MATCH",soc,cann,headers,flush=True)
libs=f["build_libraries"](cache_dir=out/"build-cache",cann=actual_cann,headers=headers,soc=soc)
lib,=libs
assert lib.resolve().is_relative_to((out/"build-cache").resolve())
assert lib.resolve()!=original.resolve()
(out/"candidate_library.txt").write_text(str(lib.resolve())+"\n")
cache={}
for line in (lib.parent/"CMakeCache.txt").read_text().splitlines():
    if line.startswith(("#","//")) or "=" not in line:continue
    key,val=line.split("=",1);cache[key.split(":",1)[0]]=val
assert Path(cache["CMAKE_HOME_DIRECTORY"]).resolve()==(src/"csrc/operators/lightning_indexer_with_fetch").resolve()
for key in ("RUN_MODE","SOC_VERSION","ASCEND_CANN_PACKAGE_PATH","KVAF_INCLUDE"):
    assert cache[key]==baseline[key],(key,cache[key],baseline[key])
print("V1_F2_BUILD_SOURCE_AND_OPTIONS_PASS",lib,flush=True)
PY
SO=$(cat "$OUT/candidate_library.txt")
test -s "$SO"
test -s "$(dirname "$SO")/complete.json"
cp "$(dirname "$SO")/build.log" "$OUT/build.log"
echo '===== 3. CHECK HOST EXPORTS AND DYNAMIC DEPENDENCIES ====='
nm -D --defined-only "$LIFETCH_LIBRARY" > "$OUT/original-symbols.txt"
nm -D --defined-only "$SO" > "$OUT/candidate-symbols.txt"
awk '$NF ~ /^kvip_/ {print $NF}' "$OUT/original-symbols.txt" | sort -u > "$OUT/original-kvip-exports.txt"
awk '$NF ~ /^kvip_/ {print $NF}' "$OUT/candidate-symbols.txt" | sort -u > "$OUT/candidate-kvip-exports.txt"
test -s "$OUT/original-kvip-exports.txt"
diff -u "$OUT/original-kvip-exports.txt" "$OUT/candidate-kvip-exports.txt"
ldd "$SO" | tee "$OUT/candidate-ldd.txt"
if grep -q 'not found' "$OUT/candidate-ldd.txt"; then exit 3; fi
sha256sum "$SO" | tee "$OUT/candidate-sha256.txt"
[[ $(sha256sum "$SO" | awk '{print $1}') != $(sha256sum "$LIFETCH_LIBRARY" | awk '{print $1}') ]]
# Do not confuse symbol compatibility with actual runtime correctness.
git -C "$SRC" diff --binary HEAD | cmp - "$OUT/applied-source.diff"
git -C "$NEW_KVIP" diff --quiet HEAD
git -C "$REPO" diff --quiet HEAD
sha256sum --strict -c "$OUT/frozen-sha256.txt"
echo V1_F2_COMPILE_ONLY_PASS
echo "OUTPUT_DIR=$OUT"
