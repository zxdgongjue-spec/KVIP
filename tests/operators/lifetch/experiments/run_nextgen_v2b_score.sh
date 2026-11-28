#!/usr/bin/env bash
# Host mode does not load torch_npu or create a device context.
set -Eeuo pipefail
umask 077
[[ $# == 3 ]] || { echo "Usage: $0 host|device URMA_MANIFEST V2A_BUILD_EVIDENCE"; exit 2; }
MODE="$1"
[[ "$MODE" == host || "$MODE" == device ]]
source "$(realpath "$2")"
BUILD=$(realpath "$3")
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
DIR="$REPO/tests/operators/lifetch/experiments"
git -C "$REPO" diff --quiet HEAD
git -C "$REPO" diff --quiet 2ebce41df37bbda33f3bfade65bbd503bd4e6138 HEAD -- csrc python
grep -Fxq V2A_SCORE_COMPILE_ONLY_PASS "$BUILD/run.log"
grep -Fxq 2ebce41df37bbda33f3bfade65bbd503bd4e6138 "$BUILD/test_commit.txt"
sha256sum --strict -c "$BUILD/binary-sha256.txt"
SO="$BUILD/build/libnextgen_v2_score_bridge.so"
[[ $(sha256sum "$SO" | awk '{print $1}') == 0f3696d6872c6f1a05ec7c97e2900fabc28227c25149aa3f193ebdf0f79261c5 ]]
OUT=$(mktemp -d "$(dirname "$VENV")/v2b-$MODE.XXXXXX")
exec > >(tee "$OUT/run.log") 2>&1
echo "OUTPUT_DIR=$OUT"
git -C "$REPO" rev-parse HEAD | tee "$OUT/test_commit.txt"
g++ -std=c++17 -O3 -shared -fPIC -ffp-contract=off "$DIR/b16_mtp_cpp_planner.cpp" -o "$OUT/libplanner.so"
unset PYTHONOPTIMIZE
"$VPY" -m py_compile "$DIR/nextgen_v2b_score_test.py" "$DIR/b16_score_queue_test.py"
"$VPY" -u "$DIR/nextgen_v2b_score_test.py" --mode host --planner "$OUT/libplanner.so"
if [[ "$MODE" == host ]]; then echo V2B_HOST_ONLY_PASS; exit 0; fi
PHYSICAL="${NEXTGEN_PHYSICAL_DEVICE:-}"
[[ "$PHYSICAL" == 6 || "$PHYSICAL" == 7 ]]
[[ "${CONFIRM_NEXTGEN_NPU_IDLE:-0}" == 1 ]]
set +u
source "$CANN/set_env.sh"
set -Eeuo pipefail
export ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL" DEVICE_ID=0 PYTHONNOUSERSITE=1
export USER="${USER:-$(id -un)}"
export PATH="$VENV/bin:$PATH" PYTHON="$VPY"
echo "SELECTED_PHYSICAL_DEVICE=$PHYSICAL"
for CASE in smoke v05 v07 v08 v09 v10; do
    npu-smi info | tee "$OUT/$CASE-npu-before.log"
    "$VPY" - "$OUT/$CASE-npu-before.log" "$PHYSICAL" <<'PY'
import re,sys
from pathlib import Path
t=Path(sys.argv[1]).read_text()
n=int(sys.argv[2])
rows=t.splitlines(); matches=[]
for i,row in enumerate(rows[:-1]):
    if re.match(rf'^\|\s*{n}\s*\|\s*Ascend950DT\s*\|',row):
        m=re.search(r'(\d+)\s*/\s*(\d+)\s*\|\s*$',rows[i+1])
        if m:matches.append((row.split('|')[3].strip(),int(m[1]),int(m[2])))
idle=bool(re.search(rf'^\|\s*No running processes found in NPU {n}\s*\|',t,re.M))
if len(matches)!=1 or matches[0][0]!='OK' or matches[0][2]!=98304 or matches[0][1]>8192 or not idle:
    raise SystemExit(f'NPU{n}_IDLE_GATE_BLOCKED {matches} idle={idle}')
print(f'NPU{n}_IDLE_GATE_PASS')
PY
    timeout --signal=TERM --kill-after=30s 900s "$VPY" -u "$DIR/nextgen_v2b_score_test.py" \
        --mode device --planner "$OUT/libplanner.so" --library "$SO" --cann "$CANN" \
        --case "$CASE" --output "$OUT/$CASE.json" 2>&1 | tee "$OUT/$CASE.log"
    grep -q "^V2B_CASE_PASS $CASE " "$OUT/$CASE.log"
done
sha256sum --strict -c "$BUILD/binary-sha256.txt"
echo V2B_SCORE_MATRIX_PASS
echo "OUTPUT_DIR=$OUT"
