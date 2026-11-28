#!/usr/bin/env bash
# Host mode does not load torch_npu or create a device context.
set -Eeuo pipefail
umask 077
[[ $# == 4 ]] || { echo "Usage: $0 host|device URMA_MANIFEST V2A_BUILD_EVIDENCE V3A_BUILD_EVIDENCE"; exit 2; }
MODE="$1"
[[ "$MODE" == host || "$MODE" == device ]]
source "$(realpath "$2")"
BUILD=$(realpath "$3")
TOPK=$(realpath "$4")
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
DIR="$REPO/tests/operators/lifetch/experiments"
git -C "$REPO" diff --quiet HEAD
git -C "$REPO" diff --quiet 74f9459d708682d982c50df2c65b328542de56b0 HEAD -- csrc python
grep -Fxq V2A_SCORE_COMPILE_ONLY_PASS "$BUILD/run.log"
grep -Fxq 2ebce41df37bbda33f3bfade65bbd503bd4e6138 "$BUILD/test_commit.txt"
sha256sum --strict -c "$BUILD/binary-sha256.txt"
SO="$BUILD/build/libnextgen_v2_score_bridge.so"
[[ $(sha256sum "$SO" | awk '{print $1}') == 0f3696d6872c6f1a05ec7c97e2900fabc28227c25149aa3f193ebdf0f79261c5 ]]
grep -Fxq V3A_TOPK_COMPILE_ONLY_PASS "$TOPK/run.log"
grep -Fxq 74f9459d708682d982c50df2c65b328542de56b0 "$TOPK/test_commit.txt"
sha256sum --strict -c "$TOPK/binary-sha256.txt"
LOCAL="$TOPK/local/libnextgen_v3_local_bridge.so"
MERGE="$TOPK/global/libnextgen_v3_global_bridge.so"
test -s "$LOCAL"
test -s "$MERGE"
grep -Fq "  $LOCAL" "$TOPK/binary-sha256.txt"
grep -Fq "  $MERGE" "$TOPK/binary-sha256.txt"
[[ $(sha256sum "$MERGE" | awk '{print $1}') == 41fe1ad28ab31d3e562a72f86ef88b49804931e1ca8db990fc7eadb8c28983a9 ]]
OUT=$(mktemp -d "$(dirname "$VENV")/v3b-$MODE.XXXXXX")
exec > >(tee "$OUT/run.log") 2>&1
echo "OUTPUT_DIR=$OUT"
git -C "$REPO" rev-parse HEAD | tee "$OUT/test_commit.txt"
g++ -std=c++17 -O3 -shared -fPIC -ffp-contract=off "$DIR/b16_mtp_cpp_planner.cpp" -o "$OUT/libplanner.so"
unset PYTHONOPTIMIZE
"$VPY" -m py_compile "$DIR/nextgen_v3b_topk_test.py" "$DIR/b16_score_queue_test.py" "$DIR/nextgen_v3b_resource_gate.py"
"$VPY" -u "$DIR/nextgen_v3b_topk_test.py" --mode host --planner "$OUT/libplanner.so"
if [[ "$MODE" == host ]]; then echo V3B_HOST_ONLY_PASS; exit 0; fi
PHYSICAL="${NEXTGEN_PHYSICAL_DEVICE:-}"
[[ "$PHYSICAL" == 0 || "$PHYSICAL" == 6 || "$PHYSICAL" == 7 ]]
[[ "${CONFIRM_NEXTGEN_NPU_IDLE:-0}" == 1 ]]
set +u
source "$CANN/set_env.sh"
set -Eeuo pipefail
export ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL" DEVICE_ID=0 PYTHONNOUSERSITE=1
export USER="${USER:-$(id -un)}"
export PATH="$VENV/bin:$PATH" PYTHON="$VPY"
echo "SELECTED_PHYSICAL_DEVICE=$PHYSICAL"
# Local lock serializes this validation entry, not other teams. The
# operator must still reserve the selected physical device exclusively.
exec 9>"$(dirname "$VENV")/v3b-npu${PHYSICAL}.lock"
flock -n 9 || { echo "V3B_VALIDATION_LOCK_BUSY"; exit 2; }
"$VPY" -u "$DIR/nextgen_v3b_resource_gate.py" --physical "$PHYSICAL" --log "$OUT/initial-npu.log"
# One process for the entire matrix. No fresh Context between cases.
# A total timeout still bounds a stuck device launch.
timeout --signal=TERM --kill-after=30s 6300s "$VPY" -u "$DIR/nextgen_v3b_topk_test.py" \
    --mode device --planner "$OUT/libplanner.so" --library "$SO" --local "$LOCAL" --merge "$MERGE" --cann "$CANN" \
    --matrix-out "$OUT" 2>&1 | tee "$OUT/matrix-device.log"
grep -q '^V3B_SINGLE_WORKER_PASS ' "$OUT/matrix-device.log"
for CASE in smoke smoke_q2 v05 v07 v08 v09 v10; do
    grep -q "^V3B_CASE_PASS $CASE " "$OUT/matrix-device.log"
done
sha256sum --strict -c "$BUILD/binary-sha256.txt"
sha256sum --strict -c "$TOPK/binary-sha256.txt"
"$VPY" - "$OUT" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);rows=[]
for case in ("smoke","smoke_q2","v05","v07","v08","v09","v10"):
    r=json.loads((out/(case+".json")).read_text())
    assert r["status"]=="PASS"
    c,a=(r["policies"][k]["timing"] for k in ("count","adaptive"))
    direct=c["direct"]["p50_us"];three=c["three_stage"]["p50_us"];dyn=a["three_stage"]["p50_us"]
    row=dict(case=case,score_count=c["score"]["p50_us"],score_adaptive=a["score"]["p50_us"],
             count_direct=direct,count_three_stage=three,adaptive_three_stage=dyn,
             reduction_vs_count_direct_percent=100*(1-dyn/direct),order_exact=r["order_exact"])
    rows.append(row);print("V3B_SUMMARY",json.dumps(row))
(out/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
PY
echo V3B_TOPK_MATRIX_PASS
echo "OUTPUT_DIR=$OUT"
