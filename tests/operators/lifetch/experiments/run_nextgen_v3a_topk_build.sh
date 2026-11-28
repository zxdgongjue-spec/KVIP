#!/usr/bin/env bash
# Independent compile-only targets. No device initialization, no KVAF service.
set -Eeuo pipefail
umask 077
[[ $# == 3 ]] || { echo "Usage: $0 URMA_MANIFEST V2A_BUILD_EVIDENCE V2B_HOST_EVIDENCE"; exit 2; }
MANIFEST=$(realpath "$1")
SCORE=$(realpath "$2")
HOST=$(realpath "$3")
source "$MANIFEST"
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
ROOT="$REPO/csrc/operators/lightning_indexer_with_fetch"
[[ "$KVIP_SHA" == 79c6a22557ed63a53da687e58f1f5bc1401fd8ed ]]
[[ "$KVAF_SHA" == 87e261ae4d07d4868cfdbb306d15f3f137edfc3d ]]
[[ "$SOC" == Ascend950DT_9582 && "$TRANSPORT" == urma ]]
[[ $(git -C "$NEW_KVIP" rev-parse HEAD) == "$KVIP_SHA" ]]
[[ $(git -C "$NEW_KVAF" rev-parse HEAD) == "$KVAF_SHA" ]]
git -C "$NEW_KVIP" diff --quiet HEAD
git -C "$REPO" diff --quiet HEAD
git -C "$REPO" diff --quiet 0381d4231e15636b3f4192f9734760be3900cbea HEAD -- csrc python \
    ':(exclude)csrc/operators/lightning_indexer_with_fetch/experiments/nextgen_v3_local_topk/**' \
    ':(exclude)csrc/operators/lightning_indexer_with_fetch/experiments/nextgen_v3_global_topk/**'
[[ $(git -C "$NEW_KVAF" diff --binary HEAD | sha256sum | awk '{print $1}') == 8132c738cb7e7a79fe1dc14f173954cc0110be322e14b25274debed15a8a9052 ]]
grep -Fxq V2A_SCORE_COMPILE_ONLY_PASS "$SCORE/run.log"
grep -Fxq 2ebce41df37bbda33f3bfade65bbd503bd4e6138 "$SCORE/test_commit.txt"
sha256sum --strict -c "$SCORE/binary-sha256.txt"
grep -Fxq V2B_HOST_ONLY_PASS "$HOST/run.log"
grep -Fxq 0381d4231e15636b3f4192f9734760be3900cbea "$HOST/test_commit.txt"
[[ $(git -C "$REPO" hash-object "$ROOT/op_kernel/indexer/topk.h") == a387b61f5c53d651eeadee703b80893070623f73 ]]
[[ $(git -C "$REPO" hash-object "$ROOT/op_kernel/indexer/topk_common.h") == b99c0e4fa43ac341c9482a5d29aa96424bebdaa1 ]]
HEADERS="$NEW_KVAF/access_operator/algorithms/get/direct/urma_aiv/include"
test -f "$HEADERS/kvaf_batch_get_device.h"
OUT=$(mktemp -d "$(dirname "$VENV")/v3a-topk-build.XXXXXX")
exec > >(tee "$OUT/run.log") 2>&1
trap 'rc=$?; if (( rc != 0 )); then echo "V3A_BUILD_FAILED rc=$rc OUT=$OUT"; fi' EXIT
echo "OUTPUT_DIR=$OUT"
git -C "$REPO" rev-parse HEAD | tee "$OUT/test_commit.txt"
cp "$MANIFEST" "$OUT/urma_manifest.env"
printf '%s\n' "$SCORE" > "$OUT/v2a_evidence_path.txt"
printf '%s\n' "$HOST" > "$OUT/v2b_host_evidence_path.txt"
sha256sum --strict -c <<EOF
8c0c71a6806f00eb9539fe0bfeaabd6e194a0c252c88999d922f8578131fb2b7  $LIFETCH_LIBRARY
EOF
sha256sum "$ROOT"/experiments/nextgen_v3_{local,global}_topk/{CMakeLists.txt,launch.cpp} \
    "$ROOT/experiments/nextgen_v3_local_topk/b16_local_topk.cpp" \
    "$ROOT/experiments/nextgen_v3_global_topk/b16_global_topk.cpp" \
    "$ROOT/op_kernel/indexer/topk.h" "$ROOT/op_kernel/indexer/topk_common.h" \
    "$HEADERS/kvaf_batch_get_device.h" > "$OUT/source-sha256.txt"
set +u
source "$CANN/set_env.sh"
set -Eeuo pipefail
export USER="${USER:-$(id -un)}"
export PATH="$VENV/bin:$PATH" PYTHON="$VPY"
export ASCEND_HOME_PATH="$CANN" ASCEND_CANN_PACKAGE_PATH="$CANN" SOC_VERSION="$SOC"
for KIND in local global; do
    SRC="$ROOT/experiments/nextgen_v3_${KIND}_topk"
    DIR="$OUT/$KIND"
    cmake -S "$SRC" -B "$DIR" -DSOC_VERSION="$SOC" \
        -DASCEND_CANN_PACKAGE_PATH="$CANN" -DKVAF_INCLUDE="$HEADERS"
    cmake --build "$DIR" --target "nextgen_v3_${KIND}_bridge" --parallel 2
    SO="$DIR/libnextgen_v3_${KIND}_bridge.so"
    test -s "$SO"
    nm -D --defined-only "$SO" > "$OUT/$KIND-symbols.txt"
    grep -Eq "[[:space:]]v9_b16_${KIND}_topk$" "$OUT/$KIND-symbols.txt"
    ldd "$SO" | tee "$OUT/$KIND-ldd.txt"
    if grep -q 'not found' "$OUT/$KIND-ldd.txt"; then exit 3; fi
    sha256sum "$SO" | tee -a "$OUT/binary-sha256.txt"
    echo "V3A_TOPK_COMPILE_PASS $KIND"
done
git -C "$REPO" diff --quiet HEAD
git -C "$NEW_KVIP" diff --quiet HEAD
sha256sum --strict -c "$SCORE/binary-sha256.txt"
sha256sum --strict -c <<EOF
8c0c71a6806f00eb9539fe0bfeaabd6e194a0c252c88999d922f8578131fb2b7  $LIFETCH_LIBRARY
EOF
echo V3A_TOPK_COMPILE_ONLY_PASS
echo "OUTPUT_DIR=$OUT"
