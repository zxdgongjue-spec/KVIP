#!/usr/bin/env bash
# Compile-only: no NPU runtime calls, no service, no install, no production build.
set -Eeuo pipefail
umask 077
[[ $# == 1 ]] || { echo "Usage: $0 URMA_MANIFEST" >&2; exit 2; }
MANIFEST=$(realpath "$1")
source "$MANIFEST"
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
SRC="$REPO/csrc/operators/lightning_indexer_with_fetch/experiments/nextgen_v2_score_queue"
[[ "$KVIP_SHA" == 79c6a22557ed63a53da687e58f1f5bc1401fd8ed ]]
[[ "$KVAF_SHA" == 87e261ae4d07d4868cfdbb306d15f3f137edfc3d ]]
[[ "$SOC" == Ascend950DT_9582 && "$TRANSPORT" == urma ]]
[[ $(git -C "$NEW_KVIP" rev-parse HEAD) == "$KVIP_SHA" ]]
[[ $(git -C "$NEW_KVAF" rev-parse HEAD) == "$KVAF_SHA" ]]
git -C "$NEW_KVIP" diff --quiet HEAD
git -C "$REPO" diff --quiet HEAD
git -C "$REPO" diff --quiet "$KVIP_SHA" HEAD -- csrc python ':(exclude)csrc/operators/lightning_indexer_with_fetch/experiments/nextgen_v2_score_queue/**'
[[ $(git -C "$NEW_KVAF" diff --binary HEAD | sha256sum | awk '{print $1}') == 8132c738cb7e7a79fe1dc14f173954cc0110be322e14b25274debed15a8a9052 ]]
HEADERS="$NEW_KVAF/access_operator/algorithms/get/direct/urma_aiv/include"
test -f "$HEADERS/kvaf_batch_get_device.h"
OUT=$(mktemp -d "$(dirname "$VENV")/v2a-score-build.XXXXXX")
exec > >(tee "$OUT/run.log") 2>&1
trap 'rc=$?; if (( rc != 0 )); then echo "V2A_BUILD_FAILED rc=$rc OUT=$OUT"; fi' EXIT
echo "OUTPUT_DIR=$OUT"
git -C "$REPO" rev-parse HEAD | tee "$OUT/test_commit.txt"
cp "$MANIFEST" "$OUT/urma_manifest.env"
sha256sum --strict -c <<EOF
8c0c71a6806f00eb9539fe0bfeaabd6e194a0c252c88999d922f8578131fb2b7  $LIFETCH_LIBRARY
EOF
sha256sum "$SRC/CMakeLists.txt" "$SRC/b16_score_queue.cpp" "$SRC/launch.cpp" \
  "$REPO/csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/indexer_vector.h" \
  "$HEADERS/kvaf_batch_get_device.h" > "$OUT/source-sha256.txt"
set +u
source "$CANN/set_env.sh"
set -Eeuo pipefail
export USER="${USER:-$(id -un)}"
export PATH="$VENV/bin:$PATH" PYTHON="$VPY"
export ASCEND_HOME_PATH="$CANN" ASCEND_CANN_PACKAGE_PATH="$CANN" SOC_VERSION="$SOC"
cmake -S "$SRC" -B "$OUT/build" \
  -DSOC_VERSION="$SOC" -DASCEND_CANN_PACKAGE_PATH="$CANN" -DKVAF_INCLUDE="$HEADERS" \
  -DPython3_EXECUTABLE="$VPY"
cmake --build "$OUT/build" --parallel 2
SO="$OUT/build/libnextgen_v2_score_bridge.so"
test -s "$SO"
nm -D --defined-only "$SO" > "$OUT/exported-symbols.txt"
grep -Eq '[[:space:]]v9_b16_score_clip$' "$OUT/exported-symbols.txt"
ldd "$SO" | tee "$OUT/ldd.txt"
if grep -q 'not found' "$OUT/ldd.txt"; then exit 3; fi
sha256sum "$SO" "$OUT/build/isolated_indexer/indexer_vector.h" | tee "$OUT/binary-sha256.txt"
git -C "$REPO" diff --quiet HEAD
git -C "$NEW_KVIP" diff --quiet HEAD
sha256sum --strict -c <<EOF
8c0c71a6806f00eb9539fe0bfeaabd6e194a0c252c88999d922f8578131fb2b7  $LIFETCH_LIBRARY
EOF
echo "V2A_SCORE_COMPILE_ONLY_PASS"
echo "OUTPUT_DIR=$OUT"
