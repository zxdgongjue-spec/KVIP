#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
SRC="$ROOT/csrc/operators/lightning_indexer_with_fetch"
: "${KVAF_INCLUDE:?Set KVAF_INCLUDE to the patched KVAF device include directory}"
ASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann-9.2.1}"
SOC_VERSION="${SOC_VERSION:-Ascend950DT_9582}"
BUILD_DIR="${BUILD_DIR:-/tmp/kvip-nextgen-v4b1-finalizer-build}"

mkdir -p "$BUILD_DIR"
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    grep -F "CMAKE_HOME_DIRECTORY:INTERNAL=$SRC" "$BUILD_DIR/CMakeCache.txt" >/dev/null
fi

cmake -S "$SRC" -B "$BUILD_DIR" \
    -DRUN_MODE=npu \
    -DSOC_VERSION="$SOC_VERSION" \
    -DASCEND_CANN_PACKAGE_PATH="$ASCEND_CANN_PACKAGE_PATH" \
    -DKVAF_INCLUDE="$KVAF_INCLUDE" \
    -DKVIP_ENABLE_NEXTGEN_DYNAMIC=ON

cmake --build "$BUILD_DIR" --target v9_bridge --parallel "${BUILD_JOBS:-8}"

LIB="$BUILD_DIR/libv9_bridge.so"
test -s "$LIB"
nm -D "$LIB" | grep -E '[[:space:]]kvip_nextgen_finalize_and_resolve$'

echo "NEXTGEN_V4_B1_FINALIZER_BUILD_PASS"
echo "LIBRARY=$LIB"
sha256sum "$LIB"
