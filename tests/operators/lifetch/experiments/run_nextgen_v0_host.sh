#!/usr/bin/env bash
# CPU-only. No NPU, no service, no build/install in frozen operator tree.
set -Eeuo pipefail
umask 077
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
DIR="$REPO/tests/operators/lifetch/experiments"
git -C "$REPO" diff --quiet HEAD
git -C "$REPO" diff --quiet 79c6a22557ed63a53da687e58f1f5bc1401fd8ed HEAD -- csrc python
for name in b16_mtp_cpp_planner.cpp b16_cpp_planner_fast.cpp nextgen_v0_host.cpp run_nextgen_v0_host.sh; do
    test "$(git -C "$REPO" hash-object "$DIR/$name")" = "$(git -C "$REPO" rev-parse "HEAD:tests/operators/lifetch/experiments/$name")"
done
BASE=${1:-/home/z00945955/Fuse/nextgen-79c6a225}
mkdir -p "$BASE"
OUT=$(mktemp -d "$BASE/v0-host.XXXXXX")
exec > >(tee "$OUT/run.log") 2>&1
echo "OUTPUT_DIR=$OUT"
git -C "$REPO" rev-parse HEAD | tee "$OUT/test_commit.txt"
g++ --version
sha256sum "$DIR/b16_mtp_cpp_planner.cpp" "$DIR/b16_cpp_planner_fast.cpp" "$DIR/nextgen_v0_host.cpp" "$DIR/run_nextgen_v0_host.sh" | tee "$OUT/source-sha256.txt"
g++ -std=c++17 -O3 -Wall -Wextra -Werror -ffp-contract=off \
    -I"$REPO/csrc/operators/lightning_indexer_with_fetch/include" \
    -I"$REPO/csrc/operators/common/lifetch" \
    "$DIR/nextgen_v0_host.cpp" -o "$OUT/nextgen_v0_host"
sha256sum "$OUT/nextgen_v0_host" | tee "$OUT/binary-sha256.txt"
"$OUT/nextgen_v0_host" | tee "$OUT/host-results.log"
test "$(grep -c '^ROUTE_MODEL ' "$OUT/host-results.log")" = 14
test "$(grep -c '^PLANNER_PASS ' "$OUT/host-results.log")" = 11
test "$(grep -c '^PLANNER_EXPECTED_REJECT ' "$OUT/host-results.log")" = 3
grep -Fxq NEXTGEN_V0_HOST_CONTRACT_PASS "$OUT/host-results.log"
echo NEXTGEN_V0_REPO_PASS
echo "OUTPUT_DIR=$OUT"
