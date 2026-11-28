#!/usr/bin/env bash
# Read-only static audit and device ELF evidence. No context/service operations.
set -Eeuo pipefail
umask 077
[[ $# == 1 ]] || { echo "Usage: $0 V1_FAILED_RUN"; exit 2; }
ROOT=$(realpath "$1")
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
OP=/tmp/KVIP-integrated-79c6a225
BUILD=/tmp/KVAF-integrated-87e261ae/build-npu/lifetch/20ed0253a932fbdbdc73
CANN=/usr/local/Ascend/cann-9.2.1
[[ $(cat "$ROOT/test_commit.txt") == 5af188b6a8377e8cd2fa8b45fb8b86072c05074d ]]
[[ $(git -C "$OP" rev-parse HEAD) == 79c6a22557ed63a53da687e58f1f5bc1401fd8ed ]]
git -C "$OP" diff --quiet HEAD
git -C "$REPO" diff --quiet HEAD
OUT=$(mktemp -d "$(dirname "$ROOT")/v1-epoch-audit.XXXXXX")
exec > >(tee "$OUT/run.log") 2>&1
echo "OUTPUT_DIR=$OUT"
git -C "$REPO" rev-parse HEAD > "$OUT/test_commit.txt"
sha256sum --strict -c <<EOF
8c0c71a6806f00eb9539fe0bfeaabd6e194a0c252c88999d922f8578131fb2b7  $BUILD/libv9_bridge.so
EOF
/usr/local/python3.12.13/bin/python -u "$REPO/tests/operators/lifetch/experiments/audit_nextgen_v1_epochs.py" "$ROOT" "$OP" "$OUT"
OBJDUMP="$CANN/tools/bisheng_compiler/bin/llvm-objdump"
READELF="$CANN/tools/hcc/bin/aarch64-target-linux-gnu-readelf"
MS="$CANN/tools/msobjdump/msobjdump"
"$OBJDUMP" --version > "$OUT/llvm-version.txt" 2>&1 || true
"$MS" --help > "$OUT/msobjdump-help.txt" 2>&1 || true
for kind in aic aiv; do
    if [[ "$kind" == aic ]]; then
        OBJ="$BUILD/v9_kernel_aic_device_dir/device.o"
    else
        OBJ="$BUILD/v9_kernel_aiv_device_dir/device_aiv.o"
    fi
    test -s "$OBJ"
    sha256sum "$OBJ" | tee -a "$OUT/object-sha256.txt"
    file "$OBJ"
    "$READELF" -h -SW -Ws "$OBJ" > "$OUT/$kind-elf.txt" 2>&1 || true
    if "$OBJDUMP" -d --line-numbers --demangle "$OBJ" > "$OUT/$kind-disassembly.txt" 2> "$OUT/$kind-disassembly.stderr"; then
        echo "DISASSEMBLY_TOOL_EXIT_ZERO $kind"
    else
        echo "DISASSEMBLY_TOOL_UNSUPPORTED $kind"
        cat "$OUT/$kind-disassembly.stderr"
    fi
    grep -E 'v9_li_kernel|v9_streaming_resolve_update' "$OUT/$kind-elf.txt" | head -n 40 || true
    wc -l "$OUT/$kind-disassembly.txt"
done
# Raw object disassembly alone is not proof of runtime PC-to-symbol mapping.
echo "V1_STATIC_AUDIT_COMPLETE"
echo "OUTPUT_DIR=$OUT"
