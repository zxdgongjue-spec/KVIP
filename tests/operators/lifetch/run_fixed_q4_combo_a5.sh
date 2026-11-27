#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python=${KVIP_PYTHON:-python3}
runner="$here/support/fixed_q4_combo_a5.py"

for arg in "$@"; do
    case "$arg" in
        --dry-run|-h|--help) exec "$python" -B "$runner" "$@" ;;
    esac
done

: "${KVIP_KVAF_ROOT:?Set KVIP_KVAF_ROOT to the built shared KVAF 813f1750 checkout}"
kvaf_pin=813f1750a50b7e6c92d5f5397f2fbc05befef6da
if [[ $(git -C "$KVIP_KVAF_ROOT" rev-parse HEAD) != "$kvaf_pin" ]]; then
    printf 'Expected KVAF %s at %s\n' "$kvaf_pin" "$KVIP_KVAF_ROOT" >&2
    exit 1
fi
build="$KVIP_KVAF_ROOT/build-npu"
for dependency in python/kvaf/__init__.py bin/kvaf_execute_batch_get_kernels.o bin/kvaf_execute_batch_put_kernels.o CMakeCache.txt; do
    if [[ ! -r "$build/$dependency" ]]; then
        printf 'Missing built dependency %s\n' "$build/$dependency" >&2
        exit 1
    fi
done
cann=$(sed -n 's/^ASCEND_CANN_PACKAGE_PATH:[^=]*=//p' "$build/CMakeCache.txt")
if [[ -z "$cann" || ! -r "$cann/set_env.sh" ]]; then
    printf 'KVAF build needs the CANN installation recorded in %s\n' "$build/CMakeCache.txt" >&2
    exit 1
fi
source "$cann/set_env.sh"
exec "$python" -B "$runner" "$@"
