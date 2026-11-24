#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../.." && pwd)
shared="$repo/experiments/lifetch_mr23_service"
program="$here/run_a5_prefetch.py"
python=${KVIP_PYTHON:-python3}
a5_root=${KVIP_A5_ROOT:-/home/w00568585/li-aligned-a5}

case ${1:-} in
    --prepare-deps|--build-deps) exec bash "$shared/run_mr28_unified_score.sh" "$@" ;;
esac
for arg in "$@"; do
    case "$arg" in
        --dry-run|--help|-h) exec "$python" -B "$program" "$@" ;;
    esac
done

export KVIP_KVAF_ROOT="${KVIP_KVAF_ROOT:-"$a5_root/KVAF-unified-813f1750"}"
export KVIP_KVAF_SERVER="${KVIP_KVAF_SERVER:-127.0.0.1:28986}"
export KVIP_A5_DEVICE="${KVIP_A5_DEVICE:-0}"
export KVIP_AB_OUTPUT="${KVIP_AB_OUTPUT:-"$a5_root/results"}"
if [[ ! -r "$KVIP_KVAF_ROOT/build-npu/CMakeCache.txt" ]]; then
    printf '%s\n' 'Build the unified KVAF Runtime in the A5 container with --build-deps.' >&2
    exit 1
fi
cann=$(sed -n 's/^ASCEND_CANN_PACKAGE_PATH:[^=]*=//p' "$KVIP_KVAF_ROOT/build-npu/CMakeCache.txt")
if [[ -z $cann || ! -r "$cann/set_env.sh" ]]; then
    printf '%s\n' 'Cannot find the CANN environment used by the unified KVAF Runtime.' >&2
    exit 1
fi
source "$cann/set_env.sh"

lidu=${KVIP_LIDU_ROOT:-"$repo/csrc/operators/lidu"}
opp=${KVIP_LIDU_OPP:-"$lidu/_custom_opp/vendors/customize"}
if [[ -r "$opp/op_api/lib/libcust_opapi.so" ]]; then
    export KVIP_LIDU_OPP="$opp"
    export ASCEND_CUSTOM_OPP_PATH="$opp${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
fi
export PYTHONPATH="$lidu/torch_extension:$KVIP_KVAF_ROOT/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
export KVAF_BATCH_GET_KERNEL="$KVIP_KVAF_ROOT/build-npu/bin/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$KVIP_KVAF_ROOT/build-npu/bin/kvaf_execute_batch_put_kernels.o"
test -r "$KVAF_BATCH_GET_KERNEL" && test -r "$KVAF_BATCH_PUT_KERNEL" || {
    printf '%s\n' 'Unified BatchGet/BatchPut kernel files are missing. Use --build-deps in the A5 container.' >&2
    exit 1
}
"$python" -c 'import acl, kvaf, torch, torch_npu, nanovllm_dsa_lidu_fused_batch_get_prepare_mtp; getattr(kvaf.NpuRuntime, "get_graph_urma_resources")'

kvaf_host=${KVIP_KVAF_SERVER%:*}
kvaf_port=${KVIP_KVAF_SERVER##*:}
if ! "$python" -c 'import socket, sys; socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1).close()' "$kvaf_host" "$kvaf_port"; then
    printf 'Start unified KVAF Runtime at %s before running this script.\n' "$KVIP_KVAF_SERVER" >&2
    exit 1
fi

exec "$python" -B "$program" "$@"
