#!/usr/bin/env bash
set -eo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mode=${1:-}
case "$mode" in
    --mr25-handoff|--mr25-score|--mr28-unified-score|--mr28-baseline-msprof) shift ;;
esac
a5_root=${KVIP_A5_ROOT:-/home/w00568585/li-aligned-a5}
unified_runtime=false
if [[ $mode == --mr25-handoff || $mode == --mr25-score || $mode == --mr28-unified-score || $mode == --mr28-baseline-msprof ]]; then
    unified_runtime=true
    checkout=$(cd -- "$here/../.." && pwd)
    export KVIP_KVAF_ROOT="${KVIP_KVAF_ROOT:-"$a5_root/KVAF-unified-813f1750"}"
    export KVIP_KVAF_SERVER="${KVIP_KVAF_SERVER:-127.0.0.1:28986}"
    lidu="$checkout/csrc/operators/lidu"
    if [[ $(git -C "$KVIP_KVAF_ROOT" rev-parse HEAD 2>/dev/null) != 813f1750a50b7e6c92d5f5397f2fbc05befef6da ]]; then
        printf '%s\n' 'Prepare the unified KVAF checkout on the host with --prepare-deps.' >&2
        exit 1
    fi
    for kind in get put; do
        if [[ ! -f "$KVIP_KVAF_ROOT/build-npu/bin/kvaf_execute_batch_${kind}_kernels.o" ]]; then
            printf '%s\n' 'Build the unified KVAF runtime in the A5 container with --build-deps.' >&2
            exit 1
        fi
    done
    export KVAF_BATCH_GET_KERNEL="$KVIP_KVAF_ROOT/build-npu/bin/kvaf_execute_batch_get_kernels.o"
    export KVAF_BATCH_PUT_KERNEL="$KVIP_KVAF_ROOT/build-npu/bin/kvaf_execute_batch_put_kernels.o"
else
    export KVIP_KVAF_ROOT="${KVIP_KVAF_ROOT:-"$a5_root/KVAF-score-t2-1c131038"}"
    export KVIP_KVAF_SERVER=127.0.0.1:28985
    lidu="$a5_root/KVIP/csrc/operators/lidu"
fi
export KVIP_A5_DEVICE="${KVIP_A5_DEVICE:-0}"
kvaf_host=${KVIP_KVAF_SERVER%:*}
kvaf_port=${KVIP_KVAF_SERVER##*:}

# Match the CANN installation used to build the existing KVAF runtime.
cann_root=$(sed -n 's/^ASCEND_CANN_PACKAGE_PATH:[^=]*=//p' \
    "$KVIP_KVAF_ROOT/build-npu/CMakeCache.txt")
if [[ -z "$cann_root" || ! -f "$cann_root/set_env.sh" ]]; then
    printf 'Cannot find the CANN environment from %s/build-npu/CMakeCache.txt\n' "$KVIP_KVAF_ROOT" >&2
    exit 1
fi
source "$cann_root/set_env.sh"
export KVIP_PYTHON="${KVIP_PYTHON:-$(command -v python3)}"

if [[ $mode != --mr28-baseline-msprof ]]; then
    opp="${KVIP_LIDU_OPP:-"$lidu/_custom_opp/vendors/customize"}"
    if [[ -z ${KVIP_LIDU_OPP:-} && -f "$opp/op_api/lib/libcust_opapi.so" ]]; then
        export KVIP_LIDU_OPP="$opp"
    fi
    if [[ -f "$opp/op_api/lib/libcust_opapi.so" ]]; then
        export ASCEND_CUSTOM_OPP_PATH="$opp${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
    fi
    if compgen -G "$lidu/torch_extension/nanovllm_dsa_lidu_fused_batch_get_prepare_mtp/_C*.so" >/dev/null; then
        export PYTHONPATH="$lidu/torch_extension${PYTHONPATH:+:$PYTHONPATH}"
    fi
    "$KVIP_PYTHON" -c 'import acl, torch, torch_npu, nanovllm_dsa_lidu_fused_batch_get_prepare_mtp'
else
    "$KVIP_PYTHON" -c 'import acl, torch, torch_npu'
fi
if [[ $unified_runtime == true ]]; then
    PYTHONPATH="$KVIP_KVAF_ROOT/build-npu/python${PYTHONPATH:+:$PYTHONPATH}" \
        "$KVIP_PYTHON" -c 'import kvaf; getattr(kvaf.NpuRuntime, "get_graph_urma_resources")'
fi

results_root=${KVIP_AB_OUTPUT:-"$a5_root/results"}
mkdir -p "$results_root"
export KVIP_AB_OUTPUT
KVIP_AB_OUTPUT=$(mktemp -d "$results_root/mr28.XXXXXX")
printf 'Run output: %s\n' "$KVIP_AB_OUTPUT"

kvaf_port_open() {
    "$KVIP_PYTHON" - "$kvaf_host" "$kvaf_port" <<'PY'
import socket
import sys

try:
    with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1):
        pass
except OSError:
    sys.exit(1)
PY
}

if ! kvaf_port_open; then
    if [[ $kvaf_host != 127.0.0.1 && $kvaf_host != localhost ]]; then
        printf 'KVAF is not listening on %s\n' "$KVIP_KVAF_SERVER" >&2
        exit 1
    fi
    service_log="$KVIP_AB_OUTPUT/kvaf-service.log"
    nohup "$KVIP_KVAF_ROOT/build-cpu/runtime/kvaf_cpu_service" \
        --listen "$kvaf_host" --port "$kvaf_port" </dev/null >"$service_log" 2>&1 &
    for attempt in {1..15}; do
        if kvaf_port_open; then
            break
        fi
        sleep 1
    done
    if ! kvaf_port_open; then
        printf 'KVAF is not listening on %s. See %s\n' "$KVIP_KVAF_SERVER" "$service_log" >&2
        tail -n 30 "$service_log" >&2
        exit 1
    fi
fi

if [[ $mode == --mr25-handoff ]]; then
    "$KVIP_PYTHON" -B "$here/run_mr25_handoff.py" "$@" 2>&1 | tee "$KVIP_AB_OUTPUT/console.log"
elif [[ $mode == --mr25-score ]]; then
    "$KVIP_PYTHON" -B "$here/run_mr25_score.py" "$@" 2>&1 | tee "$KVIP_AB_OUTPUT/console.log"
elif [[ $mode == --mr28-unified-score ]]; then
    "$KVIP_PYTHON" -B "$here/run_mr28_unified_score.py" "$@" 2>&1 | tee "$KVIP_AB_OUTPUT/console.log"
elif [[ $mode == --mr28-baseline-msprof ]]; then
    "$KVIP_PYTHON" -B "$here/run_mr28_baseline_msprof.py" "$@" 2>&1 | tee "$KVIP_AB_OUTPUT/console.log"
else
    bash "$here/run_a5.sh" "$@" 2>&1 | tee "$KVIP_AB_OUTPUT/console.log"
fi
