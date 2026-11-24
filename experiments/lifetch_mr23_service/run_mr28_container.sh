#!/usr/bin/env bash
set -eo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mode=${1:-}
case "$mode" in
    --mr25-handoff|--mr25-score) shift ;;
esac
a5_root=${KVIP_A5_ROOT:-/home/w00568585/li-aligned-a5}
export KVIP_KVAF_ROOT="${KVIP_KVAF_ROOT:-"$a5_root/KVAF-score-t2-1c131038"}"
export KVIP_A5_DEVICE="${KVIP_A5_DEVICE:-0}"
export KVIP_KVAF_SERVER=127.0.0.1:28985

# Match the CANN installation used to build the existing KVAF runtime.
cann_root=$(sed -n 's/^ASCEND_CANN_PACKAGE_PATH:[^=]*=//p' \
    "$KVIP_KVAF_ROOT/build-npu/CMakeCache.txt")
if [[ -z "$cann_root" || ! -f "$cann_root/set_env.sh" ]]; then
    printf 'Cannot find the CANN environment from %s/build-npu/CMakeCache.txt\n' "$KVIP_KVAF_ROOT" >&2
    exit 1
fi
source "$cann_root/set_env.sh"
export KVIP_PYTHON="${KVIP_PYTHON:-$(command -v python3)}"

lidu="$a5_root/KVIP/csrc/operators/lidu"
opp="$lidu/_custom_opp/vendors/customize"
if [[ -z ${KVIP_LIDU_OPP:-} && -f "$opp/op_api/lib/libcust_opapi.so" ]]; then
    export KVIP_LIDU_OPP="$opp"
fi
if compgen -G "$lidu/torch_extension/nanovllm_dsa_lidu_fused_batch_get_prepare_mtp/_C*.so" >/dev/null; then
    export PYTHONPATH="$lidu/torch_extension${PYTHONPATH:+:$PYTHONPATH}"
fi
"$KVIP_PYTHON" -c 'import acl, torch, torch_npu, nanovllm_dsa_lidu_fused_batch_get_prepare_mtp'

results_root=${KVIP_AB_OUTPUT:-"$a5_root/results"}
mkdir -p "$results_root"
export KVIP_AB_OUTPUT
KVIP_AB_OUTPUT=$(mktemp -d "$results_root/mr28.XXXXXX")
printf 'Run output: %s\n' "$KVIP_AB_OUTPUT"

kvaf_port_open() {
    "$KVIP_PYTHON" - <<'PY'
import socket
import sys

try:
    with socket.create_connection(("127.0.0.1", 28985), timeout=1):
        pass
except OSError:
    sys.exit(1)
PY
}

if ! kvaf_port_open; then
    service_log="$KVIP_AB_OUTPUT/kvaf-service.log"
    nohup "$KVIP_KVAF_ROOT/build-cpu/runtime/kvaf_cpu_service" \
        --listen 127.0.0.1 --port 28985 </dev/null >"$service_log" 2>&1 &
    for attempt in {1..15}; do
        if kvaf_port_open; then
            break
        fi
        sleep 1
    done
    if ! kvaf_port_open; then
        printf 'KVAF is not listening on 127.0.0.1:28985. See %s\n' "$service_log" >&2
        tail -n 30 "$service_log" >&2
        exit 1
    fi
fi

if [[ $mode == --mr25-handoff ]]; then
    "$KVIP_PYTHON" -B "$here/run_mr25_handoff.py" "$@" 2>&1 | tee "$KVIP_AB_OUTPUT/console.log"
elif [[ $mode == --mr25-score ]]; then
    "$KVIP_PYTHON" -B "$here/run_mr25_score.py" "$@" 2>&1 | tee "$KVIP_AB_OUTPUT/console.log"
else
    bash "$here/run_a5.sh" "$@" 2>&1 | tee "$KVIP_AB_OUTPUT/console.log"
fi
