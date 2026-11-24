#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../.." && pwd)
a5_root=${KVIP_A5_ROOT:-/home/w00568585/li-aligned-a5}
revision=813f1750a50b7e6c92d5f5397f2fbc05befef6da
export KVIP_KVAF_ROOT="${KVIP_KVAF_ROOT:-"$a5_root/KVAF-unified-813f1750"}"
export KVIP_KVAF_SERVER="${KVIP_KVAF_SERVER:-127.0.0.1:28986}"
export KVIP_A5_DEVICE="${KVIP_A5_DEVICE:-0}"
export KVIP_AB_OUTPUT="${KVIP_AB_OUTPUT:-"$a5_root/results"}"
python=${KVIP_PYTHON:-python3}

case ${1:-} in
    --comparison-dry-run|--help|-h)
        exec "$python" -B "$here/run.py" "$@"
        ;;
    --prepare-deps)
        if [[ ! -e "$KVIP_KVAF_ROOT" ]]; then
            git init "$KVIP_KVAF_ROOT"
        fi
        if ! git -C "$KVIP_KVAF_ROOT" rev-parse --verify HEAD >/dev/null 2>&1; then
            git -C "$KVIP_KVAF_ROOT" fetch --depth=1 https://gitcode.com/CacheOS/KVAF.git "refs/keep-around/$revision"
            git -C "$KVIP_KVAF_ROOT" switch --detach FETCH_HEAD
        fi
        "$python" -B "$repo/scripts/build.py" --prepare-only --transport urma --kvaf-root "$KVIP_KVAF_ROOT"
        umdk="$KVIP_KVAF_ROOT/third_party/umdk"
        if [[ ! -e "$umdk" ]]; then
            mkdir -p "$(dirname -- "$umdk")"
            git clone --depth=1 --branch br_UMDK_24.12.0_LTS https://gitcode.com/openeuler/umdk.git "$umdk"
        fi
        test -f "$umdk/include/urma_api.h"
        exit
        ;;
esac

if [[ $(git -C "$KVIP_KVAF_ROOT" rev-parse HEAD 2>/dev/null) != "$revision" ]]; then
    printf 'Run --prepare-deps on the A5 host to select KVAF %s.\n' "$revision" >&2
    exit 1
fi

build_deps=false
cache="$KVIP_KVAF_ROOT/build-npu/CMakeCache.txt"
if [[ ${1:-} == --build-deps ]]; then
    build_deps=true
    shift
    test -f "$KVIP_KVAF_ROOT/third_party/umdk/include/urma_api.h"
    cann=${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-}}
    if [[ -z "$cann" && -f "$cache" ]]; then
        cann=$(sed -n 's/^ASCEND_CANN_PACKAGE_PATH:[^=]*=//p' "$cache")
    fi
    cann=${cann:-/usr/local/Ascend/cann}
else
    if [[ ! -f "$cache" ]]; then
        printf '%s\n' 'Build the prepared dependencies with --build-deps in the A5 container.' >&2
        exit 1
    fi
    cann=$(sed -n 's/^ASCEND_CANN_PACKAGE_PATH:[^=]*=//p' "$cache")
fi
if [[ ! -f "$cann/set_env.sh" ]]; then
    printf 'CANN environment missing at %s/set_env.sh\n' "$cann" >&2
    exit 1
fi
set +u
source "$cann/set_env.sh"
set -u

if $build_deps; then
    "$python" -m pybind11 --cmakedir >/dev/null
    PYTHON="$python" PIP_NO_INDEX=1 bash "$repo/scripts/build.sh" \
        --transport urma --kvaf-root "$KVIP_KVAF_ROOT" --cann "$cann" "$@"
    exit
fi

bin="$KVIP_KVAF_ROOT/build-npu/bin"
export KVAF_BATCH_GET_KERNEL="$bin/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$bin/kvaf_execute_batch_put_kernels.o"
test -f "$KVAF_BATCH_GET_KERNEL"
test -f "$KVAF_BATCH_PUT_KERNEL"
export PYTHONPATH="$repo/python:$KVIP_KVAF_ROOT/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
"$python" -B - <<'PY'
import acl
import torch
import torch_npu
import kvaf
getattr(kvaf.NpuRuntime, "get_graph_urma_resources")
PY

host=${KVIP_KVAF_SERVER%:*}
port=${KVIP_KVAF_SERVER##*:}
port_open() {
    "$python" -B - "$host" "$port" <<'PY'
import socket
import sys
try:
    with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1):
        pass
except OSError:
    sys.exit(1)
PY
}
if ! port_open; then
    if [[ $host != 127.0.0.1 && $host != localhost ]]; then
        printf 'KVAF is not listening at %s.\n' "$KVIP_KVAF_SERVER" >&2
        exit 1
    fi
    mkdir -p "$KVIP_AB_OUTPUT"
    service_log=$(mktemp "$KVIP_AB_OUTPUT/kvaf-813f1750.XXXXXX")
    nohup "$KVIP_KVAF_ROOT/build-cpu/runtime/kvaf_cpu_service" \
        --listen "$host" --port "$port" </dev/null >"$service_log" 2>&1 &
    for attempt in {1..15}; do
        if port_open; then break; fi
        sleep 1
    done
    if ! port_open; then
        printf 'KVAF did not start. See %s.\n' "$service_log" >&2
        tail -n 30 "$service_log" >&2
        exit 1
    fi
fi
exec "$python" -B "$here/run.py" "$@"
