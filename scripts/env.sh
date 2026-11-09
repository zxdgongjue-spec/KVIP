#!/usr/bin/env bash
_kvip_env() {
    local root dependency vendor selection transport
    root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    selection="$(python3 - "$root" <<'PY'
import json,sys
from pathlib import Path
root = Path(sys.argv[1]); state = root / 'build/dependencies.json'
data = json.loads(state.read_text()) if state.is_file() else {}
print(data.get('kvaf_build_dir', str(Path(data.get('kvaf', root / 'third_party/KVAF')) / 'build-npu')))
print(data.get('transport', 'urma'))
PY
)" || return 1
    dependency="${selection%$'\n'*}"
    transport="${selection##*$'\n'}"
    export PYTHONPATH="$root/python:$dependency/python${PYTHONPATH:+:$PYTHONPATH}"
    export KVAF_BATCH_GET_KERNEL="$dependency/bin/kvaf_execute_batch_get_kernels.o"
    export KVAF_BATCH_PUT_KERNEL="$dependency/bin/kvaf_execute_batch_put_kernels.o"
    local libraries=("$root"/csrc/operators/lidu/_custom_opp/vendors/*/op_api/lib/libcust_opapi.so)
    if [[ "${#libraries[@]}" != 1 || ! -f "${libraries[0]}" ]]; then
        echo "Build KVIP first: bash $root/scripts/build.sh" >&2
        return 1
    fi
    vendor="${libraries[0]%/op_api/lib/libcust_opapi.so}"
    case ":${ASCEND_CUSTOM_OPP_PATH:-}:" in
        *":$vendor:"*) ;;
        *) export ASCEND_CUSTOM_OPP_PATH="$vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}" ;;
    esac
}
_kvip_env
_kvip_result=$?
unset -f _kvip_env
return "$_kvip_result"
