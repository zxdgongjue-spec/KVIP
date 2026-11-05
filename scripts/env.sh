#!/usr/bin/env bash
_kvip_env() {
    local root dependency vendor
    root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    dependency="$(python3 - "$root" <<'PY'
import json,sys
from pathlib import Path
root = Path(sys.argv[1]); state = root / 'build/dependencies.json'
print(json.loads(state.read_text())['kvaf'] if state.is_file() else root / 'third_party/KVAF')
PY
)" || return 1
    export PYTHONPATH="$root/python:$dependency/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
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
