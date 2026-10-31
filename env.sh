#!/usr/bin/env bash
# Source after building native-lidu. V9 itself is a ctypes bridge, not an OPP.
_kvip_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for _kvip_api in "$_kvip_root"/operators/native_lidu/_custom_opp/vendors/*/op_api/lib/libcust_opapi.so; do
  [[ -f "$_kvip_api" ]] || continue
  _kvip_vendor="$(cd "$(dirname "$_kvip_api")/../.." && pwd)"
  case ":${ASCEND_CUSTOM_OPP_PATH:-}:" in
    *":$_kvip_vendor:"*) ;;
    *) export ASCEND_CUSTOM_OPP_PATH="$_kvip_vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}" ;;
  esac
done
unset _kvip_root _kvip_api _kvip_vendor
