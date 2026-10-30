#!/usr/bin/env bash
# Source after building native-lidu. V9 itself is a ctypes bridge, not an OPP.
_kvop_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for _kvop_api in "$_kvop_root"/operators/native_lidu/_custom_opp/vendors/*/op_api/lib/libcust_opapi.so; do
  [[ -f "$_kvop_api" ]] || continue
  _kvop_vendor="$(cd "$(dirname "$_kvop_api")/../.." && pwd)"
  case ":${ASCEND_CUSTOM_OPP_PATH:-}:" in
    *":$_kvop_vendor:"*) ;;
    *) export ASCEND_CUSTOM_OPP_PATH="$_kvop_vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}" ;;
  esac
done
unset _kvop_root _kvop_api _kvop_vendor
