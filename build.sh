#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-package}" in
  package) python3 -m pip install --no-build-isolation -e "$ROOT" ;;
  v9) shift; python3 -m kvip.v9 "$@" ;;
  native-lidu) bash "$ROOT/operators/native_lidu/build_lidu_fused_batch_get_prepare_mtp.sh" ;;
  *) echo "usage: bash build.sh {package|v9 [build options]|native-lidu}" >&2; exit 2 ;;
esac
