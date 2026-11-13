#!/usr/bin/env bash
# Portable current-streaming native entry. Requires prepared NPU/KVAF environment.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
action=plan
case "${1:-}" in
    plan|run) action=$1; shift ;;
esac
exec "${KVIP_PYTHON:-python3}" "$repo/experiments/split_core/native_benchmark.py" "$action" "$@"
