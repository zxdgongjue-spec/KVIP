#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case "${1:-}" in
    --prepare-deps|--build-deps)
        exec bash "$here/run_small_batch_a5.sh" "$@"
        ;;
esac
for arg in "$@"; do
    case "$arg" in
        --window-sweep|--balanced|--cold-window-matrix|--bs2-trace-pair|--bs2-trace-pair=*|--adaptive-pw|--window-probe|--window-probe=*|--bs2-window-q|--window-q1-matrix|--window-q-matrix)
            exec bash "$here/run_small_batch_a5.sh" "$@"
            ;;
    esac
done
exec bash "$here/run_small_batch_a5.sh" --full32 "$@"
