#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runner="$here/run_small_batch_a5.sh"
case "${1:-}" in
    --prepare-deps|--build-deps|--help|-h)
        exec bash "$runner" "$@"
        ;;
    --perf-only)
        shift
        exec bash "$runner" --bs2-p4-service --warmup 2 "$@"
        ;;
    --trace-only)
        shift
        exec bash "$runner" --bs2-p4-service --trace "$@"
        ;;
    --dry-run)
        exec bash "$runner" --bs2-p4-service --warmup 2 "$@"
        ;;
esac

printf '%s\n' 'BS2 P4/W4: graph Event comparison'
bash "$runner" --bs2-p4-service --warmup 2 "$@"
printf '%s\n' 'BS2 P4/W4: checked msprof comparison'
exec bash "$runner" --bs2-p4-service --trace "$@"
