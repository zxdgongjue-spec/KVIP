#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case ${1:-} in
    --prepare-deps|--build-deps)
        exec bash "$here/run_mr28_unified_score.sh" "$@"
        ;;
esac
for arg in "$@"; do
    if [[ $arg == --dry-run || $arg == --help || $arg == -h ]]; then
        exec "${KVIP_PYTHON:-python3}" -B "$here/run_small_batch_a5.py" "$@"
    fi
done
exec bash "$here/run_mr28_container.sh" --small-batch-a5 "$@"
