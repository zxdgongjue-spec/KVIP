#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for arg in "$@"; do
    if [[ $arg == --comparison-dry-run || $arg == --help || $arg == -h ]]; then
        exec "${KVIP_PYTHON:-python3}" -B "$here/run_mr25_score.py" "$@"
    fi
done
exec bash "$here/run_mr28_container.sh" --mr25-score "$@"
