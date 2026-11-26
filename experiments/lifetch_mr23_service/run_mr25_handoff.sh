#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --comparison-dry-run ]]; then
    exec "${KVIP_PYTHON:-python3}" -B "$here/run_mr25_handoff.py" "$@"
fi
exec bash "$here/run_mr28_container.sh" --mr25-handoff "$@"
