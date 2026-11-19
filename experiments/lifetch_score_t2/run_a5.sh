#!/usr/bin/env bash
set -euo pipefail
LI_T2_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python3}" "$LI_T2_SCRIPT_DIR/run_a5.py" "$@"
