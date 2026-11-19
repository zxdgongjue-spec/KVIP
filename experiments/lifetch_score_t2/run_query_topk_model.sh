#!/usr/bin/env bash
set -euo pipefail
LI_QUERY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python3}" "$LI_QUERY_DIR/run_query_model.py" "$@"
