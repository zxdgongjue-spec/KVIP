#!/usr/bin/env bash
# Run the original MR14 history sweep, then the five-batch LIFetch comparison.
set -euo pipefail
LI_VALIDATION_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python3}" "$LI_VALIDATION_DIR/run_validation_a5.py" "$@"
