#!/usr/bin/env bash
# Three requested paths: SDK serial / numeric-direct serial / legacy fused.
set -Eeuo pipefail
LI_SERIAL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)
exec "${LI_CHAIN_PYTHON:-python3}" "$LI_SERIAL_ROOT/tests/operators/lifetch/serial_chain_compare.py" "$@"
