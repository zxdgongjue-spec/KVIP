#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q tests/v9
python tools/check_contract.py
python tools/verify_baseline.py
if command -v ruff >/dev/null 2>&1; then
    ruff check kvop tools tests/v9/test_public_contract.py tests/v9/test_public_session.py \
        tests/v9/test_consumer_fake.py tests/v9/test_transport_contract.py
fi
