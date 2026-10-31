#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
python tools/run_conformance.py "$@"
python tools/check_contract.py
python tools/verify_baseline.py
if command -v ruff >/dev/null 2>&1; then
    ruff check --config pyproject.toml kvip tools \
        tests/v9/test_public_contract.py tests/v9/test_public_session.py \
        tests/v9/test_consumer_fake.py tests/v9/test_transport_contract.py \
        tests/v9/test_owned_boundary.py tests/v9/test_inputs.py \
        tests/v9/test_launcher_binding.py tests/v9/test_graph_session.py \
        tests/v9/test_qualification_gate.py tests/v9/test_conformance_tools.py \
        tests/v9/test_demo_cli.py tests/v9/test_kvaf_compatibility.py
fi
