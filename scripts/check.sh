#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m ruff check python tests scripts
python -m ruff format --check python tests scripts
python -m pytest tests -q
