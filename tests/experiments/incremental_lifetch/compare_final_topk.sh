#!/usr/bin/env bash
set -eo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/compare.sh" --final-topk-only "$@"
