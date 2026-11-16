#!/usr/bin/env bash
set -eo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$here/compare.sh" --window-topk-only "$@"
