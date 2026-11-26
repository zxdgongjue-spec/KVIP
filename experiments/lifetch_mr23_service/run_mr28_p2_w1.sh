#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case ${1:-} in
    --prepare-deps|--build-deps)
        exec bash "$here/run_mr28_unified_score.sh" "$@"
        ;;
esac
exec bash "$here/run_mr28_dedicated_service.sh" --p2-w1 "$@"
