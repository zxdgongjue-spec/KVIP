#!/usr/bin/env bash
# Generated; serial 42-case runs for T1/T4/T8 at V56/V64 (252 conditions).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
KVIP_ROOT="${KVIP_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
for arg in "$@"; do
    case "$arg" in
        --backend|--backend=*|--suite|--suite=*|--shards|--shards=*|--budget|--budget=*|--output|--output=*)
            printf '%s\n' 'compare.sh fixes backend/suite/T/budget/output; configure root with KVIP_OUTPUT' >&2
            exit 2 ;;
    esac
done
comparison_root="${KVIP_OUTPUT:-$KVIP_ROOT/tests/results/split-compare-$(date -u +%Y%m%dT%H%M%S)-$$}"
failed=0
for budget in 28/56 32/64; do
    for shards in 1 4 8; do
        tag="t${shards}-c${budget%/*}-v${budget#*/}"
        if ! KVIP_OUTPUT="$comparison_root/$tag" KVIP_BUDGET="$budget" \
             KVIP_TOPK_SHARDS="$shards" bash "$SCRIPT_DIR/customer.sh" "$@"; then
            failed=1
        fi
    done
done
exit "$failed"
