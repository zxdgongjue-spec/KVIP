#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runner="$here/run_small_batch_a5.sh"
case "${1:-}" in
    --prepare-deps|--build-deps)
        exec bash "$runner" "$@"
        ;;
esac

args=()
perf_only=false
for arg in "$@"; do
    case "$arg" in
        --perf-only) perf_only=true ;;
        --trace) printf '%s\n' 'This runner captures trace only after timing passes.' >&2; exit 2 ;;
        -h|--help) exec bash "$runner" --retired-study --help ;;
        *) args+=("$arg") ;;
    esac
done

bash "$runner" --retired-study "${args[@]}"
if [[ "$perf_only" == false ]]; then
    bash "$runner" --retired-study "${args[@]}" --trace --warmup 0 --repeats 1
fi
