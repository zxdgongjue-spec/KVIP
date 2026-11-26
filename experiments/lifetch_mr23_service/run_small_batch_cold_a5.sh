#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help ]]; then
    printf 'Usage: %s [--dry-run]\n' "${0##*/}"
    printf '%s\n' 'Run two BS1/BS2 P2/P4 graph timing rounds, then capture one checked msprof trace per arm.'
    exit 0
fi
if (( $# > 1 )) || { (( $# == 1 )) && [[ $1 != --dry-run ]]; }; then
    printf 'Usage: %s [--dry-run]\n' "${0##*/}" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
pair=(bash "$here/run_small_batch_a5.sh" --direction1-matrix --batch 1 --batch 2 --miss-percent 10)

"${pair[@]}" --warmup 1 --iterations 5 --repeats 2 "$@"
"${pair[@]}" --trace --warmup 0 "$@"
