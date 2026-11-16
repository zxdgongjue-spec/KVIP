#!/usr/bin/env bash
set -eo pipefail
mode=--ranked-prefetch
if [[ ${1:-} == --minimum-half ]]; then mode=--ranked-floor; shift; fi
if [[ ${1:-} == --true-miss ]]; then mode=--ranked-miss; shift; fi
exec bash "$(dirname "${BASH_SOURCE[0]}")/compare.sh" "$mode" "$@"
