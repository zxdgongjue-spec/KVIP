#!/usr/bin/env bash
# Parameter sweep for one LIFetch build: epoch count first, then prefetch budget.
# Usage: bash scripts/sweep_lifetch.sh <device> <kvaf-server> [miss-percent] [extra test args]
# Prints one line per case so the whole sweep lands as a single table.
set -uo pipefail
DEVICE="${1:?device id}"
KVAF="${2:?kvaf host:port}"
MISS="${3:-50}"
shift $(( $# > 3 ? 3 : $# ))
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/tests/results/lifetch-sweep"
mkdir -p "$OUT"

run_case() {
  local tag="$1"; shift
  local log="$OUT/$tag.log"
  python3 "$ROOT/tests/operators/lifetch/test_lifetch.py" \
    --device "$DEVICE" --kvaf-server "$KVAF" \
    --batches 16 --history 131072 --resident 16384 --index-heads 32 \
    --miss-percent "$MISS" --warmup 4 --iterations 20 --trace-rounds 3 "$@" >"$log" 2>&1
  local status=$?
  local p50 core li fetch line
  p50=$(grep -oE 'prototype_with_update:[0-9.]+' "$log" | head -1 | sed 's/.*://')
  core=$(grep -oE 'prototype_core:[0-9.]+' "$log" | head -1 | sed 's/.*://')
  line=$(grep -E '^round=1 ' "$log" | tail -1)
  li=$(printf '%s' "$line" | grep -oE 'li_us=[0-9.]+' | cut -d= -f2)
  fetch=$(printf '%s' "$line" | grep -oE 'fetch_us=[0-9.]+' | cut -d= -f2)
  local reads final unused span
  reads=$(grep 'actual_reads prototype_with_update' "$log" | tail -1)
  final=$(printf '%s' "$reads" | grep -oE '"final": \[[0-9]+, [0-9]+\]' | grep -oE '[0-9]+, [0-9]+' | tr -d ' ')
  unused=$(printf '%s' "$reads" | grep -oE '"unused": \[[0-9]+, [0-9]+\]' | grep -oE '[0-9]+, [0-9]+' | tr -d ' ')
  span=$(grep -E '^total\t' "$log" | tail -1 | cut -f5)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$tag" "$status" "${p50:-?}" "${core:-?}" "${li:-?}" "${fetch:-?}" "${final:-?}" "${unused:-?}" "${span:-?}"
}

printf 'case\tstatus\tp50_us\tcore_us\tli_us\tfetch_us\tfinal\tunused\tsvc_ticks\n'
for W in 4 6 8 12; do
  run_case "W$W" --intermediate-fetches "$W"
done
for P in 256 1024 2048; do
  run_case "P$P" --intermediate-fetches 8 --prefetch-tokens "$P"
done
echo "logs: $OUT"
