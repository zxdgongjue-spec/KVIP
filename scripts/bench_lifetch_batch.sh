#!/usr/bin/env bash
# One-shot LIFetch batch: B8/B16 x miss 10/30/50, three invocations, ~1 line per case.
# Usage: bash scripts/bench_lifetch_batch.sh <tag> <device> <kvaf-server> [extra test args]
# Writes tests/results/lifetch-batch/<tag>/ and prints a compact table.
set -uo pipefail
TAG="${1:?tag, e.g. main or update-opt}"
DEVICE="${2:?device id}"
KVAF="${3:?kvaf host:port}"
shift 3
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/tests/results/lifetch-batch/$TAG"
mkdir -p "$OUT"
printf 'case\tstatus\tp50_us\tp99_us\tfull_speedup\treads\tunused\tinit\tli\tresolve\tfetch\n'
for MISS in 10 30 50; do
  LOG="$OUT/miss$MISS.log"
  python3 "$ROOT/tests/operators/lifetch/test_lifetch.py" \
    --device "$DEVICE" --kvaf-server "$KVAF" \
    --batches 8,16 --history 131072 --resident 16384 --index-heads 32 \
    --miss-percent "$MISS" --intermediate-fetches 8 --warmup 4 --iterations 20 \
    --trace-rounds 3 "$@" >"$LOG" 2>&1
  STATUS=$?
  P50=$(grep -oE 'prototype_with_update:[0-9.]+' "$LOG" | head -1 | cut -d: -f2)
  CORE=$(grep -oE 'prototype_core:[0-9.]+' "$LOG" | head -1 | cut -d: -f2)
  SPEED=$(grep -oE 'full_speedup=[0-9.]+x' "$LOG" | head -1 | cut -d= -f2)
  READS=$(grep -oE 'total=[0-9]+' "$LOG" | tail -1 | cut -d= -f2)
  UNUSED=$(grep -oE 'unused=[0-9]+' "$LOG" | tail -1 | cut -d= -f2)
  # Same-round device split from the core trace (round=0 on the primary profile pass).
  TRACE=$(grep -E '^round=0 ' "$LOG" | tail -1)
  STAGE=$(printf '%s' "$TRACE" | grep -oE '(init|li|resolve|fetch)_us=[0-9.]+' | cut -d= -f2 | paste -sd'\t' -)
  printf 'miss%s\t%s\t%s\t-\t%s\t%s\t%s\t%s\n' "$MISS" "$STATUS" "${P50:-?}" "${SPEED:-?}" "${READS:-?}" "${UNUSED:-?}" "${STAGE:-?}"
  [ "$STATUS" -ne 0 ] && tail -3 "$LOG"
done
echo "logs: $OUT"
