#!/usr/bin/env bash
# One-shot LIFetch batch: B8/B16 x miss 10/30/50, three invocations, ~1 line per case.
# Usage: bash scripts/bench_lifetch_batch.sh <tag> <device> <kvaf-server> [extra test args]
# Writes tests/results/lifetch-batch/<tag>/ and prints a compact table plus the
# same-round device split for each case.
set -uo pipefail
TAG="${1:?tag, e.g. main or update-opt}"
DEVICE="${2:?device id}"
KVAF="${3:?kvaf host:port}"
shift 3
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/tests/results/lifetch-batch/$TAG"
mkdir -p "$OUT"
printf 'case\tstatus\tp50_us\tspeedup\n'
# The event dump only exists on the instrumented branch; keep main runnable too.
EXTRA=()
if python3 "$ROOT/tests/operators/lifetch/test_lifetch.py" --help 2>/dev/null | grep -q -- "--dump-events"; then
  EXTRA+=(--dump-events)
fi
for MISS in 10 30 50; do
  LOG="$OUT/miss$MISS.log"
  python3 "$ROOT/tests/operators/lifetch/test_lifetch.py" \
    --device "$DEVICE" --kvaf-server "$KVAF" \
    --batches 8,16 --history 131072 --resident 16384 --index-heads 32 \
    --miss-percent "$MISS" --intermediate-fetches 8 --warmup 4 --iterations 20 \
    --trace-rounds 3 "${EXTRA[@]}" "$@" >"$LOG" 2>&1
  STATUS=$?
  P50=$(grep -oE 'prototype_with_update:[0-9.]+' "$LOG" | head -1 | sed 's/.*://')
  CORE=$(grep -oE 'prototype_core:[0-9.]+' "$LOG" | head -1 | sed 's/.*://')
  SPEED=$(grep -oE 'full_speedup=[0-9.]+x' "$LOG" | head -1 | sed 's/.*=//')
  printf 'miss%s\t%s\t%s\t%s\n' "$MISS" "$STATUS" "${P50:-?}" "${SPEED:-?}"
  [ -n "${CORE:-}" ] && printf '  miss%s core_only_p50=%s\n' "$MISS" "$CORE"
  grep -E '^round=[0-9]+ ' "$LOG" | sed "s/^/  miss$MISS /"
  grep -E 'actual_reads (native|prototype_with_update)' "$LOG" | sed "s/^/  miss$MISS /"
  sed -n '/^service events:/,/^mean lane span/p' "$LOG" | sed "s/^/  miss$MISS /"
  if [ "$STATUS" -ne 0 ]; then tail -5 "$LOG"; fi
done
echo "logs: $OUT"
