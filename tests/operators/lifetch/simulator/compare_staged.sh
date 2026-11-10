#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# CANN A5 simulator: standalone BS16, one resolver AIV/request, no Fetch.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
kvaf="${1:?Usage: compare_staged.sh KVAF_INCLUDE [CANN_ROOT]}"
cann="${2:-/usr/local/Ascend/cann}"
kvaf="$(cd "$kvaf" && pwd)"
run="$repo/tests/results/staged-resolver-sim/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run"
child=''
cleanup() {
  local status=$?
  if [[ -n "$child" ]]; then
    kill -TERM -- "-$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  if ((status)); then printf "FAIL exit=%s logs=%s\n" "$status" "$run" | tee "$run/summary.log"; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
command -v setsid >/dev/null || { echo 'setsid is required for simulator process cleanup' >&2; exit 1; }
set +u
source "$cann/set_env.sh"
set -u
echo "Results: $run"
cmake -S "$repo/tests/operators/lifetch/simulator" -B "$run/build" \
  -DRUN_MODE=sim -DASCEND_CANN_PACKAGE_PATH="$cann" \
  -DOP_ROOT="$repo/csrc/operators/lightning_indexer_with_fetch_safe" -DKVAF_INCLUDE="$kvaf" \
  > "$run/configure.log" 2>&1
cmake --build "$run/build" --target resolver_batch_sim staged_dma2_batch_sim -j4 \
  > "$run/build.log" 2>&1
for mode in baseline staged cross; do
  exe=resolver_batch_sim
  args=''
  if [[ "$mode" != baseline ]]; then exe=staged_dma2_batch_sim; fi
  if [[ "$mode" == cross ]]; then args=' cross-query'; fi
  echo "=== $mode: waiting for CANN simulator ==="
  setsid msprof op simulator --application="$run/build/$exe$args" --soc-version=Ascend950PR_9599 \
    --output="$run/$mode" --launch-count=1 --timeout=2880 > "$run/$mode.log" 2>&1 &
  pid=$!
  child=$pid
  start=$SECONDS
  while kill -0 "$pid" 2>/dev/null; do
    sleep 20
    if kill -0 "$pid" 2>/dev/null; then
      bytes=$(wc -c < "$run/$mode.log")
      printf 'RUNNING %s +%ss log_bytes=%s log=%s\n' "$mode" "$((SECONDS-start))" "$bytes" "$run/$mode.log"
    fi
  done
  set +e
  wait "$pid"
  status=$?
  set -e
  tail -30 "$run/$mode.log"
  if ((status != 0)); then
    grep -n -m 3 -B 4 -A 8 -E '(^FAIL|\[ERROR\]|error:|Traceback)' "$run/$mode.log" || true
    exit "$status"
  fi
  child=''
done
python3 - "$run" "$repo/tests/operators/lifetch/simulator" <<'PY'
import re
import sys
from pathlib import Path
root=Path(sys.argv[1])
sys.path.insert(0,sys.argv[2])
from compare_staged_trace import output_maxima
values={}
for name in ('baseline','staged'):
    text=(root / f'{name}.log').read_text()
    if ('PASS resolver BS=16 workers=1' not in text or 'All task success' not in text
            or (name == 'staged' and 'STAGED phase_unique_repairs_per_request=[141,205,0,0]' not in text)):
        raise SystemExit(f'{name}: simulator and correctness must both pass; see {root/name}')
    rows=re.findall(r'^core\d+\.veccore\d+\s+([\d.]+)\s+([\d.]+)', text, re.M)
    if len(rows) != 16:
        raise SystemExit(f'{name}: expected 16 per-core timings, got {len(rows)}')
    values[name]=max(float(row[0]) for row in rows)
cross=(root/'cross.log').read_text()
if ('PASS resolver BS=16 workers=1' not in cross or 'All task success' not in cross
        or 'STAGED phase_unique_repairs_per_request=[141,204,0,1]' not in cross):
    raise SystemExit(f'cross-query staged correctness missing; see {root / "cross.log"}')
base_output=output_maxima(root,'baseline',1)
q_outputs=output_maxima(root,'staged',4)
ratio=values['staged']/values['baseline']
summary=(f'BS16 Q4 one AIV/request, 410 unique misses, 512 prefetched (64 selected), 346 final; no Fetch.\n'
         f'baseline_max_core_us={values["baseline"]:.3f} staged_max_core_us={values["staged"]:.3f} '
         f'ratio={ratio:.3f} change_pct={(ratio-1)*100:+.2f}% '
         f'target_25pct={"PASS" if ratio <= 1.25 else "FAIL"}\n'
         f'baseline_output_max_us={base_output[0]:.3f} staged_q_output_max_us='
         f'{[round(x,3) for x in q_outputs]}; each maximum may be a different core.\n'
         'Cross-query different-position duplicate: PASS.\n'
         'Simulator model times are not hardware latency; correctness checks and profile are both required.\n')
(root/'summary.log').write_text(summary)
print(summary,end='')
print(f'Summary: {root / "summary.log"}')
PY
