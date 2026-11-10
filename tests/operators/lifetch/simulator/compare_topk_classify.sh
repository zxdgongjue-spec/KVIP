#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Simulator-only experiment; no LIFetch forward or network submission.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cann="${1:-/usr/local/Ascend/cann}"
run="$repo/tests/results/topk-classify-sim/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run"
child=''
cleanup(){ local rc=$?; if [[ -n "$child" ]]; then kill -TERM -- "-$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi; if ((rc)); then printf 'FAIL exit=%s logs=%s\n' "$rc" "$run"; fi; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
command -v setsid >/dev/null
set +u; source "$cann/set_env.sh"; set -u
printf 'Results: %s\n' "$run"
cmake -S "$repo/tests/operators/lifetch/simulator" -B "$run/build" -DRUN_MODE=sim -DASCEND_CANN_PACKAGE_PATH="$cann" -DOP_ROOT="$repo/csrc/operators/lightning_indexer_with_fetch_safe" > "$run/configure.log" 2>&1
cmake --build "$run/build" --target topk_classify_sim -j4 > "$run/build.log" 2>&1
for mode in topk classify; do
 args=''; [[ "$mode" != classify ]] || args=' classify'
 setsid msprof op simulator --application="$run/build/topk_classify_sim$args" --soc-version=Ascend950PR_9599 --output="$run/$mode" --launch-count=1 --timeout=2880 > "$run/$mode.log" 2>&1 &
 child=$!; start=$SECONDS
 while kill -0 "$child" 2>/dev/null; do sleep 20; if kill -0 "$child" 2>/dev/null; then printf 'RUNNING %s +%ss log=%s\n' "$mode" "$((SECONDS-start))" "$run/$mode.log"; fi; done
 set +e; wait "$child"; rc=$?; set -e; child=''; tail -25 "$run/$mode.log"; ((rc==0)) || exit "$rc"
done
python3 - "$run" <<'PY'
import re,sys
from pathlib import Path
p=Path(sys.argv[1]); vals={}
for mode in ('topk','classify'):
 s=(p/f'{mode}.log').read_text(); rows=re.findall(r'^core\d+\.veccore\d+\s+([\d.]+)\s+[\d.]+',s,re.M)
 if 'PASS topk Q4 4 AIV' not in s or 'All task success' not in s or len(rows)!=4: raise SystemExit(f'{mode}: missing correctness or four timing rows')
 vals[mode]=max(map(float,rows))
summary=(f'Q4 4 AIV, BS1, history=131072, topk=2048, hot=16384; standalone LI TopK algorithm, no fetch.\n'
 f'topk_max_us={vals["topk"]:.3f} topk_classify_max_us={vals["classify"]:.3f} delta_us={vals["classify"]-vals["topk"]:.3f}\n'
 'Each maximum may be from a different AIV. Simulator model time is not device latency.\n')
(p/'summary.log').write_text(summary); print(summary,end=''); print(f'Summary: {p / "summary.log"}')
PY
