#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Real LI resolver + Index-update A/B on CANN simulator; no KV Fetch/SFA.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../../../.." && pwd)"
kvaf_arg="${1:?Usage: run.sh KVAF_ROOT_OR_INCLUDE [CANN_ROOT]}"
cann="${2:-/usr/local/Ascend/cann}"
if [[ -d "$kvaf_arg/access_operator/demo/aiv_batch_get/include" ]]; then
 kvaf="$kvaf_arg/access_operator/demo/aiv_batch_get/include"
else kvaf="$kvaf_arg"; fi
kvaf="$(cd "$kvaf" && pwd)"
run="$repo/tests/results/resolver-index-pipeline/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run"
child=''
cleanup(){ local rc=$?; if [[ -n "$child" ]]; then kill -TERM -- "-$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi; if ((rc)); then printf 'FAIL exit=%s results=%s\n' "$rc" "$run" | tee "$run/summary.log"; fi; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
command -v setsid >/dev/null || { echo 'setsid is required to stop simulator children' >&2; exit 1; }
set +u; source "$cann/set_env.sh"; set -u
printf 'Results: %s\n' "$run"
(git -C "$repo" rev-parse HEAD; git -C "$repo" status --short; sha256sum "$here"/*.cpp "$here"/*.h) > "$run/source-identity.log" 2>&1
cmake -S "$here" -B "$run/build" -DRUN_MODE=sim -DASCEND_CANN_PACKAGE_PATH="$cann" -DOP_ROOT="$repo/csrc/operators/lightning_indexer_with_fetch_safe" -DKVAF_INCLUDE="$kvaf" > "$run/configure.log" 2>&1 || { tail -60 "$run/configure.log"; exit 1; }
cmake --build "$run/build" --target pipeline_ab_parallel_sim -j2 > "$run/build.log" 2>&1 || { grep -n -m 5 -B 5 -A 8 -E 'fatal error:|error:' "$run/build.log" || tail -60 "$run/build.log"; exit 1; }
for fixture in normal collision cross-query; do
 for mode in serial pipeline; do
  launches=8 # Include possible event-marker launches and every update kernel.
  name="$mode-$fixture"
  printf 'START %s\n' "$name"
  # CANN's --timeout is in minutes; enforce a separate wall-clock deadline.
  setsid timeout --signal=TERM --kill-after=10s 300s msprof op simulator --application="$run/build/pipeline_ab_parallel_sim $mode $fixture" --soc-version=Ascend950PR_9599 --output="$run/$name" --launch-count="$launches" --timeout=5 > "$run/$name.log" 2>&1 &
  child=$!; start=$SECONDS
  while kill -0 "$child" 2>/dev/null; do sleep 20; if kill -0 "$child" 2>/dev/null; then printf 'RUNNING %s +%ss bytes=%s\n' "$name" "$((SECONDS-start))" "$(wc -c < "$run/$name.log")"; fi; done
  set +e; wait "$child"; rc=$?; set -e; child=''
  tail -25 "$run/$name.log"
  if ((rc)); then grep -n -m 3 -B 4 -A 8 -E '(^FAIL|\[ERROR\]|error:|Traceback)' "$run/$name.log" || true; exit "$rc"; fi
 done
done
python3 - "$run" <<'PY'
import json,re,sys
from pathlib import Path
run=Path(sys.argv[1])
def timeline(mode,fixture):
 root=run/f'{mode}-{fixture}'
 paths=[p for p in root.rglob('trace.json') if re.fullmatch(r'core\d+\.veccore\d+',p.parent.name) and p.parent.parent.name=='simulator']
 by_op={}
 for path in paths:
  rows=[e for e in json.loads(path.read_text())['traceEvents'] if e.get('ph')=='X' and 'dur' in e and 'ts' in e]
  if not rows:raise SystemExit(f'empty trace {path}')
  name=path.parents[3].name
  start=min(float(e['ts']) for e in rows);end=max(float(e['ts'])+float(e['dur']) for e in rows)
  by_op.setdefault(name,[]).append((start,end))
 if mode=='serial':
  wanted=('test_ready_resolver','kvop_direct_update_kernel','kvop_direct_apply_kernel')
 else:wanted=('test_pipeline_ab_parallel',)
 groups=[(name,vals) for name,vals in by_op.items() if any(name.startswith(w) for w in wanted)]
 if len(groups)!=len(wanted):raise SystemExit(f'{mode}/{fixture}: expected {wanted}; found {sorted(by_op)}')
 start=min(a for _,vals in groups for a,b in vals);end=max(b for _,vals in groups for a,b in vals)
 return end-start,[(name,round(max(b for a,b in vals)-min(a for a,b in vals),3)) for name,vals in groups]
lines=['BS1 Q4: serial 1/1/4 AIV vs pipeline 1+1 AIV; no Fetch/SFA.',
 'Fixture preparation excluded; update mailbox, shadow-map initialization and metadata commit included.']
for fixture in ('normal','collision','cross-query'):
 rows={}
 for mode in ('serial','pipeline'):
  text=(run/f'{mode}-{fixture}.log').read_text()
  hit=re.search(r'^PASS mode=(\w+) fixture=([\w-]+) final_miss=(\d+) q_counts=\[([^]]+)\] event_status=(-?\d+) event_ms=([\d.eE+-]+)',text,re.M)
  if not hit or 'All task success' not in text: raise SystemExit(f'{mode}/{fixture}: missing verified completion')
  layout=re.search(r'^LAYOUT_HASH .* full=([0-9a-f]+) hot=([0-9a-f]+)',text,re.M)
  if not layout: raise SystemExit(f'{mode}/{fixture}: missing final layout hash')
  rows[mode]=(hit,layout)
 old,new=rows['serial'],rows['pipeline']
 if old[0].group(3,4)!=new[0].group(3,4) or old[1].group(2)!=new[1].group(2): raise SystemExit(f'{fixture}: serial/pipeline hot outputs do not match')
 a,a_ops=timeline('serial',fixture);b,b_ops=timeline('pipeline',fixture)
 perf=f'serial_trace_us={a:.3f} pipeline_trace_us={b:.3f} change_pct={(b/a-1)*100:+.2f}% serial_ops={a_ops} pipeline_ops={b_ops}'
 lines.append(f'{fixture}: PASS final_miss={old[0].group(3)} q_counts=[{old[0].group(4)}] hot layout match; cold layout may differ; {perf}')
for mode in ('serial','pipeline'):
 for row in (run/f'{mode}-normal.log').read_text().splitlines():
  if row.startswith('PIPELINE_CYCLES '):lines.append(row+' (device cycles; do not assume ns)')
lines.append('Simulator timing is not hardware latency; no network/CQ/SFA overlap is modeled.')
out='\n'.join(lines)+'\n';(run/'summary.log').write_text(out);print(out,end='');print(f'Summary: {run / "summary.log"}')
PY
