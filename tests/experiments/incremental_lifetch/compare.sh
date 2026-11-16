#!/usr/bin/env bash
set -eo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
kvaf_root=${1:?Usage: compare.sh KVAF_ROOT DEVICE SERVER [BATCHES] [MISS]}
device=${2:?Missing device}
server=${3:?Missing server}
batches=${4:-16}
miss=${5:-10}
run="$repo/tests/results/incremental-lifetch/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run"
python "$repo/tests/experiments/incremental_lifetch/prepare.py" "$run/incremental-source"
common=(--transport urma --device "$device" --kvaf-root "$kvaf_root" --kvaf-server "$server"
  --batches "$batches" --miss-percent "$miss" --history 131072 --resident 16384 --index-heads 32
  --prefetch-tokens 512 --intermediate-fetches 8 --warmup 4 --iterations 20 --timeout-seconds 600)
for variant in baseline incremental; do
  source_root="$repo"
  if [[ "$variant" == incremental ]]; then source_root="$run/incremental-source"; fi
  echo "=== $variant: BS=$batches miss=$miss%; graph timing + level 3 ==="
  python "$source_root/tests/operators/lifetch/test_lifetch.py" diagnose "${common[@]}" --level 3 \
    --output "$run/$variant" 2>&1 | tee "$run/$variant.log"
done
python - "$run" <<'PY'
import json
import sys
from pathlib import Path
run=Path(sys.argv[1])
rows={}
for variant in ('baseline','incremental'):
    files=list((run/variant).rglob('summary.json'))
    files=[p for p in files if 'cases' in json.loads(p.read_text())]
    if len(files)!=1:
        raise SystemExit(f'{variant}: expected one case summary, got {files}')
    rows[variant]={int(r['batch']):r for r in json.loads(files[0].read_text())['cases']}
lines=['BS | Check A/B | Baseline mean | Incremental mean | Reduction | Baseline p50/p99 | Incremental p50/p99 (us)']
for batch,a in rows['baseline'].items():
    b=rows['incremental'][batch]
    if not a['accepted'] or not b['accepted']:
        raise SystemExit(f'BS{batch}: correctness or timing incomplete')
    gain=100*(1-b['fused_mean_us']/a['fused_mean_us'])
    lines.append(f"{batch:2d} | PASS/PASS | {a['fused_mean_us']:13.2f} | {b['fused_mean_us']:16.2f} | {gain:+8.2f}% | "
                 f"{a['fused_p50_us']:.2f}/{a['fused_p99_us']:.2f} | {b['fused_p50_us']:.2f}/{b['fused_p99_us']:.2f}")
text='\n'.join(lines)
(run/'summary.log').write_text(text+'\n')
print(text)
print(f'Full logs and stage traces: {run}')
PY
