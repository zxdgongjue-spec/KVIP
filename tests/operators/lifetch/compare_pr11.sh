#!/usr/bin/env bash
# Compare main and PR11 using the same NPU, KVAF and workload.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
kvaf="${1:?Usage: bash tests/operators/lifetch/compare_pr11.sh KVAF_ROOT DEVICE HOST:PORT}"
device="${2:?Missing device ID}"
server="${3:?Missing KVAF CPU service address}"
kvaf="$(cd "$kvaf" && pwd)"
run="$repo/tests/results/pr11-ab/$(date +%Y%m%d_%H%M%S)_$$"
baseline="$run/main"
mkdir -p "$run"
echo "Results: $run"
echo "Main: $(git -C "$repo" rev-parse origin/main)"
echo "PR11: $(git -C "$repo" rev-parse HEAD)"
git -C "$repo" worktree add --detach "$baseline" origin/main
# The same installed LIDU extension and vendor are used by both runs.
mkdir -p "$baseline/csrc/operators/lidu/_custom_opp"
ln -s "$repo/csrc/operators/lidu/_custom_opp/vendors" \
    "$baseline/csrc/operators/lidu/_custom_opp/vendors"
source "$repo/scripts/env.sh"
for miss in 10 30 50; do
    for version in main pr11; do
        code="$repo"
        if [[ "$version" == main ]]; then code="$baseline"; fi
        echo "=== $version miss=$miss% ==="
        (
            cd "$code"
            export PYTHONPATH="$code/python:$kvaf/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
            python tests/operators/lifetch/test_lifetch.py \
                --transport urma --device "$device" --kvaf-root "$kvaf" \
                --kvaf-server "$server" --batches 16 \
                --history 131072 --resident 16384 --index-heads 32 \
                --intermediate-fetches 8 --prefetch-tokens 512 \
                --miss-percent "$miss" --warmup 4 --iterations 20 \
                --trace-rounds 0 --timeout-seconds 240 --output "$run/$version-$miss"
        ) 2>&1 | tee "$run/$version-$miss.log"
    done
done
python - "$run" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
lines = ['Graph p50 in us; correctness checks outside timing.',
         'miss | version | native_chain | lifetch_core | full_with_update']
for miss in (10, 30, 50):
    for version in ('main', 'pr11'):
        text = (root / f'{version}-{miss}.log').read_text()
        matches = re.findall(r'graph_p50_us=native_chain:([\d.]+),\s*(?:round_init:[\d.]+,\s*)?prototype_core:([\d.]+),\s*prototype_with_update:([\d.]+)', text)
        if len(matches) != 1 or 'PASS' not in text:
            raise SystemExit(f'Missing checked summary: {root / f"{version}-{miss}.log"}')
        lines.append(f'{miss}% | {version} | ' + ' | '.join(matches[0]))
result = '\n'.join(lines) + '\n'
(root / 'summary.log').write_text(result)
print(result, end='')
print(f'Summary: {root / "summary.log"}')
PY
