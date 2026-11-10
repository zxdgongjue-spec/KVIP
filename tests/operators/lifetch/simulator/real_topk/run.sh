#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
kvaf_include="${1:?usage: run.sh KVAF_INCLUDE [OUTPUT_DIR]}"
run="${2:-$repo/tests/results/real-topk/$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$run"
run="$(cd "$run" && pwd)"
cann="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}"
set +u
source "$cann/set_env.sh"
set -u
PYTHONPATH="$repo/tests/operators/lifetch/support${PYTHONPATH:+:$PYTHONPATH}" python3 - "$run" <<'PY'
from pathlib import Path
import sys
import numpy as np
from real_fetch_inputs import prepare_full_chain_fixture
root = Path(sys.argv[1])
fixture = root / 'fixture'
prepare_full_chain_fixture(fixture, 131072, 16384, 10, index_heads=32)
scores = np.load(fixture / 'logical_scores.npy')
order = np.load(fixture / 'physical_to_logical.npy')
scores[:, order].astype('<u2').tofile(root / 'scores.bin')
PY
cmake -S "$repo/tests/operators/lifetch/simulator/real_topk" -B "$run/build" \
  -DKVAF_INCLUDE="$kvaf_include" -DASCEND_CANN_PACKAGE_PATH="$cann" > "$run/configure.log" 2>&1
cmake --build "$run/build" -j4 > "$run/build.log" 2>&1
for mode in baseline classify; do
  echo "RUNNING real ProcessTopK $mode; logs=$run"
  msprof op simulator --application="$run/build/real_topk_sim $mode $run/scores.bin" \
    --soc-version=Ascend950PR_9599 --output="$run/$mode" --launch-count=1 --timeout=1200 > "$run/$mode.log" 2>&1
  grep -E 'REAL_TOPK|core[0-9]+\.veccore[0-9]+' "$run/$mode.log"
done
python3 "$repo/tests/operators/lifetch/simulator/real_topk/check_trace.py" "$run/classify" | tee "$run/synchronization.log"
python3 - "$run" <<'PY'
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
timings = {}
for mode in ('baseline', 'classify'):
    log = (run / f'{mode}.log').read_text()
    if not re.search(r'REAL_TOPK classify=[01] invalid=0 wrong=0', log):
        raise SystemExit(f'{mode}: correctness validation failed; inspect {run / (mode + ".log")}')
    cores = re.findall(r'^core\d+\.veccore\d+\s+([\d.]+)', log, re.M)
    if len(cores) != 4:
        raise SystemExit(f'{mode}: expected four simulated AIV timings, found {len(cores)}')
    timings[mode] = max(map(float, cores))
text = (f"Real ProcessTopK Q4, B1, history=131072, resident=16384, heads=32, miss10%\n"
        "Fixture protects final hot hits and maps up to 512 final cold tokens as prefetched.\n"
        f"Simulated max-core us: baseline={timings['baseline']:.3f} "
        f"classified={timings['classify']:.3f} "
        f"delta={timings['classify']-timings['baseline']:+.3f}\n"
        "TopK and classified locations/bitmap verified; no score production, Fetch or NPU timing.\n")
(run / 'summary.log').write_text(text)
print(text, end='')
print(f'Results: {run}')
PY
