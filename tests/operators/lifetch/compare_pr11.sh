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
trap 'status=$?; if (( status != 0 )); then printf "FAIL: exit=%s; full logs: %s\n" "$status" "$run" | tee "$run/summary.log"; fi' EXIT
echo "Results: $run"
echo "Main: $(git -C "$repo" rev-parse origin/main)"
echo "PR11: $(git -C "$repo" rev-parse HEAD)"
git -C "$repo" worktree add --detach "$baseline" origin/main
# The same installed LIDU extension and vendor are used by both runs.
mkdir -p "$baseline/csrc/operators/lidu/_custom_opp"
ln -s "$repo/csrc/operators/lidu/_custom_opp/vendors" \
    "$baseline/csrc/operators/lidu/_custom_opp/vendors"
source "$repo/scripts/env.sh"
# Keep kernel sources unchanged; use the same diagnostic harness on both versions.
cp "$repo/tests/operators/lifetch/test_lifetch.py" "$baseline/tests/operators/lifetch/test_lifetch.py"
cp "$repo/tests/operators/lifetch/support/lifetch.py" "$baseline/tests/operators/lifetch/support/lifetch.py"
cp "$repo/tests/operators/lifetch/support/device_trace.py" "$baseline/tests/operators/lifetch/support/device_trace.py"
for miss in 10; do
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
                --trace-rounds 3 --timeout-seconds 300 --output "$run/$version-$miss"
        ) 2>&1 | tee "$run/$version-$miss.log"
    done
done
python - "$run" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
lines = ["BS16, miss10%; full-graph profiler and unprofiled graph p50 are separate replays.",
         "Profiler durations are device intervals; fetch/update may overlap inside one kernel."]
representatives = {}
for version in ("main", "pr11"):
    text = (root / f"{version}-10.log").read_text()
    found = re.findall(r"prototype_with_update:([\d.]+)", text)
    traces = list((root / f"{version}-10").rglob("full_attribution.json"))
    if len(found) != 1 or len(traces) != 1 or "PASS" not in text:
        raise SystemExit(f"{version}: checked result or full trace missing under {root}")
    rows = json.loads(traces[0].read_text())["samples"]
    errors = [row["accounting"] for row in rows if "error" in row["accounting"]]
    if errors:
        raise SystemExit(f"{version}: trace attribution unavailable: {errors}; inspect {traces[0]}")
    rows.sort(key=lambda row: row["us"])
    row = rows[len(rows) // 2]
    representatives[version] = row
    acct = row["accounting"]
    stages = acct["stages_us"]
    pieces = [f"{key}={value:.3f}" for key, value in stages.items()]
    pieces.extend((f"gap={acct['inter_kernel_gap_us']:.3f}",
                   f"boundary={acct['boundary_residual_us']:.3f}"))
    lines.append(f"{version}: graph_p50={float(found[0]):.3f} trace_event={row['us']:.3f} "
                 + " ".join(pieces) + f" prefetch={row['prefetch']} final={row['final']}")
a, b = (representatives[k] for k in ("main", "pr11"))
x, y = a["accounting"], b["accounting"]
deltas = {key: y["stages_us"][key] - value for key, value in x["stages_us"].items()}
deltas["gap"] = y["inter_kernel_gap_us"] - x["inter_kernel_gap_us"]
deltas["boundary"] = y["boundary_residual_us"] - x["boundary_residual_us"]
lines.append("trace delta (PR - main): " + " ".join(f"{k}={v:+.3f}" for k, v in deltas.items())
             + f" sum={sum(deltas.values()):+.3f} event={b['us'] - a['us']:+.3f}")
lines.append("Representative traces are different replays; delta is not a paired causal estimate.")
result = "\n".join(lines) + "\n"
(root / "summary.log").write_text(result)
print(result, end="")
print(f"Summary: {root / 'summary.log'}")
PY
