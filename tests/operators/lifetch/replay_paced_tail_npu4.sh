#!/usr/bin/env bash
# Run the independent terminal candidate and reuse the verified B24 NPU4 archives.
set -euo pipefail
repo=${1:?supply the KVIP repository}
native=${2:?supply the candidate commit}
reader=852fd48982ddb70f55b4d9d48f833151b8920004
work=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
git -C "$repo" cat-file -e "${native}^{commit}"
git -C "$repo" cat-file -e "${reader}^{commit}"
tmp=$(mktemp -d "$work/host24-npu4-paced-tail.XXXXXX")
echo "LOG=$tmp/replay.log"
git -C "$repo" show "${reader}:tests/operators/lifetch/main_five_npu4.sh" > "$tmp/run-five.sh"
export PEER_G2_WORK="$work" PEER_PHYSICAL_NPU=4 ASCEND_RT_VISIBLE_DEVICES=4 PEER_LOCAL_ONLY=1
rc=0
bash "$tmp/run-five.sh" "$repo" "$reader" "$native" 2>&1 |
    tee "$tmp/replay.log" |
    awk '
        /^RESULT_ROOT=/ || /^REPLAY_/ || /^FAILED_ARM=/ || /^CASE_FAILED / { print; fflush(); next }
        /^CASE_BEGIN / { print $1, $2, "paced-tail"; fflush(); next }
        $2=="CHECK" || $2=="vs_mainline_archive" || $2=="vs_serial2_archive" { print; fflush(); next }
        $2=="LI_candidate" { $2="paced-tail"; print; fflush() }
    ' || rc=$?
if ((rc)); then tail -n 32 "$tmp/replay.log" >&2; exit "$rc"; fi
root=$(sed -n 's/^REPLAY_COMPLETE RESULT_ROOT=//p' "$tmp/replay.log")
test -d "$root"
python=/usr/local/python3.12.13/bin/python3
if [[ ! -x "$python" ]]; then python=python3; fi
"$python" - "$root" "$native" <<'PY'
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
reference = Path((root / "reference.txt").read_text().strip()).resolve()
serial_mean, serial_p50 = 589.598798752, 575.597625
other_mean_ok, other_p50_ok = True, True
for g in (10, 4, 17, 7, 2):
    row = json.loads((root / f"g{g}" / "comparison.json").read_text())
    if (row["production_revision"] != sys.argv[2] or row["group"] != g
            or Path(row["archive"]).resolve() != reference):
        raise SystemExit(f"G{g} candidate or archive mismatch")
    new = row["candidate"]["graph"]
    old = row["mainline"]["graph"]
    if not all(math.isfinite(v) and v > 0 for v in (*new, *old)):
        raise SystemExit(f"G{g} invalid graph timing")
    if g == 7:
        gain_mean = 100*(1-new[0]/serial_mean)
        gain_p50 = 100*(1-new[1]/serial_p50)
        g7_mean_ok, g7_p50_ok = new[0] < .8*serial_mean, new[1] < .8*serial_p50
        print(f"G7 vs_strong_serial graph_saved_mean/P50_us={serial_mean-new[0]:+.2f}/{serial_p50-new[1]:+.2f} "
              f"gain_mean/P50_pct={gain_mean:+.2f}/{gain_p50:+.2f}")
    else:
        other_mean_ok &= new[0] <= old[0]
        other_p50_ok &= new[1] <= old[1]
def verdict(value):
    return "PASS" if value else "NOT_MET"
print(f"MEASURED_GATE G7_over_20pct_mean/P50={verdict(g7_mean_ok)}/{verdict(g7_p50_ok)} "
      f"other_four_no_regression_mean/P50={verdict(other_mean_ok)}/{verdict(other_p50_ok)}")
PY
