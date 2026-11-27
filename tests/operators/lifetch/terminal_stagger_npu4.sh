#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run on the current host's physical NPU4; archive may be omitted for automatic selection.
set -euo pipefail
repo=$(cd "${1:?expected KVIP checkout}" && pwd -P)
reader=$(git -C "$repo" rev-parse --verify "${2:?expected controller revision}^{commit}")
base=d0fc5963881bd38fa97eff0f6ef40c757a6e8713
work=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
sources="$work/fused-scan-sources"
mkdir -p "$sources"
stage() (
    local sha=$1 dir="$sources/stagger-$1"
    exec 8>"$dir.lock"; flock 8
    if [[ -d $dir && $(git -C "$dir" rev-parse HEAD 2>/dev/null) == "$sha" && -z $(git -C "$dir" status --porcelain -uno) ]]; then
        echo "$dir"; return
    fi
    if [[ -e $dir ]]; then dir=$(mktemp -d "$dir.XXXXXX"); fi
    git -C "$repo" worktree add -q --detach "$dir" "$sha"
    echo "$dir"
)
controller=$(stage "$reader")
timed=$(stage "$base")
entry="$controller/tests/operators/lifetch"
export PEER_G2_WORK="$work" PEER_PHYSICAL_NPU=4 ASCEND_RT_VISIBLE_DEVICES=4 PEER_LOCAL_ONLY=1
export PYTHONDONTWRITEBYTECODE=1 PEER_TIMED_SOURCE="$timed"
export PEER_FOUR_WINDOW_RANKS=1 PEER_LARGE_COLD_WINDOWS=3 PEER_FOUR_WINDOW_CARRY=0
export FIXED_Q4_GROUPS="${PEER_STAGGER_GROUPS:-7 2}" FIXED_Q4_BATCHES="${PEER_STAGGER_BATCHES:-32}"
unset PEER_RANK_DIAG_SOURCE PEER_BS32_LI PEER_PACKED_RESUME PEER_PACKED_REFERENCE KVAF_CPU_UB_DEVICE
unset PEER_FOUR_RANK_AUDIT PEER_FOUR_RANK_FOLLOWUP PEER_AUDIT_REFERENCE PEER_AUDIT_ORIGIN
find_ref() {
    python3 -B - "$work/peer-baseline" "$base" "$FIXED_Q4_GROUPS" "$FIXED_Q4_BATCHES" "${1:-}" <<'PY'
from pathlib import Path
import sys, os
area, base, groups, batches, explicit = sys.argv[1:]
cells = {(g, b) for g in groups.split() for b in batches.split()} | {("7", "32")}
candidates = [Path(explicit)] if explicit else sorted(Path(area).glob("fixed-q4-four-ranks-one-block.*"), key=lambda p:p.stat().st_mtime_ns, reverse=True)
def text(p):
    return p.read_text().strip() if p.is_file() else ""
for p in candidates:
    machine, boot = text(Path("/etc/machine-id")), text(Path("/proc/sys/kernel/random/boot_id"))
    host_ok = text(p/"machine_id.txt") == machine if machine and text(p/"machine_id.txt") else bool(boot and text(p/"boot_id.txt") == boot)
    if (host_ok and text(p/"host.txt") == os.uname().nodename and text(p/"physical_npu.txt") == "4"
            and text(p/"native.txt") == base and all(all((p/f"g{g}/bs{b}/block1/fused"/f).is_file()
            for f in ("thp.json","placement_verified","packed_result.json")) for g, b in cells)):
        print(p.resolve()); break
PY
}
ref=$(find_ref "${3:-}")
if [[ -z $ref ]]; then
    if [[ -n ${3:-} ]]; then echo "Reference does not contain the selected same-host NPU4 cells: $3" >&2; exit 2; fi
    ref="$work/peer-baseline"
    export PEER_STAGGER_LOCAL=1
    echo "No same-card timing reference; capturing a single untimed original GET journal for each selected cell."
else
    export PEER_STAGGER_LOCAL=0
fi
export PEER_AUDIT_ORIGIN="$ref" PEER_AUDIT_REFERENCE="$ref"
echo "REFERENCE=$ref G=$FIXED_Q4_GROUPS BS=$FIXED_Q4_BATCHES NPU=4"
PEER_FOUR_RANK_AUDIT=stagger-pair bash "$entry/large_batch_npu4.sh" --four-ranks-candidate-only
