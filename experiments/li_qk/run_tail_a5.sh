#!/usr/bin/env bash
# Physical A5 only; read the frozen bundle from the existing shared mount.
set -Eeuo pipefail
LI_TAIL_SHARE=/home/w00568585/li_qk_shared
LI_TAIL_OUTPUT_ROOT=/home/w00568585
LI_TAIL_CANN=${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}
LI_TAIL_KVAF=/home/w00568585/wt-mr14/third_party/KVAF/access_operator/demo/aiv_batch_get/include
LI_TAIL_SOC=Ascend950PR_9599
LI_TAIL_PYTHON=python3
LI_TAIL_DEVICE=5
LI_TAIL_BATCH=1
LI_TAIL_HISTORY=131072
LI_TAIL_RESIDENT=16384
LI_TAIL_PARTS=4
LI_TAIL_RPARTS=1
LI_TAIL_PROFILE=0
LI_TAIL_VERIFY_ONLY=0
LI_TAIL_CONTROLS=(C0 C1 C2 C3)
die(){ printf 'ERROR: %s\n' "$*" >&2; exit 1; }
usage(){ cat <<'USAGE'
Usage: bash run_tail_a5.sh [options]
Run in the original A5 container after the host updates li_qk_shared.
  --batch 1|16          Requests (default 1)
  --history N           History tokens (default 131072)
  --resident N          Resident tokens (default 16384)
  --parts N             Cold/Final history parts (default 4)
  --resident-parts N    Independent resident TopK parts (default 1)
  --profile             Collect separate msprof traces for all four controls
  --controls C0,C1      Controls to compare (default C0,C1,C2,C3; C0 required)
  --verify-only         Verify the shared release without building/running kernels
  --device N            Physical device (default 5)
  --share-root PATH     Published shared package root
  --output-root PATH    Parent of fresh result directory
  --cann-root PATH      Matching CANN installation
  --kvaf-include PATH   Compatible KVAF headers; no Fetch is launched
  --soc NAME            Actual SoC (default Ascend950PR_9599)
  --python PATH         Python from the original environment
USAGE
}
while (($#)); do
 case "$1" in
  -h|--help) usage; exit 0 ;;
  --profile) LI_TAIL_PROFILE=1; shift ;;
  --verify-only) LI_TAIL_VERIFY_ONLY=1; shift ;;
  --controls)
   (($#>=2)) || die 'Missing controls'
   IFS=, read -r -a LI_TAIL_CONTROLS <<< "$2"
   [[ ${#LI_TAIL_CONTROLS[@]} -ge 2 && ",$2," == *,C0,* ]] || die 'Paired comparison needs C0 and another control'
   for LI_TAIL_CONTROL in "${LI_TAIL_CONTROLS[@]}"; do
    [[ $LI_TAIL_CONTROL =~ ^C[0-3]$ ]] || die "Unknown control: $LI_TAIL_CONTROL"
   done
   shift 2 ;;
  --batch|--history|--resident|--parts|--resident-parts|--device|--share-root|--output-root|--cann-root|--kvaf-include|--soc|--python)
   (($#>=2)) || die "Missing value: $1"
   case "$1" in
    --batch) LI_TAIL_BATCH=$2 ;; --history) LI_TAIL_HISTORY=$2 ;; --resident) LI_TAIL_RESIDENT=$2 ;;
    --parts) LI_TAIL_PARTS=$2 ;; --resident-parts) LI_TAIL_RPARTS=$2 ;; --device) LI_TAIL_DEVICE=$2 ;;
    --share-root) LI_TAIL_SHARE=$2 ;; --output-root) LI_TAIL_OUTPUT_ROOT=$2 ;; --cann-root) LI_TAIL_CANN=$2 ;;
    --kvaf-include) LI_TAIL_KVAF=$2 ;; --soc) LI_TAIL_SOC=$2 ;; --python) LI_TAIL_PYTHON=$2 ;;
   esac
   shift 2 ;;
  *) die "Unknown argument: $1" ;;
 esac
done
for LI_TAIL_NUM in "$LI_TAIL_BATCH" "$LI_TAIL_HISTORY" "$LI_TAIL_RESIDENT" "$LI_TAIL_PARTS" "$LI_TAIL_RPARTS" "$LI_TAIL_DEVICE"; do
 [[ $LI_TAIL_NUM =~ ^[0-9]+$ && ${#LI_TAIL_NUM} -le 7 ]] || die 'Invalid numeric argument'
done
[[ ${LD_PRELOAD:-} != *camodel* && ${LD_PRELOAD:-} != *cmodel* ]] || die 'Physical A5 required; Model preload rejected'
command -v git >/dev/null || die 'Command unavailable: git'
LI_TAIL_RELEASE=$(cd "$LI_TAIL_SHARE/current" && pwd -P) || die 'Update the shared package on the host first'
[[ $(git hash-object --no-filters "$LI_TAIL_RELEASE/li_qk_mr14_offline.bundle") == "$(cat "$LI_TAIL_RELEASE/BUNDLE_BLOB")" ]] || die 'Published bundle hash mismatch'
LI_TAIL_PUBLISHED=$(git bundle list-heads "$LI_TAIL_RELEASE/li_qk_mr14_offline.bundle" refs/remotes/origin/codex/split-core-unified-comparison)
[[ ${LI_TAIL_PUBLISHED%% *} == "$(cat "$LI_TAIL_RELEASE/SOURCE_COMMIT")" ]] || die 'Published bundle commit mismatch'
if ((LI_TAIL_VERIFY_ONLY)); then
 printf 'Shared post-score source verified: %s\nNo kernels executed.\n' "${LI_TAIL_PUBLISHED%% *}"
 exit 0
fi
for LI_TAIL_CMD in npu-smi cmake "$LI_TAIL_PYTHON"; do command -v "$LI_TAIL_CMD" >/dev/null || die "Command unavailable: $LI_TAIL_CMD"; done
mkdir -p "$LI_TAIL_OUTPUT_ROOT"
LI_TAIL_OUT=$(mktemp -d "$LI_TAIL_OUTPUT_ROOT/li_tail_$(date +%Y%m%d_%H%M%S)_XXXXXX")
finish(){ local li_tail_status=$?; trap - EXIT; tar -czf "$LI_TAIL_OUT.tar.gz" -C "$(dirname "$LI_TAIL_OUT")" "$(basename "$LI_TAIL_OUT")" || true; printf 'Results: %s\nArchive: %s.tar.gz\n' "$LI_TAIL_OUT" "$LI_TAIL_OUT"; exit "$li_tail_status"; }
trap finish EXIT
git init --bare "$LI_TAIL_OUT/repo.git" > "$LI_TAIL_OUT/git.log" 2>&1
git --git-dir="$LI_TAIL_OUT/repo.git" fetch "$LI_TAIL_RELEASE/li_qk_mr14_offline.bundle" refs/remotes/origin/codex/split-core-unified-comparison >> "$LI_TAIL_OUT/git.log" 2>&1
LI_TAIL_COMMIT=$(git --git-dir="$LI_TAIL_OUT/repo.git" rev-parse FETCH_HEAD)
[[ $LI_TAIL_COMMIT == "$(cat "$LI_TAIL_RELEASE/SOURCE_COMMIT")" ]] || die 'Published commit mismatch'
git --git-dir="$LI_TAIL_OUT/repo.git" worktree add --detach "$LI_TAIL_OUT/source" "$LI_TAIL_COMMIT" >> "$LI_TAIL_OUT/git.log" 2>&1
cd "$LI_TAIL_OUT/source"
[[ -f experiments/li_qk/tail.py ]] || die 'Published revision lacks post-score controls; update the host to the latest MR14'
printf 'commit=%s\nbatch=%s\nhistory=%s\nresident=%s\nparts=%s\nresident_parts=%s\n' "$LI_TAIL_COMMIT" "$LI_TAIL_BATCH" "$LI_TAIL_HISTORY" "$LI_TAIL_RESIDENT" "$LI_TAIL_PARTS" "$LI_TAIL_RPARTS" > "$LI_TAIL_OUT/config.txt"
npu-smi info > "$LI_TAIL_OUT/npu-smi.log"
LI_TAIL_ARGS=(--batch "$LI_TAIL_BATCH" --history "$LI_TAIL_HISTORY" --resident "$LI_TAIL_RESIDENT" --parts "$LI_TAIL_PARTS" --resident-parts "$LI_TAIL_RPARTS" --device "$LI_TAIL_DEVICE" --controls "${LI_TAIL_CONTROLS[@]}")
"$LI_TAIL_PYTHON" experiments/li_qk/tail.py plan "${LI_TAIL_ARGS[@]}" > "$LI_TAIL_OUT/plan.json"
"$LI_TAIL_PYTHON" experiments/li_qk/test_tail.py > "$LI_TAIL_OUT/checks.log" 2>&1
"$LI_TAIL_PYTHON" experiments/li_qk/tail.py build --run-mode npu --cann-root "$LI_TAIL_CANN" --kvaf-include "$LI_TAIL_KVAF" --soc "$LI_TAIL_SOC" --output "$LI_TAIL_OUT/build" > "$LI_TAIL_OUT/build-console.log" 2>&1
"$LI_TAIL_PYTHON" experiments/li_qk/tail.py bench --build "$LI_TAIL_OUT/build" --output "$LI_TAIL_OUT/results" "${LI_TAIL_ARGS[@]}" > "$LI_TAIL_OUT/bench.log" 2>&1
if ((LI_TAIL_PROFILE)); then
 "$LI_TAIL_PYTHON" experiments/li_qk/tail.py profile --build "$LI_TAIL_OUT/build" --output "$LI_TAIL_OUT/msprof" "${LI_TAIL_ARGS[@]}" > "$LI_TAIL_OUT/profile.log" 2>&1
fi
cat "$LI_TAIL_OUT/results/summary.csv"
