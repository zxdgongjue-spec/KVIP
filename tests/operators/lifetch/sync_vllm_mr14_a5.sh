#!/usr/bin/env bash
# Container: install the host-published MR14 into vLLM, rebuild, then compare three chains.
set -Eeuo pipefail
LI_VSYNC_ROOT=$PWD
LI_VSYNC_RELEASE=/home/w00568585/li_qk_shared/current
LI_VSYNC_KVAF=
LI_VSYNC_PYTHON=python3
LI_VSYNC_CANN=/usr/local/Ascend/cann
LI_VSYNC_DEVICE=0
LI_VSYNC_SERVER=127.0.0.1:28885
LI_VSYNC_BATCHES=1,16
LI_VSYNC_MISSES=0,3,5,10,15,20
LI_VSYNC_PREFETCH=512
LI_VSYNC_ROUNDS=3
LI_VSYNC_WARMUP=4
LI_VSYNC_ITERATIONS=50
LI_VSYNC_JOBS=2
LI_VSYNC_ONLY=0
LI_VSYNC_NO_BENCH=0
LI_VSYNC_STAGE=
LI_VSYNC_BACKUP=
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
usage() {
    cat <<'USAGE'
Usage: bash sync_vllm_mr14_a5.sh [options]
Run inside the A5 container, from the vLLM-DSA root. Host must publish MR14 first.
  --vllm-root PATH       vLLM checkout (default current directory)
  --release-root PATH    Shared release containing bundle/hash/commit records
  --kvaf-root PATH       Existing URMA KVAF (default <vllm>/third_party/KVAF)
  --python PROGRAM      Serving Python, >=3.12 (default python3)
  --cann-root PATH       CANN root (default /usr/local/Ascend/cann)
  --device N            NPU device (default 0)
  --kvaf-server ADDRESS Existing CPU service (default 127.0.0.1:28885)
  --batches LIST        Default 1,16
  --miss-percent LIST   Default 0,3,5,10,15,20
  --prefetch LIST       Default 512; use 0,512 for both controls
  --rounds N --warmup N --iterations N   Defaults 3 / 4 / 50
  --jobs N              Build jobs (default 2)
  --sync-only           Replace source only; no build or NPU execution
  --no-bench            Replace source and build, without benchmarking
Preserves the previous KVIP directory, including edits/builds, in a printed backup.
Imports Git locally from the shared bundle; no container GitCode fetch or Docker CLI.
USAGE
}
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --sync-only) LI_VSYNC_ONLY=1; shift ;;
        --no-bench) LI_VSYNC_NO_BENCH=1; shift ;;
        --vllm-root|--release-root|--kvaf-root|--python|--cann-root|--device|--kvaf-server|--batches|--miss-percent|--prefetch|--rounds|--warmup|--iterations|--jobs)
            (($# >= 2)) && [[ -n $2 ]] || die "Missing value for $1"
            case "$1" in
                --vllm-root) LI_VSYNC_ROOT=$2 ;; --release-root) LI_VSYNC_RELEASE=$2 ;;
                --kvaf-root) LI_VSYNC_KVAF=$2 ;; --python) LI_VSYNC_PYTHON=$2 ;;
                --cann-root) LI_VSYNC_CANN=$2 ;; --device) LI_VSYNC_DEVICE=$2 ;;
                --kvaf-server) LI_VSYNC_SERVER=$2 ;; --batches) LI_VSYNC_BATCHES=$2 ;;
                --miss-percent) LI_VSYNC_MISSES=$2 ;; --prefetch) LI_VSYNC_PREFETCH=$2 ;;
                --rounds) LI_VSYNC_ROUNDS=$2 ;; --warmup) LI_VSYNC_WARMUP=$2 ;;
                --iterations) LI_VSYNC_ITERATIONS=$2 ;; --jobs) LI_VSYNC_JOBS=$2 ;;
            esac
            shift 2 ;;
        *) die "Unknown option: $1" ;;
    esac
done
LI_VSYNC_ROOT=$(cd "$LI_VSYNC_ROOT" && pwd -P)
[[ -d $LI_VSYNC_ROOT/vllm_ascend ]] || die 'Expected vLLM-DSA root containing vllm_ascend'
LI_VSYNC_RELEASE=$(cd "$LI_VSYNC_RELEASE" && pwd -P)
for LI_VSYNC_FILE in SOURCE_COMMIT BUNDLE_BLOB li_qk_mr14_offline.bundle; do
    [[ -s $LI_VSYNC_RELEASE/$LI_VSYNC_FILE ]] || die "Shared file missing: $LI_VSYNC_FILE; update on the host first"
done
LI_VSYNC_BUNDLE=$LI_VSYNC_RELEASE/li_qk_mr14_offline.bundle
[[ $(git hash-object --no-filters "$LI_VSYNC_BUNDLE") == "$(cat "$LI_VSYNC_RELEASE/BUNDLE_BLOB")" ]] || die 'Bundle hash mismatch'
LI_VSYNC_HEAD=$(cat "$LI_VSYNC_RELEASE/SOURCE_COMMIT")
LI_VSYNC_REF=refs/remotes/origin/codex/split-core-unified-comparison
[[ $(git bundle list-heads "$LI_VSYNC_BUNDLE" "$LI_VSYNC_REF") == "$LI_VSYNC_HEAD $LI_VSYNC_REF" ]] || die 'Bundle does not contain the published MR14 revision'
[[ -n $LI_VSYNC_KVAF ]] || LI_VSYNC_KVAF=$LI_VSYNC_ROOT/third_party/KVAF
[[ -d $LI_VSYNC_KVAF ]] || die "Existing KVAF checkout required: $LI_VSYNC_KVAF"
LI_VSYNC_KVAF=$(cd "$LI_VSYNC_KVAF" && pwd -P)
if ((!LI_VSYNC_ONLY)); then
    [[ -f $LI_VSYNC_CANN/set_env.sh ]] || die "CANN environment missing: $LI_VSYNC_CANN/set_env.sh"
    set +u
    source "$LI_VSYNC_CANN/set_env.sh"
    set -Eeuo pipefail
    "$LI_VSYNC_PYTHON" -c 'import sys; assert sys.version_info >= (3, 12), "Use serving Python >=3.12"; import torch, torch_npu'
fi
cleanup() {
    LI_VSYNC_CODE=$?
    if [[ -n $LI_VSYNC_STAGE && -d $LI_VSYNC_STAGE ]]; then rm -rf -- "$LI_VSYNC_STAGE"; fi
    if ((LI_VSYNC_CODE)); then
        printf '\nSync/build/experiment failed (exit %s). KVIP: %s/third_party/KVIP\nPrevious checkout: %s\n' "$LI_VSYNC_CODE" "$LI_VSYNC_ROOT" "${LI_VSYNC_BACKUP:-unchanged}" >&2
    fi
}
trap cleanup EXIT
mkdir -p "$LI_VSYNC_ROOT/third_party"
LI_VSYNC_STAGE=$(mktemp -d "$LI_VSYNC_ROOT/third_party/.KVIP-mr14.XXXXXX")
git init "$LI_VSYNC_STAGE/KVIP"
git -C "$LI_VSYNC_STAGE/KVIP" bundle verify "$LI_VSYNC_BUNDLE"
git -C "$LI_VSYNC_STAGE/KVIP" fetch "$LI_VSYNC_BUNDLE" "$LI_VSYNC_REF"
git -C "$LI_VSYNC_STAGE/KVIP" checkout --detach "$LI_VSYNC_HEAD"
git -C "$LI_VSYNC_STAGE/KVIP" remote add origin https://gitcode.com/CacheOS/KVIP.git
[[ -f $LI_VSYNC_STAGE/KVIP/tests/operators/lifetch/serial_chain_compare.py ]] || die 'Published revision lacks three-way comparison'
[[ -f $LI_VSYNC_STAGE/KVIP/tests/operators/lifetch/support/optimized_serial_li.py ]] || die 'Published revision lacks optimized serial LI'
if ((!LI_VSYNC_ONLY)); then
    LI_VSYNC_EXPECTED_KVAF=$("$LI_VSYNC_PYTHON" - "$LI_VSYNC_STAGE/KVIP/scripts/build.py" <<'PY'
import ast, sys
from pathlib import Path
tree = ast.parse(Path(sys.argv[1]).read_text())
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'REVISION' for t in node.targets):
        print(ast.literal_eval(node.value))
        break
else:
    raise RuntimeError('Published build.py does not declare its KVAF revision')
PY
)
    LI_VSYNC_ACTUAL_KVAF=$(git -C "$LI_VSYNC_KVAF" rev-parse HEAD)
    [[ $LI_VSYNC_ACTUAL_KVAF == "$LI_VSYNC_EXPECTED_KVAF" ]] || die "KVAF mismatch: $LI_VSYNC_KVAF @ $LI_VSYNC_ACTUAL_KVAF; MR14 requires $LI_VSYNC_EXPECTED_KVAF. Existing KVIP preserved. Pass --kvaf-root to a matching checkout, e.g. /home/w00568585/wt-mr14/third_party/KVAF after checking its HEAD."
fi
LI_VSYNC_TARGET=$LI_VSYNC_ROOT/third_party/KVIP
if [[ -e $LI_VSYNC_TARGET || -L $LI_VSYNC_TARGET ]]; then
    LI_VSYNC_BACKUP=$(mktemp -d "$LI_VSYNC_ROOT/third_party/KVIP-backup.mr14.XXXXXX")/KVIP
    mv "$LI_VSYNC_TARGET" "$LI_VSYNC_BACKUP"
    printf 'Previous checkout preserved: %s\n' "$LI_VSYNC_BACKUP"
fi
if ! mv "$LI_VSYNC_STAGE/KVIP" "$LI_VSYNC_TARGET"; then
    if [[ -n $LI_VSYNC_BACKUP ]]; then mv "$LI_VSYNC_BACKUP" "$LI_VSYNC_TARGET"; fi
    die 'Could not install fresh checkout; previous checkout restored'
fi
printf 'Installed MR14: %s @ %s\n' "$LI_VSYNC_TARGET" "$(git -C "$LI_VSYNC_TARGET" rev-parse HEAD)"
if ((LI_VSYNC_ONLY)); then
    printf 'Source synchronized. Build before benchmarking; previous binaries are in the backup.\n'
    exit 0
fi
cd "$LI_VSYNC_TARGET"
mkdir -p build
"$LI_VSYNC_PYTHON" scripts/build.py --transport urma --kvaf-root "$LI_VSYNC_KVAF" --cann "$LI_VSYNC_CANN" --jobs "$LI_VSYNC_JOBS" 2>&1 | tee build/vllm-sync-build.log
source scripts/env.sh
"$LI_VSYNC_PYTHON" - "$LI_VSYNC_TARGET" <<'PY'
import json, sys
from pathlib import Path
import kvip
root = Path(sys.argv[1]).resolve()
assert Path(kvip.__file__).resolve().is_relative_to(root), kvip.__file__
print('KVIP import:', kvip.__file__)
print('Built dependencies:', json.loads((root / 'build/dependencies.json').read_text()))
PY
"$LI_VSYNC_PYTHON" tests/operators/lifetch/test_lifetch.py bench --help > build/vllm-sync-bench-help.log
if ((LI_VSYNC_NO_BENCH)); then exit 0; fi
LI_CHAIN_PYTHON=$LI_VSYNC_PYTHON bash tests/operators/lifetch/run_serial_vs_fused_a5.sh \
    --python "$LI_VSYNC_PYTHON" --device "$LI_VSYNC_DEVICE" --kvaf-root "$LI_VSYNC_KVAF" --kvaf-server "$LI_VSYNC_SERVER" \
    --batches "$LI_VSYNC_BATCHES" --miss-percent "$LI_VSYNC_MISSES" --prefetch "$LI_VSYNC_PREFETCH" \
    --intermediate-fetches 8 --rounds "$LI_VSYNC_ROUNDS" --warmup "$LI_VSYNC_WARMUP" --iterations "$LI_VSYNC_ITERATIONS"
