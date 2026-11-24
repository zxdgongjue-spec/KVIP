#!/usr/bin/env bash
set -euo pipefail

: "${KVIP_MODEL_OUTPUT:?output directory}"
: "${KVAF_INCLUDE:?KVAF device headers}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cann=${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann}
sim="$cann/tools/simulator/Ascend950PR_9599"
batch=${1:-2}
history=${2:-4096}
resident=${3:-2048}
epochs=${4:-2}
export PATH="$cann/bin:$PATH"
source "$cann/set_env.sh"
mkdir -p "$KVIP_MODEL_OUTPUT"
python3 - "$sim" "$batch" <<'PY'
import json
from pathlib import Path
import sys

sim, batch = Path(sys.argv[1]), int(sys.argv[2])
for name in ('lib/config_stars.json', 'lib/config.json', 'camodel/camodel_v100.json'):
    path = sim / name
    data = json.loads(path.read_text())
    if 'stars' in name:
        data['model_top'].update(num_aic=2*batch, num_aiv=4*batch)
        data.setdefault('pem', {}).update(parsim=1, parsim_thd_limit=min(8, 2*batch))
    elif '/camodel_' in name:
        data.setdefault('sysconf', {}).update(esl_top_thread_cnt=min(12, 4*batch))
    else:
        data['LOG']['core_enable_mask'] = ['0x0']
        data['L2CACHE']['cache_enable'] = 0
    path.write_text(json.dumps(data, indent=2))
PY

failure=0
read -r -a arms <<< "${KVIP_MODEL_ARMS:-base ub batch}"
for arm in "${arms[@]}"; do
    case "$arm" in base|ub|batch) ;; *) echo "Invalid arm: $arm" >&2; exit 2;; esac
    out="$KVIP_MODEL_OUTPUT/b${batch}-h${history}/$arm"
    mkdir -p "$out/model"
    ub=0
    chunk=0
    [[ $arm != ub ]] || ub=1
    [[ $arm != batch ]] || chunk=64
    echo "CANN build $arm"
    cmake -S "$here" -B "$KVIP_MODEL_OUTPUT/build-$arm" -DASCEND_CANN_PACKAGE_PATH="$cann" \
        -DKVAF_INCLUDE="$KVAF_INCLUDE" -DUB_CANDIDATES="$ub" -DPOST_CHUNK="$chunk" \
        > "$out/build.log" 2>&1
    cmake --build "$KVIP_MODEL_OUTPUT/build-$arm" --parallel 2 >> "$out/build.log" 2>&1
    status=0
    (
        cd "$out"
        ln -sfn /dev/null instr.bin
        export LD_LIBRARY_PATH="$sim/camodel:$sim/lib:$cann/lib64:${LD_LIBRARY_PATH:-}"
        export LD_PRELOAD="$sim/lib/libruntime_camodel.so"
        export CAMODEL_LOG_PATH="$out/model"
        echo "CANN run $arm"
        timeout --kill-after=5s "${KVIP_MODEL_TIMEOUT:-210}" \
            "$KVIP_MODEL_OUTPUT/build-$arm/service_model" "$batch" "$history" "$resident" "$arm" "$epochs" > model.log 2>&1
    ) || status=$?
    printf '%s\n' "$status" > "$out/exit_code.txt"
    grep '^SERVICE_' "$out/model.log" || true
    if (( status )); then
        tail -5 "$out/model.log"
        failure=1
    fi
done
exit "$failure"
