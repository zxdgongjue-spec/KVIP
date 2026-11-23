#!/usr/bin/env bash
set -eo pipefail
if (( $# < 4 || $# > 5 )); then
    echo "Usage: run.sh BRIDGE_SO KVIP_SOURCE KVAF_INCLUDE NEW_OUTPUT_DIRECTORY [1|2]" >&2
    exit 2
fi
bridge=$(realpath "$1")
source_root=$(realpath "$2")
kvaf=$(realpath "$3")
out=$(realpath -m "$4")
replays=${5:-1}
[[ "$replays" == 1 || "$replays" == 2 ]]
model_root=$(cd "$(dirname "$0")" && pwd)
toolkit=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0}
source "$toolkit/bin/setenv.bash" >/dev/null
set -u
sim="$toolkit/tools/simulator/dav_3510/lib"
[[ -f "$bridge" && -f "$kvaf/batch_get_abi.h" && -f "$sim/libruntime_camodel.so" ]]
mkdir "$out"
mkdir "$out/model" "$out/sim-runtime"
ln -s "$sim/libruntime_camodel.so" "$out/sim-runtime/libruntime.so"
g++ -std=c++17 -O2 "$model_root/probe.cpp" \
    -I"$source_root/csrc/operators/lightning_indexer_with_fetch/include" -I"$kvaf" -I"$toolkit/include" \
    -L"$toolkit/lib64" -Wl,-rpath-link,"$toolkit/lib64" -Wl,--allow-shlib-undefined \
    -Wl,--no-as-needed -L"$sim" -lruntime_camodel -lnpu_drv -lascendcl -ldl -o "$out/probe"
export CAMODEL_CONFIG_PATH="$model_root" CAMODEL_LOG_PATH="$out/model" ASCEND_SLOG_PRINT_TO_STDOUT=0
export LD_LIBRARY_PATH="$out/sim-runtime:$sim:${LD_LIBRARY_PATH:-}"
cd "$out"
sha256sum "$bridge" "$sim/libruntime_camodel.so" "$model_root/config.json" "$model_root/config_stars.json" > artifacts.sha256
"$out/probe" "$bridge" "$out/scores.bin" "$replays" > "$out/output.log" 2>&1 || {
    tail -25 "$out/output.log" >&2
    exit 1
}
grep -E '^(SCORE_MATCH|ROW=|REPLAY=)|Total tick' "$out/output.log"
