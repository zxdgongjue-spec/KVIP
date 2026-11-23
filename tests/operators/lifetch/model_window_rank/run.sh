#!/usr/bin/env bash
# Run with the CANN 9.2 aarch64 toolkit and its dav_3510 simulator installed.
set -euo pipefail
if (( $# != 2 )); then
    echo "usage: bash run.sh /path/to/extracted/window-audit-inputs /path/to/new-results" >&2
    exit 2
fi
src=$(cd "$(dirname "$0")" && pwd)
audit=$(cd "$1" && pwd)
mkdir -p "$2"
dest=$(cd "$2" && pwd)
cann=${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann-9.2.0}
set +u
source "$cann/set_env.sh"
set -u
export LD_LIBRARY_PATH="$cann/tools/simulator/dav_3510/lib:${LD_LIBRARY_PATH:-}"
export CAMODEL_CONFIG_PATH="$src"
python3 "$src/fixtures.py" "$audit" "$dest/samples" | tee "$dest/fixtures.jsonl"
cmake -S "$src" -B "$dest/build" -DCMAKE_BUILD_TYPE=Release -DASCEND_CANN_PACKAGE_PATH="$cann" -DBUILD_HANDOFF=ON
cmake --build "$dest/build" -j2
run_model() {
    local dir=$1 logfile=$2
    shift 2
    mkdir -p "$dir"
    (
        cd "$dir"
        export CAMODEL_LOG_PATH="$PWD"
        "$@" > "$logfile" 2>&1
    )
}
for name in g10 g4 g17 g7 g2 edges; do
    run_model "$dest/run-$name" "$dest/model-$name.log" "$dest/build/probe" "$dest/samples/$name"
done
python3 "$src/summarize.py" "$dest" "$dest/samples" | tee "$dest/results.jsonl"
python3 "$src/fixtures.py" "$audit" "$dest/all" --all-model > "$dest/all-fixtures.jsonl"
python3 "$src/handoff_fixtures.py" "$dest/all" "$dest/handoff"
for name in g10 g4 g17 g7 g2 edges; do
    run_model "$dest/run-handoff-$name" "$dest/handoff/model-$name.log" "$dest/build/handoff" "$dest/handoff/$name.bin"
done
python3 "$src/summarize.py" "$dest/handoff" "$dest/handoff" --handoff | tee "$dest/handoff/results.jsonl"
