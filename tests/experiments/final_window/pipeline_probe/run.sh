#!/usr/bin/env bash
set -euo pipefail
# Run in a configured CANN simulator environment. All generated code stays in OUT.
kind=${1:?group|split|hoisted|output|index}
out=${2:?new output directory}
kvaf_include=${3:?KVAF include directory}
here=$(cd -- "$(dirname -- "$0")" && pwd)
mode=1
case "$kind" in
 group) python3 "$here/prepare.py" "$out" ;;
 split) python3 "$here/prepare.py" "$out"; mode=3 ;;
 hoisted) python3 "$here/prepare_hoisted.py" "$out"; mode=3 ;;
 output|index) python3 "$here/prepare_module.py" "$kind" "$out" ;;
 *) exit 2 ;;
esac
ln -s "$kvaf_include" "$out/kvaf-include"
cmake -S "$out" -B "$out/build" > "$out/build.log" 2>&1
cmake --build "$out/build" -j4 >> "$out/build.log" 2>&1
msprof op simulator --application="$out/build/real_topk_sim $mode" --output="$out/profile" --soc-version=Ascend950PR_9599 > "$out/run.log" 2>&1
cat "$out/run.log"
python3 "$here/extract.py" "$out/profile" "$out/run.log" > "$out/stamps.json"
