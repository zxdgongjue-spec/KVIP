#!/usr/bin/env bash
# B20 candidate only. The serial and mainline timings are always read from archives.
set -euo pipefail
repo=${1:?supply the KVIP repository}
reader=${2:?supply the published host20 reader revision}
native=6d7f257b3d664a8b4991cfd3ebd12da4f2038d04
physical=${3:-${PEER_PHYSICAL_NPU:-0}}
case "$physical" in
    0|5) ;;
    *) echo "Expected B20 physical NPU 0 or 5, got $physical" >&2; exit 2 ;;
esac
export PEER_PHYSICAL_NPU="$physical" ASCEND_RT_VISIBLE_DEVICES="$physical" PEER_LOCAL_ONLY=1
export PEER_G2_WORK=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
git -C "$repo" cat-file -e "${native}^{commit}"
git -C "$repo" cat-file -e "${reader}^{commit}"
mkdir -p "$PEER_G2_WORK"
script=$(mktemp "$PEER_G2_WORK/host20-npu${physical}-paced-launcher.XXXXXX")
trap 'rm -f "$script"' EXIT
git -C "$repo" show "${reader}:tests/operators/lifetch/main_five_npu4.sh" > "$script"
bash "$script" "$repo" "$reader" "$native"
