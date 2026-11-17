#!/usr/bin/env bash
# Use a dedicated CANN950 Model environment; no physical-A5 timing is produced.
set -Eeuo pipefail
if (($# != 4)); then
    echo 'Usage: bash run_granularity_model.sh BUILD NEW_OUTPUT MODEL_LIB LLVM_OBJDUMP' >&2
    exit 2
fi
LI_GRAIN_BUILD=$1
LI_GRAIN_OUT=$2
LI_GRAIN_LIB=$3
LI_GRAIN_OBJDUMP=$4
LI_GRAIN_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
[[ ! -e $LI_GRAIN_OUT ]] || { echo 'Fresh output directory required' >&2; exit 1; }
mkdir -p "$LI_GRAIN_OUT"
LI_GRAIN_FOLDERS=()
capture_grain() {
    local li_h=$1 li_g=$2 li_name=$3
    shift 3
    local li_dest=$LI_GRAIN_OUT/$li_name
    python3 "$LI_GRAIN_HERE/model.py" run --build "$LI_GRAIN_BUILD" --output "$li_dest" \
        --history "$li_h" --work-grain "$li_g" --case T2_balanced --stage score \
        --core-mask 0xffffffff --timeout 600 --model-lib "$LI_GRAIN_LIB" "$@"
    python3 "$LI_GRAIN_HERE/map_model_pcs.py" "$li_dest/T2_balanced_score" \
        --build "$LI_GRAIN_BUILD" --objdump "$LI_GRAIN_OBJDUMP" \
        --output "$li_dest/T2_balanced_score/verified-kernel-instructions"
    LI_GRAIN_FOLDERS+=("$li_dest")
}
for li_h in 256 512 1024 2048; do
    capture_grain "$li_h" 128 "short-h${li_h}-g128"
done
capture_grain 8192 2048 fixed-h8192-g2048
for li_g in 512 128; do
    capture_grain 8192 "$li_g" "fixed-h8192-g${li_g}" --fixture "$LI_GRAIN_OUT/fixed-h8192-g2048/fixture"
done
python3 "$LI_GRAIN_HERE/grain_summary.py" "${LI_GRAIN_FOLDERS[@]}" --output "$LI_GRAIN_OUT/summary"
