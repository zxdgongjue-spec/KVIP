#!/usr/bin/env bash
# Host-only candidate preparation and PC evidence; no NPU launch or CPU service.
set -Eeuo pipefail
umask 077
[[ $# == 1 ]] || { echo "Usage: $0 V1_EPOCH_AUDIT_DIR"; exit 2; }
AUDIT=$(realpath "$1")
DIR=$(cd "$(dirname "$0")" && pwd)
OP=/tmp/KVIP-integrated-79c6a225
BUILD=/tmp/KVAF-integrated-87e261ae/build-npu/lifetch/20ed0253a932fbdbdc73
CANN=/usr/local/Ascend/cann-9.2.1
PY=/usr/local/python3.12.13/bin/python
test "$(cat "$AUDIT/test_commit.txt")" = 6d30cc0e66aee1b0dc7fcf3742e6c02bb40586d3
grep -Fxq 'V1_EPOCH_REACHABILITY_REPRODUCED missing=4,5,6 missing_event_records=6' "$AUDIT/run.log"
OUT=$(mktemp -d "$(dirname "$AUDIT")/v1-f1-host.XXXXXX")
exec > >(tee "$OUT/run.log") 2>&1
echo "OUTPUT_DIR=$OUT"
git -C "$DIR" rev-parse HEAD > "$OUT/test_commit.txt"
"$PY" -u "$DIR/model_nextgen_v1_closure.py" "$OP"
"$PY" -u "$DIR/prepare_nextgen_v1_f1_patch.py" "$OP" "$OUT"
OBJDUMP="$CANN/tools/bisheng_compiler/bin/llvm-objdump"
READELF="$CANN/tools/hcc/bin/aarch64-target-linux-gnu-readelf"
# The previous AIV object device_aiv.o contains AIV-only kernels, not necessarily MIX.
for spec in aic:"$BUILD/v9_kernel_aic_device_dir/device.o" mix_aiv:"$BUILD/v9_kernel_aiv_device_dir/device.o" merged:"$BUILD/v9_kernel_merge_obj_dir/device.o"; do
    kind=${spec%%:*}; obj=${spec#*:}
    test -s "$obj"
    sha256sum "$obj" | tee -a "$OUT/objects.sha256"
    "$READELF" -Ws "$obj" > "$OUT/$kind-symbols.txt"
    "$OBJDUMP" -d --line-numbers --demangle "$obj" > "$OUT/$kind-disassembly.txt" 2>"$OUT/$kind-disassembly.stderr" || echo "PC_DISASSEMBLY_UNSUPPORTED $kind"
    grep -E 'FUNC.*v9_li_kernel[^ ]*mix_(aic|aiv)([[:space:]]|$)' "$OUT/$kind-symbols.txt" || true
done
"$PY" - "$OUT" <<'PY'
import re,sys
from pathlib import Path
out=Path(sys.argv[1])
for kind,offs in (("aic",[0x39fc,0x3c4c]),("mix_aiv",[0x850c,0x8a28,0x91bc,0xd650,0xdab0])):
    syms=(out/(kind+"-symbols.txt")).read_text().splitlines()
    suffix="aic" if kind=="aic" else "aiv"
    matches=[]
    for line in syms:
        f=line.split()
        if len(f)>=8 and f[3]=="FUNC" and f[7]=="v9_li_kernel_0_mix_"+suffix:
            matches.append((int(f[1],16),int(f[2]),f[7]))
    if len(matches)!=1:
        print("PC_SYMBOL_UNRESOLVED",kind,matches);continue
    base,size,name=matches[0]
    lines=(out/(kind+"-disassembly.txt")).read_text().splitlines()
    for off in offs:
        address=base+off
        print("PC_CANDIDATE",kind,name,hex(off),hex(address),"in_symbol",off<size)
        if off>=size:continue
        hits=[i for i,s in enumerate(lines) if re.match(r"^\s*"+format(address,"x")+r":",s,re.I)]
        if len(hits)!=1:
            print("PC_INSTRUCTION_UNRESOLVED",len(hits));continue
        i=hits[0]
        print("\n".join(lines[max(0,i-12):i+13]))
PY
echo "V1_F1_HOST_PREPARED"
echo "OUTPUT_DIR=$OUT"
