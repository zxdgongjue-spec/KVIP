#!/usr/bin/env bash
# V1 timeout evidence collection only. No NPU context, no service/device mutation.
set -Eeuo pipefail
umask 077
[[ $# == 1 ]] || { echo "Usage: $0 V1_FAILED_RUN"; exit 2; }
ROOT=$(realpath "$1")
[[ $(cat "$ROOT/test_commit.txt") == 5af188b6a8377e8cd2fa8b45fb8b86072c05074d ]]
grep -Fq 'V1_FAILED rc=1' "$ROOT/run.log"
mapfile -t CASES < <(find "$ROOT/basic_bs4" -type d -name case0-b4-prototype-graph)
[[ ${#CASES[@]} == 1 ]]
CASE="${CASES[0]}"
WORKER=$(dirname "$CASE")
OUT=$(mktemp -d "$(dirname "$ROOT")/v1-timeout-evidence.XXXXXX")
exec > >(tee "$OUT/diagnostic.txt") 2>&1
echo "OUTPUT_DIR=$OUT"
echo "FAILED_RUN=$ROOT"
echo "CASE=$CASE"
echo '===== 1. RESOURCE SNAPSHOT (READ ONLY) ====='
npu-smi info | tee "$OUT/npu-smi.txt"
PID=$(cat "$ROOT/service.pid")
echo "RECORDED_SERVICE_PID=$PID"
if [[ -d "/proc/$PID" ]]; then
    ps -p "$PID" -o pid,ppid,user,stat,etime,args
    readlink "/proc/$PID/exe" || true
    printf 'SERVICE_CMDLINE='
    tr '\0' ' ' < "/proc/$PID/cmdline"; echo
else
    echo 'RECORDED_SERVICE_NOT_VISIBLE'
fi
ps -eo pid,ppid,stat,args | grep -E '[k]vaf_cpu_service|[t]est_lifetch.py|[l]ifetch.*worker' || true
if command -v ss >/dev/null; then
    ss -ltnp | grep -E ':(28999|29009)[[:space:]]' || true
fi
echo '===== 2. EXISTING LOG INVENTORY ====='
find "$WORKER" -maxdepth 3 -type f -printf '%s %p\n' | sort -n
echo '===== 3. FIRST ERROR / TASK / KERNEL / STREAM ====='
for f in "$WORKER/worker.log" "$WORKER/summary.log" "$CASE/native.log" "$CASE/error.log" "$ROOT/cpu_service.log"; do
    if [[ -f "$f" ]]; then
        echo "===== $f ====="
        grep -inE 'V1_|BENCH WALL|PRODUCTION_LEASE|FUSED_TERMINAL|quarantine|Traceback|error|timeout|task.?id|stream.?id|kernel.?name|kernelName|pending communication|retCode|CQE|completion' "$f" | head -n 100 || true
        echo '--- tail ---'
        tail -n 35 "$f"
    fi
done
echo '===== 4. PC OFFSETS (NOT SYMBOLIZED) ====='
python3 - "$CASE/error.log" <<'PY'
import re,sys
from pathlib import Path
p=Path(sys.argv[1])
if p.is_file():
    s=p.read_text(errors="replace")
    found=set()
    for core,base,pc in re.findall(r"core id is (\d+).*?pc start:\s*(0x[0-9a-fA-F]+),\s*current:\s*(0x[0-9a-fA-F]+)",s):
        row=(core,base,pc,hex(int(pc,16)-int(base,16)))
        if row not in found:
            print("PC_OFFSET",*row);found.add(row)
PY
echo '===== 5. FROZEN BINARY / BUILD INVENTORY ====='
SO=/tmp/KVAF-integrated-87e261ae/build-npu/lifetch/20ed0253a932fbdbdc73/libv9_bridge.so
sha256sum "$SO"
find "$(dirname "$SO")" -maxdepth 5 -type f \( -name '*.o' -o -name '*.a' -o -name '*.elf' -o -name '*.map' -o -name '*.json' \) -printf '%s %p\n' | head -n 120 || true
echo '===== 6. PACK EXISTING TEXT EVIDENCE ====='
# No tensor payloads, binary dumps, source tree, or process memory.
python3 - "$ROOT" "$WORKER" "$OUT" <<'PY'
import hashlib,json,sys,tarfile
from pathlib import Path
root,worker,out=map(Path,sys.argv[1:])
files=[root/n for n in ("run.log","test_commit.txt","service.pid","cpu_service.log","basic_bs4.log")]
files += [p for p in worker.rglob("*") if p.is_file() and p.suffix in (".log",".json",".txt")]
files += list((out).glob("npu-smi.txt"))
files=list(dict.fromkeys(p for p in files if p.is_file() and not p.is_symlink()))
index={}
with tarfile.open(out/"v1-timeout-logs.tar.gz","w:gz") as t:
    for p in files:
        if p.stat().st_size>32*1024*1024:
            print("SKIP_LARGE_LOG",p);continue
        rel=("failed-run/"+str(p.relative_to(root))) if p.is_relative_to(root) else ("snapshot/"+p.name)
        t.add(p,arcname=rel,recursive=False)
        index[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
(out/"evidence-sha256.json").write_text(json.dumps(index,indent=2)+"\n")
print("EVIDENCE_FILES",len(index))
PY
echo "V1_TIMEOUT_EVIDENCE_COLLECTED"
echo "OUTPUT_DIR=$OUT"
