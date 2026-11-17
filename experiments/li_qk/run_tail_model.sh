#!/usr/bin/env bash
# Run in an isolated CANN950 Model environment; fixed 32/64 topology.
set -Eeuo pipefail
(($#==4)) || { echo 'Usage: bash run_tail_model.sh BUILD NEW_OUTPUT MODEL_LIB LLVM_OBJDUMP' >&2; exit 2; }
LI_TAIL_BUILD=$1
LI_TAIL_OUT=$2
LI_TAIL_LIB=$3
LI_TAIL_OBJDUMP=$4
LI_TAIL_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
[[ ! -e $LI_TAIL_OUT ]] || { echo 'Fresh output required' >&2; exit 1; }
mkdir -p "$LI_TAIL_OUT"
capture_tail(){
 local li_tail_name=$1
 shift
 python3 "$LI_TAIL_HERE/tail.py" model --build "$LI_TAIL_BUILD" --output "$LI_TAIL_OUT/$li_tail_name" \
  --model-lib "$LI_TAIL_LIB" --objdump "$LI_TAIL_OBJDUMP" --core-mask 0xffffffff --timeout 1500 "$@"
}
# Validate the unchanged production resident primitive before long captures.
capture_tail resident_reference --history 32768 --resident 8192 --parts 4 --stage resident --controls C0
capture_tail target_all --history 131072 --resident 16384 --parts 4 --controls C0 C1
capture_tail target_final --history 131072 --resident 16384 --parts 4 --stage final --controls C0 C2 --fixture "$LI_TAIL_OUT/target_all/fixture"
capture_tail batch16_smoke --batch 16 --history 8192 --resident 4096 --parts 2 --controls C0 C1 C2 C3
capture_tail resident_split --history 32768 --resident 8192 --parts 4 --resident-parts 2 --stage resident --controls C0 --fixture "$LI_TAIL_OUT/resident_reference/fixture"
python3 - "$LI_TAIL_OUT" "$LI_TAIL_HERE" <<'PY'
import csv,json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);here=Path(sys.argv[2]);rows=[]
for manifest in sorted(root.glob('*/manifest.json')):
 m=json.loads(manifest.read_text())
 if m['status']!='MODEL_PASSED':raise RuntimeError('Incomplete group '+str(manifest))
 for r in m['rows']:
  subprocess.run([sys.executable,str(here/'tail_instructions.py'),str(manifest.parent/r['name']),'--output',str(manifest.parent/r['name']/'verified-phases')],check=True)
  rows.append(dict(group=manifest.parent.name,control=r['name'],batch=r['batch'],history=r['history'],resident=r['resident'],parts=r['parts'],resident_parts=r['resident_parts'],stage=r['stage'],whole_model_ticks=r['whole_model_ticks'],all_core_issue_envelope=r['all_core_issue_envelope'],issued_instructions=r['issued_instructions']))
with (root/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(root/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print(root/'summary.csv')
PY
