"""Match printed device stamps to simulator trace timestamps (no clock guess)."""
import json
from pathlib import Path
import re
import sys
root=Path(sys.argv[1])
log=Path(sys.argv[2]).read_text()
paths=list(root.glob('**/core0.veccore0/trace.json'))
if len(paths)!=1:
    raise ValueError(f'Expected one trace; found {len(paths)}')
events=json.loads(paths[0].read_text())['traceEvents']
by_count={}
for event in events:
    detail=event.get('args',{}).get('detail','')
    if 'SPR:SYS_CNT' in detail:
        match=re.search(r'=0x([0-9a-fA-F]+)',detail)
        if match:
            by_count[int(match[1],16)]=event['ts']
stamps={int(i):by_count[int(value)] for i,value in re.findall(r'stamp(\d+)=(\d+)',log) if int(value) in by_count}
# Source-line markers also recover the probe's per-Q stamps independently of
# host stamp printing. This is the primary timeline source for the hoisted probe.
markers={557:16,562:17,567:18,569:19}
for line,base in markers.items():
    found=sorted((e for e in events if 'SPR:SYS_CNT' in e.get('args',{}).get('detail','') and f'/indexer/incremental_methods.h:{line}\n' in e.get('args',{}).get('code','')),key=lambda e:e['ts'])
    if len(found)==4:
        for q,e in enumerate(found):stamps[base+q*4]=e['ts']
result={'source':str(paths[0]),'time_basis':'vendor trace ts in microseconds; stamp values matched to SYS_CNT instructions','stamps_us':stamps}
if 31 in stamps:
    result['queries']=[{'q':q+1,'topk_export_us':stamps[17+4*q]-stamps[16+4*q],'publish_us':stamps[18+4*q]-stamps[17+4*q],'service_us':stamps[19+4*q]-stamps[18+4*q],'end_us':stamps[19+4*q]} for q in range(4)]
print(json.dumps(result,indent=2))
