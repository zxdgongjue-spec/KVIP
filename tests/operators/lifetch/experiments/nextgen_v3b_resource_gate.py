#!/usr/bin/env python3
"""Read-only V3b device admission; default denies all unknown/INVALID rows."""
import argparse, os, re, subprocess, time
from pathlib import Path

# One incident-specific exception, disabled unless the operator confirms
# host-level PID absence and exclusive allocation. No generic INVALID bypass.
STALE_RECORD=(3642012,"<INVALID>",310,3642012)
STALE_CONFIRM="NPU7_PID3642012_310MB_HOST_ABSENT"

def inspect(raw, n, own_pid=None, allow_stale=False):
    matches=[]
    rows=raw.splitlines()
    for i,row in enumerate(rows[:-1]):
        if re.match(rf'^\|\s*{n}\s*\|\s*Ascend950DT\s*\|',row):
            m=re.search(r'(\d+)\s*/\s*(\d+)\s*\|\s*$',rows[i+1])
            # Actual npu-smi second line: empty first field, NA, empty field,
            # then utilization followed by memory and HBM values.
            last=rows[i+1].split('|')[-2].strip().split()
            if m and last and last[0].isdigit():
                matches.append((row.split('|')[3].strip(),int(m[1]),int(m[2]),int(last[0])))
    idle=bool(re.search(rf'^\|\s*No running processes found in NPU {n}\s*\|',raw,re.M))
    procs=[]
    for row in rows:
        m=re.match(rf'^\|\s*{n}\s*\|\s*(\d+)\s*\|\s*([^|]+)\|\s*(\d+)\s*\|\s*(\d+)\s*\|',row)
        if m:procs.append((int(m[1]),m[2].strip(),int(m[3]),int(m[4])))
    # Fail closed on malformed rows or contradictory idle/process evidence.
    process_rows=[row for row in rows if re.match(rf'^\|\s*{n}\s*\|\s*\d+\s*\|',row)]
    valid_rows=len(process_rows)==len(procs)
    stale=[row for row in procs if n==7 and allow_stale and row==STALE_RECORD]
    live=[row for row in procs if not (n==7 and allow_stale and row==STALE_RECORD)]
    # Match the visible container PID when available, otherwise host PID.
    own=(own_pid is not None and len(live)==1 and
         live[0][1].lower() in ("python","python3","python3.12","python3.12.13") and
         live[0][3]==own_pid)
    admitted=((idle and not procs) or
              (not idle and len(stale)<=1 and
               ((len(stale)==1 and not live and own_pid is None) or own)))
    state_ok=(len(matches)==1 and matches[0][0]=="OK" and matches[0][2]==98304
              and matches[0][1]<=8192 and matches[0][3]==0)
    return bool(state_ok and valid_rows and admitted),dict(resources=matches,idle=idle,processes=procs,own_pid=own_pid,accepted_stale=stale)

def wait_gate(n,log,own_pid=None,seconds=90):
    assert n in (0,6,7)
    confirm=os.environ.get("NEXTGEN_CONFIRMED_STALE_RECORD","")
    if confirm and (n!=7 or confirm!=STALE_CONFIRM):
        raise RuntimeError("invalid incident-specific stale-record confirmation")
    allow_stale=bool(confirm)
    end=time.monotonic()+seconds
    stable=0
    while True:
        r=subprocess.run(["npu-smi","info"],capture_output=True,text=True,timeout=30)
        with Path(log).open("a") as f:f.write(r.stdout+r.stderr+"\n")
        ok,detail=inspect(r.stdout,n,own_pid,allow_stale)
        stable=stable+1 if ok and r.returncode==0 else 0
        print(f"NPU{n}_GATE_CHECK stable={stable} {detail}",flush=True)
        if stable>=2:
            if detail["accepted_stale"]:
                print("NPU7_CONFIRMED_STALE_RECORD_ACCEPTED",detail["accepted_stale"],flush=True)
            print(f"NPU{n}_{'IDLE' if own_pid is None else 'OWN_CONTEXT'}_GATE_PASS",flush=True)
            return
        if time.monotonic()>=end:raise RuntimeError(f"NPU{n}_RESOURCE_GATE_BLOCKED {detail}")
        time.sleep(3)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--physical",type=int,required=True)
    p.add_argument("--log",type=Path,required=True)
    a=p.parse_args()
    wait_gate(a.physical,a.log)
if __name__=="__main__":main()
