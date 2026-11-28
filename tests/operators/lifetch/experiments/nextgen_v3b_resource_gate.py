#!/usr/bin/env python3
"""Read-only device admission for V3b. Unknown or INVALID rows always block."""
import argparse, os, re, subprocess, time
from pathlib import Path

def inspect(raw, n, own_pid=None):
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
    own=(own_pid is not None and len(procs)==1 and
         procs[0][1].lower() in ("python","python3","python3.12","python3.12.13") and
         (procs[0][0]==own_pid or procs[0][3]==own_pid))
    state_ok=(len(matches)==1 and matches[0][0]=="OK" and matches[0][2]==98304
              and matches[0][1]<=8192 and matches[0][3]==0)
    return bool(state_ok and valid_rows and ((idle and not procs) or (not idle and own))),dict(resources=matches,idle=idle,processes=procs,own_pid=own_pid)

def wait_gate(n,log,own_pid=None,seconds=90):
    assert n in (0,6,7)
    end=time.monotonic()+seconds
    stable=0
    while True:
        r=subprocess.run(["npu-smi","info"],capture_output=True,text=True,timeout=30)
        with Path(log).open("a") as f:f.write(r.stdout+r.stderr+"\n")
        ok,detail=inspect(r.stdout,n,own_pid)
        stable=stable+1 if ok and r.returncode==0 else 0
        print(f"NPU{n}_GATE_CHECK stable={stable} {detail}",flush=True)
        if stable>=2:
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
