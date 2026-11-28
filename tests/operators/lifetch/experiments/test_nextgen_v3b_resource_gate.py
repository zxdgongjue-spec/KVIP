#!/usr/bin/env python3
"""CPU-only gate regression: no torch, no NPU context."""
from nextgen_v3b_resource_gate import inspect, STALE_RECORD

def sample(n=7,health="OK",hbm=5281,util=0,rows=(),idle=False):
    text=f"| {n} | Ascend950DT | {health} | 380.0 37 0 / 0 |\n| | NA | {util} 0 / 0 {hbm} / 98304 |\n"
    if idle:text+=f"| No running processes found in NPU {n} |\n"
    for pid,name,mem,cpid in rows:
        text+=f"| {n} | {pid} | {name} | {mem} | {cpid} |\n"
    return text

def check(want,raw,n=7,own=None,allow=False):
    got,detail=inspect(raw,n,own,allow)
    assert got==want,(got,want,detail)

s=STALE_RECORD
o=(1234,"python3",332,55)
f=(8888,"vllm",302,333)
check(True,sample(idle=True))
check(False,sample(rows=[s]))
check(True,sample(rows=[s]),allow=True)
check(True,sample(rows=[s,o]),own=55,allow=True)
check(True,sample(rows=[o]),own=55)
check(False,sample(rows=[s,o]),allow=True)
check(False,sample(rows=[s,f]),allow=True)
check(False,sample(rows=[s,o,f]),own=55,allow=True)
check(False,sample(rows=[s,o]),own=56,allow=True)
check(False,sample(rows=[s,s]),allow=True)
check(False,sample(rows=[(s[0],s[1],311,s[3])]),allow=True)
check(False,sample(rows=[(s[0],"python",310,s[3])]),allow=True)
check(False,sample(rows=[(s[0],s[1],310,0)]),allow=True)
check(False,sample(rows=[s],hbm=8193),allow=True)
check(False,sample(rows=[s],util=1),allow=True)
check(False,sample(rows=[s],health="Warning"),allow=True)
check(False,sample(n=6,rows=[s]),n=6,allow=True)
check(False,sample(rows=[s],idle=True),allow=True)
check(False,sample(rows=[s])+ "| 7 | 999 | <INVALID> | 10 | NA |\n",allow=True)
print("V3B_RESOURCE_GATE_REGRESSION_PASS checks=19")
