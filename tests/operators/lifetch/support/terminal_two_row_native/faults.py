"""Inject errors into the actual 8,786-entry archive and require independent verification to reject each one."""
import argparse
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

def verify(exe,folder):
    rows=(folder/"input").read_text().splitlines()
    events=(folder/"events").read_text().splitlines()
    j=next(j for j,line in enumerate(rows) if len(line.split())>1 and int(line.split()[0])>32)
    def change(a,idx,fn):
        out=a[:];out[idx]=fn(a[idx]);return out
    def field(line,i,v):
        vals=line.split();vals[i]=str(v);return " ".join(vals)
    p=next(i for i,line in enumerate(events) if " post " in line)
    source=int(rows[j].split()[2])
    variants=[
        ("changed source",change(rows,j,lambda x:field(x,2,source+1)),events),
        ("aliased destination",change(rows,j+1,lambda x:field(x,1,rows[j].split()[1])),events),
        ("foreign owner",change(rows,j,lambda x:field(x,3,31)),events),
        ("duplicate Post",rows,events[:p]+[events[p]]+events[p:]),
        ("foreign Q post",rows,change(events,p,lambda x:field(x,3,(int(x.split()[3])+1)%4))),
        ("duplicate row preparation",rows,[events[0]]+events),
    ]
    with TemporaryDirectory() as t:
        t=Path(t)
        for name,lines,ev in variants:
            (t/"input").write_text("\n".join(lines)+"\n");(t/"events").write_text("\n".join(ev)+"\n")
            v=subprocess.run([exe,t/"input",t/"events",folder/"kv"],capture_output=True,text=True)
            if v.returncode!=1 or not v.stderr.startswith("FAIL "):
                raise ValueError(f"{name} escaped or malformed ({v.returncode} {v.stderr})")
            print(name,"rejected",v.stderr.strip())
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("binary");p.add_argument("fixture",type=Path)
    a=p.parse_args();verify(a.binary,a.fixture)
