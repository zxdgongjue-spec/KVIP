#!/usr/bin/env python3
"""Q4 noncausal full-group closure model; not device execution."""
import itertools,runpy,sys
from pathlib import Path
g=runpy.run_path(str(Path(sys.argv[1])/"python/kvip/operators/lifetch/geometry.py"))
H,R=131072,16384
def check(k,windows):
    b=len(k);geo=g["li_geometry"](b,"auto",windows,32)
    assert geo["cores_per_batch"]==2
    triggers=g["trigger_tokens"](b,H,R,"auto",windows,32)
    owners={c:(e,t) for e,(group,t) in enumerate(zip(geo["service_groups"],triggers)) for c in group}
    deps={};skipped=[]
    for c in range(2*b):
        end=k[c//2]
        deps[("score_end",c)]=set()
        for lane in range(2):
            final=("final",c,lane)
            deps[final]={("score_end",c)}
            if c in owners:
                e,t=owners[c];sl=geo["service_groups"][e].index(c)*2+lane
                event=("event",e,sl)
                assert event not in deps
                deps[event]={("score_end",c)} if end<t else set()
                if end<t:skipped.append(event)
                if e:deps[event].add(("event",e-1,sl))
                deps[final].add(event)
            for e in range(len(triggers)):
                for sl in range(geo["service_aivs_per_window"]):
                    deps[final].add(("event",e,sl))
    events={n for n in deps if n[0]=="event"}
    assert len(events)==len(triggers)*geo["service_aivs_per_window"]
    # Model deliberately excludes final work from predecessor publication.
    remaining=set(deps);done=set()
    while remaining:
        ready={x for x in remaining if deps[x]<=done}
        assert ready,("cycle",k,windows,remaining)
        done|=ready;remaining-=ready
    return len(skipped)
assert check([32768,65536,98304,131072],8)==6
assert check([H]*4,8)==0
assert check([H]*8+[32768]*8,8)>0
n=0
for k in itertools.product([R,32768,65536,98304,H],repeat=4):
    for windows in (1,2,4,8):
        check(k,windows);n+=1
for k in ([R]*15+[H],[H]+[R]*15,[H]*8+[R]*8,[R,H]*8):
    for windows in (1,2,4,8,16,32):
        check(k,windows);n+=1
print("V1_F1_CLOSURE_MODEL_PASS",n,"matrix scenarios; plus 3 explicit controls")
