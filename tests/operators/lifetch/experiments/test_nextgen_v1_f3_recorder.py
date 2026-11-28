#!/usr/bin/env python3
"""CPU-only regression for the F3 eager correctness recorder."""
import ast
import sys
from pathlib import Path
support=Path(__file__).resolve().parent.parent/"support"
sys.path.insert(0,str(support))
from reporting import CorrectnessRecorder, require_eager_check

def fail():
    raise AssertionError("oracle mismatch")

def runtime_error():
    raise RuntimeError("CQ failure")

def expect_error(kind, callback, message):
    try:
        callback()
    except kind as e:
        assert message in str(e),str(e)
    else:
        raise AssertionError("required failure not raised")

r=CorrectnessRecorder()
assert r.status=="NOT_RUN" and r.checked==0
require_eager_check(r,"native",lambda:None)
assert r.status=="PASS" and r.checked==1 and not r.failures
require_eager_check(r,"candidate",lambda:None)
assert r.status=="PASS" and r.checked==2 and not r.failures
r=CorrectnessRecorder()
expect_error(AssertionError,lambda:require_eager_check(r,"native",fail),"oracle mismatch")
assert r.status=="FAIL" and r.checked==1 and len(r.failures)==1
expect_error(AssertionError,lambda:require_eager_check(r,"candidate",lambda:None),"prior correctness failure")
assert r.status=="FAIL" and r.checked==2
r=CorrectnessRecorder()
expect_error(RuntimeError,lambda:require_eager_check(r,"candidate",runtime_error),"CQ failure")
# Runtime failure propagates; no caller may emit a success marker on that path.
tree=ast.parse((support/"execution.py").read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_urma_case")
calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="require_eager_check"]
assert len(calls)==2
assert {(c.args[1].value,c.args[2].id) for c in calls}=={("F3 native eager","native_check"),("F3 candidate eager","probe_verify")}
print("V1_F3_RECORDER_REGRESSION_PASS")
