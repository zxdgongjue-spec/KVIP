"""Directly execute existing host checks without installing a pytest runner.

The tiny decorator/raises adapter only imports those test functions. It does
not emulate a device or turn mocked launch success into hardware evidence.
"""

import contextlib
import importlib
import inspect
from pathlib import Path
import re
import runpy
import sys
import tempfile
import types
from unittest.mock import Mock
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "tests/operators/lifetch/support")]
p = types.ModuleType("pytest")
p.fixture = lambda *a, **k: (lambda f: f)
p.mark = types.SimpleNamespace(parametrize=lambda *a, **k: (lambda f: f))


@contextlib.contextmanager
def raises(exc, match=None):
    try:
        yield
    except exc as e:
        if match is not None:
            assert re.search(match, str(e)), (match, str(e))
    else:
        raise AssertionError("expected exception " + str(exc))


p.raises = raises
sys.modules["pytest"] = p


class Monkey:
    def __init__(self):
        self.saved = []

    def setattr(self, obj, name, value=None):
        if isinstance(obj, str):
            mod, attr = obj.rsplit(".", 1)
            obj = importlib.import_module(mod)
            value = name
            name = attr
        self.saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, value in self.saved[::-1]:
            setattr(obj, name, value)
        self.saved = []


f = runpy.run_path(str(ROOT / "tests/runtime/test_lifetch_binding.py"))
count = 0


def run(name, *args):
    global count
    m = Monkey()
    f["physical_device"](m)
    try:
        f[name](*(m if x == "MONKEY" else x for x in args))
        count += 1
        print("PASS", name)
    finally:
        m.undo()


for t in ("urma", "ubmem"):
    run("test_public_forward_only_launches_retained_workspace", t)
for b in (0, 17, 32):
    run("test_batch_limit_fails_before_device_allocation", b)
for g in ((33024, 8192, 64), (210944, 131072, 32), (65536, 2048, 64)):
    run("test_runtime_workspace_scales_without_recompilation", *g)
run("test_device_and_aiv_budget_are_checked")
run("test_layer_groups_share_only_ephemeral_storage")
with tempfile.TemporaryDirectory() as d:
    run("test_device_failure_invalidates_only_failed_owner", Path(d))
run("test_c28_b16_uses_four_records_with_independent_v56", "MONKEY")
run("test_queue_requirement_tracks_fetch_workers")
s = runpy.run_path(str(ROOT / "tests/runtime/test_lifetch_kernel_contracts.py"))
for name in ("test_shared_li_dispatch_and_shard_boundaries", "test_scale8_releases_partial_sequence_shard"):
    with tempfile.TemporaryDirectory() as d:
        s[name](Path(d))
        count += 1
        print("PASS", name)
print("EXISTING_RUNPY_CASES_PASSED", count)
T = f["Tensor"]
T.detach = lambda self: self
T.cpu = lambda self: self
T.numpy = lambda self: self.values
LIFetchBinding = importlib.import_module("kvip.operators.lifetch.binding").LIFetchBinding
streaming = importlib.import_module("kvip.operators.lifetch.streaming")

streaming.query_device_resources = lambda *a: dict(aic=28, aiv=56)
assert inspect.signature(LIFetchBinding).parameters["aiv_budget"].default is None
rotating, update = f["binding_fixture"](batch=16)
rotating.runtime_shape.aicCount = 28
rotating.runtime_shape.liCores = 16
rotating.aiv_budget = None
old = f["LIFetchForward"](rotating, update)
old()
assert old.schedule.aiv_budget == 56
for b in (24, 32):
    for route in ("auto", "history", "bsq"):
        rotating, update = f["binding_fixture"](batch=b)
        s = rotating.runtime_shape
        s.aicCount = s.liCores = 28
        s.serviceLanes = 2
        s.serviceAics = 1
        s.rounds = 32
        rotating.staged = True
        rotating.aiv_budget = 56
        rotating.row_owners = None
        rotating.arguments[5] = T(np.full(b, 4, np.int32))
        rotating.arguments[6] = T(np.full(b, s.history, np.int32))
        rotating.library.kvip_lifetch_staged_forward = Mock(return_value=0)
        rotating.library.kvip_staged_plan_bytes = Mock(return_value=64)
        rotating.library.kvip_staged_plan_version = Mock(return_value=2)
        torch = rotating.torch
        torch.uint8 = np.uint8
        torch.uint16 = np.uint16
        torch.float32 = np.float32
        torch.bfloat16 = np.uint16
        torch.full = lambda shape, value, dtype, device: T(np.full(shape, value, dtype=dtype))
        rotating.score_route = route
        forward = f["LIFetchForward"](rotating, update, aiv_budget=56)
        allocations = (torch.zeros.call_count, torch.empty.call_count)
        forward()
        forward()
        assert allocations == (torch.zeros.call_count, torch.empty.call_count)
        assert forward.schedule.owner_capacity == 16 and forward.schedule.urma_queues == 2 * b + 16
        assert all(
            w.rows <= 28 and 2 * w.rows <= 56 and w.stride % 128 == 0 and w.stride > 0
            for ws in (forward.hot, forward.cold)
            for w in ws
        )
        assert len([w for w in forward.cold if w.readyEnd]) == 31 and forward.cold[-1].readyEnd == s.history
        print("STAGED_MOCK_PASSED", b, route, "no forward allocations")

# Serial layer groups share numerical scratch, but retain their own live
# bindings. Empty owner metadata at construction is normal for vLLM startup.
pool = {}
first_pointers, first_binding = None, None
for group in range(2):
    rotating, update = f["binding_fixture"](batch=3, history=8192, resident=6144)
    s = rotating.runtime_shape
    s.aicCount = s.liCores = 4
    s.serviceLanes = 2
    s.serviceAics = 1
    s.rounds = 2
    rotating.staged = True
    rotating.score_route = "bsq"
    rotating.aiv_budget = 16
    rotating.row_owners = T(np.zeros(3, np.int32))
    rotating.streaming_scratch = pool
    rotating.arguments[5] = T(np.full(3, 4, np.int32))
    rotating.arguments[6] = T(np.full(3, 8192, np.int32))
    rotating.library.kvip_lifetch_staged_forward = Mock(return_value=0)
    rotating.library.kvip_staged_plan_bytes = Mock(return_value=64)
    rotating.library.kvip_staged_plan_version = Mock(return_value=2)
    torch = rotating.torch
    torch.uint8 = np.uint8
    torch.uint16 = np.uint16
    torch.float32 = np.float32
    torch.bfloat16 = np.uint16
    torch.full = lambda shape, value, dtype, device: T(np.full(shape, value, dtype=dtype))
    forward = f["LIFetchForward"](rotating, update, aiv_budget=16)
    pointers = (forward.canonical.data_ptr(), forward.hot[0].scores, forward.cold[0].scores)
    if group:
        assert pointers == first_pointers and forward.hot[0].bindings != first_binding
    else:
        first_pointers = pointers
        first_binding = forward.hot[0].bindings
    assert forward.prepare_metadata and forward.schedule.owner_capacity == 2
    allocations = (torch.zeros.call_count, torch.empty.call_count)
    forward()
    forward()
    assert allocations == (torch.zeros.call_count, torch.empty.call_count)
    assert rotating.library.kvip_lifetch_staged_forward.call_count == 2
    assert rotating.library.kvip_lifetch_streaming_forward_v2.call_count == 0
print("PUBLIC_SPLIT_SCORE_SHARED_SCRATCH_PASSED")
print("HOST_BINDING_CHECKS_PASSED; no Model/NPU result")
