"""Check the graph's real launch policy, device tile partition and capture-owned ready flags."""
import ctypes
import shutil
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from kvip.operators.lifetch.terminal import TerminalLeaseBinding
from kvip.operators.lifetch.tiling import RuntimeShape


def test_fixed_score_geometry_and_binding(tmp_path):
    operator = Path(__file__).resolve().parents[3] / "csrc/operators/lightning_indexer_with_fetch"
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler required for fixed Q4 scheduling")
    source = tmp_path / "policy.cpp"
    source.write_text(
        """
#include "fixed_score_groups.h"
#include <cassert>
#include <initializer_list>
extern "C" uint32_t kvip_fixed_q4_score_groups(const rotating::RuntimeShape* s) {
    return fixed_score::LaunchGroups(*s);
}
extern "C" void prove_tiles() {
    for (uint32_t batch : {1u, 2u, 4u, 8u}) {
        const uint32_t parts = fixed_score::GroupsPerRow(batch) / 2;
        for (uint32_t n = 0; n <= 1025; ++n) {
            uint32_t next = 128;
            for (uint32_t p = 0; p < parts; ++p) {
                auto range = fixed_score::Split(128, 128+n, p, parts);
                assert(range.begin == next && range.end >= range.begin);
                assert(range.end - range.begin >= n / parts);
                assert(range.end - range.begin <= (n + parts-1) / parts);
                next = range.end;
            }
            assert(next == 128+n);
        }
    }
}
"""
    )
    lib = tmp_path / "policy.so"
    subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-I", str(operator / "include"),
                    str(source), "-o", str(lib)], check=True)
    library = ctypes.CDLL(str(lib))
    library.prove_tiles()

    class Tensor:
        def __init__(self, dims=None, data=None):
            self.dims, self.data = dims, data

        def data_ptr(self):
            return id(self)

    torch = SimpleNamespace(
        zeros=lambda dims, **kw: Tensor(dims),
        empty=lambda dims, **kw: Tensor(dims),
        tensor=lambda data, **kw: Tensor(data=data),
        int32=None, int64=None, uint8=None,
    )
    shapes = [
        (1, 131328, 8), (2, 131072, 16), (4, 131328, 32), (8, 131328, 32),
        (16, 131328, 0), (20, 131328, 0), (24, 131328, 0), (28, 131328, 0),
        (32, 131328, 0), (4, 524544, 0),
    ]
    for batch, history, expected in shapes:
        shape = RuntimeShape()
        for key, value in dict(batch=batch, history=history, resident=16384, aicCount=32,
                               indexHeads=32, splitQ=1, rounds=min(4, batch*2),
                               liCores=batch*2, finalWorkers=4, tokenBytes=656).items():
            setattr(shape, key, value)
        lease = object.__new__(TerminalLeaseBinding)
        lease.rotating = SimpleNamespace(runtime_shape=shape, torch=torch, device=None, library=library)
        lease.args, lease.templates = Tensor(), Tensor()
        lease.attach_workspace(SimpleNamespace(rotating=lease.rotating, workspace=Tensor()))
        fields = struct.unpack("<6Q2I9Q", bytes(lease.workspace.data))
        if expected:
            assert lease.fixed_score_ready.dims == (2 * expected, 32)
            assert fields[11] == lease.fixed_score_ready.data_ptr()
        else:
            assert lease.fixed_score_ready is None
            assert fields[11] == 0


def test_public_forward_uses_extended_entry_only_with_ready_state(monkeypatch):
    from kvip.operators.lifetch import forward

    class Lease:
        def __init__(self, rotating):
            self.fixed_score_ready = rotating.ready

        def attach_workspace(self, owner):
            pass

    def legacy(*args):
        pass

    def sharded(*args):
        pass

    monkeypatch.setattr(forward, "TerminalLeaseBinding", Lease)
    monkeypatch.setattr(forward.StreamingLIFetchForward, "__init__", lambda *args, **kwargs: None)
    monkeypatch.setattr(forward.LIFetchForward, "set_options", lambda self: None)
    for ready in (None, object()):
        library = SimpleNamespace(kvip_parallel_tail_supported=lambda shape: 1, kvip_parallel_tail_forward=legacy)
        if ready is not None:
            library.kvip_fixed_q4_small_forward = sharded
        rotating = SimpleNamespace(library=library, shape_pointer=None, torch=None, prefetch_limit=256, ready=ready)
        op = forward.LIFetchForward(rotating, None)
        assert op.terminal_function is (sharded if ready is not None else legacy)
