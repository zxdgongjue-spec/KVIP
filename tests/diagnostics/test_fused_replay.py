# SPDX-License-Identifier: Apache-2.0
"""Native launch ordering/arguments only: no simulated NPU performance."""

import ctypes
from types import SimpleNamespace

import pytest

from kvip.diagnostics.fused_replay import _check_geometry, create_fused, fused_stages


class Library:
    def __init__(self, calls):
        self.calls, self.functions = calls, {}

    def __getattr__(self, name):
        if name not in self.functions:

            def call(*args):
                self.calls.append((name, args))
                return self.result

            self.functions[name] = call
        return self.functions[name]

    result = 0


@pytest.mark.parametrize("transport", ("ubmem", "urma"))
def test_fused_stages_preserve_exact_production_launch_order_and_arguments(transport):
    calls = []
    library, update_library = Library(calls), Library(calls)
    rotating = SimpleNamespace(library=library, bindings=SimpleNamespace(data_ptr=lambda: 0x1000), shape_pointer=0x3000)
    update = SimpleNamespace(library=update_library, bindings=SimpleNamespace(data_ptr=lambda: 0x2000))
    stream = SimpleNamespace(npu_stream=0x4000)
    lease = SimpleNamespace(check_stream=lambda: stream)
    stages = fused_stages(transport, rotating, update, lease, 16)
    for _, call in stages:
        call()
    if transport == "ubmem":
        assert [name for name, _ in stages] == [
            "init",
            "li_with_prefetch",
            "resolve_final_fetch",
            "complete",
            "index_update",
        ]
        assert calls == [(name, (0x1000, 0x4000)) for name in ("v9_init", "v9_li", "v9_resolve", "v9_fetch")] + [
            ("kvip_ubmem_update", (0x2000, 0x4000))
        ]
        assert library.functions["kvip_ubmem_update"].argtypes == [ctypes.c_void_p, ctypes.c_void_p]
    else:
        assert [name for name, _ in stages] == [
            "init",
            "li_with_prefetch",
            "resolve_final_submit",
            "index_update_while_fetch",
            "fetch_complete",
        ]
        assert calls == [(name, (0x1000, 0x4000, 0x3000)) for name in ("v9_init", "v9_li", "v9_resolve")] + [
            ("kvop_direct_update_outputs_early", (16, 0x2000, 0x4000)),
            ("aclrtlaunch_v9_fetch_update_kernel", (16, 0x4000, 0x1000, 0x2000)),
        ]
        assert (
            library.functions["aclrtlaunch_v9_fetch_update_kernel"].argtypes
            == [ctypes.c_uint32] + [ctypes.c_void_p] * 3
        )
    stream.npu_stream = 0x5000
    stages[0][1]()
    assert calls[-1][1][1] == 0x5000  # Startup capture may use a lease-approved auxiliary stream.
    library.result = 17
    with pytest.raises(RuntimeError, match="v9_init failed: 17"):
        stages[0][1]()


@pytest.mark.parametrize("batch,rounds,lanes", ((1, 2, 2), (16, 8, 8)))
def test_fused_geometry_preserves_captured_schedule(batch, rounds, lanes):
    geometry = dict(batch=batch, requested_windows=8, aic_count=32, rounds=rounds, service_lanes=lanes, final_workers=4)
    _check_geometry(geometry, 32)
    with pytest.raises(ValueError, match="fewer AICs"):
        _check_geometry(geometry, 31)
    with pytest.raises(ValueError, match="scheduling"):
        _check_geometry(dict(geometry, rounds=rounds + 1), 32)


def test_factory_rejects_wrong_transport_or_missing_page_lease_before_allocating():
    with pytest.raises(ValueError, match="transport"):
        create_fused(
            None,
            SimpleNamespace(transport="ubmem"),
            None,
            {},
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            transport="urma",
        )
    # Historical URMA builds expose no .transport property; explicit selection
    # accepts that Runtime and proceeds to the real published-page requirement.
    with pytest.raises(ValueError, match="published DDR"):
        create_fused(None, SimpleNamespace(), None, {}, None, None, None, None, None, None, None, transport="urma")
