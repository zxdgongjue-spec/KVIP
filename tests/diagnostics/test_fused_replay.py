# SPDX-License-Identifier: Apache-2.0
"""Native launch ordering/arguments only: no simulated NPU performance."""

import ctypes
from types import SimpleNamespace

import pytest

from kvip.diagnostics.fused_replay import _check_geometry, create_fused, fused_stages, fusion_tuning


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


def test_split_index_stages_retain_native_order_without_extra_work():
    calls = []
    library = Library(calls)
    rotating = SimpleNamespace(library=library, bindings=SimpleNamespace(data_ptr=lambda: 11))
    update = SimpleNamespace(bindings=SimpleNamespace(data_ptr=lambda: 22))
    lease = SimpleNamespace(check_stream=lambda: SimpleNamespace(npu_stream=33))
    stages = fused_stages("ubmem", rotating, update, lease, 16, split_index_stages=True)
    assert [name for name, _ in stages][-3:] == ["index_plan", "index_outputs", "index_apply"]
    for _, launch in stages:
        launch()
    assert calls[-3:] == [("kvip_ubmem_update_" + name, (22, 33)) for name in ("plan", "outputs", "apply")]


def test_tuning_keeps_snapshot_immutable_and_validates_only_known_options():
    geometry = dict(prefetch_limit=512, window_limit=64, requested_windows=8)
    optimized = dict(parallel_init=True, balanced_index=True, prefetch_limit=256, window_limit=32)
    assert fusion_tuning(geometry, "ubmem", optimized) == dict(
        optimized, skip_inactive=False, compact_reserved=False, grouped_copy=False
    )
    assert geometry == dict(prefetch_limit=512, window_limit=64, requested_windows=8)
    assert fusion_tuning(geometry, "ubmem")["parallel_init"] is False
    for invalid in (dict(unknown=1), dict(parallel_init=1), dict(prefetch_limit=256.0), dict(window_limit=0)):
        with pytest.raises(ValueError):
            fusion_tuning(geometry, "ubmem", invalid)
    with pytest.raises(ValueError):
        fusion_tuning(geometry, "urma", optimized)


def test_urma_tuning_has_only_real_switches_and_fixed_inactive_gate():
    geometry = dict(prefetch_limit=512, window_limit=64, requested_windows=8)
    overrides = dict(compact_reserved=True, prefetch_limit=256, window_limit=32)
    assert fusion_tuning(geometry, "urma", overrides) == dict(overrides, skip_inactive=True)
    assert fusion_tuning(geometry, "ubmem", {"grouped_copy": True})["grouped_copy"] is True
    for transport, invalid in (
        ("urma", {"grouped_copy": False}),
        ("urma", {"skip_inactive": True}),
        ("ubmem", {"grouped_copy": 1}),
        ("urma", {"prefetch_limit": True}),
        ("urma", []),
    ):
        with pytest.raises(ValueError):
            fusion_tuning(geometry, transport, invalid)


def test_urma_factory_passes_compact_switch_to_actual_build(monkeypatch, tmp_path):
    from kvip.diagnostics import fused_replay
    from kvip.runtime import build

    archive = SimpleNamespace(
        manifest=dict(
            geometry=dict(
                batch=16,
                history=131328,
                resident=16384,
                aic_count=32,
                requested_windows=8,
                heads=32,
                prefetch_limit=512,
                window_limit=64,
            )
        )
    )
    monkeypatch.setattr(fused_replay, "_check_geometry", lambda *args: None)
    monkeypatch.setattr(build, "kvaf_build_paths", lambda path: (tmp_path, tmp_path / "headers"))

    class ReachedBuild(Exception):
        pass

    def build_libraries(**kwargs):
        assert kwargs["compact_reserved"] is True
        raise ReachedBuild

    monkeypatch.setattr(build, "build_libraries", build_libraries)
    with pytest.raises(ReachedBuild):
        create_fused(
            None,
            SimpleNamespace(),
            archive,
            {"access_table": object()},
            None,
            tmp_path,
            "soc",
            None,
            None,
            tmp_path,
            32,
            transport="urma",
            tuning={"compact_reserved": True},
        )


def test_single_replay_cli_accepts_backend_tuning_without_changing_baseline_defaults():
    from kvip.diagnostics.replay import arguments

    common = ["--snapshot", "/snapshot", "--build-dir", "/build", "--soc", "soc"]
    assert arguments(common).fused_tuning is None
    assert arguments(common + ["--fused-tuning", '{"grouped_copy":true}']).fused_tuning == {"grouped_copy": True}
    urma = common + ["--transport", "urma", "--kvaf-server", "127.0.0.1:28880"]
    assert arguments(urma + ["--fused-tuning", '{"compact_reserved":true}']).fused_tuning == {"compact_reserved": True}
    for extra in (
        ["--mode", "serial", "--fused-tuning", "{}"],
        ["--inspect", "--fused-tuning", "{}"],
        ["--ubmem-ablation", "--fused-tuning", "{}"],
        ["--service-ablation", "--fused-tuning", "{}"],
        ["--fused-tuning", "{"],
        ["--fused-tuning", "[]"],
        ["--fused-tuning", '{"grouped_copy":1}'],
    ):
        with pytest.raises(SystemExit):
            arguments(common + extra)


def test_replay_inflight_cli_only_accepts_urma_serial_uint32():
    from kvip.diagnostics.replay import arguments

    common = ["--snapshot", "/snapshot", "--build-dir", "/build", "--soc", "soc"]
    serial = common + [
        "--mode",
        "serial",
        "--transport",
        "urma",
        "--kvaf-server",
        "127.0.0.1:28880",
        "--serial-workers",
        "32",
    ]
    assert arguments(serial).urma_serial_max_inflight is None
    assert arguments(serial + ["--urma-serial-max-inflight", "0"]).urma_serial_max_inflight == 0
    for base, value in (
        (common, "0"),
        (common + ["--mode", "serial"], "0"),
        (serial + ["--inspect"], "0"),
        (serial, "-1"),
        (serial, "63"),
        (serial, str(1 << 32)),
    ):
        with pytest.raises(SystemExit):
            arguments(base + ["--urma-serial-max-inflight", value])
