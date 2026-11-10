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
            "complete_index_update",
        ]
        assert calls == [(name, (0x1000, 0x4000)) for name in ("v9_init", "v9_li", "v9_resolve")] + [
            ("kvip_ubmem_complete_plan", (0x1000, 0x2000, 0x4000)),
            ("kvip_ubmem_update_outputs", (0x2000, 0x4000)),
            ("kvip_ubmem_update_apply", (0x2000, 0x4000)),
        ]
        assert library.functions["kvip_ubmem_complete_plan"].argtypes == [ctypes.c_void_p] * 3
    else:
        assert [name for name, _ in stages] == [
            "init",
            "li_with_prefetch",
            "resolve_final_plan",
            "index_update_before_final_get",
            "final_get_and_completion",
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
    assert [name for name, _ in stages][-3:] == ["complete_index_plan", "index_outputs", "index_apply"]
    for _, launch in stages:
        launch()
    assert calls[-3:] == [("kvip_ubmem_complete_plan", (11, 22, 33))] + [
        ("kvip_ubmem_update_" + name, (22, 33)) for name in ("outputs", "apply")
    ]


def test_combined_completion_stage_stops_on_plan_failure():
    calls = []
    library = Library(calls)
    rotating = SimpleNamespace(library=library, bindings=SimpleNamespace(data_ptr=lambda: 11))
    update = SimpleNamespace(bindings=SimpleNamespace(data_ptr=lambda: 22))
    lease = SimpleNamespace(check_stream=lambda: SimpleNamespace(npu_stream=33))
    stages = fused_stages("ubmem", rotating, update, lease, 16)
    library.result = 17
    with pytest.raises(RuntimeError, match="complete_plan failed: 17"):
        stages[-1][1]()
    assert calls == [("kvip_ubmem_complete_plan", (11, 22, 33))]


def test_tuning_keeps_snapshot_immutable_and_validates_only_known_options():
    geometry = dict(prefetch_limit=512, window_limit=64, requested_windows=8)
    optimized = dict(parallel_init=True, balanced_index=True, prefetch_limit=256, window_limit=32)
    assert fusion_tuning(geometry, "ubmem", optimized) == dict(
        optimized,
        skip_inactive=False,
        compact_reserved=False,
        grouped_copy=False,
        tagged_map=False,
        final_copy_stages=4,
        final_copy_rows=2,
        ordered_pair=False,
        merge_outputs=False,
        single_plan=False,
        cached_buckets=False,
        fused_init=False,
        cold_only_prefetch=False,
        threshold_only=False,
        simt_init=False,
        staged_outputs=False,
        rolling_threshold=False,
        reserved_ub=False,
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
    assert fusion_tuning(geometry, "urma", overrides) == dict(
        overrides,
        skip_inactive=True,
        parallel_final=False,
        async_final=False,
        single_planner=False,
        skip_final_partition=False,
        post_final_before_index=False,
        merge_final_completion=False,
        cold_only_prefetch=False,
        threshold_only=False,
        simt_init=False,
        staged_outputs=False,
        rolling_threshold=False,
        reserved_ub=False,
    )
    assert fusion_tuning(geometry, "urma", {"parallel_final": True})["parallel_final"] is True
    assert fusion_tuning(geometry, "ubmem", {"grouped_copy": True})["grouped_copy"] is True
    for transport, invalid in (
        ("urma", {"grouped_copy": False}),
        ("urma", {"skip_inactive": True}),
        ("ubmem", {"grouped_copy": 1}),
        ("urma", {"prefetch_limit": True}),
        ("urma", {"parallel_final": 1}),
        ("ubmem", {"parallel_final": True}),
        ("urma", {"async_final": True}),
        ("urma", {"async_final": 1, "parallel_final": True}),
        ("ubmem", {"async_final": False}),
        ("ubmem", {"ordered_pair": True}),
        ("ubmem", {"merge_outputs": 1}),
        ("urma", {"merge_outputs": True}),
        ("urma", {"single_plan": True}),
        ("ubmem", {"single_plan": 1}),
        ("urma", {"cold_only_prefetch": 1}),
        ("ubmem", {"threshold_only": 1}),
        ("ubmem", {"simt_init": True, "parallel_init": True}),
        ("urma", {"single_planner": True}),
        ("urma", {"skip_final_partition": True, "parallel_final": True, "async_final": True}),
        ("ubmem", {"single_planner": False}),
        ("ubmem", {"rolling_threshold": 1}),
        ("urma", {"rolling_threshold": True, "threshold_only": True}),
        ("urma", {"post_final_before_index": True}),
        ("urma", {"post_final_before_index": 1, "parallel_final": True}),
        ("urma", {"post_final_before_index": True, "parallel_final": True, "async_final": True}),
        ("ubmem", {"post_final_before_index": True}),
        ("urma", {"cached_buckets": False}),
        ("ubmem", {"cached_buckets": 1}),
        ("ubmem", {"reserved_ub": True}),
        ("urma", {"reserved_ub": True, "compact_reserved": True, "simt_init": True}),
        ("ubmem", {"reserved_ub": True, "compact_reserved": True, "parallel_init": True}),
        ("ubmem", {"fused_init": 1}),
        ("ubmem", {"fused_init": True, "parallel_init": True}),
        ("urma", {"fused_init": False}),
        ("ubmem", {"final_copy_stages": 6}),
        ("ubmem", {"final_copy_rows": 4}),
        ("ubmem", {"final_copy_rows": True}),
        ("ubmem", {"final_copy_rows": 4, "final_copy_stages": 6}),
        ("ubmem", {"final_copy_rows": 4, "final_copy_stages": 6, "grouped_copy": True, "ordered_pair": True}),
        ("urma", {"merge_final_completion": True, "parallel_final": True}),
        ("urma", {"merge_final_completion": 1, "parallel_final": True, "post_final_before_index": True}),
        ("ubmem", {"merge_final_completion": False}),
        ("urma", []),
    ):
        with pytest.raises(ValueError):
            fusion_tuning(geometry, transport, invalid)


def test_parallel_final_stages_join_prefetch_then_get_then_reduce():
    calls = []
    library = Library(calls)
    rotating = SimpleNamespace(
        library=library, bindings=SimpleNamespace(data_ptr=lambda: 11), shape_pointer=44, parallel_final=True
    )
    update = SimpleNamespace(library=library, bindings=SimpleNamespace(data_ptr=lambda: 22))
    lease = SimpleNamespace(check_stream=lambda: SimpleNamespace(npu_stream=33))
    stages = fused_stages("urma", rotating, update, lease, 16)
    assert [name for name, _ in stages][-3:] == ["prefetch_completion", "parallel_final_get", "final_completion"]
    for _, launch in stages:
        launch()
    assert calls[-3:] == [
        ("v9_wait", (11, 33, 44)),
        ("v9_parallel_fetch", (11, 33, 44)),
        ("aclrtlaunch_v9_fetch_update_kernel", (16, 33, 11, 22)),
    ]


def test_final_post_before_index_stage_order_and_arguments():
    calls = []
    library = Library(calls)
    rotating = SimpleNamespace(
        library=library,
        bindings=SimpleNamespace(data_ptr=lambda: 11),
        shape_pointer=44,
        parallel_final=True,
        post_final_before_index=True,
    )
    update = SimpleNamespace(library=library, bindings=SimpleNamespace(data_ptr=lambda: 22))
    lease = SimpleNamespace(check_stream=lambda: SimpleNamespace(npu_stream=33))
    stages = fused_stages("urma", rotating, update, lease, 16)
    assert [name for name, _ in stages][3:] == [
        "prefetch_join_and_final_submit",
        "index_update_during_final_get",
        "final_fetch_wait",
        "final_completion",
    ]
    for _, launch in stages:
        launch()
    assert calls[3:] == [
        ("v9_submit_final", (11, 33, 44)),
        ("kvop_direct_update_outputs_early", (16, 22, 33)),
        ("v9_parallel_fetch", (11, 33, 44)),
        ("aclrtlaunch_v9_fetch_update_kernel", (16, 33, 11, 22)),
    ]


@pytest.mark.parametrize("parallel_final,async_final", ((False, False), (True, False), (True, True)))
def test_urma_factory_passes_compact_switch_to_actual_build(monkeypatch, tmp_path, parallel_final, async_final):
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
        assert kwargs["parallel_final"] is parallel_final
        assert kwargs["async_final"] is async_final
        assert all(
            kwargs[name] is True for name in ("cold_only_prefetch", "threshold_only", "simt_init", "staged_outputs")
        )
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
            tuning={
                "compact_reserved": True,
                "parallel_final": parallel_final,
                "async_final": async_final,
                "cold_only_prefetch": True,
                "threshold_only": True,
                "simt_init": True,
                "staged_outputs": True,
            },
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


def test_async_final_stages_expose_post_then_index_then_join_and_drain():
    calls = []
    library = Library(calls)
    rotating = SimpleNamespace(
        library=library,
        bindings=SimpleNamespace(data_ptr=lambda: 11),
        shape_pointer=44,
        parallel_final=True,
        async_final=True,
    )
    update = SimpleNamespace(library=library, bindings=SimpleNamespace(data_ptr=lambda: 22))
    lease = SimpleNamespace(check_stream=lambda: SimpleNamespace(npu_stream=33))
    stages = fused_stages("urma", rotating, update, lease, 16)
    assert [name for name, _ in stages][2:] == [
        "resolve_and_submit_safe_final",
        "index_update_during_safe_final_get",
        "prefetch_completion",
        "conflicting_final_and_completion",
        "final_completion",
    ]
    for _, launch in stages:
        launch()
    assert [name for name, _ in calls][2:] == [
        "v9_resolve",
        "kvop_direct_update_outputs_early",
        "v9_wait",
        "v9_parallel_fetch",
        "aclrtlaunch_v9_fetch_update_kernel",
    ]


@pytest.mark.parametrize("split", (False, True))
def test_merged_ubmem_outputs_use_exact_five_native_launches(split):
    calls = []
    library = Library(calls)
    rotating = SimpleNamespace(
        library=library,
        bindings=SimpleNamespace(data_ptr=lambda: 11),
        shape=SimpleNamespace(merge_outputs=True),
    )
    update = SimpleNamespace(bindings=SimpleNamespace(data_ptr=lambda: 22))
    lease = SimpleNamespace(check_stream=lambda: SimpleNamespace(npu_stream=33))
    stages = fused_stages("ubmem", rotating, update, lease, 16, split_index_stages=split)
    assert [name for name, _ in stages][3:] == (
        ["complete_index_plan_outputs", "index_apply"] if split else ["complete_index_update"]
    )
    for _, launch in stages:
        launch()
    assert calls == [(name, (11, 33)) for name in ("v9_init", "v9_li", "v9_resolve")] + [
        ("kvip_ubmem_complete_plan_outputs", (11, 22, 33)),
        ("kvip_ubmem_update_apply", (22, 33)),
    ]


def test_ubmem_factory_passes_shared_optimizations_to_the_actual_specialization(monkeypatch, tmp_path):
    from kvip.diagnostics import fused_replay
    from kvip.operators.lifetch_ubmem import specialization

    archive = SimpleNamespace(
        manifest=dict(
            geometry=dict(
                batch=16,
                history=131328,
                resident=16384,
                aic_count=32,
                requested_windows=8,
                heads=32,
                token_bytes=2624,
                prefetch_limit=512,
                window_limit=64,
            )
        )
    )
    monkeypatch.setattr(fused_replay, "_check_geometry", lambda *args: None)

    class ReachedBuild(Exception):
        pass

    def build_specialization(**kwargs):
        shape = kwargs["shape"]
        assert (
            shape.cold_only_prefetch
            and shape.threshold_only
            and shape.simt_init
            and shape.single_plan
            and shape.staged_outputs
        )
        assert not shape.parallel_init
        raise ReachedBuild

    monkeypatch.setattr(specialization, "build_specialization", build_specialization)
    with pytest.raises(ReachedBuild):
        create_fused(
            None,
            SimpleNamespace(),
            archive,
            {"access_table": object()},
            None,
            tmp_path,
            "Ascend950DT_9582",
            None,
            None,
            tmp_path,
            32,
            transport="ubmem",
            tuning={
                "cold_only_prefetch": True,
                "threshold_only": True,
                "simt_init": True,
                "single_plan": True,
                "staged_outputs": True,
            },
        )


def test_single_plan_diagnostics_include_both_native_resolve_launches():
    calls = []
    library = Library(calls)
    rotating = SimpleNamespace(
        library=library,
        bindings=SimpleNamespace(data_ptr=lambda: 11),
        shape=SimpleNamespace(single_plan=True, merge_outputs=False),
    )
    update = SimpleNamespace(bindings=SimpleNamespace(data_ptr=lambda: 22))
    lease = SimpleNamespace(check_stream=lambda: SimpleNamespace(npu_stream=33))
    stages = fused_stages("ubmem", rotating, update, lease, 16, split_index_stages=True)
    assert [name for name, _ in stages] == [
        "init",
        "li_with_prefetch",
        "resolve_plan",
        "final_fetch",
        "complete_index_plan",
        "index_outputs",
        "index_apply",
    ]
    for _, launch in stages:
        launch()
    assert calls == [
        (name, (11, 33))
        for name in (
            "v9_init",
            "v9_li",
            "kvip_ubmem_resolve_plan",
            "kvip_ubmem_resolve_copy",
        )
    ] + [
        ("kvip_ubmem_complete_plan", (11, 22, 33)),
        ("kvip_ubmem_update_outputs", (22, 33)),
        ("kvip_ubmem_update_apply", (22, 33)),
    ]


def test_init_li_diagnostic_keeps_initialization_inside_timing():
    calls = []
    library = Library(calls)
    rotating = SimpleNamespace(
        library=library, bindings=SimpleNamespace(data_ptr=lambda: 11), shape=SimpleNamespace(fused_init=True)
    )
    update = SimpleNamespace(bindings=SimpleNamespace(data_ptr=lambda: 22))
    lease = SimpleNamespace(check_stream=lambda: SimpleNamespace(npu_stream=33))
    stages = fused_stages("ubmem", rotating, update, lease, 16)
    assert [name for name, _ in stages] == ["init_li_with_prefetch", "resolve_final_fetch", "complete_index_update"]
    stages[0][1]()
    assert calls == [("v9_init_li", (11, 33))]
    library.result = 17
    with pytest.raises(RuntimeError, match="v9_init_li failed"):
        stages[0][1]()


def test_merged_urma_completion_drains_in_last_stage_once():
    calls = []
    library = Library(calls)
    rotating = SimpleNamespace(
        library=library,
        bindings=SimpleNamespace(data_ptr=lambda: 11),
        shape_pointer=44,
        parallel_final=True,
        post_final_before_index=True,
        merge_final_completion=True,
    )
    update = SimpleNamespace(library=library, bindings=SimpleNamespace(data_ptr=lambda: 22))
    lease = SimpleNamespace(check_stream=lambda: SimpleNamespace(npu_stream=33))
    stages = fused_stages("urma", rotating, update, lease, 16)
    assert [name for name, _ in stages][3:] == [
        "prefetch_join_and_final_submit",
        "index_update_during_final_get",
        "final_wait_and_completion",
    ]
    for _, launch in stages:
        launch()
    assert calls[3:] == [
        ("v9_submit_final", (11, 33, 44)),
        ("kvop_direct_update_outputs_early", (16, 22, 33)),
        ("aclrtlaunch_v9_fetch_update_kernel", (16, 33, 11, 22)),
    ]


def test_urma_split_index_keeps_both_production_launches_and_exact_arguments():
    calls = []
    library = Library(calls)
    rotating = SimpleNamespace(
        library=library,
        bindings=SimpleNamespace(data_ptr=lambda: 11),
        shape_pointer=44,
        parallel_final=True,
        post_final_before_index=True,
        merge_final_completion=True,
    )
    update = SimpleNamespace(library=library, bindings=SimpleNamespace(data_ptr=lambda: 22))
    lease = SimpleNamespace(check_stream=lambda: SimpleNamespace(npu_stream=33))
    stages = fused_stages("urma", rotating, update, lease, 16, split_index_stages=True)
    assert [name for name, _ in stages][3:] == [
        "prefetch_join_and_final_submit",
        "index_outputs_plan",
        "index_apply",
        "final_wait_and_completion",
    ]
    for _, launch in stages:
        launch()
    assert calls[4:6] == [
        ("aclrtlaunch_kvop_direct_update_outputs_early_kernel", (16, 33, 22)),
        ("kvop_direct_apply_owned", (16, 22, 33)),
    ]


def test_urma_split_index_cli_uses_existing_flag_without_an_ablation(monkeypatch):
    import sys
    from kvip.diagnostics.replay import arguments

    argv = [
        "replay",
        "--snapshot",
        "/snapshot",
        "--transport",
        "urma",
        "--mode",
        "fused",
        "--build-dir",
        "/build",
        "--soc",
        "Ascend950DT_9582",
        "--kvaf-server",
        "127.0.0.1:28999",
        "--split-index-stages",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert arguments().split_index_stages is True
    argv[argv.index("fused")] = "serial"
    with pytest.raises(SystemExit):
        arguments()
