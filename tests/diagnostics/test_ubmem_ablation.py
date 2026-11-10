# SPDX-License-Identifier: Apache-2.0
"""CPU orchestration contracts only; mock timings are not NPU evidence."""

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import sys
import json
from types import SimpleNamespace

import pytest
import numpy as np

from kvip.diagnostics import replay_measure, ubmem_ablation


def geometry():
    return dict(batch=16, aic_count=32, requested_windows=8, prefetch_limit=512, window_limit=64)


def test_three_cases_change_only_tuning_and_preserve_archive():
    original = geometry()
    saved = deepcopy(original)
    cases = ubmem_ablation.plan_cases(original)
    assert [name for name, _ in cases] == ["baseline", "optimized", "optimized_half_prefetch"]
    assert cases[0][1] == dict(parallel_init=False, balanced_index=False, prefetch_limit=512, window_limit=64)
    assert cases[1][1] == dict(parallel_init=True, balanced_index=True, prefetch_limit=512, window_limit=64)
    assert cases[2][1] == dict(parallel_init=True, balanced_index=True, prefetch_limit=256, window_limit=32)
    assert original == saved
    zero = ubmem_ablation.plan_cases(dict(original, prefetch_limit=0, window_limit=1))
    assert (zero[2][1]["prefetch_limit"], zero[2][1]["window_limit"]) == (0, 1)
    with pytest.raises(ValueError, match="budgets"):
        ubmem_ablation.plan_cases(dict(original, window_limit=0))


def test_summary_joins_corresponding_index_samples_without_subtracting_full_latency(capsys):
    case = dict(
        name="optimized",
        tuning={},
        timing=dict(mean_us=51.0, p50_us=50.0),
        reads_samples=[dict(prefetch=4, final=6, unused_prefetch=2, bytes=26240)],
        stage_diagnostics=dict(
            stages={
                "init": dict(p50_us=2.0),
                "index_plan": dict(samples_us=[1, 3, 9]),
                "index_outputs": dict(samples_us=[9, 3, 1]),
                "index_apply": dict(samples_us=[2, 2, 2]),
            }
        ),
    )
    summary = ubmem_ablation.summarize_case(case)
    assert summary["full_graph_p50_us"] == 50.0
    assert summary["full_graph_mean_us"] == 51.0
    assert summary["index_diagnostic_p50_us"] == 12.0  # median([12, 8, 12]), not sum of medians.
    assert summary["init_diagnostic_p50_us"] == 2.0
    ubmem_ablation.print_summary(dict(summary=[summary]), Path("result.json"))
    text = capsys.readouterr().out
    assert "not additive" in text and "Result JSON:" in text and "optimized" in text
    del case["stage_diagnostics"]
    assert ubmem_ablation.summarize_case(case)["index_diagnostic_p50_us"] is None


def test_summary_does_not_label_merged_completion_as_index_only():
    case = dict(
        name="optimized",
        tuning={},
        timing=dict(mean_us=51, p50_us=50),
        reads_samples=[dict(prefetch=4, final=6, unused_prefetch=2, bytes=26240)],
        stage_diagnostics=dict(
            stages={
                "complete_index_plan": dict(samples_us=[5, 10]),
                "index_outputs": dict(samples_us=[20, 10]),
                "index_apply": dict(samples_us=[5, 5]),
            }
        ),
    )
    summary = ubmem_ablation.summarize_case(case)
    assert summary["index_diagnostic_p50_us"] is None
    assert summary["completion_index_diagnostic_p50_us"] == 27.5


@pytest.fixture
def native_mocks(monkeypatch, tmp_path):
    from kvip.diagnostics import ddr, fused_replay, replay_runtime
    from kvip.operators.lifetch_ubmem import specialization
    from kvip.runtime import device as device_module

    events, leases, tunings = [], [], []
    controls = dict(fail_verify=False, fail_reset=False, fail_close=False)
    table = object()
    args = SimpleNamespace(
        build_dir=tmp_path,
        device=0,
        soc="Ascend950DT_9582",
        transport="ubmem",
        mode="fused",
        warmup=0,
        iterations=1,
        stage_iterations=1,
        split_index_stages=True,
        serial_workers=None,
        output=None,
    )
    archive = SimpleNamespace(
        directory=tmp_path / "snapshot",
        manifest=dict(geometry=geometry(), ddr=dict(region_bytes=4096, slot_bytes=128), fidelity={}),
        array=lambda name: np.empty((16, 16384 if name == "pre.reserved_hot" else 2048), dtype=np.int32),
    )

    class Graph:
        def capture_begin(self, stream):
            events.append("capture_begin")

        def capture_end(self):
            events.append("capture_end")

        def replay(self, stream):
            events.append("replay")

        def reset(self):
            events.append("graph_reset")
            if controls["fail_reset"]:
                raise RuntimeError("graph reset failed")

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self, stream):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 1.0

    class Lease:
        def __init__(self):
            self.graphs = []
            self.closed = False

        def capture_scope(self):
            return nullcontext()

        def retain_graph(self, graph):
            self.graphs.append(graph)

        def close(self):
            events.append("lease_close")
            if controls["fail_close"]:
                raise RuntimeError("lease close failed")
            self.closed = True
            self.graphs.clear()

    class Runtime:
        def initialize(self, *args):
            events.append("initialize")

        def access_table_view(self, *args, **kwargs):
            events.append("table_view")
            return table

        def release_access_table(self, handle):
            assert all(lease.closed for lease in leases)
            assert handle == "handle"
            events.append("release_table")

        def shutdown(self):
            events.append("shutdown")

    runtime = Runtime()
    stream = SimpleNamespace(synchronize=lambda: events.append("sync"))
    torch = SimpleNamespace(
        device=lambda name: name,
        npu=SimpleNamespace(
            is_available=lambda: True,
            set_device=lambda device: None,
            Stream=lambda **kwargs: stream,
            set_stream=lambda stream: None,
            Event=Event,
        ),
    )
    kvaf = SimpleNamespace(__file__=str(tmp_path / "python/kvaf/__init__.py"), AclGraph=Graph)
    monkeypatch.setitem(sys.modules, "kvaf", kvaf)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(specialization, "kvaf_cann", lambda build: tmp_path / "cann")
    monkeypatch.setattr(device_module, "query_device_resources", lambda *args: dict(effective_aic=32))

    def create_runtime(*args):
        events.append("runtime")
        return runtime

    def import_pages(received_runtime, *args, **kwargs):
        assert received_runtime is runtime
        events.append("import_pages")
        return dict(access_table="handle", new_base=8192, old_base=4096)

    def factory(received_torch, received_runtime, archive, restored, received_table, *args, **kwargs):
        assert received_torch is torch and received_runtime is runtime and received_table is table
        assert all(lease.closed for lease in leases)
        events.append("factory")
        tunings.append(kwargs["tuning"])
        assert kwargs["split_index_stages"] is True
        lease = Lease()
        leases.append(lease)

        def verify():
            if controls["fail_verify"]:
                raise AssertionError("oracle failed")
            return dict(prefetch=4, final=6, unused_prefetch=2, bytes=26240)

        return SimpleNamespace(
            lease=lease,
            library=None,
            tuning=kwargs["tuning"] or {},
            reset=lambda: events.append("input_reset"),
            verify=verify,
            forward=lambda: events.append("forward"),
            stage_calls=[("init", lambda: events.append("init")), ("index_plan", lambda: events.append("index"))],
            selection_comparison=dict(status="exact"),
        )

    monkeypatch.setattr(replay_runtime, "create_runtime", create_runtime)
    monkeypatch.setattr(ddr, "import_pages", import_pages)
    monkeypatch.setattr(fused_replay, "create_fused", factory)
    monkeypatch.setattr(replay_measure, "prepare_fused_state", lambda *args: None)
    monkeypatch.setattr(replay_measure, "file_digest", lambda path: "mock_hash")
    monkeypatch.setattr(replay_measure, "source_commit", lambda: "mock_commit")
    monkeypatch.setattr(ubmem_ablation.gc, "collect", lambda: events.append("collect"))
    return SimpleNamespace(args=args, archive=archive, events=events, controls=controls, tunings=tunings, leases=leases)


def test_ablation_reuses_runtime_import_table_and_releases_every_case_before_the_next(native_mocks):
    mock = native_mocks
    retained = {}
    result = ubmem_ablation.run_ablation(mock.args, mock.archive, retained)
    assert result["status"] == "pass" and len(result["cases"]) == 3
    for event in ("runtime", "initialize", "import_pages", "table_view", "release_table", "shutdown"):
        assert mock.events.count(event) == 1
    assert mock.events.count("factory") == mock.events.count("lease_close") == mock.events.count("collect") == 3
    assert mock.events[-2:] == ["release_table", "shutdown"]
    factories = [i for i, name in enumerate(mock.events) if name == "factory"]
    for index, start in enumerate(factories):
        stop = factories[index + 1] if index + 1 < len(factories) else mock.events.index("release_table")
        case = mock.events[start:stop]
        assert case.index("graph_reset") < case.index("lease_close") < case.index("collect")
        assert case.count("graph_reset") == 3  # Full graph and two diagnostic graphs.
    assert "active_case" not in retained
    assert not hasattr(mock.args, "ubmem_tuning")
    assert result["geometry"]["prefetch_limit"] == 512
    assert result["cases"][-1]["tuning"]["prefetch_limit"] == 256


def test_explicit_variants_keep_one_import_and_each_requested_configuration(native_mocks):
    mock = native_mocks
    cases = [dict(name="current", tuning={}), dict(name="tagged", tuning={"tagged_map": True})]
    mock.args.fused_variants = ubmem_ablation.parse_variants(json.dumps(cases))
    result = ubmem_ablation.run_ablation(mock.args, mock.archive, {})
    assert result["ablation"] == "ubmem_explicit_variants"
    assert [case["name"] for case in result["cases"]] == ["current", "tagged"]
    assert mock.tunings == [{}, {"tagged_map": True}]
    assert mock.events.count("import_pages") == mock.events.count("runtime") == 1
    assert mock.events.count("factory") == mock.events.count("lease_close") == 2


def test_explicit_variants_reject_malformed_and_conflicting_cli_options(tmp_path):
    from kvip.diagnostics.replay import arguments

    variants = '[{"name":"current","tuning":{}},{"name":"tagged","tuning":{"tagged_map":true}}]'
    common = [
        "--snapshot",
        "/snapshot",
        "--build-dir",
        "/build",
        "--soc",
        "soc",
        "--output",
        str(tmp_path / "result.json"),
    ]
    assert arguments(common + ["--fused-variants", variants]).fused_variants[-1][1] == {"tagged_map": True}
    for value in (
        "{}",
        "[]",
        '[{"name":"a","tuning":{}}]',
        variants.replace('"tagged"', '"current"'),
        variants.replace('"tagged_map"', '"unknown"'),
    ):
        with pytest.raises(SystemExit):
            arguments(common + ["--fused-variants", value])
    for extra in (["--inspect"], ["--mode", "serial"], ["--ubmem-ablation"], ["--fused-tuning", "{}"]):
        with pytest.raises(SystemExit):
            arguments(common + ["--fused-variants", variants] + extra)


@pytest.mark.parametrize("failure", ("fail_verify", "fail_reset", "fail_close"))
def test_failed_case_keeps_active_bundle_graphs_and_shared_runtime(native_mocks, failure):
    mock = native_mocks
    mock.controls[failure] = True
    retained = {}
    with pytest.raises((RuntimeError, AssertionError)):
        ubmem_ablation.run_ablation(mock.args, mock.archive, retained)
    assert "runtime" in retained and "table" in retained and "bundle" in retained["active_case"]
    assert "release_table" not in mock.events and "shutdown" not in mock.events and "collect" not in mock.events
    assert mock.events.count("factory") == 1
    if failure != "fail_verify":
        assert retained["active_case"]["graphs"]


def test_normal_run_preserves_one_case_api(native_mocks):
    mock = native_mocks
    result = replay_measure.run(mock.args, mock.archive, {})
    assert result["status"] == "pass" and result["mode"] == "fused"
    assert mock.events.count("factory") == 1 and mock.events[-2:] == ["release_table", "shutdown"]


def test_service_matrix_reuses_ddr_and_pairs_identical_derived_inputs(native_mocks, monkeypatch):
    from kvip.diagnostics import active_rows

    mock = native_mocks
    selected = []

    def mask(source, count):
        assert source is mock.archive
        selected.append(count)
        return source

    monkeypatch.setattr(active_rows, "mask_active_rows", mask)
    mock.args.service_ablation = True
    mock.args.active_batches = [1, 4, 16]
    report = ubmem_ablation.run_ablation(mock.args, mock.archive, {})
    assert selected == [1, 4, 16]
    assert len(report["cases"]) == 8
    assert mock.events.count("import_pages") == mock.events.count("shutdown") == 1
    for case, tuning in zip(report["cases"], mock.tunings):
        assert tuning["parallel_init"] is False and tuning["balanced_index"] is True
        assert tuning["prefetch_limit"] == 256 and tuning["window_limit"] == 32
        assert tuning["skip_inactive"] is (not case["name"].endswith("dense_baseline"))
        assert tuning["compact_reserved"] is case["name"].endswith("compact_skip")
    assert "not reconstruct the full service step" in report["interpretation"]


def test_service_matrix_rejects_invalid_counts_before_allocating(native_mocks):
    for counts in ([], [0], [17]):
        with pytest.raises(ValueError, match="Active request counts"):
            ubmem_ablation.service_cases(native_mocks.archive, counts)
    assert not native_mocks.events


def test_cli_service_ablation_requires_ubmem_fused_and_an_output(tmp_path):
    from kvip.diagnostics.replay import arguments

    common = ["--snapshot", str(tmp_path), "--build-dir", str(tmp_path), "--soc", "Ascend950DT_9582"]
    args = arguments(
        common + ["--service-ablation", "--active-batches", "1", "4", "16", "--output", str(tmp_path / "x.json")]
    )
    assert args.service_ablation and args.active_batches == [1, 4, 16]
    for extra in (["--service-ablation"], ["--active-batches", "1"], ["--service-ablation", "--mode", "serial"]):
        with pytest.raises(SystemExit):
            arguments(common + extra)


def test_cli_call_preparation_accepts_fused_device_paths_and_rejects_inspect_serial(tmp_path):
    from kvip.diagnostics.replay import arguments

    common = ["--snapshot", str(tmp_path), "--build-dir", str(tmp_path), "--soc", "Ascend950DT_9582"]
    assert not arguments(common).measure_call_preparation
    assert arguments(common + ["--measure-call-preparation"]).measure_call_preparation
    urma = arguments(common + ["--transport", "urma", "--kvaf-server", "127.0.0.1:28880", "--measure-call-preparation"])
    assert urma.measure_call_preparation and urma.stage_iterations == 0
    for extra in (["--inspect"], ["--mode", "serial"]):
        with pytest.raises(SystemExit):
            arguments(common + extra + ["--measure-call-preparation"])
    matrix = arguments(
        common
        + [
            "--service-ablation",
            "--measure-call-preparation",
            "--stage-iterations",
            "2",
            "--output",
            str(tmp_path / "x.json"),
        ]
    )
    assert matrix.service_ablation and matrix.measure_call_preparation and matrix.stage_iterations == 2


def test_summary_keeps_call_wall_cpu_and_optional_h2d_separate(capsys):
    calls = []
    for name, dense, h2d in (("dense", 1048576, 8.0), ("compact", 0, None)):
        call = dict(
            wall=dict(mean_us=120.0, p50_us=115.0),
            cpu_prepare=dict(p50_us=9.0),
            payload=dict(candidate_h2d_bytes=131072, dense_protection_h2d_bytes=dense),
        )
        if h2d is not None:
            call["h2d_diagnostic"] = dict(p50_us=h2d)
        calls.append(dict(name=name, candidate_call=call))
    ubmem_ablation.print_summary(dict(summary=[], cases=calls))
    text = capsys.readouterr().out
    assert "group_call_mean_us" in text and "h2d_diag_p50_us" in text
    assert "dense  120.000  115.000  9.000  8.000  1179648" in text
    assert "compact  120.000  115.000  9.000  -  131072" in text
    assert "Do not multiply by 21" in text and "excludes full-step/shared preparation" in text


@pytest.mark.parametrize("fail", (False, True))
def test_call_preparation_integration_runs_before_graph_release_and_reports_distinct_scope(
    native_mocks, monkeypatch, fail
):
    from kvip.diagnostics import call_preparation

    mock = native_mocks
    mock.args.measure_call_preparation = True
    preparation = object()

    def create_preparation(torch, archive, bundle):
        assert archive is mock.archive
        mock.events.append("candidate_prepare_setup")
        return preparation

    def measure_call(torch, stream, bundle, args, replay, retained):
        assert bundle.candidate_preparation is preparation
        assert retained["bundle"] is bundle and retained["graphs"]
        assert "graph_reset" not in mock.events and not mock.leases[0].closed
        mock.events.append("candidate_call")
        replay()
        if fail:
            raise RuntimeError("candidate call failed")
        return dict(scope="single_group_candidate_ready_to_output_ready", wall=dict(p50_us=123.0))

    monkeypatch.setattr(call_preparation, "CandidatePreparation", create_preparation)
    monkeypatch.setattr(call_preparation, "measure_candidate_call", measure_call)
    retained = {}
    if fail:
        with pytest.raises(RuntimeError, match="candidate call failed"):
            replay_measure.run(mock.args, mock.archive, retained)
        assert retained["active_case"]["bundle"].candidate_preparation is preparation
        assert retained["active_case"]["graphs"] and "runtime" in retained
        assert "graph_reset" not in mock.events and "release_table" not in mock.events and "shutdown" not in mock.events
        return
    report = replay_measure.run(mock.args, mock.archive, retained)
    assert mock.events.index("candidate_prepare_setup") < mock.events.index("candidate_call")
    assert mock.events.index("candidate_call") < mock.events.index("graph_reset") < mock.events.index("lease_close")
    assert report["timing"]["p50_us"] == 1000.0  # Mock device event interval, separate from mock call wall.
    assert report["candidate_call"]["wall"]["p50_us"] == 123.0
    assert report["reserved_metadata"]["h2d_in_timing"] is False
    assert report["reserved_metadata"]["h2d_in_candidate_call"] is True


@pytest.mark.parametrize("transport,mode", (("urma", "fused"), ("ubmem", "serial")))
def test_ablation_rejects_other_paths_before_allocation(transport, mode):
    with pytest.raises(ValueError, match="requires"):
        ubmem_ablation.run_ablation(SimpleNamespace(transport=transport, mode=mode), None, {})
