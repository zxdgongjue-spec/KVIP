# SPDX-License-Identifier: Apache-2.0
"""An upstream reference must execute upstream code, not feature flags set to zero."""

import json
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from kvip.diagnostics import upstream
from kvip.diagnostics import compare, replay, replay_measure

REVISION = "1" * 40


def _source(tmp_path):
    root = tmp_path / "upstream"
    package = root / "python/kvip"
    (package / "runtime").mkdir(parents=True)
    (root / "csrc/operators").mkdir(parents=True)
    (package / "__init__.py").write_text("ORIGIN = 'upstream initializer'\n")
    (package / "runtime/__init__.py").write_text("")
    (package / "runtime/build.py").write_text(
        "from pathlib import Path\n"
        "def native_source():\n"
        "    return str(Path(__file__).resolve().parents[3] / 'csrc/operators')\n"
    )
    return root


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE).strip()


def test_checkout_requires_exact_revision_and_unmodified_production_sources(tmp_path):
    root = _source(tmp_path)
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    revision = _git(root, "rev-parse", "HEAD")
    assert upstream.validate_checkout(root, revision) == root
    with pytest.raises(ValueError, match="exactly"):
        upstream.validate_checkout(root, REVISION)
    with pytest.raises(ValueError, match="full 40-character"):
        upstream.validate_checkout(root, revision[:8])
    (root / "python/kvip/new_kernel.py").write_text("# untracked production code\n")
    with pytest.raises(ValueError, match="modified or untracked"):
        upstream.validate_checkout(root, revision=revision)
    (root / "python/kvip/new_kernel.py").unlink()
    (root / "python/kvip/__init__.py").write_text("ORIGIN = 'changed'\n")
    with pytest.raises(ValueError, match="modified or untracked"):
        upstream.validate_checkout(root, revision=revision)


def test_fresh_process_loads_upstream_initializer_build_and_only_feature_diagnostics(tmp_path):
    root = _source(tmp_path)
    code = """
import json, sys
from pathlib import Path
from types import SimpleNamespace
from kvip.diagnostics import upstream
from kvip.diagnostics import archive
diagnostic = archive.__file__
upstream.validate_checkout = lambda path, revision: Path(path).resolve()
upstream.activate(sys.argv[1], sys.argv[2])
import kvip
from kvip.runtime import build
assert kvip.ORIGIN == 'upstream initializer'
assert str(Path(sys.argv[1]) / 'csrc/operators') == build.native_source()
assert archive.__file__ == diagnostic
try:
    import kvip.cache.index_storage
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('Production imports fell back to feature code')
report = upstream.provenance(sys.argv[1], sys.argv[2])
sys.modules['kvip.rogue'] = SimpleNamespace(__file__=diagnostic)
try:
    upstream.provenance(sys.argv[1], sys.argv[2])
except RuntimeError as error:
    assert 'Mixed production imports' in str(error)
else:
    raise AssertionError('Mixed production source was not rejected')
print(json.dumps(report))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(root), REVISION], capture_output=True, text=True, check=True
    )
    report = json.loads(result.stdout)
    assert report["commit"] == REVISION
    assert set(report["python_modules"]) == {"kvip", "kvip.runtime", "kvip.runtime.build"}
    assert all(Path(path).is_relative_to(root) for path in report["python_modules"].values())


def test_activation_rejects_preloaded_production_before_mutating_package(tmp_path, monkeypatch):
    root = _source(tmp_path)
    monkeypatch.setattr(upstream, "validate_checkout", lambda path, revision: root)
    monkeypatch.setitem(sys.modules, "kvip.fake_production", object())
    original = sys.modules["kvip"]
    with pytest.raises(RuntimeError, match="before production imports"):
        upstream.activate(root, REVISION)
    assert sys.modules["kvip"] is original


def test_reused_lidu_requires_identical_actual_source_bytes(tmp_path, monkeypatch):
    first, second = (tmp_path / name for name in ("main", "feature"))
    relative = "csrc/operators/lidu/kernel.cpp"
    for root in (first, second):
        (root / relative).parent.mkdir(parents=True)
        (root / relative).write_text("same source")
    monkeypatch.setattr(upstream, "_git", lambda *args: relative)
    report = upstream.verify_lidu_sources(first, second)
    assert report["sources_identical"]
    (second / relative).write_text("different source")
    with pytest.raises(ValueError, match="sources differ"):
        upstream.verify_lidu_sources(first, second)


def _case(mean, *, tuning=None, main=False):
    return dict(
        status="pass",
        snapshot_sha256="same",
        geometry={},
        scope="complete graph",
        soc="A5",
        device=0,
        reads_samples=[
            dict(
                cold_miss_rate=0.1,
                cold_demand=2,
                selected_unique=20,
                prefetch=2,
                final=1,
                unused_prefetch=1,
                unused_prefetch_bytes=656,
                bytes=1968,
                useful_transfer_bytes=1312,
            )
        ],
        timing=dict(mean_us=mean, p50_us=mean, count=10, min_us=mean, max_us=mean, samples_us=[mean] * 10),
        runtime_workers=32,
        selection_comparison=dict(selected_set_sha256="ids", ordered_output_sha256="ordered"),
        threshold_comparison=dict(
            initial_keys=[[39000 if main else 40000]],
            final_keys=[[41000]],
            mean_gap=2 if main else 1,
            equal_fraction=0 if main else 0.5,
        ),
        score_comparison=dict(status="bit_exact"),
        tuning=tuning,
        timing_boundary="complete graph",
        production_source=dict(kind="exact_upstream_checkout" if main else "feature_checkout", commit=REVISION),
    )


def _reports(transport="urma"):
    result = {
        f"{transport}-upstream-serial": _case(120, main=True),
        f"{transport}-upstream-fused": _case(110, main=True),
    }
    for name, tuning in replay_measure.study_cases(SimpleNamespace(study="skip")).items():
        result[f"{transport}-recent-{name}"] = _case(100 if name == "before" else 90, tuning=tuning)
    return result


def _arguments(tmp_path, extra=()):
    build, snapshot = tmp_path / "build", tmp_path / "snapshot"
    (build / "bin").mkdir(parents=True, exist_ok=True)
    (build / "bin/kvaf_execute_batch_get_kernels.o").touch()
    snapshot.mkdir(exist_ok=True)
    (snapshot / "manifest.json").write_text("{}")
    return [
        "--snapshot",
        str(snapshot),
        "--build-dir",
        str(build),
        "--kvaf-server",
        "127.0.0.1:29095",
        "--result-dir",
        str(tmp_path / "results"),
        *extra,
    ]


def test_compare_defaults_to_urma_and_upstream_requires_explicit_pinned_source(tmp_path, monkeypatch):
    args = compare.arguments(_arguments(tmp_path))
    assert args.transports == "urma" and args.study == "threshold"
    with pytest.raises(SystemExit):
        compare.arguments(_arguments(tmp_path, ["--study", "upstream"]))
    with pytest.raises(SystemExit):
        compare.arguments(_arguments(tmp_path, ["--upstream-root", str(tmp_path), "--upstream-revision", REVISION]))
    validated = []
    monkeypatch.setattr(
        upstream, "validate_checkout", lambda root, revision: validated.append((root, revision)) or root
    )
    args = compare.arguments(
        _arguments(tmp_path, ["--study", "upstream", "--upstream-root", str(tmp_path), "--upstream-revision", REVISION])
    )
    assert validated == [(tmp_path, REVISION)]
    assert args.transports == "urma"


@pytest.mark.parametrize("transports", ["urma", "ubmem", "both"])
def test_upstream_plan_uses_isolated_reference_processes_and_paired_recent_cases(
    tmp_path, monkeypatch, capsys, transports
):
    monkeypatch.setattr(upstream, "validate_checkout", lambda root, revision: root)
    args = compare.arguments(
        _arguments(
            tmp_path,
            [
                "--study",
                "upstream",
                "--transports",
                transports,
                "--upstream-root",
                str(tmp_path / "main"),
                "--upstream-revision",
                REVISION,
            ],
        )
    )
    calls = []

    def child(command, **kwargs):
        calls.append(command)
        backend, mode = (command[command.index("--" + key) + 1] for key in ("transport", "mode"))
        reports = _reports(backend)
        if "--upstream-root" in command:
            assert command[command.index("--upstream-revision") + 1] == REVISION
            assert command[command.index("--study") + 1] == "threshold"
            assert command[command.index("--threshold-policy") + 1] == "resident"
            result = reports[f"{backend}-upstream-{mode}"]
        else:
            assert mode == "fused" and command[command.index("--study") + 1] == "skip"
            result = {"cases": {name: reports[f"{backend}-recent-{name}"] for name in ("before", "after")}}
        assert ("--serial-workers" in command) is (mode == "serial")
        Path(command[command.index("--output") + 1]).write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compare.subprocess, "run", child)
    monkeypatch.setattr(compare, "source_commit", lambda: "feature")
    summary = compare.run(args)
    backends = 2 if transports == "both" else 1
    assert len(calls) == 3 * backends and len(summary["table"]) == 4 * backends
    for backend, effect in summary["effects"].items():
        assert effect["serial_reference"] == backend + "-upstream-serial"
        assert effect["skip_reduction_pct"] == pytest.approx(10)
        assert effect["recent_after_vs_upstream_reduction_pct"] == pytest.approx(100 * (1 - 90 / 110))
        assert summary["table"][backend + "-upstream-fused"]["timing"]["samples_us"] == [110] * 10
    assert "upstream production sources" in capsys.readouterr().out


@pytest.mark.parametrize("change", ["score", "ordered", "source", "revision", "recent_threshold", "final_threshold"])
def test_upstream_summary_does_not_relax_correctness_or_mislabel_feature_flags_as_main(change):
    reports = deepcopy(_reports())
    main, before = reports["urma-upstream-fused"], reports["urma-recent-before"]
    if change == "score":
        main["score_comparison"]["status"] = "unverified"
    elif change == "ordered":
        main["selection_comparison"]["ordered_output_sha256"] = "different-but-tied"
    elif change == "source":
        main["production_source"]["kind"] = "feature_checkout"
    elif change == "revision":
        main["production_source"]["commit"] = "2" * 40
    elif change == "recent_threshold":
        before["threshold_comparison"]["initial_keys"] = [[123]]
    else:
        main["threshold_comparison"]["final_keys"] = [[123]]
    with pytest.raises(ValueError):
        compare.summarize(reports, "upstream")


def test_replay_activates_upstream_before_context_allocation(tmp_path, monkeypatch):
    from kvip.diagnostics import ddr, snapshot

    args = SimpleNamespace(
        upstream_root=tmp_path / "main",
        upstream_revision=REVISION,
        quiet=False,
        output=tmp_path / "result.json",
        snapshot=tmp_path,
        transport="urma",
        mode="serial",
        study="threshold",
        threshold_policy="resident",
    )
    calls = []
    monkeypatch.setattr(replay, "arguments", lambda argv: args)
    monkeypatch.setattr(upstream, "verify_lidu_sources", lambda *unused: calls.append("lidu_sources") or {})
    monkeypatch.setattr(upstream, "activate", lambda *unused: calls.append("activate") or {})
    monkeypatch.setattr(
        replay,
        "load_snapshot",
        lambda path: SimpleNamespace(manifest={"content": "real_li_synthetic_kv", "ddr": {}}, directory=path),
    )
    monkeypatch.setattr(snapshot, "validate_capture_inputs", lambda archive: None)
    monkeypatch.setattr(ddr, "validate_pages", lambda *unused: None)

    def run(*unused):
        assert calls == ["lidu_sources", "activate"]
        calls.append("run")
        return _case(100, main=True)

    monkeypatch.setattr(replay_measure, "run", run)
    replay.main([])
    assert calls[-1] == "run" and args.output.is_file()


@pytest.mark.parametrize("mode", ["serial", "fused"])
def test_replay_cli_rejects_ambiguous_reference_options(tmp_path, mode):
    base = [
        "--snapshot",
        str(tmp_path),
        "--build-dir",
        str(tmp_path),
        "--transport",
        "urma",
        "--mode",
        mode,
        "--kvaf-server",
        "127.0.0.1:29095",
        "--soc",
        "A5",
        "--output",
        str(tmp_path / "out.json"),
    ]
    with pytest.raises(SystemExit):
        replay.arguments(base + ["--upstream-root", str(tmp_path)])
    with pytest.raises(SystemExit):
        replay.arguments(base + ["--upstream-root", str(tmp_path), "--upstream-revision", REVISION, "--study", "skip"])
    parsed = replay.arguments(base + ["--upstream-root", str(tmp_path), "--upstream-revision", REVISION])
    assert parsed.upstream_root == tmp_path and parsed.upstream_revision == REVISION


@pytest.mark.parametrize("transport", ["urma", "ubmem"])
def test_upstream_factory_uses_old_builder_and_binding_signatures(tmp_path, monkeypatch, transport):
    from kvip.adapters import kvaf
    from kvip.diagnostics import fused_replay
    from kvip.operators.lifetch import binding, forward, index_update, workspace
    from kvip.operators.lifetch_ubmem import binding as ub_binding, build as ub_build
    from kvip.runtime import build

    calls = []
    tensor = SimpleNamespace(view=lambda *args: tensor)
    inputs = SimpleNamespace(query_ends=tensor, row_owners=tensor)

    class Workspace:
        def __init__(self, *args, **kwargs):
            self.inputs, self.scratch, self.tail_bounds, self.topk = inputs, object(), tensor, tensor

        def arguments(self, storage):
            return [tensor] * 14

        def allocate_rotation(self, *args, **kwargs):
            pass

    class Lease:
        sizes = {"status": 128}

        def __init__(self, *args):
            pass

        def register_table(self, table):
            pass

    class Binding:
        def __init__(self, *args, **kwargs):
            assert "threshold_bounds" not in kwargs
            calls.append(("binding", kwargs))
            self.library = SimpleNamespace()  # Upstream has no recent option getters.

    def build_urma(*, cache_dir, cann, headers, soc):
        calls.append(("urma_build", cache_dir))
        return (tmp_path / "upstream.so",)

    def build_ubmem(*, cache_dir, cann, shape):
        calls.append(("ubmem_build", cache_dir))
        return tmp_path / "upstream.so"

    monkeypatch.setattr(build, "build_libraries", build_urma)
    monkeypatch.setattr(build, "kvaf_build_paths", lambda unused: (tmp_path, tmp_path))
    monkeypatch.setattr(ub_build, "build_ubmem_library", build_ubmem)
    monkeypatch.setattr(workspace, "LIFetchWorkspace", Workspace)
    monkeypatch.setattr(workspace, "UbmemWorkspace", Workspace)
    monkeypatch.setattr(kvaf, "KvafGetLease", Lease)
    monkeypatch.setattr(kvaf, "UbmemLease", Lease)
    monkeypatch.setattr(kvaf, "KvafQueueView", lambda *args: object())
    monkeypatch.setattr(binding, "LIFetchBinding", Binding)
    monkeypatch.setattr(ub_binding, "UbmemBinding", Binding)
    monkeypatch.setattr(index_update, "DirectIndexUpdate", lambda *args, **kwargs: object())
    monkeypatch.setattr(index_update, "UbmemIndexUpdate", lambda *args, **kwargs: object())
    monkeypatch.setattr(forward, "LIFetchForward", lambda *args: object())
    storage = SimpleNamespace(keys=tensor, scales=tensor, addresses=tensor, logical_ids=tensor, hot_valid=tensor)
    monkeypatch.setattr(fused_replay, "IndexCacheStorage", lambda *args: storage)
    archive = SimpleNamespace(
        manifest={
            "geometry": dict(
                batch=16,
                history=131328,
                resident=16384,
                heads=32,
                token_bytes=656,
                aic_count=32,
                requested_windows=8,
                output_stride=16896,
                owners=17,
            )
        },
        tensor=lambda *args: tensor,
    )
    torch = SimpleNamespace(zeros=lambda *args, **kwargs: tensor, empty=lambda *args, **kwargs: tensor, int32="int32")
    bundle = fused_replay.create_fused(
        torch,
        object(),
        archive,
        {},
        SimpleNamespace(addresses=tensor, valid=tensor),
        tmp_path,
        "Ascend950DT_9582",
        object(),
        object(),
        tmp_path,
        32,
        transport=transport,
        upstream=True,
    )
    assert calls[0][0] == transport + "_build" and calls[1][0] == "binding"
    assert not any(
        bundle.tuning[name] for name in ("recent_threshold", "recent_pipeline", "recent_unified_topk", "recent_skip")
    )
    assert bundle.tuning["prefetch_limit"] == 512 and bundle.tuning["window_limit"] == 64


def _threshold_only_reports():
    reports = _reports()
    for name, tuning in replay_measure.study_cases(SimpleNamespace(study="threshold-only")).items():
        reports[f"urma-recent-{name}"]["tuning"] = tuning | {"prefetch_limit": 512, "window_limit": 64}
    return reports


def test_threshold_only_study_changes_one_flag_and_preserves_native_input_checks(capsys):
    reports = _threshold_only_reports()
    before, after = (reports[f"urma-recent-{name}"] for name in ("before", "after"))
    assert before["tuning"] | {"recent_threshold_only": True} == after["tuning"]
    assert before["tuning"]["recent_skip"] is False
    before["reads_samples"][0].update(prefetch=8, unused_prefetch=4, bytes=6560)
    after["reads_samples"][0].update(prefetch=4, unused_prefetch=1, bytes=3936)
    summary = compare.summarize(reports, "threshold-only")
    assert [row["display_name"] for row in summary["table"].values()] == [
        "urma-main-serial",
        "urma-main-resident",
        "urma-recent-recompute",
        "urma-recent-kth-only",
    ]
    effect = summary["effects"]["urma"]
    assert effect["optimization_reduction_pct"] == pytest.approx(10)
    assert effect["traffic_before_to_after"] == pytest.approx(
        dict(unused_prefetch_reduction_pct=75, total_read_bytes_reduction_pct=40)
    )
    assert summary["table"]["urma-recent-before"]["reads"]["prefetch_waste_pct"] == 50
    compare.print_summary(summary)
    printed = capsys.readouterr().out
    assert "recent-kth-only" in printed and "unused_prefetch_reduction_pct=75.00%" in printed
    assert "total_read_bytes_reduction_pct=40.00%" in printed and "waste%" in printed
    assert compare.traffic_change({"unused_prefetch": 0, "bytes": 0}, {"unused_prefetch": 0, "bytes": 0}) == {
        "unused_prefetch_reduction_pct": None,
        "total_read_bytes_reduction_pct": None,
    }


@pytest.mark.parametrize("field", ["recent_skip", "recent_pipeline", "recent_threshold_only", "window_limit"])
def test_threshold_only_summary_rejects_other_changes_or_missing_new_binary_flag(field):
    reports = _threshold_only_reports()
    reports["urma-recent-after"]["tuning"][field] = (
        32 if field == "window_limit" else not reports["urma-recent-after"]["tuning"][field]
    )
    with pytest.raises(ValueError, match="Threshold-only A/B"):
        compare.summarize(reports, "threshold-only")


def test_threshold_only_cli_runs_main_and_one_shared_arena_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(upstream, "validate_checkout", lambda root, revision: root)
    base = _arguments(
        tmp_path,
        ["--study", "threshold-only", "--upstream-root", str(tmp_path / "main"), "--upstream-revision", REVISION],
    )
    with pytest.raises(SystemExit):
        compare.arguments(base + ["--transports", "both"])
    with pytest.raises(SystemExit):
        compare.arguments(base + ["--diagnostic-replays", "-1"])
    args = compare.arguments(base + ["--diagnostic-replays", "1"])
    calls, reports = [], _threshold_only_reports()

    def child(command, **kwargs):
        calls.append(command)
        parsed = replay.arguments(command[4:])
        if parsed.upstream_root:
            assert parsed.study == "threshold" and parsed.threshold_policy == "resident"
            result = reports[f"urma-upstream-{parsed.mode}"]
        else:
            assert parsed.study == "threshold-only" and parsed.threshold_policy == "both"
            result = {"cases": {name: reports[f"urma-recent-{name}"] for name in ("before", "after")}}
        assert parsed.diagnostic_replays == int(parsed.mode == "fused")
        parsed.output.write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compare.subprocess, "run", child)
    monkeypatch.setattr(compare, "source_commit", lambda: "feature")
    summary = compare.run(args)
    assert len(calls) == 3 and len(summary["table"]) == 4
    assert summary["study"] == "threshold-only"


def test_graph_bandwidth_uses_complete_time_and_distinguishes_useful_payload(capsys):
    reports = _threshold_only_reports()
    source = reports["urma-recent-after"]
    source["reads_samples"] = [
        dict(source["reads_samples"][0], bytes=12000, useful_transfer_bytes=9000),
        dict(source["reads_samples"][0], bytes=18000, useful_transfer_bytes=9000),
    ]
    source["timing"]["mean_us"] = 100
    summary = compare.summarize(reports, "threshold-only")
    row = summary["table"]["urma-recent-after"]
    assert row["graph_total_GBps"] == row["graph_payload_GBps"] == pytest.approx(0.15)
    assert row["graph_useful_GBps"] == pytest.approx(0.09)
    assert row["payload_efficiency_pct"] == pytest.approx(60)
    assert row["hardware_bandwidth_utilization_pct"] is None
    compare.print_summary(summary)
    output = capsys.readouterr().out
    assert "total=0.150GB/s useful=0.090GB/s payload_efficiency=60.00% hardware_utilization=N/A" in output
    assert "not isolated Fetch bandwidth" in output
    for sample in source["reads_samples"]:
        sample.update(bytes=0, useful_transfer_bytes=0)
    empty = compare.summarize(reports, "threshold-only")["table"]["urma-recent-after"]
    assert empty["graph_total_GBps"] == 0 and empty["graph_useful_GBps"] == 0
    assert empty["payload_efficiency_pct"] is None


def _seed_reports():
    reports = {"urma-upstream-fused": _case(110, main=True)}
    for name, tuning in replay_measure.study_cases(SimpleNamespace(study="seed-ub")).items():
        reports[f"urma-recent-{name}"] = _case(
            {"current": 120, "release": 115, "seed": 100}[name],
            tuning=tuning | dict(prefetch_limit=512, window_limit=64),
        )
    for name in ("current", "release", "seed"):
        reports[f"urma-recent-{name}"]["seed_ub_eligibility"] = dict(
            predicted_eligible_queries=64, total_queries=64, predicted_fallback_queries=0
        )
    return reports


def test_seed_plan_separates_marker_removal_and_seed_reuse_without_serial(capsys):
    reports = _seed_reports()
    summary = compare.summarize(reports, "seed-ub")
    assert [row["display_name"] for row in summary["table"].values()] == [
        "urma-main-resident",
        "urma-current-kth",
        "urma-release-kth",
        "urma-seed-ub",
    ]
    effect = summary["effects"]["urma"]
    assert effect["main_reference"] == "urma-upstream-fused"
    assert "serial_reference" not in effect
    assert effect["marker_removal_reduction_pct"] == pytest.approx(100 * (1 - 115 / 120))
    assert effect["seed_reuse_reduction_pct"] == pytest.approx(100 * (1 - 100 / 115))
    assert effect["seed_vs_main_reduction_pct"] == pytest.approx(100 * (1 - 100 / 110))
    compare.print_summary(summary)
    text = capsys.readouterr().out
    assert "marker_removal=" in text and "seed_reuse=" in text and "MAIN_GAIN urma p50/mean" in text
    assert "vs_serial" not in text


@pytest.mark.parametrize(
    "change", ["score", "ordered", "initial", "final", "profile", "seed", "budget", "instrumented", "serial"]
)
def test_seed_study_rejects_changed_contracts(change):
    reports = _seed_reports()
    row = reports["urma-recent-seed"]
    if change == "score":
        row["score_comparison"]["status"] = "unverified"
    elif change == "ordered":
        row["selection_comparison"]["ordered_output_sha256"] = "other"
    elif change in ("initial", "final"):
        row["threshold_comparison"][change + "_keys"] = [[0]]
    elif change in ("profile", "seed"):
        row["tuning"]["recent_profile" if change == "profile" else "recent_seed_ub"] = change == "profile"
    elif change == "budget":
        row["tuning"]["window_limit"] = 32
    elif change == "instrumented":
        row["native_diagnostics"] = {"count": 1}
    else:
        reports["urma-upstream-serial"] = _case(100, main=True)
    with pytest.raises(ValueError):
        compare.summarize(reports, "seed-ub")


def test_seed_cli_runs_only_main_fused_and_feature_three_variants(tmp_path, monkeypatch):
    monkeypatch.setattr(upstream, "validate_checkout", lambda root, revision: root)
    base = _arguments(
        tmp_path, ["--study", "seed-ub", "--upstream-root", str(tmp_path / "main"), "--upstream-revision", REVISION]
    )
    with pytest.raises(SystemExit):
        compare.arguments(base + ["--diagnostic-replays", "1"])
    with pytest.raises(SystemExit):
        compare.arguments(base + ["--transports", "both"])
    args, calls, reports = compare.arguments(base), [], _seed_reports()

    def child(command, **kwargs):
        calls.append(command)
        parsed = replay.arguments(command[4:])
        assert parsed.mode == "fused" and parsed.diagnostic_replays == 0
        assert "--serial-workers" not in command
        if parsed.upstream_root:
            result = reports["urma-upstream-fused"]
        else:
            assert parsed.study == "seed-ub" and parsed.threshold_policy == "both"
            with pytest.raises(SystemExit):
                replay.arguments(command[4:] + ["--diagnostic-replays", "1"])
            result = {"cases": {name: reports[f"urma-recent-{name}"] for name in ("current", "release", "seed")}}
        parsed.output.write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compare.subprocess, "run", child)
    monkeypatch.setattr(compare, "source_commit", lambda: "feature")
    result = compare.run(args)
    assert len(calls) == 2 and len(result["table"]) == 4
    assert len(list(args.result_dir.glob("*.json"))) == 3


def test_seed_runtime_is_shared_and_all_three_full_oracles_are_required(monkeypatch):
    from kvip.diagnostics import seed_eligibility

    monkeypatch.setattr(
        seed_eligibility,
        "predict_seed_eligibility",
        lambda archive: dict(predicted_eligible_queries=64, total_queries=64, predicted_fallback_queries=0),
    )
    args = SimpleNamespace(study="seed-ub", mode="fused", transport="urma", threshold_policy="both")
    calls, context, retained = [], object(), {}
    monkeypatch.setattr(replay_measure, "allocate_context", lambda *args: calls.append("allocate") or context)
    monkeypatch.setattr(replay_measure, "release_context", lambda received: calls.append(("release", received)))

    def measure(received, archive, allocated, retained):
        assert allocated is context
        calls.append(dict(received.fused_tuning))
        return _case(100, tuning=dict(received.fused_tuning))

    monkeypatch.setattr(replay_measure, "measure_case", measure)
    result = replay_measure.run(args, object(), retained)
    assert list(result["cases"]) == ["current", "release", "seed"]
    assert calls == ["allocate", *replay_measure.study_cases(args).values(), ("release", context)]
    assert list(retained["completed_cases"]) == ["current", "release", "seed"]


def test_tail_skew_keeps_all_samples_and_reports_p50_and_mean_gains(capsys):
    reports = _seed_reports()
    # Nine typical runs and one delayed run reproduce the observed G17 mean
    # and median. This models a long tail without claiming its physical cause.
    raw = [465.604] * 9 + [2128.204]
    main = reports["urma-upstream-fused"]
    main["timing"] = dict(
        count=10, min_us=min(raw), p50_us=465.604, mean_us=sum(raw) / 10, max_us=max(raw), samples_us=list(raw)
    )
    release = reports["urma-recent-release"]
    release["timing"].update(mean_us=456.812, p50_us=456.063)
    summary = compare.summarize(reports, "seed-ub")
    stored = summary["table"]["urma-upstream-fused"]
    assert stored["timing"]["samples_us"] == raw
    assert stored["mean_us"] == pytest.approx(631.864)
    spread = stored["timing_spread"]
    assert spread["flagged"] and spread["high_tail_samples"] == 1
    assert spread["count"] == 10 and spread["max_us"] == 2128.204
    assert spread["mean_to_p50"] == pytest.approx(631.864 / 465.604)
    paired = summary["effects"]["urma"]["timing_comparisons"]["release"]
    assert paired["p50_reduction_pct"] == pytest.approx(100 * (1 - 456.063 / 465.604))
    assert paired["mean_reduction_pct"] == pytest.approx(100 * (1 - 456.812 / 631.864))
    assert paired["timing_spread_flagged"] is True
    compare.print_summary(summary)
    text = capsys.readouterr().out
    assert "release=+2.05%/+27.70%" in text
    assert "TIMING_NOTE tail_spread" in text and "all samples kept, not a significance test" in text
    line = next(line for line in text.splitlines() if line.startswith("urma-main-resident"))
    assert all(value in line for value in ("10", "465.604", "631.864", "2128.204"))


@pytest.mark.parametrize(
    "timing,expected",
    [
        (dict(p50_us=100, mean_us=101, min_us=98, max_us=104, count=10), False),
        (dict(p50_us=100, mean_us=120, min_us=98, max_us=140, count=10), True),
        (dict(p50_us=100, mean_us=105, min_us=98, max_us=160, count=10), True),
        (dict(p50_us=100, mean_us=100), False),
    ],
)
def test_tail_warning_uses_explicit_rules_without_inventing_missing_extrema(timing, expected):
    spread = compare.timing_spread(timing)
    assert spread["flagged"] is expected
    assert spread["high_tail_samples"] is None
    assert spread["count"] == timing.get("count")
    assert spread["max_us"] == timing.get("max_us")


def _local_reports():
    reports = {"urma-upstream-fused": _case(110, main=True)}
    for name, tuning in replay_measure.study_cases(SimpleNamespace(study="local-victims")).items():
        reports[f"urma-recent-{name}"] = _case(
            100 if name == "release" else 95, tuning=tuning | dict(prefetch_limit=512, window_limit=64)
        )
    return reports


def test_local_victims_changes_only_its_flag_and_uses_main_fusion_as_reference(capsys):
    cases = replay_measure.study_cases(SimpleNamespace(study="local-victims"))
    assert cases["release"] | {"recent_local_victims": True} == cases["local"]
    assert cases["release"]["recent_threshold_only"] is True
    assert all(cases["release"][key] is False for key in ("recent_skip", "recent_profile", "recent_seed_ub"))
    summary = compare.summarize(_local_reports(), "local-victims")
    assert [row["display_name"] for row in summary["table"].values()] == [
        "urma-main-resident",
        "urma-release-kth",
        "urma-local-victims",
    ]
    effect = summary["effects"]["urma"]
    assert "serial_reference" not in effect and "seed_ub_eligibility" not in effect
    assert effect["local_victims_reduction_pct"] == pytest.approx(5)
    compare.print_summary(summary)
    text = capsys.readouterr().out
    assert "MAIN_GAIN urma p50/mean release=" in text
    assert "STEP_GAIN urma p50/mean local_victims=+5.00%/+5.00%" in text
    assert "SEED_UB_ELIGIBILITY" not in text and "SERIAL_REFERENCE_GAIN" not in text


@pytest.mark.parametrize("change", ["score", "ordered", "initial", "final", "profile", "seed", "budget", "local"])
def test_local_victims_never_weakens_oracle_or_changes_another_option(change):
    reports = _local_reports()
    row = reports["urma-recent-local"]
    if change == "score":
        row["score_comparison"]["status"] = "unchecked"
    elif change == "ordered":
        row["selection_comparison"]["ordered_output_sha256"] = "other"
    elif change in ("initial", "final"):
        row["threshold_comparison"][change + "_keys"] = [[0]]
    else:
        field, value = {
            "profile": ("recent_profile", True),
            "seed": ("recent_seed_ub", True),
            "budget": ("window_limit", 32),
            "local": ("recent_local_victims", False),
        }[change]
        row["tuning"][field] = value
    with pytest.raises(ValueError):
        compare.summarize(reports, "local-victims")


def test_local_victims_cli_has_two_fused_processes_and_no_profile_or_serial(tmp_path, monkeypatch):
    monkeypatch.setattr(upstream, "validate_checkout", lambda root, revision: root)
    base = _arguments(
        tmp_path,
        ["--study", "local-victims", "--upstream-root", str(tmp_path / "main"), "--upstream-revision", REVISION],
    )
    for extra in (["--diagnostic-replays", "1"], ["--transports", "both"]):
        with pytest.raises(SystemExit):
            compare.arguments(base + extra)
    args, calls, reports = compare.arguments(base), [], _local_reports()

    def child(command, **kwargs):
        calls.append(command)
        parsed = replay.arguments(command[4:])
        assert parsed.mode == "fused" and parsed.diagnostic_replays == 0
        assert "--serial-workers" not in command
        if parsed.upstream_root:
            result = reports["urma-upstream-fused"]
        else:
            assert parsed.study == "local-victims" and parsed.threshold_policy == "both"
            with pytest.raises(SystemExit):
                replay.arguments(command[4:] + ["--diagnostic-replays", "1"])
            result = {"cases": {name: reports[f"urma-recent-{name}"] for name in ("release", "local")}}
        parsed.output.write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compare.subprocess, "run", child)
    monkeypatch.setattr(compare, "source_commit", lambda: "feature")
    result = compare.run(args)
    assert len(calls) == 2 and len(result["table"]) == 3
    assert len(list(args.result_dir.glob("*.json"))) == 3


def test_local_victims_exports_are_required_and_match_native_settings():
    from kvip.diagnostics.fused_replay import validate_fused_tuning, verify_build_options

    options = dict(recent_threshold_only=True, recent_local_victims=True)
    assert validate_fused_tuning(options, "urma") == options
    with pytest.raises(ValueError, match="only by URMA"):
        validate_fused_tuning(options, "ubmem")
    with pytest.raises(ValueError, match="requires recent_threshold_only"):
        validate_fused_tuning(dict(recent_local_victims=True), "urma")
    with pytest.raises(AttributeError, match="kvip_recent_local_victims"):
        verify_build_options(SimpleNamespace(), {"recent_local_victims": True})
    with pytest.raises(RuntimeError, match="recent_local_victims=True"):
        verify_build_options(SimpleNamespace(kvip_recent_local_victims=lambda: 0), {"recent_local_victims": True})
