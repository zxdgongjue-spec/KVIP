# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the two LI-only Varlen CLI admission gates.

The public entry point must admit H131072/R8192 and forward its arguments
to the internal URMA worker. No NPU process or KVAF server is launched.
"""
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture(scope="module")
def lifetch_cli():
    path = Path(__file__).with_name("test_lifetch.py")
    spec = importlib.util.spec_from_file_location("_lifetch_gate_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cli_args(history=131072, resident=8192, prefetch=64, *extra):
    return [
        "--transport", "urma",
        "--kvaf-server", "127.0.0.1:28999",
        "--kvaf-root", "/tmp/nonexistent-kvaf",  # no connection is opened
        "--soc", "Ascend950DT_9582",
        "--batches", "4",
        "--history", str(history),
        "--resident", str(resident),
        "--index-heads", "32",
        "--miss-percent", "0",
        "--prefetch-tokens", str(prefetch),
        "--intermediate-fetches", "8",
        "--li-only-varlen",
        *extra,
    ]


@pytest.mark.parametrize("prefetch", [0, 64])
def test_public_gate_admits_h131k_r8192_and_forwards_to_worker(lifetch_cli, monkeypatch, prefetch):
    commands = []
    monkeypatch.setattr(
        lifetch_cli.subprocess, "call",
        lambda command: commands.append(command) or 0,
    )
    args = cli_args(prefetch=prefetch)
    if prefetch == 64:
        args += ["--epoch-audit-profile"]
    assert lifetch_cli.main(args) == 0
    assert len(commands) == 1
    child = commands[0]
    assert "--internal-urma" in child
    assert child[child.index("--history") + 1] == "131072"
    assert child[child.index("--resident") + 1] == "8192"
    assert child[child.index("--prefetch-tokens") + 1] == str(prefetch)
    if prefetch == 64:
        assert "--epoch-audit-profile" in child


@pytest.mark.parametrize("prefetch", [0, 64])
def test_internal_gate_admits_same_geometry(lifetch_cli, prefetch):
    args = cli_args(prefetch=prefetch)
    args.remove("--transport")
    args.remove("urma")
    if prefetch == 64:
        args += ["--epoch-audit-profile"]
    worker = lifetch_cli.arguments(args)
    assert worker.history == 131072
    assert worker.resident == 8192
    assert worker.li_only_varlen
    assert worker.prefetch_tokens == prefetch


def test_p1_c0_payload_fixture_does_not_expand_silently(lifetch_cli):
    with pytest.raises(SystemExit) as exc:
        lifetch_cli.main(cli_args(
            131072, 8192, 64, "--epoch-audit-profile", "--prefetch-payload-audit",
        ))
    assert exc.value.code == 2


def test_p1_c1_cold_probe_stays_h8192_r2048(lifetch_cli, capsys):
    # R=2048 must pass the P1-C0 guard and reach the P1-C1 H=8192 guard.
    with pytest.raises(SystemExit) as exc:
        lifetch_cli.main(cli_args(
            131072, 2048, 64, "--epoch-audit-profile", "--prefetch-payload-audit",
            "--cold-prefetch-probe", "--miss-percent", str(100 / 2048),
        ))
    assert exc.value.code == 2
    assert "P1-C1 cold-prefetch probe retains its H8192/R2048 fixture" in capsys.readouterr().err


@pytest.mark.parametrize("mode,expected", [("auto", False), ("q4", True)])
def test_li_tiling_selection_is_forwarded_only_for_q4(lifetch_cli, monkeypatch, mode, expected):
    commands = []
    monkeypatch.setattr(lifetch_cli.subprocess, "call", lambda command: commands.append(command) or 0)
    args = cli_args(prefetch=0) + ["--li-tiling", mode]
    assert lifetch_cli.main(args) == 0
    child = commands[0]
    assert ("--li-tiling" in child) is expected
    if expected:
        assert child[child.index("--li-tiling") + 1] == "q4"
    internal_args = args[2:]
    assert lifetch_cli.arguments(internal_args).li_tiling == mode


@pytest.mark.parametrize("prefetch", [64, 512])
def test_q4_rejects_prefetch_to_isolate_li_kernel(lifetch_cli, prefetch):
    with pytest.raises(SystemExit) as exc:
        lifetch_cli.main(cli_args(prefetch=prefetch) + ["--li-tiling", "q4"])
    assert exc.value.code == 2


def test_q4_rejects_non_li_only_scope(lifetch_cli):
    args = cli_args(prefetch=0) + ["--li-tiling", "q4"]
    args.remove("--li-only-varlen")
    with pytest.raises(SystemExit) as exc:
        lifetch_cli.main(args)
    assert exc.value.code == 2


@pytest.mark.parametrize("tiling", ["auto", "q4"])
def test_li_core_sweep_forwarded_in_both_cli_layers(lifetch_cli, monkeypatch, tiling):
    commands = []
    monkeypatch.setattr(lifetch_cli.subprocess, "call", lambda command: commands.append(command) or 0)
    args = cli_args(prefetch=0) + ["--li-core-sweep", "--li-tiling", tiling]
    assert lifetch_cli.main(args) == 0
    assert "--li-core-sweep" in commands[0]
    internal = lifetch_cli.arguments(args[2:])
    assert internal.li_core_sweep is True
    assert internal.li_tiling == tiling


def test_li_core_sweep_requires_prefetch_zero(lifetch_cli):
    with pytest.raises(SystemExit) as exc:
        lifetch_cli.main(cli_args(prefetch=64) + ["--li-core-sweep"])
    assert exc.value.code == 2


def test_li_core_sweep_scenarios_are_opt_in():
    path = Path(__file__).with_name("support") / "lifetch.py"
    source = path.read_text()
    for name in ("balanced_full_q4", "mixed_k_full_q4", "extreme_k_full_q4"):
        assert name in source
    assert 'if getattr(args, "li_core_sweep", False):' in source


@pytest.mark.parametrize("tiling", ["auto", "q4"])
def test_li_core_perf_forwarded_only_for_prefetch_zero(lifetch_cli, monkeypatch, tiling):
    commands = []
    monkeypatch.setattr(lifetch_cli.subprocess, "call", lambda command: commands.append(command) or 0)
    args = cli_args(prefetch=0) + ["--li-core-perf", "--li-tiling", tiling]
    assert lifetch_cli.main(args) == 0
    assert "--li-core-perf" in commands[0]
    worker = lifetch_cli.arguments(args[2:])
    assert worker.li_core_perf
    assert worker.li_tiling == tiling


def test_li_core_perf_requires_prefetch_zero(lifetch_cli):
    with pytest.raises(SystemExit) as exc:
        lifetch_cli.main(cli_args(prefetch=64) + ["--li-core-perf"])
    assert exc.value.code == 2


def test_li_core_perf_rejects_full_chain(lifetch_cli):
    args = cli_args(prefetch=0) + ["--li-core-perf"]
    args.remove("--li-only-varlen")
    with pytest.raises(SystemExit) as exc:
        lifetch_cli.main(args)
    assert exc.value.code == 2


def test_li_core_perf_capture_isolated_from_v9_init_and_wait():
    source = (Path(__file__).with_name("support") / "lifetch.py").read_text()
    anchor = source.index('if getattr(args, "li_core_perf", False):')
    begin = source.index("with torch.npu.graph(graph, stream=capture_stream):", anchor)
    end = source.index("capture_stream.synchronize()", begin)
    capture_body = source[begin:end]
    assert 'rotating._call("li")' in capture_body
    assert 'rotating._call("init")' not in capture_body
    assert 'rotating._call("wait")' not in capture_body


@pytest.mark.parametrize("repeats", [1, 8, 16, 32])
def test_li_core_perf_graph_repeats_forwarded(lifetch_cli, monkeypatch, repeats):
    commands = []
    monkeypatch.setattr(lifetch_cli.subprocess, "call", lambda command: commands.append(command) or 0)
    args = cli_args(prefetch=0) + [
        "--li-core-perf", "--li-core-perf-graph-repeats", str(repeats)
    ]
    assert lifetch_cli.main(args) == 0
    cmd = commands[0]
    assert cmd[cmd.index("--li-core-perf-graph-repeats") + 1] == str(repeats)
    worker = lifetch_cli.arguments(args[2:])
    assert worker.li_core_perf_graph_repeats == repeats


@pytest.mark.parametrize("repeats", [0, 65])
def test_li_core_perf_graph_repeats_reject_out_of_range(lifetch_cli, repeats):
    with pytest.raises(SystemExit) as exc:
        lifetch_cli.main(
            cli_args(prefetch=0) +
            ["--li-core-perf", "--li-core-perf-graph-repeats", str(repeats)]
        )
    assert exc.value.code == 2


def test_li_core_perf_graph_repeats_capture_scope():
    source = (Path(__file__).with_name("support") / "lifetch.py").read_text()
    anchor = source.index('if getattr(args, "li_core_perf", False):')
    begin = source.index("with torch.npu.graph(graph, stream=capture_stream):", anchor)
    end = source.index("capture_stream.synchronize()", begin)
    capture = source[begin:end]
    assert "for _ in range(graph_repeats)" in capture
    assert 'rotating._call("li")' in capture
    assert 'rotating._call("init")' not in capture
    assert 'rotating._call("wait")' not in capture
    assert "elapsed_us / graph_repeats" in source[anchor:]


def test_li_q2_trunk_control_forwards_and_admits_auto(lifetch_cli, monkeypatch):
    commands = []
    monkeypatch.setattr(lifetch_cli.subprocess, "call", lambda cmd: commands.append(cmd) or 0)
    args = cli_args(prefetch=0) + ["--li-core-perf", "--li-tiling", "auto",
                                   "--li-q2-trunk8k-control"]
    assert lifetch_cli.main(args) == 0
    assert "--li-q2-trunk8k-control" in commands[0]
    worker = lifetch_cli.arguments(args[2:])
    assert worker.li_q2_trunk8k_control is True


def test_li_q2_trunk_control_rejects_q4(lifetch_cli):
    with pytest.raises(SystemExit) as exc:
        lifetch_cli.main(cli_args(prefetch=0) + [
            "--li-core-perf", "--li-tiling", "q4", "--li-q2-trunk8k-control"
        ])
    assert exc.value.code == 2


def test_li_q2_trunk_control_rejects_prefetch(lifetch_cli):
    with pytest.raises(SystemExit) as exc:
        lifetch_cli.main(cli_args(prefetch=64) + [
            "--li-core-perf", "--li-q2-trunk8k-control"
        ])
    assert exc.value.code == 2


def test_li_q2_trunk_control_rejects_without_perf(lifetch_cli):
    with pytest.raises(SystemExit) as exc:
        lifetch_cli.main(cli_args(prefetch=0) + ["--li-q2-trunk8k-control"])
    assert exc.value.code == 2


def test_li_q2_trunk_control_compile_flag_is_opt_in():
    root = Path(__file__).resolve().parents[3]
    cmake_path = root / "csrc/operators/lightning_indexer_with_fetch/CMakeLists.txt"
    kernel_path = root / "csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/indexer_vector.h"
    cmake_source = cmake_path.read_text()
    kernel_source = kernel_path.read_text()
    assert 'set(V9_Q2_TRUNK8K_CONTROL 0 CACHE STRING' in cmake_source
    assert '-DV9_Q2_TRUNK8K_CONTROL=${V9_Q2_TRUNK8K_CONTROL}' in cmake_source
    assert '#if defined(V9_Q2_TRUNK8K_CONTROL) && V9_Q2_TRUNK8K_CONTROL' in kernel_source
    assert 'trunkLen_ = s1BaseSize_ == 2 ? TRUNK_LEN_16K : 8192;' in kernel_source


@pytest.mark.parametrize("tiling", ["auto", "q4"])
def test_li_s2_slope_sweep_forwarded_to_worker(lifetch_cli, monkeypatch, tiling):
    commands = []
    monkeypatch.setattr(lifetch_cli.subprocess, "call", lambda cmd: commands.append(cmd) or 0)
    args = cli_args(prefetch=0) + [
        "--li-s2-slope-sweep", "--li-core-perf", "--li-tiling", tiling
    ]
    assert lifetch_cli.main(args) == 0
    assert "--li-s2-slope-sweep" in commands[0]
    internal = lifetch_cli.arguments(args[2:])
    assert internal.li_s2_slope_sweep is True
    assert internal.li_tiling == tiling


@pytest.mark.parametrize("extra", [
    [],
    ["--li-core-perf", "--li-core-sweep"],
    ["--li-core-perf", "--history", "98304"],
])
def test_li_s2_slope_sweep_rejects_nonisolated_scope(lifetch_cli, extra):
    with pytest.raises(SystemExit) as exc:
        lifetch_cli.main(cli_args(prefetch=0) + ["--li-s2-slope-sweep"] + extra)
    assert exc.value.code == 2


def test_li_s2_slope_sweep_needs_prefetch_zero(lifetch_cli):
    with pytest.raises(SystemExit) as exc:
        lifetch_cli.main(
            cli_args(prefetch=64) + ["--li-s2-slope-sweep", "--li-core-perf"]
        )
    assert exc.value.code == 2


def test_li_s2_slope_scenarios_have_q1_and_q4_and_five_k_lengths():
    source = (Path(__file__).with_name("support") / "lifetch.py").read_text()
    assert 'if getattr(args, "li_s2_slope_sweep", False):' in source
    assert 'for qcount in (1, 4)' in source
    assert 'for key_len in (resident, quarter, half, three_quarter, history)' in source
    assert '"query_counts": (qcount, 0, 0, 0)' in source


@pytest.mark.parametrize("cores", [1, 2, 4])
def test_s2_score_probe_cli_and_internal_worker(lifetch_cli, monkeypatch, cores):
    commands = []
    monkeypatch.setattr(lifetch_cli.subprocess, "call", lambda cmd: commands.append(cmd) or 0)
    args = cli_args(prefetch=0) + [
        "--li-s2-score-probe", "--li-s2-score-cores", str(cores),
        "--li-core-perf", "--li-tiling", "q4",
    ]
    assert lifetch_cli.main(args) == 0
    assert "--li-s2-score-probe" in commands[0]
    assert commands[0][commands[0].index("--li-s2-score-cores") + 1] == str(cores)
    worker = lifetch_cli.arguments(args[2:])
    assert worker.li_s2_score_probe
    assert worker.li_s2_score_cores == cores


@pytest.mark.parametrize("extra", [
    [],
    ["--li-core-perf"],
    ["--li-core-perf", "--li-tiling", "q4", "--li-core-sweep"],
    ["--li-core-perf", "--li-tiling", "q4", "--li-s2-slope-sweep"],
    ["--li-core-perf", "--li-tiling", "q4", "--epoch-audit-profile"],
])
def test_s2_score_probe_rejects_incompatible_modes(lifetch_cli, extra):
    with pytest.raises(SystemExit) as exc:
        lifetch_cli.main(cli_args(prefetch=0) + ["--li-s2-score-probe"] + extra)
    assert exc.value.code == 2


def test_s2_score_probe_rejects_nonzero_prefetch(lifetch_cli):
    with pytest.raises(SystemExit) as exc:
        lifetch_cli.main(cli_args(prefetch=64) + [
            "--li-s2-score-probe", "--li-core-perf", "--li-tiling", "q4",
        ])
    assert exc.value.code == 2


def test_s2_score_probe_kernel_is_separate_opt_in_entry():
    root = Path(__file__).resolve().parents[3]
    kernel = (root / "csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer_s2_score.cpp").read_text()
    cmake = (root / "csrc/operators/lightning_indexer_with_fetch/CMakeLists.txt").read_text()
    host = (root / "csrc/operators/lightning_indexer_with_fetch/op_host/launch.cpp").read_text()
    harness = (root / "tests/operators/lifetch/support/s2_score_device.py").read_text()
    assert 'void v9_s2_score_kernel' in kernel
    assert 'op.SetScoreOnlyS2(true);' in kernel
    assert 'op.ConfigureSingleQueryScoreShard(groups);' in kernel
    assert 'if(V9_S2_SCORE_PROBE)' in cmake
    assert 'aclrtlaunch_v9_s2_score_kernel' in host
    assert 'shape->history != 131072' in host
    assert 'raw bits mismatch' in harness
    assert '_stitch(' in harness
