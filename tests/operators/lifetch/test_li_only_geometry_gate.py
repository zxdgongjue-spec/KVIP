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
