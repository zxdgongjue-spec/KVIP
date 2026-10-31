"""Run explicit host or complete CPU-Torch conformance; neither certifies hardware.

host deselects Torch-dependent tests before execution. cpu requires Torch and
the critical suites, and treats any collection/setup/call skip as failure.
Installed wheels can be tested with an external tests tree and --provider installed.
"""

import argparse
import ast
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CRITICAL_SUITES = frozenset(
    {
        "test_package.py",
        "test_public_contract.py",
        "test_public_session.py",
        "test_graph_session.py",
        "test_consumer_fake.py",
        "test_owned_boundary.py",
        "test_transport_contract.py",
        "test_qualification_gate.py",
        "test_inputs.py",
        "test_launcher_binding.py",
        "test_conformance_tools.py",
    }
)


class HostOnlyImports:
    """Keep host coverage independent of whether Torch happens to be installed."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("host conformance excludes Torch; use --tier cpu")
        return None


def require_cpu_torch(importer=importlib.import_module):
    try:
        torch = importer("torch")
        value = torch.zeros(1, dtype=torch.float8_e4m3fn, device="cpu")
        if value.device.type != "cpu":
            raise ValueError("Torch CPU allocation failed")
        return torch.__version__
    except (ImportError, AttributeError, RuntimeError, ValueError) as error:
        raise ValueError(
            "complete cpu conformance requires CPU-capable Torch with FP8 types; install .[cpu-test]"
        ) from error


def torch_call(node):
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return any(
            name.name == "torch" or name.name.startswith("torch.")
            for name in node.names
        ) or (
            isinstance(node, ast.ImportFrom) and (node.module or "").startswith("torch")
        )
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "importorskip"
        and (
            bool(node.args)
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "torch"
        )
    )


def host_exclusions(tests):
    ignored, deselected = [], []
    for path in sorted(tests.glob("test_*.py")):
        tree = ast.parse(path.read_text())
        functions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        dependent = [
            node
            for node in functions
            if any(torch_call(child) for child in ast.walk(node))
        ]
        top_level = [
            node
            for node in tree.body
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if any(
            torch_call(child) for node in top_level for child in ast.walk(node)
        ) or any(not node.name.startswith("test_") for node in dependent):
            ignored.append(path)
        else:
            deselected.extend((path.resolve(), node.name) for node in dependent)
    return ignored, deselected


class CompleteRun:
    def __init__(self, excluded_functions=()):
        self.collected = set()
        self.skipped = []
        self.excluded_functions = set(excluded_functions)
        self.deselected = []

    def pytest_collection_modifyitems(self, config, items):
        selected = []
        for item in items:
            function = getattr(item, "originalname", None) or item.name.split("[", 1)[0]
            key = (Path(str(item.path)).resolve(), function)
            if key in self.excluded_functions:
                self.deselected.append(item)
            else:
                selected.append(item)
        if self.deselected:
            config.hook.pytest_deselected(items=self.deselected)
            items[:] = selected
        self.collected.update(Path(str(item.path)).name for item in items)

    def pytest_collectreport(self, report):
        if report.skipped:
            self.skipped.append(report.nodeid)

    def pytest_runtest_logreport(self, report):
        if report.skipped:
            self.skipped.append(report.nodeid)

    def validate(self, tier):
        if self.skipped:
            raise ValueError(
                f"conformance refuses skipped checks: {', '.join(self.skipped)}"
            )
        missing = CRITICAL_SUITES - self.collected if tier == "cpu" else set()
        if missing:
            raise ValueError(
                f"missing critical conformance suites: {', '.join(sorted(missing))}"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("host", "cpu"), default="cpu")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--provider", choices=("source", "installed"), default="source")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    tests = root / "tests/v9"
    if not tests.is_dir():
        raise SystemExit(
            "a complete external/source tests/v9 tree is required; installed-package scope alone is not conformance"
        )
    # A clean subprocess prevents invoking cwd/source imports from accidentally
    # satisfying an installed-wheel check. Source and installed are explicit.
    if not args._worker:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root) if args.provider == "source" else ""
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--tier",
            args.tier,
            "--root",
            str(root),
            "--provider",
            args.provider,
            "--_worker",
        ]
        raise SystemExit(subprocess.run(command, cwd=tests, env=env).returncode)
    try:
        if args.tier == "host":
            sys.meta_path.insert(0, HostOnlyImports())
        import pytest

        import kvop

        provider = Path(kvop.__file__).resolve()
        if args.provider == "installed" and provider.is_relative_to(root / "kvop"):
            raise ValueError("installed provider resolves to the source checkout")
        version = require_cpu_torch() if args.tier == "cpu" else "not_required"
        compiler = shutil.which("c++") or shutil.which("g++")
        if args.tier == "cpu" and compiler is None:
            raise ValueError(
                "complete cpu conformance requires a C++ compiler for launcher ABI tests"
            )
        options = [str(tests), "-q", f"--confcutdir={tests}", "--import-mode=prepend"]
        ignored, deselected = (
            host_exclusions(tests) if args.tier == "host" else ([], [])
        )
        options.extend(f"--ignore={path}" for path in ignored)
        results = CompleteRun(deselected)
        status = pytest.main(options, plugins=[results])
        results.validate(args.tier)
        print(
            json.dumps(
                {
                    "tier": args.tier,
                    "provider": str(provider),
                    "torch": version,
                    "cpp_compiler": compiler,
                    "passed": status == 0,
                    "skipped": len(results.skipped),
                    "torch_excluded_modules": [path.name for path in ignored],
                    "torch_deselected_functions": len(deselected),
                    "torch_deselected_cases": len(results.deselected),
                    "hardware_function_and_performance": "not_verified",
                },
                indent=2,
            )
        )
    except (ImportError, ValueError) as error:
        raise SystemExit(str(error)) from error
    raise SystemExit(int(status))


if __name__ == "__main__":
    main()
