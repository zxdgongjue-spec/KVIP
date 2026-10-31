"""Run the existing standalone benchmark as a bounded correctness demo.

host runs the CPU contract model; npu-memory runs marked device-memory tests;
kvaf uses real communication. Graph means the benchmark's single-output
NPUGraph, not the serving framework's GraphFrame begin/end contract.
Dry-run reports source evidence only and never certifies a runtime or hardware.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEMO_HISTORY = 16385
DEMO_RESIDENT = 12288
DEMO_TAIL = 129
DEVICE_HEADER = Path("access_operator/demo/aiv_batch_get/include/kvaf_batch_get_device.h")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_provenance(root):
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain=v1"], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None, "scope": "source-path-and-sha-only"}
    return {"revision": revision, "dirty": bool(status.strip())}


def local_provider(root):
    """Load this checkout explicitly, regardless of inherited PYTHONPATH/wheels."""
    sys.path.insert(0, str(root))
    package = importlib.import_module("kvip")
    if Path(package.__file__).resolve() != (root / "kvip/__init__.py").resolve():
        raise ValueError("KVIP was already imported from a different provider")
    return package, importlib.import_module("kvip.v9")


def kvaf_source(args):
    root = args.kvaf_root.resolve()
    if not root.is_dir():
        raise ValueError(f"KVAF directory does not exist: {root}")
    header = root / DEVICE_HEADER
    if not header.is_file():
        raise ValueError(f"KVAF device header is missing: {header}")
    if args.profile == "kvaf":
        from kvip.v9.kvaf_compatibility import source_evidence

        evidence = source_evidence(root, expected_revision=args.expected_kvaf_revision, profile="v9-eager")
    else:
        evidence = {
            "profile": "device-headers-only",
            "git": git_provenance(root),
            "runtime_surface": "not_required_for_npu_memory_transport",
        }
        if args.expected_kvaf_revision is not None and evidence["git"]["revision"] != args.expected_kvaf_revision:
            raise ValueError("KVAF revision mismatch for device-header source (expected full commit SHA)")
    return {
        "root": str(root),
        "device_header": str(header),
        "device_header_sha256": sha256(header),
        "source_evidence": evidence,
        "runtime_certified": False,
        "graph_scope": "standalone-single-output-NPUGraph" if args.execution == "graph" else "eager",
        "framework_graph_frame_required": False,
    }


def build_plan(root, args):
    benchmark = root / "tests/v9/benchmark_contract.py"
    if not benchmark.is_file():
        raise ValueError(f"standalone benchmark source is missing: {benchmark}; use a complete KVIP checkout")
    package, provider = local_provider(root)
    source = provider.native_source().resolve()
    if not source.is_relative_to(root):
        raise ValueError("demo operator source must belong to the selected KVIP checkout")
    hashes = {
        path.relative_to(source).as_posix(): sha256(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
        and path.suffix in (".h", ".inc", ".cpp", ".py", ".txt", ".json")
        and "__pycache__" not in path.parts
    }
    output = (
        args.output.resolve() if args.output is not None else root / "results/demo" / f"{args.profile}-{args.execution}"
    )
    command = [
        sys.executable,
        str(benchmark),
        "--backend",
        "cpu" if args.profile == "host" else "npu",
        "--transport",
        "npu-memory" if args.profile == "npu-memory" else "kvaf",
        "--execution",
        args.execution,
        "--case-suite",
        args.case_suite,
        "--batches",
        "1",
        "--queries",
        "4",
        "--history",
        str(DEMO_HISTORY),
        "--resident",
        str(DEMO_RESIDENT),
        "--tail",
        str(DEMO_TAIL),
        "--layers",
        "1",
        "--steps",
        "1",
        "--intermediate-fetches",
        "4",
        "--intermediate-profiling",
        "off",
        "--native-baseline",
        "off",
        "--qsfa",
        "off",
        "--warmup",
        "1",
        "--iterations",
        "1",
        "--output",
        str(output),
    ]
    external = None
    if args.profile != "host":
        external = kvaf_source(args)
        command.extend(
            (
                "--kvaf-root",
                external["root"],
                "--cann",
                str(args.cann.resolve()),
                "--soc",
                args.soc,
                "--device",
                str(args.device),
            )
        )
        if args.profile == "kvaf":
            command.extend(("--kvaf-server", args.kvaf_server))
    capability = provider.query_capabilities()
    return {
        "profile": args.profile,
        "execution": args.execution,
        "case_suite": args.case_suite,
        "command": command,
        "output": str(output),
        "kvip_root": str(root),
        "kvip": {
            "package_version": package.__version__,
            "contract_version": capability.contract_version,
            **git_provenance(root),
        },
        "benchmark": {"path": str(benchmark), "sha256": sha256(benchmark)},
        "operator_sources": {"path": str(source), "sha256": hashes},
        "kvaf": external,
        "qualification": {
            "hardware_certified": capability.hardware_certified,
            "runtime_validated": False,
            "runtime_surface_checked": False,
            "serving_framework_graph_frame": "not_tested_by_this_demo",
            "hardware_function_and_performance": "not_qualified_by_this_demo",
            "host_timing": "not_hardware_performance" if args.profile == "host" else "not_applicable",
            "npu_memory_timing": "not_network_performance" if args.profile == "npu-memory" else "not_applicable",
        },
    }


def require_npu(args):
    cann = args.cann.resolve()
    compiler = cann / "compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake"
    libraries = (cann / "lib64/libascendcl.so", cann / "aarch64-linux/lib64/libascendcl.so")
    if not compiler.is_file() or not any(path.is_file() for path in libraries):
        raise ValueError(f"CANN SDK/compiler/runtime files are missing under {cann}; NPU execution cannot be skipped")
    if shutil.which("cmake") is None:
        raise ValueError("cmake is required for the NPU demo")
    try:
        torch = importlib.import_module("torch")
        importlib.import_module("torch_npu")
        if not torch.npu.is_available() or args.device < 0 or args.device >= torch.npu.device_count():
            raise ValueError(f"NPU device {args.device} is unavailable")
    except (ImportError, AttributeError, RuntimeError) as error:
        raise ValueError("NPU demo requires usable Torch NPU/torch_npu; no CPU fallback is allowed") from error


def require_kvaf_runtime(root, args):
    path = root / "tests/v9/aiv1_fetch_support.py"
    spec = importlib.util.spec_from_file_location("kvip_demo_fetch_support", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    config = helper.discover(server=args.kvaf_server, kvaf_root=args.kvaf_root)
    python_dir = config["kvaf_python_dir"]
    sys.path.insert(0, str(python_dir))
    runtime = importlib.import_module("kvaf")
    if not Path(runtime.__file__).resolve().is_relative_to(python_dir):
        raise ValueError("KVAF runtime was imported from a different build bundle")
    from kvip.v9.kvaf_compatibility import validate_runtime_surface

    surface = validate_runtime_surface(runtime.NpuRuntime, profile="v9-eager")
    return {"module_path": str(Path(runtime.__file__).resolve()), "surface": surface}


def run_child(command, root):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    return subprocess.run(command, cwd=root, env=environment).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("host", "npu-memory", "kvaf"), default="host")
    parser.add_argument("--execution", choices=("eager", "graph"), default="eager")
    parser.add_argument("--case-suite", choices=("contract", "full"), default="contract")
    parser.add_argument("--kvaf-root", type=Path)
    parser.add_argument("--expected-kvaf-revision")
    parser.add_argument("--kvaf-server")
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann"))
    parser.add_argument("--soc", default="Ascend950DT_9574")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.profile == "host" and args.execution == "graph":
        parser.error("host cannot validate NPUGraph; select an NPU profile for graph execution")
    if args.profile != "host" and args.kvaf_root is None:
        parser.error("NPU profiles require an explicit external --kvaf-root")
    if args.profile == "kvaf" and not args.kvaf_server:
        parser.error("kvaf profile requires --kvaf-server, including a complete dry-run command")
    if args.profile == "host" and (
        args.kvaf_root is not None or args.expected_kvaf_revision is not None or args.kvaf_server is not None
    ):
        parser.error("host profile does not use KVAF arguments")
    root = Path(__file__).resolve().parents[1]
    try:
        plan = build_plan(root, args)
        if not args.dry_run and args.profile != "host":
            require_npu(args)
            if args.profile == "kvaf":
                plan["kvaf"]["runtime_surface_evidence"] = require_kvaf_runtime(root, args)
                plan["qualification"]["runtime_surface_checked"] = True
        print(json.dumps(plan, indent=2), flush=True)
        if not args.dry_run:
            raise SystemExit(run_child(plan["command"], root))
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
