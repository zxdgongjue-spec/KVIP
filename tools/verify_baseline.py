"""Compare normalized source trees, or explicitly verify an installed package.

Source comparison requires Git history or --baseline-directory. Package scope
checks native source hashes/public discovery, not baseline or hardware behavior.
"""

import argparse
import ast
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

BASE = "2f49ba0e140311aa8efd40dbb539b1169b959747"
PROTECTED = (
    "operators",
    "kvop/v9/build.py",
    "kvop/v9/geometry.py",
    "kvop/v9/device_resources.py",
    "kvop/v9/inputs.py",
    "tests/v9/contract_reference.py",
    "tests/v9/contract_simulator.py",
    "tests/v9/contract_index_reference.py",
    "tests/v9/contract_payload.py",
)
ORACLE = """
import hashlib, json, numpy as np
from contract_fixture import CASES, make_fixture
from contract_simulator import Simulator, seed_output
result = {}
for batch in (1, 2):
    for windows in (0, 4):
        for case in CASES:
            g, a, scores = make_fixture(batch=batch, case=case)
            seed_output(a, g)
            model = Simulator(g, a)
            model.launch(scores, rounds=windows)
            digest = hashlib.sha256()
            for key in sorted(a):
                value = np.asarray(a[key])
                digest.update(key.encode())
                digest.update(str((value.shape, value.dtype)).encode())
                digest.update(value.tobytes())
            result[f'b{batch}-w{windows}-{case.name}'] = digest.hexdigest()
print(json.dumps(result, sort_keys=True))
"""


def oracle(root):
    env = dict(
        os.environ, PYTHONPATH=os.pathsep.join((str(root), str(root / "tests/v9")))
    )
    output = subprocess.check_output(
        [sys.executable, "-c", ORACLE], cwd=root, env=env, text=True
    )
    return json.loads(output)


def protected_sources(root):
    """Include untracked source additions; ignore generated Python/build output."""
    files = {}
    for name in PROTECTED:
        path = root / name
        if not path.exists():
            raise ValueError(f"missing protected source: {path}")
        candidates = path.rglob("*") if path.is_dir() else (path,)
        for source in candidates:
            relative = source.relative_to(root)
            if any(
                part == "__pycache__" or part == "build" or part.startswith("build_")
                for part in relative.parts
            ):
                continue
            if (
                source.is_file()
                and source.suffix not in (".pyc", ".pyo")
                and source.name != ".DS_Store"
            ):
                files[relative.as_posix()] = source.read_bytes()
    return files


def execution_methods(source):
    names = {
        "_call",
        "launch_compute",
        "wait",
        "launch_final_submit",
        "finish",
        "launch",
        "completion",
    }
    return {
        (cls.name, node.name): ast.dump(node, include_attributes=False)
        for cls in ast.parse(source).body
        if isinstance(cls, ast.ClassDef)
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }


def compare_sources(root, baseline_root):
    previous, current = protected_sources(baseline_root), protected_sources(root)
    changed = sorted(
        name
        for name in previous.keys() | current.keys()
        if previous.get(name) != current.get(name)
    )
    if changed:
        raise ValueError(f"protected source difference: {', '.join(changed)}")
    if execution_methods(
        (baseline_root / "kvop/v9/launch.py").read_text()
    ) != execution_methods((root / "kvop/v9/launch.py").read_text()):
        raise ValueError("low-level launch/completion execution changed")
    expected, actual = oracle(baseline_root), oracle(root)
    mismatches = sorted(
        name
        for name in expected.keys() | actual.keys()
        if expected.get(name) != actual.get(name)
    )
    if mismatches or not expected:
        raise ValueError(f"host numerical difference or empty oracle: {mismatches}")
    return {
        "scope": "source-baseline",
        "protected_source_files": len(previous),
        "device_and_oracle_sources": "unchanged",
        "build_geometry_packing_sources": "unchanged",
        "low_level_launch_completion_ast": "unchanged",
        "identical_host_cases": len(actual),
        "npu_network_graph_performance": "not_verified",
    }


def extract_git_baseline(root, revision, destination):
    """Export the independent root even when called from a framework subtree."""
    try:
        prefix = subprocess.check_output(
            ["git", "rev-parse", "--show-prefix"], cwd=root, text=True
        ).strip("/\n")
        git_root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], cwd=root, text=True
            ).strip()
        )
        candidates = ("", prefix) if prefix else ("",)
        for candidate in candidates:
            path = (
                f"{candidate}/kvop/v9/launch.py" if candidate else "kvop/v9/launch.py"
            )
            exists = subprocess.run(
                ["git", "cat-file", "-e", f"{revision}:{path}"],
                cwd=git_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if exists.returncode == 0:
                tree = f"{revision}:{candidate}" if candidate else revision
                archive = subprocess.check_output(
                    ["git", "archive", tree], cwd=git_root
                )
                break
        else:
            raise ValueError(
                f"baseline {revision} has no KVOP root or matching subtree"
            )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            "source-baseline requires Git/history; provide --baseline-directory for an exported baseline, "
            "or explicitly select --scope installed-package"
        ) from error
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for entry in bundle:
            target = destination / entry.name
            if (
                not target.resolve().is_relative_to(destination.resolve())
                or entry.issym()
                or entry.islnk()
            ):
                raise ValueError("baseline archive contains an unsafe path or link")
            if entry.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif entry.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bundle.extractfile(entry).read())
    return tree


def installed_package_checks():
    import kvop
    from kvop.v9 import ABI_VERSION, CONTRACT_VERSION, native_source, negotiate_contract
    from kvop.v9.api import (
        CONTRACT_MAJOR,
        CONTRACT_MINOR,
        GRAPH_TRANSPORT_ABI,
        INDEX_WRITER_ABI,
    )
    from kvop.v9.contract_info import describe_contract

    capability = negotiate_contract()
    description = describe_contract()
    expected = {
        "package_version": kvop.__version__,
        "contract_version": CONTRACT_VERSION,
        "contract_major": CONTRACT_MAJOR,
        "contract_minor": CONTRACT_MINOR,
        "device_abi": ABI_VERSION,
        "graph_transport_abi": GRAPH_TRANSPORT_ABI,
        "index_writer_abi": INDEX_WRITER_ABI,
    }
    if any(
        getattr(capability, name) != value or description.get(name) != value
        for name, value in expected.items()
    ):
        raise ValueError("installed public discovery/schema version mismatch")
    if tuple(map(int, CONTRACT_VERSION.split(".")[:2])) != (
        CONTRACT_MAJOR,
        CONTRACT_MINOR,
    ):
        raise ValueError("installed public contract semantic version mismatch")
    source = native_source()
    hashes = json.loads((source / "SOURCE.json").read_text()).get("sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("packaged source manifest has no hashes")
    for name, expected_hash in hashes.items():
        path = source / name
        if not path.resolve().is_relative_to(source.resolve()) or not path.is_file():
            raise ValueError(f"packaged source missing or unsafe: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"packaged source hash mismatch: {name}")
    return {
        "scope": "installed-package",
        "package": str(Path(kvop.__file__).resolve()),
        "public_contract": capability.contract_version,
        "schema_versions": expected,
        "packaged_native_source_hashes": len(hashes),
        "source_baseline_and_host_equivalence": "not_checked_in_this_scope",
        "npu_network_graph_performance": "not_verified",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASE)
    parser.add_argument("--baseline-directory", type=Path)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--scope",
        choices=("source-baseline", "installed-package"),
        default="source-baseline",
    )
    args = parser.parse_args(argv)
    try:
        if args.scope == "installed-package":
            if args.baseline_directory is not None:
                parser.error("--baseline-directory belongs to source-baseline scope")
            result = installed_package_checks()
        elif args.baseline_directory is not None:
            result = compare_sources(
                args.root.resolve(), args.baseline_directory.resolve()
            )
            result["baseline_directory"] = str(args.baseline_directory.resolve())
        else:
            with tempfile.TemporaryDirectory(prefix="kvop-baseline-") as temporary:
                baseline_root = Path(temporary)
                tree = extract_git_baseline(
                    args.root.resolve(), args.baseline, baseline_root
                )
                result = compare_sources(args.root.resolve(), baseline_root)
            result["baseline_tree"] = tree
    except (ValueError, OSError, ImportError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
