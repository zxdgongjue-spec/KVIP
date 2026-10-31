"""Compare unchanged device/oracle sources and deterministic host outputs to Git base."""

import argparse
import ast
import json
import os
import subprocess
import sys
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASE)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["git", "diff", "--exit-code", args.baseline, "--", *PROTECTED],
        cwd=root,
        check=True,
    )
    previous_launch = subprocess.check_output(
        ["git", "show", f"{args.baseline}:kvop/v9/launch.py"],
        cwd=root,
        text=True,
    )

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

    if execution_methods(previous_launch) != execution_methods(
        (root / "kvop/v9/launch.py").read_text()
    ):
        raise SystemExit("low-level launch/completion execution changed")
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "operators"],
        cwd=root,
        text=True,
    )
    if untracked.strip():
        raise SystemExit("untracked device files would invalidate source preservation")
    with tempfile.TemporaryDirectory(prefix="kvop-baseline-") as temporary:
        archive = subprocess.Popen(
            ["git", "archive", args.baseline], cwd=root, stdout=subprocess.PIPE
        )
        subprocess.run(["tar", "-x", "-C", temporary], stdin=archive.stdout, check=True)
        archive.stdout.close()
        if archive.wait() != 0:
            raise SystemExit("baseline archive failed")
        expected, actual = oracle(Path(temporary)), oracle(root)
    mismatches = [name for name in expected if expected[name] != actual.get(name)]
    if mismatches or set(expected) != set(actual):
        raise SystemExit(f"host numerical difference: {mismatches}")
    print(
        json.dumps(
            {
                "baseline": args.baseline,
                "device_and_oracle_sources": "unchanged",
                "build_geometry_packing_sources": "unchanged",
                "low_level_launch_completion_ast": "unchanged",
                "identical_host_cases": len(actual),
                "npu_network_graph_performance": "not_verified",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
