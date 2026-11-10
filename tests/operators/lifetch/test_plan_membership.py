# SPDX-License-Identifier: Apache-2.0
"""Execute both native planners against an independent ordered-set oracle."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_actual_planners_reuse_membership_without_changing_record_order(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for the native planner model")
    root = Path(__file__).resolve().parents[3]
    header = root / "csrc/operators/common/lifetch/index_cache_update.h"
    source = header.read_text()
    # Observe actual membership calls without replacing any planner logic.
    # Counting the entry calls catches reintroducing the redundant second pass.
    for name in ("PlanContains", "ExtendedContains"):
        begin = source.index("{", source.index(name + "(")) + 1
        source = source[:begin] + "\n  ++membershipQueries;" + source[begin:]
    (tmp_path / "planner_under_test.h").write_text(source)
    binary = tmp_path / "plan-membership"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            "-I" + str(tmp_path),
            str(Path(__file__).with_name("plan_membership_model.cpp")),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True, timeout=30)
