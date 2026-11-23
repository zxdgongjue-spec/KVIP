#!/usr/bin/env python3
"""Stage and check the P1 experiment on its pinned source in isolation."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY_BASE = "7b4e3d52d7a85451d1982a6e04823e4e18ecabfb"
PATCH = HERE / "p1-overlap.patch"
SHARED_GEOMETRY = (
    "csrc/operators/lightning_indexer_with_fetch/include/runtime_shape.h",
    "csrc/operators/common/lifetch/sequence_schedule.h",
)


def prepare(destination):
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Choose a new destination: {destination}")
    with TemporaryDirectory(prefix="kvip-p1-", dir=destination.parent) as directory:
        staged = Path(directory) / "source"
        staged.mkdir()
        archive = Path(directory) / "source.tar"
        subprocess.run(
            ["git", "archive", "--format=tar", f"--output={archive}", LEGACY_BASE],
            cwd=ROOT,
            check=True,
            timeout=60,
        )
        with tarfile.open(archive, "r") as source:
            members = source.getmembers()
            for member in members:
                name = PurePosixPath(member.name)
                if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unsupported archive entry: {member.name}")
            source.extractall(staged, members=members)
        # Establish an independent Git root so an output directory nested
        # under the caller's checkout cannot resolve that checkout's index.
        subprocess.run(["git", "init", "--quiet"], cwd=staged, check=True, timeout=20)
        subprocess.run(
            ["git", "apply", "--whitespace=error", str(PATCH)],
            cwd=staged,
            check=True,
            timeout=20,
        )
        staged.rename(destination)


def check(compiler):
    with TemporaryDirectory(prefix="kvip-p1-check-") as directory:
        area = Path(directory)
        source = area / "source"
        prepare(source)
        for name in SHARED_GEOMETRY:
            if (ROOT / name).read_bytes() != (source / name).read_bytes():
                raise RuntimeError(f"Main and the P1 baseline differ in {name}; recheck the schedule")
        executable = area / "schedule_check"
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(source / "experiments/lifetch_score_t2/p1_overlap_schedule_test.cpp"),
                "-o",
                str(executable),
            ],
            check=True,
            timeout=60,
        )
        subprocess.run([str(executable)], check=True, timeout=20)


def check_archived_prefix():
    evidence = json.loads((HERE / "model-results.json").read_text())
    pair = evidence["bs2_prefix_pair"]["arms"]
    if pair["serial"]["binary_sha256"] != pair["dual"]["binary_sha256"]:
        raise ValueError("The archived comparison uses different executables")
    spans = {}
    for mode, arm in pair.items():
        intervals = arm["closed_kernel_intervals"]
        span = max(intervals["Post0"][1], intervals["Score1"][1]) - intervals["Score0"][0]
        if span != arm["comparable_prefix_ticks"] or span <= 0:
            raise ValueError(f"{mode}: archived endpoint is inconsistent")
        spans[mode] = span
    print(f"Archived P1 prefix: {spans['serial']} -> {spans['dual']} scheduler ticks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path, metavar="DESTINATION", help="emit an isolated legacy source tree")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"), help="C++17 compiler for the host checks")
    args = parser.parse_args()
    available = subprocess.run(
        ["git", "cat-file", "-e", f"{LEGACY_BASE}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        timeout=20,
    )
    if available.returncode:
        parser.error(f"Missing {LEGACY_BASE}. Fetch refs/heads/codex/lifetch-score-t2 from KVIP origin.")
    if args.prepare:
        # Do not resolve the final element before rejecting existing symlinks.
        destination = args.prepare.absolute()
        prepare(destination)
        print(f"Staged {LEGACY_BASE} at {destination}")
    else:
        check(args.cxx)
        check_archived_prefix()
    print(f"Patch SHA256 {hashlib.sha256(PATCH.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
