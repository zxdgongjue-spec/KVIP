# SPDX-License-Identifier: Apache-2.0
"""Exercise the detached launch exactly as invoked from the remote Git checkout."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tempfile
from textwrap import dedent
import unittest

ENTRY = Path("tests/operators/lifetch")
LAUNCHER = ENTRY / "resume_four_ranks.py"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@unittest.skipUnless(Path("/proc/sys/kernel/random/boot_id").is_file(), "Linux validation server")
class DetachedResumeTest(unittest.TestCase):
    def test_keeps_timed_revision_and_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_detached(Path(directory))

    def run_detached(self, tmp_path):
        repo = tmp_path / "repo"
        (repo / ENTRY / "support").mkdir(parents=True)
        git(repo, "init", "-q")
        git(repo, "config", "user.name", "Fixture")
        git(repo, "config", "user.email", "fixture@example.invalid")
        (repo / "timed").write_text("original")
        git(repo, "add", ".")
        git(repo, "commit", "-qm", "original")
        native = git(repo, "rev-parse", "HEAD")
        script = Path(__file__).with_name("resume_four_ranks.py").read_bytes()
        (repo / LAUNCHER).write_bytes(script)
        evidence = Path(__file__).parent / "support/replay_interrupt_evidence.py"
        (repo / ENTRY / "support/replay_interrupt_evidence.py").write_bytes(evidence.read_bytes())
        (repo / "timed").write_text("controller")
        (repo / ENTRY / "large_batch_npu4.sh").write_text("""set -eu
    test "$(cat "$PEER_TIMED_SOURCE/timed")" = original
    test "$1" = --four-ranks-candidate-only
    test "$PEER_PHYSICAL_NPU:$ASCEND_RT_VISIBLE_DEVICES:$FIXED_Q4_BATCHES" = '3:3:32 28 24 20'
    test "$FIXED_Q4_GROUPS" = '7 2 10 4 17'
    cd "$PEER_PACKED_RESUME"
    echo run >> attempts
    while test ! -e release; do sleep 0.05; done
    exit "$(cat release)"
    """)
        (repo / ENTRY / "support/report_large_li_phases.py").write_text(dedent("""
    def report(root, native, batches, groups):
        assert (root / 'native.txt').read_text() == native
        return [dict(group=g, batch=b) for b in batches for g in groups]
    """))
        git(repo, "add", ".")
        git(repo, "commit", "-qm", "recovery control")
        revision = git(repo, "rev-parse", "HEAD")
        root = tmp_path / "peer-baseline/fixed-q4-four-ranks-one-block.fixture"
        root.mkdir(parents=True)
        (tmp_path / "fused-scan-sources").mkdir()
        for key, value in dict(candidate_blocks="1", selection_oracle="captured_logical_scores",
                               physical_npu="3", host=os.uname().nodename, native=native,
                               boot_id=Path("/proc/sys/kernel/random/boot_id").read_text()).items():
            (root / f"{key}.txt").write_text(value)

        def start():
            return subprocess.run([sys.executable, "-B", "-", str(repo), revision, str(root)],
                                  input=script, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  check=True, timeout=10).stdout.decode()

        def wait_for(path, marker):
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if path.is_file() and marker in path.read_text():
                    return
                time.sleep(0.05)
            self.fail(f"Missing {marker!r} in {path}: {(root / 'resume.log').read_text()}")

        try:
            assert "RESUME_STARTED" in start()
            wait_for(root / "attempts", "run")
            assert "ALREADY_RUNNING" in start()
        finally:
            (root / "release").write_text("124")
        wait_for(root / "resume.log", "RESUME_FAILED child_returncode=124")
        (root / "release").write_text("0")
        deadline = time.monotonic() + 8
        while "ALREADY_RUNNING" in start():
            assert time.monotonic() < deadline
            time.sleep(0.05)
        wait_for(root / "resume.log", "RESUME_COMPLETE cells=20")
        assert len(json.loads((root / "li_phase_comparison.json").read_text())) == 20
        assert (root / "attempts").read_text().splitlines() == ["run", "run"]


if __name__ == "__main__":
    unittest.main()
