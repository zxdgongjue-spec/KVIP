# SPDX-License-Identifier: Apache-2.0
"""Completed case files survive an uncatchable exit without claiming study pass."""

import json
import os
import signal
import subprocess
import sys

import pytest

from kvip.diagnostics.replay_journal import ReplayJournal


def test_completed_case_survives_sigkill_in_next_case(tmp_path):
    output = tmp_path / "study.json"
    command = """
import os, signal, sys
from pathlib import Path
from kvip.diagnostics.replay_journal import ReplayJournal
journal = ReplayJournal(Path(sys.argv[1]))
journal.case('before', {'status': 'pass', 'timing': {'mean_us': 123.0}})
journal.phase({'variant': 'after', 'phase': 'graph_capture_and_fence', 'step': None})
os.kill(os.getpid(), signal.SIGKILL)
"""
    child = subprocess.run([sys.executable, "-c", command, str(output)], env=dict(os.environ), check=False)
    assert child.returncode == -signal.SIGKILL
    assert not output.exists()
    assert json.loads((tmp_path / "study.cases/before.json").read_text())["timing"]["mean_us"] == 123.0
    progress = json.loads((tmp_path / "study.progress.json").read_text())
    assert progress["status"] == "in_progress"
    assert progress["completed_cases"] == ["before"]
    assert progress["progress"] == dict(variant="after", phase="graph_capture_and_fence", step=None)


def test_existing_reports_and_checkpoints_are_not_overwritten(tmp_path):
    output = tmp_path / "study.json"
    output.write_text("existing final report")
    with pytest.raises(FileExistsError):
        ReplayJournal(output)
    assert output.read_text() == "existing final report"
    output.unlink()
    journal = ReplayJournal(output)
    journal.case("before", {"timing": 1})
    with pytest.raises(FileExistsError):
        journal.case("before", {"timing": 2})
    assert json.loads((tmp_path / "study.cases/before.json").read_text()) == {"timing": 1}
    with pytest.raises(FileExistsError):
        ReplayJournal(output)
    assert not list(tmp_path.rglob(".replay-*"))
