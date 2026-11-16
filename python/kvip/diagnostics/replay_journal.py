# SPDX-License-Identifier: Apache-2.0
"""Small untimed checkpoints that survive a killed replay process."""

import json
import os
from pathlib import Path
import tempfile


def _write_json(path, value, *, replace=False):
    """Publish a complete file; successful case checkpoints are never replaced."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".replay-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2)
            stream.write("\n")
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class ReplayJournal:
    """Keep partial results separate from the final report and its pass status."""

    def __init__(self, output):
        output = Path(output)
        self.directory = output.with_suffix(".cases")
        self.progress = output.with_suffix(".progress.json")
        if output.exists() or self.progress.exists():
            raise FileExistsError(f"Replay output/checkpoint already exists: {output}")
        self.directory.mkdir(parents=True, exist_ok=False)
        self.completed = []

    def phase(self, progress):
        # Exact iteration/step progress remains in memory for normal failures.
        # Disk breadcrumbs are coarse: no IO between replay iterations.
        if progress.get("step") is not None:
            return
        self._save(progress, status="in_progress")

    def finish(self, progress, *, status, report):
        self._save(progress, status=status, report=str(report) if report is not None else None)

    def _save(self, progress, **state):
        _write_json(self.progress, dict(
            pid=os.getpid(), progress=dict(progress), completed_cases=list(self.completed), **state,
        ), replace=True)

    def case(self, name, result):
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Invalid checkpoint case name: {name!r}")
        _write_json(self.directory / f"{name}.json", result)
        self.completed.append(name)
