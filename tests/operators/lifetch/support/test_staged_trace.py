# SPDX-License-Identifier: Apache-2.0
"""Check staged simulator trace accounting and strict missing-stage handling."""
import importlib.util
import json
from pathlib import Path

import pytest

source = Path(__file__).resolve().parents[1] / 'simulator/compare_staged_trace.py'
spec = importlib.util.spec_from_file_location('compare_staged_trace', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_output_copy_times_and_missing_copy(tmp_path):
    events = [
        {'name': 'process_name', 'ph': 'M', 'pid': 70, 'args': {'name': 'MTE3'}},
        {'name': 'init', 'ph': 'X', 'pid': 1, 'ts': 1.0, 'dur': 0.5},
    ]
    for timestamp in (10, 20, 30, 40):
        events.append({'name': 'MOV_SRC_TO_DST_ALIGNv2', 'ph': 'X', 'pid': 70,
                       'ts': timestamp, 'dur': 0.25})
    path = tmp_path / 'trace.json'
    path.write_text(json.dumps({'traceEvents': events}))
    assert module.copy_completions(path, 4) == [9.25, 19.25, 29.25, 39.25]
    with pytest.raises(ValueError, match='expected 1 output copies'):
        module.copy_completions(path, 1)
