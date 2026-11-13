"""Compile all entry/worker sources, including lazily imported NPU-only modules."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('source', sorted(ROOT.rglob('*.py')), ids=lambda path: str(path.relative_to(ROOT)))
def test_python_source_compiles(source):
    compile(source.read_bytes(), str(source), 'exec')
