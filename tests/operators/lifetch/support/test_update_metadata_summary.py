import json

import pytest

from test_core_timeline import fixture
from summarize_update_metadata_ab import summarize


def test_update_comparison_reports_work_and_waits(tmp_path):
    for version, duration in [('A', 100), ('B', 90)]:
        root = tmp_path / version
        root.mkdir()
        (root / 'report.json').write_text(json.dumps({'timing': {'prototype_with_update': {'p50_us': duration, 'p99_us': duration+10}}}))
        sample = {'round': 0, 'accounting': {'stages_us': {'resolve_update': 50}},
                  'streaming_cycles': [{'core_timeline_raw': fixture()}]}
        (root / 'full_attribution.json').write_text(json.dumps({'samples': [sample]}))
    text = summarize(tmp_path)
    assert 'change=-10.00%' in text
    assert 'last_Update=1' in text
    assert 'metadata | 0.240 | 0.250 | 0.010' in text
    assert 'Q_wait_sum' in text
    (tmp_path / 'B' / 'report.json').unlink()
    with pytest.raises(ValueError, match='no checked reports'):
        summarize(tmp_path)
