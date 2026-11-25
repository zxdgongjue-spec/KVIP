# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
from pr30_feature_repeats import GROUPS, summarize


def test_reverse_order_and_stage_isolation(tmp_path):
    for block in (1, 2):
        for group in GROUPS:
            path = tmp_path / f'block{block}' / f'g{group}'
            path.mkdir(parents=True)
            (path / 'order.txt').write_text('dualcut paced' if block == 1 else 'paced dualcut')
            timings = {
                'mainline': {'graph': [300, 299], 'LI': [240, 240], 'tail': [55, 54]},
                'dualcut': {'graph': [290, 290], 'LI': [226, 226], 'tail': [58, 58]},
                'paced': {'graph': [290 - block, 288], 'LI': [226, 226], 'tail': [58 - block, 56]},
            }
            (path / 'feature_steps.json').write_text(json.dumps(dict(
                group=group, reference='/same-machine/archive', timing=timings, measured={'paced': {'split_q': group}},
                calls_per_graph={'mainline': 20, 'dualcut': 10, 'paced': 10},
                device_text={'main_tail_equals_dualcut': True, 'dualcut_li_equals_paced': True})))
    summarize(tmp_path)
    result = json.loads((tmp_path / 'feature_comparison.json').read_text())
    assert result['equal_weight']['A_vs_main'] == {'graph': [10, 10], 'LI': [14, 14], 'tail': [-3, -3]}
    assert result['groups']['7']['B_vs_A']['tail'] == [1, 2]
    assert result['groups']['7']['paced_split_Q_per_640'] == [7, 7]

    path = tmp_path / 'block2/g7'
    (path / 'order.txt').write_text('dualcut paced')
    with pytest.raises(ValueError, match='reverse order'):
        summarize(tmp_path)
    (path / 'order.txt').write_text('paced dualcut')
    report = path / 'feature_steps.json'
    data = json.loads(report.read_text())
    data['device_text']['dualcut_li_equals_paced'] = False
    report.write_text(json.dumps(data))
    summarize(tmp_path)
    result = json.loads((tmp_path / 'feature_comparison.json').read_text())
    assert result['groups']['7']['unmodified_stage_match'] == {
        'A_vs_main': [True, True], 'B_vs_A': [True, False]}
    data['calls_per_graph']['paced'] = 20
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='expected 10 measured candidate calls'):
        summarize(tmp_path)
    data['calls_per_graph']['paced'] = 10
    for block in (2, 1):
        report = tmp_path/f'block{block}/g7/feature_steps.json'
        if block == 1:
            data = json.loads(report.read_text())
        data['reference'] = None
        del data['timing']['mainline'], data['calls_per_graph']['mainline']
        data['device_text']['main_tail_equals_dualcut'] = None
        report.write_text(json.dumps(data))
        if block == 2:
            with pytest.raises(ValueError, match='different upstream baselines'):
                summarize(tmp_path)
    summarize(tmp_path)
    result = json.loads((tmp_path / 'feature_comparison.json').read_text())
    assert result['equal_weight']['A_vs_main'] is None
    assert result['groups']['7']['A_vs_main'] is None
    assert result['equal_weight']['B_vs_A']['tail'] == [1, 2]
