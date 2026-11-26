# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'support'))
import pr_head_anchor as anchor
from cap_pair_report import SDK, SDK_TREE
from pr30_feature_steps import REVISIONS


def test_previous_pr30_reuses_only_the_recorded_local_placement_and_oracle(tmp_path, monkeypatch):
    machine = '01234567-89ab-cdef-0123-456789abcdef'
    current = tmp_path / 'current'
    current.mkdir()
    owner = tmp_path / 'owner'
    owner.mkdir()
    prior = tmp_path / 'archive'
    markers = dict(host='node20', machine_id=machine, boot_id=machine, physical_npu='4')
    for name, value in markers.items():
        (current / f'{name}.txt').write_text(value)
    for i in (1, 2):
        block = prior / f'block{i}'
        folder = block / 'g7/paced/results/mainline/cases/diagnose/fused'
        folder.mkdir(parents=True)
        for name, value in {**markers, **REVISIONS, 'kvaf':SDK, 'kvaf_tree':SDK_TREE, 'live_main':'1'}.items():
            (block / f'{name}.txt').write_text(value)
        (folder/'path_result.json').write_text('{}')
        (folder/'score_topk_check.json').write_text(json.dumps({'native_reference':{'archive_path':str(owner)}}))
        (block/'g7/feature_steps.json').write_text(json.dumps(dict(group=7, live_main=True, calls_per_graph={'paced':10}, timing={'paced':{'event_mean_us':437.1}})))
        (block/'g7/paced/thp.json').write_text(json.dumps(dict(status='measured', page_mode='thp', bytes=100, coverage_min_pct=100, ub_device='adapter', numa_pages_overlapping_vmas={'3':256})))
    monkeypatch.setattr(anchor, 'owner_data', lambda root, group: None)
    assert anchor.anchor(prior, current, 7) == (owner, prior/'block1/g7/paced/thp.json')
    (prior/'block2/machine_id.txt').write_text('ffffffff-ffff-ffff-ffff-ffffffffffff')
    with pytest.raises(ValueError, match='different machine'):
        anchor.anchor(prior, current, 7)
