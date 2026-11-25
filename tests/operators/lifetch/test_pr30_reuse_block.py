# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent/'support'))
import pr30_feature_steps as steps
from pr30_reuse_block import active_writers, reuse_block


def test_live_writer_between_cases_and_orphans(tmp_path):
    root = tmp_path/'prior'
    proc = tmp_path/'proc'
    (proc/'self/fd').mkdir(parents=True)
    for pid, argv, stdout in (
            (1001, ['tee', str(root/'run.log')], 'pipe:[201]'),
            (1002, ['python', '/opt/main_five_run.py', '--output', str(root/'block1/g7/paced')], 'pipe:[202]'),
            (1003, ['kvaf_cpu_service'], str(root/'block1/g7/paced/cpu.log')),
            (1004, ['tail', str(root/'run.log')], 'pipe:[205]'),
            (1005, ['tee', str(tmp_path/'prior-other/run.log')], 'pipe:[206]')):
        p = proc/str(pid)
        (p/'fd').mkdir(parents=True)
        (p/'fd/1').symlink_to(stdout)
        (p/'cmdline').write_bytes(b'\0'.join(s.encode() for s in argv))
    assert active_writers(root, proc) == [(1001, 'tee'), (1002, 'python'), (1003, 'kvaf_cpu_service')]


@pytest.fixture
def blocks(tmp_path):
    src, dst = tmp_path/'old', tmp_path/'new/block1'
    owner = tmp_path/'oracle'
    owner.mkdir()
    ids = {'host': 'b24', 'machine_id': 'f'*32, 'boot_id': 'x', 'physical_npu': '4'}
    for key, value in ids.items():
        src.mkdir(exist_ok=True)
        (src/f'{key}.txt').write_text(value)
    (src/'reader.txt').write_text('a'*40)
    for b in (src/'block1', dst):
        (b/'g10').mkdir(parents=True)
        for key, value in {**ids, **steps.REVISIONS, 'reader':'a'*40, 'kvaf':steps.SDK,
                           'kvaf_tree':steps.SDK_TREE, 'reference':'unavailable'}.items():
            (b/f'{key}.txt').write_text(value)
        (b/'g10/order.txt').write_text('dualcut paced')
    for arm in ('dualcut', 'paced'):
        folder = src/f'block1/g10/{arm}/results/mainline/cases/diagnose/fused'
        folder.mkdir(parents=True)
        for filename in ('path_result.json', 'score_topk_check.json'):
            (folder/filename).write_text('{}')
    return src, dst, owner


def test_reanalyzes_original_pair_without_touching_source(blocks, monkeypatch):
    src, dst, owner = blocks
    original = src/'block1/g10'
    (original/'feature_steps.json').write_text('old timing schema')
    def report(block, group, oracle):
        assert (block, group, oracle) == (dst, 10, owner)
        assert all((block/'g10'/a).resolve() == original/a for a in ('dualcut', 'paced'))
        (block/'g10/feature_steps.json').write_text('corrected timing')
    monkeypatch.setattr(steps, 'report', report)
    assert reuse_block(src, dst, 10, owner)
    assert (original/'feature_steps.json').read_text() == 'old timing schema'
    assert (dst/'g10/feature_steps.json').read_text() == 'corrected timing'
    assert json.loads((dst/'g10/reused_pair.json').read_text())['source'] == str(src)
    with pytest.raises(ValueError, match='overwrite'):
        reuse_block(src, dst, 10, owner)


def test_partial_and_corrupt_pairs_leave_no_arms_in_new_result(blocks, monkeypatch):
    src, dst, owner = blocks
    marker = next((src/'block1/g10/paced').rglob('score_topk_check.json'))
    marker.unlink()
    assert not reuse_block(src, dst, 10, owner)
    marker.write_text('{}')
    def report(block, group, oracle):
        raise ValueError('malformed sample intervals')
    monkeypatch.setattr(steps, 'report', report)
    assert not reuse_block(src, dst, 10, owner)
    assert list((dst/'g10').iterdir()) == [dst/'g10/order.txt']
    (src/'machine_id.txt').write_text('e'*32)
    with pytest.raises(ValueError, match='different or unidentified'):
        reuse_block(src, dst, 10, owner)
