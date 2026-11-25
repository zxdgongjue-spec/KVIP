# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'support'))
import feature_host_reference as refs

MACHINE = '1234567890abcdef1234567890abcdef'
BOOT = '55a61dee-2cd7-4d46-a22a-eaf0571a2a08'
LOCAL = ('localhost', MACHINE, BOOT)


def markers(root, **entries):
    root.mkdir(parents=True, exist_ok=True)
    for key, value in entries.items():
        (root/(key+'.txt')).write_text(str(value))


def place(path, coverage=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(status='measured', page_mode='thp', bytes=2**21,
                                    coverage_min_pct=coverage, ub_device='ub42',
                                    numa_pages_overlapping_vmas={'3': 512})))


def test_only_local_placement_with_valid_transport(tmp_path):
    root = tmp_path/'fused.local'
    markers(root, host='localhost', physical_npu=4, boot_id=BOOT)
    path = root/'g7/old-arm/thp.json'
    place(path)
    bad = tmp_path/'outside'/'block1'
    markers(bad, host='localhost', physical_npu=4, machine_id='a'*32, boot_id=BOOT)
    place(bad/'g7/dualcut/thp.json')
    assert refs.find_reference(tmp_path, 7, 'placement', LOCAL) == path
    other = root/'g4/another-arm/thp.json'
    place(other)
    assert refs.find_reference(tmp_path, 7, 'placement', LOCAL) == path
    (root/'boot_id.txt').unlink()
    assert refs.find_reference(tmp_path, 7, 'placement', LOCAL) is None
    markers(root, machine_id=MACHINE, boot_id='old')
    assert refs.find_reference(tmp_path, 7, 'placement', LOCAL) == path
    place(path, coverage=90)
    assert refs.find_reference(tmp_path, 7, 'placement', LOCAL) == other
    place(other, coverage=80)
    assert refs.find_reference(tmp_path, 7, 'placement', LOCAL) is None


def test_baseline_provenance_and_incomplete_archives(tmp_path, monkeypatch):
    runtime = dict(host='localhost', visible='4', npu='Ascend950', torch='t1', torch_npu='n1', cann='c1')
    monkeypatch.setattr(refs, 'current_runtime', lambda host: dict(runtime))
    owner = tmp_path/'foreign-owner'
    markers(owner, **{'logical-ties':refs.OWNER, 'host':'another-host'})
    def owner_data(path, group):
        assert path == owner and group == 7
        return 'oracle'
    monkeypatch.setattr(refs, 'owner_data', owner_data)
    assert refs.find_reference(tmp_path, 7, 'owner', LOCAL) == owner
    for name, groups, machine in [('full', refs.GROUPS, MACHINE),
                                   ('partial', (7,), MACHINE), ('foreign', refs.GROUPS, 'f'*32)]:
        root = tmp_path/name
        markers(root, main=refs.REVISION, kvaf=refs.SDK, kvaf_tree=refs.SDK_TREE,
                host='localhost', machine_id=machine, physical_npu=4)
        for group in groups:
            folder = root/f'g{group}/mainline/results/run/cases/diagnose/fused'
            folder.mkdir(parents=True)
            (folder/'path_result.json').write_text('{}')
            (folder/'mainline_checks.json').write_text(json.dumps({'runtime':runtime}))
            (folder/'score_topk_check.json').write_text(json.dumps({'native_reference':{'archive_path':str(owner)}}))
            place(root/f'g{group}/mainline/thp.json')
    def measured(root, group, arm, owner_arg, oracle, host):
        assert arm == 'mainline' and host == 'localhost' and owner_arg == owner
        assert oracle == 'oracle' and group == 7
        if root.name == 'full' and (root/'incomplete').exists():
            raise ValueError('incomplete measured timings')
    monkeypatch.setattr(refs, 'collect_arm', measured)
    assert refs.find_reference(tmp_path, 7, 'baseline', LOCAL) == tmp_path/'full'
    check = tmp_path/'full/g7/mainline/results/run/cases/diagnose/fused/mainline_checks.json'
    check.write_text(json.dumps({'runtime': dict(runtime, torch_npu='older')}))
    assert refs.find_reference(tmp_path, 7, 'baseline', LOCAL) == tmp_path/'partial'
    check.write_text(json.dumps({'runtime':runtime}))
    (tmp_path/'full/incomplete').touch()
    assert refs.find_reference(tmp_path, 7, 'baseline', LOCAL) == tmp_path/'partial'
    markers(tmp_path/'partial', kvaf='another-sdk')
    assert refs.find_reference(tmp_path, 7, 'baseline', LOCAL) is None


def test_missing_local_data_needs_no_runtime(tmp_path, monkeypatch):
    def unexpected(host):
        raise AssertionError('no archive requires runtime probing')
    monkeypatch.setattr(refs, 'current_runtime', unexpected)
    assert refs.find_reference(tmp_path, 7, 'baseline', LOCAL) is None
