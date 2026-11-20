# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent/'support'))
import ranked_index_report as report


def test_reuse_requires_matching_sha_binary_device_capture_and_placement(tmp_path, monkeypatch):
    group = 2
    original = tmp_path/'original'
    root = tmp_path/'candidate'
    control = tmp_path/f'help-{report.CONTROL[:8]}-run'
    name = f'ranked-{report.CONTROL[:8]}.binary.sha256'
    for folder in (original, root, control):
        (folder/'g2').mkdir(parents=True)
        (folder/'g2/input.sha256').write_text('b'*64)
        (folder/'host.txt').write_text('container20')
        (folder/'physical_npu.txt').write_text('0')
        (folder/name).write_text('c'*64)
    (control/'harness.txt').write_text(report.CONTROL_HARNESS)
    (control/'candidate.txt').write_text(report.CONTROL)
    (control/'g2/auto').mkdir()
    (control/'g2/auto/pass').touch()
    monkeypatch.setattr(report, 'load', lambda *args: {})
    monkeypatch.setattr(report, 'routed_arm', lambda *args: {})
    monkeypatch.setattr(report, 'verify_pair', lambda *args: None)

    report.choose_control(root, original, group, fresh=False)
    saved = root/'g2/reuse-auto.txt'
    assert saved.read_text().strip() == str(control.resolve())
    report.choose_control(root, original, group, fresh=True)
    assert not saved.exists()
    assert (root/'g2/control-archive.txt').read_text().strip() == str(control.resolve())
    for path in ('candidate.txt', name, 'physical_npu.txt', 'host.txt', 'harness.txt', 'g2/input.sha256'):
        file = control/path
        data = file.read_text()
        file.write_text('different')
        report.choose_control(root, original, group, fresh=False)
        assert not saved.exists()
        file.write_text(data)

    def mismatched(*args):
        raise ValueError('NUMA')

    monkeypatch.setattr(report, 'verify_pair', mismatched)
    report.choose_control(root, original, group, fresh=False)
    assert not saved.exists()


def test_event_delta_includes_time_outside_invocations_and_overlap():
    def timing(li, tail, gap, overlap, outside):
        return {'timing': dict(stage_mean_us=dict(li=li, resolve_update=tail),
                               inter_kernel_gap_mean_us=gap, overlap_mean_us=overlap,
                               outside_invocations_mean_us=outside,
                               event_mean_us=li+tail+gap-overlap+outside)}
    old = timing(240, 70, .5, .1, 2)
    candidate = timing(235, 65, .2, .4, 4)
    saved = report.graph_savings(old, candidate)
    assert saved['li'] == saved['resolve_update'] == 5
    assert abs(saved['other_event_time'] + 1.4) < 1e-9
