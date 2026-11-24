from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent/'support'))
import pr30_parity_group as parity


def test_report_requires_a_packed_graph_and_independent_states(tmp_path, monkeypatch, capsys):
    old, new, single = tmp_path/'old', tmp_path/'new', tmp_path/'single'
    for root in (new, single):
        root.mkdir()
        (root/'main.txt').write_text('candidate')
    ctrl = SimpleNamespace(protocol=parity.PACKED, states=20)
    monkeypatch.setattr(parity, 'find_arm', lambda *a: ('cap128k', 'pr30'))
    monkeypatch.setattr(parity, 'case_folder', lambda root, *a: root)
    def read_arm(root, *_args, **kwargs):
        if root == new:
            assert kwargs == {'verify_library': True}
        return {'protocol': parity.INDIVIDUAL if root == single else (ctrl.protocol if root == new else parity.PACKED),
                'binary': 'candidate' if root != old else 'pr30', 'geometry': {}, 'oracle': {},
                'capture': {}, 'cap_state': {},
                'stats': dict(event_mean=400 if root == new else 410, kernel_span=(394,390),
                              LI1_to_LI3=(240,239), final_topk_through_tail=(151,149))}
    monkeypatch.setattr(parity, 'measure_arm', read_arm)
    monkeypatch.setattr(parity, 'same_work', lambda *a: True)
    monkeypatch.setattr(parity, 'same_controls', lambda *a: True)
    monkeypatch.setattr(parity, 'read', lambda p: dict(library_sha256=p.parent.name,
                        independent_buffers=dict.fromkeys(('index','kv','winners','window'), ctrl.states)))
    monkeypatch.setattr(parity, 'terminal_detail', lambda p: {'status': 'UNAVAILABLE'})
    monkeypatch.setattr(parity, 'print_terminal_detail', lambda *a: None)
    parity.report(new, 7, old, tmp_path, single)
    output = capsys.readouterr().out
    assert 'candidate_minus_PR30 graph_mean/P50_us=-10.00/+0.00' in output
    assert 'same_candidate_packed_minus_single event_mean_us=-10.00' in output
    ctrl.protocol = 'one_state_individual_graphs'
    with pytest.raises(ValueError, match='comparison checks'):
        parity.report(new, 7, old, tmp_path, single)
    ctrl.protocol, ctrl.states = parity.PACKED, 1
    with pytest.raises(ValueError, match='comparison checks'):
        parity.report(new, 7, old, tmp_path, single)
