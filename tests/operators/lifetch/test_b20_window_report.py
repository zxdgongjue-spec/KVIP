from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent/'support'))
import b20_window_report as report
from reporting import atomic_json


@pytest.mark.parametrize('card', [0, 4])
def test_b20_comparison_uses_archived_metrics_and_labels_machine_difference(tmp_path, monkeypatch, capsys, card):
    for name, value in {'main': report.MAIN, 'kvaf': report.KVAF, 'candidate': 'a'*40, 'physical_npu': str(card)}.items():
        (tmp_path/f'{name}.txt').write_text(value)
    hashes = ['b'*64]*20
    baseline = report.historical_times()
    for group in report.GROUPS:
        folder = tmp_path/f'g{group}'/'li-window'
        folder.mkdir(parents=True)
        now = {k: [t/2 for t in pair] for k, pair in baseline[str(group)].items()}
        args = (folder, {}, {'topk_set_digests': hashes}, now, {}, {'runtime': {'visible':str(card), 'host':'B20'}}, {})
        monkeypatch.setattr(report, 'measured_artifacts', lambda *a, **kw: args)
        check = dict(group=group, queries=1280, calls=20, topk_set_digests=list(hashes),
                     request_state_sha256=report.REQUEST_STATE[group],
                     score_domain_sha256=report.ARCHIVE[group][0], score_multiset_sha256=report.ARCHIVE[group][1])
        atomic_json(folder/'portable_score_check.json', check)
        report.group_report(tmp_path, group, card)
        with pytest.raises(ValueError, match='NPU mismatch'):
            report.group_report(tmp_path, group, 4 - card)
        args[5]['runtime']['visible'] = str(4 - card)
        with pytest.raises(ValueError, match='disagree'):
            report.group_report(tmp_path, group, card)
        args[5]['runtime']['visible'] = str(card)
        check['topk_set_digests'] = hashes[:-1]
        atomic_json(folder/'portable_score_check.json', check)
        with pytest.raises(ValueError, match='disagree'):
            report.group_report(tmp_path, group, card)
    report.summary(tmp_path, card)
    with pytest.raises(ValueError, match='different physical NPU'):
        report.summary(tmp_path, 4 - card)
    lines = capsys.readouterr().out
    assert f'B20_physical_npu={card} B24_physical_npu=4' in lines
    assert 'queries=1280/1280' in lines and 'B24_archive_minus_B20_candidate cross_host' in lines
    assert 'delta_pct=+50.000 same_host_10pct_target=UNVERIFIED' in lines
