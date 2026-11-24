# SPDX-License-Identifier: Apache-2.0
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
from pr30_terminal_phases import PHASES, inspect, print_terminal_detail, terminal_detail


def timeline():
    cores = []
    for b in range(64):
        begin = 10000 + (5 if b == 42 else 0)
        clock = begin
        intervals = []
        phases = list(PHASES)
        phases.insert(7, 'control_gap')
        for phase in phases:
            ticks = 7 if b == 42 and phase == 'final_CQ_join' else 1
            stop = clock + ticks
            intervals.append(dict(name=phase, start_cycles=clock, end_cycles=stop, start_us=(clock-10000)/10,
                                  end_us=(stop-10000)/10, duration_us=ticks/10))
            clock = stop
        cores.append(dict(block=b, start_us=(begin-10000)/10, end_us=(clock-10000)/10,
                          duration_us=(clock-begin)/10, stages=intervals))
    return dict(clock_mhz=10., cores=cores, last_block=42, observed_span_us=2.4)


def test_reconcile_last_worker_only_and_print_delta(tmp_path, capsys):
    t = timeline()
    s = inspect(t)
    assert s['span'] == pytest.approx(2.4)
    assert s['stages']['start_delay'] == .5 and s['stages']['final_CQ_join'] == .7
    assert sum(s['stages'].values()) == pytest.approx(s['span'])
    folder = tmp_path/'diagnose/round-1'
    folder.mkdir(parents=True)
    samples = [dict(round=i, source_round=1, streaming_cycles=[dict(terminal_timeline=t)]) for i in range(20)]
    (folder/'detail_samples.json').write_text(json.dumps(samples))
    report = terminal_detail(tmp_path)
    assert report['last_blocks'] == {42: 20} and report['tie_calls'] == 0
    print_terminal_detail(7, dict(pr30=report, active=report))
    assert 'active_minus_pr30_last_worker_phase_mean_us' in capsys.readouterr().out


def test_refuse_partial_missing_or_invented_duration(tmp_path):
    assert terminal_detail(tmp_path)['status'] == 'UNAVAILABLE'
    t = timeline()
    broken = copy.deepcopy(t)
    broken['cores'].pop(4)
    with pytest.raises(ValueError, match='request/Q'):
        inspect(broken)
    broken = copy.deepcopy(t)
    broken['cores'][42]['stages'][3]['duration_us'] = 22.
    with pytest.raises(ValueError, match='device ticks'):
        inspect(broken)
    t['clock_mhz'] = float('nan')
    with pytest.raises(ValueError, match='calibration'):
        inspect(t)
