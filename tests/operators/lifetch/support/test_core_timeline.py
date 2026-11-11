import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('timeline', Path(__file__).with_name('summarize_core_timeline.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    cores = [[100, 1000] + [0]*30 for _ in range(6)]
    p = [0]*64
    p[1:4] = [110, 120, 130]
    for q in range(4):
        p[10+5*q:15+5*q] = list(range(140+q*20, 145+q*20))
    u = [0]*32
    u[10] = 110
    for q in range(5):
        u[12+2*q:14+2*q] = [230+q*20, 240+q*20]
    u[22:25] = [340, 350, 360]
    p[32:] = u
    f = [0]*128
    for q in range(4):
        cores[2+q][2:7] = [120, 400, 420, 430, 500]
        f[q*24:q*24+3] = [120, 300, 310]
    f[96:101] = [400, 800, 850, 900, 450]
    for i in range(6):
        f[104+2*i:106+2*i] = [460+i*20, 470+i*20]
    return dict(batch=1, workers_per_query=1, clock_mhz=1000,
                cores=cores, producer_update=[p], fetch=[f], flags=[[[1, 0, 0]]*12])


def test_same_replay_latest_core_and_stages():
    raw = fixture()
    raw['cores'][3][1] = 1100
    result = module.decode(raw)
    assert result['last_block'] == 3
    assert result['observed_span_us'] == 1
    assert result['cores'][3]['role'] == 'fetch Q2 lane0'
    assert len([s for s in result['cores'][0]['stages'] if not s['name'].startswith('control_gap')]) == 27
    assert any(s['name'] == 'wait.Q4_fetch' for s in result['cores'][0]['stages'])


@pytest.mark.parametrize('mutation', ['missing_exit', 'missing_wait', 'publication_order'])
def test_incomplete_or_inconsistent_timeline_fails(mutation):
    raw = copy.deepcopy(fixture())
    if mutation == 'missing_exit': raw['cores'][0][1] = 0
    if mutation == 'missing_wait': raw['fetch'][0][105] = 0
    if mutation == 'publication_order': raw['fetch'][0][1] = 130
    with pytest.raises(ValueError):
        module.decode(raw)
