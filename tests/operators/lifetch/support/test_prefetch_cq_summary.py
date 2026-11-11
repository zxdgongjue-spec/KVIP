import importlib.util
from pathlib import Path

import pytest

from test_core_timeline import fixture

spec = importlib.util.spec_from_file_location(
    'cq_summary', Path(__file__).with_name('summarize_prefetch_cq.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample():
    raw = fixture()
    raw['fetch'][0][101:104] = [4, 30, 440]
    return {'round': 0, 'streaming_cycles': [{'core_timeline_raw': raw}]}


def test_drain_and_surrounding_use_same_invocation():
    result = module.analyze(sample())
    row = result['prefetch'][0]
    assert row['total_us'] == .05
    assert row['drain_us'] == .03
    assert row['surrounding_us'] == .02
    assert row['polls'] == 4
    assert row['drain_begin_us'] == .31
    assert row['last_observed_us'] == .34
    assert result['final_fetch'][0]['get_us'] == .07


@pytest.mark.parametrize('index,value', [(102, 51), (103, 451), (100, 399)])
def test_invalid_capture_rejected(index, value):
    data = sample()
    data['streaming_cycles'][0]['core_timeline_raw']['fetch'][0][index] = value
    with pytest.raises(ValueError):
        module.analyze(data)
