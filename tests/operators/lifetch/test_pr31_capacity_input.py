from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
import pr31_capacity_input as refresh
from main_five_run import parse_args
from near512_locality_input import expand_locality
from test_near512_locality_input import donor


def test_wide_shapes_use_ten_independent_states():
    base = ['--main', '.', '--trace', '.', '--owner-archive', '.', '--kvaf-root', '.', '--output', '.',
            '--kvaf-server', '127.0.0.1:42', '--group', '4', '--revision', refresh.PR31, '--ablation-policy', 'paced']
    for shape in ['wide-short', 'wide-long']:
        for worker in ([], ['--worker']):
            cfg = parse_args([*base, '--pr31-shape', shape, '--narrow-proof', 'archive', *worker])
            assert (cfg.iterations, cfg.production_graph, cfg.history_cap) == (10, True, 524544)
        with pytest.raises(SystemExit):
            parse_args([*base, '--pr31-shape', shape])


def test_narrow_state_is_checked_even_when_short_scores_match(tmp_path):
    cap = dict(capture_history=131328, captured_page_count=1, effective_ends=[[111]],
               query_ends_sha256='q', capture_page_table_sha256='p', capture_state_sha256={'captured_lru': 'a'})
    proof = dict(production_revision=refresh.PR31, runtime=dict(visible='4'), capture_operand_hashes={'query': 'b'},
                 measured_calls=10, history_cap=dict(cap, bound_history=131328))
    path = tmp_path / 'narrow.json'
    path.write_text(json.dumps(proof))
    refresh.check_donor(path, {'query': 'b'}, cap)
    with pytest.raises(ValueError, match='capture_state_sha256'):
        refresh.check_donor(path, {'query': 'b'}, dict(cap, capture_state_sha256={'captured_lru': 'c'}))


def test_long_reference_checks_measured_ids_against_original_donor_and_long_reference(monkeypatch):
    short = donor()
    short.geometry['output_stride'] = short.geometry['resident'] + 512
    long, detail = expand_locality(short, 16384)

    def topk(inputs):
        scores = np.where(inputs.logical_score_valid, inputs.logical_score_keys.astype(np.int32), -1)
        return np.argsort(-scores, kind='stable', axis=-1)[..., :2048].astype(np.int32)

    original = topk(short)
    expanded = topk(long)
    monkeypatch.setattr(refresh, 'serial_reference', lambda *args: expanded)
    native_check = Mock()
    monkeypatch.setattr(refresh, 'validate_native_sets', native_check)
    fake_torch = SimpleNamespace(npu=Mock(), device=lambda name: name)
    ids, receipt = refresh.long_oracle(short, long, detail, original, fake_torch, None)
    assert receipt['new_winners'] == 0 and receipt['short_unique'] == receipt['long_unique']
    long.validate_topk(ids)
    assert native_check.call_count == 1
    bad = ids.copy()
    bad[0, 0, 0] = np.flatnonzero(~np.isin(np.arange(16384) // 128, detail['source_page_to_long_page'][0]))[0]
    with pytest.raises((ValueError, AssertionError)):
        long.validate_topk(bad)
    assert native_check.call_count == 1
