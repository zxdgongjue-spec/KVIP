import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
from score_equivalence import CapturedScoreOracle, check_scores, save_checked
from workloads import ReplayRound


def example(batch=1):
    history = 2055
    keys = np.ones((batch, 4, history), dtype=np.uint16)
    keys[:, :, :2046] = 2
    keys[:, :, -1] = 0
    inputs = SimpleNamespace(geometry=dict(history=history, batch=batch), logical_score_keys=keys,
                             logical_score_valid=np.ones_like(keys, dtype=bool),
                             query_ends=np.tile(np.arange(history - 3, history + 1), (batch, 1)))
    inputs.validate_topk = lambda ids: ReplayRound.validate_topk(inputs, ids)
    return inputs, np.broadcast_to(np.arange(2048, dtype=np.int32), (batch, 4, 2048)).copy()


def test_interchange_only_cutoff_ties():
    data, serial = example()
    candidate = serial[..., ::-1].copy()
    candidate[0, 0, 0] = 2048
    check = check_scores(data, serial, [candidate, serial])
    assert (check['queries'], check['exact_calls'], check['exact_queries']) == (8, 1, 7)
    with pytest.raises(AssertionError, match='ranking'):
        missing = serial.copy()
        missing[0, 0, 5] = 2049
        check_scores(data, serial, [missing])
    with pytest.raises(AssertionError, match='ranking'):
        below = serial.copy()
        below[0, 3, -1] = 2054
        check_scores(data, serial, [below])
    with pytest.raises(AssertionError, match='duplicates'):
        duplicate = serial.copy()
        duplicate[0, 3, -1] = duplicate[0, 3, -2]
        check_scores(data, serial, [duplicate])
    with pytest.raises(ValueError, match='logical query bound'):
        out = serial.copy()
        out[0, 0, -1] = 2052
        check_scores(data, serial, [out])
    data.logical_score_valid[0, 0, 2048] = False
    with pytest.raises(AssertionError, match='causal prefix'):
        check_scores(data, serial, [candidate])
    with pytest.raises(AssertionError, match='causal prefix'):
        check_scores(data, candidate, [serial])


def test_ten_scored_calls_verifiably_bound_to_ten_measured_outputs(tmp_path):
    from correctness import topk_fingerprint
    from score_equivalence_report import measured_sets

    data, serial = example(16)
    outputs = np.broadcast_to(serial, (10, *serial.shape))
    save_checked(tmp_path, 4, data, serial, {}, outputs, expected_calls=10)
    receipt = json.loads((tmp_path/'score_topk_check.json').read_text())
    current = (None, [topk_fingerprint(serial)] * 10)
    assert (receipt['calls'], receipt['queries']) == (10, 640)
    assert measured_sets(tmp_path, 4, current, serial, receipt, expected_calls=10).all()
    with pytest.raises(ValueError, match='Incomplete'):
        measured_sets(tmp_path, 4, current, serial, receipt)
    with pytest.raises(ValueError, match='Expected 10'):
        save_checked(tmp_path, 4, data, serial, {}, outputs[:9], expected_calls=10)


def test_cached_score_oracle_checks_every_selection_against_verified_population():
    inputs, serial = example(16)
    inputs.logical_score_valid[-1, -1, 2051] = False
    inputs.logical_score_keys[0, 0] = 0  # Every admissible token ties in this query.
    good = serial.copy()
    good[:, :, -1] = 2048
    outputs = [good] * 9 + [good.copy()]
    reference = check_scores(inputs, serial, outputs)
    oracle = CapturedScoreOracle(inputs, serial)
    inputs.validate_topk = oracle.validate
    assert check_scores(inputs, serial, outputs) == reference
    for position, token, error in [(0, 2049, 'ranking'), (-1, 2054, 'ranking'),
                                   (-1, 2046, 'duplicates'), (-1, 2051, 'causal prefix'),
                                   (-1, -1, 'logical query bound'), (-1, 2055, 'logical query bound')]:
        bad = good.copy()
        bad[-1, -1, position] = token
        with pytest.raises(AssertionError, match=f'Call 10: .*{error}'):
            check_scores(inputs, serial, [*outputs[:9], bad])
    for bad in (good[..., :-1], good.astype(np.float32)):
        with pytest.raises(AssertionError, match='shape/type'):
            oracle.validate(bad)
    inputs, bad_serial = example()
    bad_serial[0, 0, 0] = 2048
    with pytest.raises(AssertionError, match='ranking'):
        CapturedScoreOracle(inputs, bad_serial)


def test_report_binds_score_checked_ids_to_measured_and_archive(tmp_path, monkeypatch, capsys):
    import score_equivalence_report as report
    from correctness import topk_fingerprint
    from native_li_reference import digest
    current, archive = tmp_path / 'run', tmp_path / 'archived-handoff'
    relative = Path('g4/logical-ties/results/job/cases/diagnose/fused')
    data, serial = example(16)
    outputs = np.broadcast_to(serial, (20, *serial.shape)).copy()
    outputs[3:, 0, 0, -1] = 2048
    placements = dict(coverage_min_pct=100, ub_device='urma0', numa_pages_overlapping_vmas={'3':1000})
    ids = dict(inputs=dict(query=digest(serial)))
    meta = dict(identity=ids, replay_identity=ids, archive_path=str(archive/relative/'native_li_compare.json'),
                origin='completed-archive')
    rows = [[('a'*64, 'b'*64)]*16 for _ in range(20)]
    def timings(root, graph, li, tail, fingerprints):
        folder = root / relative
        profile = folder / 'diagnose/round-1'
        profile.mkdir(parents=True)
        (root/'physical_npu.txt').write_text('4')
        accounting = dict(stages_us=dict(li=li), intervals_us=dict(li=[(li*i, li*i+li/4) for i in range(4)]))
        (profile/'full_attribution.json').write_text(json.dumps(dict(samples=[dict(accounting=accounting)]*20)))
        return (data.geometry, fingerprints, placements, graph, graph, root.name, [], [],
                tail, 100, tail, rows, None, li, '/kvaf')
    old = timings(archive, 405, 280, 117, [topk_fingerprint(serial)]*20)
    new = timings(current, 394, 270, 117, [topk_fingerprint(x) for x in outputs])
    (archive/'logical-ties.txt').write_text('67b49a2748fde8f8779c4551410416596b35cf76')
    monkeypatch.setattr(report, 'measurement', lambda *_: (old, {}, meta, ({},64,True,'PASS',serial),
                                                            archive/relative/'native_li_compare.json'))
    monkeypatch.setattr(report, 'load', lambda *_, **__: new)
    save_checked(current/relative, 4, data, serial, meta, outputs)
    report.report(current, 4, archive)
    printed = capsys.readouterr().out
    assert 'exact_ID_sets=1263/1280 cutoff_tie_only_sets=17/1280' in printed
    assert 'graph_archive_minus_candidate_mean/P50_us=+11.00/+11.00' in printed
    new[1][0] = 'wrong'
    with pytest.raises(ValueError, match='measured and KV-checked'):
        report.report(current, 4, archive)
