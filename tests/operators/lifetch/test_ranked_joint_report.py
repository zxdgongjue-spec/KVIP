# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from ranked_five_report import JOINT_MAGIC
from ranked_joint_report import drift, early_index_stats, joint_endpoints, run


def sample():
    ticks = np.zeros((1, 6656), dtype=np.int64)
    core = ticks[:, 4096:6144].reshape(1, 64, 32)
    core[..., 0] = 100
    core[0, 48, 16:26] = [180, 200, 270, 210, 220, 210, 220, 3, 10, 30]
    ticks[:, :1024].reshape(1, 16, 64)[..., 56] = 260
    fetch = ticks[:, 2048:4096].reshape(1, 16, 128)
    fetch[..., 99] = 301
    fetch[0, 0, 16] = 1
    stamp = ticks[:, 6144:].reshape(1, 64, 8)
    for i in range(64):
        r, f = divmod(i, 3) if i < 48 else (i-48, 3)
        stamp[0, i] = [JOINT_MAGIC, r, i << 32 | f, 30, 150, 240, 160, 0]
    return {'times': ticks, 'cycles_per_us': 2}, core, fetch, stamp


def test_split_prefix_can_finish_on_another_block():
    raw, core, _, _ = sample()
    ends = joint_endpoints(raw)
    assert (ends['wins'], ends['remote_wins'], ends['precq_moves'], ends['unprocessed_by_prefix_moves']) == (1,1,10,20)
    assert ends['last_index_minus_last_cq']['p50'] == -5
    core[0, 48, 17] = 225
    with pytest.raises(ValueError, match='Invalid final'):
        joint_endpoints(raw)


def test_empty_get_and_no_help():
    raw, core, fetch, _ = sample()
    core[0, 48, 16:28] = 0
    fetch[0, 0, 16] = 0
    ends = joint_endpoints(raw)
    assert ends['first_doorbell']['p50'] is None
    assert ends['attempts'] == 0
    assert ends['last_index']['p50'] == 80


def test_capture_drift_separates_mean_and_median():
    assert drift([1, 1, 1, 1, 1, 2, 2, 2, 2, 12]) == (3,1)


def test_local_comparison_rejects_external_baseline(tmp_path):
    with pytest.raises(ValueError, match='same run directory'):
        run(tmp_path/'candidate-machine', tmp_path/'baseline-machine', None, 2)


def test_idle_row_core_may_finish_its_assigned_remote_row():
    raw, _, _, stamp = sample()
    # Block 0 can finish remote row 1 after its local duties have completed.
    stamp[0, 49, 2] = 3
    assert joint_endpoints(raw)['cross_row_completions'] == 1
    stamp[0, 50, 2] = 3
    with pytest.raises(ValueError, match='physical owner'):
        joint_endpoints(raw)


@pytest.mark.parametrize('workers', (1, 2, 4))
def test_final_queue_count_and_ticket_row_partition(workers):
    raw, core, fetch, stamp = sample()
    raw['q_workers'] = workers
    # Exercise the very last queue in each geometry so decoding eight
    # fixed lanes would either lose this fetch or read a nonexistent one.
    queue = 4*workers - 1
    core[0, 48+queue, 16:28] = core[0, 48, 16:28]
    core[0, 48, 16:28] = 0
    core[0, 48+queue, 23] = 16
    fetch[0, 0, 16] = 0
    fetch[0, workers-1, 3*24+16] = 1
    assert joint_endpoints(raw)['nonempty_lanes'] == 1
    if workers < 4:
        # The last standby may claim the fourth quarter of its paired comm row.
        peer = 15 % (4*workers)
        stamp[0, 48+peer, 2] = 63 << 32 | 3
        joint_endpoints(raw)
    core[0, 48+queue, 23] = 1
    with pytest.raises(ValueError, match='Pre-CQ ticket'):
        joint_endpoints(raw)


def test_early_reads_ignore_communication_core_fields_and_track_real_row():
    raw, cores, _, _ = sample()
    cores[0, 63, 16:21] = [125, 140, 159, 20, 8]
    raw['times'][:, :1024].reshape(1, 16, 64)[0, 7, 54] = 135
    raw['times'][:, 6144:].reshape(1, 64, 8)[0, 48+7, 2] = (63 << 32) | 3
    early = early_index_stats(raw)
    assert (early['hits'], early['eligible_core_calls'], early['moves']) == (1, 8, 20)
    assert early['reads_by_row'][7] == 1
    assert early['finished_by_plan'] == 0 and early['preplan_read_us']['p50'] == 5
    assert (early['entry_to_read_us']['p50'], early['read_us']['p50'], early['wait_after_read_us']['p50']) == (12.5, 7.5, 9.5)
    cores[0, 63, 18] = 119
    with pytest.raises(ValueError, match='early Index'):
        early_index_stats(raw)
    raw['q_workers'] = 4
    early = early_index_stats(raw)
    assert (early['eligible_core_calls'], early['hits'], early['read_us']['p50']) == (0, 0, None)


def test_archived_original_and_eight_core_control_use_candidate_only(tmp_path, monkeypatch):
    import json
    import ranked_joint_report as report

    control, current = tmp_path / 'previous', tmp_path / 'current'
    candidate = 'c' * 40
    for root, tag in ((control, report.JOINT_CONTROL), (current, candidate)):
        (root / 'g2').mkdir(parents=True)
        for name, value in (('candidate', tag), ('baseline', report.RANKED_BASE),
                            ('physical_npu', '0'), ('host', 'node20')):
            (root / f'{name}.txt').write_text(value)
    raw, *_ = sample()
    np.savez(current / 'lean_ticks.npz', **raw)
    calls = []
    def load(root, group, label, revision, abi):
        calls.append((root, label))
        return dict(folder=root, revision=revision, binary=revision, placement='same')
    def compare(left, right):
        return dict(same_binary=False, placement='matched', same_capture_sha256=True,
                    exact_topk_matched=dict(n=2, before_calls=2, span_p50_saved_us=3.0))
    def summarize(arm):
        return dict(**arm, event_mean_us=12., profiler_p50_us=11., span_us=[10., 12.],
                    graph_calls=2, stages_p50_max_us={k: [1., 2.] for k in report.STAGES},
                    kernel_mean_p50_us={'li': [5., 5.], 'resolve_update': [7., 7.]},
                    tail={k: {'p50': 1., 'p95': 2.} for k in (
                        'last_fourth_minus_last_primary_us', 'row_handoff_gap_us', 'fourth_copy_start_minus_plan_us')})
    for root, rev in ((control, report.RANKED_BASE), (control, report.JOINT_CONTROL), (current, candidate)):
        (root / f'ranked-{rev[:8]}.binary.sha256').write_text(rev)
    monkeypatch.setattr(report, 'load', load)
    monkeypatch.setattr(report, 'compare', compare)
    monkeypatch.setattr(report, 'summary', lambda arm: {k: v for k, v in summarize(arm).items() if k != 'folder'})
    monkeypatch.setattr(report, 'pair_placement', lambda *_args, **_kwargs: 'matched')
    report.run(current, control, None, 2, joint_control=True)
    result = json.loads((current / 'g2/comparison.json').read_text())
    assert set(result['arms']) == {'original', 'previous8', 'candidate'}
    assert result['previous8_topk_match']['n'] == 2
    assert len(calls) == 3 and (current, 'baseline') not in calls
    (current / 'physical_npu.txt').write_text('4')
    with pytest.raises(ValueError, match='physical device'):
        report.run(current, control, None, 2, joint_control=True)


def test_two_staged_prefixes_keep_separate_rows_and_waits():
    from ranked_index_pair_stats import stage_windows

    raw, cores, _, stamp = sample()
    # AIV 56 serves home row 8 and paired row 0. Its second read spans
    # 40 ticks but only spends 18 of those ticks doing useful reads.
    cores[0, 56, 16:30] = [110, 140, 165, 28, 9, 144, 184, 191, 26, 1, 16, 11, 2, 18]
    plan = raw['times'][:, :1024].reshape(1, 16, 64)[..., 54]
    plan[0, [0, 8]] = [180, 150]
    cores[0, [2, 26], 27] = [185, 170]
    stamp[0, [48, 56], 2] = (56 << 32) | 3
    both = stage_windows(raw, candidate=True)
    assert both['counts'] == dict(first_reads=1, second_reads=1, second_moves=26)
    assert both['p50_us']['first_ready_to_scatter'] == 0
    assert both['p95_us']['second_ready_to_scatter'] == 3
    assert both['p50_us']['second_busy_read'] == 9
    control = stage_windows(raw, candidate=False)
    assert control['counts'] == dict(first_reads=1)
    cores[0, 56, 26] = 257
    with pytest.raises(ValueError, match='slice budget'):
        stage_windows(raw, candidate=True)
    cores[0, 56, 26] = 16
    cores[0, 56, 25] = 9
    with pytest.raises(ValueError, match='Repeated'):
        stage_windows(raw, candidate=True)
