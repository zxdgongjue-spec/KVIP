"""CPU regressions for the unified LIFetch workloads module."""
import hashlib
import json

import numpy as np
import pytest
import torch
from correctness import require_stateful_round, stateful_inputs, validate_topk
from workloads import (
    BASE_ARRAYS,
    CHECK_SUITES,
    REAL_ARRAYS,
    SCORE_ARRAYS,
    CaptureReference,
    ReplayPagePlan,
    assemble_round,
    checked_path,
    configuration,
    convert_capture_bundle,
    load_score_capture,
    prepare_compact_fixture,
    prepare_full_chain_fixture,
    score_values,
)


def test_defaults_and_check_has_no_geometry_options():
    check = configuration(['check'])
    assert check.transport == 'urma' and check.device == 0
    assert check.suite == CHECK_SUITES[:-1]
    bench = configuration(['bench'])
    assert bench.seed == 43 and bench.batches == (1, 4, 8, 16)
    assert bench.queries == 4
    with pytest.raises(SystemExit):
        configuration(['check', '--batches', '16'])


def test_real_preserves_source_selection_and_rejects_even_explicit_default_geometry():
    args = ['bench', '--input', 'real', '--trace-root', '/trace', '--group', '10', '--round-range', '51:51']
    parsed = configuration(args)
    assert parsed.round_range == (51, 51) and parsed.batches is None
    for key, value in [('seed', '43'), ('queries', '4'), ('batches', '16'), ('history', '131072')]:
        with pytest.raises(SystemExit):
            configuration(args + ['--' + key, value])


@pytest.mark.parametrize('args', [
    ['check', '--suite', 'real'], ['check', '--suite', 'all,basic'],
    ['bench', '--queries', '2'], ['bench', '--batches', '17'],
    ['bench', '--batches', '1,1'], ['bench', '--trace-root', '/trace'],
    ['bench', '--input', 'real'], ['bench', '--seed', '-1'],
    ['diagnose', '--level', 'stage'], ['diagnose', '--stages', 'unknown'],
    ['diagnose', '--stages', 'all,fetch'],
    ['diagnose', '--algorithm-stats'], ['diagnose', '--fetch-bandwidth'],
])
def test_invalid_or_removed_interfaces_fail_early(args):
    with pytest.raises(SystemExit):
        configuration(args)


def test_diagnose_reuses_bench_config_and_has_only_two_extra_selectors():
    args = configuration(['diagnose', '--level', '3', '--stages', 'update,fetch'])
    assert args.stages == ('update', 'fetch')
    assert args.iterations == 20 and args.seed == 43
    assert 'algorithm_stats' not in vars(args)


@pytest.mark.parametrize('flag', ['--early-fetch', '--avoid-prefetch-slots', '--legacy-fetch-layout',
                                   '--parallel-fetch-finalize', '--optimized-resolve', '--trace-rounds'])
def test_old_pipeline_flags_are_not_public(flag):
    with pytest.raises(SystemExit):
        configuration(['bench', flag])


def test_compact_fixture_layout_and_residency(tmp_path):
    roots = [tmp_path/'low', tmp_path/'high']
    infos = [prepare_compact_fixture(root, 8192, 4096, miss, seed=17)
             for root, miss in zip(roots, (5, 50))]
    assert infos[0]['model'] == 'compact_builtin_v1'
    assert sum(infos[0]['final_misses_per_query']) < sum(infos[1]['final_misses_per_query'])
    for root in roots:
        scores = np.load(root/'logical_scores.npy')
        order = np.load(root/'physical_to_logical.npy')
        physical = scores[:, order]
        selected = np.argpartition(physical, -2048, axis=1)[:, -2048:]
        validate_topk(selected, physical)
        assert np.isfinite(scores).all()
        assert (root/'arg0.bin').stat().st_size == 4*32*128
        assert (root/'arg1.bin').stat().st_size == 8192*128
    assert (roots[0]/'arg0.bin').read_bytes() == (roots[1]/'arg0.bin').read_bytes()
    # Residency changes positions, not the score distribution.
    np.testing.assert_array_equal(np.sort(np.load(roots[0]/'logical_scores.npy'), axis=1),
                                  np.sort(np.load(roots[1]/'logical_scores.npy'), axis=1))


def test_score_oracle_accepts_ties_and_rejects_bad_or_duplicate_rows():
    scores = np.concatenate((np.full(100, 4.), np.full(2000, 2.), np.zeros(1000)))[None, :]
    valid = np.arange(2048)
    validate_topk(valid, scores)
    alternate = valid.copy()
    alternate[-1] = 2099
    validate_topk(alternate, scores)
    invalid = valid.copy()
    invalid[-1] = 3000
    with pytest.raises(AssertionError):
        validate_topk(invalid, scores)
    invalid[-1] = 0
    with pytest.raises(AssertionError):
        validate_topk(invalid, scores)


def reference(tmp_path, rank=0, number=1):
    directory = tmp_path / f'dp{rank}-round{number}'
    directory.mkdir()
    path = capture_file(directory)
    manifest = json.loads(path.read_text())
    manifest['geometry'].update(token_bytes=656, output_stride=2176)
    path.write_text(json.dumps(manifest))
    return CaptureReference(rank, number, 100 + number, 10, path, (('request', 0),))


def test_merge_preserves_dp_identity_and_query_bytes(tmp_path):
    first, second = reference(tmp_path), reference(tmp_path, rank=1)
    round_input = assemble_round([first, second])
    assert round_input.geometry['batch'] == 2
    assert round_input.identities == ((0, 'request', 0), (1, 'request', 0))
    assert round_input.page_keys[0][0] == (0, 0)
    assert round_input.page_keys[1][0] == (1, 0)
    assert round_input.weights.dtype == np.uint16  # Preserve BF16 bits.
    assert round_input.query.shape == (2, 4, 2, 128)


def test_current_path_layout_drives_index_not_captured_hot_layout(tmp_path):
    round_input = assemble_round([reference(tmp_path)])
    current = np.arange(4096, dtype=np.int64)[None]
    current[:, [0, 3000]] = current[:, [3000, 0]]
    keys, scales, valid = round_input.index_for_layout(current)
    assert scales[0, 0] == 1095 and scales[0, 3000] == 4095
    assert keys[0, 0, 0] == 1095 % 256
    assert valid.all()
    # Preparing another round must not mutate captured evidence or supplied state.
    assert round_input.captured_layout[0, 0] == 4095
    assert current[0, 0] == 3000
    current[0, 0] = 0
    with pytest.raises(ValueError, match='permutation'):
        round_input.index_for_layout(current)


def test_replay_rejects_unaligned_or_duplicate_dp_inputs(tmp_path):
    first = reference(tmp_path)
    second = reference(tmp_path, rank=1, number=2)
    with pytest.raises(ValueError, match='aligned'):
        assemble_round([first, second])
    with pytest.raises(ValueError, match='Duplicate DP'):
        assemble_round([first, first])


@pytest.mark.parametrize('mutation, message', [
    ('identity', 'Request identity'), ('bounds', 'bounds regressed'),
    ('pages', 'DDR page identity'),
])
def test_sequence_rejects_invalid_continuity(tmp_path, monkeypatch, mutation, message):
    import workloads as replay_inputs
    first = reference(tmp_path)
    second = reference(tmp_path, number=2)
    initial = assemble_round([first])
    following = assemble_round([second])
    if mutation == 'identity':
        following.identities = ((0, 'another-request', 0),)
    elif mutation == 'bounds':
        following.query_ends[0, 0] -= 1
    else:
        following.page_keys = (((0, 999), *following.page_keys[0][1:]),)
    monkeypatch.setattr(replay_inputs, 'select_series', lambda *args: [first, second])
    monkeypatch.setattr(replay_inputs, 'assemble_round',
                        lambda refs: initial if refs[0].source_round == 1 else following)
    sequence = replay_inputs.ReplaySequence(tmp_path, 10, (1, 2))
    iterator = iter(sequence)
    assert next(iterator) is initial
    with pytest.raises(ValueError, match=message):
        next(iterator)


def test_sequence_iteration_does_not_carry_test_state_between_repetitions(tmp_path, monkeypatch):
    import workloads as replay_inputs
    references = [reference(tmp_path), reference(tmp_path, number=2)]
    monkeypatch.setattr(replay_inputs, 'select_series', lambda *args: references)
    sequence = replay_inputs.ReplaySequence(tmp_path, 10, (1, 2))
    first_pass = list(sequence)
    first_pass[0].captured_hot_valid[:] = 0
    first_pass[0].logical_keys[:] = 0
    second_pass = list(sequence)
    assert second_pass[0].captured_hot_valid.all()
    assert np.any(second_pass[0].logical_keys)
    assert [entry.source_round for entry in second_pass] == [1, 2]


def test_page_payload_preserves_aliases_and_later_abc_publication(tmp_path):
    from workloads import ReplayPagePlan
    first = assemble_round([reference(tmp_path), reference(tmp_path, rank=1)])
    # Duplicate logical pages within DP0 model a shared captured page alias.
    first.page_keys = (((0, 0), (0, 0), *first.page_keys[0][2:]), first.page_keys[1])
    later = assemble_round([reference(tmp_path, number=2), reference(tmp_path, rank=1, number=2)])
    later.page_keys = first.page_keys
    first.page_keys = ((None, *first.page_keys[0][1:]), first.page_keys[1])
    plan = ReplayPagePlan([first, later])
    assert plan.page_indices[0, 0] == plan.page_indices[0, 1]
    assert plan.page_indices[0, 0] != plan.page_indices[1, 0]
    np.testing.assert_array_equal(plan.token_payload(0, [0]), plan.token_payload(0, [128]))
    np.testing.assert_array_equal(plan.token_payload(0, np.arange(128)),
                                  plan.page_payload([plan.page_indices[0, 0]]))
    assert not np.array_equal(plan.token_payload(0, [0]), plan.token_payload(1, [0]))


def test_topk_oracle_accepts_cutoff_ties_but_not_bad_rank_or_duplicates(tmp_path):
    inputs = assemble_round([reference(tmp_path)])
    selected = np.broadcast_to(np.arange(2048, dtype=np.int32), (1, 4, 2048)).copy()
    inputs.validate_topk(selected)
    # All original scores tie; selecting a different set is valid.
    inputs.validate_topk(selected + 2048)
    inputs.logical_score_keys[0, 0, 3000] += 1
    with pytest.raises(AssertionError, match='ranking differs'):
        inputs.validate_topk(selected)
    selected[0, 0, 0] = 3000
    inputs.validate_topk(selected)
    selected[0, 0, 1] = 3000
    with pytest.raises(AssertionError, match='duplicates'):
        inputs.validate_topk(selected)


def capture_file(tmp_path, *, real=True):
    b, h, r, heads = 1, 4096, 2048, 2
    arrays = {name: np.zeros(0, dtype=np.int32) for name in BASE_ARRAYS | SCORE_ARRAYS}
    metadata = np.zeros((5, 4), dtype=np.int64)
    metadata[2] = np.arange(4)
    arrays.update({
        'pre.logical_ids': np.arange(h, dtype=np.int32)[None, ::-1].copy(),
        'pre.hot_valid': np.ones((b, r), dtype=np.uint8),
        'pre.address_valid': np.ones((b, h), dtype=np.uint8),
        'pre.row_owners': np.zeros(b, dtype=np.int32),
        'pre.metadata': metadata,
        'pre.query_ends': np.full((b, 4), h, dtype=np.int32),
        'pre.lru_slots': np.arange(r, dtype=np.int32)[None],
        'pre.tail_bounds': np.array([[h - 128, h]], dtype=np.int32),
        'pre.ddr_page_slots': np.arange(h // 128, dtype=np.int32)[None],
        'post.topk': np.broadcast_to(np.arange(2048, dtype=np.int32), (b, 4, 2048)).copy(),
        'post.score_keys': np.full((b, 4, h), 0xBF80, dtype=np.uint16),
    })
    if real:
        arrays.update({
            'real.keys': np.broadcast_to((np.arange(h) % 256).astype(np.uint8)[None, :, None], (b, h, 128)).copy(),
            'real.key_scales': np.arange(h, dtype=np.float32)[None],
            'real.query': np.zeros((b, 4, heads, 128), dtype=np.uint8),
            'real.weights': np.zeros((b, 4, heads), dtype=np.uint16),
            'real.query_scale': np.ones((b, 4, heads), dtype=np.float32),
        })
    archive = tmp_path / 'scores.npz'
    np.savez_compressed(archive, **arrays)
    manifest = {
        'schema_version': 4, 'kind': 'kvaf_decode_score_trace', 'content': 'scores',
        'status': 'complete', 'validation': {'passed': True, 'kind': 'score_trace_integrity'},
        'real_inputs_saved': real, 'metadata': {'capture_real_inputs': real},
        'geometry': {'batch': b, 'history': h, 'resident': r, 'heads': heads},
        'archive': {'file': archive.name, 'bytes': archive.stat().st_size,
                    'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()},
        'arrays': {name: {'dtype': value.dtype.name, 'shape': list(value.shape)}
                   for name, value in arrays.items()},
    }
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return path


def test_actual_npz_roundtrip_remaps_physical_index_by_logical_identity(tmp_path):
    capture = load_score_capture(capture_file(tmp_path))
    keys, scales = capture.logical_keys(0)
    assert capture.executable and REAL_ARRAYS <= capture.arrays.keys()
    np.testing.assert_array_equal(scales, np.arange(4096, dtype=np.float32)[::-1])
    assert keys[0, 0] == 255 and keys[-1, 0] == 0


def test_score_only_can_analyze_but_cannot_replay(tmp_path):
    path = capture_file(tmp_path, real=False)
    with pytest.raises(ValueError, match='No real Q/Index'):
        load_score_capture(path)
    capture = load_score_capture(path, require_real=False)
    assert not capture.executable
    with pytest.raises(ValueError, match='no executable'):
        capture.logical_keys(0)


def test_tampered_archive_and_manifest_rejected(tmp_path):
    path = capture_file(tmp_path)
    with (tmp_path / 'scores.npz').open('ab') as output:
        output.write(b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        load_score_capture(path)
    path = capture_file(tmp_path)
    manifest = json.loads(path.read_text())
    manifest['arrays']['real.query']['shape'][0] = 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='real.query'):
        load_score_capture(path)


def test_prefix_topk_and_identity_validation(tmp_path):
    capture = load_score_capture(capture_file(tmp_path))
    from workloads import validate_inputs
    capture.arrays['post.topk'][0, 0, 0] = 4096
    with pytest.raises(ValueError, match='prefix'):
        validate_inputs(capture)
    capture.arrays['post.topk'][0, 0, 0] = 0
    capture.arrays['pre.logical_ids'][0, 0] = 0
    with pytest.raises(ValueError, match='permutation'):
        validate_inputs(capture)


def test_path_escape_and_bf16_sortable_decoding(tmp_path):
    outside = tmp_path.parent / 'outside-trace-test'
    outside.write_text('not part of capture')
    try:
        with pytest.raises(ValueError, match='outside capture'):
            checked_path(tmp_path, '../outside-trace-test')
    finally:
        outside.unlink()
    np.testing.assert_array_equal(score_values([0x407F, 0x8000, 0xBF80]), [-1., 0., 1.])


def test_series_uses_source_round_and_preserves_request_identity(tmp_path):
    from workloads import select_series
    dp = tmp_path / 'dp2'
    dp.mkdir()
    rounds = []
    for number in (1, 2):
        directory = dp / f'captured-{number}'
        directory.mkdir()
        path = capture_file(directory)
        manifest = json.loads(path.read_text())
        manifest['metadata'].update({
            'dp_rank': 2, 'group': 10,
            'context': {'round': 100 + number, 'global_decode_confirmed': True},
            'requests': [{'request_id': 'req-a', 'owner': 0, 'query_begin': 0, 'query_stop': 4}],
        })
        path.write_text(json.dumps(manifest))
        rounds.append({'round_index': number, 'scheduler_round': 100 + number,
                       'groups': [{'group': 10, 'manifest': str(path.relative_to(dp))}]})
    series = {
        'dp_rank': 2, 'dp_ranks': [2], 'schema_version': 1,
        'kind': 'kvaf_decode_snapshot_series', 'status': 'complete', 'content': 'scores',
        'batch': 1, 'queries': 4, 'completed_rounds': 2, 'expected_rounds': 2,
        'request_owner_signature': [{'request_id': 'req-a', 'owner': 0}], 'rounds': rounds,
    }
    series_path = dp / 'series.json'
    series_path.write_text(json.dumps(series))
    references = select_series(tmp_path, 10, (2, 2))
    assert len(references) == 1
    assert references[0].source_round == 2 and references[0].scheduler_round == 102
    assert references[0].dp_rank == 2 and references[0].request_signature == (('req-a', 0),)
    with pytest.raises(ValueError, match='absent'):
        select_series(tmp_path, 10, (2, 3))
    series['dp_ranks'] = [0, 2]
    series_path.write_text(json.dumps(series))
    with pytest.raises(ValueError, match='DP capture is missing'):
        select_series(tmp_path, 10, (1, 2))
    series['dp_ranks'] = [2]
    series['rounds'][1]['scheduler_round'] = 104
    series_path.write_text(json.dumps(series))
    with pytest.raises(ValueError, match='Nonconsecutive'):
        select_series(tmp_path, 10, (1, 2))


def test_stateful_changes_query_only_and_has_exact_topk_without_ties(tmp_path):
    sequence = stateful_inputs(tmp_path)
    rounds = list(sequence)
    assert [row.source_round for row in rounds] == [1, 2, 3, 4]
    for inputs in rounds:
        assert np.array_equal(inputs.logical_keys, rounds[0].logical_keys)
        assert np.array_equal(inputs.captured_layout, rounds[0].captured_layout)
        assert inputs.geometry['resident'] == 4096
        # All operands are FP8 {0,1}; compute real positive dot products and
        # sum 32 identical heads, independently from the supplied score oracle.
        assert set(np.unique(inputs.query)) == {0, 0x38}
        assert np.all(inputs.query == inputs.query[:, :, :1])
        q = (inputs.query[:, :, 0] == 0x38).astype(np.float32)
        k = (inputs.logical_keys == 0x38).astype(np.float32)
        scores = np.einsum('bqd,btd->bqt', q, k) * 32
        selected = np.stack([np.stack([np.flatnonzero(query == 32) for query in row]) for row in scores])
        inputs.validate_topk(selected)
        assert all(len(np.unique(row)) == (4096 if inputs.source_round == 3 else 2048)
                   for row in selected)
    assert np.array_equal(rounds[0].query, rounds[1].query)
    assert np.array_equal(rounds[0].query, rounds[3].query)
    assert not np.array_equal(rounds[0].query, rounds[2].query)
    assert ReplayPagePlan(sequence).page_count == 2 * 12288 // 128


@pytest.mark.parametrize('round_number,reads,tokens', [
    (1, 2048, range(4096, 6144)), (2, 0, range(4096, 6144)),
    (3, 4096, range(6144, 10240)), (4, 2048, range(4096, 6144)),
])
def test_each_transition_requires_actual_reads_and_persistent_residency(round_number, reads, tokens):
    require_stateful_round(round_number, [tokens, tokens], [0, 0], [reads, reads])
    with pytest.raises(AssertionError, match='reads='):
        require_stateful_round(round_number, [tokens], [0], [reads + 1])
    with pytest.raises(AssertionError, match='persistent hot'):
        require_stateful_round(round_number, [[]], [0], [reads])


def test_eviction_cannot_be_faked_by_only_checking_new_tokens():
    with pytest.raises(AssertionError, match='evict A'):
        require_stateful_round(3, [range(4096, 10240)], [0], [4096])


def test_stateful_job_records_actual_fixed_geometry_and_budget():
    from correctness import check_jobs
    from workloads import configuration
    [(name, config)] = check_jobs(configuration(['check', '--suite', 'stateful']))
    assert name == 'stateful'
    assert (config.batches, config.history, config.resident, config.prefetch_tokens) == ((2,), 12288, 4096, 0)
    assert config.warmup == 0 and config.iterations == 1


def test_seed_is_reproducible_and_changes_real_li_inputs(tmp_path):
    roots = [tmp_path / str(i) for i in range(3)]
    for root, seed in zip(roots, [43, 43, 44]):
        info = prepare_full_chain_fixture(root, 8192, 4096, 10, index_heads=32, seed=seed)
        assert info['seed'] == seed
    for name in ('arg1.bin', 'arg4.bin', 'logical_scores.npy'):
        assert (roots[0] / name).read_bytes() == (roots[1] / name).read_bytes()
        assert (roots[0] / name).read_bytes() != (roots[2] / name).read_bytes()
    for root in roots:
        scores = np.load(root / 'logical_scores.npy')
        assert scores.shape == (4, 8192)


def test_exact_miss_boundaries_for_every_query(tmp_path):
    for heads in (32, 64):
        for miss in (0, 10, 100):
            root = tmp_path / f'{heads}-{miss}'
            info = prepare_full_chain_fixture(root, 131072, 16384, miss, index_heads=heads, seed=43)
            target = round(2048 * miss / 100)
            assert info['final_misses_per_query'] == [target] * 4
            scores = np.load(root / 'logical_scores.npy')
            order = np.load(root / 'physical_to_logical.npy')
            for query in scores:
                cutoff = np.partition(query, -2048)[-2048]
                assert np.count_nonzero(query >= cutoff) == 2048
                assert np.count_nonzero(query[order[16384:]] >= cutoff) == target


def test_current_schema_preserved(tmp_path):
    from workloads import load_score_capture
    source_dir = tmp_path/'source'
    source_dir.mkdir()
    source = capture_file(source_dir)
    convert_capture_bundle(source, tmp_path/'output')
    capture = load_score_capture(tmp_path/'output/schema4/manifest.json')
    assert capture.executable
    assert capture.arrays['real.keys'].shape == (1,4096,128)


def test_compact_generator_needs_no_profile_and_is_reproducible():
    from workloads import generate_operands
    first = generate_operands(8192, seed=17)
    second = generate_operands(8192, seed=17)
    assert first['query'].shape == (4, 32, 128)
    assert first['keys'].shape == (8192, 128)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
        assert np.isfinite(first[name]).all()
    zero = np.all(first['keys'] == 0, axis=1)
    assert zero.sum() == round(8192 * .0099182)
    assert np.all(first['key_scale'][zero] == 1)
    assert not np.array_equal(first['query'], generate_operands(8192, seed=18)['query'])
    with pytest.raises(ValueError):
        generate_operands(1)


def test_legacy_executable_replay_preserves_real_bits_and_padding(tmp_path):
    from workloads import (
        ReplayPagePlan,
        ReplaySequence,
        convert_legacy_replay,
        load_score_capture,
        prepare_full_chain_fixture,
    )
    root = tmp_path/'fixture'
    prepare_full_chain_fixture(root, 8192, 4096, 10, index_heads=32)
    def array(index, dtype, shape):
        return np.fromfile(root/f'arg{index}.bin', dtype).reshape(shape)
    order = np.load(root/'physical_to_logical.npy')
    scores = np.load(root/'logical_scores.npy')
    top = np.argpartition(scores, -2048, axis=1)[:, -2048:]
    q = array(0, np.uint8, (4, 32, 128))
    keys = array(1, np.uint8, (8192, 128))
    source = tmp_path/'legacy.pt'
    torch.save(dict(metadata=dict(schema=1, sample=0, group=3), before=dict(
        query=torch.from_numpy(np.concatenate((q, np.zeros_like(q)))),
        keys=torch.from_numpy(np.stack((np.zeros_like(keys), keys))),
        key_scale=torch.from_numpy(np.stack((np.zeros(8192, np.float32), array(4, np.float32, (8192,))))),
        weights=torch.from_numpy(np.tile(array(2, np.float32, (4,32)), (2,1))).to(torch.bfloat16),
        query_scale=torch.from_numpy(np.tile(array(3, np.float32, (4,32)), (2,1))),
        row_owners=torch.tensor([1,-1]), query_valid=torch.tensor([1,1,1,1,0,0,0,0]),
        unpack_indices=torch.arange(8), logical_ids=torch.from_numpy(np.stack((order,order))),
        hot_valid=torch.ones(2,4096,dtype=torch.int32),
        query_ends=torch.full((8,),8192), tail_bounds=torch.tensor([[8192,8192],[0,0]]),
        lru_slots=torch.arange(2048).repeat(2,1)),
        after=dict(logical_topk=torch.from_numpy(np.stack((top,np.full_like(top,-1)))))), source)
    bundle = tmp_path/'bundle'
    convert_capture_bundle(source,bundle)
    output = tmp_path/'replay'
    report = convert_legacy_replay(bundle,output)
    assert report['batch'] == 1 and report['group'] == 3
    capture = load_score_capture(output)
    np.testing.assert_array_equal(capture.arrays['real.keys'][0], keys)
    np.testing.assert_array_equal(capture.arrays['real.query'][0], q)
    sequence = ReplaySequence(output,3,(1,1))
    first = next(iter(sequence))
    first.validate_topk(top[None].astype(np.int32))
    replay_keys, replay_scales, valid = first.index_for_layout(first.captured_layout)
    np.testing.assert_array_equal(replay_keys[0], keys)
    assert ReplayPagePlan(sequence).page_count == 64
    with pytest.raises(ValueError, match='independent'):
        ReplaySequence(output,3,(1,2))
    with pytest.raises(ValueError, match='group'):
        ReplaySequence(output,0,(1,1))
    from execution import dispatch
    for kind, input_path in [('raw', source), ('bundle', bundle)]:
        destination = tmp_path / ('converted-' + kind)
        config = configuration(['convert-trace', str(input_path), '--output', str(destination)])
        assert dispatch(config) == 0
        result = json.loads((destination / 'conversion.json').read_text())
        assert result['status'] == 'COMPLETE' and len(result['captures']) == 1
        converted = load_score_capture(destination / 'capture-00000')
        for key in capture.arrays:
            np.testing.assert_array_equal(converted.arrays[key], capture.arrays[key])
        with pytest.raises(ValueError, match='already exists'):
            dispatch(config)
    with (output/'inputs.npz').open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        load_score_capture(output)


def test_legacy_bundle_preserves_bf16_and_rejects_overwrite(tmp_path):
    source = tmp_path / 'legacy.pt'
    values = torch.tensor([[-1.0, 1.5]], dtype=torch.bfloat16)
    torch.save(dict(metadata=dict(schema=1), before=dict(weights=values), after={}), source)
    destination = tmp_path / 'bundle'
    manifest = convert_capture_bundle(source, destination)
    assert manifest['arrays']['before.weights']['encoding'] == 'raw_bytes'
    with np.load(destination / 'arrays.npz', allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['before.weights'], values.view(torch.uint8).numpy())
    with pytest.raises(ValueError, match='already exists'):
        convert_capture_bundle(source, destination)
