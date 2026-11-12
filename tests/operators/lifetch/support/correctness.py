# SPDX-License-Identifier: Apache-2.0
"""Fixed workload selection and public operator contract checks."""
from types import SimpleNamespace

from workloads import configuration

# Cases share the existing full-chain oracle; geometry is not public check input.
FIXED_CASES = {
    'basic': [(f'basic_bs{batch}', batch, 10., 512) for batch in (1, 2, 4, 8, 16)],
    'boundary': [('all_hit_budget0', 2, 0., 0), ('all_miss', 2, 100., 512),
                 ('prefetch_zero', 2, 10., 0), ('boundary_regular', 2, 10., 512)],
    'stateful': [('stateful', 2, 100., 0)],
}


def check_jobs(config):
    """Build named jobs; never accept a case filter that silently selects nothing."""
    available = {name for suite in config.suite for name, *_ in FIXED_CASES.get(suite, [])}
    available.update(suite for suite in config.suite if suite in ('init', 'contract', 'real'))
    selected = set(config.cases or available)
    unknown = selected - available
    if unknown:
        raise ValueError(f'Unknown cases for selected suites: {sorted(unknown)}; available={sorted(available)}')
    jobs = []
    for suite in config.suite:
        if suite in ('init', 'contract', 'real'):
            if suite in selected:
                jobs.append((suite, SimpleNamespace(**vars(config), operation=suite)))
            continue
        for name, batch, miss, budget in FIXED_CASES[suite]:
            if name not in selected:
                continue
            defaults = vars(configuration(['bench']))
            common = {key: getattr(config, key) for key in
                      ('transport', 'device', 'kvaf_root', 'kvaf_server', 'output', 'timeout_seconds')}
            values = {**defaults, **common, 'mode': 'check', 'operation': 'pipeline',
                      'check_suite': suite, 'check_case': name, 'batches': (batch,),
                      'miss_percent': miss, 'prefetch_tokens': budget, 'warmup': 0, 'iterations': 1}
            if suite == 'stateful':
                values.update(history=12288, resident=4096)
            jobs.append((name, SimpleNamespace(**values)))
    return jobs


def check_contracts(transport):
    """Exercise production validation rather than merely the CLI parser."""
    from kvip.operators.lifetch.geometry import prefetch_budget

    rows = []
    def rejected(name, callback):
        try:
            callback()
        except ValueError as error:
            rows.append(dict(case=name, correctness='PASS', detail=str(error)))
        else:
            rows.append(dict(case=name, correctness='FAIL', detail='invalid input accepted'))

    for batch in (0, 17, 32):
        if transport == 'urma':
            from kvip.operators.lifetch.binding import LIFetchBinding
            rejected(f'reject_bs{batch}', lambda b=batch: LIFetchBinding(
                None, None, [], None, batch=b, history=131072, resident=16384))
        else:
            from kvip.operators.lifetch_ubmem.build import UbmemGeometry
            rejected(f'reject_bs{batch}', lambda b=batch: UbmemGeometry(
                b, 131072, 16384, 32, 656, 32, 8, 'Ascend950PR_9599'))
    for budget in (-1, 2049):
        rejected(f'reject_budget{budget}', lambda n=budget: prefetch_budget(n, intermediate_fetches=8))
    rejected('reject_windows0', lambda: prefetch_budget(512, intermediate_fetches=0))
    return rows


def boundary_evidence(selected, initial_hot_valid, initial_layout, prefetch, final_posts):
    """Observed branch coverage, using logical identities and actual device counts."""
    import numpy as np

    selected = np.asarray(selected)
    validity, layout = np.asarray(initial_hot_valid), np.asarray(initial_layout)
    prefetch, posts = np.asarray(prefetch), np.asarray(final_posts)
    batch = len(selected)
    if selected.ndim != 3 or selected.shape[1] != 4 or validity.ndim != 2:
        raise ValueError('Boundary evidence requires B/Q4 selections and B/R hot validity')
    if layout.shape != validity.shape or len(validity) != batch or prefetch.shape != (batch,) or posts.shape != (batch, 4):
        raise ValueError('Boundary evidence shapes differ')
    if np.any(prefetch < 0) or np.any(posts < 0) or np.any((validity != 0) & (validity != 1)):
        raise ValueError('Boundary evidence contains invalid counters/validity')
    requests = []
    for row in range(batch):
        demand = np.unique(selected[row])
        cold = int((~np.isin(demand, layout[row, validity[row] != 0])).sum())
        seen, duplicate = set(), []
        for query in selected[row]:
            tokens = set(map(int, query))
            duplicate.append(len(seen & tokens))
            seen.update(tokens)
        requests.append(dict(request=row, selected_unique=len(demand), initial_cold=cold,
                             initial_hot_valid=int(validity[row].sum()), capacity=validity.shape[1],
                             cross_query_duplicates=duplicate, prefetch=int(prefetch[row]),
                             final_by_query=posts[row].astype(int).tolist(), final=int(posts[row].sum())))
    return requests


def require_boundary(case, phases):
    """A named boundary passes only when measured inputs/counts prove it ran."""
    regular = phases['regular']
    if case == 'all_hit_budget0':
        assert all(r['initial_cold'] == 0 and r['prefetch'] == 0 and r['final'] == 0
                   for r in regular), 'all-hit with zero budget must issue zero reads'
    elif case == 'all_miss':
        assert all(r['initial_hot_valid'] == r['capacity'] and
                   r['initial_cold'] == r['selected_unique'] > 0 and
                   r['prefetch'] + r['final'] >= r['selected_unique']
                   for r in regular), 'all-miss must start with a full unrelated hot cache'
    elif case == 'prefetch_zero':
        assert all(r['prefetch'] == 0 and r['initial_cold'] > 0 and
                   r['final'] >= r['initial_cold'] for r in regular), 'zero budget must retain final repair'
    elif case == 'boundary_regular':
        assert all(sum(r['cross_query_duplicates'][1:]) > 0 and
                   r['final_by_query'][2:] == [0, 0] for r in regular), 'duplicate Q3/Q4 must issue no new repair'
        cold = phases['cold-start']
        assert all(r['initial_hot_valid'] == 0 and r['initial_cold'] == r['selected_unique'] and
                   r['prefetch'] + r['final'] >= r['selected_unique']
                   for r in cold), 'cold-start must actually clear validity and fetch all selected tokens'


def stateful_inputs(directory):
    """Real LI input: fixed Index/KV, query-only A/A/B+C/A residency cycle."""
    from pathlib import Path
    import numpy as np
    from replay_inputs import ReplayRound
    from reporting import atomic_json

    batch, history, resident, heads = 2, 12288, 4096, 32
    initial = np.broadcast_to(np.arange(history), (batch, history)).copy()
    keys = np.zeros((batch, history, 128), dtype=np.uint8)
    for feature, begin in enumerate((4096, 6144, 8192)):
        keys[:, begin:begin + 2048, feature] = 0x38  # Exactly 1 in E4M3.
    geometry = dict(batch=batch, history=history, resident=resident, heads=heads,
                    token_bytes=656, output_stride=resident + 128)
    manifest = Path(directory) / 'stateful_input.json'
    phases = ('load_A', 'hit_A', 'evict_A_with_BC', 'reload_A')
    atomic_json(manifest, dict(input='synthetic', geometry=geometry, phases=phases,
                              expected_reads_per_request=[2048, 0, 4096, 2048], prefetch_tokens=0,
                              state_reset='sequence boundary only; fixed Index and KV; queries change'))
    identities = tuple((0, f'stateful-{row}', row + 1) for row in range(batch))
    class Sequence:
        rounds = tuple((SimpleNamespace(source_round=n, manifest_path=manifest),) for n in range(1, 5))
        def __iter__(self):
            for number in range(1, 5):
                features = (1, 2, 1, 2) if number == 3 else (0, 0, 0, 0)
                query = np.zeros((batch, 4, heads, 128), dtype=np.uint8)
                scores = np.full((batch, 4, history), 0x8000, dtype=np.uint16)
                for q, feature in enumerate(features):
                    query[:, q, :, feature] = 0x38
                    begin = (4096, 6144, 8192)[feature]
                    scores[:, q, begin:begin + 2048] = 0xC200  # sortable BF16(+32).
                ends = np.broadcast_to(np.arange(history - 3, history + 1), (batch, 4)).copy()
                yield ReplayRound(number, number, identities, geometry, keys.copy(),
                    np.ones((batch, history), np.float32), query,
                    np.full((batch, 4, heads), 0x3F80, np.uint16),
                    np.ones((batch, 4, heads), np.float32), ends,
                    np.zeros((batch, 2), np.int64), np.ones((batch, history), np.uint8),
                    tuple(tuple((row, page) for page in range(history // 128)) for row in range(batch)),
                    initial.copy(), np.ones((batch, resident), np.int32),
                    np.broadcast_to(np.arange(2048), (batch, 2048)).copy(), (str(manifest),),
                    scores, np.arange(history)[None, None, :] < ends[:, :, None])
    return Sequence()


def require_stateful_round(source_round, hot_tokens, prefetch, final):
    """Assert actual reads and eviction, not a host-simulated LRU trajectory."""
    import numpy as np
    expected_reads = {1: 2048, 2: 0, 3: 4096, 4: 2048}[source_round]
    expected = set(range(6144, 10240)) if source_round == 3 else set(range(4096, 6144))
    assert np.all(np.asarray(prefetch) == 0), 'stateful case must not speculate'
    assert np.all(np.asarray(final) == expected_reads), (
        f'round={source_round}: reads={list(final)}, expected={expected_reads}/request')
    for tokens in hot_tokens:
        tokens = set(map(int, tokens))
        assert expected <= tokens, f'round={source_round}: selected tokens are not persistent hot'
        if source_round == 3:
            assert not tokens.intersection(range(4096, 6144)), 'B/C failed to evict A'
