# SPDX-License-Identifier: Apache-2.0
"""Numerical oracles, invariants and fixed check coverage."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from workloads import configuration

# Cases share the existing full-chain oracle; geometry is not public check input.
FIXED_CASES = {
    'basic': [(f'basic_bs{batch}', batch, 10., 512) for batch in (1, 2, 4, 8, 16)],
    'boundary': [('all_hit_budget0', 2, 0., 0), ('all_miss', 2, 100., 512),
                 ('prefetch_zero', 2, 10., 0), ('boundary_regular', 2, 10., 512)],
    'stateful': [('stateful', 2, 100., 0)],
}


TOPK, BLOCK_TOKENS, TOKEN_BYTES = 2048, 128, 656
FETCH_AB_MAGIC = 0x4B5641464247444F


def validate_terminal_receipts(counts, statuses, queue_views):
    """Check each Q's own KVAF acknowledgments after the graph has finished."""
    queues = statuses[np.asarray(queue_views)].reshape(*counts.shape, -1)
    long_rows = counts.sum(axis=1, keepdims=True) >= 256
    expected = np.where(long_rows, np.minimum(counts, 4), counts != 0)
    valid = (
        (queues[:, :, 0] == FETCH_AB_MAGIC)
        & (queues[:, :, 1] == counts)
        & (queues[:, :, 2] == 0)
        & (queues[:, :, 3] == expected)
    )
    if not valid.all():
        row, query = np.argwhere(~valid)[0]
        raise RuntimeError(
            f"Final Get row={row} Q={query + 1} expected {counts[row, query]} requests and "
            f"{expected[row, query]} completions, KVAF status={queues[row, query, :4].tolist()}"
        )


def validate_topk(ids, scores):
    ids = np.asarray(ids).reshape(-1, TOPK)
    scores = np.asarray(scores).reshape(-1, scores.shape[-1])
    for i, selected in enumerate(ids):
        row = scores[i % len(scores)]
        if selected.min() < 0 or selected.max() >= len(row) or len(np.unique(selected)) != TOPK:
            raise AssertionError('Compact TopK has invalid/duplicate positions')
        cutoff = np.partition(row, -TOPK)[-TOPK]
        # One BF16 spacing bounds CPU/device reduction differences at the cutoff.
        tolerance = max(float(abs(cutoff)) * 2**-7, np.finfo(np.float32).tiny)
        if row[selected].min() < cutoff-tolerance:
            raise AssertionError('Compact TopK selected a score below the cutoff')
        mandatory = np.flatnonzero(row > cutoff+tolerance)
        if not np.isin(mandatory, selected).all():
            raise AssertionError('Compact TopK omitted a score above the cutoff')


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
                      'miss_percent': miss, 'prefetch_tokens': budget,
                      'warmup': getattr(config, 'check_warmup', 0),
                      'iterations': getattr(config, 'check_iterations', 1)}
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
    from kvip.operators.lifetch.geometry import li_geometry, trigger_tokens
    for name, history, resident in (
        ('unaligned_history', 131073, 16384), ('unaligned_resident', 131072, 16385),
        ('resident_too_small', 131072, 1024), ('no_cold_history', 16384, 16384),
    ):
        rejected(f'reject_{name}', lambda h=history, r=resident: trigger_tokens(2, h, r))
    for cores in (0, 33):
        rejected(f'reject_aic{cores}', lambda n=cores: li_geometry(2, aic_count=n))
    for window in (0, 481):
        rejected(f'reject_window_budget{window}', lambda n=window: prefetch_budget(512, window_limit=n))
    if transport == 'ubmem':
        defaults = dict(batch=2, history=131072, resident=16384, heads=32,
                        token_bytes=656, aic_count=32, intermediate_fetches=8, soc='Ascend950PR_9599')
        for key, value in (('heads', 16), ('token_bytes', 128), ('intermediate_fetches', 33),
                           ('soc', 'invalid'), ('history', 211200)):
            rejected(f'reject_{key}', lambda k=key, v=value: UbmemGeometry(**(defaults | {k: v})))
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
    if case not in {'all_hit_budget0', 'all_miss', 'prefetch_zero', 'boundary_regular'}:
        raise ValueError(f'Unknown boundary case: {case}')
    regular = phases['regular']
    assert regular, 'Boundary case has no observed requests'
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
        require_transfer_coverage(phases.get('transfer', []))
        assert phases.get('padding') == 'PASS', 'Padding was not checked'
        cold = phases['cold-start']
        assert all(r['initial_hot_valid'] == 0 and r['initial_cold'] == r['selected_unique'] and
                   r['prefetch'] + r['final'] >= r['selected_unique']
                   for r in cold), 'cold-start must actually clear validity and fetch all selected tokens'


def stateful_inputs(directory):
    """Real LI input: fixed Index/KV, query-only A/A/B+C/A residency cycle."""
    from pathlib import Path

    import numpy as np
    from reporting import atomic_json
    from workloads import ReplayRound

    batch, history, resident, heads = 2, 12288, 4096, 32
    initial = np.broadcast_to(np.arange(history), (batch, history)).copy()
    keys = np.zeros((batch, history, 128), dtype=np.uint8)
    for feature, begin in enumerate((4096, 6144, 8192)):
        keys[:, begin:begin + 2048, feature] = 0x38  # Exactly 1 in E4M3.
    geometry = dict(batch=batch, history=history, resident=resident, heads=heads,
                    # Serving bindings reserve three 128-token ABC pages, even with an empty tail.
                    token_bytes=656, output_stride=resident + 3 * 128)
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


def transfer_coverage(posts, destinations, lru_slots, stride, resident, *, allocated_slots=None, prior_counts=None):
    """Prove conflict descriptors and fallback destinations from actual output."""
    import numpy as np
    posts, destinations, lru_slots = map(np.asarray, (posts, destinations, lru_slots))
    destinations = destinations.reshape(-1)
    evidence = []
    for row in range(len(posts)):
        if allocated_slots is not None:
            count = int(posts[row, :, 0].sum())
            begin = int(prior_counts[row])
            selected = np.asarray(allocated_slots[row, begin:begin+count], dtype=np.int64)
            if len(selected) != count or np.any(selected < 0) or np.any(selected >= resident):
                raise RuntimeError('Invalid terminal hot-slot allocations')
            if len(np.unique(selected)) != count:
                raise RuntimeError('Terminal repairs alias one destination')
            evidence.append(dict(request=row, final=count, conflicts=0, path='parallel',
                fallback_slots=len(set(selected.tolist()) - set(map(int, lru_slots[row])))))
            continue
        slots, conflicts = [], 0
        for post in posts[row]:
            count, start, safe = map(int, (post[0], post[4], post[5]))
            if not 0 <= safe <= count or start < 0 or start + count > len(destinations):
                raise RuntimeError('Invalid final descriptor range')
            raw = destinations[start:start + count].astype(np.int64)
            if np.any(raw < 0):
                raise RuntimeError('Negative final destination')
            selected = raw % stride
            if np.any(selected < 0) or np.any(selected >= resident):
                raise RuntimeError('Final fetch did not target persistent hot slots')
            slots.extend(selected.tolist())
            conflicts += count - safe
        fallback = len(set(slots) - set(map(int, lru_slots[row])))
        evidence.append(dict(request=row, final=len(slots), conflicts=conflicts,
                             fallback_slots=fallback))
    return evidence


def require_transfer_coverage(rows):
    assert rows and all(row['fallback_slots'] > 0 for row in rows), 'Hot-pool fallback was not exercised'
    assert all(row.get('path') == 'parallel' or row['conflicts'] > 0 for row in rows), 'Prefetch destination conflict was not exercised'


def check_sfa(torch, tensors, geometry, before, after, *, output_path=None):
    """Actual service C8 API, plus independent logical-token numerical oracle.

    Scope is attention on the LI-selected 2048 history tokens. The 384-slot
    tail is allocated/seeded but is not appended to sparse indices. This does
    not claim to validate a model's ABC scheduling or mixed prefill batches.
    """
    import torch_npu
    from workloads import payload, unpack

    g = geometry
    rng = np.random.default_rng(181)
    query = torch.from_numpy(rng.uniform(-0.2, 0.2, (g.batch * g.queries, 8, 576)).astype(np.float32))
    query = query.to(torch.bfloat16)
    qcpu = query.float().numpy()
    scale = 576**-0.5
    expected = np.empty((g.batch * g.queries, 8, 512), np.float32)
    indices = np.empty((g.batch * g.queries, 1, TOPK), np.int32)
    # Get the ordered LI -> slot mapping, never assume packed destination order.
    for b, pool in enumerate(before["pool_rows"]):
        for q in range(g.queries):
            row = b * g.queries + q
            tokens = after["topk"][row, 0]
            if np.any(tokens < 0):
                raise AssertionError("128K case must supply 2048 valid TopK indices")
            indices[row, 0] = after["forward"][pool, tokens]
            if np.any((indices[row] < 0) | (indices[row] >= before["budgets"][b])):
                raise AssertionError("LI token has no valid hot-slot mapping")
            key, value = unpack(payload(int(pool) * g.num_layers + g.layer_id, tokens))
            logits = qcpu[row] @ key.T * scale
            logits -= logits.max(axis=1, keepdims=True)
            prob = np.exp(logits)
            prob /= prob.sum(axis=1, keepdims=True)
            prob = torch.from_numpy(prob).to(torch.bfloat16).float().numpy()
            expected[row] = prob @ value
    device = tensors["query"].device
    kv = tensors["resident_output"].reshape(g.resident_blocks, BLOCK_TOKENS, 1, TOKEN_BYTES)
    kv = kv.view(torch.float8_e4m3fn)
    mapping = tensors["forward"].index_select(0, tensors["pool_rows"].long())
    mapped = torch.gather(mapping, 1, tensors["topk"].reshape(g.batch, -1).long())
    mapped = mapped.reshape(g.batch * g.queries, 1, TOPK)
    if not np.array_equal(mapped.cpu().numpy(), indices):
        raise AssertionError("device token-to-slot gather differs from host mapping")
    actual = torch_npu.npu_kv_quant_sparse_flash_attention(
        query=query.to(device),
        key=kv,
        value=kv,
        sparse_indices=mapped,
        scale_value=scale,
        sparse_block_size=1,
        block_table=tensors["resident_table"],
        actual_seq_lengths_query=tensors["cumulative_query"],
        actual_seq_lengths_kv=tensors["logical_lens"],
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        key_quant_mode=2,
        value_quant_mode=2,
        rope_head_dim=64,
    )
    actual = actual[0] if isinstance(actual, (tuple, list)) else actual
    actual = actual.float().cpu().numpy().reshape(expected.shape)
    if output_path is not None:
        np.save(output_path, actual)
    error = float(np.max(np.abs(actual - expected)))
    if not np.all(np.isfinite(actual)) or not np.allclose(actual, expected, atol=0.006, rtol=0.025):
        raise AssertionError(f"A5 C8 SFA numerical mismatch: max_abs={error}")
    return {
        "status": "pass",
        "max_abs": error,
        "atol": 0.006,
        "rtol": 0.025,
        "heads": 8,
        "scope": "LI-selected history TopK; actual service C8 SFA API; logical-token CPU oracle",
    }
