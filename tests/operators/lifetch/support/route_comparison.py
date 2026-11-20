# SPDX-License-Identifier: Apache-2.0
"""Compare fused routes through the ordinary multi-batch bench entry.

Every route keeps the complete requested batch list and shared pool. Timing
and numerical checks remain in test_lifetch; this module only selects sources,
forwards configuration and joins validated case results.
"""
from __future__ import annotations

import csv
import difflib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[4]
KVAF = '1c1310387ebfb6776a07316874e924265b209ada'
MODES = {'mr23': 'original', 'window_budget': 'window', 'query_budget': 'window_query',
         'aligned_budget': 'aligned'}
DEFAULT_PARTS = {1: 16, 2: 16, 4: 8, 8: 4, 16: 2}
COMPONENT_FIELDS = ('score_hot_us', 'score_cold_us', 'theta_us', 'query_topk_us',
                    'service_us', 'final_us', 'score_full_us', 'full_topk_classify_us')
COUNT_FIELDS = ('prefetch_requests_per_call', 'final_get_requests_per_call', 'total_get_requests_per_call')
SOURCE_PATHS = ('python/kvip', 'tests/operators/lifetch',
                'csrc/operators/lightning_indexer_with_fetch',
                'csrc/operators/common', 'csrc/operators/lidu',
                'experiments/lifetch_score_t2/run_a5.py')


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(source):
    """Hash effective source, including local edits, excluding generated builds."""
    source = Path(source)
    files = {}
    excluded = {'__pycache__', '_custom_opp', 'build', 'build-npu', 'dist'}
    extensions = {'.py', '.cpp', '.h', '.hpp', '.cc', '.cxx', '.c', '.inc', '.cmake', '.json'}
    for relative in SOURCE_PATHS:
        root = source/relative
        paths = [root] if root.is_file() else root.rglob('*')
        for path in sorted(paths):
            name = path.relative_to(source)
            if (path.is_file() and not excluded.intersection(name.parts) and
                    (path.suffix in extensions or path.name == 'CMakeLists.txt')):
                files[str(name)] = checksum(path)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(sha256=digest, files=files)


def dependency_fingerprint(dependency):
    root = Path(dependency['root'])/'build-npu/bin'
    return {name: checksum(root/name) for name in dependency['binaries']}


def positive_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def window_configuration(record):
    """Compare common Score/admission geometry, allowing the Q-parallel change."""
    geometry = record['score_geometry']
    return dict(score_parts=record['score_parts'], effective_aic=record['effective_aic'],
        li_schedule=record['li_schedule'],
        **{key: geometry.get(key) for key in ('half_window', 'waves', 'score_post',
                                            'cold_predicate', 'final_topk', 'resolve_fetch')})


def pipeline_count_summary(result, config, batch):
    """Average actual submitted token requests per complete batch invocation."""
    empty = dict(pipeline_count_status='MISSING', pipeline_count_reason='completed-replay counters not collected',
                 **{field: None for field in COUNT_FIELDS})
    report = result.get('pipeline_counts')
    if not isinstance(report, dict):
        return empty
    try:
        if report.get('status') != 'PASS' or report.get('unit') != 'token Get requests':
            raise ValueError('counter status or units invalid')
        records = report['records']
        if not isinstance(records, list) or not records:
            raise ValueError('counter records missing')
        groups = {}
        seen = set()
        for record in records:
            key = tuple(record[field] for field in ('source_round', 'input', 'request'))
            if any(not isinstance(v, int) or isinstance(v, bool) for v in key):
                raise ValueError('counter identity is not an integer')
            if key in seen or key[0] < 0 or not 0 <= key[1] < config.iterations or not 0 <= key[2] < batch:
                raise ValueError('counter identity duplicated or outside requested inputs')
            seen.add(key)
            counts = [record[field] for field in ('prefetch_requests', 'final_get_requests', 'total_get_requests')]
            if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in counts):
                raise ValueError('counter value is not a nonnegative integer')
            if counts[0] > config.prefetch_tokens or counts[2] != counts[0]+counts[1]:
                raise ValueError('counter total or prefetch budget inconsistent')
            if record.get('row_final_requests') != counts[1] or sum(record['final_get_by_worker']) != counts[1]:
                raise ValueError('Final Get counter cross-check failed')
            groups.setdefault(key[0], []).append(counts)
        if any(len(values) != batch*config.iterations for values in groups.values()):
            raise ValueError('counter coverage differs from measured graph calls/batch')
        graph_count = result.get('all_rounds', {}).get('measured_graphs', 1)
        if len(groups) != graph_count:
            raise ValueError('counter source rounds differ from measured graphs')
        calls = config.iterations*len(groups)
        totals = [sum(values[index] for group in groups.values() for values in group) / calls
                  for index in range(3)]
        return dict(pipeline_count_status='PASS', pipeline_count_reason='', **dict(zip(COUNT_FIELDS, totals)))
    except (ValueError, KeyError, TypeError) as error:
        empty.update(pipeline_count_status='INVALID', pipeline_count_reason=str(error))
        return empty


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def reference_module():
    path = ROOT/'experiments/lifetch_score_t2/run_a5.py'
    spec = importlib.util.spec_from_file_location('_lifetch_comparison_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_reporting_overlay(source, output):
    """Read the same completion counters after the fixed MR23 measured graph.

    Only host reporting changes: operator/launch source and event/profiler
    measurement stay as exported. Record the exact reporting diff separately.
    """
    relative = Path('tests/operators/lifetch/support/execution.py')
    path = Path(source)/relative
    original = path.read_text()
    (Path(output)/'baseline-execution-original.py').write_text(original)
    replacements = (
        ('        rounds = {}\n        for inputs in sequence:',
         '        rounds = {}\n        pipeline_counts = None\n        if fused:\n'
         '            from pipeline_counts import collect_pipeline_counts, new_pipeline_count_report\n'
         '            pipeline_counts = new_pipeline_count_report()\n        for inputs in sequence:'),
        ("            validate_all(inputs, 'measured')\n            rounds[inputs.source_round] = measured['timing']",
         "            validate_all(inputs, 'measured')\n            if fused:\n"
         "                pipeline_counts['records'].extend(collect_pipeline_counts(states, inputs.source_round))\n"
         "            rounds[inputs.source_round] = measured['timing']"),
        ('                    failures=checks.failures, sequence_graph_sum=dict(',
         '                    failures=checks.failures, pipeline_counts=pipeline_counts, sequence_graph_sum=dict('),
    )
    effective = original
    for before, after in replacements:
        if effective.count(before) != 1:
            raise RuntimeError('Pinned MR23 reporting context changed; refusing an ambiguous overlay')
        effective = effective.replace(before, after)
    helper_relative = relative.parent/'pipeline_counts.py'
    helper = Path(source)/helper_relative
    if helper.exists():
        raise RuntimeError('Pinned MR23 already contains a counter helper; reporting overlay needs review')
    helper.write_bytes((ROOT/helper_relative).read_bytes())
    path.write_text(effective)
    (Path(output)/'baseline_reporting.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(keepends=True), effective.splitlines(keepends=True),
        fromfile=str(relative), tofile=str(relative))))
    evidence = dict(path=str(relative), original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        effective_sha256=checksum(path), helper=str(helper_relative), helper_sha256=checksum(helper),
        reason='Identical read-only counter reporting after measured event/completion; no kernel, launch or timer change')
    save(Path(output)/'baseline_reporting_override.json', evidence)
    return evidence


def parts_configuration(config):
    from workloads import score_parts_for_batch
    result = {}
    for batch in config.batches:
        requested = score_parts_for_batch(config, batch)
        fallback = DEFAULT_PARTS.get(batch)
        if fallback is None:
            fallback = max(p for p in (1, 2, 4, 8, 16) if batch*p <= 32)
        result[batch] = requested or fallback
    return result


def misses(config):
    value = config.miss_percent
    return list(value) if isinstance(value, (tuple, list)) else [value]


def build_commands(config, output, sources, parts_map):
    """One complete multi-batch bench worker per route and measured round."""
    jobs = []
    first = 0 if config.comparison_discard_first else 1
    for round_id in range(first, config.comparison_rounds + 1):
        order = list(config.compare_routes)
        if round_id % 2 == 0:
            order.reverse()
        for case in order:
            name = f'r{round_id}_{case}'
            folder = Path(output)/name
            args = [sys.executable, '-u', str(Path(sources[case])/'tests/operators/lifetch/test_lifetch.py'),
                'bench', '--input', 'synthetic', '--transport', config.transport,
                '--device', str(config.device), '--kvaf-root', str(Path(config.kvaf_root).resolve()),
                '--kvaf-server', config.kvaf_server,
                '--batches', ','.join(map(str, config.batches)),
                '--miss-percent', ','.join(f'{x:g}' for x in misses(config)),
                '--history', str(config.history), '--resident', str(config.resident),
                '--queries', str(config.queries), '--index-heads', str(config.index_heads),
                '--seed', str(config.seed), '--prefetch-tokens', str(config.prefetch_tokens),
                '--prefetch-rank-start', str(config.prefetch_rank_start),
                '--intermediate-fetches', str(config.intermediate_fetches),
                '--warmup', str(config.warmup), '--iterations', str(config.iterations),
                '--timeout-seconds', str(config.timeout_seconds), '--output', str(folder)]
            if case != 'mr23':
                args += ['--li-score-mode', MODES[case], '--li-score-parts-map',
                         ','.join(f'{b}:{parts_map[b]}' for b in config.batches)]
            jobs.append(dict(name=name, case=case, round=round_id, output=str(folder),
                             log=str(Path(output)/(name+'.log')), args=args,
                             cwd=str(sources[case]), parts_map=parts_map))
    return jobs


def dependency_preflight(config):
    """Keep the established MR23/KVAF ABI; do not modify selected dependencies."""
    root = Path(config.kvaf_root).resolve()
    head = git('-C', str(root), 'rev-parse', 'HEAD')
    if head != KVAF:
        raise RuntimeError(f'Route comparison requires validated KVAF @{KVAF}; found {head} at {root}. '
                           'Point --kvaf-root at the existing pinned build. The selected checkout is unchanged.')
    required = ('CMakeCache.txt', 'python/kvaf/__init__.py',
                'bin/kvaf_execute_batch_get_kernels.o', 'bin/kvaf_execute_batch_put_kernels.o')
    for relative in required:
        if not (root/'build-npu'/relative).is_file():
            raise RuntimeError(f'Missing built dependency: {root/"build-npu"/relative}')
    return dict(commit=head, root=str(root), binaries={
        name: checksum(root/'build-npu/bin'/name) for name in
        ('kvaf_execute_batch_get_kernels.o', 'kvaf_execute_batch_put_kernels.o')})


def environment_for(source, config):
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(Path(source)/'python'),
        str(Path(config.kvaf_root).resolve()/'build-npu/python'), env.get('PYTHONPATH', '')])
    vendors = [str(p) for p in (ROOT/'csrc/operators/lidu/_custom_opp/vendors').glob('*')
               if (p/'op_api/lib/libcust_opapi.so').is_file()]
    if vendors:
        env['ASCEND_CUSTOM_OPP_PATH'] = os.pathsep.join(
            vendors + list(filter(None, [env.get('ASCEND_CUSTOM_OPP_PATH')])))
    return env


def device_preflight(config):
    probe = ('import json,sys; import torch,torch_npu; '
             'from pathlib import Path; from kvip.runtime.build import kvaf_build_paths; '
             'from kvip.runtime.device import query_device_resources; '
             'torch.npu.set_device(int(sys.argv[1])); '
             'cann,_=kvaf_build_paths(Path(sys.argv[2])/"build-npu"); '
             'print(json.dumps(query_device_resources(cann,int(sys.argv[1]))))')
    result = subprocess.check_output([sys.executable, '-c', probe, str(config.device),
        str(Path(config.kvaf_root).resolve())], env=environment_for(ROOT, config), text=True)
    return json.loads(result.splitlines()[-1])


def execute_job(job, environment):
    with Path(job['log']).open('w') as log:
        process = subprocess.Popen(job['args'], cwd=job['cwd'], env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            return process.wait()
        except BaseException:
            # Stop the child bench and its worker before returning control.
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        finally:
            process.stdout.close()


def fixture_hashes(folder):
    return {p.name: checksum(p) for p in sorted(Path(folder).glob('*'))
            if p.suffix in ('.bin', '.npy')}


def collect_job(job, config):
    records = {}
    pools = sorted(Path(job['output']).rglob('pool.json'))
    pool = json.loads(pools[0].read_text()) if len(pools) == 1 else {}
    if not isinstance(pool, dict):
        raise ValueError('Malformed shared pool record')
    pool_valid = (pool.get('transport') == config.transport and
                  pool.get('max_batch') == max(config.batches) and
                  pool.get('runtime_workers') == 40 and
                  pool.get('cases') == 2*len(config.batches)*len(misses(config)) and
                  pool.get('put_bytes') == max(config.batches)*config.history*656)
    fixture_cache = {}
    reference = reference_module() if job['case'] != 'mr23' else None
    for path in sorted(Path(job['output']).rglob('case-b*/report.json')):
        report = json.loads(path.read_text())
        if not isinstance(report, dict):
            raise ValueError(f'Malformed case report: {path}')
        key = (report['batch'], report['miss_percent'])
        if (not isinstance(key[0], int) or isinstance(key[0], bool) or
                key[0] not in config.batches or key[1] not in misses(config)):
            raise ValueError(f'Unexpected case geometry: {key}')
        if key in records:
            raise RuntimeError(f'Duplicate case result for {job["name"]}: {key}')
        paths = {}
        for name in ('serial', 'fused'):
            result_path = path.parent/name/'path_result.json'
            paths[name] = json.loads(result_path.read_text()) if result_path.is_file() else {}
            if not isinstance(paths[name], dict):
                raise ValueError(f'Malformed {name} path result: {result_path}')
        fixture = path.parent.parent/'fixture'
        if fixture not in fixture_cache:
            fixture_cache[fixture] = fixture_hashes(fixture)
        reasons = []
        if job.get('source_unchanged') is False:
            reasons.append('source changed during comparison')
        if job.get('dependency_unchanged') is False:
            reasons.append('dependency binaries changed during comparison')
        if report.get('execution') != 'COMPLETED':
            reasons.append('case incomplete')
        if report.get('correctness') != 'PASS' or any(p.get('correctness') != 'PASS' for p in paths.values()):
            reasons.append('correctness check failed or missing')
        if not pool_valid:
            reasons.append('shared pool evidence missing or differs from requested multi-batch configuration')
        if not fixture_cache[fixture]:
            reasons.append('fixture operand hashes missing')
        for name, result in paths.items():
            timing = result.get('all_rounds', {})
            if not isinstance(timing, dict):
                raise ValueError(f'Malformed {name} timing record: {path}')
            if timing.get('timing_protocol') != 'one_profiled_graph':
                reasons.append(f'{name}: same-graph timing evidence missing')
            if not positive_number(timing.get('event_mean_us')):
                reasons.append(f'{name}: event timing missing or invalid')
            stages = timing.get('stage_mean_us', {})
            required = ('li', 'resolve_update') if name == 'fused' else ('li', 'lidu_prepare', 'fetch')
            if not isinstance(stages, dict) or any(not positive_number(stages.get(s)) for s in required):
                reasons.append(f'{name}: required same-graph stages missing or invalid')
        fused = paths['fused']
        mapping = job['parts_map']
        expected_parts = 0 if job['case'] == 'mr23' else mapping.get(key[0], mapping.get(str(key[0])))
        if fused.get('score_parts', 0) != expected_parts:
            reasons.append('actual score partition count differs from requested route')
        if job['case'] != 'mr23' and (fused.get('score_mode') != MODES[job['case']] or
                                     not fused.get('score_geometry') or not fused.get('li_schedule')):
            reasons.append('candidate mode or geometry evidence missing')
        s, f = [paths[name].get('all_rounds', {}) for name in ('serial', 'fused')]
        stages = f.get('stage_mean_us', {})
        if not isinstance(stages, dict):
            stages = {}
        components = {}
        geometry = fused.get('score_geometry')
        launches = 4 if job['case'] == 'mr23' else None
        if job['case'] != 'mr23':
            try:
                if not isinstance(geometry, dict) or geometry.get('mode') != MODES[job['case']]:
                    raise ValueError('actual component geometry/mode missing')
                waves = geometry['waves']
                if not isinstance(waves, list) or not waves:
                    raise ValueError('actual Score windows missing')
                expected = 2 if job['case'] == 'aligned_budget' else len(waves)*(3 if job['case']=='query_budget' else 2)+2
                if geometry.get('li_launches') != expected:
                    raise ValueError('actual LI launch count differs from route contract')
                components = reference.li_components(path.parent/'fused', MODES[job['case']], geometry)
                if not components or any(not positive_number(v) for v in components.values()):
                    raise ValueError('same-graph component durations missing or invalid')
                if not positive_number(stages.get('li')) or abs(sum(components.values())-stages['li']) > .05:
                    raise ValueError('component means disagree with same-graph LI stage')
                launches = expected
            except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
                reasons.append(f'same-graph LI components rejected: {error}')
                components = {}
        records[key] = dict(case=job['case'], round=job['round'], batch=key[0], miss_percent=key[1],
            execution=report.get('execution', 'INCOMPLETE'), correctness=report.get('correctness', 'NOT_RUN'),
            accepted=not reasons, reason='; '.join(reasons), serial_event_us=s.get('event_mean_us'),
            fused_event_us=f.get('event_mean_us'), li_us=stages.get('li'),
            resolve_fetch_us=stages.get('resolve_update'), score_parts=expected_parts,
            score_mode=MODES[job['case']], score_geometry=geometry,
            li_schedule=fused.get('li_schedule'), li_kernel_launches=launches,
            effective_aic=fused.get('effective_aic'), pool=pool, report=str(path),
            pipeline_counts=fused.get('pipeline_counts'), **pipeline_count_summary(fused, config, key[0]),
            **{field: components.get(field) for field in COMPONENT_FIELDS},
            score_total_us=(components.get('score_full_us') if job['case']=='aligned_budget' else
                            components.get('score_hot_us', 0)+components.get('score_cold_us', 0)) if components else None,
            fixture_sha256=fixture_cache[fixture], failures={name: result.get('failures', [])
                                                           for name, result in paths.items()})
    return records


def summarize(output, jobs, config):
    samples = []
    collected = {}
    for job in jobs:
        if job['round'] == 0:
            continue
        try:
            collected[(job['round'], job['case'])] = collect_job(job, config)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            collected[(job['round'], job['case'])] = {}
            job['collection_error'] = str(error)
    for case in config.compare_routes:
        for batch in config.batches:
            for miss in misses(config):
                records = [collected.get((r, case), {}).get((batch, miss))
                           for r in range(1, config.comparison_rounds+1)]
                present = [record for record in records if record is not None]
                if present and any(record.get('score_geometry') != present[0].get('score_geometry') or
                                   record.get('li_schedule') != present[0].get('li_schedule') for record in present):
                    for record in present:
                        record.update(accepted=False, reason=(record['reason']+
                                      '; actual geometry changed across measured rounds').strip('; '))
    for round_id in range(1, config.comparison_rounds+1):
        for batch in config.batches:
            for miss in misses(config):
                key = (batch, miss)
                reference = collected.get((round_id, 'mr23'), {}).get(key)
                for case in config.compare_routes:
                    record = collected.get((round_id, case), {}).get(key)
                    if record is None:
                        error = next((j.get('collection_error') for j in jobs
                                      if j['round'] == round_id and j['case'] == case), None)
                        record = dict(case=case, round=round_id, batch=batch, miss_percent=miss,
                                      execution='INCOMPLETE', correctness='NOT_RUN',
                                      accepted=False, reason='missing structured case result' +
                                      (f'; {error}' if error else ''))
                    else:
                        record = dict(record)
                    if not reference or not reference['accepted']:
                        record.update(accepted=False, reason=(record['reason']+'; MR23 reference not accepted').strip('; '))
                    elif record.get('fixture_sha256') != reference['fixture_sha256']:
                        record.update(accepted=False, reason=(record['reason']+'; operand mismatch across routes').strip('; '))
                    samples.append(record)
    save(Path(output)/'samples.json', samples)
    rows = []
    for batch in config.batches:
        for miss in misses(config):
            reference = [s for s in samples if s['case'] == 'mr23' and s['batch'] == batch and s['miss_percent'] == miss]
            window = [s for s in samples if s['case'] == 'window_budget' and s['batch'] == batch and s['miss_percent'] == miss]
            for case in config.compare_routes:
                values = [s for s in samples if s['case'] == case and s['batch'] == batch and s['miss_percent'] == miss]
                accepted = all(v['accepted'] for v in values)
                row = dict(case=case, batch=batch, miss_percent=miss, prefetch=config.prefetch_tokens,
                    measured_rounds=config.comparison_rounds, accepted=accepted,
                    execution='COMPLETED' if all(v['execution']=='COMPLETED' for v in values) else 'INCOMPLETE',
                    correctness='PASS' if all(v['correctness']=='PASS' for v in values) else 'FAIL',
                    reason='; '.join(dict.fromkeys(v['reason'] for v in values if v['reason'])),
                    score_parts=values[0].get('score_parts'), score_mode=MODES[case],
                    li_kernel_launches=values[0].get('li_kernel_launches'),
                    serial_event_us=None, fused_event_us=None, li_us=None, resolve_fetch_us=None,
                    **{field: None for field in (*COMPONENT_FIELDS, 'score_total_us')},
                    **{field: None for field in COUNT_FIELDS},
                    pipeline_count_status='MISSING', pipeline_count_reason='',
                    over_mr23=None, net_gain_vs_mr23_us=None,
                    over_window_budget=None, net_gain_vs_window_us=None, window_comparison_reason='')
                if accepted:
                    for field in ('serial_event_us', 'fused_event_us', 'li_us', 'resolve_fetch_us',
                                  *COMPONENT_FIELDS, 'score_total_us'):
                        if all(isinstance(v.get(field), (int, float)) for v in values):
                            row[field] = sum(v[field] for v in values)/len(values)
                    old = sum(v['fused_event_us'] for v in reference)/len(reference)
                    row.update(over_mr23=row['fused_event_us']/old, net_gain_vs_mr23_us=old-row['fused_event_us'])
                # Completed-state counters remain useful when numerical checks
                # reject the performance sample; they never authorize a gain.
                if all(v.get('pipeline_count_status') == 'PASS' for v in values):
                    row.update(pipeline_count_status='PASS', **{
                        field: sum(v[field] for v in values)/len(values) for field in COUNT_FIELDS})
                row['pipeline_count_reason'] = '; '.join(dict.fromkeys(
                    v.get('pipeline_count_reason', 'counter records missing') for v in values
                    if v.get('pipeline_count_status') != 'PASS'))
                if any(v.get('pipeline_count_status') == 'INVALID' for v in values):
                    row['pipeline_count_status'] = 'INVALID'
                if case == 'query_budget':
                    if not accepted:
                        row['window_comparison_reason'] = 'query result not accepted'
                    elif len(window) != len(values) or not all(v['accepted'] for v in window):
                        row['window_comparison_reason'] = 'window reference not accepted or not selected'
                    else:
                        pairs = [(v, next(w for w in window if w['round']==v['round'])) for v in values]
                        if any(v['fixture_sha256'] != w['fixture_sha256'] or
                               window_configuration(v) != window_configuration(w) for v, w in pairs):
                            row['window_comparison_reason'] = 'window/query operands or common Score/window configuration differ'
                        else:
                            old_window = sum(v['fused_event_us'] for v in window)/len(window)
                            row.update(over_window_budget=row['fused_event_us']/old_window,
                                       net_gain_vs_window_us=old_window-row['fused_event_us'])
                rows.append(row)
    with (Path(output)/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    save(Path(output)/'summary.json', rows)
    print('Full multi-batch bench | us/call | same-graph event mean', flush=True)
    print('case             B miss% pf   Check Accepted   Serial    Fused       LI Resolve/Get /MR23 /Window Gain/window', flush=True)
    for row in rows:
        def number(key):
            value = row[key]
            return '—' if value is None else f'{value:.2f}'
        print(f'{row["case"]:16} {row["batch"]:2} {row["miss_percent"]:5g} {row["prefetch"]:4} '
              f'{row["correctness"]:5} {str(row["accepted"]):8} {number("serial_event_us"):>8} '
              f'{number("fused_event_us"):>8} {number("li_us"):>8} {number("resolve_fetch_us"):>11} '
              f'{number("over_mr23"):>5} {number("over_window_budget"):>7} {number("net_gain_vs_window_us"):>11}', flush=True)
    print('Same-graph LI components | us/call | Service includes admission/packing/submission/control', flush=True)
    print('case             B miss% HotScore ColdScore   theta QueryTopK Service   Final LI launches', flush=True)
    for row in rows:
        if row['score_mode'] not in ('window', 'window_query'):
            continue
        def component(key):
            value = row[key]
            return 'inline' if key=='query_topk_us' and row['score_mode']=='window' else '—' if value is None else f'{value:.2f}'
        print(f'{row["case"]:16} {row["batch"]:2} {row["miss_percent"]:5g} '
              + ' '.join(f'{component(k):>8}' for k in ('score_hot_us', 'score_cold_us', 'theta_us',
                                                      'query_topk_us', 'service_us', 'final_us'))
              + f' {str(row["li_kernel_launches"]):>11}', flush=True)
    print('Completed-replay Get requests/call | read after timing | across all requests in the batch', flush=True)
    print('case             B miss% Counter status Prefetch FinalGet TotalGet', flush=True)
    for row in rows:
        counts = ['—' if row[field] is None else f'{row[field]:.2f}' for field in COUNT_FIELDS]
        print(f'{row["case"]:16} {row["batch"]:2} {row["miss_percent"]:5g} '
              f'{row["pipeline_count_status"]:14} '+ ' '.join(f'{value:>8}' for value in counts), flush=True)
    return rows


def run(config):
    reference = reference_module()
    if not config.kvaf_server:
        raise ValueError('--compare-routes requires --kvaf-server')
    parts_map = parts_configuration(config)
    parent = Path(config.output).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='route_comparison_'+time.strftime('%Y%m%d_%H%M%S')+'_', dir=parent))
    print('Route comparison results: '+str(output), flush=True)
    manifest = dict(status='PLANNED', baseline_commit=reference.BASE, candidate_commit=git('rev-parse', 'HEAD'),
        candidate_status=git('status', '--short'), kvaf_expected_commit=KVAF,
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(config).items()},
        parts_map=parts_map, batch_pool='Each route keeps the complete batch list in one shared max-batch pool',
        timing='Unmodified test_lifetch same-graph event means; requested warmup/iterations forwarded',
        discard_first=config.comparison_discard_first, completed_jobs=[])
    save(output/'manifest.json', manifest)
    (output/'candidate.diff').write_text(git('diff', 'HEAD'))
    try:
        sources = {case: reference.baseline(output) if case == 'mr23' else ROOT for case in config.compare_routes}
        manifest['baseline_build_override'] = json.loads((output/'baseline_build_override.json').read_text())
        manifest['baseline_reporting_override'] = baseline_reporting_overlay(sources['mr23'], output)
        manifest['source_fingerprints'] = {case: source_fingerprint(source) for case, source in sources.items()}
        jobs = build_commands(config, output, sources, parts_map)
        save(output/'jobs.json', jobs)
        if config.comparison_dry_run:
            manifest.update(status='NOT_RUN', planned_jobs=len(jobs))
            print(f'Planned {len(jobs)} complete multi-batch bench workers; no NPU or KVAF initialized.', flush=True)
            return 0
        manifest['kvaf'] = dependency_preflight(config)
        resources = device_preflight(config)
        required = max([2*max(config.batches)] +
                       [b*parts_map[b] for b in config.batches if any(c!='mr23' for c in config.compare_routes)])
        if resources['effective_aic'] < required:
            raise RuntimeError(f'Routes require {required} effective AIC, selected logical device has {resources}')
        manifest.update(status='RUNNING', device_resources=resources)
        for job in jobs:
            save(output/'manifest.json', manifest)
            print(f'[{time.strftime("%H:%M:%S")}] {job["name"]} batches={list(config.batches)}; Log: {job["log"]}', flush=True)
            case = job['case']
            job['source_before'] = source_fingerprint(sources[case])
            job['returncode'] = execute_job(job, environment_for(sources[job['case']], config))
            job['source_after'] = source_fingerprint(sources[case])
            job['source_unchanged'] = (job['source_before'] == job['source_after'] ==
                                       manifest['source_fingerprints'][case])
            job['dependency_unchanged'] = dependency_fingerprint(manifest['kvaf']) == manifest['kvaf']['binaries']
            manifest['completed_jobs'].append(dict(name=job['name'], returncode=job['returncode']))
            save(output/'jobs.json', jobs)
            if job['returncode']:
                print(f'{job["name"]}: exit={job["returncode"]}; case results retained; continuing independent routes.', flush=True)
        final_sources = {case: source_fingerprint(source) for case, source in sources.items()}
        final_dependencies = dependency_fingerprint(manifest['kvaf'])
        for job in jobs:
            job['source_unchanged'] &= final_sources[job['case']] == manifest['source_fingerprints'][job['case']]
            job['dependency_unchanged'] &= final_dependencies == manifest['kvaf']['binaries']
        manifest['source_fingerprints_after'] = final_sources
        manifest['dependency_binaries_after'] = final_dependencies
        save(output/'jobs.json', jobs)
        rows = summarize(output, jobs, config)
        save(output/'jobs.json', jobs)
        failed = any(job.get('returncode') for job in jobs) or not all(row['accepted'] for row in rows)
        manifest['status'] = 'FAILED' if failed else 'PASS'
        print('Summary: '+str(output/'summary.csv'), flush=True)
        return int(failed)
    except BaseException as error:
        manifest.update(status='FAILED', error=str(error))
        raise
    finally:
        save(output/'manifest.json', manifest)
