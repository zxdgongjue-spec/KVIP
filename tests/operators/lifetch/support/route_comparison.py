# SPDX-License-Identifier: Apache-2.0
"""Compare fused routes through the ordinary multi-batch bench entry.

Every route keeps the complete requested batch list and shared pool. Timing
and numerical checks remain in test_lifetch; this module only selects sources,
forwards configuration and joins validated case results.
"""
from __future__ import annotations

import csv
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
MODES = {'mr23': 'original', 'window_budget': 'window', 'aligned_budget': 'aligned'}
DEFAULT_PARTS = {1: 16, 2: 16, 4: 8, 8: 4, 16: 2}
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


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def reference_module():
    path = ROOT/'experiments/lifetch_score_t2/run_a5.py'
    spec = importlib.util.spec_from_file_location('_lifetch_comparison_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
                                     not fused.get('score_geometry')):
            reasons.append('candidate mode or geometry evidence missing')
        s, f = [paths[name].get('all_rounds', {}) for name in ('serial', 'fused')]
        stages = f.get('stage_mean_us', {})
        if not isinstance(stages, dict):
            stages = {}
        records[key] = dict(case=job['case'], round=job['round'], batch=key[0], miss_percent=key[1],
            execution=report.get('execution', 'INCOMPLETE'), correctness=report.get('correctness', 'NOT_RUN'),
            accepted=not reasons, reason='; '.join(reasons), serial_event_us=s.get('event_mean_us'),
            fused_event_us=f.get('event_mean_us'), li_us=stages.get('li'),
            resolve_fetch_us=stages.get('resolve_update'), score_parts=expected_parts,
            score_mode=MODES[job['case']], score_geometry=fused.get('score_geometry'),
            effective_aic=fused.get('effective_aic'), pool=pool, report=str(path),
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
            for case in config.compare_routes:
                values = [s for s in samples if s['case'] == case and s['batch'] == batch and s['miss_percent'] == miss]
                accepted = all(v['accepted'] for v in values)
                row = dict(case=case, batch=batch, miss_percent=miss, prefetch=config.prefetch_tokens,
                    measured_rounds=config.comparison_rounds, accepted=accepted,
                    execution='COMPLETED' if all(v['execution']=='COMPLETED' for v in values) else 'INCOMPLETE',
                    correctness='PASS' if all(v['correctness']=='PASS' for v in values) else 'FAIL',
                    reason='; '.join(dict.fromkeys(v['reason'] for v in values if v['reason'])),
                    score_parts=values[0].get('score_parts'), score_mode=MODES[case],
                    serial_event_us=None, fused_event_us=None, li_us=None, resolve_fetch_us=None,
                    over_mr23=None, net_gain_vs_mr23_us=None)
                if accepted:
                    for field in ('serial_event_us', 'fused_event_us', 'li_us', 'resolve_fetch_us'):
                        if all(isinstance(v.get(field), (int, float)) for v in values):
                            row[field] = sum(v[field] for v in values)/len(values)
                    old = sum(v['fused_event_us'] for v in reference)/len(reference)
                    row.update(over_mr23=row['fused_event_us']/old, net_gain_vs_mr23_us=old-row['fused_event_us'])
                rows.append(row)
    with (Path(output)/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    save(Path(output)/'summary.json', rows)
    print('Full multi-batch bench | us/call | same-graph event mean', flush=True)
    print('case             B miss% pf   Check Accepted   Serial    Fused       LI Resolve/Get /MR23', flush=True)
    for row in rows:
        def number(key):
            value = row[key]
            return '—' if value is None else f'{value:.2f}'
        print(f'{row["case"]:16} {row["batch"]:2} {row["miss_percent"]:5g} {row["prefetch"]:4} '
              f'{row["correctness"]:5} {str(row["accepted"]):8} {number("serial_event_us"):>8} '
              f'{number("fused_event_us"):>8} {number("li_us"):>8} {number("resolve_fetch_us"):>11} '
              f'{number("over_mr23"):>5}', flush=True)
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
