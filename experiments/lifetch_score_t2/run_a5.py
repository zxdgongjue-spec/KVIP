#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MR14 direct window score/predicate producers + MR23 ranked T2/Fetch tail.

No Git fetch or Docker command. Optional --prepare-kvaf builds a separate pinned
checkout from local history. Both sources use the MR23 graph timing protocol.
"""
import argparse
import csv
from decimal import Decimal
import hashlib
import inspect
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = 'a56154c37bebb0928da4a4095e9670898e6b21c7'
KVAF = '1c1310387ebfb6776a07316874e924265b209ada'
MR14 = 'c684d79f3a4f8432bf0915edeba392c7d54ba289'
ROUTES = {
    'mr23': ('1,2,4,8,16', 0, 'original'),
    'window_budget': ('1,2,4,8,16', None, 'window'),
    'query_budget': ('1,2,4,8,16', None, 'window_query'),
    'full_budget': ('1,2,4,8,16', None, 'full'),
    'window_p16': ('1', 16, 'window'),
    'window_p2': ('16', 2, 'window'),
    'score_t2_p2': ('1,16', 2, 'full'),
    'score_t2_p16': ('1', 16, 'full'),
}
DEFAULT_PARTS = {1:16, 2:16, 4:8, 8:4, 16:2}


def batch_configuration(batches_csv, parts_csv):
    batches = [int(x) for x in batches_csv.split(',')]
    entries = [tuple(int(v) for v in x.split(':')) for x in parts_csv.split(',')]
    if (not batches or len(set(batches)) != len(batches) or
            any(b not in DEFAULT_PARTS for b in batches)):
        raise ValueError('batches must be distinct values from 1,2,4,8,16')
    if any(len(e) != 2 for e in entries):
        raise ValueError('parts-map must use B:P entries')
    parts_map = dict(entries)
    if (len(parts_map) != len(entries) or any(b not in DEFAULT_PARTS for b in parts_map) or
            any(not 1 <= v <= 32 or b*v > 32 for b,v in parts_map.items()) or
            any(b not in parts_map for b in batches)):
        raise ValueError('parts-map must cover requested batches with distinct B:P entries, P=1..32 and B*P<=32')
    return batches, parts_map


def plan_routes(selected, batches, parts_map):
    routes=[]
    for case in selected:
        allowed,fixed,mode=ROUTES[case]
        for b in batches:
            if str(b) not in allowed.split(','):
                continue
            parts=parts_map[b] if fixed is None else fixed
            if parts and b*parts>32:
                raise ValueError(f'{case}: B{b}/P{parts} exceeds 32 AIC')
            routes.append((case,str(b),parts,mode))
    if not routes or any(not any(r[0]==case for r in routes) for case in selected):
        raise ValueError('Each selected route must support at least one requested batch')
    return routes


def li_components(folder, mode, geometry):
    """Named kernels from the same measured graph, excluding detail replays.

    The known window launch contract identifies Hot as the first score launch
    inside each recorded invocation; no phase is inferred from its duration.
    """
    if mode not in ('window','window_query'):
        return None
    profiles=sorted(folder.glob('diagnose/round-*/graph_profile.json'))
    if not profiles:
        raise RuntimeError(f'Missing same-graph component trace: {folder}')
    totals=[]
    aliases={'score':'v9_li_kernel_score_window_t2', 'service':'v9_li_window_service',
             'theta':'v9_li_kernel_1', 'final':'v9_li_kernel_3'}
    if mode=='window_query':
        aliases.update(service='v9_li_window_post',query_topk='v9_li_window_topk')
    for path in profiles:
        profile=json.loads(path.read_text())
        trace=json.loads((path.parent/'trace.json').read_text())
        events=trace['traceEvents'] if isinstance(trace,dict) else trace
        scopes=[e for e in events if e.get('name')=='lifetch_measured_graph' and e.get('ph')=='X']
        if len(scopes)!=1:
            raise RuntimeError('Expected exactly one measured graph scope for component accounting')
        origin=Decimal(str(scopes[0]['ts']))
        per_call=[]
        for sample in profile['samples']:
            intervals=sample['intervals_us']['li']
            parts={key:[] for key in aliases}
            for event in events:
                n=event.get('name','').lower();cat=(event.get('cat') or '').lower()
                if event.get('ph')!='X' or (cat and 'kernel' not in cat):continue
                key=next((key for key,prefix in aliases.items()
                          if n==prefix or (n.startswith(prefix+'_') and re.fullmatch(r'(?:_\d+)+',n[len(prefix):]))),None)
                if key is None:continue
                start=float(Decimal(str(event['ts']))-origin);duration=float(event['dur'])
                if any(abs(start-a)<.01 and abs(start+duration-b)<.01 for a,b in intervals):
                    parts[key].append((start,duration))
            waves=len(geometry['waves'])
            expected={'score':waves,'service':waves,'theta':1,'final':1}
            if mode=='window_query':expected['query_topk']=waves
            if {k:len(v) for k,v in parts.items()}!=expected:
                raise RuntimeError(f'Incomplete/duplicate LI components: {path}, invocation {sample["invocation"]}')
            scores=sorted(parts['score'])
            item=dict(score_hot_us=scores[0][1],score_cold_us=sum(v for _,v in scores[1:]),
                      theta_us=parts['theta'][0][1],service_us=sum(v for _,v in parts['service']),
                      final_us=parts['final'][0][1])
            if mode=='window_query':item['query_topk_us']=sum(v for _,v in parts['query_topk'])
            if abs(sum(item.values())-sample['stages_us']['li'])>.05:
                raise RuntimeError('LI component sum disagrees with measured stage duration')
            per_call.append(item)
        if not per_call:
            raise RuntimeError(f'Empty measured graph: {path}')
        totals.append({k:sum(s[k] for s in per_call)/len(per_call) for k in per_call[0]})
    return {k:sum(s[k] for s in totals)/len(totals) for k in totals[0]}


def command(args, *, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def baseline(output):
    # Export committed source only. Local build directories, dangling symlinks
    # and unrelated edits can never enter the baseline archive.
    if command(['git', 'rev-parse', BASE + '^{commit}']) != BASE:
        raise RuntimeError('MR23 commit is not in local Git history; update code on the host first')
    archive = output / 'baseline.tar'
    with archive.open('wb') as f:
        subprocess.run(['git', 'archive', BASE], cwd=ROOT, stdout=f, check=True)
    source = (output / 'baseline-source').resolve()
    source.mkdir()
    with tarfile.open(archive) as tar:
        if 'filter' in inspect.signature(tar.extractall).parameters:
            tar.extractall(source, filter='data')
        else:
            # Python 3.9 hosts lack extraction filters. Reject archive paths
            # and links that escape the newly created source directory.
            for member in tar.getmembers():
                destination = (source/member.name).resolve()
                if source != destination and source not in destination.parents:
                    raise RuntimeError('Baseline archive path escapes source root')
                if member.issym() or member.islnk():
                    target = ((destination.parent if member.issym() else source)/member.linkname).resolve()
                    if source != target and source not in target.parents:
                        raise RuntimeError('Baseline archive link escapes source root')
            tar.extractall(source)
    archive.unlink()
    # Apply the same startup-only cache recovery to both routes. MR23 kernel,
    # launch and benchmark timing sources remain exactly as archived.
    relative = Path('python/kvip/runtime/build.py')
    old_builder = source / relative
    (output / 'baseline-build-original.py').write_bytes(old_builder.read_bytes())
    original_hash = checksum(old_builder)
    old_builder.write_bytes((ROOT / relative).read_bytes())
    save(output / 'baseline_build_override.json', dict(
        path=str(relative), original_sha256=original_hash, effective_sha256=checksum(old_builder),
        reason='Shared startup-only fresh-attempt build recovery; no operator or timing changes'))
    return source


def fixture_hashes(folder):
    # The fixture path changes; the actual operands and score oracle must not.
    return {p.name: checksum(p) for p in sorted(folder.glob('*'))
            if p.suffix in ('.bin', '.npy')}


def summarize(output, jobs, measured_rounds):
    rows = []
    fixtures = {}
    resources = json.loads((output/'manifest.json').read_text()).get('device_resources', {})
    for job in jobs:
        if job['round'] == 0:
            continue
        reports = sorted(Path(job['output']).rglob('case-b*/report.json'))
        if not reports:
            raise RuntimeError(f"No completed reports for {job['name']}; inspect {job['log']}")
        fixture_dirs = sorted(Path(job['output']).rglob('fixture/fixture.json'))
        # A run may retain additional fixture variants; compare per requested miss.
        for path in fixture_dirs:
            miss_dir = path.parent.parent.name
            key = (job['pf'], job['batch'], miss_dir)
            digest = fixture_hashes(path.parent)
            if not digest:
                raise RuntimeError(f'No fixture operand files: {path.parent}')
            if key in fixtures and fixtures[key] != digest:
                raise RuntimeError(f'Operand mismatch across routes: {key}')
            fixtures[key] = digest
        for path in reports:
            report = json.loads(path.read_text())
            if report['batch'] != job['batch']:
                raise RuntimeError(f'Unexpected batch in isolated job: {path}')
            if report.get('correctness') != 'PASS' or report.get('execution') != 'COMPLETED':
                raise RuntimeError(f'Failed correctness/completion: {path}')
            serial, fused = [json.loads((path.parent / name / 'path_result.json').read_text())
                             for name in ('serial', 'fused')]
            if serial['all_rounds']['timing_protocol'] != 'one_profiled_graph' or fused['all_rounds']['timing_protocol'] != 'one_profiled_graph':
                raise RuntimeError('Timing protocol changed; do not mix old p50 with graph-event means')
            s, f = serial['all_rounds'], fused['all_rounds']
            stage = f['stage_mean_us']
            mode = job.get('mode','full')
            geometry = fused.get('score_geometry')
            schedule = fused.get('li_schedule')
            if mode in ('window','window_query') and (not geometry or not schedule):
                raise RuntimeError(f'Missing actual window geometry/schedule: {path}')
            components = li_components(path.parent/'fused', mode, geometry)
            if components and abs(sum(components.values())-stage['li']) > .05:
                raise RuntimeError('Component means disagree with all-round LI mean')
            native_stages = s['stage_mean_us']
            peak = max(w['score_aics'] for w in geometry['waves']) if mode in ('window','window_query') else report['batch']*(job['parts'] or 2)
            # Kernel names are retained with invocation boundaries in raw trace.
            rows.append(dict(round=job['round'], case=job['case'], prefetch=job['pf'],
                batch=report['batch'], miss_percent=report['miss_percent'], score_parts=job['parts'], score_mode=mode,
                score_geometry=geometry, score_peak_aic=peak, score_peak_aiv=2*peak,
                actual_windows=schedule['rounds'] if schedule else None,
                service_aivs=schedule['serviceLanes'] if schedule else None,
                li_kernel_launches=geometry['li_launches'] if mode in ('window','window_query') else 4,
                serial_event_us=s['event_mean_us'], fused_event_us=f['event_mean_us'],
                serial_li_us=native_stages['li'], serial_lidu_prepare_us=native_stages['lidu_prepare'],
                serial_fetch_us=native_stages['fetch'],
                fused_profiler_p50_us=f['p50_us'], li_us=stage['li'],
                resolve_fetch_us=stage['resolve_update'], gap_us=f['inter_kernel_gap_mean_us'],
                other_and_edges_us=f['event_mean_us']-stage['li']-stage['resolve_update']-f['inter_kernel_gap_mean_us'],
                **{key:components.get(key) if components else None for key in
                   ('score_hot_us','score_cold_us','theta_us','service_us','final_us','query_topk_us')},
                score_total_us=(components['score_hot_us']+components['score_cold_us']) if components else None,
                effective_aic=fused.get('effective_aic', resources.get('effective_aic')),
                effective_aiv=fused.get('effective_aiv', resources.get('effective_aiv')),
                li_schedule=fused.get('li_schedule'),
                trace_paths=[str(p) for p in (path.parent/'fused').rglob('trace.json')]))
    save(output / 'samples.json', rows)
    groups = {}
    for row in rows:
        groups.setdefault((row['case'], row['prefetch'], row['batch'], row['miss_percent']), []).append(row)
    fields = ('serial_event_us', 'fused_event_us', 'fused_profiler_p50_us', 'li_us', 'resolve_fetch_us', 'gap_us',
              'other_and_edges_us','serial_li_us','serial_lidu_prepare_us','serial_fetch_us',
              'score_hot_us','score_cold_us','score_total_us','theta_us','service_us','final_us','query_topk_us')
    table = []
    for (case, pf, b, miss), samples in groups.items():
        if sorted(s['round'] for s in samples) != list(range(1, measured_rounds+1)):
            raise RuntimeError('Missing or duplicated measured rounds')
        row = dict(case=case, prefetch=pf, batch=b, miss_percent=miss,
                   score_parts=samples[0]['score_parts'], score_mode=samples[0]['score_mode'],
                   score_geometry=samples[0]['score_geometry'], effective_aic=samples[0]['effective_aic'],
                   li_schedule=samples[0]['li_schedule'])
        for key in ('score_parts','score_mode','score_geometry','li_schedule','score_peak_aic','score_peak_aiv',
                    'actual_windows','service_aivs','li_kernel_launches'):
            if any(s[key]!=samples[0][key] for s in samples):
                raise RuntimeError(f'Geometry changed across measured rounds: {key}')
            row[key]=samples[0][key]
        row.update({f: sum(s[f] for s in samples)/len(samples) if all(s[f] is not None for s in samples) else None
                    for f in fields})
        ref = groups[('mr23', pf, b, miss)]
        old = sum(s['fused_event_us'] for s in ref)/len(ref)
        row.update(over_mr23=row['fused_event_us']/old,
                   over_serial=row['fused_event_us']/row['serial_event_us'],
                   serial_drift_over_mr23=row['serial_event_us']/(sum(s['serial_event_us'] for s in ref)/len(ref)),
                   per_round_over_mr23=[s['fused_event_us']/next(r['fused_event_us'] for r in ref if r['round']==s['round']) for s in samples])
        row['net_gain_vs_mr23_us']=old-row['fused_event_us']
        for key in ('li','resolve_fetch','gap','other_and_edges'):
            row[f'{key}_gain_vs_mr23_us']=sum(s[key+'_us'] for s in ref)/len(ref)-row[key+'_us']
        if abs(sum(row[f'{k}_gain_vs_mr23_us'] for k in ('li','resolve_fetch','gap','other_and_edges'))-
               row['net_gain_vs_mr23_us'])>.05:
            raise RuntimeError('Full-chain gain decomposition is not additive')
        window=groups.get(('window_budget',pf,b,miss))
        old_window=sum(s['fused_event_us'] for s in window)/len(window) if window else None
        row['over_window_budget']=row['fused_event_us']/old_window if old_window else None
        row['net_gain_vs_window_us']=old_window-row['fused_event_us'] if old_window else None
        row['per_round_over_window']=[s['fused_event_us']/next(r['fused_event_us'] for r in window
            if r['round']==s['round']) for s in samples] if window else None
        table.append(row)
    with (output/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, list(table[0]))
        writer.writeheader(); writer.writerows(table)
    save(output/'fixture_hashes.json', [{'prefetch':k[0], 'batch':k[1], 'miss_directory':k[2], 'sha256':v} for k,v in fixtures.items()])
    print('us/call | same-graph event means | positive gain is faster', flush=True)
    print('SerialBefore = SDK LI + LIDU + completed BatchGet; MR14 SerialAfter is not measured here.', flush=True)
    print('case             pf  B  P miss% SerialBefore    Fused      LI Resolve/Get   /MR23   Gain us', flush=True)
    for r in table:
        print(f"{r['case']:16} {r['prefetch']:4} {r['batch']:2} {r['score_parts']:2} {r['miss_percent']:5g} {r['serial_event_us']:12.2f} {r['fused_event_us']:8.2f} {r['li_us']:7.2f} {r['resolve_fetch_us']:11.2f} {r['over_mr23']:7.3f} {r['net_gain_vs_mr23_us']:9.2f}")
    print('Window components (us/call): case              pf  B miss%  HotScore ColdScore   theta QueryTopK Service   Final   LI launches', flush=True)
    for r in table:
        if r['score_mode'] in ('window','window_query'):
            q='inline' if r['query_topk_us'] is None else f"{r['query_topk_us']:.2f}"
            print(f"{r['case']:16} {r['prefetch']:4} {r['batch']:2} {r['miss_percent']:5g} {r['score_hot_us']:9.2f} {r['score_cold_us']:9.2f} {r['theta_us']:7.2f} {q:>9} {r['service_us']:7.2f} {r['final_us']:7.2f} {r['li_kernel_launches']:13}")
    return table


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', type=int, default=5)
    p.add_argument('--cases', default='mr23,window_budget',
                   help='comma-separated: '+','.join(ROUTES)+'; include mr23 as reference')
    p.add_argument('--batches', default='1,2,4,8,16', help='distinct batch sizes, one batch per job')
    p.add_argument('--parts-map', default=','.join(f'{b}:{v}' for b,v in DEFAULT_PARTS.items()),
                   help='history P for budget routes; B:P CSV with B*P<=32, no implicit waves')
    p.add_argument('--kvaf-root', type=Path, required=True)
    p.add_argument('--prepare-kvaf', action='store_true',
                   help='use kvaf-root as local source; prepare/build a separate pinned dependency')
    p.add_argument('--pinned-kvaf-root', type=Path, default=ROOT.parent/'KVAF-score-t2-1c131038')
    p.add_argument('--kvaf-server', default='127.0.0.1:28885')
    p.add_argument('--miss-percent', default='0,3,5,10,15,20')
    p.add_argument('--prefetch', default='1024', help='comma-separated budgets; add 0 as an overlap control')
    p.add_argument('--intermediate-fetches', type=int, default=4)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--warmup', type=int, default=4)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--timeout-seconds', type=int, default=1800)
    p.add_argument('--output', type=Path)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    selected=a.cases.split(',')
    if 'mr23' not in selected or any(c not in ROUTES for c in selected) or len(set(selected))!=len(selected):
        p.error('cases must include mr23 and contain distinct supported routes')
    if a.rounds < 1 or a.warmup < 1 or a.iterations < 1 or not 2 <= a.intermediate_fetches <= 32:
        p.error('rounds/warmup/iterations must be positive; T2 requires 2..32 windows')
    try:
        misses = [float(x) for x in a.miss_percent.split(',')]
        prefetches = [int(x) for x in a.prefetch.split(',')]
        batches, parts_map = batch_configuration(a.batches, a.parts_map)
        routes = plan_routes(selected, batches, parts_map)
    except ValueError as error:
        p.error(f'invalid experiment configuration: {error}')
    if (not misses or any(not 0 <= x <= 20 for x in misses) or len(set(misses)) != len(misses) or
            not prefetches or any(not 0 <= x <= 2048 or x % a.intermediate_fetches for x in prefetches) or
            len(set(prefetches)) != len(prefetches)):
        p.error('miss values must be distinct in 0..20; prefetch must divide evenly into windows')
    output = a.output.resolve() if a.output else Path(tempfile.mkdtemp(prefix='lifetch_score_t2_'+time.strftime('%Y%m%d_%H%M%S')+'_', dir='/home/w00568585' if Path('/home/w00568585').is_dir() else None))
    if a.output:
        output.mkdir(parents=True, exist_ok=False)
    output = output.resolve()
    print(f'Results directory: {output}', flush=True)
    print(f'Source checkout: {ROOT}', flush=True)
    kvaf = a.kvaf_root.resolve()
    if a.prepare_kvaf and not a.dry_run:
        from prepare_kvaf import prepare_and_build
        preparation = dict(status='PREPARING', source=str(kvaf),
                           destination=str(a.pinned_kvaf_root.resolve()))
        save(output/'dependency_preparation.json', preparation)
        try:
            kvaf = prepare_and_build(kvaf, a.pinned_kvaf_root, log=output/'dependency-build.log')
            preparation.update(status='READY', effective_kvaf_root=str(kvaf))
        except BaseException as error:
            preparation.update(status='FAILED', error=str(error))
            raise
        finally:
            save(output/'dependency_preparation.json', preparation)
    sources = {case: baseline(output) if case=='mr23' else ROOT for case in selected}
    manifest = dict(status='PLANNED', baseline_commit=BASE, mr14_reference=MR14,
                    integration='adapted direct window producers; original MR23 ranked T2/Final/URMA tail',
                    requested_batches=batches, budget_parts_map=parts_map,
                    serial_reference='SerialBefore: SDK LI + LIDU + completed BatchGet; no MR14 SerialAfter adapter',
                    score_budget='B*P<=32 AIC; no implicit oversubscribed waves',
                    batch_isolation='One batch size per job; baseline/candidate use identical DDR pool capacity', candidate_commit=command(['git','rev-parse','HEAD']),
                    candidate_status=command(['git','status','--short']), configuration=vars(a).copy(),
                    baseline_build_override=json.loads((output/'baseline_build_override.json').read_text()),
                    effective_kvaf_root=str(kvaf),
                    completion='URMA Fetch + Index update + output slot validation; SFA checked outside timing')
    manifest['configuration'] = {k:str(v) if isinstance(v, Path) else v for k,v in manifest['configuration'].items()}
    save(output/'manifest.json', manifest)
    (output/'candidate.diff').write_text(command(['git','diff','HEAD']))
    jobs=[]
    for round_id in range(a.rounds+1):
        ordered_routes = routes if round_id % 2 == 0 else list(reversed(routes))
        order=[(case,bs,parts,mode,pf) for pf in prefetches for b in batches
               for case,bs,parts,mode in ordered_routes if int(bs)==b]
        shift=round_id % len(order)
        for case,bs,parts,mode,pf in order[shift:]+order[:shift]:
            name=f'r{round_id}_pf{pf}_{case}_b{bs}'
            folder=output/name
            args=[sys.executable,'-u',str(sources[case]/'tests/operators/lifetch/test_lifetch.py'),'bench',
                '--transport','urma','--device',str(a.device),'--kvaf-root',str(kvaf),
                '--kvaf-server',a.kvaf_server,'--batches',bs,'--miss-percent',a.miss_percent,
                '--history','131072','--resident','16384','--index-heads','32',
                '--seed','43','--prefetch-tokens',str(pf),'--prefetch-rank-start','1024',
                '--intermediate-fetches',str(a.intermediate_fetches),'--warmup',str(a.warmup),
                '--iterations',str(a.iterations),'--timeout-seconds',str(a.timeout_seconds),
                '--output',str(folder)]
            if parts:
                args += ['--li-score-parts',str(parts),'--li-score-mode',mode]
            jobs.append(dict(name=name,case=case,batch=int(bs),parts=parts,mode=mode,pf=pf,round=round_id,
                             output=str(folder),log=str(output/(name+'.log')),args=args))
    save(output/'jobs.json', jobs)
    if a.dry_run:
        print(f'Planned {len(jobs)} commands; no device or transport initialized. See jobs.json.')
        return
    actual_kvaf = command(['git','-C',str(kvaf),'rev-parse','HEAD'])
    if actual_kvaf != KVAF:
        message = (f'KVAF version mismatch at {kvaf}: expected {KVAF}, found {actual_kvaf}. '
                   f'Rerun the same command with --prepare-kvaf to clone/patch/build the pinned version '
                   f'at {a.pinned_kvaf_root.resolve()} from local Git history; the source checkout is preserved.')
        manifest.update(status='FAILED', actual_kvaf_commit=actual_kvaf, error=message)
        save(output/'manifest.json', manifest)
        raise RuntimeError(message)
    for file in ('bin/kvaf_execute_batch_get_kernels.o','bin/kvaf_execute_batch_put_kernels.o','CMakeCache.txt'):
        if not (kvaf/'build-npu'/file).is_file():
            raise RuntimeError(f'Missing built dependency: {kvaf/"build-npu"/file}')
    probe = ('import json,sys; import torch,torch_npu; '
             'from kvip.runtime.build import kvaf_build_paths; '
             'from kvip.runtime.device import query_device_resources; '
             'from pathlib import Path; '
             'torch.npu.set_device(int(sys.argv[1])); '
             'cann,_=kvaf_build_paths(Path(sys.argv[2])/"build-npu"); '
             'print(json.dumps(query_device_resources(cann,int(sys.argv[1]))))')
    probe_env = dict(os.environ)
    probe_env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'python'), str(kvaf/'build-npu/python'),
                                            probe_env.get('PYTHONPATH', '')])
    probe_output = subprocess.check_output([sys.executable, '-c', probe, str(a.device), str(kvaf)],
                                           env=probe_env, text=True)
    resources = json.loads(probe_output.splitlines()[-1])
    resources['effective_aiv'] = 2*resources['effective_aic']
    required_aic = max(max(2*int(bs), int(bs)*parts) for _,bs,parts,_ in routes)
    if resources['effective_aic'] < required_aic:
        raise RuntimeError(f'Selected score/T2 routes require {required_aic} effective AIC; measured {resources}')
    manifest.update(status='RUNNING', kvaf_commit=KVAF,
        device_resources=resources,
        kvaf_get_sha256=checksum(kvaf/'build-npu/bin/kvaf_execute_batch_get_kernels.o'),
        kvaf_put_sha256=checksum(kvaf/'build-npu/bin/kvaf_execute_batch_put_kernels.o'))
    vendor_paths = sorted(str(p) for p in (ROOT/'csrc/operators/lidu/_custom_opp/vendors').glob('*')
                          if (p/'op_api/lib/libcust_opapi.so').is_file())
    manifest['shared_lidu_vendors'] = vendor_paths
    save(output/'manifest.json',manifest)
    try:
        for i,job in enumerate(jobs):
            print(f"[{time.strftime('%H:%M:%S')}] {i+1}/{len(jobs)} {job['name']}"+(' (discard)' if job['round']==0 else ''),flush=True)
            print('Log: '+job['log'],flush=True)
            env=dict(os.environ)
            env['PYTHONPATH']=os.pathsep.join([str(sources[job['case']]/'python'),str(kvaf/'build-npu/python'),env.get('PYTHONPATH','')])
            if vendor_paths:
                env['ASCEND_CUSTOM_OPP_PATH'] = os.pathsep.join(
                    vendor_paths + list(filter(None, [env.get('ASCEND_CUSTOM_OPP_PATH')])))
            with open(job['log'],'w') as log:
                result=subprocess.run(job['args'],cwd=sources[job['case']],env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                print(Path(job['log']).read_text(errors='replace')[-12000:],flush=True)
                raise RuntimeError(f"{job['name']} exit {result.returncode}; inspect its logs")
        summarize(output,jobs,a.rounds)
        manifest['status']='PASS'
    except BaseException as error:
        manifest.update(status='FAILED',error=str(error)); raise
    finally:
        save(output/'manifest.json',manifest)
    print(f'Completed. Summary: {output/"summary.csv"}',flush=True)


if __name__=='__main__':
    main()
