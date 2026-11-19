#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MR14 direct window score/predicate producers + MR23 ranked T2/Fetch tail.

No Git fetch or Docker command. Optional --prepare-kvaf builds a separate pinned
checkout from local history. Both sources use the MR23 graph timing protocol.
"""
import argparse
import csv
import hashlib
import inspect
import json
import os
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
    'mr23': ('1,16', 0, 'original'),
    'window_p16': ('1', 16, 'window'),
    'window_p2': ('16', 2, 'window'),
    'score_t2_p2': ('1,16', 2, 'full'),
    'score_t2_p16': ('1', 16, 'full'),
}


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
            key = (job['pf'], miss_dir)
            digest = fixture_hashes(path.parent)
            if not digest:
                raise RuntimeError(f'No fixture operand files: {path.parent}')
            if key in fixtures and fixtures[key] != digest:
                raise RuntimeError(f'Operand mismatch across routes: {key}')
            fixtures[key] = digest
        for path in reports:
            report = json.loads(path.read_text())
            if report.get('correctness') != 'PASS' or report.get('execution') != 'COMPLETED':
                raise RuntimeError(f'Failed correctness/completion: {path}')
            serial, fused = [json.loads((path.parent / name / 'path_result.json').read_text())
                             for name in ('serial', 'fused')]
            if serial['all_rounds']['timing_protocol'] != 'one_profiled_graph' or fused['all_rounds']['timing_protocol'] != 'one_profiled_graph':
                raise RuntimeError('Timing protocol changed; do not mix old p50 with graph-event means')
            s, f = serial['all_rounds'], fused['all_rounds']
            stage = f['stage_mean_us']
            # Kernel names are retained with invocation boundaries in raw trace.
            rows.append(dict(round=job['round'], case=job['case'], prefetch=job['pf'],
                batch=report['batch'], miss_percent=report['miss_percent'], score_parts=job['parts'], score_mode=job.get('mode','full'),
                score_geometry=fused.get('score_geometry'),
                serial_event_us=s['event_mean_us'], fused_event_us=f['event_mean_us'],
                fused_profiler_p50_us=f['p50_us'], li_us=stage['li'],
                resolve_fetch_us=stage['resolve_update'], gap_us=f['inter_kernel_gap_mean_us'],
                effective_aic=fused.get('effective_aic', resources.get('effective_aic')),
                effective_aiv=fused.get('effective_aiv', resources.get('effective_aiv')),
                li_schedule=fused.get('li_schedule'),
                trace_paths=[str(p) for p in (path.parent/'fused').rglob('trace.json')]))
    save(output / 'samples.json', rows)
    groups = {}
    for row in rows:
        groups.setdefault((row['case'], row['prefetch'], row['batch'], row['miss_percent']), []).append(row)
    fields = ('serial_event_us', 'fused_event_us', 'fused_profiler_p50_us', 'li_us', 'resolve_fetch_us', 'gap_us')
    table = []
    for (case, pf, b, miss), samples in groups.items():
        if sorted(s['round'] for s in samples) != list(range(1, measured_rounds+1)):
            raise RuntimeError('Missing or duplicated measured rounds')
        row = dict(case=case, prefetch=pf, batch=b, miss_percent=miss,
                   score_parts=samples[0]['score_parts'], score_mode=samples[0]['score_mode'],
                   score_geometry=samples[0]['score_geometry'], effective_aic=samples[0]['effective_aic'],
                   li_schedule=samples[0]['li_schedule'])
        row.update({f: sum(s[f] for s in samples)/len(samples) for f in fields})
        ref = groups[('mr23', pf, b, miss)]
        old = sum(s['fused_event_us'] for s in ref)/len(ref)
        row.update(over_mr23=row['fused_event_us']/old,
                   over_serial=row['fused_event_us']/row['serial_event_us'],
                   serial_drift_over_mr23=row['serial_event_us']/(sum(s['serial_event_us'] for s in ref)/len(ref)),
                   per_round_over_mr23=[s['fused_event_us']/next(r['fused_event_us'] for r in ref if r['round']==s['round']) for s in samples])
        table.append(row)
    with (output/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, list(table[0]))
        writer.writeheader(); writer.writerows(table)
    save(output/'fixture_hashes.json', [{'prefetch':k[0], 'miss_directory':k[1], 'sha256':v} for k,v in fixtures.items()])
    print('case             B miss%   Serial    Fused    LI    Resolve/Get   /MR23', flush=True)
    for r in table:
        print(f"{r['case']:16} {r['batch']:2} {r['miss_percent']:5g} {r['serial_event_us']:8.2f} {r['fused_event_us']:8.2f} {r['li_us']:7.2f} {r['resolve_fetch_us']:11.2f} {r['over_mr23']:7.3f}")
    return table


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', type=int, default=5)
    p.add_argument('--cases', default='mr23,window_p16,window_p2',
                   help='comma-separated: '+','.join(ROUTES)+'; include mr23 as reference')
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
    except ValueError:
        p.error('invalid miss/prefetch CSV')
    if (not misses or any(not 0 <= x <= 20 for x in misses) or len(set(misses)) != len(misses) or
            not prefetches or any(not 0 <= x <= 2048 or x % a.intermediate_fetches for x in prefetches) or
            len(set(prefetches)) != len(prefetches)):
        p.error('miss values must be distinct in 0..20; prefetch must divide evenly into windows')
    output = a.output.resolve() if a.output else Path(tempfile.mkdtemp(prefix='lifetch_score_t2_'+time.strftime('%Y%m%d_%H%M%S')+'_', dir='/home/w00568585' if Path('/home/w00568585').is_dir() else None))
    if a.output:
        output.mkdir(parents=True, exist_ok=False)
    output = output.resolve()
    print(f'Results directory: {output}', flush=True)
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
                    batch_isolation='One batch size per job; baseline/candidate use identical DDR pool capacity', candidate_commit=command(['git','rev-parse','HEAD']),
                    candidate_status=command(['git','status','--short']), configuration=vars(a).copy(),
                    baseline_build_override=json.loads((output/'baseline_build_override.json').read_text()),
                    effective_kvaf_root=str(kvaf),
                    completion='URMA Fetch + Index update + output slot validation; SFA checked outside timing')
    manifest['configuration'] = {k:str(v) if isinstance(v, Path) else v for k,v in manifest['configuration'].items()}
    save(output/'manifest.json', manifest)
    (output/'candidate.diff').write_text(command(['git','diff','HEAD']))
    jobs=[]
    routes=[(case,bs,parts,mode) for case in selected
            for batches,parts,mode in [ROUTES[case]] for bs in batches.split(',')]
    for round_id in range(a.rounds+1):
        order=[(case,bs,parts,mode,pf) for pf in prefetches for case,bs,parts,mode in routes]
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
    if resources['effective_aic'] < 32:
        raise RuntimeError(f'B16/P2 requires 32 effective AIC; measured {resources}')
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
