#!/usr/bin/env python3
"""Physical A5 msprof capture, separate from the paired timing experiment."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from run import geometry, make_fixture, sha, suite_cases
from cases import lease


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('plan','run'))
    p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,default=5)
    p.add_argument('--suite',choices=('orthogonal7','orthogonal8','history7','batch16','batch16_cores'),default='batch16')
    p.add_argument('--history',type=int,default=131072)
    p.add_argument('--metrics',choices=('PipeUtilization','Memory','ResourceConflictRatio'),default='PipeUtilization')
    p.add_argument('--timeout',type=int,default=900)
    p.add_argument('--work-grain',type=int,default=2048)
    p.add_argument('--fine-grain',type=int,choices=(128,512),default=128)
    a=p.parse_args();a.build=a.build.resolve();a.output=a.output.resolve()
    if a.suite=='batch16_cores':
        from b16_cores import configs as core_configs
        configs=core_configs(a.fine_grain,a.history)
    else:configs=suite_cases(a.suite) if a.suite in ('batch16','history7') else [dict(batch=1,history_parts=16 if a.suite=='orthogonal7' else 32,query_parts=1,schedule='waves')]
    cases=[]
    for c in configs:
        c=dict(c);b=c['batch'];c.setdefault('aics',30 if b==16 else 32);c.setdefault('aivs',60 if b==16 else 64)
        c.setdefault('name',f"B{b}_T{c['history_parts']}_{c['schedule']}")
        grain=c.get('work_grain',a.work_grain if c['schedule']!='waves' else 2048)
        c['geometry']=geometry(c['history_parts'],1,a.history,c['aics'],c['aivs'],b,c['schedule'],grain);cases.append(c)
    if a.action=='plan':print(json.dumps(cases,indent=2));return
    if any(x in os.environ.get('LD_PRELOAD','') for x in ('camodel','cmodel')):p.error('physical A5 profile rejects Model preload')
    profiler=shutil.which('msprof')
    if not profiler:p.error('msprof unavailable; activate CANN environment')
    driver=a.build/'build/li_qk';build=json.loads((a.build/'build-manifest.json').read_text())
    if build['status']!='BUILT' or sha(driver)!=build['binary_sha256']:p.error('unaccepted build')
    caps=json.loads(subprocess.check_output([str(driver),'--capabilities'],text=True))
    if caps['run_mode']!='npu':p.error('physical NPU binary required')
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='RUNNING',scope='physical A5 msprof; excluded from benchmark summary',cases=cases,
                  binary_sha256=sha(driver),build_manifest=build,device=a.device)
    record=a.output/'manifest.json'
    with lease('npu',a.device,Path(f'/tmp/kvip-li-qk-device-{a.device}.lease')):
        fixtures={}
        for c in cases:
            b=c['batch']
            if b not in fixtures:
                fixtures[b]=make_fixture(a.output/f'fixture_b{b}',b,a.history,32)
            c['fixture']=fixtures[b];folder=a.output/c['name'];folder.mkdir()
            command=[str(driver),str(a.output/f'fixture_b{b}'),str(folder/'oracle'),str(a.history),'32',
                     str(c['aics']),str(c['aivs']),str(c['history_parts']),'1','4','5',str(a.device),str(b),c['schedule'],'li',str(c['geometry']['work_grain'])]
            # File preserves argv safely even when paths contain shell metacharacters.
            app=folder/'application.sh';app.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec '+shlex.join(command)+'\n');app.chmod(0o700)
            argv=[profiler,'--application='+shlex.quote(str(app)),'--output='+str(folder/'trace'),
                  '--task-time=on','--aic-metrics='+a.metrics]
            c.update(argv=argv,status='RUNNING');record.write_text(json.dumps(manifest,indent=2)+'\n')
            print('Profiling '+c['name'],flush=True)
            with (folder/'msprof.log').open('w') as log:
                try:r=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout);c['exit_code']=r.returncode
                except subprocess.TimeoutExpired:c.update(exit_code=None,status='TIMEOUT');record.write_text(json.dumps(manifest,indent=2)+'\n');raise
            proof=folder/'oracle/report.json'
            if c['exit_code'] or not proof.exists():
                c['status']='FAILED';record.write_text(json.dumps(manifest,indent=2)+'\n');raise RuntimeError('profile failed: '+str(folder/'msprof.log'))
            export=[profiler,'--export=on','--output='+str(folder/'trace')]
            c['export_argv']=export
            with (folder/'export.log').open('w') as log:
                result_export=subprocess.run(export,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
            c['export_exit_code']=result_export.returncode
            if result_export.returncode:raise RuntimeError('msprof export failed: '+str(folder/'export.log'))
            result=json.loads(proof.read_text())
            if result['status']!='PASS' or not result['topk_threshold_oracle'] or sha(folder/'oracle/score-keys.bin')!=fixtures[b]['hashes']['oracle']:
                raise RuntimeError('profile oracle rejected')
            traces=list((folder/'trace').rglob('*.json'))+list((folder/'trace').rglob('*.csv'))
            if not traces:raise RuntimeError('msprof returned without exported trace; inspect '+str(folder/'trace'))
            expected_names=['li_qk_topk','li_qk_score_balanced' if c['schedule']!='waves' else 'li_qk_score']
            if not c['geometry']['canonical_alias']:expected_names.append('li_qk_scatter')
            found=set()
            for trace in traces:
                with trace.open('r',errors='replace') as f:
                    for line in f:
                        found.update(n for n in expected_names if n in line)
            if found!=set(expected_names):raise RuntimeError('exported trace missing LI kernel names: '+str(set(expected_names)-found))
            c.update(status='PROFILE_PASSED',report_sha256=sha(proof),trace_files=[str(x.relative_to(a.output)) for x in traces])
        manifest['status']='PROFILE_PASSED';record.write_text(json.dumps(manifest,indent=2)+'\n')
    print(a.output)


if __name__=='__main__':main()
