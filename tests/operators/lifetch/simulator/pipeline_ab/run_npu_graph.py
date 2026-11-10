#!/usr/bin/env python3
"""Graph replay A/B for batched Resolver + Index update (no LI/Fetch)."""
import argparse
import csv
import datetime
import json
import re
import statistics
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def run(command, log, timeout, label):
    started = time.monotonic()
    with log.open('w') as out:
        process = subprocess.Popen([str(x) for x in command], cwd=REPO,
                                   stdout=out, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed > timeout:
                import os, signal
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f'{label} timed out after {timeout}s; log={log}')
            try:
                process.wait(timeout=min(20, max(0.1, timeout-elapsed)))
            except subprocess.TimeoutExpired:
                print(f'RUNNING {label} +{time.monotonic()-started:.0f}s log={log}', flush=True)
    text = log.read_text(errors='replace')
    if process.returncode:
        raise RuntimeError(f'{label} exit={process.returncode}; log={log}\n' + '\n'.join(text.splitlines()[-30:]))
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', type=int, default=5)
    p.add_argument('--batches', default='1,2,4,8,16')
    p.add_argument('--kvaf-root', required=True, type=Path)
    p.add_argument('--soc', default='Ascend950PR_9599')
    p.add_argument('--cann', type=Path, default=Path('/usr/local/Ascend/cann'))
    p.add_argument('--warmup', type=int, default=4)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--profile-replays', type=int, default=3)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    try:
        batches = [int(x) for x in args.batches.split(',')]
    except ValueError:
        p.error('batches must be comma-separated integers')
    if not batches or len(set(batches)) != len(batches) or any(x not in (1, 2, 4, 8, 16) for x in batches):
        p.error('batches must contain unique values from 1,2,4,8,16')
    if args.device < 0 or args.warmup < 0 or args.iterations < 1 or args.timeout < 1 or args.profile_replays < 1:
        p.error('invalid device/warmup/iterations/timeout')
    kvaf = args.kvaf_root.resolve() / 'access_operator/demo/aiv_batch_get/include'
    if not kvaf.is_dir():
        p.error(f'KVAF include missing: {kvaf}')
    output = (args.output or REPO / 'tests/results/resolver-index-graph' /
              datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = output / 'summary.log'
    lines = ['Graph replay A/B, separate graphs; every sample restores inputs outside timed event.',
             'BS | serial_p50_us | parallel_p50_us | change | serial_range_us | parallel_range_us']
    summary.write_text('\n'.join(lines)+'\n')
    print(f'Summary: {summary}', flush=True)
    build = output/'build'
    try:
        run(['cmake','-S',HERE,'-B',build,'-DRUN_MODE=npu',f'-DSOC_VERSION={args.soc}',
             f'-DASCEND_CANN_PACKAGE_PATH={args.cann}',f'-DKVAF_INCLUDE={kvaf}',
             f'-DOP_ROOT={REPO / "csrc/operators/lightning_indexer_with_fetch_safe"}'], output/'configure.log',120,'configure')
        run(['cmake','--build',build,'--target','pipeline_ab_parallel_sim','-j2'], output/'build.log',600,'build')
        executable=build/'pipeline_ab_parallel_sim'
        pattern=re.compile(r'^GRAPH_SAMPLE iteration=(\d+) event_status=(-?\d+) event_ms=([0-9.eE+-]+)',re.M)
        pass_pattern=re.compile(r'^PASS mode=(\w+) fixture=([\w-]+) final_miss=(\d+)',re.M)
        results={}
        for bs in batches:
            results[str(bs)]={}
            for fixture in ('normal','collision','cross-query'):
                for mode in ('serial','pipeline'):
                    repeats=args.warmup+args.iterations if fixture=='normal' else 1
                    text=run([executable,mode,fixture,args.device,'graph',repeats,bs],
                             output/f'bs{bs}-{mode}-{fixture}.log',args.timeout,
                             f'BS={bs} {mode} {fixture}')
                    passed=pass_pattern.search(text)
                    if not passed or passed.group(1,2)!=(mode,fixture):
                        raise RuntimeError(f'BS={bs} {mode} {fixture}: missing verified PASS')
                    samples=pattern.findall(text)
                    if len(samples)!=repeats or any(int(status)!=0 or float(ms)<=0 for _,status,ms in samples):
                        raise RuntimeError(f'BS={bs} {mode} {fixture}: missing or invalid event samples')
                    if fixture=='normal':
                        results[str(bs)][mode]=[float(ms)*1000 for _,_,ms in samples[args.warmup:]]
            a=results[str(bs)]['serial'];b=results[str(bs)]['pipeline'];am=statistics.median(a);bm=statistics.median(b)
            line=f'{bs} | {am:.3f} | {bm:.3f} | {(bm/am-1)*100:+.2f}% | {min(a):.3f}..{max(a):.3f} | {min(b):.3f}..{max(b):.3f}'
            lines.append(line);summary.write_text('\n'.join(lines)+'\n');print(line,flush=True)
            stage_values={}
            for mode in ('serial','pipeline'):
                prof=output/f'profile-bs{bs}-{mode}'
                command=shlex.join([str(executable),mode,'normal',str(args.device),'graph',str(args.profile_replays),str(bs)])
                text=run(['msprof',f'--output={prof}','--task-time=on','--runtime-api=on',f'--application={command}'],
                         output/f'profile-bs{bs}-{mode}.log',args.timeout,f'profile BS={bs} {mode}')
                if not pass_pattern.search(text):
                    raise RuntimeError(f'profile BS={bs} {mode}: application did not report PASS')
                if not list(prof.rglob('op_summary*.csv')):
                    run(['msprof','--export=on',f'--output={prof}'],output/f'profile-export-bs{bs}-{mode}.log',args.timeout,f'export BS={bs} {mode}')
                groups={}
                for file in prof.rglob('op_summary*.csv'):
                    with file.open(newline='',encoding='utf-8-sig') as stream:
                        for row in csv.DictReader(stream):
                            name=' '.join(str(value) for value in row.values())
                            for kernel in ('test_ready_resolver','kvop_direct_update_kernel','kvop_direct_apply_kernel','test_pipeline_ab_parallel'):
                                if kernel not in name:
                                    continue
                                duration=next((v for k,v in row.items() if k and 'task duration' in k.lower() and 'us' in k.lower()),None)
                                try:
                                    value=float(duration)
                                except (ValueError,TypeError):
                                    continue
                                if value>0:groups.setdefault(kernel,[]).append(value)
                wanted=('test_ready_resolver','kvop_direct_update_kernel','kvop_direct_apply_kernel') if mode=='serial' else ('test_pipeline_ab_parallel',)
                for kernel in wanted:
                    values=groups.get(kernel,[])
                    if len(values)<args.profile_replays+1:
                        raise RuntimeError(f'profile BS={bs} {mode}: insufficient {kernel} tasks ({len(values)}); raw={prof}')
                    # One eager warmup precedes capture; keep the final graph replays.
                    stage_values[kernel]=statistics.median(values[-args.profile_replays:])
                api_values=[]
                for file in prof.rglob('api_statistic*.csv'):
                    with file.open(newline='',encoding='utf-8-sig') as stream:
                        for row in csv.DictReader(stream):
                            if 'aclmdlRIExecuteAsync' not in str(row.get('API Name','')):
                                continue
                            try:
                                api_values.append(float(row['Avg(us)']))
                            except (ValueError,KeyError):
                                pass
                stage_values[f'{mode}_host_submit_avg']=api_values[0] if api_values else None
                if mode=='pipeline':
                    cycles=[line for line in text.splitlines() if line.startswith('PIPELINE_CYCLES ')]
                    if len(cycles)!=bs:
                        raise RuntimeError(f'profile BS={bs}: expected {bs} cycle records, got {len(cycles)}')
                    results[str(bs)]['pipeline_cycles']=cycles
            stage=f'STAGES BS={bs} device_task_p50_us: A_resolve={stage_values["test_ready_resolver"]:.3f} A_plan={stage_values["kvop_direct_update_kernel"]:.3f} A_apply={stage_values["kvop_direct_apply_kernel"]:.3f} B_fused_kernel={stage_values["test_pipeline_ab_parallel"]:.3f}'
            lines.append(stage);print(stage,flush=True)
            host=f'HOST BS={bs} profiled aclmdlRIExecuteAsync avg_us: A={stage_values["serial_host_submit_avg"]} B={stage_values["pipeline_host_submit_avg"]}; None means not exported'
            lines.append(host);print(host,flush=True)
            for cycle in results[str(bs)]['pipeline_cycles']:
                lines.append(cycle);print(cycle,flush=True)
            summary.write_text('\n'.join(lines)+'\n')
        lines.append('PASS: each replay checks per-request status; full output/oracle checked on final replay for each graph.')
        lines.append('Graph event includes replay scheduling; no LI, Fetch/CQ, SFA, or input restoration. Separate graphs, not paired replay.')
        lines.append('Stage task times come from a separate profiled pass; B per-request timestamps are raw device cycles, not microseconds.')
        summary.write_text('\n'.join(lines)+'\n');(output/'samples.json').write_text(json.dumps(results,indent=2)+'\n')
        print('\n'.join(lines[-3:]),flush=True)
    except (RuntimeError,OSError) as exc:
        lines.append('FAIL: '+str(exc));summary.write_text('\n'.join(lines)+'\n');print(lines[-1],flush=True);return 1
    finally:
        print(f'Summary: {summary}',flush=True)
    return 0


if __name__=='__main__':
    sys.exit(main())
