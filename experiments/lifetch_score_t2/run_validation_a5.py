#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sequential physical-A5 checks of MR14 history7 and MR23+MR14 LIFetch.

The history suite is the original B1 numeric LI comparison; the chain suite
uses the current fusion candidates and SDK SerialBefore. Their timing and
fixtures stay separate. This launcher has no network fetch or Docker commands.
"""
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MR14 = 'c684d79f3a4f8432bf0915edeba392c7d54ba289'
KVAF = '1c1310387ebfb6776a07316874e924265b209ada'
PARTS = (1, 2, 4, 8, 16, 32, 30)
CANN_CMAKE = 'compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake'


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True,
                                   stderr=subprocess.STDOUT).strip()


def cache_values(root):
    cache = Path(root)/'build-npu/CMakeCache.txt'
    values = {}
    if cache.is_file():
        for line in cache.read_text().splitlines():
            if line and not line.startswith(('#', '//')) and '=' in line:
                key, value = line.split('=', 1)
                values[key.split(':', 1)[0]] = value
    return values


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--only', choices=('both', 'history7', 'chain'), default='both')
    p.add_argument('--device', type=int, default=0, help='logical device in the existing experiment container')
    p.add_argument('--kvaf-root', type=Path, required=True)
    p.add_argument('--kvaf-server', default='127.0.0.1:28985', help='20-machine server, used only by the chain experiment')
    p.add_argument('--mr14-source', type=Path, help='clean checkout at the pinned MR14 commit, for history7 only')
    p.add_argument('--cann-root', '--cann', dest='cann_root', type=Path,
                   help='matching CANN root; default from KVAF build cache or CANN environment')
    p.add_argument('--soc', help='actual SoC; default from KVAF build cache or SOC_VERSION')
    p.add_argument('--kvaf-include', type=Path, help='compatible history device headers; default from KVAF root')
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--prepare-kvaf', action='store_true', help='build a separate pinned dependency from local Git history first')
    p.add_argument('--pinned-kvaf-root', type=Path, default=ROOT.parent/'KVAF-score-t2-1c131038')
    p.add_argument('--batches', default='1,2,4,8,16')
    p.add_argument('--cases', default='mr23,window_budget,aligned_budget')
    p.add_argument('--parts-map', default='1:16,2:16,4:8,8:4,16:2')
    p.add_argument('--miss-percent', default='5,10,15,20,25,30')
    p.add_argument('--prefetch', default='0', help='aligned_budget requires 0')
    p.add_argument('--intermediate-fetches', type=int, default=4)
    p.add_argument('--rounds', type=int, default=3, help='measured rounds for both suites; r0 is always discarded')
    p.add_argument('--warmup', type=int, default=4)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--timeout-seconds', type=int, default=1800)
    p.add_argument('--profile', action='store_true', help='collect separate history7 msprof traces after history timing')
    p.add_argument('--output', type=Path, help='fresh parent result directory')
    p.add_argument('--dry-run', '--plan-only', dest='dry_run', action='store_true', help='write both plans; do not build, initialize NPU or access KVAF service')
    return p


def validate(a, p):
    if (a.device < 0 or a.rounds < 1 or not 1 <= a.warmup <= 100 or
            not 1 <= a.iterations <= 1000 or a.jobs < 1 or a.timeout_seconds < 1):
        p.error('valid nonnegative device and positive sampling/build parameters required')
    # Validate even in plan mode, before exporting source or creating results.
    from run_a5 import ROUTES, batch_configuration, plan_routes
    try:
        batches, mapping = batch_configuration(a.batches, a.parts_map)
        selected = a.cases.split(',')
        if 'mr23' not in selected or len(set(selected)) != len(selected) or any(c not in ROUTES for c in selected):
            raise ValueError('cases must include mr23 and contain distinct supported routes')
        plan_routes(selected, batches, mapping)
        misses = [float(x) for x in a.miss_percent.split(',')]
        budgets = [int(x) for x in a.prefetch.split(',')]
        if not misses or len(set(misses)) != len(misses) or any(not math.isfinite(x) or not 0 <= x <= 100 for x in misses):
            raise ValueError('miss values must be distinct and finite in 0..100')
        if (not 2 <= a.intermediate_fetches <= 32 or not budgets or
                len(set(budgets)) != len(budgets) or
                any(not 0 <= x <= 2048 or x % a.intermediate_fetches for x in budgets)):
            raise ValueError('prefetch must divide evenly into 2..32 intermediate windows')
        if 'aligned_budget' in selected and budgets != [0]:
            raise ValueError('aligned_budget requires --prefetch 0')
    except ValueError as error:
        p.error(str(error))


def extract_source(archive, source):
    with tarfile.open(archive) as tar:
        if 'filter' in inspect.signature(tar.extractall).parameters:
            tar.extractall(source, filter='data')
        else:
            for member in tar.getmembers():
                target = (source/member.name).resolve()
                if source != target and source not in target.parents:
                    raise RuntimeError('MR14 archive path escapes its source directory')
                if member.issym() or member.islnk():
                    link = ((target.parent if member.issym() else source)/member.linkname).resolve()
                    if source != link and source not in link.parents:
                        raise RuntimeError('MR14 archive link escapes its source directory')
            tar.extractall(source)


def mr14_source(output, provided=None):
    output = Path(output).resolve()
    if provided:
        source = Path(provided).resolve()
        if git(source, 'rev-parse', 'HEAD') != MR14 or git(source, 'status', '--porcelain', '--untracked-files=no'):
            raise RuntimeError(f'--mr14-source must be a clean tracked checkout at {MR14}: {source}')
    else:
        if git(ROOT, 'rev-parse', MR14+'^{commit}') != MR14:
            raise RuntimeError('Pinned MR14 is missing from local Git history; fetch the MR14 branch on the host first')
        source = output/'mr14-source'
        source.mkdir()
        archive = output/'mr14-source.tar'
        with archive.open('wb') as f:
            subprocess.run(['git', '-C', str(ROOT), 'archive', MR14], stdout=f, check=True)
        extract_source(archive, source)
        archive.unlink()
        # Original prepare_score_source requires Git HEAD/status. A plain archive
        # inside this checkout would otherwise report the candidate's parent HEAD.
        subprocess.run(['git', 'init', '--quiet', str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'fetch', '--quiet', '--no-tags', '--depth', '1',
                        str(ROOT), MR14], check=True)
        subprocess.run(['git', '-C', str(source), 'update-ref', '--no-deref', 'HEAD', MR14], check=True)
        subprocess.run(['git', '-C', str(source), 'read-tree', MR14], check=True)
    required = ('experiments/li_qk/run.py', 'experiments/li_qk/test_history.py',
                'experiments/li_qk/profile.py', 'experiments/split_core/prepare_score_source.py')
    evidence = dict(commit=git(source, 'rev-parse', 'HEAD'), source=str(source),
                    external_checkout=bool(provided), files={})
    for name in required:
        path = source/name
        evidence['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(output/'mr14-source.json', evidence)
    return source


def toolchain(a, kvaf, *, required=True):
    values = cache_values(kvaf)
    cann = a.cann_root or values.get('ASCEND_CANN_PACKAGE_PATH') or os.environ.get('ASCEND_CANN_PACKAGE_PATH') or os.environ.get('ASCEND_HOME_PATH')
    soc = a.soc or values.get('SOC_VERSION') or os.environ.get('SOC_VERSION')
    if not cann:
        cann = next((p for p in (Path('/usr/local/Ascend/ascend-toolkit/latest'), Path('/usr/local/Ascend/cann-9.2.0'))
                     if (p/CANN_CMAKE).is_file()), None)
    headers = a.kvaf_include or kvaf/'access_operator/demo/aiv_batch_get/include'
    result = dict(cann_root=str(Path(cann).resolve()) if cann else None,
                  soc=soc, kvaf_include=str(Path(headers).resolve()))
    if required:
        if not cann or not (Path(cann)/CANN_CMAKE).is_file():
            raise RuntimeError('Set --cann-root to the matching CANN installation containing '+CANN_CMAKE)
        if not soc:
            raise RuntimeError('Set --soc to the actual A5 SoC; it was absent from the KVAF build cache and environment')
        if any(not (Path(headers)/n).is_file() for n in ('kvaf_batch_get_device.h', 'batch_get_abi.h')):
            raise RuntimeError('Set --kvaf-include to compatible KVAF device headers')
        cache_cann, cache_soc = values.get('ASCEND_CANN_PACKAGE_PATH'), values.get('SOC_VERSION')
        if cache_cann and Path(cache_cann).resolve() != Path(cann).resolve():
            raise RuntimeError('CANN differs from the existing KVAF build; rebuild a separate dependency with --prepare-kvaf')
        if cache_soc and cache_soc != soc:
            raise RuntimeError('SoC differs from the existing KVAF build; rebuild a separate dependency with --prepare-kvaf')
    return result


def activated_environment(cann):
    env = dict(os.environ)
    for directory in ('/usr/local/dcmi', '/usr/local/Ascend/driver/tools'):
        if (Path(directory)/'npu-smi').is_file():
            env['PATH'] = directory+os.pathsep+env.get('PATH', '')
    if cann:
        for script in (Path(cann)/'set_env.sh', Path(cann).parent/'set_env.sh'):
            if script.is_file():
                result = subprocess.run(['bash', '-c', 'set +u; source "$1" >&2; env -0',
                                         'kvip-validation-env', str(script)], env=env, check=True,
                                        stdout=subprocess.PIPE)
                env.update(os.fsdecode(item).split('=', 1) for item in result.stdout.split(b'\0') if b'=' in item)
                break
    return env


def planned_commands(a, output, source, kvaf, tools):
    commands = []
    def add(stage, argv, cwd):
        commands.append(dict(stage=stage, argv=argv, cwd=str(cwd), log=str(output/(stage+'.log'))))
    if a.only != 'chain':
        runner = str(source/'experiments/li_qk/run.py')
        common = ['--suite', 'history7', '--history', '131072', '--aics', '32', '--aivs', '64']
        add('history-checks', [sys.executable, str(source/'experiments/li_qk/test_history.py')], source)
        add('history-build', [sys.executable, runner, 'build', '--run-mode', 'npu',
            '--cann-root', tools['cann_root'] or '<matching CANN root>', '--soc', tools['soc'] or '<actual SoC>',
            '--kvaf-include', tools['kvaf_include'], '--jobs', str(a.jobs), '--output', str(output/'history7/build')], source)
        add('history-run', [sys.executable, '-u', runner, 'run', *common,
            '--device', str(a.device), '--warmup', str(a.warmup), '--iterations', str(a.iterations),
            '--rounds', str(a.rounds), '--timeout', str(a.timeout_seconds),
            '--build', str(output/'history7/build'), '--output', str(output/'history7/results')], source)
        if a.profile:
            add('history-profile', [sys.executable, '-u', str(source/'experiments/li_qk/profile.py'), 'run',
                '--suite', 'history7', '--history', '131072', '--device', str(a.device),
                '--timeout', str(a.timeout_seconds), '--build', str(output/'history7/build'),
                '--output', str(output/'history7/msprof')], source)
    if a.only != 'history7':
        add('chain-run', [sys.executable, '-u', str(HERE/'run_a5.py'), '--device', str(a.device),
            '--kvaf-root', str(kvaf), '--kvaf-server', a.kvaf_server, '--batches', a.batches,
            '--cases', a.cases, '--parts-map', a.parts_map, '--miss-percent', a.miss_percent,
            '--prefetch', a.prefetch, '--intermediate-fetches', str(a.intermediate_fetches),
            '--rounds', str(a.rounds), '--warmup', str(a.warmup), '--iterations', str(a.iterations),
            '--timeout-seconds', str(a.timeout_seconds), '--output', str(output/'chain')], ROOT)
    return commands


def run_logged(item, env):
    print(f"[{time.strftime('%H:%M:%S')}] {item['stage']}\nLog: {item['log']}", flush=True)
    with Path(item['log']).open('w') as log:
        process = subprocess.Popen(item['argv'], cwd=item['cwd'], env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                log.write(line); log.flush(); print(line, end='', flush=True)
            result = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise
        finally:
            process.stdout.close()
    if result:
        raise RuntimeError(f"{item['stage']} exit {result}; see {item['log']}")


def physical_info(output, env):
    with (output/'npu-smi.log').open('w') as log:
        subprocess.run(['npu-smi', 'info'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def main(argv=None):
    p = parser(); a = p.parse_args(argv); validate(a, p)
    output = a.output.resolve() if a.output else Path(tempfile.mkdtemp(
        prefix='lifetch_validation_'+time.strftime('%Y%m%d_%H%M%S')+'_',
        dir='/home/w00568585' if Path('/home/w00568585').is_dir() else None))
    if a.output:
        output.mkdir(parents=True, exist_ok=False)
    print('Validation results: '+str(output), flush=True)
    manifest = dict(status='PLANNED', stage='source', configuration={k:str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
                    candidate_commit=git(ROOT, 'rev-parse', 'HEAD'), suites=['history7', 'chain'] if a.only=='both' else [a.only],
                    serial_reference='Chain SerialBefore is SDK LI + LIDU + completed BatchGet; no MR14 SerialAfter path',
                    history_scope='B1/Q4/H131072/G32/D128/K2048; score and numeric LI; no classify/Fetch/Service/Index',
                    statistics='History: mean of round p50 and paired T1 ratios; chain: same-graph event means. Separate fixtures and summaries.',
                    source_mr14_commit=MR14, kvaf_commit=KVAF, completed_stages=[])
    save(output/'manifest.json', manifest)
    try:
        if not a.dry_run and any(name in os.environ.get('LD_PRELOAD', '') for name in ('camodel', 'cmodel')):
            raise RuntimeError('Physical A5 validation rejects Model preload')
        source = mr14_source(output, a.mr14_source) if a.only!='chain' else None
        kvaf = a.kvaf_root.resolve()
        if a.prepare_kvaf and not a.dry_run:
            from prepare_kvaf import prepare_and_build
            initial_tools = toolchain(a, kvaf, required=False)
            if not initial_tools['cann_root'] or not initial_tools['soc']:
                raise RuntimeError('Set --cann-root and --soc for a new dependency; an actual SoC is required')
            os.environ.update(activated_environment(initial_tools['cann_root']))
            kvaf = prepare_and_build(kvaf, a.pinned_kvaf_root, jobs=a.jobs,
                                     cann=initial_tools['cann_root'], soc=initial_tools['soc'],
                                     log=output/'dependency-build.log')
        elif a.prepare_kvaf:
            kvaf = a.pinned_kvaf_root.resolve()
            manifest['dependency_preparation'] = 'PLANNED; use local source history, preserve source checkout'
        tools = toolchain(a, kvaf, required=not a.dry_run)
        manifest.update(toolchain=tools, effective_kvaf_root=str(kvaf))
        commands = planned_commands(a, output, source, kvaf, tools)
        save(output/'commands.json', commands)
        if source:
            plan = json.loads(subprocess.check_output([sys.executable, str(source/'experiments/li_qk/run.py'),
                'plan', '--suite', 'history7', '--history', '131072', '--aics', '32', '--aivs', '64', '--details'],
                cwd=source, text=True))
            save(output/'history7-plan.json', plan)
            manifest['history_case_count_including_r0'] = 7*(a.rounds+1)
        if a.dry_run:
            if a.only != 'history7':
                chain = next(c for c in commands if c['stage']=='chain-run')
                run_logged(dict(chain, argv=chain['argv']+['--dry-run']), dict(os.environ))
            manifest.update(status='NOT_RUN', stage='planned')
            print('Plans saved; no CANN build, NPU or KVAF service execution.', flush=True)
            return output
        if git(kvaf, 'rev-parse', 'HEAD') != KVAF:
            raise RuntimeError(f'KVAF must be pinned at {KVAF}; use --prepare-kvaf for a separate dependency')
        env = activated_environment(tools['cann_root'])
        if any(name in env.get('LD_PRELOAD', '') for name in ('camodel', 'cmodel')):
            raise RuntimeError('Physical A5 validation rejects Model preload')
        for executable in ('cmake', 'c++', 'npu-smi'):
            if not shutil.which(executable, path=env.get('PATH')):
                raise RuntimeError('Required executable unavailable: '+executable)
        if a.profile and not shutil.which('msprof', path=env.get('PATH')):
            raise RuntimeError('msprof unavailable; activate the matching CANN installation')
        physical_info(output, env)
        manifest['npu_smi_log'] = str(output/'npu-smi.log')
        manifest['status'] = 'RUNNING'
        for item in commands:
            manifest['stage'] = item['stage']; save(output/'manifest.json', manifest)
            run_logged(item, env)
            manifest['completed_stages'].append(item['stage'])
        manifest.update(status='PASS', stage='complete')
        print('Completed validation. Separate summaries:', flush=True)
        if a.only != 'chain': print(output/'history7/results/summary.csv', flush=True)
        if a.only != 'history7': print(output/'chain/summary.csv', flush=True)
    except BaseException as error:
        manifest.update(status='FAILED', error=str(error)); raise
    finally:
        save(output/'manifest.json', manifest)
    return output


if __name__ == '__main__':
    main()
