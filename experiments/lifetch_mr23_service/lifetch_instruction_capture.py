# SPDX-License-Identifier: Apache-2.0
"""Run a checked live P2 case under msOpProf application replay."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from lifetch_msprof_timeline import PHASES


STAGES = {stage.lower().replace('_', '-'): kernel for stage, kernel, _ in PHASES}
PIPES = {'mte1', 'mte2', 'mte3', 'vector', 'cube', 'fixp'}
TARGET_SOC = 'Ascend950PR_9599'


def instruction_plan(stage, pipes):
    selected = pipes.split('|')
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(PIPES):
        raise ValueError('Choose distinct instruction pipes separated by | from ' + ', '.join(sorted(PIPES)))
    kernels = list(STAGES.values()) if stage == 'all' else [STAGES[stage]]
    # v9_li_kernel is a prefix for Hot, Threshold and Cold on the stock route.
    selector = 'v9_li_kernel|v9_parallel_terminal' if stage == 'all' else kernels[0]
    return dict(
        kernels=kernels, selector=selector, launch_count=len(kernels), pipes=pipes,
        metrics='Default,InstrTimeline', replay_mode='application', tool_warmup=0,
    )


def collector_options(plan, output):
    return [
        '--output=' + str(output), '--aic-metrics=' + plan['metrics'],
        '--instr-timeline-pipe=' + plan['pipes'], '--kernel-name=' + plan['selector'],
        '--launch-count=' + str(plan['launch_count']), '--replay-mode=' + plan['replay_mode'],
        '--warm-up=' + str(plan['tool_warmup']),
    ]


def checked_collector(profiler, log):
    standalone = os.environ.get('KVIP_MSOPPROF')
    if standalone:
        program = shutil.which(standalone)
        if program is None:
            raise FileNotFoundError(f'Cannot run KVIP_MSOPPROF={standalone}')
        command = [program]
    else:
        command = [profiler, 'op']
    result = subprocess.run([*command, '--help'], capture_output=True, text=True, timeout=20, check=False)
    help_text = result.stdout + result.stderr
    log.write_text(help_text)
    required = ('--instr-timeline-pipe', '--replay-mode', 'application', '--warm-up')
    metric = re.search(r'\bInstrTimeline\b', help_text, flags=re.IGNORECASE)
    if not metric or not all(option in help_text.lower() for option in required):
        raise RuntimeError('The installed msOpProf does not advertise InstrTimeline with application replay. '
                           f'See {log}. A compatible standalone msopprof can be selected with KVIP_MSOPPROF.')
    return command, metric.group()


def collect_instructions(profiler, output, kvaf, config, plan, fixture_sha256):
    from execution import supervise, worker_environment
    from kvip.runtime.build import build_libraries, kvaf_build_paths
    from reporting import atomic_json

    output.mkdir()
    status_file = output / 'instruction_status.json'
    status = dict(requested=plan, status='RUNNING')
    try:
        collector, metric = checked_collector(profiler, output / 'msopprof-help.log')
        build = kvaf / 'build-npu'
        cann, headers = kvaf_build_paths(build)
        (library,) = build_libraries(cache_dir=build / 'lifetch', cann=cann, headers=headers, soc=TARGET_SOC)
        config.msprof_instruction_replay = True
        config.msprof_instruction_library = str(library)
        config.warmup = 0
        config_file = output / 'replay_template.json'
        atomic_json(config_file, {key: str(value) if isinstance(value, Path) else value
                                 for key, value in vars(config).items()})
        raw = output / 'raw'
        raw.mkdir()
        # Run the bridge in the tool's own application process. A nested
        # Python subprocess may escape the operator collector's hooks.
        collector_plan = {**plan, 'metrics': 'Default,' + metric}
        command = [*collector, *collector_options(collector_plan, raw), sys.executable, '-B', '-u',
                   str(Path(__file__).resolve()), str(config_file)]
        status.update(command=command, library=str(library))
        atomic_json(status_file, status)
        rc = supervise(command, output, timeout_seconds=config.timeout_seconds,
                       environment=worker_environment(kvaf), show_case_details=False)
        status['worker_exit_code'] = rc
        if rc:
            raise RuntimeError(f'Application replay incomplete. See {output / "summary.log"}')
        case_name = f'case-b{config.batches[0]}'
        miss_dir = f'miss{config.miss_percent:g}'
        attempts = sorted((output / 'attempts').glob('replay-*'))
        reports = []
        for attempt in attempts:
            result = json.loads((attempt / 'worker-result.json').read_text())
            report = attempt / miss_dir / case_name / 'instruction_replay_report.json'
            details = json.loads(report.read_text())
            launches = [(row['kernel_entry'], row['launches']) for row in details.get('kernel_launch_plan', [])]
            if (result.get('exit_code') != 0 or details.get('correctness') != 'PASS'
                    or details.get('invocations') != 1 or details.get('source_trace_enabled') is not False
                    or details.get('inner_profiler_enabled') is not False
                    or details.get('fixture_sha256') != fixture_sha256
                    or launches != [(kernel, 1) for kernel in STAGES.values()]):
                raise ValueError(f'Instruction replay failed correctness or invocation checks: {attempt}')
            reports.append(str(report))
        files = [path for path in sorted(raw.glob('OPPROF_*/**/visualize_data.bin'))
                 if path.is_file() and path.stat().st_size > 0]
        if not reports or not files:
            raise ValueError(f'No checked worker report or nonempty native operator artifact under {output}')
        status.update(status='ARTIFACTS_CAPTURED', checked_replay_reports=reports,
                      visualize_data=[str(path) for path in files],
                      named_artifact_kernels=sorted({kernel for kernel in plan['kernels']
                                                    if any(kernel in path.relative_to(raw).parts for path in files)}),
                      instruction_track_verified=False)
        print(f'Native instruction artifacts: {status_file}', flush=True)
        for path in files:
            print(f'Open in MindStudio Insight: {path}', flush=True)
        return status
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        status.update(status='FAILED', error=str(error))
        raise
    finally:
        atomic_json(status_file, status)


def replay_once(template):
    """Create fresh KVAF and report state for every msOpProf application pass."""
    repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(repo / 'python'), str(repo / 'tests/operators/lifetch/support')]
    from execution import run_config
    from reporting import atomic_json

    root = template.parent
    attempts = root / 'attempts'
    attempts.mkdir(exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix='replay-', dir=attempts))
    marker = root / 'worker-result.json'
    atomic_json(marker, dict(exit_code=None, attempt=str(attempt)))
    config_file = attempt / 'config.json'
    config_file.write_bytes(template.read_bytes())
    code = run_config(config_file)
    atomic_json(marker, dict(exit_code=code, attempt=str(attempt)))
    return code


if __name__ == '__main__':
    sys.exit(replay_once(Path(sys.argv[1]).resolve()))
