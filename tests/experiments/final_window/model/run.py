#!/usr/bin/env python3
"""CANN simulator comparison: identical four-Q scores, candidates and LRU state."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command(args, log):
    with log.open('w') as stream:
        subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--kvaf-include', required=True, type=Path)
    parser.add_argument('--baseline-ref', default='c2c0a7b')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = Path(__file__).resolve().parent
    repo = model.parents[3]
    old = output / 'baseline-tools'
    old.mkdir()
    for name in ('prepare.py', 'terminal.h'):
        text = subprocess.check_output(['git', 'show',
            f'{args.baseline_ref}:tests/experiments/final_window/{name}'], cwd=repo, text=True)
        (old / name).write_text(text)
    results = {}
    for name, script, modes in (
        ('current_final', old / 'prepare.py', (1,)),
        ('optimized', model.parent / 'prepare.py', (0, 1)),
    ):
        source = output / name
        load(script, name).prepare(repo, source)
        for file in ('kernel.cpp', 'driver.cpp', 'CMakeLists.txt'):
            shutil.copy2(model / file, source / file)
        (source / 'kvaf-include').symlink_to(args.kvaf_include.resolve(), target_is_directory=True)
        command(['cmake', '-S', str(source), '-B', str(source / 'build')], source / 'configure.log')
        command(['cmake', '--build', str(source / 'build'), '-j2'], source / 'build.log')
        for mode in modes:
            case = 'middle' if mode == 0 else name
            log = source / f'run-{mode}.log'
            print(f'Running {case}; log={log}', flush=True)
            command(['msprof', 'op', 'simulator',
                f'--application={source}/build/real_topk_sim {mode}',
                f'--output={source}/profile-{mode}', '--soc-version=Ascend950PR_9599'], log)
            text = log.read_text()
            if f'mode={mode} posted=146 bad=0' not in text:
                raise RuntimeError(f'{case}: output/WQE verification failed; {log}')
            cores = [(core, float(duration), float(running)) for core, duration, running in
                re.findall(r'(core\d+\.veccore\d+)\s+([\d.]+)\s+([\d.]+)', text)]
            if len(cores) != (1 if mode == 0 else 4):
                raise RuntimeError(f'{case}: missing simulator core timing; {log}')
            results[case] = {'cores': cores, 'max_core_us': max(c[1] for c in cores),
                             'posted': 146, 'correctness': 'PASS', 'log': str(log)}
    (output / 'summary.json').write_text(json.dumps(results, indent=2))
    print('Path | simulator max-core us | Check')
    for name, row in results.items():
        print(f"{name} | {row['max_core_us']:.2f} | {row['correctness']}")
    print(f'Results: {output}')


if __name__ == '__main__':
    main()
