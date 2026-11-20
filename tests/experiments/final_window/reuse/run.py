#!/usr/bin/env python3
"""Compare unmodified RotateFetch with intermediate/final input budgets."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kvaf-include', type=Path, required=True)
    a = p.parse_args()
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    model = Path(__file__).resolve().parent
    repo = model.parents[3]
    # Symlink original sources: no substitutions, generated headers or patches.
    declarations = (repo / 'csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer.cpp').read_text().split('namespace {', 1)[0]
    (root / 'window_types.h').write_text(declarations + '\n#endif\n')
    (root / 'csrc').symlink_to(repo / 'csrc', target_is_directory=True)
    (root / 'kvaf-include').symlink_to(a.kvaf_include.resolve(), target_is_directory=True)
    for name in ('kernel.cpp', 'driver.cpp', 'CMakeLists.txt'):
        shutil.copy2(model / name, root / name)

    def run(command, log):
        with log.open('w') as f:
            subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)

    run(['cmake', '-S', str(root), '-B', str(root / 'build')], root / 'configure.log')
    run(['cmake', '--build', str(root / 'build'), '-j2'], root / 'build.log')
    results = {}
    for name, mode in (('intermediate', 0), ('final', 1), ('final_abc_unaligned', 2)):
        log = root / f'{name}.log'
        print(f'Running {name}: {log}', flush=True)
        run(['msprof', 'op', 'simulator',
             f'--application={root}/build/real_topk_sim {mode}',
             f'--output={root}/profile-{name}', '--soc-version=Ascend950PR_9599'], log)
        text = log.read_text()
        if f'mode={mode} posted=146 bad=0' not in text:
            raise RuntimeError(f'TopK/mapping/WQE mismatch: {log}')
        row = {k: int(v) for k, v in re.findall(r'(\w+_ticks)=(\d+)', text)}
        durations = re.findall(r'core0\.veccore0\s+([\d.]+)', text)
        if len(durations) != 1:
            raise RuntimeError(f'Expected one AIV timing: {log}')
        row.update(kernel_us=float(durations[0]), posted=146, correctness='PASS')
        results[name] = row
    (root / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print('Path | Kernel us | TopK ticks | Service ticks | Scan ticks | Candidate ticks | Post ticks')
    for name, r in results.items():
        print(f"{name} | {r['kernel_us']:.2f} | " + ' | '.join(str(r[k]) for k in
              ('topk_ticks', 'service_ticks', 'scan_ticks', 'candidate_ticks', 'post_ticks')))
    print('Scope: TopK + request submission only; no CQ completion or public output replay.')


if __name__ == '__main__':
    main()
