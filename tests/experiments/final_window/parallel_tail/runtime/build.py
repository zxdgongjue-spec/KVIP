"""Build isolated prefix and private terminal libraries; never install globally."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from prepare_native import generate
from prepare_prefix import prepare_prefix


def run_logged(command, log):
    """Keep full build output and surface compiler diagnostics on failure."""
    try:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as error:
        log.flush()
        lines = Path(log.name).read_text(errors='replace').splitlines()
        # Preserve the first actual compiler errors even if the make tail is long.
        indices = set(range(max(0, len(lines)-100), len(lines)))
        for i, line in enumerate(lines):
            if 'error:' in line.lower() or 'undefined symbol' in line.lower():
                indices.update(range(max(0, i-2), min(len(lines), i+8)))
        detail = '\n'.join(lines[i] for i in sorted(indices))
        raise RuntimeError(f'Build failed: {command}\nLog: {log.name}\n{detail}') from error


def build(output, headers, cann, soc, jobs=2):
    output = Path(output).resolve()
    headers, cann = Path(headers).resolve(), Path(cann).resolve()
    if not (headers / 'kvaf_batch_get_device.h').is_file():
        raise ValueError('KVAF patched device headers not found')
    if output.exists():
        raise ValueError(f'Use a new output directory: {output}')
    output.mkdir(parents=True)
    prefix = output / 'prefix'
    prefix_source = prepare_prefix(prefix)
    terminal = generate(output / 'terminal')
    (terminal / 'kvaf-include').symlink_to(headers, target_is_directory=True)
    products = {}
    for name, source, directory, filename in (
        ('prefix', prefix_source, prefix, 'libv9_bridge.so'),
        ('terminal', terminal, terminal, 'libparallel_terminal.so'),
    ):
        build_dir = directory / 'build'
        with (directory / 'build.log').open('w') as log:
            run_logged(['cmake', '-S', str(source), '-B', str(build_dir),
                            f'-DKVAF_INCLUDE={headers}', f'-DASCEND_CANN_PACKAGE_PATH={cann}',
                            f'-DSOC_VERSION={soc}'], log=log)
            run_logged(['cmake', '--build', str(build_dir), '-j', str(jobs)],
                           log=log)
        library = build_dir / filename
        if not library.is_file():
            raise RuntimeError(f'Missing native library: {library}')
        products[name] = dict(path=str(library), sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    products['qualification'] = 'compile only; real runtime correctness and timing are not yet accepted'
    (output / 'libraries.json').write_text(json.dumps(products, indent=2) + '\n')
    return products


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--kvaf-include', required=True, type=Path)
    parser.add_argument('--cann-root', default='/usr/local/Ascend/cann', type=Path)
    parser.add_argument('--soc', default='Ascend950PR_9599')
    parser.add_argument('--jobs', default=2, type=int)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    print(json.dumps(build(args.output, args.kvaf_include, args.cann_root, args.soc, args.jobs), indent=2))
