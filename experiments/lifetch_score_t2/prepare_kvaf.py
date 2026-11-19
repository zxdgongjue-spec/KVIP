#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare a separate pinned URMA dependency from local Git history.

Never fetches or checks out the source repository, and never copies its build
binaries. The optional A5 build uses KVIP's existing dependency patch/build flow.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
REVISION = '1c1310387ebfb6776a07316874e924265b209ada'


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def cache_values(path):
    values = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line and not line.startswith(('#', '//')) and '=' in line:
                key, value = line.split('=', 1)
                values[key.split(':', 1)[0]] = value
    return values


def find_umdk(source):
    candidates = [os.environ.get('KVAF_UMDK_ROOT'), source/'third_party/umdk',
                  source.parent/'third_party/umdk', source.parent/'umdk']
    for folder in ('build-cpu', 'build-npu'):
        values = cache_values(source/folder/'CMakeCache.txt')
        candidates.append(values.get('KVAF_URMA_ROOT'))
    return next((Path(p).resolve() for p in candidates if p and (Path(p)/'include/urma_api.h').is_file()), None)


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise RuntimeError('The pinned dependency must be a separate sibling directory')
    head = git(source, 'rev-parse', 'HEAD')
    available = subprocess.run(['git', '-C', str(source), 'cat-file', '-e', REVISION+'^{commit}'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not available:
        raise RuntimeError(f'Local KVAF history @{head} lacks {REVISION}. On the networked host, run '
                           f'git -C {source} fetch origin {REVISION}, then rerun. No network fetch was attempted.')
    marker = destination/'.kvip-score-t2-source.json'
    if destination.exists():
        if not marker.is_file() or git(destination, 'rev-parse', 'HEAD') != REVISION:
            raise RuntimeError(f'Destination is not a managed pinned checkout: {destination}; choose a new path')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Copy Git objects rather than borrowing/hard-linking them. A later
        # update or deletion of the source checkout cannot affect this one.
        with tempfile.TemporaryDirectory(prefix='.kvaf-score-t2-', dir=destination.parent) as temporary:
            checkout = Path(temporary)/'KVAF'
            subprocess.run(['git', 'clone', '--no-hardlinks', '--no-checkout', str(source), str(checkout)], check=True)
            subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach', REVISION], check=True)
            (checkout/'.kvip-score-t2-source.json').write_text(json.dumps(
                dict(source=str(source), source_head=head, revision=REVISION), indent=2)+'\n')
            checkout.rename(destination)
    # A local header source prevents build.sh from trying to clone UMDK.
    # Its build products go into the new dependency, never the original UMDK.
    umdk = find_umdk(source)
    link = destination/'third_party/umdk'
    if umdk and not link.exists():
        link.parent.mkdir(exist_ok=True)
        link.symlink_to(umdk, target_is_directory=True)
    print(f'PINNED KVAF: {destination} @ {REVISION}', flush=True)
    print(f'SOURCE KVAF: {source} @ {head} (preserved)', flush=True)
    return destination, umdk


def prepare_and_build(source, destination, *, build=True, jobs=2, cann=None, soc=None, log=None):
    destination, umdk = prepare(source, destination)
    # Import only the dependency preparer. This modifies the new checkout and
    # applies the exact four URMA patches required by this KVIP revision.
    import importlib.util
    spec = importlib.util.spec_from_file_location('kvip_pinned_builder', ROOT/'scripts/build.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.prepare_dependency(destination, 'urma')
    if not build:
        return destination
    if not umdk and not (destination/'third_party/umdk/include/urma_api.h').is_file():
        raise RuntimeError('No local UMDK headers found. Set KVAF_UMDK_ROOT to an existing '
                           'directory containing include/urma_api.h; the original KVAF is preserved.')
    values = cache_values(Path(source)/'build-npu/CMakeCache.txt')
    cann = Path(cann or values.get('ASCEND_CANN_PACKAGE_PATH') or
                os.environ.get('ASCEND_CANN_PACKAGE_PATH') or os.environ.get('ASCEND_HOME_PATH') or
                '/usr/local/Ascend/cann').resolve()
    soc = soc or values.get('SOC_VERSION') or os.environ.get('SOC_VERSION') or 'Ascend950PR_9599'
    env = dict(os.environ, KVAF_UMDK_BUILD_DIR=str(destination/'build-umdk'))
    args = [sys.executable, str(ROOT/'scripts/build.py'), '--transport', 'urma',
            '--kvaf-root', str(destination), '--cann', str(cann), '--soc', soc, '--jobs', str(jobs)]
    print('BUILD pinned KVAF + LIDU/LIFetch with the serving Python; this can take several minutes.', flush=True)
    if log:
        print(f'Build log: {log}', flush=True)
        with Path(log).open('w') as output:
            result = subprocess.run(args, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            print(Path(log).read_text(errors='replace')[-12000:], flush=True)
            raise RuntimeError(f'Pinned dependency build exit {result.returncode}; see {log}')
    else:
        subprocess.run(args, cwd=ROOT, env=env, check=True)
    print(f'READY KVAF: {destination}', flush=True)
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--destination', type=Path, default=ROOT.parent/'KVAF-score-t2-1c131038')
    p.add_argument('--prepare-only', action='store_true', help='clone and patch without an A5 build')
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--cann', type=Path)
    p.add_argument('--soc')
    a = p.parse_args()
    if a.jobs < 1:
        p.error('jobs must be positive')
    prepare_and_build(a.source, a.destination, build=not a.prepare_only, jobs=a.jobs, cann=a.cann, soc=a.soc)


if __name__ == '__main__':
    main()
