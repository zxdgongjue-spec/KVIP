"""Recreate archived simulator candidates; this does not modify production code."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def prepare(version: str, output: Path) -> None:
    here = Path(__file__).resolve().parent
    evidence = here / 'evidence' / version
    hash_file = evidence / ('measured_sources.json' if version == 'v346' else 'hashes.json')
    expected = json.loads(hash_file.read_text())
    names = ['kernel.cpp', 'driver.cpp', 'parallel_post.h', 'emit.h']
    names += [name for name in ('helpers.h', 'shadow_exchange_parallel.h')
              if name in expected]
    names += [name for name in ('prefetch.h', 'CMakeLists.txt') if name in expected]
    for name in names:
        actual = hashlib.sha256((evidence / name).read_bytes()).hexdigest()
        if actual != expected[name]:
            raise ValueError(f'Archived source checksum mismatch: {version}/{name}')
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    subprocess.run([sys.executable, str(here / 'prepare.py'), str(output)], check=True)
    kernel_dir = output / 'csrc/operators/lightning_indexer_with_fetch/op_kernel'
    for name in names:
        target = kernel_dir / name if name in ('shadow_exchange_parallel.h', 'prefetch.h') else output / name
        shutil.copy2(evidence / name, target)
    manifest_path = output / 'source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    # Regenerate every existing digest after overlays, including generated headers.
    for name in list(manifest['sha256']):
        manifest['sha256'][name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    shadow = kernel_dir / 'shadow_exchange_parallel.h'
    manifest['sha256'][str(shadow.relative_to(output))] = hashlib.sha256(shadow.read_bytes()).hexdigest()
    manifest['candidate'] = version
    manifest['scope'] = 'isolated simulator; constructed scores; fake SQ/doorbell/CQ; not production acceptance'
    manifest['archived_inputs'] = '1024 unique misses' if version in ('v492', 'v502') else '146 unique misses'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Prepared {version}: {output}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('version', choices=('v346', 'v492', 'v494', 'v502', 'v522', 'v534', 'v537'))
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    prepare(args.version, args.output.resolve())


if __name__ == '__main__':
    main()
