"""Build an isolated baseline bridge with an additive three-stage entry."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'csrc/operators/common/lifetch').is_dir())


def prepare_prefix(destination):
    if destination.exists():
        raise ValueError(f'Prefix destination already exists: {destination}')
    target = destination / 'csrc/operators/lightning_indexer_with_fetch'
    source = ROOT / 'csrc/operators/lightning_indexer_with_fetch'
    shutil.copytree(source, target)
    shutil.copytree(ROOT / 'csrc/operators/common', destination / 'csrc/operators/common')
    shutil.copytree(ROOT / 'csrc/operators/lidu', destination / 'csrc/operators/lidu')
    launch = target / 'op_host/launch.cpp'
    launch.write_text(launch.read_text() + '\n' + (HERE / 'launch_prefix.h').read_text())
    hashes = {}
    for relative_root in ('op_kernel', 'include'):
        for original in sorted((source / relative_root).rglob('*')):
            if not original.is_file():
                continue
            relative = original.relative_to(source)
            if original.read_bytes() != (target / relative).read_bytes():
                raise AssertionError(f'Producer source was changed: {relative}')
            hashes[str(relative)] = hashlib.sha256(original.read_bytes()).hexdigest()
    for original in sorted((ROOT / 'csrc/operators/common').rglob('*')):
        if original.is_file():
            relative = original.relative_to(ROOT / 'csrc/operators/common')
            copied = destination / 'csrc/operators/common' / relative
            if original.read_bytes() != copied.read_bytes():
                raise AssertionError(f'Shared producer source was changed: {relative}')
            hashes['../common/' + str(relative)] = hashlib.sha256(original.read_bytes()).hexdigest()
    (destination / 'producer_sources.json').write_text(json.dumps(hashes, indent=2) + '\n')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    print(prepare_prefix(parser.parse_args().output.resolve()))
