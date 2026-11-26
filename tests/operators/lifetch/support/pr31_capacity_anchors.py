# SPDX-License-Identifier: Apache-2.0
"""Load the completed 128K PR31 controls before starting the two 512K replays."""
import argparse
from pathlib import Path
from shutil import copyfile

from feature_host_reference import current_runtime, same_local_device, usable_placement
from main_five_report import case_folder, read
from pr30_archive_compare import marker
from pr30_feature_steps import FRESH_CALLS, GROUPS
from pr_head_parity_report import REVISIONS, identity


def prepare(archive, out):
    saved = identity(archive)
    machine = tuple(marker(out, name) for name in ('host', 'machine_id', 'boot_id'))
    if not same_local_device(archive, machine) or marker(out, 'main') != saved['pr31']:
        raise ValueError('The archived PR31 measurements must match this machine, physical NPU4 and PR31 source')
    if any(marker(out, name) != saved[name] for name in ('kvaf', 'kvaf_tree', 'physical_npu')):
        raise ValueError('The SDK, patch tree or physical device differs from the 128K archive')
    summary = read(archive / 'pr_head_parity.json')
    if summary['identity'] != saved or set(summary['groups']) != {str(g) for g in GROUPS}:
        raise ValueError('Expected a completed five-group narrow PR head comparison')
    runtime = current_runtime(machine[0])
    for block in ('block1', 'block2'):
        dest = out / block
        dest.mkdir()
        for name in ('host', 'machine_id', 'boot_id', 'main', 'reader', 'kvaf', 'kvaf_tree', 'physical_npu', 'reference'):
            field = out / (name + '.txt')
            if field.exists():
                copyfile(field, dest / field.name)
        for group in GROUPS:
            parent = archive / block
            receipt = read(parent / f'g{group}/pr_head_parity.json')
            folder = case_folder(parent, group, 'pr31')
            proof = read(folder / 'mainline_checks.json')
            owner = Path(read(folder / 'score_topk_check.json')['native_reference']['archive_path']).resolve(strict=True)
            if (receipt['group'] != group or receipt['identity'] != saved
                    or receipt['controls']['runtime'] != runtime or proof['runtime'] != runtime
                    or proof['measured_calls'] != FRESH_CALLS or proof['production_revision'] != REVISIONS['pr31']
                    or not usable_placement(parent / f'g{group}/pr31/thp.json')):
                raise ValueError(f'{block} G{group}: runtime, source, device placement or ten-call measurement differs')
            directory = dest / f'g{group}'
            directory.mkdir()
            (directory / 'owner.txt').write_text(str(owner) + '\n')
            (directory / 'narrow_proof.txt').write_text(str(folder / 'mainline_checks.json') + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    prepare(args.archive.resolve(strict=True), args.out.resolve(strict=True))
