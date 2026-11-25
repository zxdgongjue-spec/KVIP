# SPDX-License-Identifier: Apache-2.0
"""Reanalyze both completed arms of an earlier NPU4 feature block from raw samples."""
import argparse
import os
from pathlib import Path
import sys
from zipfile import BadZipFile


def active_writers(root, proc=Path('/proc')):
    """The launcher keeps tee alive between workers; an interrupted worker may outlive it."""
    if not (proc / 'self/fd').is_dir():
        raise OSError(f'Cannot inspect running replay processes under {proc}')
    found = []
    for entry in proc.iterdir():
        if not entry.name.isdecimal() or int(entry.name) == os.getpid():
            continue
        try:
            args = [os.fsdecode(a) for a in (entry/'cmdline').read_bytes().split(b'\0') if a]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if not args:
            continue
        name = Path(args[0]).name
        matching = name == 'tee' and str(root/'run.log') in args[1:]
        if not matching and '--output' in args and any(a.endswith('/main_five_run.py') for a in args):
            i = args.index('--output') + 1
            matching = i < len(args) and root in Path(args[i]).parents
        if not matching:
            for fd in ('1', '2'):
                try:
                    dest = Path(os.readlink(entry/'fd'/fd))
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    continue
                if root in dest.parents and dest.name in ('worker.log', 'cpu.log', 'thp.log'):
                    matching = True
                    break
        if matching:
            found.append((int(entry.name), name))
    return sorted(found)


def reuse_block(previous, block, group, owner):
    from feature_host_reference import same_local_device
    from main_five_report import case_folder
    from pr30_archive_compare import marker
    from pr30_feature_steps import check_roots, group_reference, report
    from reporting import atomic_json

    previous, block, owner = (Path(p).resolve(strict=True) for p in (previous, block, owner))
    if block.name not in ('block1', 'block2') or block.parent == previous:
        raise ValueError('Expected a new block directory separate from the source run')
    target_group = block/f'g{group}'
    original_block = previous/block.name
    original_group = original_block/f'g{group}'
    reference = group_reference(block, group)
    if not same_local_device(previous, tuple(marker(block, key) for key in ('host', 'machine_id', 'boot_id'))):
        raise ValueError(f'Prior run is on a different or unidentified host/card: {previous}')
    if not original_block.is_dir() or not (original_group/'order.txt').is_file():
        return False
    check_roots(original_block, group_reference(original_block, group))
    if (reference != group_reference(original_block, group) or
            (original_group/'order.txt').read_text().split() != (target_group/'order.txt').read_text().split()):
        print(f'G{group} {block.name} prior reference or order differs; measuring the pair again')
        return False
    reader = marker(original_block, 'reader')
    if len(reader) != 40 or reader != marker(previous, 'reader'):
        raise ValueError('Prior block has inconsistent reader provenance')

    # The parent worker saves the score check and finishes path_result after
    # the device worker exits. A graph_event or an arm directory alone is insufficient.
    arms = ('dualcut', 'paced')
    try:
        for arm in arms:
            folder = case_folder(original_block, group, arm)
            if not (folder/'score_topk_check.json').is_file():
                return False
    except ValueError:
        return False
    if (original_group/arms[0]).resolve().parent != (original_group/arms[1]).resolve().parent:
        raise ValueError('Source arms come from different original measurement blocks')
    owned = []
    validated = False
    summary = target_group/'feature_steps.json'
    provenance = target_group/'reused_pair.json'
    if any(p.exists() or p.is_symlink() for p in (summary, provenance, *(target_group/a for a in arms))):
        raise ValueError(f'Refusing to overwrite existing arm evidence in {target_group}')
    try:
        for arm in arms:
            link = target_group/arm
            link.symlink_to((original_group/arm).resolve(strict=True), target_is_directory=True)
            owned.append(link)
        try:
            # report independently checks raw calls, correctness, loaded images,
            # transport placement and the original comparison oracle before publishing.
            report(block, group, owner)
        except (OSError, ValueError, KeyError, IndexError, TypeError, EOFError, BadZipFile) as error:
            print(f'G{group} {block.name} prior pair could not be revalidated: {error}', file=sys.stderr)
            return False
        atomic_json(provenance, dict(source=str(previous), group=group, block=block.name,
                                     reader=reader, arms={a: str((original_group/a).resolve()) for a in arms}))
        validated = True
    finally:
        if not validated:
            for link in owned:
                link.unlink()
            summary.unlink(missing_ok=True)
    print(f'G{group} REUSED_PAIR block={block.name} source={original_group}')
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--active', action='store_true', help='Check whether the prior run still owns a writer')
    parser.add_argument('previous', type=Path)
    parser.add_argument('block', nargs='?', type=Path)
    parser.add_argument('group', nargs='?', type=int, choices=(10, 4, 17, 7, 2))
    parser.add_argument('owner', nargs='?', type=Path)
    args = parser.parse_args(argv)
    root = args.previous.resolve(strict=True)
    if not root.is_dir():
        parser.error('previous must be a replay result directory')
    if args.active and (args.block, args.group, args.owner) == (None, None, None):
        writers = active_writers(root)
        if writers:
            print('PRIOR_REPLAY_ACTIVE ' + ' '.join(f'{pid}/{name}' for pid, name in writers) + f' root={root}')
            return 0
        return 3
    if not args.active and None not in (args.block, args.group, args.owner):
        return 0 if reuse_block(root, args.block, args.group, args.owner) else 3
    parser.error('use --active PRIOR_ROOT or PRIOR_ROOT NEW_BLOCK GROUP OWNER')


if __name__ == '__main__':
    sys.exit(main())
