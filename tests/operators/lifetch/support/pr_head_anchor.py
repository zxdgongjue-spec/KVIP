# SPDX-License-Identifier: Apache-2.0
"""Locate PR30's recorded oracle and CPU placement for a same-card 128K replay."""
import argparse
from pathlib import Path
from statistics import mean
import sys

from feature_host_reference import INVALID, owner_data, same_local_device, usable_placement
from main_five_report import case_folder, read
from peer_delta import pair_placement
from pr30_archive_compare import marker
from pr30_feature_steps import FRESH_CALLS, GROUPS, check_roots


def anchor(archive, current, group):
    identity = tuple(marker(current, name) for name in ('host', 'machine_id', 'boot_id'))
    owners, placements, events = [], [], []
    for i in (1, 2):
        block = archive / f'block{i}'
        check_roots(block, None)
        if marker(block, 'live_main') != '1' or not same_local_device(block, identity):
            raise ValueError('PR30 archive belongs to a different machine or replay protocol')
        folder = case_folder(block, group, 'paced')
        check = read(folder / 'score_topk_check.json')
        owners.append(Path(check['native_reference']['archive_path']).resolve(strict=True))
        placement = block / f'g{group}/paced/thp.json'
        if not usable_placement(placement):
            raise ValueError(f'Incomplete placement receipt for {block.name}')
        placements.append(placement)
        record = read(block / f'g{group}/feature_steps.json')
        if record['group'] != group or not record['live_main'] or record['calls_per_graph']['paced'] != FRESH_CALLS:
            raise ValueError(f'Unexpected PR30 block for G{group}')
        events.append(record['timing']['paced']['event_mean_us'])
    if owners[0] != owners[1] or pair_placement(*(read(p) for p in placements), local_baseline=True) != 'matched':
        raise ValueError('The published blocks used different oracles or placement')
    owner_data(owners[0], group)
    print(f'G{group} published_pr30_event_per_call_us={mean(events):.2f} source={archive}', file=sys.stderr)
    return owners[0], placements[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('current', type=Path)
    parser.add_argument('group', type=int, choices=GROUPS)
    args = parser.parse_args()
    try:
        owner, placement = anchor(args.archive.resolve(strict=True), args.current, args.group)
    except INVALID as error:
        print(f'G{args.group} previous PR30 placement or oracle unavailable: {error}', file=sys.stderr)
        sys.exit(3)
    print(owner)
    print(placement)
