# SPDX-License-Identifier: Apache-2.0
"""Compare ten-call packed device events with the published PR30 blocks, then gate 45 cells."""
import argparse
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import mean
import sys

# B20 / NPU4 published paced arm, two event-per-call means per trace (microseconds, rounded to .01 here).
# The original archive supplies the unrounded values for the acceptance decision.
PUBLISHED_EVENT_PER_CALL = {10: (298.59, 310.30), 4: (310.18, 308.84),
                            17: (333.97, 334.96), 7: (444.03, 430.16), 2: (287.97, 286.43)}
BATCHES = (1, 2, 4, 8, 16, 20, 24, 28, 32)
PACED = 'b261b4885edd79dabac822a2e6e3790c7f226ad6'


def read(path):
    return json.loads(path.read_text())


def packed_block(root, group, arm, revision, owner, check_library):
    from cap_pair_report import oracle_for
    from main_five_report import case_folder, measured_parallel
    from pr30_feature_steps import feature_times, receipt_summary

    folder = case_folder(root, group, arm)
    times, _, thp, proof, config = measured_parallel(root, group, arm, revision, owner, *oracle_for(owner, group),
                                                      verify_library=check_library, expected_calls=10)
    if not proof.get('independent_buffers') or set(proof['independent_buffers'].values()) != {10}:
        raise ValueError(f'{folder}: graph does not contain ten independent states')
    receipts = read(folder / 'ablation_receipts.json')
    if receipts['policy'] != 'paced' or receipts['independent_status_allocations'] != 10:
        raise ValueError(f'{folder}: invalid independent queue policy')
    for phase in ('measured', 'detail'):
        receipt_summary(receipts[phase], paced=True)
    detail = feature_times(read(folder / 'diagnose/round-1/full_attribution.json'), times)
    event = detail['event_mean_us']
    if not isfinite(event) or event <= 0:
        raise ValueError(f'{folder}: invalid packed event')
    varying = {'production_root', 'production_revision', 'output', 'trace_root', 'kvaf_server', 'kvaf_root'}
    controls = {key: value for key, value in config.items() if key not in varying}
    snapshot = read(folder.parent/'replay_input.json')
    signature = (proof['runtime'], proof['capture_operand_hashes'], controls, thp['bytes'])
    return event, thp, signature, snapshot


def evaluate_packed(out, group, archive):
    from feature_host_reference import same_local_device
    from peer_delta import pair_placement

    def local_field(name):
        path = out / f'{name}.txt'
        return path.read_text().strip() if path.is_file() else ''

    revision = local_field('native')
    matched = archive.is_dir() and same_local_device(archive, (local_field('host'), local_field('machine_id'),
                                              Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    comparable_archive = matched
    if comparable_archive:
        if (local_field('kvaf') != (archive/'kvaf.txt').read_text().strip() or
                local_field('kvaf_tree') != (archive/'kvaf_tree.txt').read_text().strip()):
            raise ValueError('Published PR30 and candidate used different KVAF sources')
        if ((archive/'paced.txt').read_text().strip() != PACED or
                (archive/'kvaf_tree.txt').read_text().strip() != '1de18e8315137d7a0227c3ae4c82afd45b5fe743'):
            raise ValueError(f'Unexpected PR30 build or SDK in {archive}')
    actual, published = [], []
    for i in (1, 2):
        block = out/'pr30'/f'block{i}'
        owner = Path((block/f'g{group}/candidate/oracle.txt').read_text().strip())
        event, placement, controls, snapshot = packed_block(block, group, 'candidate', revision, owner, True)
        actual.append(event)
        expected = PUBLISHED_EVENT_PER_CALL[group][i-1]
        if comparable_archive:
            from li_restore_archive import check_snapshot
            from ranked_integration_sets import INPUT_AND_RESULTS
            from main_five_report import case_folder
            previous = archive/f'block{i}'
            folder = case_folder(previous, group, 'paced')
            prior_owner = Path(read(folder/'score_topk_check.json')['native_reference']['archive_path'])
            ref, old_placement, old_controls, prior_snapshot = packed_block(previous, group, 'paced', PACED, prior_owner, False)
            if not isclose(ref, expected, abs_tol=.006):
                raise ValueError(f'{archive}: G{group} block {i} differs from the published event')
            if snapshot != prior_snapshot:
                check_snapshot(group, INPUT_AND_RESULTS[group][0], prior_snapshot, snapshot)
            if controls != old_controls or pair_placement(placement, old_placement, local_baseline=True) != 'matched':
                matched = False
            published.append(ref)
        else:
            published.append(expected)
    diff = mean(published) - mean(actual)
    return dict(group=group, new_packed_us=actual, published_packed_us=published, saved_us=diff,
                context_matched=bool(matched), passed=bool(matched and diff >= 0))


def combined_pass(cells, packed):
    wanted = {(g, b) for g in PUBLISHED_EVENT_PER_CALL for b in BATCHES}
    observed = {(c['group'], c['batch']) for c in cells}
    groups = set(PUBLISHED_EVENT_PER_CALL)
    if (observed != wanted or len(cells) != len(wanted)
            or {p['group'] for p in packed} != groups or len(packed) != len(groups)):
        raise ValueError('Incomplete full-matrix verification')
    serial = 0
    for cell in cells:
        baseline, fused = cell['serial_graph_us'], cell['fused_graph_us']
        if not all(isfinite(v) and v > 0 for v in (baseline, fused)):
            raise ValueError('Invalid single-event graph time')
        serial += fused <= 0.8 * baseline
    parity = sum(row['context_matched'] and mean(row['new_packed_us']) <= mean(row['published_packed_us'])
                 for row in packed)
    return serial, parity


def main(out):
    summary = read(out/'fixed_q4_summary.json')
    support = Path((out/'pr30/reader_path.txt').read_text().strip())
    sys.path.insert(0, str(support))
    archive = Path((out/'pr30/published_archive.txt').read_text().strip())
    packed = []
    for group in PUBLISHED_EVENT_PER_CALL:
        row = evaluate_packed(out, group, archive)
        packed.append(row)
        print(f'PR30 G{group} BS16 packed_graph_mean_us={mean(row["new_packed_us"]):.2f} '
              f'published_mean_us={mean(row["published_packed_us"]):.2f} saved_us={row["saved_us"]:+.2f} '
              f'context={"matched" if row["context_matched"] else "unverified"} '
              f'gate={"PASS" if row["passed"] else "NOT_MET"}')
    serial, parity = combined_pass(summary['cells'], packed)
    ok = (serial, parity) == (45, 5)
    payload = dict(packed=packed, serial_cells_passed=serial, pr30_cells_passed=parity, passed=ok)
    (out/'fixed_q4_acceptance.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(f'ACCEPTANCE serial20={serial}/45 BS16_PR30={parity}/5 goal={"MET" if ok else "NOT_MET"} RESULT_ROOT={out}')
    return 0 if ok else 3


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    sys.exit(main(parser.parse_args().result.resolve()))
