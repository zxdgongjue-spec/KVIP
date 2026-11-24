"""Reconcile five archived runs with the operator before its history limit was raised."""
import argparse
from functools import lru_cache
from math import isclose
from pathlib import Path
import re
from statistics import mean, median
import subprocess
import sys

from cap_pair_report import CAPS, GROUPS, SDK, SDK_TREE, cap_times, inspect_cap
from main_five_report import archived_oracle, case_folder, main_times, measured_parallel, read
from peer_delta import pair_placement

PR30 = '54acdb9c54853441d08898a2623407f4492a83a4'
FEATURE = '20915a78b682378ea60889afbcd14b9a21123913'
FEATURE_SDK = 'a2857b460a74704114fd30a849ae38666cc2c372'
PACKED = 'one_profiled_graph'
INDIVIDUAL = 'one_state_individual_graphs'


def marker(root, name):
    path = root / (name + '.txt')
    return path.read_text().strip() if path.is_file() else ''


@lru_cache(maxsize=256)
def exact_revision(revision, repo):
    if revision == PR30:
        return True
    if not repo or not re.fullmatch('[0-9a-f]{40}', revision):
        return False
    # A merge commit can have a different ID and the same full Git tree.
    def tree(rev):
        p = subprocess.run(['git', '-C', str(repo), 'rev-parse', '--verify', f'{rev}^{{tree}}'],
                           capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    preceding = tree(PR30)
    return preceding is not None and tree(revision) == preceding


def find_arm(root, role, repo=None, group=10):
    if marker(root, 'physical_npu') != '4':
        return None
    # Combined run roots can advertise both main and candidate revisions.
    # Read which one actually produced the saved arm.
    choices = []
    for arm in ('mainline', 'li-window', 'cap128k', 'cap512k'):
        configs = list((root/f'g{group}'/arm/'results').glob('*/cases/diagnose/config.json'))
        if len(configs) != 1:
            continue
        cfg = read(configs[0])
        revision = cfg['production_revision']
        if revision not in {marker(root, 'main'), marker(root, 'candidate')}:
            continue
        if role == 'candidate':
            if arm == 'cap128k':
                continue
        elif role == 'feature':
            if revision != FEATURE or marker(root, 'kvaf') != FEATURE_SDK:
                continue
        elif (not exact_revision(revision, repo) or marker(root, 'kvaf') != SDK or
              marker(root, 'kvaf_tree') != SDK_TREE):
            continue
        choices.append((arm, revision))
    if len(choices) > 1:
        raise ValueError(f'{root}: ambiguous {role} arms: {choices}')
    return choices[0] if choices else None


def decomposition(profile, timing, checked):
    """Keep the actual NPU event. Spans and within-span gaps are measured separately."""
    spans = []
    gaps = []
    for sample in profile['samples']:
        acc = sample['accounting']
        li = sorted(acc['intervals_us']['li'])
        tail, = acc['intervals_us']['topk_resolve_update']
        ordered = li + [tail]
        span = tail[1] - li[0][0]
        gaps.append(sum(second[0]-first[1] for first, second in zip(ordered, ordered[1:])))
        spans.append(span)
    metrics = {key: val for key, val in checked.items() if key != 'graph'}
    metrics['kernel_span'] = mean(spans), median(spans)
    metrics['inter_kernel_gap'] = mean(gaps), median(gaps)
    event = timing['event_mean_us']
    metrics['event_outside_span_mean'] = event - mean(spans)
    metrics['event_mean'] = event
    reconciled = sum(metrics[key][0] for key in ('LI1_to_LI3', 'final_topk_through_tail', 'inter_kernel_gap'))
    if not isclose(reconciled, mean(spans), abs_tol=.025):
        raise ValueError('Profile span and stage means do not reconcile')
    return metrics


def measure_arm(root, arm, revision, group, *, verify_library=False):
    folder = case_folder(root, group, arm)
    timing = read(folder/'path_result.json')['all_rounds']
    protocol = timing['timing_protocol']
    timing_reader = {PACKED: main_times, INDIVIDUAL: cap_times}.get(protocol)
    if timing_reader is None:
        raise ValueError(f'{root}/g{group}: unsupported timing protocol {protocol}')
    receipt = read(folder/'score_topk_check.json')['native_reference']
    owner = Path(receipt['archive_path'])
    ids, old_receipt = archived_oracle(owner, group)
    archived = read(case_folder(owner, group, 'logical-ties').parent/'replay_input.json')
    cap_meta = 'capture_input.json' if (folder.parent/'capture_input.json').exists() else None
    checked, _, thp, proof, cfg = measured_parallel(root, group, arm, revision, owner, ids, old_receipt, archived,
                                                  verify_library=verify_library, timing_reader=timing_reader,
                                                  capture_meta_name=cap_meta)
    runtime = proof['runtime']
    if runtime['host'] != marker(root, 'host') or runtime['visible'] != '4':
        raise ValueError(f'{root}: archived host or device differs from the measured runtime')
    raw = read(folder/'diagnose/round-1/full_attribution.json')
    stats = decomposition(raw, timing, checked)
    replay = read(folder.parent/'replay_input.json')
    cap_state = None
    if cap_meta:
        cap = proof['history_cap']
        bound = replay['geometry']['history']
        label = next((label for label, history in CAPS.items() if history == bound), None)
        if label is None or cfg['history'] != bound:
            raise ValueError(f'G{group}: replay capacity differs from config')
        inspect_cap(label, read(folder.parent/cap_meta), replay, cap, receipt)
        cap_state = (cap['capture_page_table_sha256'], cap['capture_state_sha256'])
    comparable_cfg = {k: v for k, v in cfg.items() if k not in {
        'history', 'kvaf_server', 'output', 'trace_root', 'kvaf_root', 'production_root', 'production_revision'}}
    return dict(stats=stats, geometry=replay['geometry'], config=comparable_cfg, capture=proof['capture_operand_hashes'],
                binary=proof['library_sha256'], cap_state=cap_state, protocol=protocol, runtime=runtime, thp=thp, oracle=receipt['identity'])


def print_arm(role, group, arm):
    s = arm['stats']
    def pair(k):
        return '/'.join(f'{v:.2f}' for v in s[k])
    print(f'G{group} {role} protocol={arm["protocol"]} event_mean_us={s["event_mean"]:.2f} '
          f'four_kernel_span_mean/P50_us={pair("kernel_span")} LI_mean/P50_us={pair("LI1_to_LI3")} '
          f'tail_mean/P50_us={pair("final_topk_through_tail")} '
          f'inter_kernel_gap_mean_us={s["inter_kernel_gap"][0]:.2f} '
          f'event_minus_span_mean_us={s["event_outside_span_mean"]:.2f}')


def same_host(left_root, right_root, left, right):
    left_mid, right_mid = marker(left_root, 'machine_id'), marker(right_root, 'machine_id')
    return (left['runtime'] == right['runtime'] and marker(left_root, 'host') == marker(right_root, 'host') and
            (not left_mid or not right_mid or left_mid == right_mid))


def same_work(old, now):
    return (now['capture'] == old['capture'] and now['oracle'] == old['oracle'] and
            (old['cap_state'] is None or old['cap_state'] == now['cap_state']) and
            old['geometry']['history'] == CAPS['cap128k'] and
            now['geometry'] == {**old['geometry'], 'history': CAPS['cap512k']})


def same_controls(old_root, new_root, old, new):
    return (same_host(old_root, new_root, old, new) and old['config'] == new['config'] and
            old['thp']['bytes'] == new['thp']['bytes'] and
            pair_placement(old['thp'], new['thp'], local_baseline=True) == 'matched')


def exact_roots(area, repo):
    return sorted({path.parent for name in ('main.txt', 'candidate.txt')
                   for path in area.glob('*/'+name)
                   if marker(path.parent, 'kvaf') == SDK and exact_revision(marker(path.parent, name[:-4]), repo)})


def find_exact_group(candidate, group, repo, anchor=None):
    if (marker(candidate, 'kvaf'), marker(candidate, 'kvaf_tree')) != (SDK, SDK_TREE):
        raise ValueError('Unexpected lookup anchor KVAF source')
    # Deterministic path order prevents choosing archives for their timing.
    if anchor is None:
        arm = find_arm(candidate, 'candidate', repo, group)
        if not arm:
            raise ValueError(f'Missing candidate group {group} timing in {candidate}')
        anchor = measure_arm(candidate, *arm, group)
    for root in exact_roots(candidate.parent, repo):
        try:
            arm = find_arm(root, 'exact', repo, group)
            if not arm:
                continue
            old = measure_arm(root, *arm, group)
        except (OSError, ValueError, KeyError) as error:
            print(f'EXACT_REFERENCE_REJECTED G{group} {root}: {error}', file=sys.stderr)
            continue
        if old['protocol'] == PACKED and same_work(old, anchor) and same_controls(root, candidate, old, anchor):
            return root
    return None


def compare(role, roots, candidate, new, repo):
    deltas, gates = [], []
    for g, root in roots.items():
        arm = find_arm(root, 'feature' if role.startswith('published') else 'exact', repo, g)
        if not arm:
            raise ValueError(f'G{g}: unexpected version, SDK or device for {role}: {root}')
        old = measure_arm(root, *arm, g)
        now = new[g]
        if not same_work(old, now):
            raise ValueError(f'G{g}: captures, strong-serial oracle or physical shapes differ')
        print(f'G{g} REFERENCE role={role} archive={root} native={arm[1]} SDK={marker(root, "kvaf")}')
        print_arm(role, g, old)
        placement = pair_placement(old['thp'], now['thp'], local_baseline=True)
        runtime_match = same_host(root, candidate, old, now)
        protocol_match = old['protocol'] == now['protocol']
        comparable = (role == 'exact_PR30' and protocol_match and
                      same_controls(root, candidate, old, now))
        ns, os = now['stats'], old['stats']
        diff = (ns['event_mean']-os['event_mean'], ns['kernel_span'][0]-os['kernel_span'][0],
                ns['LI1_to_LI3'][0]-os['LI1_to_LI3'][0],
                ns['final_topk_through_tail'][0]-os['final_topk_through_tail'][0],
                ns['inter_kernel_gap'][0]-os['inter_kernel_gap'][0],
                ns['event_outside_span_mean']-os['event_outside_span_mean'])
        print(f'G{g} candidate_minus_{role} event_mean_us={diff[0]:+.2f} '
              f'four_kernel_span_mean_us={diff[1]:+.2f} LI_mean_us={diff[2]:+.2f} '
              f'tail_mean_us={diff[3]:+.2f} inter_kernel_gap_mean_us={diff[4]:+.2f} '
              f'event_minus_span_mean_us={diff[5]:+.2f} same_protocol={int(protocol_match)} '
              f'same_runtime={int(runtime_match)} placement={placement} comparable_operator_event={int(comparable)}')
        deltas.append(diff)
        gates.append(comparable)
    if len(roots) == len(GROUPS):
        ev, span, li, tail, gap, outside = map(mean, zip(*deltas))
        print(f'FIVE_GROUP_EQUAL_WEIGHT vs={role} comparable_operator_event={int(all(gates))} '
              f'candidate_minus_reference_event_mean_us={ev:+.3f} four_kernel_span_mean_us={span:+.3f} '
              f'LI_mean_us={li:+.3f} tail_mean_us={tail:+.3f} inter_kernel_gap_mean_us={gap:+.3f} '
              f'event_minus_span_mean_us={outside:+.3f}')


def run(candidate, args):
    if (marker(candidate, 'kvaf'), marker(candidate, 'kvaf_tree')) != (SDK, SDK_TREE):
        raise ValueError('Unexpected candidate KVAF source')
    new = {}
    builds = set()
    for g in GROUPS:
        arm = find_arm(candidate, 'candidate', args.repo, g)
        if not arm:
            raise ValueError(f'No completed NPU4 G{g} candidate under {candidate}')
        new[g] = measure_arm(candidate, *arm, g)
        builds.add((arm[1], new[g]['binary'], new[g]['protocol']))
        if len(builds) != 1:
            raise ValueError('Five candidate arms must share one revision, binary and timing protocol')
        if new[g]['geometry']['history'] != CAPS['cap512k']:
            raise ValueError('Candidate must bind the actual 512K capacity')
        print(f'G{g} CANDIDATE archive={candidate} native={arm[1]} SDK={SDK}')
        print_arm('candidate', g, new[g])

    if args.reference_map:
        roots = {int(g): Path(r).resolve() for g, r in read(args.reference_map).items()}
        if set(roots) != set(GROUPS):
            raise ValueError('Expected exactly G10/G4/G17/G7/G2 in reference map')
    elif args.reference:
        roots = dict.fromkeys(GROUPS, args.reference.resolve())
    else:
        roots = {}
        for g in GROUPS:
            root = find_exact_group(candidate, g, args.repo, new[g])
            if root:
                roots[g] = root
            else:
                print(f'G{g} EXACT_PR30_MISSING no_verified_matched_pre512k_unified_sdk_measurement')
    compare('exact_PR30', roots, candidate, new, args.repo)
    feature = args.feature_reference or candidate.parent/'li-borrow-four-batch.8hoXzA'
    if feature.is_dir():
        try:
            compare('published_PR30_experimental_old_KVAF', dict.fromkeys(GROUPS, feature), candidate, new, args.repo)
        except (OSError, ValueError, KeyError) as error:
            if args.feature_reference:
                raise
            print(f'PUBLISHED_FEATURE_COMPARISON_UNAVAILABLE reason={error}')
    elif args.feature_reference:
        raise ValueError(f'Explicit feature reference {feature} is unavailable')
    else:
        print('PUBLISHED_FEATURE_ARCHIVE_MISSING')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--repo', type=Path, help='Identify identical merge trees through this local KVIP checkout')
    ref = parser.add_mutually_exclusive_group()
    ref.add_argument('--reference', type=Path, help='Optional exact pre-512K PR30 archive')
    ref.add_argument('--reference-map', type=Path, help='JSON map of group number to exact PR30 root')
    ref.add_argument('--find-exact-group', type=int, choices=GROUPS, help='Print a reusable packed PR30 group archive, exit 3 if absent')
    parser.add_argument('--feature-reference', type=Path, help='Optional published old-SDK experiment archive')
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    try:
        if args.find_exact_group:
            root = find_exact_group(candidate, args.find_exact_group, args.repo)
            if not root:
                sys.exit(3)
            print(root)
        else:
            run(candidate, args)
    except (OSError, KeyError, ValueError) as error:
        sys.exit(f'ARCHIVE_AUDIT_FAILED: {error}')
