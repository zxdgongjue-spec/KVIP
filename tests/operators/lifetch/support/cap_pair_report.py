"""Measure the cost of binding the same 128K replay to two different service capacities."""
import argparse
from math import isclose, isfinite
from pathlib import Path
from statistics import mean, median

from main_five_report import archived_oracle, case_folder, measured_parallel, read
from peer_delta import pair_placement, single_numa_node
from reporting import atomic_json


GROUPS = (10, 4, 17, 7, 2)
CAPS = {'cap128k': 131328, 'cap512k': 524544}
NATIVE = 'cfe2b52f6403d5edd612c314bc7110525a5f5789'
SDK = '813f1750a50b7e6c92d5f5397f2fbc05befef6da'
SDK_TREE = '1de18e8315137d7a0227c3ae4c82afd45b5fe743'


def cap_times(folder, timing):
    """Graph statistics use per-call NPU events; only stages use profiler kernel spans."""
    profile = read(folder / 'diagnose/round-1/full_attribution.json')
    samples = profile['samples']
    if ((timing['timing_protocol'], timing['measured_graphs'], timing['graph_calls']) !=
            ('one_state_individual_graphs', 20, 20) or len(samples) != 20 or profile['graph_calls'] != 20):
        raise ValueError('Expected 20 reset-and-replay samples on one reusable state')
    events, li, tail = [], [], []
    for i, sample in enumerate(samples):
        acc = sample['accounting']
        groups = acc['intervals_us']
        if (acc['invocation'] != i or set(groups) != {'li', 'topk_resolve_update'}
                or len(groups['li']) != 3 or len(groups['topk_resolve_update']) != 1):
            raise ValueError(f'Call {i}: incomplete LI or tail profiler trace')
        spans = sorted(groups['li']) + groups['topk_resolve_update']
        if (any(not (isfinite(a) and isfinite(b) and a < b) for a, b in spans) or
                any(end > begin for (_, end), (begin, _) in zip(spans, spans[1:]))):
            raise ValueError(f'Call {i}: invalid kernel intervals')
        event = acc['event_us']
        before = sum(b-a for a, b in spans[:3])
        after = spans[-1][1] - spans[-1][0]
        if (not isfinite(event) or event <= 0 or
                not isclose(sample['us'], spans[-1][1]-spans[0][0], abs_tol=.025) or
                not isclose(acc['stages_us']['li'], before, abs_tol=.025) or
                not isclose(acc['stages_us']['topk_resolve_update'], after, abs_tol=.025)):
            raise ValueError(f'Call {i}: profiler or event timing failed its cross-check')
        events.append(event)
        li.append(before)
        tail.append(after)
    if (len(timing['samples_us']) != len(events)
            or any(not isclose(a, b, abs_tol=.025) for a, b in zip(timing['samples_us'], events))
            or not isclose(mean(events), timing['event_mean_us'], abs_tol=.025)
            or not isclose(mean(events), timing['mean_us'], abs_tol=.025)
            or not isclose(median(events), timing['p50_us'], abs_tol=.025)):
        raise ValueError('Recorded graph mean/P50 must use the same measured per-call events')
    return {key: (mean(values), median(values)) for key, values in
            (('graph', events), ('LI1_to_LI3', li), ('final_topk_through_tail', tail))}


def inspect_cap(arm, capture, actual, cap, source):
    expected = CAPS[arm]
    geometry = capture['geometry']
    if (geometry['history'] != CAPS['cap128k'] or geometry['batch'] != 16 or
            actual != {**capture, 'geometry': {**geometry, 'history': expected}}):
        raise ValueError(f'{arm}: input shape, rows, pages or source manifests changed')
    if (cap['capture_history'] != geometry['history'] or cap['bound_history'] != expected or
            cap['effective_ends'] != source['effective_native_ends'] or
            cap['query_ends_sha256'] != source['identity']['inputs']['query_ends'] or
            cap['captured_page_count'] != capture['page_count'] or cap['bound_page_count'] != capture['page_count']):
        raise ValueError(f'{arm}: effective bounds, original pages or physical capacity mismatch')
    hashed_state = {'captured_hot_valid', 'captured_lru', 'tail_bounds', 'logical_address_valid',
                    'logical_score_keys', 'logical_score_valid'}
    if set(cap['capture_state_sha256']) != hashed_state:
        raise ValueError(f'{arm}: incomplete capture state verification')
    counts = cap['measured_requests']
    for key in ('prefetch', 'final'):
        if len(counts[key]) != 20 or any(type(n) is not int or n < 0 for n in counts[key]):
            raise ValueError(f'{arm}: incomplete measured request counts')


def compare(left, right):
    return {k: tuple(b-a for a, b in zip(v, right[k])) for k, v in left.items()}


def fmt_times(data, signed=False):
    f = '+.2f' if signed else '.2f'
    return ' '.join(f'{key}_mean/P50_us={format(a, f)}/{format(b, f)}' for key, (a, b) in data.items())


def verify_root(root, revision):
    for file, value in [('main', revision), ('kvaf', SDK), ('kvaf_tree', SDK_TREE), ('physical_npu', '4')]:
        if (root / (file + '.txt')).read_text().strip() != value:
            raise ValueError(f'Unexpected {file} for capacity comparison')
    return (root/'host.txt').read_text().strip()


def read_arm(root, group, arm, revision, owner, oracle, *, verify_library):
    serial, receipt, archived = oracle
    folder = case_folder(root, group, arm)
    times, match, thp, proof, cfg = measured_parallel(
        root, group, arm, revision, owner, serial, receipt, archived,
        verify_library=verify_library, timing_reader=cap_times, capture_meta_name='capture_input.json')
    capture = read(folder.parent / 'capture_input.json')
    actual = read(folder.parent / 'replay_input.json')
    cap = proof['history_cap']
    inspect_cap(arm, capture, actual, cap, receipt['native_reference'])
    if cfg['history'] != CAPS[arm]:
        raise ValueError(f'G{group} {arm}: replay config and measured capacity disagree')
    runtime = proof['runtime']
    if runtime['host'] != (root/'host.txt').read_text().strip() or runtime['visible'] != '4':
        raise ValueError(f'G{group} {arm}: runtime used a different host or NPU')
    controls = dict(capture=capture, runtime=runtime, library=proof['library_sha256'],
                    operand_hashes=proof['capture_operand_hashes'], state_hashes=cap['capture_state_sha256'],
                    page_hash=cap['capture_page_table_sha256'], sdk_root=str(Path(cfg['kvaf_root']).resolve()),
                    cpu_registered_bytes=thp['bytes'])
    return times, thp, controls, {k: mean(v) for k, v in cap['measured_requests'].items()}, int(match.sum())


def oracle_for(owner, group):
    serial, receipt = archived_oracle(owner, group)
    archived = read(case_folder(owner, group, 'logical-ties').parent / 'replay_input.json')
    return serial, receipt, archived


def report(root, group, owner):
    host = verify_root(root, NATIVE)
    order = (root / f'g{group}/order.txt').read_text().split()
    if sorted(order) != sorted(CAPS):
        raise ValueError(f'G{group}: expected one measured block per capacity')
    oracle = oracle_for(owner, group)
    source = oracle[1]['native_reference']
    times, thps, controls, traffic, exact = {}, {}, {}, {}, {}
    for arm in CAPS:
        times[arm], thps[arm], controls[arm], traffic[arm], exact[arm] = read_arm(
            root, group, arm, NATIVE, owner, oracle, verify_library=True)
    left, right = CAPS
    changed = [key for key in controls[left] if controls[left][key] != controls[right][key]]
    if changed:
        raise ValueError(f'G{group}: arms used different controls: {changed}')
    placement = pair_placement(thps[left], thps[right], local_baseline=True)
    if placement != 'matched':
        raise ValueError(f'G{group}: unequal or unknown placement: {placement}')
    ends = source['effective_native_ends']
    delta = compare(times[left], times[right])
    result = dict(group=group, host=host, physical_npu=4, order=order, timing=times, delta=delta,
                  runtime=controls[left]['runtime'], library_sha256=controls[left]['library'],
                  kvaf_root=controls[left]['sdk_root'], exact_id_sets=exact, traffic=traffic,
                  effective_end_min=min(map(min, ends)), effective_end_max=max(map(max, ends)),
                  numa=single_numa_node(thps[left]), ub_device=thps[left]['ub_device'],
                  registered_bytes=thps[left]['bytes'])
    atomic_json(root / f'g{group}/capacity_comparison.json', result)
    print(f'G{group} CHECK both_caps TopK_score_vs_strong_serial=PASS queries_per_arm=1280/1280 '
          f'KV=PASS ordered_Index=PASS rows_per_arm=320/320 exact_ID_sets_128k/512k={exact[left]}/{exact[right]} '
          f'order={",".join(order)} bound_slots=131328/524544 effective_end_min/max={result["effective_end_min"]}/'
          f'{result["effective_end_max"]} placement=matched numa={result["numa"]} HCA={result["ub_device"]}')
    for arm in CAPS:
        print(f'G{group} {arm} {fmt_times(times[arm])} prefetch/final_tokens_per_call_mean='
              f'{traffic[arm]["prefetch"]:.2f}/{traffic[arm]["final"]:.2f}')
    print(f'G{group} cap512k_minus_cap128k {fmt_times(delta, signed=True)} '
          f'graph_mean_change_pct={100*delta["graph"][0]/times[left]["graph"][0]:+.2f}')


def summary(root):
    rows = [read(root/f'g{group}/capacity_comparison.json') for group in GROUPS]
    identity = ('host', 'physical_npu', 'runtime', 'library_sha256', 'kvaf_root')
    if (tuple(r['group'] for r in rows) != GROUPS or
            any(r[key] != rows[0][key] for key in identity for r in rows)):
        raise ValueError('The five group reports have different host, NPU or build identities')
    avg = {arm: mean(row['timing'][arm]['graph'][0] for row in rows) for arm in CAPS}
    delta = avg['cap512k'] - avg['cap128k']
    print(f'FIVE_GROUP_EQUAL_WEIGHT graph_mean_us cap128k={avg["cap128k"]:.3f} cap512k={avg["cap512k"]:.3f} '
          f'cap512k_minus_cap128k_us={delta:+.3f} graph_mean_change_pct={100*delta/avg["cap128k"]:+.2f} '
          f'groups_slower={sum(row["delta"]["graph"][0]>0 for row in rows)}/5')


def compare_active(root, archive, group, owner):
    revision = (root/'main.txt').read_text().strip()
    if revision == NATIVE or verify_root(root, revision) != verify_root(archive, NATIVE):
        raise ValueError("Expected a new candidate measured on the archive host")
    if (root/'reference.txt').read_text().strip() != str(archive.resolve()):
        raise ValueError("Candidate reference differs from the selected archive")
    oracle = oracle_for(owner, group)
    times, thps, controls, traffic, exact = {}, {}, {}, {}, {}
    phases = {}
    for label, folder, arm, rev in (('cap128k', archive, 'cap128k', NATIVE),
                                    ('cap512k', archive, 'cap512k', NATIVE),
                                    ('candidate', root, 'cap512k', revision)):
        times[label], thps[label], controls[label], traffic[label], exact[label] = read_arm(
            folder, group, arm, rev, owner, oracle, verify_library=label == 'candidate')
        attr = read(case_folder(folder, group, arm) / 'diagnose/round-1/full_attribution.json')
        durations = [[b-a for a,b in sorted(s['accounting']['intervals_us']['li'])] for s in attr['samples']]
        phases[label] = {f"LI{i}": (mean(values), median(values)) for i, values in enumerate(zip(*durations), 1)}
    if controls['cap128k']['library'] != controls['cap512k']['library']:
        raise ValueError(f"G{group}: archived capacities used different binaries")
    for label in CAPS:
        different = [key for key in controls[label] if key != 'library' and controls[label][key] != controls['candidate'][key]]
        if different:
            raise ValueError(f"G{group}: archived input or runtime differs: {different}")
        placement = pair_placement(thps[label], thps['candidate'], local_baseline=True)
        if placement != 'matched':
            raise ValueError(f"G{group}: candidate vs archived {label} placement: {placement}")
    prior = read(archive/f'g{group}/capacity_comparison.json')
    if (prior['group'] != group or prior['timing'] != {
            label: {k:list(v) for k,v in times[label].items()} for label in CAPS}):
        raise ValueError(f"G{group}: recorded capacity comparison differs from its measured files")
    proof = controls['candidate']
    atomic_json(root/f'g{group}/active_comparison.json', dict(group=group, reference=str(archive.resolve()),
        timing=times, li_phases=phases, traffic=traffic, runtime=proof['runtime'], native=revision, binary=proof['library'],
        exact_id_sets=exact))
    print(f'G{group} CHECK active_512k TopK_score_vs_strong_serial=PASS calls=20/20 queries=1280/1280 KV=PASS '
          f'ordered_Index=PASS rows=320/320 placement=matched numa={single_numa_node(thps["candidate"])} '
          f'exact_ID_sets={exact["candidate"]}/1280')
    print(f'G{group} active_512k {fmt_times(times["candidate"])} prefetch/final_tokens_per_call_mean='
          f'{traffic["candidate"]["prefetch"]:.2f}/{traffic["candidate"]["final"]:.2f}')
    for label in (*CAPS, 'candidate'):
        print(f'G{group} {label}_li_phases {fmt_times(phases[label])}')
    for label in CAPS:
        delta = compare(times[label], times['candidate'])
        print(f'G{group} candidate_minus_archived_{label} {fmt_times(delta, signed=True)} '
              f'graph_mean_change_pct={100*delta["graph"][0]/times[label]["graph"][0]:+.2f}')


def summary_active(root, archive):
    rows = [read(root/f'g{group}/active_comparison.json') for group in GROUPS]
    if tuple(r['group'] for r in rows) != GROUPS or any(
            r['reference'] != str(archive.resolve()) or
            any(r[k] != rows[0][k] for k in ('runtime', 'native', 'binary')) for r in rows):
        raise ValueError("Five groups must use the same runtime, candidate and reference")
    means = {label: mean(r['timing'][label]['graph'][0] for r in rows) for label in (*CAPS, 'candidate')}
    gap = means['candidate'] - means['cap128k']
    print(f'FIVE_GROUP_EQUAL_WEIGHT graph_mean_us archive_cap128k={means["cap128k"]:.3f} '
          f'archive_cap512k={means["cap512k"]:.3f} active_512k={means["candidate"]:.3f} '
          f'candidate_minus_archived_cap128k_us={gap:+.3f} '
          f'candidate_minus_archived_cap128k_pct={100*gap/means["cap128k"]:+.2f} '
          f'saved_vs_archived_cap512k_us={means["cap512k"]-means["candidate"]:+.3f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('group', type=int, choices=GROUPS, nargs='?')
    p.add_argument('owner', type=Path, nargs='?')
    p.add_argument('--summary', action='store_true')
    p.add_argument('--reference', type=Path, help='Reuse the measured cap pair and compare one fresh 512K candidate')
    args = p.parse_args()
    if args.summary and args.group is None and args.owner is None:
        if args.reference:
            summary_active(args.root, args.reference)
        else:
            summary(args.root)
    elif not args.summary and args.group is not None and args.owner is not None:
        if args.reference:
            compare_active(args.root, args.reference, args.group, args.owner)
        else:
            report(args.root, args.group, args.owner)
    else:
        p.error('Use ROOT GROUP OWNER or ROOT --summary')
