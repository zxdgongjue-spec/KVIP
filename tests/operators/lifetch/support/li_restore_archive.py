"""Compare a saved 6bca fixed arm with the independent logical-order native LI."""
import hashlib
import json
import re
from math import isclose, isfinite
from pathlib import Path
from statistics import mean, median
from zipfile import BadZipFile

import numpy as np

from correctness import topk_fingerprint
from peer_delta import read
from ranked_auto_report import FIXED, load

BASELINE_RUN = 'help-96211844-fcb8095d52d34b1c62be'
# Host20/NPU4 fused.b8TJ3b read these archive values directly from host20/NPU0
# help-96211844-fcb8095d52d34b1c62be. Each pair is a measured mean and P50.
PUBLISHED = {7: ((286.20, 285.87), (25.53, 25.60)), 10: ((265.90, 265.50), (25.51, 25.58)),
             17: ((268.91, 268.77), (26.01, 25.98)), 4: ((267.45, 267.62), (25.74, 25.69))}
SHA = re.compile(r'[a-f0-9]{64}')


def published(group, reason, li_pair, li4_pair):
    li, final = PUBLISHED[group]
    return [f'G{group} 6bca602a_archive reason={reason} vs_strong_serial_set=UNVERIFIED '
            f'published_host20_NPU0 historical_unpaired LI_mean/P50_us={pair(li)} LI4_mean/P50_us={pair(final)} '
            f'LI_saved_mean/P50_us={difference(li, li_pair)} LI4_saved_mean/P50_us={difference(final, li4_pair)}']


def dp_capture_id(group, meta):
    """Reproduce the input.sha256 that peer_g2_input.sh saved for the fixed run."""
    paths = list(map(Path, meta['source_paths']))
    if len(paths) != 8 or len(set(paths)) != 8:
        return None
    archive_hashes = []
    for path in paths:
        archive_hash = read(path)['archive']['sha256']
        if not SHA.fullmatch(archive_hash):
            raise ValueError(f'invalid archive SHA256 in {path}')
        archive_hashes.append(archive_hash)
    source_round = 51 if group == 10 else 2
    record = [group, source_round, meta['identities'], archive_hashes]
    return hashlib.sha256(json.dumps(record, separators=(',', ':')).encode()).hexdigest()


def normalized_inputs(group, meta):
    """Compare verified payloads if the current replay uses the merged G10 export."""
    from workloads import ReplaySequence

    paths = [Path(s).resolve() for s in meta['source_paths']]
    if len(paths) == 8 and len(set(paths)) == 1:
        capture = paths[0].parent
    elif len(paths) == 8 and len(set(paths)) == 8:
        capture = paths[0].parents[3]
    else:
        raise ValueError('expected eight DP references')
    sequence = ReplaySequence(capture, group, (1, 1))
    if [r.manifest_path.resolve() for r in sequence.rounds[0]] != paths:
        raise ValueError('source references differ from the resolved capture')
    row = next(iter(sequence))
    if list(map(list, row.identities)) != meta['identities'] or row.geometry != meta['geometry']:
        raise ValueError('request identities or geometry in capture differ')
    digest = hashlib.sha256()
    for key in ('logical_keys', 'logical_scales', 'query', 'weights', 'query_scale',
                'query_ends', 'tail_bounds', 'captured_layout', 'captured_hot_valid', 'captured_lru',
                'logical_address_valid', 'logical_score_keys', 'logical_score_valid'):
        arr = np.ascontiguousarray(getattr(row, key))
        digest.update(f'{key}/{arr.shape}/{arr.dtype.str}/'.encode())
        digest.update(memoryview(arr).cast('B'))
    return digest.digest()


def check_snapshot(group, recorded, saved, current):
    for key in ('geometry', 'identities', 'rounds', 'page_count', 'kv_payload'):
        if saved[key] != current[key]:
            raise ValueError(f'snapshot_mismatch:{key}')
    if not SHA.fullmatch(recorded):
        raise ValueError('invalid archived input.sha256')
    current_id = dp_capture_id(group, current)
    if current_id is not None:
        if current_id != recorded:
            raise ValueError('snapshot_mismatch:8DP_payload_SHA256')
        return 'eight_DP_archive_SHA256'
    # G10 may arrive as one merged export (eight references to one manifest).
    # First bind the old sources to the measurement, then compare decoded operands.
    if dp_capture_id(group, saved) != recorded:
        raise ValueError('old_sources_not_bound_to_archived_input_sha256')
    if normalized_inputs(group, saved) != normalized_inputs(group, current):
        raise ValueError('snapshot_mismatch:normalized_operands_layout_or_bounds')
    return 'verified_normalized_capture'


def kernel_timings(arm):
    li, final = [], []
    for sample in arm['samples']:
        intervals = sorted(sample['intervals_us']['li'])
        if len(intervals) != 4:
            raise ValueError('expected four LI kernel intervals')
        durations = [end-start for start, end in intervals]
        if (not all(isfinite(d) and d > 0 for d in durations)
                or any(right[0] < left[1] - 1e-5 for left, right in zip(intervals, intervals[1:]))
                or not isclose(sum(durations), sample['stages_us']['li'], abs_tol=1e-5)):
            raise ValueError('invalid archived LI kernel intervals')
        li.append(sample['stages_us']['li'])
        final.append(durations[-1])
    if not isclose(mean(li), arm['timing']['stage_mean_us']['li'], abs_tol=1e-5):
        raise ValueError('LI intervals disagree with stored kernel mean')
    return (mean(li), median(li)), (mean(final), median(final))


def check_arm(base, group):
    if (base/'fixed.txt').read_text().strip() != FIXED or (base/'physical_npu.txt').read_text().strip() != '0':
        raise ValueError('wrong_fixed_SHA_or_reference_NPU')
    arm = load(base, group, 'fixed', FIXED)
    cfg = arm['config']
    if (cfg.get('ranked_auto_route') or cfg.get('route_policy') or arm['route']
            or cfg['ranked_q_workers'] != 2 or cfg['device'] != 0 or cfg['iterations'] != 20
            or cfg['prefetch_tokens'] != 1024 or cfg['intermediate_fetches'] != 4
            or cfg['prefetch_rank_start'] != 1024 or cfg.get('window_limit') not in (None, 256)
            or arm['binary'] != (base/f'ranked-{FIXED[:8]}.binary.sha256').read_text().strip()):
        raise ValueError('wrong_fixed_binary_route_budget_or_runtime')
    return arm


def pair(values):
    return '/'.join(f'{value:.2f}' for value in values)


def difference(left, right):
    return '/'.join(f'{a-b:+.2f}' for a, b in zip(left, right))


def archive_lines(root, group, input_meta, native_ids, li_pair, li4_pair):
    root = Path(root)
    base = root.parent / BASELINE_RUN
    if not base.is_dir():
        return published(group, f'missing:{base}', li_pair, li4_pair)
    try:
        npu = (root/'physical_npu.txt').read_text().strip()
        if not npu.isdecimal():
            raise ValueError('current physical NPU ordinal unreadable')
        arm = check_arm(base, group)
        recorded = (base/f'g{group}'/'input.sha256').read_text().strip()
        provenance = check_snapshot(group, recorded, arm['inputs'], input_meta)
        if native_ids.shape != (16, 4, 2048) or native_ids.dtype.kind not in 'iu':
            raise ValueError('expected real strong serial logical IDs [16,4,2048]')
        digest = topk_fingerprint(native_ids)
        matches = sum(d == digest for d in arm['digests'])
        repeat_sets = len(set(arm['digests']))
        li, final = kernel_timings(arm)
        label = 'different_physical_NPU' if npu != '0' else 'same_NPU_ordinal'
        return [f'G{group} archived_6bca602a source={base}/g{group}/fixed input={provenance} '
                f'host20_reference_NPU=0 current_NPU={npu} comparison={label} '
                f'vs_strong_serial_set={"PASS" if matches == 20 else "MISMATCH"} '
                f'same_64Q_sets_calls={matches}/20 distinct_64Q_sets={repeat_sets} '
                f'native_reference=first_logical_strong_serial',
                f'G{group} archived_6bca602a LI_mean/P50_us={pair(li)} LI4_mean/P50_us={pair(final)} '
                f'LI_saved_mean/P50_us={difference(li, li_pair)} LI4_saved_mean/P50_us={difference(final, li4_pair)}']
    except (OSError, ValueError, KeyError, IndexError, TypeError, BadZipFile) as exc:
        return published(group, f'unverified:{exc}', li_pair, li4_pair)
