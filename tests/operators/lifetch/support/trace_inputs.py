# SPDX-License-Identifier: Apache-2.0
"""Read the existing schema-4 service capture without importing device libraries.

The format originates in kvip.diagnostics.score_trace. Captured post state is
reference evidence only: replay must evolve its own state, not restore it each
round. Score-only archives are suitable for calibration, not executable LI.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TOPK = 2048
QUERIES = 4
BLOCK = 128
BASE_ARRAYS = frozenset({
    'pre.row_owners', 'pre.logical_ids', 'pre.hot_valid', 'pre.metadata',
    'pre.query_ends', 'pre.tail_bounds', 'pre.lru_slots', 'pre.address_valid',
    'pre.ddr_page_slots', 'post.logical_ids', 'post.hot_valid',
    'post.logical_output', 'post.slot_output', 'post.topk',
    'post.prefetch_records', 'post.final_records', 'post.read_counts',
})
SCORE_ARRAYS = frozenset({
    'post.score_keys', 'post.threshold_records', 'post.window_posted',
    'post.window_processed_ends', 'post.prefetch_ordinals',
})
REAL_ARRAYS = frozenset({
    'real.keys', 'real.key_scales', 'real.query', 'real.weights', 'real.query_scale',
})


def checked_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Trace member missing or outside capture: {relative}')
    return path


def score_values(keys):
    """Decode sortable BF16 keys; a numeric uint16 cast is not a score."""
    keys = np.asarray(keys, dtype=np.uint16)
    bits = np.where(keys & 0x8000, keys ^ 0x8000, ~keys).astype(np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32)


@dataclass
class ScoreCapture:
    path: Path
    manifest: dict
    arrays: dict

    @property
    def executable(self):
        return self.manifest['real_inputs_saved']

    def logical_keys(self, row):
        """Return LI rows in logical-token order for subsequent layout remapping."""
        if not self.executable:
            raise ValueError('Score-only capture has no executable Q/Index inputs')
        ids = self.arrays['pre.logical_ids'][row]
        keys = np.empty_like(self.arrays['real.keys'][row])
        scales = np.empty_like(self.arrays['real.key_scales'][row])
        keys[ids] = self.arrays['real.keys'][row]
        scales[ids] = self.arrays['real.key_scales'][row]
        return keys, scales


def load_score_capture(manifest_path, *, require_real=True):
    path = Path(manifest_path).resolve()
    if path.is_dir():
        path /= 'manifest.json'
    manifest = json.loads(path.read_text())
    legacy = manifest.get('kind') == 'lifetch_legacy_replay_snapshot'
    if legacy:
        if (manifest.get('schema_version') != 1 or manifest.get('status') != 'complete'
                or manifest.get('provenance', {}).get('identity_scope') != 'independent_snapshot_only'):
            raise ValueError('Invalid legacy executable snapshot')
    elif (manifest.get('schema_version') != 4
            or manifest.get('kind') != 'kvaf_decode_score_trace'
            or manifest.get('content') != 'scores'
            or manifest.get('status') != 'complete'
            or manifest.get('validation', {}).get('passed') is not True
            or manifest['validation'].get('kind') != 'score_trace_integrity'):
        raise ValueError(f'Expected complete schema-4 score capture: {path}')
    real = manifest.get('real_inputs_saved')
    if type(real) is not bool or real != manifest['metadata'].get('capture_real_inputs', False):
        raise ValueError('Real LI capture marker is inconsistent')
    if require_real and not real:
        raise ValueError(f'No real Q/Index saved in {path}; score-only data cannot replay LI')
    description = manifest['archive']
    archive = checked_path(path.parent, description['file'])
    with archive.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    if archive.stat().st_size != description['bytes'] or digest != description['sha256']:
        raise ValueError(f'Trace archive size/checksum differs: {archive}')
    expected = BASE_ARRAYS | SCORE_ARRAYS | (REAL_ARRAYS if real else frozenset())
    if legacy:
        if not real:
            raise ValueError('Legacy replay requires real operands')
        expected = REAL_ARRAYS | {name for name in BASE_ARRAYS if name.startswith('pre.')} | {'post.topk', 'post.score_keys'}
    with np.load(archive, allow_pickle=False) as source:
        if set(source.files) != expected or set(manifest['arrays']) != expected:
            raise ValueError('Missing or unexpected capture arrays')
        arrays = {name: source[name] for name in source.files}
    for name, array in arrays.items():
        spec = manifest['arrays'][name]
        if array.dtype.hasobject or array.dtype.name != spec['dtype'] or list(array.shape) != spec['shape']:
            raise ValueError(f'Trace array differs from manifest: {name}')
    capture = ScoreCapture(path, manifest, arrays)
    validate_inputs(capture)
    return capture


def validate_inputs(capture):
    """Validate executable shapes, logical identities and captured TopK evidence."""
    a, g = capture.arrays, capture.manifest['geometry']
    b, h, r, heads = (g[name] for name in ('batch', 'history', 'resident', 'heads'))
    if (not all(type(n) is int for n in (b, h, r, heads))
            or not 1 <= b <= 16 or not TOPK <= r <= h
            or h % BLOCK or r % BLOCK or heads <= 0):
        raise ValueError('Unsupported capture geometry')
    specs = {
        'pre.logical_ids': ('int32', (b, h)),
        'pre.hot_valid': ('uint8', (b, r)),
        'pre.address_valid': ('uint8', (b, h)),
        'pre.row_owners': ('int32', (b,)),
        'pre.metadata': ('int64', (5, b * QUERIES)),
        'pre.query_ends': ('int32', (b, QUERIES)),
        'pre.lru_slots': ('int32', (b, TOPK)),
        'pre.tail_bounds': ('int32', (b, 2)),
        'pre.ddr_page_slots': ('int32', (b, h // BLOCK)),
        'post.topk': ('int32', (b, QUERIES, TOPK)),
        'post.score_keys': ('uint16', (b, QUERIES, h)),
    }
    if capture.executable:
        specs.update({
            'real.keys': ('uint8', (b, h, 128)),
            'real.key_scales': ('float32', (b, h)),
            'real.query': ('uint8', (b, QUERIES, heads, 128)),
            'real.weights': ('uint16', (b, QUERIES, heads)),
            'real.query_scale': ('float32', (b, QUERIES, heads)),
        })
    for name, (dtype, shape) in specs.items():
        if a[name].dtype.name != dtype or a[name].shape != shape:
            raise ValueError(f'Capture input dtype/shape invalid: {name}')
    if np.any(a['pre.hot_valid'] > 1) or np.any(a['pre.address_valid'] > 1):
        raise ValueError('Validity fields must be binary')
    if not np.array_equal(np.sort(a['pre.metadata'][2]), np.arange(b * QUERIES)):
        raise ValueError('Packed query mapping is not a permutation')
    if not np.array_equal(a['pre.metadata'][4].view(np.int32)[:b], a['pre.row_owners']):
        raise ValueError('Packed owner mapping differs from capture rows')
    for row in range(b):
        if not np.array_equal(np.sort(a['pre.logical_ids'][row]), np.arange(h)):
            raise ValueError('Logical Index IDs are not a permutation')
        lru = a['pre.lru_slots'][row]
        if np.any(lru < 0) or np.any(lru >= r) or len(np.unique(lru)) != TOPK:
            raise ValueError('Invalid LRU candidate slots')
        for q in range(QUERIES):
            end = int(a['pre.query_ends'][row, q])
            selected = a['post.topk'][row, q]
            if not TOPK <= end <= h or np.any(selected < 0) or np.any(selected >= end):
                raise ValueError('TopK exceeds valid query prefix')
            if len(np.unique(selected)) != TOPK:
                raise ValueError('Duplicate positions in captured TopK')
            scores = a['post.score_keys'][row, q, :end]
            if np.any(np.isnan(score_values(scores))):
                raise ValueError('NaN in captured valid score prefix')
            cutoff = np.partition(scores, end - TOPK)[end - TOPK]
            chosen = scores[selected]
            if np.min(chosen) != cutoff or np.sum(chosen > cutoff) != np.sum(scores > cutoff):
                raise ValueError('Captured TopK differs from scores (cutoff ties allowed)')


@dataclass(frozen=True)
class CaptureReference:
    dp_rank: int
    source_round: int
    scheduler_round: int
    group: int
    manifest_path: Path
    request_signature: tuple


def select_series(root, group, round_range):
    """Select capture rounds through series.json, preserving DP/request identity.

    Source round means round_index in the capture, not the scheduler's decode
    counter. Payloads remain lazy so a long series is not all loaded into RAM.
    """
    root = Path(root).resolve()
    first, last = round_range
    if type(group) is not int or group < 0 or not 1 <= first <= last:
        raise ValueError('Invalid group or source round range')
    series_paths = sorted(root.glob('dp[0-9]*/series.json'))
    if not series_paths:
        raise ValueError(f'No DP series manifests under {root}')
    references, seen_ranks, expected_ranks = [], set(), None
    for path in series_paths:
        series = json.loads(path.read_text())
        rank = series.get('dp_rank')
        if (type(rank) is not int or rank < 0 or path.parent.name != f'dp{rank}'
                or rank in seen_ranks or series.get('schema_version') != 1
                or series.get('kind') != 'kvaf_decode_snapshot_series'
                or series.get('status') != 'complete' or series.get('content') != 'scores'):
            raise ValueError(f'Invalid or incomplete score series: {path}')
        seen_ranks.add(rank)
        configured = set(series['dp_ranks'])
        if expected_ranks is None:
            expected_ranks = configured
        if configured != expected_ranks:
            raise ValueError('DP series disagree on configured ranks')
        rounds = series['rounds']
        if series['completed_rounds'] != series['expected_rounds'] or len(rounds) != series['expected_rounds']:
            raise ValueError(f'Incomplete round coverage: {path}')
        if last > len(rounds):
            raise ValueError(f'Requested round {last} absent in {path}')
        previous_scheduler = None
        signature = tuple((item['request_id'], item['owner'])
                          for item in series['request_owner_signature'])
        if len(signature) != series['batch'] or len(set(signature)) != len(signature):
            raise ValueError('Invalid request identity signature')
        for number, entry in enumerate(rounds, 1):
            schedule = entry['scheduler_round']
            if (entry['round_index'] != number or type(schedule) is not int
                    or (previous_scheduler is not None and schedule != previous_scheduler + 1)):
                raise ValueError(f'Nonconsecutive captured rounds: {path}')
            previous_scheduler = schedule
            if not first <= number <= last:
                continue
            matches = [item for item in entry['groups'] if item['group'] == group]
            if len(matches) != 1:
                raise ValueError(f'Round {number} requires exactly one group {group}')
            manifest_path = checked_path(path.parent, matches[0]['manifest'])
            manifest = json.loads(manifest_path.read_text())
            metadata = manifest.get('metadata', {})
            requests = metadata.get('requests', [])
            actual_signature = tuple((item['request_id'], item['owner']) for item in requests)
            if (metadata.get('dp_rank') != rank or metadata.get('group') != group
                    or metadata.get('context', {}).get('round') != schedule
                    or metadata.get('context', {}).get('global_decode_confirmed') is not True
                    or actual_signature != signature
                    or manifest.get('geometry', {}).get('batch') != series['batch']
                    or series['queries'] != QUERIES
                    or any(item['query_stop'] - item['query_begin'] != QUERIES for item in requests)):
                raise ValueError(f'Round identity differs from series: {manifest_path}')
            references.append(CaptureReference(rank, number, schedule, group, manifest_path, signature))
    if seen_ranks != expected_ranks:
        raise ValueError('Configured DP capture is missing')
    return sorted(references, key=lambda ref: (ref.source_round, ref.dp_rank))
