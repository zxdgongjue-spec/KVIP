"""Certify the raw fused LI outputs against captured scores and the archived native TopK."""
import numpy as np

from correctness import topk_fingerprint
from native_li_reference import native_ends
from reporting import atomic_json


class CapturedScoreOracle:
    """Bind the cutoff and number of strict winners once for an immutable replay round."""

    def __init__(self, inputs, serial):
        # This is the full capture oracle, including a partition of every valid
        # population. It must run before we install the faster selection check.
        inputs.validate_topk(serial)
        self.ends = native_ends(inputs)[..., None]
        if np.any(serial >= self.ends):
            raise ValueError('Archived serial token exceeds its logical query bound')
        self.keys = inputs.logical_score_keys
        self.valid = inputs.logical_score_valid
        self.shape = serial.shape
        scores = np.take_along_axis(self.keys, serial, axis=-1)
        self.cutoff = scores.min(axis=-1, keepdims=True)
        self.strict_count = (scores > self.cutoff).sum(axis=-1)

    def validate(self, selected):
        ids = np.asarray(selected)
        if ids.shape != self.shape or ids.dtype.kind not in 'iu':
            raise AssertionError('Replay TopK shape/type differs from Q4 contract')
        if np.any(ids < 0) or np.any(ids >= self.keys.shape[-1]) or np.any(ids >= self.ends):
            raise AssertionError('Replay TopK token outside logical query bound')
        ordered = np.sort(ids, axis=-1)
        if np.any(ordered[..., 1:] == ordered[..., :-1]) or not np.take_along_axis(self.valid, ids, axis=-1).all():
            raise AssertionError('Replay TopK duplicates or exceeds causal prefix')
        scores = np.take_along_axis(self.keys, ids, axis=-1)
        # IDs are distinct and admissible. Matching the population's strict count
        # means every strict winner is present, even if the tie IDs change.
        wrong = (scores.min(axis=-1) != self.cutoff[..., 0]) | ((scores > self.cutoff).sum(axis=-1) != self.strict_count)
        if np.any(wrong):
            row, query = np.argwhere(wrong)[0]
            raise AssertionError(f'Replay TopK ranking differs: request={row} Q={query + 1}')


def check_scores(inputs, serial, outputs):
    # The replay oracle checks cardinality, uniqueness, valid positions, the exact
    # cutoff score and complete coverage of every token strictly above the cutoff.
    inputs.validate_topk(serial)
    ends = native_ends(inputs)[..., None]
    if np.any(serial >= ends):
        raise ValueError('Archived serial token exceeds its logical query bound')
    serial_sets = np.sort(serial, axis=-1)
    serial_scores = np.take_along_axis(inputs.logical_score_keys, serial, axis=-1)
    cutoff = serial_scores.min(axis=-1, keepdims=True)
    matches = []
    for i, selected in enumerate(outputs):
        try:
            inputs.validate_topk(selected)
        except AssertionError as error:
            raise AssertionError(f'Call {i + 1}: {error}') from error
        if np.any(selected >= ends):
            raise ValueError(f'Call {i + 1}: candidate token exceeds its logical query bound')
        # The capture oracle validated both sets against the same population. In
        # particular, every differing ID must have exactly the serial cutoff score.
        scores = np.take_along_axis(inputs.logical_score_keys, selected, axis=-1)
        if not np.array_equal(scores.min(axis=-1, keepdims=True), cutoff):
            raise ValueError(f'Call {i + 1}: candidate and serial cutoff scores differ')
        matches.append(np.all(np.sort(selected, axis=-1) == serial_sets, axis=-1))
    match = np.asarray(matches)
    return dict(calls=len(outputs), queries=int(match.size), exact_calls=int(match.all(axis=(1, 2)).sum()),
                exact_queries=int(match.sum()), serial_set_digest=topk_fingerprint(serial))


def save_checked(directory, group, inputs, reference, metadata, outputs, *, expected_calls=20):
    if len(outputs) != expected_calls or inputs.geometry['batch'] != 16:
        raise ValueError(f'Expected {expected_calls} measured Q4 calls with batch 16')
    result = check_scores(inputs, reference, outputs)
    # The standalone native LI reference is part of this result archive, even
    # if a later run prunes its build or native LI cache.
    extra = {'strong_serial': reference} if metadata.get('origin') in ('fresh', 'validated-cache', 'near512-sharded-native') else {}
    np.savez_compressed(directory / 'score_topk_ids.npz', fused=np.asarray(outputs), **extra)
    atomic_json(directory / 'score_topk_check.json', dict(group=group, native_reference=metadata, **result))
