"""Certify the raw fused LI outputs against captured scores and the archived native TopK."""
import numpy as np

from correctness import topk_fingerprint
from native_li_reference import native_ends
from reporting import atomic_json


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


def save_checked(directory, group, inputs, reference, metadata, outputs):
    if len(outputs) != 20 or inputs.geometry['batch'] != 16:
        raise ValueError('Expected 20 measured Q4 calls with batch 16')
    result = check_scores(inputs, reference, outputs)
    np.savez_compressed(directory / 'score_topk_ids.npz', fused=np.asarray(outputs))
    atomic_json(directory / 'score_topk_check.json', dict(group=group, native_reference=metadata, **result))
