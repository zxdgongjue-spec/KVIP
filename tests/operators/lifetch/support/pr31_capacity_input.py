# SPDX-License-Identifier: Apache-2.0
"""Reuse the certified PR31 short source for the two 512K capacity measurements."""
from copy import copy
import json

import numpy as np

from cap_pair_input import bind_capacity
from correctness import topk_fingerprint
from native_li_reference import digest
from near512_locality_input import expand_locality
from near512_profile import bind_source_check, compare_profiles, _categories, _locality
from near512_oracle import serial_reference, validate_native_sets
from score_equivalence import CapturedScoreOracle

PR31 = '58155892a4f3b7a3bf9789d6055fa4a0f1e95c07'
NARROW, WIDE = 131328, 524544


def check_donor(proof_path, hashes, cap):
    narrow = json.loads(proof_path.read_text())
    if (narrow['production_revision'] != PR31 or narrow['runtime']['visible'] != '4'
            or narrow['capture_operand_hashes'] != hashes or narrow['measured_calls'] != 10
            or narrow['history_cap']['bound_history'] != NARROW):
        raise ValueError(f'128K donor differs from the completed narrow PR31 arm: {proof_path}')
    for field in ('capture_history', 'captured_page_count', 'effective_ends', 'query_ends_sha256',
                  'capture_page_table_sha256', 'capture_state_sha256'):
        if cap[field] != narrow['history_cap'][field]:
            raise ValueError(f'Archived narrow donor differs in {field}')
    return narrow


def long_oracle(source, inputs, expansion, donor_ids, torch, torch_npu):
    # A real native LI reference recomputed over the new full causal domain.
    # Validate the chosen IDs against both this reference and the short donor.
    torch.npu.set_device(0)
    device = torch.device('npu:0')
    reference = serial_reference(inputs, torch, torch_npu, device)
    source.validate_topk(donor_ids)
    page_map = expansion['source_page_to_long_page']
    bind_source_check(source, inputs, page_map)
    native = CapturedScoreOracle(inputs, reference)
    short_profile = dict(native_unique=_locality(_categories(source), donor_ids))
    comparison = compare_profiles(short_profile, inputs, reference)
    verified = {}

    def validate(selected):
        native.validate(selected)
        inputs.check_donor_topk(selected)
        validate_native_sets(inputs, selected, reference, torch, torch_npu, device, verified)

    inputs.validate_topk = validate
    torch.npu.empty_cache()
    return reference, dict(
        source_kind='rank_preserving_short_capture', source_oracle_digest=topk_fingerprint(donor_ids),
        source_page_map_sha256=digest(np.asarray(page_map, np.int32)),
        new_winners=0, native_queries=comparison['checked_queries'],
        short_unique=comparison['short'], long_unique=comparison['long'])


def bind_workload(args, sequence, source, pages, operand_hashes, donor_ids, torch, torch_npu):
    # Make the source proof before changing either its capacities or its query bounds.
    short = args.pr31_shape == 'wide-short'
    sequence, inputs, pages, cap = bind_capacity(sequence, source, pages, WIDE if short else NARROW, digest)
    check_donor(args.narrow_proof, operand_hashes, cap)
    proof = dict(pr31_shape=args.pr31_shape, source_geometry=source.geometry)
    if short:
        # The dormant suffix has no valid score or logical address and cannot supply a winner.
        inputs.validate_topk = source.validate_topk
        return sequence, inputs, pages, cap, donor_ids, proof

    inputs, expansion = expand_locality(source, WIDE)
    sequence = copy(sequence)
    sequence.cached_ranked_round = inputs
    if (next(iter(sequence)) is not inputs or not np.array_equal(inputs.query_ends, source.query_ends + WIDE-NARROW)
            or inputs.query_ends.min() < 510*1024 or inputs.query_ends.max() > 512*1024+3):
        raise ValueError('Incorrect extended causal bounds or uncached long input')
    pages = type(pages)(sequence)
    serial, locality = long_oracle(source, inputs, expansion, donor_ids, torch, torch_npu)
    locality['long_operand_sha256'] = {name: digest(getattr(inputs, name)) for name in operand_hashes}
    locality['long_state_sha256'] = {name: digest(getattr(inputs, name)) for name in cap['capture_state_sha256']}
    proof['long_workload'] = locality
    cap.update(bound_history=WIDE, bound_page_count=pages.page_count, effective_ends=inputs.query_ends.tolist(),
               query_ends_sha256=digest(inputs.query_ends))
    return sequence, inputs, pages, cap, serial, proof
