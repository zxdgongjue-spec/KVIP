"""Per-call logical TopK set fingerprints of the archived 1d2486a7 baseline runs."""
import numpy as np

from correctness import topk_fingerprint
from li_restore_archive import dp_capture_id

# fused.rs7TG7/g{group}/baseline/results/*/cases/diagnose/fused/path_result.json.
# The input fingerprint includes group, original round, ordered request identities and eight
# capture payload SHA256s. The replay reader checks the actual payloads before using them.
INPUT_AND_RESULTS = {
    2: ('86b35676301c9497a36f99710d0f8779d1bf532830995597d1aee65f2def2c78', {
        'c138af2109a1678b6adf88f3d37aff3b23c8058910b40ab7aab6c3974da30a13': 20,
    }),
    7: ('e444e32559b4b59e98d41ea2e2ff31965441706c9d9f75ab8e1fab9b357f0989', {
        '85ac041bccf42c57ae0a5983e09afc824e1534ac0c28787e5298e8c6cc5d5124': 9,
        'de8b33ed7359f108819b1fdf0e4f41ca419a46735925d1fd6e6040de0772c1cf': 4,
        '1f7fc97568b47a0d8a0dd896d6d09a957c38bea7c058b28b54a70c69f106d202': 4,
        '926a19530e9cf3c95f85cbd324cd88a127d9c1198e28a63fa3c6a46266b0afae': 2,
        'a4b436f7fd9aa7c2e3d853e0216e56f04a8aed7a572427ef5c2dee6eb69a26ae': 1,
    }),
    10: ('503ec4a68022e61f29e0fe69ae8ebb48f647dee24ac04b142f947684973c1e4f', {
        '84d0ea70bde22ff45743efda3c4cbb49c4e7026882eefa2518a0e4f0aa3b6af0': 20,
    }),
    17: ('c5d461f834ad97bb28b828b3468a7322056e841c64dd91f11e5cc3fc526da274', {
        '8c138feb624341efc53276691954ab4adc1176a08fa755ed6f2808f1b93d66ff': 20,
    }),
    4: ('5ec2a3afec547b312a092928682fa15553f293c153ed3da8d88e84c9d11ff4a3', {
        '416f2817b5615e96931cbf994fb9b465ea0860e5ec2c0093c69daec5ddd0f1bc': 19,
        '2e161537d36f6da834d3960713f5aaa78d2d144d14074b753a6a226dedac45c3': 1,
    }),
}


def archived_integration_line(group, input_meta, native_ids):
    prefix = f'G{group} archived_1d2486a7'
    input_hash, calls = INPUT_AND_RESULTS[group]
    try:
        identity = dp_capture_id(group, input_meta)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return f'{prefix} vs_first_strong_serial=UNVERIFIED input_error={error}'
    if identity != input_hash:
        return f'{prefix} vs_first_strong_serial=UNVERIFIED current_input_SHA256={identity or "unavailable"}'
    ids = np.asarray(native_ids)
    if ids.shape != (16, 4, 2048) or ids.dtype.kind not in 'iu':
        raise ValueError('Expected 16x4 native logical TopK outputs')
    matched = calls.get(topk_fingerprint(ids), 0)
    verdict = 'PASS' if matched == 20 else 'MISMATCH'
    return (f'{prefix} vs_first_strong_serial={verdict} same_64Q_sets_calls={matched}/20 '
            f'distinct_64Q_sets={len(calls)} input_SHA256={identity}')
