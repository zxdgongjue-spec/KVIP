# SPDX-License-Identifier: Apache-2.0
"""Save one untimed stock LI reference for both candidates on a local capture."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from native_li_reference import check_positions, digest, get_reference, native_li
from reporting import atomic_json


def load_oracle(directory, group):
    receipt = json.loads((directory / 'native_score_oracle.json').read_text())
    source = receipt['native_reference']
    identity = source['identity']
    ends = np.asarray(source['effective_native_ends'])
    if (receipt['group'] != group or receipt['reference_kind'] != 'native-only'
            or identity['visible'] != '4' or identity['operator'] != 'torch_npu.npu_quant_lightning_indexer'
            or ends.shape != (16, 4) or source['sparse_count'] != 2048):
        raise ValueError(f'G{group}: unexpected standalone native TopK reference')
    ids = np.load(directory / 'strong_serial.npy', allow_pickle=False)
    check_positions(ids, ends)
    if digest(ids) != receipt['ids_sha256']:
        raise ValueError(f'G{group}: standalone native TopK checksum mismatch')
    return ids, receipt


def prepare(trace, group, directory):
    import torch
    import torch_npu
    from workloads import ReplayPagePlan, ReplaySequence

    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != '4':
        raise ValueError('Expected one visible physical NPU4')
    device = torch.device('npu:0')
    torch.npu.set_device(device)
    sequence = ReplaySequence(trace, group, (1, 1))
    sequence.cache_ranked_round()
    inputs = next(iter(sequence))
    if (len(sequence.rounds), inputs.geometry['batch'], inputs.geometry['token_bytes']) != (1, 16, 656):
        raise ValueError('Expected one captured BS16 Q4 656B round')
    directory.mkdir(parents=True, exist_ok=True)
    ids, source = get_reference(inputs, torch, torch_npu, device, native_li, directory, strong_only=True)
    capture = dict(geometry=inputs.geometry, identities=inputs.identities, rounds=1,
                   page_count=ReplayPagePlan(sequence).page_count,
                   kv_payload='synthetic identity C8; actual KV not captured',
                   source_paths=[str(ref.manifest_path) for refs in sequence.rounds for ref in refs])
    np.save(directory/'strong_serial.npy', ids['strong_serial'], allow_pickle=False)
    atomic_json(directory/'native_score_oracle.json', dict(group=group, reference_kind='native-only',
                native_reference=source, ids_sha256=digest(ids['strong_serial']), capture=capture))
    load_oracle(directory, group)
    print(f"G{group} correctness_oracle={source['origin']} native_only=1 timing_runs=0 path={directory}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('group', type=int, choices=(10, 4, 17, 7, 2))
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    prepare(args.trace, args.group, args.directory)
