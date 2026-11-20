"""One comparison per real G group. Readbacks run after graph timing."""
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

import numpy as np

from peer_delta import pair_placement, read


def owner_receipts(posts):
    if posts.shape != (16, 4, 32):
        raise ValueError(f"Unexpected FinalPost shape {posts.shape}")
    row, q = np.indices((16, 4))
    view = np.where(q == 0, 2 * row + 1, 32 + 3 * row + q - 1)
    if (not np.array_equal(posts[:, :, 8], 4 * row + q + 1)
            or not np.array_equal(posts[:, :, 9], view)
            or not np.array_equal(posts[:, :, 10], posts[:, :, 0])
            or np.any(posts[:, :, 1:4]) or np.any(posts[:, :, 0] > 2048)):
        raise ValueError("FinalPost owner, queue, completion count or CQ status differs")
    return dict(owners=64, queues=int(np.unique(view).size), processed=int(posts[:, :, 10].sum()))


def load(root, group, label):
    path, = (root / f'g{group}' / label / 'results').glob('**/fused/path_result.json')
    out = read(path)
    cfg = read(path.parents[1] / 'config.json')
    inputs = read(path.parents[1] / 'replay_input.json')
    revision = (root / f'{label}.txt').read_text().strip()
    expected = dict(batch=16, token_bytes=656, history=131328, resident=16384)
    if (out['production_revision'] != revision or out['correctness'] != 'PASS' or out['failures']
            or out['transport'] != 'urma' or out['runtime_workers'] != 40 or cfg['device'] != 0
            or cfg['window_limit'] != 64 or cfg['query_owner_arm'] != label or cfg['group'] != group or cfg['intermediate_fetches'] != 9 or cfg['prefetch_tokens'] != 512
            or cfg['level'] != 1 or cfg['warmup'] != 4 or cfg['iterations'] != 20
            or any(inputs['geometry'][k] != v for k, v in expected.items())
            or inputs['rounds'] != 1 or out['input_paths'] != inputs['source_paths']
            or Counter(i[0] for i in inputs['identities']) != {dp: 2 for dp in range(8)}
            or len(set(map(tuple, inputs['identities']))) != 16):
        raise ValueError(f'G{group} {label}: unexpected input, runtime, configuration or failed validation')
    timing = out['all_rounds']
    times = timing['samples_us']
    rows = read(path.parent / 'diagnose/round-1/full_attribution.json')['samples']
    if (len(times) != len(rows) or len(times) != 20 or timing['measured_graphs'] != 1
            or timing['timing_protocol'] != 'one_profiled_graph' or min(times) <= 0
            or len(out['topk_set_digests']) != len(rows)):
        raise ValueError(f'{label}: incomplete graph measurement')
    binary = (root / f'ranked-{revision[:8]}.binary.sha256').read_text().strip()
    if out['binary_sha256'] != binary:
        raise ValueError(f'{label}: replay loaded a different native library')
    for sample in rows:
        requests = sample['algorithm']['requests']
        if len(requests) != 16 or any(len(r['windows']) != 9 or r['prefetch'] > 512
                                      or any(w['posted'] > 64 for w in r['windows'])
                                      for r in requests):
            raise ValueError(f'{label}: observed window count or prefetch budget differs')
    traffic = [v['algorithm']['totals'] for v in rows]
    if any(v['KV_read_bytes'] != 656 * (v['final'] + v['prefetch']) for v in traffic):
        raise ValueError(f'{label}: inconsistent physical Get bytes')
    if label == 'candidate':
        receipts = out['query_owners']
        if len(receipts) != len(rows) or any(r != dict(owners=64, queues=64, processed=v['final'])
                                              for r, v in zip(receipts, traffic)):
            raise ValueError('Query-owner CQ counts differ from actual final Get traffic')
    miss = 100 * sum(v['initial_cold'] for v in traffic) / sum(v['selected_unique'] for v in traffic)
    kib = [mean(v[k] for v in traffic) * 656 / 1024 for k in ('unused_prefetch', 'final')]
    mib = mean(v['KV_read_bytes'] for v in traffic) / 2**20
    stages = timing['stage_mean_us']
    drift = 100 * (mean(times[-5:]) / mean(times[:5]) - 1)
    m, p = timing['mean_us'], median(times)
    print(f'G{group} {label:<9s} full_mean/P50_us={m:.2f}/{p:.2f} '
          f'LI/tail_mean_us={stages["li"]:.2f}/{stages["resolve_update"]:.2f} '
          f'span_drift_pct={drift:+.2f} miss_pct={miss:.3f} '
          f'waste/repair_KiB={kib[0]:.1f}/{kib[1]:.1f} total_MiB={mib:.4f}', flush=True)
    thp = read(root / f'g{group}' / label / 'thp.json')
    return inputs, set(out['topk_set_digests']), thp, m, p, out['binary_sha256']


def main(root, group):
    old, new = [load(root, group, name) for name in ('baseline', 'candidate')]
    if old[0] != new[0] or old[5] == new[5]:
        raise ValueError('Expected identical inputs and distinct binaries')
    placement = pair_placement(old[2], new[2], local_baseline=True)
    same = len(old[1] | new[1]) == 1
    if placement != 'matched' or not same:
        print(f'G{group} NOT_COMPARABLE placement={placement} selected_ids_identical={same} owners=64/64')
        return
    gains = [100 * (1 - new[i] / old[i]) for i in (3, 4)]
    verdict = 'BOTH_FASTER' if min(gains) > 0 else 'KEEP_BASELINE'
    print(f'G{group} {verdict} gain_mean/P50_pct={gains[0]:+.2f}/{gains[1]:+.2f} owners=64/64', flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]), int(sys.argv[2]))
