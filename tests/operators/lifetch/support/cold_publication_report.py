"""Summarize physical score frontiers and the window readers that consume them."""
import json
from pathlib import Path
from statistics import mean
import sys


def summarize(calls):
    first_ready = [0, 0, 0, 0]
    total = [0, 0, 0, 0]
    shared_wait = [[], [], [], []]
    visibility = [0, 0, 0]
    for call in calls:
        clock, workers = call['clock_mhz'], call['workers']
        by_aiv = {w['aiv']: w for w in workers}
        batch = len(workers) // 4
        by_service = {(w['service_epoch'], w['service_row']): w for w in workers}
        if len(by_service) != 4 * batch or len(by_aiv) != len(workers):
            raise ValueError('Incomplete score or service geometry')
        for w in workers:
            epoch = w['service_epoch']
            total[epoch] += 1
            first_ready[epoch] += int(w['first_poll_complete'])
        for aiv in range(0, len(workers), 2):
            pair = [by_aiv[aiv], by_aiv[aiv+1]]
            epoch = pair[0]['service_epoch']
            if pair[1]['service_epoch'] != epoch:
                raise ValueError('Mixed service epochs on one scoring core')
            if not epoch:
                continue
            # The previous service reads its final clock immediately before
            # setting complete. Exclude the consumer's final polling overhead.
            overlap = 0
            if all(not w['first_poll_complete'] for w in pair):
                earlier_finish = min(by_service[epoch-1, w['service_row']]['post_end'] for w in pair)
                overlap = max(0, earlier_finish-max(w['first_poll'] for w in pair))
            shared_wait[epoch].append(overlap / clock)
        last = max((w for w in workers if w['service_epoch'] == 3),
                   key=lambda w: (w['post_end'], -w['service_row']))
        for w in workers:
            if w['scoring_row'] != last['service_row']:
                continue
            for tile, issued, done in w['segments'][-1]:
                if done <= last['freeze_begin']:
                    visibility[0] += 1
                elif issued >= last['freeze_end']:
                    visibility[2] += 1
                else:
                    visibility[1] += 1
    return first_ready, total, shared_wait, visibility


def main(root):
    if (root/'physical_npu.txt').read_text().strip() != '4':
        raise ValueError('Expected physical NPU4')
    for group in (10, 4, 17, 7, 2):
        records = json.loads((root/f'g{group}/cold_publication.json').read_text())
        calls = records['calls']
        if (records['group'] != group or records['schema'] != 1 or len(calls) != 20 or
                any(c['call'] != i or len(c['workers']) != 64 for i,c in enumerate(calls))):
            raise ValueError(f'Incomplete G{group} capture')
        yes, n, wait, vis = summarize(calls)
        if sum(vis) != 20*4*4:
            raise ValueError('Missing last-row publications')
        print(f'G{group} SCORE_FRONTIER calls=20 workers_per_call=64 observations={20*64*12} '
              'predecessor_ready_first_read_W1_W2_W3=' +
              ','.join(f'{yes[i]}/{n[i]}' for i in (1,2,3)) +
              ' sibling_shared_wait_mean_us_W1_W2_W3=' +
              '/'.join(f'{mean(wait[i]):.2f}' for i in (1,2,3)) +
              ' last_row_last_segment_marks_before_during_after_snapshot=' + '/'.join(map(str,vis)))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
