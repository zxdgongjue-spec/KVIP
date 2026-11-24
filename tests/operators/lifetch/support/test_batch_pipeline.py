# SPDX-License-Identifier: Apache-2.0
"""CPU checks for A5 pipeline ordering and live queue ownership checks."""
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[4] / 'python')]
from batch_pipeline import check_slot_queues, enqueue_groups, triple_overlap  # noqa: E402


class Event:
    def __init__(self):
        self.recorded = None

    def record(self, stream):
        self.recorded = stream.end


class Stream:
    def __init__(self):
        self.end = 0

    def wait_event(self, event):
        if event.recorded is None:
            raise AssertionError('An event must be recorded before adding a wait')
        self.end = max(self.end, event.recorded)

    def launch(self, duration):
        start, self.end = self.end, self.end + duration
        return start, self.end


def replay(serial):
    compute = Stream()
    owners = [Stream() for _ in range(3)]
    ready = [[Event() for _ in range(4)] for _ in range(4)]
    done = [Event() for _ in range(4)]
    intervals = []

    def run(stage, group, window, stream, duration):
        start, end = stream.launch(duration)
        intervals.append(dict(stage=stage, group=group, window=window, start_us=start, end_us=end))

    enqueue_groups(compute, owners, ready, done, serial,
                   lambda g, w: run('score', g, w, compute, 4),
                   lambda g, w: run('service', g, w, owners[g % 3], 6),
                   lambda g: run('terminal', g, None, owners[g % 3], 30))
    return intervals, done


class BatchPipelineTests(unittest.TestCase):
    def test_dependency_and_three_stage_overlap(self):
        intervals, done = replay(False)
        spans = {(r['stage'], r['group'], r['window']): (r['start_us'], r['end_us']) for r in intervals}
        for group in range(4):
            for window in range(4):
                self.assertGreaterEqual(spans['service', group, window][0], spans['score', group, window][1])
            self.assertGreaterEqual(spans['terminal', group, None][0], spans['service', group, 3][1])
        for group in range(1, 4):
            self.assertGreaterEqual(spans['score', group, 0][0], spans['score', group - 1, 3][1])
        self.assertGreaterEqual(spans['score', 3, 0][0], done[0].recorded)
        self.assertGreater(triple_overlap(intervals), 0)

    def test_same_kernels_for_serial_control_with_terminal_gate(self):
        serial, serial_done = replay(True)
        parallel, parallel_done = replay(False)
        self.assertEqual([(r['stage'], r['group'], r['window']) for r in serial],
                         [(r['stage'], r['group'], r['window']) for r in parallel])
        for record in serial:
            if record['stage'] == 'score' and record['group']:
                self.assertGreaterEqual(record['start_us'], serial_done[record['group'] - 1].recorded)
        self.assertEqual(triple_overlap(serial), 0)
        self.assertLess(parallel_done[-1].recorded, serial_done[-1].recorded)

    def test_physical_queue_alias_across_owner_streams(self):
        lane0, lane1 = ((1, 2, 3, 4, 5),), ((6, 7, 8, 9, 10),)
        check_slot_queues([(0, lane0), (1, lane1), (0, lane0)])
        for column in range(5):
            aliased = list(lane1[0])
            aliased[column] = lane0[0][column]
            with self.assertRaisesRegex(RuntimeError, 'physical queue alias'):
                check_slot_queues([(0, lane0), (1, (tuple(aliased),))])
        with self.assertRaisesRegex(RuntimeError, 'changed its physical queues'):
            check_slot_queues([(0, lane0), (0, lane1)])


if __name__ == '__main__':
    unittest.main()
