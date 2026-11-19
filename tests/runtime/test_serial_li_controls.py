"""CPU contracts for the three-way LI serial/fused experiment."""
import ctypes
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/operators/lifetch'))
sys.path.insert(0, str(ROOT / 'tests/operators/lifetch/support'))
import full_chain_compare
import optimized_serial_li
import serial_chain_compare
from workloads import configuration


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.device = 'cpu'

    def to(self, device):
        return self

    def data_ptr(self):
        return self.value.ctypes.data

    def __eq__(self, other):
        return self.value == other


class CpuTorch:
    uint8, int16, int32, float32 = np.uint8, np.int16, np.int32, np.float32
    from_numpy = staticmethod(Tensor)

    @staticmethod
    def empty(shape, dtype, device):
        return Tensor(np.empty(shape, dtype=dtype))

    @staticmethod
    def full(shape, value, dtype, device):
        return Tensor(np.full(shape, value, dtype=dtype))


class Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class SerialControls(unittest.TestCase):
    def test_owner_coverage_and_wave_capacity(self):
        for batch, capacity, requested in ((1, 28, 0), (16, 32, 0), (16, 28, 0), (16, 30, 2), (1, 28, 32)):
            parts, stride, waves = optimized_serial_li.task_plan(batch, 131072, capacity, requested)
            self.assertTrue(all(0 < len(wave) <= capacity for wave in waves))
            for owner in range(batch):
                segments = sorted((t[2], t[3]) for wave in waves for t in wave if t[4] == owner)
                self.assertEqual(segments, [(p*stride, (p+1)*stride) for p in range(parts)])

    def test_three_way_plan_preserves_legacy(self):
        args = full_chain_compare.arguments(['--kvaf-server', 'example:28885'], serial=True)
        jobs = list(serial_chain_compare.plan(args, Path('/tmp/plan')))
        self.assertEqual(len(jobs), 8)
        for job in jobs:
            config = configuration(job['argv'][2:])
            self.assertTrue(config.compare_optimized_serial)
            self.assertEqual((config.score_route, config.score_post, config.score_service), ('legacy', 'separate', 'separate'))
        self.assertEqual(len(list(full_chain_compare.plan(args, Path('/tmp/old-plan')))), 24)

    def test_serial_binding_bytes_and_owner_tasks(self):
        h, heads = 131072, 32
        fixture = dict(query=Tensor(np.empty((1,4,heads,128), np.uint8)),
                       key=Tensor(np.empty((h,128), np.uint8)),
                       actual_key=Tensor(np.array([h], np.int32)),
                       block_table=Tensor(np.arange(h//128, dtype=np.int32)[None]),
                       native_inputs=dict(weights=Tensor(np.empty((1,4,heads), np.uint16)),
                                          query_scale=Tensor(np.empty((1,4,heads), np.float32)),
                                          key_scale=Tensor(np.empty(h, np.float32))))
        class Library:
            li_serial_context_bytes = Function(lambda: 320)
            li_serial_context_init = Function(lambda *args: 0)
            li_serial_score = Function(lambda *args: 0)
            li_serial_topk = Function(lambda *args: 0)
        with patch('optimized_serial_li.ctypes.CDLL', return_value=Library()):
            li = optimized_serial_li.OptimizedSerialLI(CpuTorch, fixture, Path(__file__),
                selected_shape=(4,1,2048), batch=1, history=h, heads=heads, aics=8, aivs=16, parts=16)
        self.assertEqual(len(li.waves), 2)
        for rows, inputs, binding, sidecar in li.waves:
            self.assertEqual(inputs.value.size, 88)
            self.assertEqual(sidecar.value.size, 56)
            pointers = inputs.value[:72].view(np.uint64)
            self.assertEqual(pointers[0], fixture['query'].data_ptr())
            self.assertEqual(pointers[3], fixture['block_table'].data_ptr())
            np.testing.assert_array_equal(inputs.value[72:].view(np.uint32), [heads,h,8192,rows])
            words = sidecar.value[:48].view(np.uint64)
            self.assertEqual(words[0], li.canonical.data_ptr())
            self.assertEqual(words[5], li.selected.data_ptr())
            np.testing.assert_array_equal(sidecar.value[48:].view(np.uint32), [h,1])
            tasks = np.frombuffer(ctypes.string_at(int(pointers[8]), rows*20), np.uint32).reshape(rows,5)
            self.assertTrue(np.all(tasks[:,4] == 0))
            self.assertTrue(np.all(tasks[:,3] - tasks[:,2] == 8192))


if __name__ == '__main__':
    unittest.main()
