import ctypes
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from kvip.adapters.kvaf import KvafGetLease, KvafQueueView, LeasedRuntime


class Function:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


class TransportTest(unittest.TestCase):
    def setUp(self):
        self.stream = SimpleNamespace(npu_stream=123)
        self.launches = []
        self.library = SimpleNamespace()
        for name, size in (("args", 112), ("status", 1216), ("entity", 256), ("queue", 64)):
            setattr(self.library, f"refactor_fetch_{name}_bytes", Function(lambda size=size: size))

        def pack(blob, size, fields, count):
            self.assertEqual((size, count), (112, 18))
            self.fields = list(fields)
            ctypes.memmove(blob, struct.pack("=10Q8I", *fields), size)
            return 0

        self.library.refactor_fetch_pack_args = Function(pack)
        self.native_get = Function(lambda *args: self.launches.append(args) or 0)
        self.runtime = Mock()
        self.output = torch.empty((128, 656), dtype=torch.uint8)
        self.context = dict(
            lease_id=1,
            worker_count=2,
            jettys_per_core=2,
            token_bytes=656,
            tokens_per_block=128,
            output_address=self.output.data_ptr(),
            output_bytes=self.output.numel(),
            device_channel_entity=100,
            device_core_status=200,
            device_queue_state=300,
        )
        self.runtime.acquire_fused_batch_get_context.return_value = self.context
        self.npu = SimpleNamespace(current_stream=lambda device: self.stream)
        self.torch = SimpleNamespace(
            npu=self.npu,
            tensor=torch.tensor,
            zeros=torch.zeros,
            uint8=torch.uint8,
            int32=torch.int32,
            int64=torch.int64,
            uint64=getattr(torch, "uint64", object()),
        )
        with (
            patch("kvip.adapters.kvaf.load_lifetch_library", return_value=self.library),
            patch("kvip.adapters.kvaf.load_native_get", return_value=(object(), self.native_get)),
        ):
            self.owner = KvafGetLease(self.torch, self.runtime, "test", self.output, self.stream)

    def inputs(self):
        return (
            torch.ones(2, dtype=torch.int64),
            torch.ones(2, dtype=torch.uint8),
            torch.zeros(8, dtype=torch.int32),
            torch.zeros(8, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(8, dtype=torch.int32),
            self.output,
        )

    def test_replay_retains_arguments_and_does_not_upload(self):
        inputs = self.inputs()
        plan = self.owner.prepare_get(*inputs)
        self.assertEqual(self.fields[:7], [t.data_ptr() for t in inputs])
        self.assertEqual(struct.unpack("=10Q8I", plan.arguments.raw[:112]), tuple(self.fields))
        self.assertEqual(self.fields[7:], [200, 100, 300, 2, 8, 128, 656, 128, 2, 2, 2048])
        with patch.object(self.torch, "tensor", side_effect=AssertionError("replay must not allocate")):
            plan()
            plan()
        self.assertEqual(self.launches, [(plan.arguments, 123)] * 2)
        self.runtime.register_fused_batch_get_output.assert_not_called()
        self.owner.close()
        self.owner.close()
        self.runtime.release_fused_batch_get_context.assert_called_once_with(self.context, self.stream)
        with self.assertRaisesRegex(RuntimeError, "released"):
            plan()

    def test_cached_native_launch_reads_current_device_metadata(self):
        runtime = LeasedRuntime(self.owner)
        inputs = self.inputs()
        observed = []

        def launch(arguments, stream):
            fields = struct.unpack("=10Q8I", arguments.raw[:112])
            count = ctypes.cast(fields[4], ctypes.POINTER(ctypes.c_int32))[0]
            index = ctypes.cast(fields[2], ctypes.POINTER(ctypes.c_int32))[0]
            observed.append((count, index, fields[8], fields[9], stream))
            return 0

        self.owner.get = launch
        for count in (4, 8, 2, 0, 8):
            inputs[4].fill_(count)
            inputs[2].fill_(count + 10)
            runtime.batch_get(*inputs)
        self.assertEqual(observed, [(n, n + 10, 100, 300, 123) for n in (4, 8, 2, 0, 8)])
        self.assertEqual(len(runtime.bindings), 1)
        self.owner.close()

    def test_loader_uses_the_runtime_sdk_dependency_scope(self):
        from kvip.runtime.loader import load_native_get

        native = SimpleNamespace(KvafExecuteBatchGetLaunch=Function(lambda *args: 0))
        with (
            patch("kvip.runtime.loader.ctypes.CDLL", return_value=native) as load,
            patch(
                "kvip.runtime.loader.importlib.import_module", return_value=SimpleNamespace(__file__="/sdk/_kvaf.so")
            ) as module,
        ):
            library, launch = load_native_get(self.runtime)
        module.assert_called_once_with(type(self.runtime).__module__)
        load.assert_called_once_with("/sdk/_kvaf.so")
        self.assertIs(library, native)
        self.assertIs(launch, native.KvafExecuteBatchGetLaunch)
        self.assertEqual(launch.argtypes, [ctypes.c_void_p, ctypes.c_void_p])
        self.assertIs(launch.restype, ctypes.c_int32)
        self.owner.close()

    def test_output_registration_is_cached_and_shares_owner(self):
        other = torch.empty((256, 656), dtype=torch.uint8)
        self.runtime.register_fused_batch_get_output.return_value = dict(self.context, output_address=other.data_ptr())
        self.owner.register_output(other)
        self.owner.register_output(other)
        self.runtime.register_fused_batch_get_output.assert_called_once()
        self.assertIs(self.runtime.register_fused_batch_get_output.call_args.args[0], self.context)
        self.owner.close()

    def test_failed_release_keeps_storage_and_wrong_stream_is_rejected(self):
        plan = self.owner.prepare_get(*self.inputs())
        self.npu.current_stream = lambda device: SimpleNamespace(npu_stream=124)
        with self.assertRaisesRegex(RuntimeError, "lease stream"):
            plan()
        self.assertFalse(self.launches)
        self.runtime.release_fused_batch_get_context.side_effect = RuntimeError("uncertain DMA")
        with self.assertRaisesRegex(RuntimeError, "uncertain DMA"):
            self.owner.close()
        self.assertFalse(self.owner.closed)
        self.assertEqual(len(self.owner.outputs), 1)
        self.assertEqual(self.owner.plans, [plan])

    def test_fused_views_share_native_queues_without_new_runtime(self):
        addresses = torch.ones(8, dtype=torch.int64)
        valid = torch.ones(8, dtype=torch.uint8)
        view = KvafQueueView(self.owner, self.output, addresses, valid)
        arguments = view.build_device_args("same ABI library")
        self.assertEqual(tuple(arguments.shape), (4, 112))
        for queue in range(4):
            fields = struct.unpack("=10Q8I", bytes(arguments[queue].tolist()))
            self.assertEqual(fields[:2], (addresses.data_ptr(), valid.data_ptr()))
            self.assertEqual(fields[7], view.core_status.data_ptr() + queue * 1216)
            self.assertEqual(fields[8:10], (100 + queue * 256, 300 + queue * 64))
            self.assertEqual(fields[10:], (8, 8192, 128, 656, 128, 1, 1, 0))
        with patch.object(self.torch, "tensor", side_effect=AssertionError("must reuse arguments")):
            self.assertIs(view.build_device_args("same ABI library"), arguments)
        self.runtime.acquire_fused_batch_get_context.assert_called_once()
        self.runtime.register_fused_batch_get_output.assert_not_called()
        self.owner.close()

    def test_descriptor_capacity_mismatch(self):
        inputs = list(self.inputs())
        inputs[3] = torch.zeros(7, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "capacities"):
            self.owner.prepare_get(*inputs)
        self.owner.close()

    def test_capture_launches_on_capture_stream_and_fences_only_at_startup(self):
        plan = self.owner.prepare_get(*self.inputs())
        self.npu.synchronize = Mock()
        with self.owner.capture_scope():
            self.npu.current_stream = lambda device: SimpleNamespace(npu_stream=456)
            plan()
            self.assertEqual(self.launches[-1][1], 456)
        self.assertEqual(self.npu.synchronize.call_count, 2)
        with self.assertRaisesRegex(RuntimeError, "lease stream"):
            plan()
        self.npu.current_stream = lambda device: self.stream
        plan()
        self.assertEqual(self.npu.synchronize.call_count, 2)
        self.owner.close()

    def test_failed_capture_fence_retains_transport(self):
        self.npu.synchronize = Mock(side_effect=[None, RuntimeError("device failure")])
        with self.assertRaisesRegex(RuntimeError, "device failure"), self.owner.capture_scope():
            pass
        with self.assertRaisesRegex(RuntimeError, "remains retained"):
            self.owner.close()
        self.runtime.release_fused_batch_get_context.assert_not_called()

    def test_runtime_routes_table_and_tensor_views_without_new_upload(self):
        runtime = LeasedRuntime(self.owner)
        inputs = self.inputs()
        runtime.batch_get(*inputs)
        table = SimpleNamespace(addresses=inputs[0], valid=inputs[1])
        self.stream.synchronize = Mock()
        with patch.object(self.torch, "tensor", side_effect=AssertionError("replay upload")):
            runtime.batch_get(table, *[t.view_as(t) for t in inputs[2:]], synchronize=True)
        self.assertEqual(len(runtime.bindings), 1)
        self.assertEqual(len(self.launches), 2)
        self.stream.synchronize.assert_called_once()
        self.runtime.batch_get.assert_not_called()
        runtime.shutdown()
        self.runtime.shutdown.assert_called_once()

    def test_runtime_does_not_destroy_transport_after_failed_release(self):
        runtime = LeasedRuntime(self.owner)
        runtime.batch_get(*self.inputs())
        self.runtime.release_fused_batch_get_context.side_effect = RuntimeError("outstanding DMA")
        with self.assertRaisesRegex(RuntimeError, "outstanding DMA"):
            runtime.shutdown()
        self.runtime.shutdown.assert_not_called()
        self.assertEqual(len(runtime.bindings), 1)
