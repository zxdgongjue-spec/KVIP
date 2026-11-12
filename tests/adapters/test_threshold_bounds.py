"""Service binders retain per-group threshold ranges for both transports."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from kvip.adapters.kvaf import LeasedRuntime, UbmemLeasedRuntime
from kvip.adapters.vllm import binding


class ThresholdBindingTest(unittest.TestCase):
    def test_both_transports_bind_distinct_persistent_group_ranges(self):
        for transport in ("urma", "ubmem"):
            with self.subTest(transport=transport):
                batch, groups = 2, 2
                bounds = torch.arange(groups * (batch + 1) * 2, dtype=torch.int32).view(groups, batch + 1, 2)
                workspace = NS(
                    arguments=lambda storage: (),
                    inputs=NS(query_ends=torch.zeros(batch, 4, dtype=torch.int32), row_owners=object()),
                    tail_bounds=torch.zeros(batch, 2, dtype=torch.int32),
                )
                storage = NS(history=4096, hot_valid=object(), addresses=object())
                setup = NS(
                    batch=batch,
                    cann="cann",
                    headers="headers",
                    soc="soc",
                    aics=32,
                    windows=8,
                    shapes={656: object()},
                    libraries={656: object()},
                )
                runtime_type = UbmemLeasedRuntime if transport == "ubmem" else LeasedRuntime
                writer = NS(
                    groups=[NS(geometry=(656, 128)) for _ in range(groups)],
                    runtimes={(656, 128): runtime_type(NS())},
                )
                policy = NS(
                    fused_setup=setup,
                    fused_workspace=workspace,
                    writer=writer,
                    indexer_selectors={},
                    fused_index_storage=[storage] * groups,
                    bytes_outputs=[object()] * groups,
                    threshold_bounds=bounds,
                    history_capacity=2048,
                    slot_capacity=2432,
                    addresses=torch.zeros(1),
                    valid=torch.zeros(1),
                    pool=NS(device="cpu"),
                )
                config = NS(transport=transport, build_dir="unused", prefetch_tokens=512)
                with (
                    patch.object(binding, "build_libraries", return_value=(object(),)),
                    patch.object(binding, "KvafQueueView"),
                    patch.object(binding, "group_scratch", return_value={}),
                    patch.object(binding, "index_update_arguments", return_value={}),
                    patch.object(binding, "bind_group_selector"),
                    patch.object(binding, "DirectIndexUpdate"),
                    patch.object(binding, "UbmemIndexUpdate"),
                    patch.object(binding, "LIFetchBinding") as urma,
                    patch.object(binding, "UbmemBinding") as ubmem,
                ):
                    binding.bind_selectors(torch, policy, config)
                selected, unused = (ubmem, urma) if transport == "ubmem" else (urma, ubmem)
                unused.assert_not_called()
                self.assertEqual(selected.call_count, groups)
                for group, call in enumerate(selected.call_args_list):
                    value = call.kwargs["threshold_bounds"]
                    self.assertEqual(value.shape, (batch, 2))
                    self.assertEqual(value.dtype, torch.int32)
                    self.assertTrue(value.is_contiguous())
                    self.assertEqual(value.data_ptr(), bounds[group].data_ptr())
                    self.assertIs(call.kwargs["tail_bounds"], workspace.tail_bounds)
                    self.assertNotEqual(value.data_ptr(), workspace.tail_bounds.data_ptr())
                    bounds[group, 0, 0] = 1234 + group
                    self.assertEqual(int(value[0, 0]), 1234 + group)


if __name__ == "__main__":
    unittest.main()
