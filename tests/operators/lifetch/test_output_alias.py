# SPDX-License-Identifier: Apache-2.0
"""Service feedback views share native output storage without input aliasing."""

import gc
import importlib
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import weakref

import pytest
import torch

from kvip.operators.lifetch.index_update import DirectIndexUpdate
from kvip.operators.lifetch_ubmem.index_update import UbmemIndexUpdate


def fixture():
    batch, owners, history, resident = 2, 3, 256, 128
    outputs = torch.zeros((2, batch * 4, 1, 2048), dtype=torch.int32)
    rotating = NS(
        shape=NS(batch=batch, history=history, resident=resident),
        arguments=[torch.empty(1) for _ in range(14)],
        locations=torch.empty((batch, 4, 2048), dtype=torch.int32),
        change_slots=torch.empty((batch, 10240), dtype=torch.int32),
        changes=torch.empty((batch, 10240), dtype=torch.int32),
        change_count=torch.empty(batch, dtype=torch.int32),
        rows=torch.empty((batch, 256), dtype=torch.int32),
        hot_valid=torch.empty((owners, resident), dtype=torch.int32),
        row_status=torch.empty((batch, 32), dtype=torch.int32),
        final_posts=torch.empty((batch, 4, 32), dtype=torch.int32),
        prefetch_status=torch.empty((batch * 2, 1216), dtype=torch.uint8),
        final_status=torch.empty((batch * 2, 1216), dtype=torch.uint8),
        ubmem_status=torch.empty((batch, 16), dtype=torch.int64),
        region=torch.empty(128, dtype=torch.uint8),
        final_workers=4,
        owner=NS(abi_sizes={"status": 1216}),
        output=torch.empty((8 * 2048 * 4,), dtype=torch.uint8),
        lease=NS(bindings=[]),
        library=NS(kvip_ubmem_plan_words=Mock(return_value=40968)),
    )
    rotating.owner.output = rotating.output
    rotating.arguments[9] = torch.empty((batch, 4, 2048), dtype=torch.int32)
    return dict(
        keys=torch.empty((owners, history, 128), dtype=torch.uint8),
        scales=torch.empty((owners, history), dtype=torch.float32),
        addresses=torch.empty((owners, history), dtype=torch.int64),
        logical_ids=torch.empty((owners, history), dtype=torch.int64),
        row_owners=torch.tensor([2, 1], dtype=torch.int32),
        rotating=rotating,
        status=torch.empty(batch, dtype=torch.int32),
        query_inputs=NS(unpack_indices=torch.arange(batch * 4), output_valid=torch.ones(batch * 4, dtype=torch.int64)),
        outputs=outputs,
    )


def make(transport, inputs, **options):
    if transport == "ubmem":
        return UbmemIndexUpdate(torch, **inputs, **options)
    library = NS(
        kvop_direct_update_abi_version=Mock(return_value=3),
        kvop_direct_update_plan_words=Mock(return_value=40968),
        kvop_direct_update_work_words=Mock(return_value=40960),
        kvop_direct_update_outputs=Mock(return_value=0),
    )
    with patch("kvip.operators.lifetch.index_update.ctypes.CDLL", return_value=library):
        return DirectIndexUpdate(torch, "test", **inputs, **options)


@pytest.mark.parametrize("transport", ("urma", "ubmem"))
def test_feedback_alias_binds_only_slot_pointer_and_retains_the_view(transport):
    inputs = fixture()
    allocation = torch.full((12, 1, 2048), 123, dtype=torch.int32)
    output = allocation[2:10]
    reference = weakref.ref(output)
    update = make(transport, inputs, slot_output=output)
    assert update.slot_output is output
    assert int(update.bindings[22]) == inputs["outputs"][0].data_ptr()
    assert int(update.bindings[23]) == output.data_ptr()
    # The native output pointer is the service feedback view itself. Writing
    # it changes feedback directly, while guard rows and logical output survive.
    update.slot_output.fill_(17)
    assert torch.all(allocation[2:10] == 17)
    assert torch.all(allocation[:2] == 123) and torch.all(allocation[10:] == 123)
    assert torch.all(update.logical_output == 0)
    del output
    gc.collect()
    assert reference() is update.slot_output


@pytest.mark.parametrize("transport", ("urma", "ubmem"))
def test_default_output_binding_is_unchanged(transport):
    inputs = fixture()
    update = make(transport, inputs)
    assert int(update.bindings[23]) == inputs["outputs"][1].data_ptr()


@pytest.mark.parametrize("transport", ("urma", "ubmem"))
@pytest.mark.parametrize(
    "kind", ("shape", "dtype", "device", "stride", "logical", "partial", "topk", "locations", "kv")
)
def test_invalid_or_overlapping_slot_outputs_are_rejected(transport, kind):
    inputs = fixture()
    output = {
        "shape": lambda: torch.empty((7, 1, 2048), dtype=torch.int32),
        "dtype": lambda: torch.empty((8, 1, 2048), dtype=torch.int64),
        "device": lambda: torch.empty((8, 1, 2048), dtype=torch.int32, device="meta"),
        "stride": lambda: torch.empty((8, 1, 4096), dtype=torch.int32)[:, :, ::2],
        "logical": lambda: inputs["outputs"][0],
        "partial": lambda: inputs["outputs"].reshape(-1)[1 : 8 * 2048 + 1].view(8, 1, 2048),
        "topk": lambda: inputs["rotating"].arguments[9].view(8, 1, 2048),
        "locations": lambda: inputs["rotating"].locations.view(8, 1, 2048),
        "kv": lambda: inputs["rotating"].output.view(torch.int32).view(8, 1, 2048),
    }[kind]()
    with pytest.raises(ValueError, match="Slot output"):
        make(transport, inputs, slot_output=output)


def test_standalone_update_cannot_accept_service_slot_output():
    inputs = fixture()
    inputs.pop("query_inputs")
    with pytest.raises(ValueError, match="service query inputs"):
        make("urma", inputs, slot_output=torch.empty((8, 1, 2048), dtype=torch.int32))


def test_slot_output_cannot_overlap_a_later_padded_address_row():
    inputs = fixture()
    backing = torch.empty((3, 8192), dtype=torch.int64)
    inputs["addresses"] = backing[:, :256]
    output = backing.view(torch.int32).reshape(-1)[8192 : 8192 + 8 * 2048].view(8, 1, 2048)
    with pytest.raises(ValueError, match="must not overlap"):
        make("urma", inputs, slot_output=output)


@pytest.mark.parametrize("transport", ("urma", "ubmem"))
def test_service_binder_passes_each_groups_existing_feedback_view(transport):
    module = importlib.import_module("kvip.adapters.vllm." + ("binding" if transport == "urma" else "ubmem_binding"))
    batch = 2
    workspace = NS(
        scratch={},
        inputs=NS(query_ends=torch.zeros(batch * 4), row_owners=torch.zeros(batch, dtype=torch.int32)),
        tail_bounds=object(),
        prefix_reset=object(),
        index_plan=object(),
        index_work=object(),
        arguments=lambda storage: (),
    )
    stores = [
        NS(history=256, **{name: object() for name in ("keys", "scales", "addresses", "logical_ids", "hot_valid")})
        for _ in range(2)
    ]
    runtime_type = module.LeasedRuntime if transport == "urma" else module.UbmemLeasedRuntime
    runtimes = {(width, 128): Mock(spec=runtime_type) for width in (656, 2624)}
    for runtime in runtimes.values():
        runtime.lease = object()
    policy = NS(
        pool=NS(device=torch.device("cpu")),
        fused_setup=NS(
            cann="/cann",
            headers="/include",
            soc="soc",
            batch=batch,
            windows=8,
            aics=32,
            shapes={656: object(), 2624: object()},
            libraries={656: object(), 2624: object()},
        ),
        fused_workspace=workspace,
        writer=NS(groups=[NS(geometry=(width, 128)) for width in (656, 2624)], runtimes=runtimes),
        indexer_selectors={},
        fused_index_storage=stores,
        bytes_outputs=[object(), object()],
        addresses=torch.empty(1),
        valid=torch.empty(1),
        fused_fetch_rows=torch.empty(2, batch, 256),
        spare_slots=torch.empty(2, batch, 2048),
        fused_reserved_hot=torch.empty(2, batch, 128),
        history_capacity=128,
        slot_capacity=512,
        fused_errors=torch.empty(2, batch),
        fused_outputs=torch.empty(2, 2, batch * 4, 1, 2048),
        destinations=torch.empty(2, batch * 4 + 3, 1, 2048, dtype=torch.int32),
        abc_descriptors=torch.empty(2, batch, 4),
    )
    config = NS(transport=transport, build_dir="/build", prefetch_tokens=512)
    binding_name = "LIFetchBinding" if transport == "urma" else "UbmemBinding"
    update_name = "DirectIndexUpdate" if transport == "urma" else "UbmemIndexUpdate"
    with (
        patch.object(module, binding_name),
        patch.object(module, update_name) as update,
        patch.object(module, "VllmIndexSelector"),
    ):
        if transport == "urma":
            with (
                patch.object(module, "build_libraries", return_value=("li", "update")),
                patch.object(module, "KvafQueueView"),
            ):
                module.bind_selectors(torch, policy, config)
        else:
            module.bind_selectors(torch, policy, config)
    assert update.call_count == 2
    for group, call in enumerate(update.call_args_list):
        view = call.kwargs["slot_output"]
        assert view.data_ptr() == policy.destinations[group].data_ptr()
        assert view.shape == (batch * 4, 1, 2048) and view.is_contiguous()
