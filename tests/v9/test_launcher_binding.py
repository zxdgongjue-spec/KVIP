"""Real ContractLaunch host binding against an explicitly CPU-only C shim.

The shim checks the ABI/exported symbols and records dispatch. It performs no
selection, device work, DMA, or communication; numerical/hardware acceptance
remains a separate qualification tier.
"""

import ctypes
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
from contract_fixture import make_fixture, to_device

from kvop.testing.owner import StubQueueOwner
from kvop.v9 import ContractLaunch, PreparedSession, Status, native_source
from kvop.v9.access import OUTPUT_FIELDS
from kvop.v9.contract import (
    GEOMETRY_FIELDS,
    POINTER_FIELDS,
    DeviceContract,
    allocate_workspace,
    validate_buffers,
)
from kvop.v9.geometry import final_aivs, li_geometry
from kvop.v9.index_cache import cache_bytes


@pytest.fixture(scope="module")
def host_library(tmp_path_factory):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.fail("host binding conformance requires a C++ compiler")
    g, _, _ = make_fixture(candidate=256, budget=2048, tail=0)
    execution = li_geometry(g.batch)
    metadata = dict(
        abi_version=13,
        contract_mode=1,
        contract_bytes=ctypes.sizeof(DeviceContract),
        context_bytes=136,
        row_bytes=1024,
        batch=g.batch,
        queries=g.queries,
        history=g.candidate_capacity,
        resident=g.resident_capacity,
        li_cores=execution["rotating_aic"],
        aic_count=32,
        simulation=0,
        rounds=execution["intermediate_fetches"],
        service_lanes=execution["service_aivs_per_window"],
        final_workers=final_aivs(g.batch),
        final_replicated=1,
    )
    getters = "\n".join(
        f'extern "C" uint32_t v9_{name}() {{ return {value}u'
        + (" - (bad_metadata == 1)" if name == "abi_version" else "")
        + "; }"
        for name, value in metadata.items()
    )
    source = (
        r"""
#include "contract_abi.h"
static uint32_t count = 0, failed_stage = 0, bad_metadata = 0;
static uint32_t stages[16];
static uintptr_t streams[16], argument_table = 0;
extern "C" void v9_host_reset() { count = failed_stage = bad_metadata = 0; argument_table = 0; }
extern "C" void v9_host_fail(uint32_t stage) { failed_stage = stage; }
extern "C" void v9_host_bad_metadata(uint32_t value) { bad_metadata = value; }
extern "C" uint32_t v9_host_count() { return count; }
extern "C" uint32_t v9_host_stage(uint32_t index) { return stages[index]; }
extern "C" uintptr_t v9_host_stream(uint32_t index) { return streams[index]; }
extern "C" uintptr_t v9_host_arguments() { return argument_table; }
extern "C" uint32_t v9_contract_offset(uint32_t index) {
#define OFFSET(name) offsetof(rotating::Contract, name),
    static const uint32_t offsets[] = {V9_CONTRACT_POINTERS(OFFSET) V9_CONTRACT_GEOMETRY(OFFSET)};
#undef OFFSET
    return offsets[index] + (bad_metadata == 2 && index == 0 ? 8 : 0);
}
static uint32_t record(uint32_t stage, void *arguments, void *stream) {
    argument_table = reinterpret_cast<uintptr_t>(arguments);
    stages[count] = stage; streams[count] = reinterpret_cast<uintptr_t>(stream); ++count;
    return stage == failed_stage ? 77 : 0;
}
#define STAGE(name, number) extern "C" uint32_t v9_##name(void *a, void *s) { return record(number, a, s); }
STAGE(init, 1) STAGE(li, 2) STAGE(wait, 3) STAGE(resolve, 4) STAGE(fetch, 5)
"""
        + getters
    )
    directory = tmp_path_factory.mktemp("kvop-host-shim")
    library = directory / ("host.dylib" if sys.platform == "darwin" else "host.so")
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-x",
            "c++",
            "-I",
            str(native_source()),
            "-o",
            str(library),
            "-",
        ],
        input=source,
        text=True,
        check=True,
        capture_output=True,
    )
    return library


class HostTorch:
    def __init__(self, torch, stream):
        self._torch = torch
        self.npu = SimpleNamespace(current_stream=lambda device: stream)

    def __getattr__(self, name):
        return getattr(self._torch, name)


class HostBindingOwner(StubQueueOwner):
    """CPU lifetime/queue-view provider, deliberately not a KVAF provider."""

    def __init__(self, torch, output, stream):
        super().__init__(output, stream)
        self.abi_sizes = {"status": 16}
        self.core_status = torch.zeros((2, 16), dtype=torch.uint8)
        self.contexts = torch.zeros((5, 136), dtype=torch.uint8)
        self.completions = 0

    def build_device_args(self, library):
        return self.contexts

    def mark_completed(self, *, completion_verified):
        super().mark_completed(completion_verified=completion_verified)
        self.completions += 1


@pytest.fixture
def binding_setup(monkeypatch, host_library):
    torch = pytest.importorskip("torch")
    g, arrays, _ = make_fixture(candidate=256, budget=2048, tail=0)
    tensors = to_device(torch, arrays, g, torch.device("cpu"))
    workspace = allocate_workspace(torch, g, "cpu")
    backing = torch.zeros(cache_bytes(g) + 127, dtype=torch.uint8)
    offset = -backing.data_ptr() % 128
    caches = [backing[offset : offset + cache_bytes(g)]]
    events = []
    stream = SimpleNamespace(
        device="cpu", npu_stream=0x1234, synchronize=lambda: events.append("sync")
    )
    owner = HostBindingOwner(torch, tensors["resident_output"], stream)
    # Retain the real metadata/range validator; adapt only its expected device
    # type. No fake Tensor class or weakened output schema is introduced.
    monkeypatch.setattr(
        "kvop.v9.launch.validate_buffers",
        lambda g, t, w: validate_buffers(g, t, w, device_type="cpu"),
    )
    library = ctypes.CDLL(str(host_library))
    library.v9_host_reset()
    for name in ("v9_host_stream", "v9_host_arguments"):
        getattr(library, name).restype = ctypes.c_size_t

    def prepare():
        return ContractLaunch(
            HostTorch(torch, stream),
            host_library,
            tensors,
            owner,
            g,
            workspace=workspace,
            index_caches=caches,
        )

    return SimpleNamespace(
        torch=torch,
        g=g,
        tensors=tensors,
        workspace=workspace,
        owner=owner,
        stream=stream,
        events=events,
        library=library,
        prepare=prepare,
    )


def test_real_launcher_packs_every_pointer_and_geometry_field_and_dispatches_four_stages(
    binding_setup,
):
    setup = binding_setup
    launcher = setup.prepare()
    descriptor = DeviceContract.from_buffer_copy(launcher.descriptor.numpy().tobytes())
    supplied = launcher.tensors | launcher.workspace
    for name in POINTER_FIELDS:
        assert getattr(descriptor, name) == supplied[name].data_ptr(), name
    for name in GEOMETRY_FIELDS:
        assert getattr(descriptor, name) == getattr(setup.g, name), name
    assert set(launcher.outputs) == set(OUTPUT_FIELDS)
    for name in OUTPUT_FIELDS:
        value = launcher.outputs[name]
        shape, dtype = setup.g.tensor_spec()[name]
        assert tuple(value.shape) == shape and str(value.dtype).split(".")[-1] == dtype
        assert value is setup.tensors[name]
    session = PreparedSession(launcher, stream=setup.stream)
    selection = session.enqueue()
    assert all(selection[name] is setup.tensors[name] for name in OUTPUT_FIELDS)
    library = setup.library
    assert library.v9_host_count() == 4
    assert [library.v9_host_stage(i) for i in range(4)] == [1, 2, 4, 5]
    assert [library.v9_host_stream(i) for i in range(4)] == [0x1234] * 4
    assert library.v9_host_arguments() == launcher.bindings.data_ptr()
    assert launcher.bindings.tolist() == [
        value.data_ptr() for value in launcher.arguments
    ]
    assert setup.events == []
    # Synthetic host status verifies completion plumbing, never kernel math.
    setup.tensors["status"].fill_(Status.OK)
    completion = session.complete()
    assert completion.drained and completion.success and setup.events == ["sync"]
    assert setup.owner.completions == 1 and not setup.owner.in_flight
    session.close()
    assert setup.owner.closed


@pytest.mark.parametrize(
    "mismatch,match", [(1, "binary abi_version"), (2, "offset mismatch")]
)
def test_real_launcher_refuses_mismatching_native_exports_before_dispatch(
    binding_setup, mismatch, match
):
    setup = binding_setup
    setup.library.v9_host_bad_metadata(mismatch)
    with pytest.raises(ValueError, match=match):
        setup.prepare()
    assert setup.library.v9_host_count() == 0 and not setup.owner.in_flight


@pytest.mark.parametrize("invalid", ["output", "queue_count", "output_schema"])
def test_real_launcher_rejects_owner_or_tensor_binding_mismatch(binding_setup, invalid):
    setup = binding_setup
    if invalid == "output":
        setup.owner.output = setup.owner.output.clone()
        match = "registered output"
    elif invalid == "queue_count":
        setup.owner.contexts = setup.owner.contexts[:4]
        match = "too few exclusive"
    else:
        setup.tensors["destination"] = setup.tensors["destination"].flatten()
        match = "destination"
    with pytest.raises(ValueError, match=match):
        setup.prepare()
    assert setup.library.v9_host_count() == 0


def test_real_native_return_code_stops_later_dispatch_and_quarantines_session(
    binding_setup,
):
    setup = binding_setup
    launcher = setup.prepare()
    session = PreparedSession(launcher, stream=setup.stream)
    setup.library.v9_host_fail(2)
    with pytest.raises(RuntimeError, match="quarantined.*li_launch_failed:77"):
        session.enqueue()
    assert setup.library.v9_host_count() == 2 and setup.owner.in_flight
    assert setup.owner.keepalive
    with pytest.raises(RuntimeError, match="cannot release"):
        session.close()


@pytest.mark.parametrize(
    "uncertain",
    [
        "pending_row",
        "pending_post",
        "queue_error",
        "pending_status",
        "communication_status",
    ],
)
def test_real_completion_refuses_unverified_rows_posts_and_status(
    binding_setup, uncertain
):
    setup = binding_setup
    launcher = setup.prepare()
    session = PreparedSession(launcher, stream=setup.stream)
    session.enqueue()
    if uncertain == "pending_row":
        launcher.rows[0, 211] = 1
    elif uncertain == "pending_post":
        launcher.final_posts[0, 0, 1] = 1
    elif uncertain == "queue_error":
        launcher.final_posts[0, 0, 2] = 1
    else:
        setup.tensors["status"].fill_(
            Status.PENDING if uncertain == "pending_status" else Status.COMMUNICATION
        )
    with pytest.raises(RuntimeError, match="quarantined"):
        session.complete()
    assert (
        setup.owner.in_flight and setup.owner.completions == 0 and setup.owner.keepalive
    )
