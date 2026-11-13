"""Exercise the URMA constructor and packed ABI with CPU-backed NPU tensors."""

import ctypes
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from kvip.operators.lifetch import binding as urma
from kvip.operators.lifetch.geometry import final_aivs, li_geometry
from kvip.operators.lifetch.tiling import RuntimeShape


@dataclass(frozen=True)
class Device:
    type: str = "npu"
    index: int = 0


class Tensor:
    def __init__(self, values, device=Device()):
        self.values = np.asarray(values)
        self.device = device

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def shape(self):
        return self.values.shape

    def is_contiguous(self):
        return self.values.flags.c_contiguous

    def data_ptr(self):
        return self.values.ctypes.data

    def numel(self):
        return self.values.size

    def element_size(self):
        return self.values.itemsize

    def stride(self, index):
        return self.values.strides[index] // self.element_size()

    def __getitem__(self, key):
        return Tensor(self.values[key], self.device)

    def reshape(self, *shape):
        return Tensor(self.values.reshape(*shape), self.device)

    def to(self, device):
        self.device = device
        return self

    def fill_(self, value):
        self.values.fill(value)
        return self

    def zero_(self):
        return self.fill_(0)

    def copy_(self, other):
        self.values[...] = other.values
        return self


def fixture(monkeypatch, batch=2):
    history, resident, heads = 4096, 2048, 32
    torch = SimpleNamespace(
        int32=np.int32,
        int64=np.int64,
        uint8=np.uint8,
        uint64=np.uint64,
        from_numpy=Tensor,
        tensor=lambda values, dtype, device: Tensor(np.asarray(values, dtype=dtype), device),
        zeros=Mock(side_effect=lambda size, dtype, device: Tensor(np.zeros(size, dtype=dtype), device)),
        ones=lambda size, dtype, device: Tensor(np.ones(size, dtype=dtype), device),
        full=lambda size, value, dtype, device: Tensor(np.full(size, value, dtype=dtype), device),
    )
    arguments = [Tensor(np.zeros(32, dtype=np.uint8)) for _ in range(14)]
    arguments[0] = Tensor(np.zeros((batch, 4, heads, 128), dtype=np.uint8))

    def prepare(pointer):
        s = ctypes.cast(pointer, ctypes.POINTER(RuntimeShape)).contents
        g = li_geometry(s.batch, "auto", s.rounds, s.aicCount)
        s.liCores = g["rotating_aic"]
        s.rounds, s.serviceLanes = g["intermediate_fetches"], g["service_aivs_per_window"]
        s.finalWorkers = final_aivs(s.batch, "auto", s.aicCount)
        s.finalCores, s.serviceAics = s.batch * s.finalWorkers, s.serviceLanes // 2
        # Nonzero sentinels prove that the entire 20-word shape precedes the
        # new pointer; the appended field must not overwrite its final words.
        s.postGroups, s.postThreads, s.postBank, s.batchedRows, s.scanTokens = 1, 512, 448, 1, 23808
        return 0

    library = SimpleNamespace(kvip_prepare_shape=Mock(side_effect=prepare))
    properties = {
        "token_bytes": "tokenBytes",
        "final_workers": "finalWorkers",
        "aic_count": "aicCount",
        "service_lanes": "serviceLanes",
        "rounds": "rounds",
        "li_cores": "liCores",
        "batch": "batch",
        "history": "history",
        "resident": "resident",
    }
    for name, attribute in properties.items():
        setattr(
            library,
            "v9_" + name,
            Mock(side_effect=lambda p, a=attribute: getattr(ctypes.cast(p, ctypes.POINTER(RuntimeShape)).contents, a)),
        )
    for name, value in {
        "abi_version": 13,
        "context_bytes": 336,
        "row_bytes": 1024,
        "simulation": 0,
        "final_replicated": 1,
        "init": 0,
        "li": 0,
    }.items():
        setattr(library, "v9_" + name, Mock(return_value=value))
    monkeypatch.setattr(urma, "load_lifetch_library", lambda path: library)
    metadata = dict(
        query_ends=Tensor(np.full((batch, 4), 3900, dtype=np.int32)),
        hot_valid=Tensor(np.ones((batch + 1, resident), dtype=np.int32)),
        row_owners=Tensor(np.arange(1, batch + 1, dtype=np.int32)),
        token_addresses=Tensor(np.zeros((batch + 1, history), dtype=np.int64)),
        tail_bounds=Tensor(np.tile([3840, 3900], (batch, 1)).astype(np.int32)),
        threshold_bounds=Tensor(np.tile([3712, 3900], (batch, 1)).astype(np.int32)),
        output_stride=resident + 384,
    )
    # The final packed row is padding. Bounds are persistent input, not reset
    # scratch; zero rows must remain zero and must still have a valid pointer.
    metadata["query_ends"].values[-1] = 0
    metadata["threshold_bounds"].values[-1] = 0
    owner = SimpleNamespace(
        abi_sizes={"status": 1216},
        core_status=Tensor(np.zeros((max(batch * 5, 2 * batch + 16), 1216), dtype=np.uint8)),
        build_device_args=Mock(return_value=Tensor(np.zeros((max(batch * 5, 2 * batch + 16), 128), dtype=np.uint8))),
    )

    def construct(**overrides):
        return urma.LIFetchBinding(
            torch,
            "library",
            arguments,
            owner,
            batch=batch,
            history=history,
            resident=resident,
            intermediate_fetches=8,
            **(metadata | overrides),
        )

    return SimpleNamespace(construct=construct, metadata=metadata, library=library, torch=torch)


@pytest.mark.parametrize("batch", [1, 2, 16])
def test_context_appends_retained_threshold_range(monkeypatch, batch):
    bytes_, shape_offset, tail_offset, threshold_offset = 336, 224, 216, 320
    f = fixture(monkeypatch, batch)
    bound = f.construct()
    blob = bound.context.values.tobytes()
    assert len(blob) == bytes_
    assert ctypes.sizeof(RuntimeShape) == 80
    assert blob[shape_offset : shape_offset + 80] == bytes(bound.runtime_shape)

    def pointer(offset):
        return int.from_bytes(blob[offset : offset + 8], "little")

    assert pointer(threshold_offset) == f.metadata["threshold_bounds"].data_ptr()
    assert pointer(tail_offset) == f.metadata["tail_bounds"].data_ptr()
    assert pointer(threshold_offset) != pointer(tail_offset)
    assert bound.threshold_bounds is f.metadata["threshold_bounds"]
    assert bound.query_ends.shape == (batch, 4)
    assert bound.hot_valid.shape[0] == batch + 1  # Stable-owner guard row is independent of B.
    np.testing.assert_array_equal(bound.threshold_bounds.values[-1], [0, 0])
    f.metadata["threshold_bounds"].values[0] = [3600, 3900]
    bound.reset()
    np.testing.assert_array_equal(bound.threshold_bounds.values[0], [3600, 3900])
    baseline = f.construct(threshold_bounds=None)
    assert int.from_bytes(baseline.context.values[threshold_offset:].tobytes(), "little") == 0
    assert baseline.threshold_bounds is None


@pytest.mark.parametrize("bad", ["shape", "dtype", "stride", "device"])
def test_threshold_range_rejects_wrong_tensor_contract(monkeypatch, bad):
    f = fixture(monkeypatch)
    tensor = f.metadata["threshold_bounds"]
    if bad == "shape":
        tensor = Tensor(np.zeros((2, 4), dtype=np.int32))
    elif bad == "dtype":
        tensor = Tensor(np.zeros((2, 2), dtype=np.int64))
    elif bad == "stride":
        tensor = Tensor(np.zeros((2, 4), dtype=np.int32)[:, ::2])
    else:
        tensor = Tensor(tensor.values, Device(index=1))
    with pytest.raises(ValueError, match="contiguous int32"):
        f.construct(threshold_bounds=tensor)
    f.library.v9_init.assert_not_called()
    f.library.v9_li.assert_not_called()


@pytest.mark.parametrize("field,value", [("v9_abi_version", 12), ("v9_context_bytes", 328)])
def test_old_native_abi_rejected_before_scratch_allocation(monkeypatch, field, value):
    f = fixture(monkeypatch)
    getattr(f.library, field).return_value = value
    with pytest.raises(ValueError, match="binary contract"):
        f.construct()
    f.torch.zeros.assert_not_called()


@pytest.mark.parametrize("batch", [17, 32])
def test_binding_batch_limit_still_precedes_allocation(monkeypatch, batch):
    f = fixture(monkeypatch, batch)
    with pytest.raises(ValueError, match="BS1..16"):
        f.construct()
    f.torch.zeros.assert_not_called()
    f.library.kvip_prepare_shape.assert_not_called()


def test_admission_controls_are_validated_and_keep_a_stable_address(monkeypatch):
    f = fixture(monkeypatch)
    values = np.array([[512, 64, 0, 0], [2048, 480, 65535, 1]], dtype=np.int32)
    bound = f.construct(admission_controls=values)
    pointer = bound.admission_controls.data_ptr()
    assert int.from_bytes(bound.context.values[328:336].tobytes(), "little") == pointer
    np.testing.assert_array_equal(bound.admission_controls.values, values)
    values[0] = [0, 1, 2, 1]
    bound.set_admission_controls(values)
    assert bound.admission_controls.data_ptr() == pointer
    bound.reset()
    np.testing.assert_array_equal(bound.admission_controls.values, values)
    bound.set_admission_controls(None)
    assert bound.admission_controls is None
    assert not bound.context.values[328:336].any()
    bound.set_admission_controls(values)
    assert bound.admission_controls.data_ptr() == pointer
    np.testing.assert_array_equal(bound.admission_controls.values, values)


@pytest.mark.parametrize("column,value", [(0, -1), (0, 2049), (1, 0), (1, 481), (2, -1), (2, 65536), (3, -1), (3, 2)])
def test_admission_rejects_invalid_bounds_before_allocation(monkeypatch, column, value):
    f = fixture(monkeypatch)
    controls = np.tile(np.array([512, 64, 0, 0], dtype=np.int32), (2, 1))
    controls[0, column] = value
    with pytest.raises(ValueError, match="admission bounds"):
        f.construct(admission_controls=controls)
    f.torch.zeros.assert_not_called()


@pytest.mark.parametrize("bad", [np.zeros((2, 4), dtype=np.int64), np.zeros((2, 3), dtype=np.int32),
                                 np.zeros((2, 8), dtype=np.int32)[:, ::2], Tensor(np.zeros((2, 4), np.int32))])
def test_admission_rejects_casts_and_hidden_device_readback(monkeypatch, bad):
    f = fixture(monkeypatch)
    bound = f.construct()
    original = bound.context.values.copy()
    with pytest.raises(ValueError, match="CPU int32"):
        bound.set_admission_controls(bad)
    np.testing.assert_array_equal(bound.context.values, original)
    assert bound.admission_controls is None


def test_admission_cannot_update_during_capture(monkeypatch):
    f = fixture(monkeypatch)
    bound = f.construct()
    f.torch.npu = SimpleNamespace(is_current_stream_capturing=lambda: True)
    with pytest.raises(RuntimeError, match="outside graph capture"):
        bound.set_admission_controls(np.tile(np.array([512, 64, 0, 0], np.int32), (2, 1)))
    assert bound.admission_controls is None
