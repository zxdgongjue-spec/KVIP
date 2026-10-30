"""Exercise the real V9/native initialization with a bounded KVAF test double."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from contract_fixture import make_fixture
from contract_payload import payload, unpack
from contract_session import ContractSession


class Tensor(np.ndarray):
    def __new__(cls, values):
        return np.asarray(values).view(cls)

    def to(self, target):
        return self.astype(target) if isinstance(target, type) else self.copy()

    def clone(self):
        return self.copy()

    def numel(self):
        return self.size

    def data_ptr(self):
        return self.ctypes.data

    def is_contiguous(self):
        return self.flags.c_contiguous

    def copy_(self, other):
        self[...] = other
        return self


@pytest.fixture
def transport(monkeypatch, tmp_path):
    events, runtimes = [], []

    class Runtime:
        def __init__(self, kernel, **options):
            self.options = options
            self.blocks = {}
            self.table_calls = 0
            self.chunks = []
            runtimes.append(self)

        def link_to(self, server):
            assert server == "test-server"

        def initialize(self, capacity_bytes, block_bytes):
            self.block_bytes = block_bytes
            self.capacity = capacity_bytes // block_bytes

        def batch_put(self, keys, seed, offsets, sizes):
            # Model an exactly sized pool. A replacement PUT cannot reserve
            # another set of slots after the initial payload fills the pool.
            if len(self.blocks) + len(keys) > self.capacity:
                raise RuntimeError("not enough free slots for PreparePut")
            assert not self.blocks.keys() & set(keys), "duplicate PreparePut"
            if self.chunks:
                assert events[-1] == "sync"
            self.chunks.append(len(keys))
            flat = seed.reshape(-1)
            for key, offset, size in zip(keys, offsets, sizes):
                assert size == self.block_bytes
                self.blocks[key] = flat[offset : offset + size].copy()
            events.append("put")

        def prepare_access_table(self, keys):
            assert events[-1] == "sync"
            assert keys == list(self.blocks)
            self.table_calls += 1
            return keys

        def access_table_view(self, table, *, device, capacity):
            assert capacity == len(table) == self.capacity
            return SimpleNamespace(
                addresses=Tensor(np.arange(capacity, dtype=np.uint64) * self.block_bytes + 0x100000000),
                valid=Tensor(np.ones(capacity, np.uint8)),
            )

        def batch_get(self):
            """Verified service binding with miss_count."""
            raise AssertionError("setup must not launch GET")

        def acquire_fused_batch_get_context(self):
            raise AssertionError("setup must not acquire")

        def release_fused_batch_get_context(self):
            raise AssertionError("setup must not release")

    monkeypatch.setitem(sys.modules, "kvaf", SimpleNamespace(NpuRuntime=Runtime, __file__=str(tmp_path / "kvaf.py")))
    kernel = tmp_path / "kernel.o"
    kernel.touch()
    config = dict(server="test-server", get_kernel=kernel, put_kernel=kernel, kvaf_python_dir=tmp_path)
    torch = SimpleNamespace(
        uint8=np.uint8,
        int32=np.int32,
        int64=np.int64,
        uint64=np.uint64,
        zeros=lambda shape, dtype, device=None: Tensor(np.zeros(shape, dtype)),
        full=lambda shape, value, dtype, device=None: Tensor(np.full(shape, value, dtype)),
        arange=lambda *args, dtype=None, device=None: Tensor(np.arange(*args, dtype=dtype)),
        stack=lambda rows: Tensor(np.stack(rows)),
        from_numpy=Tensor,
        npu=SimpleNamespace(synchronize=lambda: events.append("sync")),
    )
    return torch, config, runtimes


@pytest.mark.parametrize("batch,layers,candidate", [(1, 1, 16385), (2, 2, 2051)])
def test_contract_setup_puts_final_payload_once(monkeypatch, transport, batch, layers, candidate):
    torch, config, runtimes = transport
    g, arrays, _ = make_fixture(batch=batch, layers=layers, candidate=candidate, budget=12288)
    owner = ContractSession(torch, SimpleNamespace(index=0), g.max_requests * layers, g.qcap * 128, 2048)
    owner.contract_geometry = g
    owner.initial_output = Tensor(arrays["resident_output"])
    owner.output = owner.initial_output.clone()
    monkeypatch.setattr(owner.support, "discover", lambda **kwargs: config)

    # Production quarantine terminates the process. Make a regression observable
    # as a pytest failure instead, without permitting setup to continue.
    def fail_quarantine(*args):
        raise AssertionError(f"unexpected quarantine: {args}")

    session_class = owner.support._session_class()
    monkeypatch.setattr(session_class, "quarantine", fail_quarantine)
    monkeypatch.setattr(owner.support, "_session_class", lambda: session_class)
    monkeypatch.setattr(owner, "quarantine", fail_quarantine)
    monkeypatch.setattr(sys, "path", list(sys.path))
    owner._prepare_contract("test-server", None, None, None, None, workers=2)

    assert len(runtimes) == 1
    runtime = runtimes[0]
    capacity = g.max_requests * layers * g.qcap
    assert len(runtime.blocks) == runtime.capacity == capacity
    assert runtime.chunks == [min(128, capacity - begin) for begin in range(0, capacity, 128)]
    assert runtime.table_calls == 1
    assert runtime.options["jettys_per_core"] == 2
    assert runtime.options["npu_core_count"] == 2
    assert owner.session.seed.shape == (capacity, 128 * 656)
    remote = np.stack(list(runtime.blocks.values())).reshape(g.max_requests * layers, g.qcap * 128, 656)
    for row in range(g.max_requests * layers):
        assert np.array_equal(remote[row], payload(row, np.arange(g.qcap * 128)))
    assert np.array_equal(remote, owner.session.cpu_payload)
    assert np.isfinite(unpack(remote)[0]).all()
    assert np.array_equal(owner.ddr_block_addresses, owner.session.device_table.addresses)
    assert np.all(owner.ddr_valid == 1)


def test_native_default_payload_remains_unchanged(transport):
    torch, config, runtimes = transport
    owner = ContractSession(torch, SimpleNamespace(index=0), 1, 2048, 2048)
    placeholders = torch.zeros((1, 2048), dtype=np.int32)
    session = owner.support._session_class().open(
        torch,
        owner.device,
        1,
        2048,
        2048,
        config["server"],
        config["get_kernel"],
        config["put_kernel"],
        placeholders,
        placeholders,
        None,
        None,
        None,
        None,
        temporary_only=True,
        npu_core_count=2,
    )
    tokens = np.arange(2048)[:, None]
    columns = np.arange(656)[None, :]
    expected = ((tokens * 17 + (tokens >> 8) * 29 + columns * 7 + (columns >> 8)) % 256).astype(np.uint8)
    for part in range(4):
        expected[:, part::16] = (tokens >> (part * 8)) & 255
    expected[:, 4::16] = expected[:, 5::16] = 0
    assert np.array_equal(session.cpu_payload[0], expected)
    assert runtimes[0].table_calls == 1
