"""NPU-local transport stub. Needs NPU/CANN/headers, no KVAF runtime/server."""

import ctypes
from types import SimpleNamespace

import numpy as np
from contract_session import ContractSession, _contract_payload


class MemorySession(ContractSession):
    transport = "npu-memory-test"

    def build_device_args(self, library):
        abi = ctypes.CDLL(str(library))
        for name, expected in (("entity", 192), ("queue", 64)):
            fn = getattr(abi, f"refactor_fetch_{name}_bytes")
            fn.argtypes, fn.restype = [], ctypes.c_uint32
            if fn() != expected:
                raise ValueError(f"test transport {name} ABI changed")
        return super().build_device_args(library)

    def _prepare_contract(self, server, get_kernel, put_kernel, python_dir, root, workers):
        torch, g = self.torch, self.contract_geometry
        self.workers = workers
        cpu = _contract_payload(torch, self.batch, self.history)
        self.source_payload = cpu.to(self.device)
        self.session = SimpleNamespace(cpu_payload=cpu, close=lambda: None)
        count = g.max_requests * g.num_layers * g.qcap
        self.ddr_block_addresses = torch.zeros(g.table_capacity, dtype=torch.int64, device=self.device)
        self.ddr_block_addresses[:count] = (
            torch.arange(count, dtype=torch.int64, device=self.device) * (128 * 656) + self.source_payload.data_ptr()
        )
        self.ddr_valid = torch.zeros(g.table_capacity, dtype=torch.uint8, device=self.device)
        self.ddr_valid[:count].fill_(1)
        self.keepalive.extend((self.source_payload, self.ddr_block_addresses, self.ddr_valid))

    def acquire(self):
        if self.closed or self.lease is not None or self.in_flight:
            raise ValueError("invalid memory transport acquisition")
        torch, owners, depth = self.torch, self.workers * 2, 16384
        self.sq = torch.zeros((owners, depth, 16), dtype=torch.int32, device=self.device)
        self.queues = torch.zeros((owners, 64), dtype=torch.uint8, device=self.device)
        self.doorbells = torch.zeros((owners, 128), dtype=torch.uint8, device=self.device)
        entities = np.zeros((owners, 48), dtype=np.uint32)
        words64 = entities.view(np.uint64)
        for i in range(owners):
            entities[i, :2] = [0x4B564155, 1]
            words64[i, 1:3] = [self.source_payload.data_ptr(), self.source_payload.numel()]
            words64[i, 4:6] = [self.output.data_ptr(), self.output.numel()]
            entities[i, 12] = 1
            words64[i, 7:9] = [self.sq[i].data_ptr(), self.doorbells[i].data_ptr()]
            entities[i, 19] = depth
            entities[i, 21] = 64
            words64[i, 13:15] = [self.doorbells[i].data_ptr(), self.doorbells[i].data_ptr() + 64]
            entities[i, 31:33] = [depth, 64]
        self.entities = torch.from_numpy(entities.view(np.uint8)).to(self.device)
        self.keepalive.extend((self.sq, self.queues, self.doorbells, self.entities))
        self.lease = dict(
            worker_count=self.workers,
            jettys_per_core=2,
            device_channel_entity=self.entities.data_ptr(),
            device_queue_state=self.queues.data_ptr(),
            output_address=self.output.data_ptr(),
            output_bytes=self.output.numel(),
            token_bytes=656,
            tokens_per_block=128,
        )
        torch.npu.synchronize()
        return dict(self.lease)

    def release(self):
        if self.in_flight:
            self.quarantine("release before checked completion", "pending NPU copy")
        self.torch.npu.synchronize()
        self.lease = None

    def quarantine(self, reason, error):
        # Raw kernel launches also retain pointers on the memory test path.
        # Use the same process boundary as the real owner on unknown completion.
        self.support.quarantine_all(self, reason, error)
