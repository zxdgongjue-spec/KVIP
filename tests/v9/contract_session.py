"""V9-local KVAF owner using the existing runtime, two jettys and PUT policy."""

import sys
from pathlib import Path

import numpy as np
from contract_payload import payload
from kvop.v9.contract import TOKEN_BYTES, TOPK


def _base_class():
    from real_fetch_session import RealFetchSession

    return RealFetchSession


def _contract_payload(torch, batch, tokens):
    """Supply finite C8 data to the native session's single initialization PUT."""
    packed = np.stack([payload(row, np.arange(tokens)) for row in range(batch)])
    return torch.from_numpy(packed)


class ContractSession(_base_class()):
    @classmethod
    def open(
        cls,
        torch,
        device,
        geometry,
        initial_output,
        *,
        server,
        kvaf_root=None,
        workers=32,
        get_kernel=None,
        put_kernel=None,
        kvaf_python_dir=None,
    ):
        owner = cls(torch, device, geometry.max_requests * geometry.num_layers, geometry.qcap * 128, TOPK)
        owner.contract_geometry = geometry
        owner.geometry.update(output_tokens=geometry.output_tokens, output_bytes=geometry.output_tokens * TOKEN_BYTES)
        owner.initial_output = initial_output
        owner.output = initial_output.clone()
        try:
            owner._prepare_contract(server, get_kernel, put_kernel, kvaf_python_dir, kvaf_root, workers)
            return owner
        except BaseException:
            owner.close()
            raise

    def _prepare_contract(self, server, get_kernel, put_kernel, python_dir, root, workers):
        config = self.support.discover(
            server=server, get_kernel=get_kernel, put_kernel=put_kernel, kvaf_python_dir=python_dir, kvaf_root=root
        )
        loaded = sys.modules.get("kvaf")
        if loaded is not None:
            origin = getattr(loaded, "__file__", None)
            if origin is None or not Path(origin).resolve().is_relative_to(config["kvaf_python_dir"]):
                raise ValueError("KVAF already imported from a different runtime")
        sys.path.insert(0, str(config["kvaf_python_dir"]))
        torch, device = self.torch, self.device
        placeholders = torch.zeros((self.batch, TOPK), dtype=torch.int32, device=device)
        counts = torch.zeros(self.batch, dtype=torch.int32, device=device)
        entries = torch.arange(self.batch, dtype=torch.int32, device=device)
        budgets = torch.full((self.batch,), TOPK, dtype=torch.int32, device=device)
        lengths = torch.full((self.batch,), self.history, dtype=torch.int32, device=device)
        # A remote row is (pool_row, layer), never packed batch index.
        # This retains V9's real runtime, 128-block PUT and two-jetty lease.
        self.session = self.support._session_class().open(
            torch,
            device,
            self.batch,
            self.history,
            TOPK,
            config["server"],
            config["get_kernel"],
            config["put_kernel"],
            placeholders,
            placeholders,
            counts,
            entries,
            budgets,
            lengths,
            temporary_only=True,
            put_chunk_blocks=128,
            npu_core_count=workers,
            payload_factory=_contract_payload,
        )
        # The native owner has already PUT the final C8 payload and published
        # its access table. Retain that seed/table and do not PUT the keys again.
        g = self.contract_geometry
        raw = self.support.unsigned_address_view(self.session.device_table.addresses, torch)
        valid = self.session.device_table.valid
        capacity = g.max_requests * g.num_layers * g.qcap
        if raw.numel() < capacity or valid.numel() < capacity:
            raise ValueError("KVAF access table is smaller than the directory geometry")
        self.ddr_block_addresses = torch.zeros(g.table_capacity, dtype=torch.int64, device=device)
        self.ddr_valid = torch.zeros(g.table_capacity, dtype=torch.uint8, device=device)
        self.ddr_block_addresses[:capacity].copy_(raw.reshape(-1)[:capacity].view(torch.int64))
        self.ddr_valid[:capacity].copy_(valid.reshape(-1)[:capacity].to(torch.uint8))
        self.keepalive.extend(
            (
                placeholders,
                counts,
                entries,
                budgets,
                lengths,
                raw,
                valid,
                self.initial_output,
                self.output,
                self.ddr_block_addresses,
                self.ddr_valid,
            )
        )
        torch.npu.synchronize()

    def seed_resident(self, arrays):
        """Install mapped resident payload and tail before leasing; never timed."""
        if self.lease is not None or self.in_flight:
            raise ValueError("seed requires an unleased idle owner")
        g = self.contract_geometry
        cpu = self.initial_output.cpu().reshape(-1, TOKEN_BYTES)
        payload = self.session.cpu_payload
        for b, p in enumerate(arrays["pool_rows"]):
            remote_row = int(p) * g.num_layers + g.layer_id
            budget = int(arrays["budgets"][b])
            for slot, token in enumerate(arrays["reverse"][int(p), :budget]):
                if token >= 0:
                    physical = int(arrays["resident_table"][b, slot // 128]) * 128 + slot % 128
                    cpu[physical].copy_(payload[remote_row, int(token)])
            candidate, logical = int(arrays["candidate_lens"][b]), int(arrays["logical_lens"][b])
            for token in range(candidate, logical):
                slot = budget + token - candidate
                physical = int(arrays["resident_table"][b, slot // 128]) * 128 + slot % 128
                cpu[physical].copy_(payload[remote_row, token])
        self.initial_output.copy_(cpu.reshape(self.initial_output.shape).to(self.device))
        self.output.copy_(self.initial_output)

    def payload(self, pool, tokens):
        row = int(pool) * self.contract_geometry.num_layers + self.contract_geometry.layer_id
        return self.session.cpu_payload[row, np.asarray(tokens, dtype=np.int64)].numpy()
