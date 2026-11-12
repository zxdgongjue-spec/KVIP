"""Actual mtp_xzw_0903 MTP union/LIDU/descriptor operator, not four Q1 calls."""

import os

import kvip
from aiv1_fetch_support import PADDED_TOKENS, TOKEN_BYTES
from mtp_fixture import QUERIES, SOURCE, TOPK


class NativeMtp:
    def __init__(self, torch, device, owner, fixture, *, source_tokens=SOURCE, padded_tokens=PADDED_TOKENS):
        if source_tokens < TOPK or padded_tokens < source_tokens or padded_tokens % 128:
            raise ValueError("native_mtp_invalid_source_geometry")
        self.source_tokens, self.padded_tokens = source_tokens, padded_tokens
        self.torch, self.owner, self.fixture = torch, owner, fixture
        self.batch, self.hot = owner.batch, owner.resident_budget
        # Test-only diagnostic. Profile status is stored in column 9 of
        # each request's lane-0 profile; mode=1 can perturb timings.
        self.diagnose = os.environ.get("KVAF_NATIVE_MTP_DIAG", "0") == "1"
        self.initial = owner.cache_map.clone()
        self.forward = self.initial.clone()
        self.reverse_initial = torch.arange(self.hot, dtype=torch.int32, device=device).repeat(self.batch, 1)
        for row, mapping in enumerate(self.initial.cpu().tolist()):
            inverse = [0] * self.hot
            for token, slot in enumerate(mapping):
                if slot >= 0:
                    inverse[slot] = token
            self.reverse_initial[row] = torch.tensor(inverse, dtype=torch.int32, device=device)
        self.reverse = self.reverse_initial.clone()
        self.cursors = torch.zeros(self.batch, dtype=torch.int32, device=device)
        self.entries = torch.arange(self.batch, dtype=torch.int32, device=device)
        self.budgets = torch.full((self.batch,), self.hot, dtype=torch.int32, device=device)
        self.preferred = torch.arange(TOPK, dtype=torch.int32, device=device).repeat(self.batch, 1)
        bitmap = torch.zeros((self.batch, 512), dtype=torch.int32)
        bitmap[:, : TOPK // 32] = -1
        self.bitmap = bitmap.to(device)
        self.block_table = torch.arange(self.batch * self.hot // 128, dtype=torch.int32, device=device).reshape(
            self.batch, -1
        )
        self.enabled = torch.full(
            (1,), 1 if self.diagnose else 0, dtype=torch.int32, device=device
        )
        self.destinations = torch.empty((self.batch * QUERIES, 1, TOPK), dtype=torch.int32, device=device)
        self.counts = torch.empty(self.batch + 1, dtype=torch.int32, device=device)
        self.indices = torch.empty(self.batch * QUERIES * TOPK, dtype=torch.int32, device=device)
        self.offsets = torch.empty_like(self.indices)
        self.outputs = torch.empty_like(self.indices)
        self.resident_lengths = torch.empty(self.batch, dtype=torch.int32, device=device)
        self.profiling = (
            torch.zeros((self.batch, 8, 32), dtype=torch.int64, device=device)
            if self.diagnose
            else torch.empty((self.batch, 8, 32), dtype=torch.int64, device=device)
        )
        self.initial_payload = (
            torch.stack(
                [owner.session.cpu_payload[row, ids] for row, ids in enumerate(self.reverse_initial.cpu().long())]
            )
            .reshape(-1, TOKEN_BYTES)
            .to(device)
        )
        self.output = self.initial_payload.clone()
        self.total = self.counts[self.batch :]

    def reset(self):
        self.forward.copy_(self.initial)
        self.reverse.copy_(self.reverse_initial)
        self.cursors.zero_()
        self.output.copy_(self.initial_payload)
        if self.diagnose:
            self.profiling.zero_()
            self.counts.fill_(-999)

    @staticmethod
    def _counts_invalid(counts, batch):
        return any(value < 0 or value > QUERIES * TOPK for value in counts[:batch]) or (
            counts[batch] != sum(counts[:batch])
        )

    def _invalid_counts_error(self, counts):
        if not self.diagnose:
            return ValueError(f"native_mtp_counts_invalid:{counts}")
        profile = self.profiling.detach().cpu()
        # Kernel PROFILE_STATUS is field 9; status is only published if the
        # global Query Layout gate accepts all request rows. All-zero profile
        # with negative counts suggests an early global validation failure.
        statuses = profile[:, 0, 9].tolist()
        codes = {
            0: "STATUS_OK_OR_NOT_PROFILED",
            -100: "QUERY_LAYOUT",
            -101: "ROW_METADATA",
            -102: "DUPLICATE_POOL_ROW",
            -103: "TOPK_TOKEN",
            -105: "COUNT_ACCOUNTING",
            -106: "UNIQUE_MISS_CAPACITY",
            -107: "INSUFFICIENT_EVICTION_SLOTS",
            -109: "RESOLVED_SLOT",
            -110: "DESTINATION_SLOT",
            -111: "RESIDENT_BLOCK",
        }
        details = {
            "counts": counts,
            "native_profile_status": statuses,
            "status_names": [codes.get(v, "UNKNOWN") for v in statuses],
            "actual_query_cumulative": self.fixture["actual_query"].detach().cpu().tolist(),
            "logical_and_candidate_lengths": self.fixture["actual_key"].detach().cpu().tolist(),
            "resident_budgets": self.budgets.detach().cpu().tolist(),
            "request_pool_rows": self.entries.detach().cpu().tolist(),
            "selected_shape": list(self.destinations.shape),
            "source_capacity": self.source_tokens,
            "resident_capacity": self.hot,
            "note": "Diagnostic profile mode=1 changes timing; statuses can be zero if the global gate exits before WriteProfile.",
        }
        return ValueError(f"native_mtp_counts_invalid:diagnostics={details}")

    def launch(self, selected):
        self.prepare(selected)
        if self.diagnose:
            # Do not invoke KVAF BatchGet if native LIDU has published -1
            # counts; inspect the original kernel failure first.
            self.torch.npu.synchronize()
            counts = self.counts.detach().cpu().tolist()
            if self._counts_invalid(counts, self.batch):
                raise self._invalid_counts_error(counts)
        self.fetch()

    def prepare(self, selected):
        if self.owner.lease is not None:
            raise ValueError("native_mtp_requires_unleased_batch_get_queues")
        kvip.lidu(
            selected,
            self.fixture["actual_query"],
            self.entries,
            self.forward,
            self.reverse,
            self.cursors,
            self.budgets,
            self.fixture["actual_key"],
            self.preferred,
            self.bitmap,
            self.block_table,
            self.fixture["actual_key"],
            self.enabled,
            self.destinations,
            self.counts,
            self.indices,
            self.offsets,
            self.outputs,
            self.resident_lengths,
            self.profiling,
            0,
            1,
            self.batch,
            self.padded_tokens // 128,
        )

    def fetch(self):
        if self.owner.lease is not None:
            raise ValueError("native_mtp_requires_unleased_batch_get_queues")
        self.owner.session.batch_get_attempted = True
        self.owner.session.runtime.batch_get(
            self.owner.session.device_table.addresses,
            self.owner.session.device_table.valid,
            self.indices,
            self.offsets,
            self.total,
            self.outputs,
            self.output,
            self.torch.npu.current_stream(),
            False,
        )

    def verify(self, selected):
        ids = selected.detach().cpu().reshape(self.batch, QUERIES * TOPK).long()
        # Native MTP compacts hits/misses and does not preserve query input
        # order in destinationSlots. Resolve selected tokens through its
        # updated forward map, then check the exported slot set separately.
        exported = self.destinations.cpu().reshape(self.batch, QUERIES, TOPK).long()
        forward = self.forward.cpu().long()
        output = self.output.cpu().reshape(self.batch, self.hot, TOKEN_BYTES)
        counts = self.counts.cpu().tolist()
        if self._counts_invalid(counts, self.batch):
            raise self._invalid_counts_error(counts)
        for row in range(self.batch):
            slots = forward[row, ids[row]]
            if bool(((ids[row] < 0) | (ids[row] >= self.source_tokens) | (slots < 0) | (slots >= self.hot)).any()):
                raise ValueError(f"native_mtp_location_range:row={row}")
            for q in range(QUERIES):
                expected_slots = slots[q * TOPK : (q + 1) * TOPK]
                if not self.torch.equal(expected_slots.sort().values, exported[row, q].sort().values):
                    raise ValueError(f"native_mtp_exported_slots_mismatch:row={row}:query={q}")
            actual = output[row, slots]
            expected = self.owner.session.cpu_payload[row, ids[row]]
            if not self.torch.equal(actual, expected):
                raise ValueError(f"native_mtp_payload_mismatch:row={row}")
        return counts
