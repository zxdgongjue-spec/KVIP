"""Native LI + prepare + KVAF baseline using the same caller snapshot."""

import numpy as np
from contract_payload import payload
from contract_reference import frozen_hot, verify_selection
from kvop.v9.contract import TOPK


class NativeBaseline:
    def __init__(self, torch, torch_npu, owner, geometry, tensors):
        self.torch, self.torch_npu, self.owner = torch, torch_npu, owner
        self.g = geometry
        self.t = {name: value.clone() for name, value in tensors.items()}
        self.t["resident_output"] = owner.output
        device = owner.device
        self.indices = torch.empty(
            geometry.batch * (TOPK if geometry.queries == 1 else 8192), dtype=torch.int32, device=device
        )
        self.offsets, self.outputs = torch.empty_like(self.indices), torch.empty_like(self.indices)
        # Keep the exact registered allocation/view alive through graph replay.
        self.output_rows = owner.output.reshape(-1, 656)
        self.graph_zero_count = torch.zeros_like(self.t["miss_counts"][-1:])
        self.graph_output_prewarmed = False
        self.profiling = torch.zeros((geometry.batch, 8, 32), dtype=torch.int64, device=device)

    def prewarm_graph_output(self):
        """Register caller output outside capture with a synchronous zero-count GET.

        Direct-HCCP retains the registration until transport shutdown. Capture
        must reuse it: first use also allocates/copies channel entities, which
        cannot safely run inside graph capture. No LI/prepare or fixture maps
        are touched here, and this setup is outside benchmark timing.
        """
        if self.graph_output_prewarmed:
            return
        for tensor in (self.indices, self.offsets, self.outputs, self.graph_zero_count):
            tensor.zero_()
        self.owner.session.batch_get_attempted = True
        self.owner.session.runtime.batch_get(
            self.owner.ddr_block_addresses,
            self.owner.ddr_valid,
            self.indices,
            self.offsets,
            self.graph_zero_count,
            self.outputs,
            self.output_rows,
            self.torch.npu.current_stream(self.owner.device),
            True,
        )
        # Do not mark a failed registration as ready for capture.
        self.graph_output_prewarmed = True

    def restore(self, before):
        for name in before:
            if name not in self.t or name in ("address_table", "valid_table"):
                continue
            source = self.torch.from_numpy(before[name])
            if name in ("query", "key"):
                source = source.view(self.torch.float8_e4m3fn)
            self.t[name].copy_(source.to(self.owner.device, dtype=self.t[name].dtype))

    def launch(self, phase_events=None):
        t, g = self.t, self.g
        result = self.torch_npu.npu_quant_lightning_indexer(
            query=t["query"],
            key=t["key"],
            weights=t["weights"],
            query_dequant_scale=t["query_scale"],
            key_dequant_scale=t["key_scale"],
            actual_seq_lengths_query=t["cumulative_query"],
            actual_seq_lengths_key=t["candidate_lens"],
            block_table=t["index_table"],
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=TOPK,
            sparse_mode=3,
        )
        if phase_events is not None:
            phase_events[0].record(self.torch.npu.current_stream(self.owner.device))
        selected = result[0] if isinstance(result, (tuple, list)) else result
        # Match original V9: pass the LI output directly to prepare.
        # Keep the view alive for graph replay and post-run verification.
        t["topk"] = selected.reshape_as(t["topk"])
        prefix = [t["topk"]]
        if g.queries > 1:
            prefix.append(t["cumulative_query"])
        args = prefix + [
            t[name]
            for name in (
                "pool_rows",
                "forward",
                "reverse",
                "cursors",
                "budgets",
                "candidate_lens",
                "victims",
                "victim_bitmap",
                "resident_table",
                "logical_lens",
                "profile",
                "destination",
                "miss_counts",
            )
        ]
        args += [
            self.indices,
            self.offsets,
            self.outputs,
            t["resident_lengths"],
            self.profiling,
            g.layer_id,
            g.num_layers,
            g.max_requests,
            g.qcap,
        ]
        name = "lidu_fused_batch_get_prepare_mtp_out" if g.queries > 1 else "lidu_fused_batch_get_prepare_out"
        getattr(self.torch.ops.nanovllm_dsa, name)(*args)
        if phase_events is not None:
            phase_events[1].record(self.torch.npu.current_stream(self.owner.device))
        self.owner.session.batch_get_attempted = True
        self.owner.session.runtime.batch_get(
            self.owner.ddr_block_addresses,
            self.owner.ddr_valid,
            self.indices,
            self.offsets,
            t["miss_counts"][-1:],
            self.outputs,
            self.output_rows,
            self.torch.npu.current_stream(self.owner.device),
            False,
        )

    def verify(self, before, scores):
        g = self.g
        actual = {
            n: self.t[n].cpu().numpy()
            for n in ("topk", "forward", "reverse", "cursors", "destination", "miss_counts", "resident_output")
        }
        ids = actual["topk"].reshape(g.batch, g.queries, TOPK)
        dst = actual["destination"].reshape(g.batch, g.queries, TOPK)
        out = actual["resident_output"].reshape(-1, 656)
        counts = []
        for b, p in enumerate(before["pool_rows"]):
            verify_selection(ids[b], scores[b])
            c, budget = int(before["candidate_lens"][b]), int(before["budgets"][b])
            hot = frozen_hot(before["forward"][p], before["reverse"][p], before["victims"][b], c, budget)
            # Native prepare still eagerly invalidates every original victim.
            victims = set(map(int, before["victims"][b]))
            hot = {t: s for t, s in hot.items() if s not in victims}
            counts.append(len({int(t) for t in ids[b].flat if t >= 0 and t not in hot}))
            for q in range(g.queries):
                valid = ids[b, q][ids[b, q] >= 0]
                slots = actual["forward"][p, valid]
                if np.any((slots < 0) | (slots >= budget)):
                    raise AssertionError("native baseline returned an invalid resident slot")
                if set(dst[b, q][dst[b, q] >= 0]) != set(slots):
                    raise AssertionError("native baseline destination set mismatch")
                physical = before["resident_table"][b, slots // 128] * 128 + slots % 128
                if not np.array_equal(out[physical], payload(int(p) * g.num_layers + g.layer_id, valid)):
                    raise AssertionError("native baseline payload mismatch")
        if not np.array_equal(actual["miss_counts"], counts + [sum(counts)]):
            raise AssertionError("native baseline logical miss count mismatch")
        return {"status": "pass", "logical_misses": counts, "contract": "TND/sparse3"}
