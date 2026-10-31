"""Optional downstream QSFA numeric acceptance, outside fused timing."""

import numpy as np
from contract_payload import payload, unpack
from kvip.v9.contract import TOPK


def check_qsfa(torch, tensors, geometry, before, after):
    # The framework Python wrapper is registered only during model startup.
    # Standalone validation loads vllm_ascend_C and calls the same native op.
    op = getattr(torch.ops._C_ascend, "npu_dsa_a5_sparse_tail_attention_c8", None)
    if op is None:
        return {"status": "skipped", "reason": "native npu_dsa_a5_sparse_tail_attention_c8 is not registered"}
    g = geometry
    rng = np.random.default_rng(181 + g.layer_id)
    query = torch.from_numpy(rng.uniform(-0.2, 0.2, (g.batch * g.queries, 8, 576)).astype(np.float32)).to(
        torch.bfloat16
    )
    qcpu = query.float().numpy()
    tail = int(before["logical_lens"][0] - before["candidate_lens"][0])
    slots = np.full((g.batch * g.queries, 1, TOPK + tail), -1, np.int32)
    selected_lengths = np.zeros(g.batch * g.queries, np.int32)
    expected = np.empty((g.batch * g.queries, 8, 512), np.float32)
    exported = after["destination"].reshape(g.batch, g.queries, TOPK)
    scale = 576**-0.5
    for b, p in enumerate(before["pool_rows"]):
        c, budget = int(before["candidate_lens"][b]), int(before["budgets"][b])
        for q in range(g.queries):
            row = b * g.queries + q
            valid = exported[b, q][exported[b, q] >= 0]
            # Attention metadata has a packed selected prefix and a contiguous
            # per-query visible tail. The full resident length remains S+tail.
            visible_tail = max(0, tail - g.queries + q + 1)
            tail_slots = np.arange(budget, budget + visible_tail, dtype=np.int32)
            all_slots = np.concatenate((valid, tail_slots))
            slots[row, 0, : len(all_slots)] = all_slots
            selected_lengths[row] = len(all_slots)
            tokens = np.concatenate((after["reverse"][p, valid], np.arange(c, c + visible_tail)))
            key, value = unpack(payload(int(p) * g.num_layers + g.layer_id, tokens))
            logits = qcpu[row] @ key.T * scale
            if len(tokens):
                logits -= logits.max(axis=1, keepdims=True)
                prob = np.exp(logits)
                prob /= prob.sum(axis=1, keepdims=True)
                # Native QSFA accumulates a BF16 probability/value product.
                prob = torch.from_numpy(prob).to(torch.bfloat16).float().numpy()
                expected[row] = prob @ value
            else:
                expected[row].fill(0)
    device = tensors["query"].device
    actual = op(
        query.to(device),
        tensors["resident_output"].reshape(g.resident_blocks, 128, 1, 656).view(torch.float8_e4m3fn),
        torch.from_numpy(slots).to(device),
        scale,
        tensors["resident_table"].repeat_interleave(g.queries, dim=0),
        torch.arange(1, g.batch * g.queries + 1, dtype=torch.int32, device=device),
        tensors["resident_lengths"].repeat_interleave(g.queries),
        torch.from_numpy(selected_lengths).to(device),
    )
    actual = actual[0] if isinstance(actual, (tuple, list)) else actual
    actual = actual.float().cpu().numpy().reshape(expected.shape)
    if not np.all(np.isfinite(actual)) or not np.allclose(actual, expected, atol=0.006, rtol=0.025):
        raise AssertionError(f"QSFA numerical mismatch: max_abs={float(np.max(np.abs(actual - expected)))}")
    return {
        "status": "pass",
        "max_abs": float(np.max(np.abs(actual - expected))),
        "max_rel": float(np.max(np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-6))),
        "heads": 8,
        "atol": 0.006,
        "rtol": 0.025,
        "tail": tail,
    }
