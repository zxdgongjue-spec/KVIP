import numpy as np

TOPK, BLOCK_TOKENS, TOKEN_BYTES = 2048, 128, 656


def check_sfa(torch, tensors, geometry, before, after):
    """Actual service C8 API, plus independent logical-token numerical oracle.

    Scope is attention on the LI-selected 2048 history tokens. The 384-slot
    tail is allocated/seeded but is not appended to sparse indices. This does
    not claim to validate a model's ABC scheduling or mixed prefill batches.
    """
    import torch_npu
    from contract_payload import payload, unpack

    g = geometry
    rng = np.random.default_rng(181)
    query = torch.from_numpy(rng.uniform(-0.2, 0.2, (g.batch * g.queries, 8, 576)).astype(np.float32))
    query = query.to(torch.bfloat16)
    qcpu = query.float().numpy()
    scale = 576**-0.5
    expected = np.empty((g.batch * g.queries, 8, 512), np.float32)
    indices = np.empty((g.batch * g.queries, 1, TOPK), np.int32)
    # Get the ordered LI -> slot mapping, never assume packed destination order.
    for b, pool in enumerate(before["pool_rows"]):
        for q in range(g.queries):
            row = b * g.queries + q
            tokens = after["topk"][row, 0]
            if np.any(tokens < 0):
                raise AssertionError("128K case must supply 2048 valid TopK indices")
            indices[row, 0] = after["forward"][pool, tokens]
            if np.any((indices[row] < 0) | (indices[row] >= before["budgets"][b])):
                raise AssertionError("LI token has no valid hot-slot mapping")
            key, value = unpack(payload(int(pool) * g.num_layers + g.layer_id, tokens))
            logits = qcpu[row] @ key.T * scale
            logits -= logits.max(axis=1, keepdims=True)
            prob = np.exp(logits)
            prob /= prob.sum(axis=1, keepdims=True)
            prob = torch.from_numpy(prob).to(torch.bfloat16).float().numpy()
            expected[row] = prob @ value
    device = tensors["query"].device
    kv = tensors["resident_output"].reshape(g.resident_blocks, BLOCK_TOKENS, 1, TOKEN_BYTES)
    kv = kv.view(torch.float8_e4m3fn)
    mapping = tensors["forward"].index_select(0, tensors["pool_rows"].long())
    mapped = torch.gather(mapping, 1, tensors["topk"].reshape(g.batch, -1).long())
    mapped = mapped.reshape(g.batch * g.queries, 1, TOPK)
    if not np.array_equal(mapped.cpu().numpy(), indices):
        raise AssertionError("device token-to-slot gather differs from host mapping")
    actual = torch_npu.npu_kv_quant_sparse_flash_attention(
        query=query.to(device),
        key=kv,
        value=kv,
        sparse_indices=mapped,
        scale_value=scale,
        sparse_block_size=1,
        block_table=tensors["resident_table"],
        actual_seq_lengths_query=tensors["cumulative_query"],
        actual_seq_lengths_kv=tensors["logical_lens"],
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        key_quant_mode=2,
        value_quant_mode=2,
        rope_head_dim=64,
    )
    actual = actual[0] if isinstance(actual, (tuple, list)) else actual
    actual = actual.float().cpu().numpy().reshape(expected.shape)
    error = float(np.max(np.abs(actual - expected)))
    if not np.all(np.isfinite(actual)) or not np.allclose(actual, expected, atol=0.006, rtol=0.025):
        raise AssertionError(f"A5 C8 SFA numerical mismatch: max_abs={error}")
    return {
        "status": "pass",
        "max_abs": error,
        "atol": 0.006,
        "rtol": 0.025,
        "heads": 8,
        "scope": "LI-selected history TopK; actual service C8 SFA API; logical-token CPU oracle",
    }
