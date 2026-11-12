"""Adapt the existing controlled LI fixture to batched real transport inputs."""

import json
import shutil
from pathlib import Path

import numpy as np


def prepare_full_chain_fixture(
    root, history, resident, miss_percent, prefetch_decoys=0,
    index_heads=64, cold_probe_per_query=0, native_union_fit=False,
    actual_key_lengths=None,
):
    """Reuse controlled cold masks, give each final TopK an unambiguous boundary.

    Hot feature bands independently choose exactly the remaining winners for
    each query. This is an explicitly synthetic controlled-overlap fixture;
    no tie-breaking rule is added to either implementation.
    """
    from miss_fixture import build

    target = round(2048 * miss_percent / 100)
    if actual_key_lengths is not None and (cold_probe_per_query or prefetch_decoys):
        raise ValueError("P0-A2 actual-K fixture disallows cold probe and speculative decoys")
    if not 0 <= target <= min(2048, history - resident):
        raise ValueError("requested miss count exceeds available cold candidates")
    if cold_probe_per_query:
        if cold_probe_per_query != 1 or target != 1 or index_heads != 32:
            raise ValueError("cold probe requires one final miss per query and 32 index heads")
        if prefetch_decoys:
            raise ValueError("cold probe cannot be combined with speculative decoys")
        if history - resident < 128:
            raise ValueError("cold probe requires a complete first cold tile")
    tiles = (history - resident) // 128
    # Cold probe overrides the naturally dispersed cold tokens with one strong
    # winner per feature band inside the FIRST cold tile. This makes them
    # observable by the earliest active post-resident Rotation window.
    mean = 0 if cold_probe_per_query or actual_key_lengths is not None else min(256, 2 * (target + 0.125) / tiles)
    build(root, 0, "dispersed", history, resident, mean, independent_queries=True)
    root = Path(root)
    encoded = np.fromfile(root / "arg1.bin", np.uint8).reshape(history, 128)
    scale = np.fromfile(root / "arg4.bin", np.float32)
    weights = np.fromfile(root / "arg2.bin", np.float32).reshape(4, 64)
    query_scale = np.fromfile(root / "arg3.bin", np.float32).reshape(4, 64)
    if index_heads == 32:
        # The controlled fixture repeats each feature head twice. Merge exact
        # binary weight*scale factors; both native and fused read 32 real heads.
        weights = (weights * query_scale).reshape(4, 2, 32).sum(axis=1)
        query_scale = np.ones((4, 32), np.float32)
        query = np.fromfile(root / "arg0.bin", np.uint8).reshape(4, 64, 128)
        query[:, :32].copy().tofile(root / "arg0.bin")
        weights.tofile(root / "arg2.bin")
        query_scale.tofile(root / "arg3.bin")
    elif index_heads != 64:
        raise ValueError("index_heads must be 32 or 64")
    if cold_probe_per_query:
        # Independent Q bands: only the matching query can score this token.
        # Resident has 2047 winners after injecting one cold winner.
        for query in range(4):
            token = resident + 16 * query
            encoded[token, query * 32 : (query + 1) * 32] = 0x44
    # Spread background scores before BF16 quantization instead of making all
    # nonwinning hot rows equal to the resident TopK threshold. Binary scales
    # keep the independent score oracle reproducible. Winner value3 stays
    # above every background value (<=1), even at the minimum scale0.5.
    rng = np.random.default_rng(20261104)
    for query in range(4):
        background = rng.integers(0x08, 0x39, resident, dtype=np.uint8)
        encoded[:resident, query * 32 : (query + 1) * 32] = background[:, None]
    scale[:resident] = rng.integers(256, 513, resident).astype(np.float32) / 512
    if actual_key_lengths is not None:
        # Place all winning cold tokens before the smallest valid K. A short-K
        # row then still has exactly 2048 winners, with no quantized-score tie.
        minimum_k = min(actual_key_lengths)
        if resident + 4 * 256 + target > minimum_k:
            raise ValueError("actual_k_fixture_insufficient_early_cold_space")
        for q in range(4):
            winners = resident + q * 256 + np.arange(target)
            columns = q * 32 + np.arange(32)
            encoded[np.ix_(winners, columns)] = 0x44
            scale[winners] = 8.0
    if prefetch_decoys < 0:
        raise ValueError("prefetch_decoys must be nonnegative")
    logical, actual_misses, selected_cold, prefetched = [], [], set(), set()
    # The legacy q*1024 hot shift spans every resident slot at R=4096.
    # Optional Q4 overlap keeps the Native LIDU union within one R-sized cache.
    hot_stride = 128 if native_union_fit else 1024
    order = np.load(root / "physical_to_logical.npy")
    for query in range(4):
        columns = np.arange(index_heads) % 32 + query * 32
        cold = np.count_nonzero(encoded[resident:, query * 32])
        candidates = np.flatnonzero(encoded[resident:, query * 32] == 0) + resident
        if prefetch_decoys > len(candidates):
            raise ValueError("prefetch_decoys exceeds available nonwinning cold candidates")
        if prefetch_decoys:
            take = np.floor(np.arange(prefetch_decoys) * len(candidates) / prefetch_decoys).astype(int)
            # FP8 0x20 = 0.125; cold scale8 gives value1, above the
            # background threshold and below every final hot winner (>=1.5).
            encoded[np.ix_(candidates[take], np.unique(columns))] = 0x20
        winners = (np.arange(2048 - cold) + query * hot_stride) % resident
        encoded[np.ix_(winners, np.unique(columns))] = 0x44
        # Decode positive finite E4M3 values used by the actual input bytes.
        codes = encoded[:, columns]
        exponent = (codes >> 3).astype(np.int32)
        mantissa = (codes & 7).astype(np.float32)
        values = np.where(exponent == 0, mantissa * 2.0**-9, (1 + mantissa / 8) * np.exp2(exponent - 7)).astype(
            np.float32
        )
        scores = (values * (weights[query] * query_scale[query])).sum(-1) * scale
        bits = ((scores.view(np.uint32) + 0x8000) >> 16).astype(np.uint16)
        cutoff = np.partition(bits, -2048)[-2048]
        if int(np.count_nonzero(bits >= cutoff)) != 2048:
            raise ValueError(f"fixture_final_boundary_not_unique:query={query}")
        mapped = np.empty_like(bits)
        mapped[order] = bits
        logical.append(mapped)
        actual_misses.append(int(cold))
        if cold_probe_per_query and cold != 1:
            raise AssertionError(f"cold probe has {cold} cold winners for query {query}")
        selected_cold.update((np.flatnonzero(bits[resident:] >= cutoff) + resident).tolist())
        resident_cutoff = np.partition(bits[:resident], -2048)[-2048]
        prefetched.update((np.flatnonzero(bits[resident:] > resident_cutoff)[:2048] + resident).tolist())
    # LIDU must materialize every query's TopK in one R-slot resident cache.
    # Scores are in logical order; order maps physical token -> logical token.
    physical_scores = np.stack(logical)[:, order]
    selected_by_query = [np.argpartition(row, -2048)[-2048:] for row in physical_scores]
    selected_union = np.unique(np.concatenate(selected_by_query))
    unique_hot = int(np.count_nonzero(selected_union < resident))
    unique_cold = int(np.count_nonzero(selected_union >= resident))
    if selected_union.size > resident:
        raise ValueError(
            f"native_mtp_union_exceeds_resident: unique_topk={selected_union.size} "
            f"hot={unique_hot} cold={unique_cold} resident={resident} "
            f"hot_stride={hot_stride}; use --native-union-fit with enough "
            "resident capacity for all Q4 selected tokens"
        )
    actual_key_admission = None
    if actual_key_lengths is not None:
        from actual_k_admission import score_oracle_for_actual_k
        per_request, _ = score_oracle_for_actual_k(
            np.stack(logical), order, actual_key_lengths, resident)
        actual_key_admission = dict(
            fixed_history=history,
            key_lengths=list(actual_key_lengths),
            requests=per_request,
            cold_winners="all_in_shortest_valid_prefix",
        )
    encoded.tofile(root / "arg1.bin")
    scale.tofile(root / "arg4.bin")
    np.save(root / "logical_scores.npy", np.stack(logical))
    config = json.loads((root / "fixture.json").read_text())
    config["full_chain_fixture"] = dict(
        model="controlled_query_feature_bands",
        index_heads=index_heads,
        requested_miss_percent=miss_percent,
        final_misses_per_query=actual_misses,
        hot_background="seed20261104_positive_fp8_binary_scales_v2",
        final_boundary="exactly_2048_at_or_above_cutoff",
        temporal_trace=False,
        prefetch_decoys=prefetch_decoys,
        cold_probe_per_query=cold_probe_per_query,
        cold_probe_physical_tokens=(
            [resident + 16 * query for query in range(4)]
            if cold_probe_per_query else []
        ),
        expected_final_supplement=len(selected_cold - prefetched),
        actual_key_admission=actual_key_admission,
        native_union_admission=dict(
            enabled=bool(native_union_fit),
            hot_stride=hot_stride,
            unique_topk=int(selected_union.size),
            unique_hot=unique_hot,
            unique_cold=unique_cold,
            resident_capacity=resident,
            spare_slots=resident - int(selected_union.size),
            admissible=True,
        ),
    )
    (root / "fixture.json").write_text(json.dumps(config, indent=2))
    return config["full_chain_fixture"]


def batch_inputs(common, destination, batch, history, cores):
    if batch < 1:
        raise ValueError("positive batch required")
    shutil.copytree(common, destination)
    for index in (0, 2, 3, 5, 6, 7, 9):
        path = Path(destination) / f"arg{index}.bin"
        path.write_bytes(path.read_bytes() * batch)
    path = Path(destination) / "arg13.bin"
    blob = path.read_bytes()
    prefix = 4 * 2048 * 6 + 1024
    tail = prefix + 4 * history * 2
    path.write_bytes(blob[:prefix] * batch + blob[prefix:tail] * batch + blob[tail:] * batch)
    with (Path(destination) / "arg12.bin").open("wb") as workspace:
        workspace.truncate(max(16 * 1024 * 1024, cores * 4 * history * 4 + 1024 * 1024))


def install_remote_addresses(data, addresses, batch, history):
    addresses = np.asarray(addresses, dtype=np.uint64).reshape(batch, history)
    path = Path(data) / "arg13.bin"
    blob = np.fromfile(path, np.uint8)
    prefix_bytes = 4 * 2048 * 6 + 1024 + 4 * history * 2
    tail_bytes = history * 8 + 8192 * 64 + 4 * 128
    for row in range(batch):
        begin = batch * prefix_bytes + row * tail_bytes
        blob[begin : begin + history * 8] = addresses[row].view(np.uint8)
    blob.tofile(path)


def native_fixture(torch, arguments, batch, history):
    """Zero-copy dtype/layout views; native and fused consume identical bytes."""
    device = arguments[0].device
    return {
        "query": arguments[0].view(torch.float8_e4m3fn).reshape(batch, 4, 64, 128),
        "key": arguments[1].view(torch.float8_e4m3fn).reshape(history // 128, 128, 1, 128),
        # Native MTP LIDU uses cumulative TND row boundaries. The LI adapter
        # below separately supplies BSND lengths to match the installed source.
        "actual_query": torch.arange(1, batch + 1, dtype=torch.int32, device=device) * 4,
        "li_query_lengths": torch.full((batch,), 4, dtype=torch.int32, device=device),
        "actual_key": torch.full((batch,), history, dtype=torch.int32, device=device),
        "block_table": arguments[7].view(torch.int32).reshape(batch, history // 128),
        "native_inputs": {
            "weights": arguments[2].view(torch.bfloat16).reshape(batch, 4, 64),
            "query_scale": arguments[3].view(torch.float32).reshape(batch, 4, 64),
            "key_scale": arguments[4].view(torch.float32).reshape(history // 128, 128, 1),
        },
    }


def native_li(torch_npu, fixture):
    """Same BSND/sparse=0 contract as the accepted native Stage2 benchmark."""
    result = torch_npu.npu_quant_lightning_indexer(
        query=fixture["query"],
        key=fixture["key"],
        weights=fixture["native_inputs"]["weights"],
        query_dequant_scale=fixture["native_inputs"]["query_scale"],
        key_dequant_scale=fixture["native_inputs"]["key_scale"],
        actual_seq_lengths_query=fixture["li_query_lengths"],
        actual_seq_lengths_key=fixture["actual_key"],
        block_table=fixture["block_table"],
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="BSND",
        layout_key="PA_BSND",
        sparse_count=2048,
        sparse_mode=0,
    )
    selected = result[0] if isinstance(result, (list, tuple)) else result
    return selected.reshape(-1, 1, 2048)
