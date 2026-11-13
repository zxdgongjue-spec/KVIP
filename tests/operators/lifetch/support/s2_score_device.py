# SPDX-License-Identifier: Apache-2.0
"""P0-C5a A5 Score-only Q1 S2 split admission and timing.

Reference: original V9 LI Score workspace, on identical Q1/K128K inputs.
Probe: 1/2/4 AIC groups, disjoint per-AIC Score workspace regions.
Strict raw 16-bit score equality, ownership and no TopK/Hit writes.
"""
import numpy as np

from s2_split_oracle import plan_static_s2


def _score_words(arguments):
    data = arguments[12].detach().cpu().numpy()
    if data.dtype != np.uint8 or data.nbytes % 2:
        raise AssertionError("Score workspace must be uint8/even length")
    return data.view(np.uint16).copy()


def _stitch(words, shards, history):
    """Private ScoreGM rows hold Q0 at each shard's global S2 offset."""
    stride = 4 * history  # 4 Query slots, uint16 per S2 token/AIC
    sentinel = np.uint16(0xA5A5)
    output = np.full(history, sentinel, dtype=np.uint16)
    covered = np.zeros(history, dtype=bool)
    for shard in shards:
        begin, end = shard.token_start, shard.token_end
        row = words[shard.core * stride:(shard.core + 1) * stride]
        if row.size != stride:
            raise AssertionError("Score workspace shorter than shard allocation")
        if covered[begin:end].any():
            raise AssertionError("S2 score shard overlap")
        output[begin:end] = row[begin:end]
        covered[begin:end] = True
        if np.any(row[:begin] != sentinel) or np.any(row[end:] != sentinel):
            raise AssertionError(
                f"AIC group {shard.core} wrote Score outside [{begin},{end})"
            )
    if not covered.all():
        raise AssertionError("S2 Score coverage incomplete")
    return output


def run_s2_score_probe(
    *, args, torch, device, owner, arguments, fixture, native_li, torch_npu,
    rotating, topk, stream, inputs, query_lengths, key_lengths,
):
    history, resident, groups = args.history, args.resident, args.li_s2_score_cores
    if (history, resident) != (131072, 8192) or groups not in (1, 2, 4):
        raise ValueError("P0-C5a fixed H131072/R8192 and groups=1/2/4")
    if "s2_score" not in rotating.functions:
        raise RuntimeError("v9_s2_score not exported: build with V9_S2_SCORE_PROBE=1")
    sentinel = np.uint16(0xA5A5)
    q_counts = (1, 0, 0, 0)
    k_lengths = (history, 0, 0, 0)

    inputs.prepare(q_counts, query_starts=(0, 1, 1, 1), query_owners=(0, 1, 2, 3))
    inputs.bind_positions(torch.tensor([history - 1], device=device, dtype=torch.int64))
    stream.synchronize()
    observed_q = tuple(int(x) for x in query_lengths.cpu().tolist())
    observed_k = tuple(int(x) for x in key_lengths.cpu().tolist())
    if (observed_q, observed_k) != (q_counts, k_lengths):
        raise AssertionError(f"Actual Q/K mismatch {observed_q}/{observed_k}")

    fixture["li_query_lengths"] = query_lengths
    fixture["actual_key"] = key_lengths
    native_output = native_li(torch_npu, fixture).reshape(4, 4, 2048)

    # Same-input full original V9 LI is the binary Score reference. Unlike
    # Resident-only TopK equality, this reads ALL 128K original Score tokens.
    arguments[12].fill_(0xA5)
    topk.fill_(0x5A5A5A5A)
    rotating.reset()
    owner.reset()
    rotating._call("init")
    rotating._call("li")
    rotating._call("wait")
    stream.synchronize()
    if not torch.equal(
        torch.sort(native_output[0, 0]).values,
        torch.sort(topk[0, 0]).values,
    ):
        raise AssertionError("Original V9 topk does not match Native on Q1/K128K")

    baseline = _score_words(arguments)[:history].copy()
    count_sentinel = int(np.count_nonzero(baseline == sentinel))
    if count_sentinel:
        raise AssertionError(f"Original V9 missing {count_sentinel} Score positions")

    topk.fill_(0x5A5A5A5A)
    original_shared = arguments[13].clone()
    arguments[12].fill_(0xA5)
    stream.synchronize()
    rotating._call("s2_score")
    stream.synchronize()

    shards = plan_static_s2(history, groups, resident=resident)
    actual = _stitch(_score_words(arguments), shards, history)
    bad = np.flatnonzero(actual != baseline)
    if bad.size:
        preview = bad[:12].tolist()
        raise AssertionError(
            f"S2 Score raw bits mismatch: {len(bad)} at first IDs {preview} "
            f"want={baseline[preview].tolist()} got={actual[preview].tolist()}"
        )
    if not bool(torch.all(topk == 0x5A5A5A5A).item()):
        raise AssertionError("Score-only kernel wrote final TopK")
    if not torch.equal(original_shared, arguments[13]):
        raise AssertionError("Score-only kernel wrote shared prefix/Hit storage")

    # Device-event timing of ONLY the Score producer. Packed Graph reduces
    # Host enqueue gaps. This is an amortized cost, not standalone TPOT.
    graph = torch.npu.NPUGraph()
    owner.graphs.append(graph)
    capture_stream = torch.npu.Stream(device=device)
    stream.synchronize()
    with torch.npu.graph(graph, stream=capture_stream):
        for _ in range(args.li_core_perf_graph_repeats):
            rotating._call("s2_score")
    capture_stream.synchronize()
    graph.replay()
    stream.synchronize()
    for _ in range(args.warmup):
        graph.replay()
    stream.synchronize()

    samples = []
    for _ in range(args.iterations):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record(stream)
        graph.replay()
        end.record(stream)
        end.synchronize()
        samples.append(float(start.elapsed_time(end) * 1000.0
                             / args.li_core_perf_graph_repeats))
    stream.synchronize()
    if not np.all(np.isfinite(samples)):
        raise AssertionError("Invalid Score Graph timing samples")
    if not np.array_equal(_stitch(_score_words(arguments), shards, history), baseline):
        raise AssertionError("Score changed after repeated Graph replay")
    if not bool(torch.all(topk == 0x5A5A5A5A).item()):
        raise AssertionError("Score Graph replay wrote final TopK")
    if not torch.equal(original_shared, arguments[13]):
        raise AssertionError("Score Graph replay wrote shared Hit metadata")

    perf = {
        "aic_groups": groups,
        "actual_q": list(observed_q),
        "actual_k": history,
        "actual_k_by_request": list(observed_k),
        "shards": [dict(core=s.core, start=s.token_start, end=s.token_end) for s in shards],
        "raw_score_equal": True,
        "score_tokens_checked": history,
        "topk_unchanged": True,
        "shared_prefix_unchanged": True,
        "native_baseline_topk_equal": True,
        "graph_repeats": args.li_core_perf_graph_repeats,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "samples_us": samples,
        "p50_us": float(np.median(samples)),
        "p95_us": float(np.percentile(samples, 95)),
        "cv_percent": float(np.std(samples) / np.mean(samples) * 100),
        "scope": "packed Graph, score producer only; excludes TopK/Fetch/Index update",
    }
    return {
        "test": "P0-C5a-isolated-score-only",
        "result": "PASS",
        "li_tiling": "q4",
        "li_cores": rotating.runtime_shape.liCores,
        "rotation_epochs": rotating.runtime_shape.rounds,
        "history": history, "resident": resident, "batch": 4,
        "rounds": [{"name": f"q1_k128k_{groups}aic", "result": "PASS"}],
        "score_probe": perf,
        "executed_stages": ["reference_v9_li", "v9_s2_score"],
        "excluded_stages": ["Score-only TopK", "MIS", "BatchGet", "Index update"],
    }
