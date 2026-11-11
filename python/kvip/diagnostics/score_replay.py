# SPDX-License-Identifier: Apache-2.0
"""Analyze captured LI scores; optionally merge real LI inputs for one-card replay.

Score-only traces never create a timing fixture. A real-input fixture preserves
the original LI bits and uses deterministic KV bytes with DP-local DDR aliases.
"""

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from kvip.diagnostics.archive import checked_path
from kvip.diagnostics.token_replay import QUERIES, TOPK, _array, _inputs, materialize_score_trace
from kvip.operators.lifetch.geometry import final_aivs, li_geometry


def score_value(key):
    """Inverse FloatToSortableKey<bfloat16_t>; ranking uses the original keys."""
    key = np.asarray(key, dtype=np.uint16)
    bits = np.where(key & 0x8000, key ^ 0x8000, ~key).astype(np.uint16)
    words = np.asarray(bits, dtype=np.uint32)
    # NumPy 1.x can promote a zero-dimensional uint32 << Python int to
    # int64. Keep both operands and the destination uint32 before bitcasting.
    np.left_shift(words, np.uint32(16), out=words)
    return words.view(np.float32)


def _identity(trace):
    metadata = trace.manifest["metadata"]
    requests = metadata["requests"]
    by_owner = {item["owner"]: item for item in requests}
    owners = trace.array("pre.row_owners")
    if len(by_owner) != len(owners) or set(by_owner) != set(map(int, owners)):
        raise ValueError("Request identities do not cover active owners exactly")
    identities = []
    for row, owner in enumerate(owners):
        item = by_owner[int(owner)]
        if item["query_begin"] != row * QUERIES or item["query_stop"] != (row + 1) * QUERIES:
            raise ValueError("Request identity differs from its packed Q4 row")
        identities.append((metadata["dp_rank"], str(item["request_id"]), int(owner)))
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate request identity")
    return identities


def analyze_scores(trace, previous):
    """Check complete causal scores and evaluate the previous TopK on this round.

    The rescored previous set is a proposed initial threshold, not a claim that
    the captured kernel used it. No Q/K reconstruction or device timing occurs.
    """
    geometry, inputs = _inputs(trace)
    if trace.manifest.get("content") != "scores":
        raise ValueError("Score analysis requires a complete score trace")
    b, h = geometry["batch"], geometry["history"]
    scores = _array(trace, "post.score_keys", (b, QUERIES, h), np.uint16)
    records = _array(trace, "post.threshold_records", (b, QUERIES, (h + 16383) // 16384 + 2, 4), np.int32)
    source = trace.manifest["metadata"]
    old_source = previous.manifest["metadata"]
    if (
        old_source["context"]["round"] + 1 != source["context"]["round"]
        or old_source["dp_rank"] != source["dp_rank"]
        or old_source["group"] != source["group"]
        or old_source["layer_names"] != source["layer_names"]
    ):
        raise ValueError("Previous TopK seed requires the immediately preceding scheduler round and same DP/group")
    _, old = _inputs(previous)
    identities, previous_identities = _identity(trace), _identity(previous)
    if set(identities) != set(previous_identities):
        raise ValueError("Previous TopK seed request/owner lifetime differs")
    previous_rows = {identity: row for row, identity in enumerate(previous_identities)}
    tnd = np.argsort(inputs["metadata"][2])
    old_tnd = np.argsort(old["metadata"][2])
    queries = []
    for row, identity in enumerate(identities):
        inverse = np.argsort(inputs["ids"][inputs["row_owners"][row]])
        for q in range(QUERIES):
            end = int(inputs["ends"][row, q])
            keys = scores[row, q, :end]
            if np.isnan(score_value(keys)).any():
                raise ValueError(f"NaN score inside the causal prefix: DP{identity[0]} row{row} q{q}")
            final = int(np.partition(keys, len(keys) - TOPK)[-TOPK])
            selection = inputs["selected"][tnd[row * QUERIES + q]]
            selected_scores = scores[row, q, inverse[selection]]
            if np.any(selected_scores < final) or np.count_nonzero(selected_scores > final) != np.count_nonzero(
                keys > final
            ):
                raise ValueError(f"Captured TopK omits a higher-scored token: DP{identity[0]} row{row} q{q}")
            observed = []
            for valid_end, applies_from, key, kind in records[row, q]:
                valid_end, applies_from, key, kind = map(int, (valid_end, applies_from, key, kind))
                if not kind:
                    if valid_end or applies_from or key:
                        raise ValueError("Unused threshold record contains data")
                    continue
                if kind not in (1, 2, 3, 4) or not TOPK <= valid_end <= end or not valid_end <= applies_from <= h:
                    raise ValueError("Captured threshold prefix or application boundary is invalid")
                expected = int(np.partition(keys[:valid_end], valid_end - TOPK)[-TOPK])
                if key != expected or kind == 4 and (valid_end != end or applies_from != h):
                    raise ValueError("Captured threshold disagrees with its complete score prefix")
                observed.append(
                    dict(
                        kind=kind,
                        valid_end=valid_end,
                        applies_from=applies_from,
                        key=key,
                        value=float(score_value(key)),
                    )
                )
            if not observed or observed[-1]["kind"] != 4:
                raise ValueError("Captured score trace is missing its final threshold")
            old_row = previous_rows[identity]
            seed = old["selected"][old_tnd[old_row * QUERIES + q]]
            if np.any(seed >= end):
                raise ValueError("Previous TopK contains a token outside the current causal prefix")
            seed_key = int(scores[row, q, inverse[seed]].min())
            if seed_key > final:
                raise AssertionError("A K-token seed cannot exceed the full-history Kth threshold")
            initial = next((item for item in observed if item["kind"] != 4), None)
            queries.append(
                dict(
                    dp_rank=identity[0],
                    request_id=identity[1],
                    owner=identity[2],
                    packed_row=row,
                    query=q,
                    query_end=end,
                    final_key=final,
                    final_value=float(score_value(final)),
                    observed_initial_key=None if initial is None else initial["key"],
                    observed_initial_value=None if initial is None else initial["value"],
                    previous_seed_key=seed_key,
                    previous_seed_value=float(score_value(seed_key)),
                    previous_seed_candidates=int(np.count_nonzero(keys >= seed_key)),
                    previous_seed_above=int(np.count_nonzero(keys > seed_key)),
                    previous_seed_equal=int(np.count_nonzero(keys == seed_key)),
                    final_candidates=int(np.count_nonzero(keys >= final)),
                    final_above=int(np.count_nonzero(keys > final)),
                    final_equal=int(np.count_nonzero(keys == final)),
                    thresholds=observed,
                )
            )
    # The shared trace validator recomputes miss/transfer classes from captured IDs.
    from kvip.diagnostics.token_trace import trace_summary

    actual = trace_summary(trace)
    if actual != trace.manifest["summary"]:
        raise ValueError("Captured miss/transfer summary differs from actual IDs")
    return dict(
        status="pass",
        content="scores",
        dp_rank=source["dp_rank"],
        group=source["group"],
        scheduler_round=source["context"]["round"],
        previous_scheduler_round=old_source["context"]["round"],
        score_scope="All causal physical Index positions; uint16 sortable BF16 keys, exact captured thresholds",
        seed_scope="Previous captured TopK rescored on current scores; a proposed seed, not observed kernel execution",
        candidate_count_rule="*_candidates includes >= threshold ties; *_above is strict >, *_equal is ties. Native admission additionally applies residency, window and budget gates.",
        queries=queries,
        **actual,
    )


def merge_traces(traces, *, numa_node):
    """Merge real inputs without changing LI bits; isolate DP arenas by slot range."""
    if not traces or type(numa_node) is not int or numa_node < -1:
        raise ValueError("Merging requires traces and an explicit target NUMA node")
    ranks = [trace.manifest["metadata"]["dp_rank"] for trace in traces]
    if len(set(ranks)) != len(ranks):
        raise ValueError("Duplicate DP rank")
    first = traces[0].manifest
    geometry = copy.deepcopy(first["geometry"])
    fields = (
        "history",
        "resident",
        "heads",
        "token_bytes",
        "output_stride",
        "aic_count",
        "requested_windows",
        "tokens_per_block",
        "slot_bytes",
        "prefetch_limit",
        "window_limit",
        "soc",
    )
    source = first["metadata"]
    batch = sum(trace.manifest["geometry"]["batch"] for trace in traces)
    if not 1 <= batch <= 16:
        raise ValueError("Merged Q4 batch must be in 1..16")
    arrays, requests, subarenas, sources = {}, [], [], []
    row_offset, slot_offset = 0, 0
    metadata_parts = []
    names = (
        "pre.logical_ids",
        "pre.hot_valid",
        "pre.address_valid",
        "pre.query_ends",
        "pre.tail_bounds",
        "pre.lru_slots",
        "post.logical_output",
        "post.score_keys",
        "real.keys",
        "real.key_scales",
        "real.query",
        "real.weights",
        "real.query_scale",
    )
    parts = {name: [] for name in names}
    page_parts = []
    for trace in traces:
        g, validated = _inputs(trace)
        m = trace.manifest["metadata"]
        if trace.manifest.get("content") != "scores" or trace.manifest.get("real_inputs_saved") is not True:
            raise ValueError("Real LI replay requires saved real inputs on every selected DP")
        if any(g.get(name) != geometry.get(name) for name in fields) or (
            m["context"]["round"] != source["context"]["round"]
            or m["group"] != source["group"]
            or m["layer_names"] != source["layer_names"]
            or m["context"].get("global_decode_confirmed") is not True
        ):
            raise ValueError("DP traces differ in scheduler round, group or core geometry")
        identities = _identity(trace)
        size, slot = g["region_bytes"], g["slot_bytes"]
        if type(size) is not int or type(slot) is not int or size <= 0 or slot <= 0 or size % slot:
            raise ValueError("Source DDR arena must contain complete fixed-size slots")
        pages = validated["page_slots"].astype(np.int64)
        if np.any(pages >= size // slot):
            raise ValueError("Published slot is outside its source DP arena")
        pages[pages >= 0] += slot_offset
        if pages.max() > np.iinfo(np.int32).max:
            raise ValueError("Merged slot namespace exceeds the captured int32 slot format")
        page_parts.append(pages.astype(np.int32))
        local = validated["metadata"].copy()
        local[0] += row_offset * QUERIES
        local[2] += row_offset * QUERIES
        local[4] = 0
        metadata_parts.append(local)
        for name in names:
            parts[name].append(trace.array(name))
        by_owner = {item["owner"]: item for item in m["requests"]}
        for row, (_, request_id, owner) in enumerate(identities):
            item = copy.deepcopy(by_owner[owner])
            item.update(
                request_id=f"dp{m['dp_rank']}:{request_id}",
                owner=row_offset + row + 1,
                query_begin=(row_offset + row) * QUERIES,
                query_stop=(row_offset + row + 1) * QUERIES,
                source_dp_rank=m["dp_rank"],
                source_request_id=request_id,
                source_owner=owner,
            )
            requests.append(item)
        subarenas.append(
            dict(
                dp_rank=m["dp_rank"],
                slot_offset=slot_offset,
                region_bytes=size,
                slot_bytes=slot,
                source_page_mode=g["page_mode"],
                source_numa_node=g["numa_node"],
            )
        )
        sources.append(copy.deepcopy(m))
        row_offset += g["batch"]
        slot_offset += size // slot
    arrays.update({name: np.concatenate(values, axis=0) for name, values in parts.items()})
    arrays["pre.metadata"] = np.concatenate(metadata_parts, axis=1)
    arrays["pre.row_owners"] = np.arange(1, batch + 1, dtype=np.int32)
    arrays["pre.metadata"][4].view(np.int32)[:batch] = arrays["pre.row_owners"]
    arrays["pre.ddr_page_slots"] = np.concatenate(page_parts)
    layout = li_geometry(batch, "auto", geometry["requested_windows"], geometry["aic_count"])
    geometry.update(
        batch=batch,
        owners=batch + 1,
        token_count=batch * QUERIES,
        rounds=layout["intermediate_fetches"],
        service_lanes=layout["service_aivs_per_window"],
        final_workers=final_aivs(batch, "auto", geometry["aic_count"]),
        region_bytes=slot_offset * geometry["slot_bytes"],
        page_mode="thp",
        numa_node=numa_node,
    )
    metadata = copy.deepcopy(source)
    metadata.update(
        dp_rank=0,
        dp_size=1,
        source_dp_ranks=ranks,
        requests=requests,
        merged_from=sources,
        ddr_subarenas=subarenas,
        capture_real_inputs=True,
    )
    metadata["geometry"] = dict(metadata.get("geometry", {}), batch=batch)
    metadata["context"]["min_context"] = min(m["context"]["min_context"] for m in sources)
    metadata["context"]["max_context"] = max(m["context"]["max_context"] for m in sources)
    manifest = dict(
        schema_version=4,
        kind="kvaf_decode_score_trace",
        content="scores",
        status="complete",
        validation=dict(passed=True, kind="score_trace_integrity"),
        real_inputs_saved=True,
        geometry=geometry,
        metadata=metadata,
    )
    merged = SimpleNamespace(manifest=manifest, arrays=arrays, array=lambda name: arrays[name])
    _inputs(merged)
    return merged


def load_round(root, ranks, round_index, group):
    """Resolve manifest paths from each series, never infer a snapshot from filenames."""
    from kvip.diagnostics.token_trace import load_token_trace

    if round_index < 2 or not ranks or len(set(ranks)) != len(ranks) or any(rank < 0 for rank in ranks):
        raise ValueError("Select distinct DP ranks and an analysis round >=2 (round 1 is the seed)")
    current, previous = [], []
    for rank in ranks:
        directory = Path(root) / f"dp{rank}"
        series = json.loads((directory / "series.json").read_text())
        if (
            series.get("status") != "complete"
            or series.get("content") != "scores"
            or series.get("dp_rank") != rank
            or series.get("consecutive_required") is not True
            or series.get("seed_rounds") != 1
        ):
            raise ValueError("Expected a complete consecutive scores series with one seed round")
        for number, output in ((round_index - 1, previous), (round_index, current)):
            matches = [item for item in series["rounds"] if item["round_index"] == number]
            if len(matches) != 1:
                raise ValueError(f"DP{rank} capture round {number} is missing or duplicated")
            entries = [item for item in matches[0]["groups"] if item["group"] == group]
            if len(entries) != 1:
                raise ValueError(f"DP{rank} round {number} group {group} is missing or duplicated")
            path = checked_path(directory, entries[0]["manifest"])
            trace = load_token_trace(path.parent)
            m = trace.manifest["metadata"]
            if m["dp_rank"] != rank or m["group"] != group or m["context"]["round"] != matches[0]["scheduler_round"]:
                raise ValueError("Snapshot identity differs from its series entry")
            signature = [dict(request_id=item["request_id"], owner=item["owner"]) for item in m["requests"]]
            if signature != series.get("request_owner_signature"):
                raise ValueError("Snapshot request cohort differs from its completed series")
            output.append(trace)
    if len({trace.manifest["metadata"]["context"]["round"] for trace in current}) != 1:
        raise ValueError("Selected DPs did not capture the same scheduler round")
    first = current[0].manifest
    for trace in current[1:]:
        m, g = trace.manifest["metadata"], trace.manifest["geometry"]
        if m["layer_names"] != first["metadata"]["layer_names"] or any(
            g.get(name) != first["geometry"].get(name)
            for name in (
                "batch",
                "history",
                "resident",
                "heads",
                "token_bytes",
                "output_stride",
                "aic_count",
                "requested_windows",
                "prefetch_limit",
                "window_limit",
                "soc",
            )
        ):
            raise ValueError("Selected DP score traces have different layer coverage or core geometry")
    return current, previous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--round", dest="round_index", required=True, type=int, help="Capture index, including seed round 1"
    )
    parser.add_argument("--group", required=True, type=int)
    parser.add_argument("--dp-ranks", nargs="+", required=True, type=int)
    parser.add_argument(
        "--output", type=Path, help="New real-LI/synthetic-KV schema1 fixture; requires saved real inputs"
    )
    parser.add_argument(
        "--numa-node", type=int, help="Explicit single-card target NUMA node; -1 uses the target default"
    )
    parser.add_argument(
        "--layer-index", type=int, help="Project one zero-based source group layer to independent 656-byte KV slots"
    )
    parser.add_argument("--quiet", action="store_true", help="Print only the final fixture/analysis summary")
    args = parser.parse_args(argv)
    if args.output is not None and args.numa_node is None:
        parser.error("--output requires explicit --numa-node (-1 or a nonnegative node)")
    if args.output is None and args.numa_node is not None:
        parser.error("--numa-node requires --output")
    if args.output is None and args.layer_index is not None:
        parser.error("--layer-index requires --output")
    try:
        current, previous = load_round(args.root, args.dp_ranks, args.round_index, args.group)
        reports = [analyze_scores(trace, old) for trace, old in zip(current, previous)]
        for report in reports:
            if args.quiet:
                continue
            q = report["queries"]
            initial = [item["observed_initial_value"] for item in q if item["observed_initial_value"] is not None]

            def span(values):
                return f"{min(values):.6g}..{max(values):.6g}" if values else "none"

            print(
                f"SCORE_PASS dp={report['dp_rank']} round={args.round_index} scheduler={report['scheduler_round']} "
                f"group={args.group} queries={len(q)} cold={report['demand']['cold_demand']} "
                f"unused={report['reads']['unused_prefetch']} initial={span(initial)} "
                f"final={span([item['final_value'] for item in q])} "
                f"previous_seed={span([item['previous_seed_value'] for item in q])}"
            )
        if args.output is not None:
            merged = merge_traces(current, numa_node=args.numa_node)
            merged.manifest["metadata"]["score_analysis"] = reports
            output = materialize_score_trace(
                merged, args.output, numa_node=args.numa_node, layer_index=args.layer_index
            )
            demand = output.manifest["synthetic"]["captured_selection_demand"]
            print(
                f"REAL_LI_SYNTHETIC_KV_READY {output.directory} batch={merged.manifest['geometry']['batch']} "
                f"queries=4 layers={','.join(output.manifest['metadata']['layer_names'])} "
                f"token_bytes={output.manifest['geometry']['token_bytes']} "
                f"captured_cold={demand['cold_demand']} selected_unique={demand['selected_unique']} "
                f"cold_miss_percent={100 * demand['cold_miss_rate']:.4f} "
                f"ddr_GiB={output.manifest['ddr']['region_bytes'] / (1 << 30):.3f} "
                f"numa={args.numa_node} page_mode=thp; native score/KV validation remains required; not DP service timing"
            )
            if args.quiet:
                print(
                    f"SOURCE root={args.root} round={args.round_index} group={args.group} "
                    f"dp_ranks={','.join(map(str, args.dp_ranks))} "
                    f"layer_index={'all' if args.layer_index is None else args.layer_index}"
                )
        else:
            print("SCORES_ONLY: CPU threshold/TopK/miss validation; no native LI or fusion performance measurement")
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"SCORE_REPLAY_FAILED: {error}\n")


if __name__ == "__main__":
    main()
