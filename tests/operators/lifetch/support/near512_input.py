"""Select a captured long Q4 round or assemble an explicitly synthetic long history."""

from collections import deque
from copy import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from native_li_reference import digest
from workloads import ReplaySequence, checked_path

CAPACITY = 524544
MIN_CONTEXT = 510 * 1024
PAGE = 128
TRACE_ROUNDS = {10: 51, 4: 2, 17: 2, 7: 2, 2: 2}


def _near_ends(ends, capacity):
    ends = np.asarray(ends)
    return (
        ends.ndim == 2
        and ends.shape[1] == 4
        and ends.dtype.kind in "iu"
        and np.all(np.diff(ends.astype(np.int64), axis=1) == 1)
        and np.all(ends[:, 0] >= MIN_CONTEXT)
        and np.all(ends[:, 0] <= 512 * 1024)
        and np.all(ends[:, -1] <= capacity)
    )


def _manifest_matches(path, group):
    m = json.loads(path.read_text())
    g = m.get("geometry", {})
    if (
        m.get("real_inputs_saved") is not True
        or m.get("metadata", {}).get("group") != group
        or not MIN_CONTEXT <= g.get("history", 0) <= CAPACITY
        or g["history"] % 256
        or (g.get("heads"), g.get("resident"), g.get("token_bytes")) != (32, 16384, 656)
    ):
        return False
    archive = checked_path(path.parent, m["archive"]["file"])
    with np.load(archive, allow_pickle=False) as data:
        return _near_ends(data["pre.query_ends"], g["history"])


def _long_capture_roots(data):
    # Only visit the capture parent; never traverse every DP, build or result tree.
    for directory, dirs, _ in os.walk(data):
        root = Path(directory)
        if (root / "dp0/series.json").is_file():
            yield root
            dirs[:] = []
            continue
        skip = {"fused-scan-sources", "adaptive-sources", "objects", "tensors", "ddr", "results", "peer-baseline", "KVAF"}
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "build", "cpu-")) and d not in skip)


def _near_capture(root, group):
    series = root / "dp0/series.json"
    if not series.is_file():
        return
    record = json.loads(series.read_text())
    for step in record.get("rounds", ()):
        num = step["round_index"]
        for item in step.get("groups", ()):
            if item["group"] != group:
                continue
            manifest = checked_path(series.parent, item["manifest"])
            if not _manifest_matches(manifest, group):
                continue
            seq = ReplaySequence(root, group, (num, num))
            refs = seq.rounds[0]
            if sum(len(ref.request_signature) for ref in refs) != 16:
                continue
            if any(not _manifest_matches(ref.manifest_path, group) for ref in refs if ref.dp_rank != 0):
                continue
            seq.cache_ranked_round()
            yield seq


def _native_input(seq):
    from cap_pair_input import bind_capacity
    from native_li_reference import digest
    from workloads import ReplayPagePlan

    original = next(iter(seq))
    # bind_capacity expects one consecutive Q4 capture and leaves every visible
    # byte and address unchanged. ReplayPagePlan re-created by the worker can
    # reserve additional invalid rows but these never have a usable address.
    bound, round_, _, _ = bind_capacity(seq, original, ReplayPagePlan(seq), CAPACITY, digest)
    return bound, round_


def _donor_pages(source):
    table = []
    for row in range(source.geometry["batch"]):
        # A real, fully populated page seen by all four source Qs, with available KV.
        valid = source.logical_address_valid[row].astype(bool) & source.logical_score_valid[row].all(axis=0)
        pages = np.flatnonzero(valid.reshape(-1, PAGE).all(axis=1))
        if len(pages) == 0:
            raise ValueError(f"No valid captured full pages in donor row {row}")
        table.append(pages)
    return table


def _page_plan(table, row, count):
    # Begin with this request's own published pages (so its hot slots can
    # be carried over). Interleave 32-page chunks from the other requests.
    donors = deque((row + d) % len(table) for d in range(1, len(table)))
    segments = [(row, table[row])]
    cursor = {d: 0 for d in donors}
    while donors:
        d = donors.popleft()
        at = cursor[d]
        segments.append((d, table[d][at : at + 32]))
        cursor[d] += 32
        if cursor[d] < len(table[d]):
            donors.append(d)
    donor = np.concatenate([np.full(len(p), d, dtype=np.int32) for d, p in segments])
    page = np.concatenate([p for _, p in segments])
    if len(donor) < count:
        # Distinct pages on the target, identified by (request,logical page),
        # even when input bytes have to be sampled a second time.
        copies = (count + len(donor) - 1) // len(donor)
        donor, page = np.tile(donor, copies), np.tile(page, copies)
    return donor[:count], page[:count]


def _map_resident(source, row, table, row_ends, capacity):
    resident = source.geometry["resident"]
    old = source.captured_layout[row, :resident]
    slots = np.full(source.geometry["history"] // PAGE, -1, np.int32)
    slots[table[row]] = np.arange(len(table[row]), dtype=np.int32)
    pages = slots[old // PAGE]
    mapped = (pages * PAGE + old % PAGE).astype(np.int32)
    keep = (pages >= 0) & (mapped < row_ends[0])
    taken = np.zeros(capacity, bool)
    taken[mapped[keep]] = True
    # New backing for any old hot slot whose source page was unpublished.
    # Fill from early visible tokens, preserving the captured slot-valid and LRU order.
    mapped[~keep] = np.flatnonzero(~taken[: row_ends[0]])[: np.count_nonzero(~keep)]
    taken[mapped] = True
    return np.concatenate((mapped, np.flatnonzero(~taken))).astype(np.int32), int(keep.sum())


def _stitched(source, ends, capacity):
    g = source.geometry
    batch = g["batch"]
    if ends.shape != (batch, 4) or capacity % 256 or np.any(ends < g["resident"]) or np.any(ends > capacity):
        raise ValueError("Invalid stitched effective lengths or storage geometry")
    table = _donor_pages(source)
    keys = np.zeros((batch, capacity, 128), np.uint8)
    scales = np.ones((batch, capacity), np.float32)
    layouts = np.empty((batch, capacity), np.int32)
    address = np.zeros((batch, capacity), np.uint8)
    tails, page_keys, retained, source_distinct, payload_hashes = [], [], [], [], set()
    for row in range(batch):
        end = int(ends[row, -1])
        active_pages = (end + PAGE - 1) // PAGE
        donor, old_page = _page_plan(table, row, active_pages)
        source_distinct.append(len(set(zip(donor.tolist(), old_page.tolist()))))
        keys_by_page = keys[row].reshape(-1, PAGE, 128)
        scales_by_page = scales[row].reshape(-1, PAGE)
        for d in np.unique(donor):
            dest = np.flatnonzero(donor == d)
            select = old_page[dest]
            keys_by_page[dest] = source.logical_keys[d].reshape(-1, PAGE, 128)[select]
            scales_by_page[dest] = source.logical_scales[d].reshape(-1, PAGE)[select]
        payload_hashes.add(hashlib.sha256(memoryview(keys[row])).hexdigest())
        width = int(source.tail_bounds[row, 1] - source.tail_bounds[row, 0])
        if not 0 <= width <= g["output_stride"] - g["resident"]:
            raise ValueError(f"Invalid captured ABC span for donor {row}: {width}")
        # Move the captured recent ring range by the change in sequence length.
        tail_end = min(end, int(source.tail_bounds[row, 1]) + end - int(source.query_ends[row, -1]))
        tails.append((tail_end - width, tail_end) if width else (0, 0))
        address[row, : active_pages * PAGE] = 1
        row_layout, kept = _map_resident(source, row, table, ends[row], capacity)
        layouts[row] = row_layout
        retained.append(kept)
        page_keys.append(tuple((row, i) if i < active_pages else None for i in range(capacity // PAGE)))
    if len(payload_hashes) != batch:
        raise ValueError("Donor rows produced identical sampled input signatures")
    valid = np.arange(capacity) < ends[:, :, None]
    # The capture score oracle belongs to the old Q/Key pairs. Mark scores as
    # unavailable until the worker installs its independent native LI checker.
    # This read-only field has only one backing element; diagnostic printing on
    # the old replay ABI still expects its shape.
    scores = np.broadcast_to(np.uint16(0xFFC1), valid.shape)
    inputs = replace(
        source,
        geometry=dict(g, history=capacity),
        logical_keys=keys,
        logical_scales=scales,
        query_ends=ends,
        tail_bounds=np.asarray(tails, dtype=np.int32),
        logical_address_valid=address,
        captured_layout=layouts,
        page_keys=tuple(page_keys),
        logical_score_keys=scores,
        logical_score_valid=valid,
    )
    inputs.validate_topk = _need_native_oracle
    return inputs, dict(
        source_pages_available_per_row=[len(p) for p in table],
        distinct_source_row_page_pairs_per_target_row=source_distinct,
        captured_hot_slot_ids_remapped_per_row=retained,
        unique_target_page_addresses=True,
        distinct_target_key_rows=len(payload_hashes),
        score_source="independent native LI on the assembled inputs; no donor scores are reused",
    )


def _need_native_oracle(selected):
    raise RuntimeError("Stitched input requires the independent native TopK oracle")


def _find_long_input(work, group):
    override = os.environ.get("PEER_NEAR512_TRACE")
    roots = (Path(override),) if override else _long_capture_roots(Path(work).parent)
    for root in roots:
        try:
            seq = next(_near_capture(root, group), None)
        except (OSError, ValueError, KeyError) as error:
            if override:
                raise
            print(f"NEAR512_CAPTURE_SKIPPED root={root} reason={error}", flush=True)
            continue
        if seq is not None:
            return seq, root
    if override:
        raise ValueError(f"No complete BS16/Q4 G{group} near-512K DP capture in PEER_NEAR512_TRACE={override}")
    return None, None


def _verified_donor(seq, source, group, proof_path):
    """Identify the short input before it is extended or its Index is remapped."""
    refs, = seq.rounds
    ranks = [ref.dp_rank for ref in refs]
    identities = tuple((ref.dp_rank, request, owner) for ref in refs for request, owner in ref.request_signature)
    if (ranks != list(range(8)) or len(identities) != 16 or identities != source.identities
            or source.source_round != 1 or any(ref.group != group or ref.source_round != source.source_round
                                              or ref.scheduler_round != source.scheduler_round for ref in refs)):
        raise ValueError(f"G{group}: group, round or request identities differ from the prepared DP0–7 trace")

    proof_path = Path(proof_path)
    proof = json.loads(proof_path.read_text())
    archived = proof["capture_operand_hashes"]
    fields = ("query", "weights", "query_scale", "logical_keys", "logical_scales", "query_ends", "captured_layout")
    hashes = {name: digest(getattr(source, name)) for name in fields}
    mismatches = [name for name in fields if hashes[name] != archived.get(name)]
    state = proof.get("history_cap", {}).get("capture_state_sha256", {})
    for name in ("captured_lru", "captured_hot_valid", "logical_address_valid", "tail_bounds"):
        hashes[name] = digest(getattr(source, name))
        if name in state and hashes[name] != state[name]:
            mismatches.append(name)
    if mismatches:
        raise ValueError(f"G{group}: 128K source differs from {proof_path}: {', '.join(mismatches)}")
    return dict(group=group, selector_round=TRACE_ROUNDS[group], replay_round=source.source_round,
                scheduler_round=source.scheduler_round, dp_ranks=ranks, requests=len(identities),
                operand_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                field_sha256=hashes, archive_proof=str(proof_path))


def make_input(work, group, fallback_trace, report_dir, *, stitch_source=False, source_proof=None, preserve_workload=False):
    """Return a cached one-round sequence, its ReplayRound and a printable provenance receipt."""
    if preserve_workload:
        stitch_source = True
    if stitch_source and (group not in TRACE_ROUNDS or source_proof is None):
        raise ValueError("Five-trace stitching requires a supported group and its archived 128K input proof")
    seq, root = (None, None) if stitch_source else _find_long_input(work, group)
    if seq is not None:
        seq, inputs = _native_input(seq)
        provenance = dict(
            kind="capture",
            source_kind="captured_near512k",
            source_root=str(root),
            score_source="archived native score keys",
        )
    else:
        if fallback_trace is None:
            prepare = Path(__file__).resolve().parents[1] / "peer_g2_input.sh"
            result = subprocess.run(
                ["bash", str(prepare), str(work), str(group)],
                env={**os.environ, "PEER_LOCAL_ONLY": "1"},
                stdout=subprocess.PIPE,
                text=True,
                check=True,
            )
            fallback_trace = Path(result.stdout.strip())
        src_seq = ReplaySequence(fallback_trace, group, (1, 1))
        src_seq.cache_ranked_round()
        source = next(iter(src_seq))
        if (
            source.geometry["batch"],
            source.geometry["heads"],
            source.geometry["history"],
            source.geometry["resident"],
            source.geometry["token_bytes"],
        ) != (16, 32, 131328, 16384, 656):
            raise ValueError(f"G{group}: expected a captured BS16/Q4 128K donor with 32 heads and 656B tokens")
        donor = _verified_donor(src_seq, source, group, source_proof) if stitch_source else None
        if preserve_workload:
            from near512_locality_input import expand_locality
            from near512_profile import archive_profile, bind_source_check

            archived_state = json.loads(Path(source_proof).read_text())["history_cap"]["capture_state_sha256"]
            for name in ("logical_score_keys", "logical_score_valid"):
                if digest(getattr(source, name)) != archived_state[name]:
                    raise ValueError(f"G{group}: captured {name} differs from the archived short input")
            inputs, proof = expand_locality(source, CAPACITY)
            bind_source_check(source, inputs, proof["source_page_to_long_page"])
            proof["short_profile"] = archive_profile(source, group, source_proof)
        else:
            ends = 524280 - 17 * np.arange(16, dtype=np.int32)[:, None] + np.arange(4, dtype=np.int32)
            inputs, proof = _stitched(source, ends, CAPACITY)
        seq = copy(src_seq)
        seq.cached_ranked_round = inputs
        provenance = dict(
            kind="workload_preserved" if preserve_workload else "stitched",
            source_kind="rank_preserving_short_capture" if preserve_workload else "stitched_short_capture_pages",
            source_root=str(fallback_trace),
            **proof,
        )
        if donor is not None:
            provenance["donor"] = donor
    provenance.update(
        actual_group=group,
        capacity=CAPACITY,
        effective_ends=inputs.query_ends.tolist(),
        source_manifests=list(inputs.source_paths),
        token_bytes=inputs.geometry["token_bytes"],
        batch=inputs.geometry["batch"],
        warm_kv_payload="deterministic 656B page and token identity",
    )
    (Path(report_dir) / "near512_input.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return seq, inputs, provenance
