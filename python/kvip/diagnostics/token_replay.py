# SPDX-License-Identifier: Apache-2.0
"""Materialize one token-ID trace for synthetic production-operator validation.

The actual captured TopK, residency, causal boundaries and LRU candidates are
retained. Scores, KV bytes and DDR placement are constructed, so replay timing
cannot reproduce the captured model's LI thresholds or transport performance.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from kvip.adapters.vllm.inputs import reserved_hot_reference
from kvip.diagnostics.archive import TensorArchive, array_digest, file_digest
from kvip.diagnostics.ddr import validate_pages
from kvip.diagnostics.demand import SnapshotDemand
from kvip.diagnostics.fused_replay import _check_geometry
from kvip.diagnostics.oracle import DecodeOracle
from kvip.diagnostics.serial_replay import serial_inputs
from kvip.diagnostics.snapshot import relocate_snapshot_addresses, validate_capture_inputs

TOPK, QUERIES, BLOCK, KEY_WIDTH = 2048, 4, 128, 128
FP8_ONE, BF16_ONE = 0x38, 0x3F80
SYNTHETIC_BASE = 0x100000000
FIDELITY = (
    "Synthetic score-membership Q/K/weights/scales and deterministic KV bytes, not real model inputs. "
    "Captured active-owner logical permutations, valid hot slots, ABC bounds, LRU candidates, causal queries, "
    "published-token availability and TopK sets are preserved. DDR slots are compact with captured shared-page aliases preserved; original addresses, page placement, inactive-owner state, scores and LI/prefetch timing are not reproduced."
)


def synthetic_payload(owner, logical, width):
    """Distinct owner/token tags plus deterministic bytes across the whole KV row."""
    logical = np.asarray(logical, dtype="<u8").reshape(-1)
    columns = np.arange(width, dtype=np.uint64)
    result = ((logical[:, None] * 131 + columns[None, :] * 17 + int(owner) * 29) % 251).astype(np.uint8)
    result[:, :8] = logical.view(np.uint8).reshape(-1, 8)
    result[:, 8:12] = np.asarray([owner], dtype="<u4").view(np.uint8)
    return result


def _array(trace, name, shape, dtype):
    value = np.asarray(trace.array(name))
    if value.shape != shape or value.dtype != np.dtype(dtype):
        raise ValueError(f"Token trace {name} must have shape {shape} and dtype {dtype}")
    return value


def _inputs(trace):
    manifest, geometry = trace.manifest, dict(trace.manifest["geometry"])
    if (
        manifest.get("kind") != "kvaf_decode_token_trace"
        or manifest.get("content") != "token_ids"
        or manifest.get("status") != "complete"
        or manifest.get("validation", {}).get("passed") is not True
    ):
        raise ValueError("Expected a lightweight token-ID trace, not a full-input snapshot")
    b, h, r, width, stride, heads, owner_count = (
        geometry[name] for name in ("batch", "history", "resident", "token_bytes", "output_stride", "heads", "owners")
    )
    if (
        any(type(value) is not int for value in (b, h, r, width, stride, heads, owner_count))
        or not 1 <= b <= 16
        or h % 256
        or r % 256
        or not TOPK <= r < h
        or width not in (656, 2624)
        or heads not in (32, 64)
        or stride % BLOCK
        or stride < r + 384
        or owner_count <= b
        or geometry.get("tokens_per_block", BLOCK) != BLOCK
    ):
        raise ValueError("Token trace geometry is outside the production Q4 replay contract")
    _check_geometry(geometry, geometry["aic_count"])
    metadata = _array(trace, "pre.metadata", (5, b * QUERIES), np.int64)
    row_owners = _array(trace, "pre.row_owners", (b,), np.int32)
    if (
        np.any(row_owners <= 0)
        or np.any(row_owners >= owner_count)
        or len(np.unique(row_owners)) != b
        or not np.array_equal(metadata[4].view(np.int32)[:b], row_owners)
        or not np.all(metadata[1:4:2] == 1)
    ):
        raise ValueError("Token trace requires distinct real owners and full Q4 input/output validity")
    unpack = metadata[2]
    if not np.array_equal(np.sort(unpack), np.arange(b * QUERIES)):
        raise ValueError("Token trace TND unpack must be a permutation of all packed Q4 rows")
    compact_ids = _array(trace, "pre.logical_ids", (b, h), np.int32)
    compact_hot = _array(trace, "pre.hot_valid", (b, r), np.uint8)
    address_valid = _array(trace, "pre.address_valid", (b, h), np.uint8)
    page_slots = _array(trace, "pre.ddr_page_slots", (b, h // BLOCK), np.int32)
    ends = _array(trace, "pre.query_ends", (b, QUERIES), np.int32)
    bounds = _array(trace, "pre.tail_bounds", (b, 2), np.int32)
    candidates = _array(trace, "pre.lru_slots", (b, TOPK), np.int32)
    selected = _array(trace, "post.logical_output", (b * QUERIES, TOPK), np.int32)
    if np.any((compact_hot != 0) & (compact_hot != 1)) or np.any(address_valid > 1):
        raise ValueError("Token trace residency and address availability must be Boolean")
    ids = np.tile(np.arange(h, dtype=np.int64), (owner_count, 1))
    hot = np.zeros((owner_count, r), dtype=np.int32)
    published = []
    for row, owner in enumerate(row_owners):
        if not np.array_equal(np.sort(compact_ids[row]), np.arange(h)):
            raise ValueError("Token trace logical IDs are not a full Index permutation")
        ids[owner], hot[owner] = compact_ids[row], compact_hot[row]
        for end in ends[row]:
            if not TOPK <= end <= h or np.any(compact_ids[row, :end] >= end):
                raise ValueError("Token trace physical/logical causal prefixes differ")
        logical_valid = np.empty(h, dtype=np.uint8)
        logical_valid[compact_ids[row]] = address_valid[row]
        committed = int(logical_valid.sum())
        if (
            committed % BLOCK
            or not np.all(logical_valid[:committed])
            or np.any(logical_valid[committed:])
            or committed < bounds[row, 0]
            or not np.array_equal(np.repeat(page_slots[row] >= 0, BLOCK), logical_valid != 0)
            or np.any(page_slots[row] < -1)
        ):
            raise ValueError("Token trace published availability must cover complete contiguous committed pages")
        published.append(committed)
    for tnd, source in enumerate(unpack):
        values = selected[tnd]
        if np.any(values < 0) or np.any(values >= ends.reshape(-1)[source]) or len(np.unique(values)) != TOPK:
            raise ValueError("Token trace selection contains duplicates or violates its causal bound")
    return geometry, dict(
        ids=ids,
        hot=hot,
        row_owners=row_owners,
        metadata=metadata,
        ends=ends,
        bounds=bounds,
        candidates=candidates,
        selected=selected,
        published=published,
        page_slots=page_slots,
    )


def _token_payload(owner, logical, inputs, width):
    logical = np.asarray(logical, dtype=np.int64).reshape(-1)
    row = int(np.flatnonzero(inputs["row_owners"] == owner)[0])
    pages = inputs["page_slots"][row, logical // BLOCK]
    result = np.empty((len(logical), width), dtype=np.uint8)
    for page in np.unique(pages):
        mask = pages == page
        result[mask] = (
            synthetic_payload(int(page), logical[mask] % BLOCK, width)
            if page >= 0
            else synthetic_payload(int(owner) | (1 << 31), logical[mask], width)
        )
    return result


def _write_ddr(directory, geometry, inputs):
    width, h = geometry["token_bytes"], geometry["history"]
    slot = ((width * BLOCK + 4095) // 4096) * 4096
    directory.mkdir()
    addresses = np.zeros(inputs["ids"].shape, dtype=np.int64)
    identities = list(dict.fromkeys(int(slot) for slot in inputs["page_slots"].reshape(-1) if slot >= 0))
    if not identities:
        raise ValueError("Operator replay requires at least one actually published DDR page")
    compact = {identity: index for index, identity in enumerate(identities)}
    with (directory / "payload.bin").open("xb") as output:
        for identity in identities:
            output.write(synthetic_payload(identity, np.arange(BLOCK), width).tobytes())
    for row, owner in enumerate(inputs["row_owners"]):
        logical_addresses = np.zeros(h, dtype=np.int64)
        for block, identity in enumerate(inputs["page_slots"][row]):
            if identity >= 0:
                logical_addresses[block * BLOCK : (block + 1) * BLOCK] = (
                    SYNTHETIC_BASE + compact[int(identity)] * slot + np.arange(BLOCK) * width
                )
        addresses[owner] = logical_addresses[inputs["ids"][owner]]
    count = len(identities)
    keys = np.arange(count, dtype=np.int64)
    offsets = keys.astype(np.uint64) * slot
    arrays = dict(
        keys=keys,
        slot_offsets=offsets,
        addresses=offsets + SYNTHETIC_BASE,
        valid_lengths=np.full(count, width * BLOCK, dtype=np.uint64),
    )
    files = {}
    for name, value in arrays.items():
        path = directory / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        files[name] = dict(path=path.name, sha256=file_digest(path))
    files["payload"] = dict(path="payload.bin", sha256=file_digest(directory / "payload.bin"))
    return addresses, dict(
        schema_version=1,
        transport="ubmem",
        device_id=0,
        page_mode="base",
        numa_node=-1,
        generation="synthetic-token-replay",
        region_base=SYNTHETIC_BASE,
        region_bytes=count * slot,
        slot_bytes=slot,
        token_bytes=width,
        tokens_per_block=BLOCK,
        record_count=count,
        payload_bytes=count * width * BLOCK,
        files=files,
        synthetic_slot_identities=identities,
    )


def _synthetic_tensors(geometry, inputs, addresses):
    b, h, r, width, stride, heads = (
        geometry[name] for name in ("batch", "history", "resident", "token_bytes", "output_stride", "heads")
    )
    ids, owners = inputs["ids"], inputs["row_owners"]
    keys = np.zeros((*ids.shape, KEY_WIDTH), dtype=np.uint8)
    # Q is zero in these columns. Token/owner fingerprints make Index key
    # permutations observable even for tokens with identical membership scores.
    for owner in owners:
        for bit in range(32):
            keys[owner, :, 4 + bit] = ((ids[owner] >> bit) & 1).astype(np.uint8) * FP8_ONE
            keys[owner, :, 36 + bit] = ((int(owner) >> bit) & 1) * FP8_ONE
    query = np.zeros((b * QUERIES, heads, KEY_WIDTH), dtype=np.uint8)
    weights = np.zeros((b * QUERIES, heads), dtype=np.uint16)
    weights[:, 0] = BF16_ONE
    # Captured outputs are TND. Scatter each set back into its packed query.
    for tnd, source in enumerate(inputs["metadata"][2]):
        row, q = divmod(int(source), QUERIES)
        owner = owners[row]
        keys[owner, np.isin(ids[owner], inputs["selected"][tnd]), q] = FP8_ONE
        query[source, 0, q] = FP8_ONE
    output = np.zeros((len(ids), stride, width), dtype=np.uint8)
    for row, owner in enumerate(owners):
        hot_slots = np.flatnonzero(inputs["hot"][owner])
        output[owner, hot_slots] = _token_payload(owner, ids[owner, hot_slots], inputs, width)
        abc = np.arange(*inputs["bounds"][row], dtype=np.int64)
        output[owner, r + abc % (stride - r)] = _token_payload(owner, abc, inputs, width)
    return dict(
        keys=keys,
        key_scales=np.ones(ids.shape, dtype=np.float32),
        addresses=addresses,
        logical_ids=ids,
        hot_valid=inputs["hot"],
        output=output.reshape(-1, width),
        metadata=inputs["metadata"],
        query=query,
        weights=weights,
        query_scale=np.ones((b * QUERIES, heads), dtype=np.float32),
        query_ends=inputs["ends"].reshape(-1),
        query_lengths=np.full(b, QUERIES, dtype=np.int32),
        key_lengths=np.full(b, h, dtype=np.int32),
        block_table=(owners[:, None] * (h // BLOCK) + np.arange(h // BLOCK)).astype(np.int32),
        lru_slots=inputs["candidates"],
        reserved_hot=reserved_hot_reference(inputs["candidates"], inputs["ends"], r),
        tail_bounds=inputs["bounds"],
    )


def materialize_token_trace(trace, destination):
    """Build one self-contained synthetic fixture; device correctness is checked by replay.py."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite synthetic fixture: {destination}")
    geometry, inputs = _inputs(trace)
    destination.mkdir(parents=True)
    (destination / "INCOMPLETE").write_text("Synthetic fixture construction has not completed.\n")
    manifest = dict(
        schema_version=1,
        status="incomplete",
        content="synthetic_token_replay",
        source_transport="ubmem",
        geometry=geometry,
        tensors={},
        metadata=copy.deepcopy(trace.manifest["metadata"]),
        provenance=dict(
            source_token_manifest_sha256=array_digest(
                np.frombuffer(json.dumps(trace.manifest, sort_keys=True).encode(), dtype=np.uint8)
            )
        ),
        scope="synthetic_token_ID_membership_LI_TopK_Fetch_IndexUpdate_validation",
        fidelity=FIDELITY,
        synthetic=dict(
            score_reference="Exact FP8 0/1 membership: selected=1, other=0; BF16 boundary gap=1",
            payload_reference="shared page/token tags; owner/logical-ID tags for unpublished ABC",
            native_validated=False,
        ),
    )
    manifest["metadata"]["content"] = "synthetic_token_replay"
    addresses, manifest["ddr"] = _write_ddr(destination / "ddr", geometry, inputs)
    archive = TensorArchive(destination, manifest)
    tensors = _synthetic_tensors(geometry, inputs, addresses)
    for name, value in tensors.items():
        archive.save("pre." + name, value, dtype={"query": "float8_e4m3fn", "weights": "bfloat16"}.get(name))
    archive.save("post.logical_output", inputs["selected"])
    # Oracle identity is (stable owner, logical token), not captured physical slots.
    tags = []
    seen = set()
    for tnd, source in enumerate(inputs["metadata"][2]):
        owner = int(inputs["row_owners"][source // QUERIES])
        for token in inputs["selected"][tnd]:
            identity = (owner, int(token))
            if identity not in seen:
                seen.add(identity)
                tags.append(identity)
    tags = np.asarray(tags, dtype=np.int64)
    payload = np.empty((len(tags), geometry["token_bytes"]), dtype=np.uint8)
    for owner in np.unique(tags[:, 0]):
        mask = tags[:, 0] == owner
        payload[mask] = _token_payload(owner, tags[mask, 1], inputs, geometry["token_bytes"])
    archive.save("post.selected_tags", tags)
    archive.save("post.selected_payload", payload)
    validate_capture_inputs(archive)
    validate_pages(destination / "ddr", manifest["ddr"])
    relocate_snapshot_addresses(archive, SYNTHETIC_BASE)
    serial_inputs(archive, SYNTHETIC_BASE)  # Reject layouts native LI/LIDU cannot consume.
    oracle = DecodeOracle(archive)
    for owner in np.unique(tags[:, 0]):
        mask = tags[:, 0] == owner
        if not np.array_equal(oracle.expected(owner, tags[mask, 1]), payload[mask]):
            raise AssertionError("Synthetic hot/ABC/DDR bytes disagree for a captured selected ID")
    manifest["oracle"] = dict(passed=True, kind="synthetic_fixture_consistency", native_validated=False)
    manifest["synthetic"]["captured_selection_demand"] = SnapshotDemand(archive).classify(inputs["selected"])
    manifest["status"] = "complete"
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "INCOMPLETE").unlink()
    return archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path, help="One selected token-ID group directory")
    parser.add_argument("--output", required=True, type=Path, help="New synthetic replay fixture directory")
    args = parser.parse_args(argv)
    from kvip.diagnostics.token_trace import load_token_trace

    archive = materialize_token_trace(load_token_trace(args.snapshot), args.output)
    print(
        f"SYNTHETIC_FIXTURE_READY {archive.directory}; native validation remains required; real-model timing is not reproduced"
    )


if __name__ == "__main__":
    main()
