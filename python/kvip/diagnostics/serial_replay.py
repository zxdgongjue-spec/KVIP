# SPDX-License-Identifier: Apache-2.0
"""Replay the serving native LI -> LIDU -> BatchGet -> slot-map chain.

Snapshot Index rows are hot-first permutations. Native CANN LI instead consumes
logical paged history with its causal mask. Restore that representation once,
outside measurement; do not replace LI with the saved TopK. Ordinary BatchGet
uses no fused lease and retains its native queue state between graph replays.
"""

from types import SimpleNamespace

import numpy as np

from kvip.diagnostics.archive import cpu_array
from kvip.diagnostics.demand import classify_initial_slots, sum_demand
from kvip.diagnostics.oracle import DecodeOracle
from kvip.diagnostics.selection import SelectionOracle

TOPK = 2048
BLOCK = 128
QUERIES = 4
# Match the existing LIDU ABI, including the ABC extension and 32-byte alignment.
MAX_CACHE_TOKENS = (1 << 17) + 20 * BLOCK
BITMAP_WORDS = ((MAX_CACHE_TOKENS + 31) // 32 + 7) // 8 * 8


def serial_inputs(archive, new_base):
    """Restore native logical layouts from the captured bytes on the host."""
    g, ddr = archive.manifest["geometry"], archive.manifest["ddr"]
    b, h, r, width, stride = (g[n] for n in ("batch", "history", "resident", "token_bytes", "output_stride"))
    if (
        not 1 <= b <= 16
        or h % BLOCK
        or r < TOPK
        or r % BLOCK
        or width not in (656, 2624)
        or stride % BLOCK
        or not r + 384 <= stride <= MAX_CACHE_TOKENS
        or h > (1 << 18) + 32
        or ddr["tokens_per_block"] != BLOCK
    ):
        raise ValueError("Snapshot geometry is outside the production serial LIDU contract")
    ids = archive.array("pre.logical_ids")
    owner_count = ids.shape[0]
    if ids.shape != (g["owners"], h) or ids.dtype != np.int64:
        raise ValueError("Snapshot Index owner/history layout is invalid")
    identity = np.arange(h)
    inverse = np.empty_like(ids)
    for owner, row in enumerate(ids):
        if not np.array_equal(np.sort(row), identity):
            raise ValueError("Snapshot Index logical IDs must be complete permutations")
        inverse[owner, row] = identity
    metadata = archive.array("pre.metadata")
    if metadata.shape != (5, b * QUERIES) or not np.all(metadata[1:4:2] == 1):
        raise ValueError("Serial replay requires a full real Q4 batch")
    unpack = metadata[2].reshape(b, QUERIES)
    packed_rows = unpack[:, 0] // QUERIES
    if not np.array_equal(np.sort(packed_rows), np.arange(b)) or not np.array_equal(
        unpack, packed_rows[:, None] * QUERIES + np.arange(QUERIES)
    ):
        raise ValueError("Native TND requests must each contain four consecutive packed queries")
    packed_owners = metadata[4].view(np.int32)[:b]
    owners = packed_owners[packed_rows]
    if np.any(owners <= 0) or np.any(owners >= owner_count) or len(np.unique(owners)) != b:
        raise ValueError("Serial replay requires distinct live request owners")
    ends = archive.array("pre.query_ends").reshape(b, QUERIES)[packed_rows]
    if np.any(np.diff(ends, axis=1) != 1) or np.any(ends < TOPK) or np.any(ends > h):
        raise ValueError("Native LI requires four real consecutive causal query positions")
    for owner, row_ends in zip(owners, ends):
        for end in row_ends:
            # Fusion masks the physical prefix; native LI masks logical IDs.
            # A serving Index may permute committed rows, never future tokens.
            if np.any(ids[owner, :end] >= end):
                raise ValueError("Snapshot physical and logical causal prefixes differ")
    bounds = archive.array("pre.tail_bounds")[packed_rows]
    tail = stride - r
    if (
        bounds.shape != (b, 2)
        or np.any(bounds[:, 0] % BLOCK)
        or np.any(bounds[:, 0] > ends[:, 0] - 1)
        or np.any(bounds[:, 1] < ends[:, -1])
        or np.any(bounds[:, 1] > h)
        or np.any(bounds[:, 1] - bounds[:, 0] > tail)
    ):
        raise ValueError("Snapshot ABC bounds do not cover the native query positions")
    keys, scales = archive.array("pre.keys"), archive.array("pre.key_scales")
    if keys.shape != (owner_count, h, 128) or scales.shape != ids.shape:
        raise ValueError("Snapshot Index key/scale geometry differs from its logical IDs")
    ordered_keys = np.take_along_axis(keys, inverse[:, :, None], axis=1)
    ordered_scales = np.take_along_axis(scales, inverse, axis=1)
    old_addresses = np.take_along_axis(archive.array("pre.addresses").view(np.uint64), inverse, axis=1)
    page_addresses = np.zeros((owner_count, h // BLOCK), dtype=np.uint64)
    page_valid = np.zeros(page_addresses.shape, dtype=np.uint8)
    exported = set(map(int, np.load(archive.directory / "ddr" / ddr["files"]["keys"]["path"], allow_pickle=False)))
    for row, owner in enumerate(owners):
        committed = int(bounds[row, 0])
        pages = old_addresses[owner, :committed].reshape(-1, BLOCK)
        bases = pages[:, 0]
        expected = bases[:, None] + np.arange(BLOCK, dtype=np.uint64) * width
        if np.any(bases == 0) or not np.array_equal(pages, expected):
            raise ValueError("Native Get requires complete contiguous published logical pages")
        for page, old_base in enumerate(bases):
            offset = int(old_base) - ddr["region_base"]
            if offset < 0 or offset % ddr["slot_bytes"] or offset // ddr["slot_bytes"] not in exported:
                raise ValueError("Native Get logical page is absent from the frozen DDR export")
            address = int(new_base) + offset
            if not 0 < address < (1 << 64):
                raise ValueError("Relocated native Get page address overflows uint64")
            page_addresses[owner, page] = address
            page_valid[owner, page] = 1
    forward = np.full((owner_count, h), -1, dtype=np.int32)
    reverse = np.full((owner_count, stride), -1, dtype=np.int32)
    hot_valid = archive.array("pre.hot_valid")
    for row, owner in enumerate(owners):
        hot = np.flatnonzero(hot_valid[owner] != 0)
        tokens = ids[owner, hot]
        forward[owner, tokens], reverse[owner, hot] = hot, tokens
        abc = np.arange(*bounds[row], dtype=np.int32)
        # ABC takes precedence, as in DecodeOracle.expected and serving maps.
        previous = forward[owner, abc]
        reverse[owner, previous[previous >= 0]] = -1
        slots = r + abc % tail
        forward[owner, abc], reverse[owner, slots] = slots, abc
    candidates = archive.array("pre.lru_slots").reshape(b, TOPK)[packed_rows].copy()
    bitmap = np.zeros((b, BITMAP_WORDS), dtype=np.uint32)
    for row in range(b):
        if np.any(candidates[row] < 0) or np.any(candidates[row] >= r) or len(np.unique(candidates[row])) != TOPK:
            raise ValueError("Snapshot LRU candidates must be 2048 distinct hot slots")
        np.bitwise_or.at(bitmap[row], candidates[row] // 32, np.uint32(1) << (candidates[row] % 32).astype(np.uint32))
    blocks = owners[:, None] * (stride // BLOCK) + np.arange(stride // BLOCK, dtype=np.int32)
    return dict(
        owners=owners.copy(),
        unpack=unpack.reshape(-1).copy(),
        query_ends=ends.copy(),
        tail_bounds=bounds.copy(),
        keys=ordered_keys,
        scales=ordered_scales,
        forward=forward,
        reverse=reverse,
        candidates=candidates,
        bitmap=bitmap.view(np.int32),
        page_addresses=page_addresses.view(np.int64),
        page_valid=page_valid,
        index_blocks=(owners[:, None] * (h // BLOCK) + np.arange(h // BLOCK)).astype(np.int32),
        resident_blocks=blocks.astype(np.int32),
    )


def _verify_serial(archive, inputs, oracle, logical, slots, forward, reverse, output, counts, destinations):
    """Check dynamic native selection, selected bytes and both resident maps."""
    g = archive.manifest["geometry"]
    b, h, r, stride, width = (g[n] for n in ("batch", "history", "resident", "output_stride", "token_bytes"))
    logical, slots = logical.reshape(b, QUERIES, TOPK), slots.reshape(b, QUERIES, TOPK)
    if counts.shape != (b + 1,) or np.any(counts < 0) or counts[-1] != counts[:-1].sum():
        raise AssertionError("Native LIDU miss counts are invalid")
    output = output.reshape(-1, width)[: len(forward) * stride].reshape(len(forward), stride, width)
    destinations = destinations.reshape(b, QUERIES, TOPK)
    demand = []
    for row, owner in enumerate(inputs["owners"]):
        selected = logical[row]
        if np.any(selected < 0) or np.any(selected >= inputs["query_ends"][row, :, None]):
            raise AssertionError("Native LI violated a causal query boundary")
        if any(len(np.unique(query)) != TOPK for query in selected):
            raise AssertionError("Native LI selected duplicate logical tokens")
        unique = np.unique(selected)
        initial_slots = inputs["forward"][owner, unique]
        # Production LIDU deliberately treats a resident candidate slot as a
        # miss too: those reserved slots may be overwritten during this round.
        classified = classify_initial_slots(initial_slots, inputs["candidates"][row], r)
        demand.append(classified)
        misses = classified["cold_demand"] + classified["candidate_hot_refetch"]
        if int(counts[row]) != misses:
            raise AssertionError("Native LIDU reads differ from its cold/candidate-slot miss classification")
        mapped = forward[owner, selected]
        if np.any(mapped < 0) or np.any(mapped >= stride) or not np.array_equal(mapped, slots[row]):
            raise AssertionError("Native TND output disagrees with the updated logical-to-slot map")
        if not np.array_equal(np.sort(destinations[row], axis=-1), np.sort(mapped, axis=-1)):
            raise AssertionError("Native LIDU destinations differ from the selected resident slots")
        if not np.array_equal(reverse[owner, mapped], selected):
            raise AssertionError("Native selected slots have inconsistent reverse ownership")
        if not np.array_equal(output[owner, mapped.reshape(-1)], oracle.expected(owner, selected.reshape(-1))):
            raise AssertionError("Native selected KV payload differs from the frozen service input")
        live = np.flatnonzero(forward[owner] >= 0)
        physical = forward[owner, live]
        if (
            np.any(physical >= stride)
            or len(np.unique(physical)) != len(physical)
            or not np.array_equal(reverse[owner, physical], live)
        ):
            raise AssertionError("Native resident maps are not inverse one-to-one ownership maps")
        for start in range(0, len(live), 256):
            ids = live[start : start + 256]
            if not np.array_equal(output[owner, forward[owner, ids]], oracle.expected(owner, ids)):
                raise AssertionError("Native updated resident KV is inconsistent with its logical ownership")
    return dict(
        prefetch=0,
        final=int(counts[-1]),
        transfers=int(counts[-1]),
        bytes=int(counts[-1]) * width,
        final_per_request=[int(value) for value in counts[:-1]],
        unused_prefetch=0,
        unused_prefetch_bytes=0,
        useful_transfers=int(counts[-1]),
        useful_transfer_bytes=int(counts[-1]) * width,
        **sum_demand(demand),
    )


def create_serial(
    torch, runtime, archive, restored, device_table, build, soc, device, stream, cann, aics, *, transport
):
    """Create an unchanged native LI/LIDU/Get graph with restored serving inputs."""
    if transport not in ("ubmem", "urma") or getattr(runtime, "transport", transport) != transport:
        raise ValueError("Serial replay transport differs from its Runtime")
    if restored.get("access_table") is None:
        raise ValueError("Serial replay requires pinned imported DDR pages")
    from kvip.operators.lidu import lidu
    import torch_npu

    g = archive.manifest["geometry"]
    b, h, r, stride, width = (g[n] for n in ("batch", "history", "resident", "output_stride", "token_bytes"))
    inputs = serial_inputs(archive, restored["new_base"])
    owners = len(inputs["forward"])
    retained = dict(runtime=runtime, device_table=device_table, restored=restored, inputs=inputs)

    def tensor(values):
        return torch.from_numpy(np.ascontiguousarray(values).copy()).to(device)

    def query_input(name):
        description = archive.manifest["tensors"]["pre." + name]
        array = archive.array("pre." + name)[inputs["unpack"]]
        raw = tensor(array.view(np.uint8).reshape(-1))
        return raw.view(getattr(torch, description["dtype"])).reshape(array.shape)

    query, weights, query_scale = (query_input(name) for name in ("query", "weights", "query_scale"))
    keys = tensor(inputs["keys"]).view(torch.float8_e4m3fn).reshape(-1, BLOCK, 1, 128)
    scales = tensor(inputs["scales"]).reshape(-1, BLOCK, 1)
    index_blocks, resident_blocks = (tensor(inputs[name]) for name in ("index_blocks", "resident_blocks"))
    ends = tensor(inputs["query_ends"][:, -1].astype(np.int32))
    cumulative = torch.arange(1, b + 1, dtype=torch.int32, device=device) * QUERIES
    entries = tensor(inputs["owners"].astype(np.int32))
    budgets = torch.full((b,), stride, dtype=torch.int32, device=device)
    preferred, bitmap = (tensor(inputs[name]) for name in ("candidates", "bitmap"))
    addresses, valid = (tensor(inputs[name]).reshape(-1) for name in ("page_addresses", "page_valid"))
    initial_forward, initial_reverse = (tensor(inputs[name]) for name in ("forward", "reverse"))
    forward_map, reverse_map = initial_forward.clone(), initial_reverse.clone()
    initial_output = archive.tensor("pre.output", torch, device)
    output = initial_output.clone().view(-1, width)
    cursors = torch.zeros(owners, dtype=torch.int32, device=device)
    profile = torch.zeros(1, dtype=torch.int32, device=device)
    destinations = torch.empty((b * QUERIES, 1, TOPK), dtype=torch.int32, device=device)
    counts = torch.empty(b + 1, dtype=torch.int32, device=device)
    indices = torch.empty(b * QUERIES * TOPK, dtype=torch.int32, device=device)
    offsets, output_indices = torch.empty_like(indices), torch.empty_like(indices)
    resident_lengths = torch.empty(b, dtype=torch.int32, device=device)
    profiling = torch.empty((b, 8, 32), dtype=torch.int64, device=device)
    bases = tensor(np.repeat(inputs["owners"].astype(np.int64), QUERIES) * h).view(-1, 1, 1)
    total = counts[b:]
    state = {}
    generated_outputs = []
    oracle = DecodeOracle(archive)
    selection = SelectionOracle(archive, inputs)
    selection_comparison = {}
    retained.update(locals())
    retained.pop("retained", None)

    def native_li():
        result = torch_npu.npu_quant_lightning_indexer(
            query=query,
            key=keys,
            weights=weights,
            query_dequant_scale=query_scale,
            key_dequant_scale=scales,
            actual_seq_lengths_query=cumulative,
            actual_seq_lengths_key=ends,
            block_table=index_blocks,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=TOPK,
            sparse_mode=3,
        )
        result = result[0] if isinstance(result, (tuple, list)) else result
        if result.dtype != torch.int32 or result.numel() != b * QUERIES * TOPK:
            raise ValueError("Native LI output differs from the serving int32 TND TopK contract")
        state["logical"] = result.reshape(b * QUERIES, 1, TOPK)
        # Full-graph and diagnostic stage captures can create distinct output
        # tensors. Keep both until the caller destroys every captured graph.
        # Graph replay itself does not execute Python or grow this list.
        generated_outputs.append(state["logical"])

    def prepare():
        lidu(
            state["logical"],
            cumulative,
            entries,
            forward_map,
            reverse_map,
            cursors,
            budgets,
            ends,
            preferred,
            bitmap,
            resident_blocks,
            ends,
            profile,
            destinations,
            counts,
            indices,
            offsets,
            output_indices,
            resident_lengths,
            profiling,
            0,
            1,
            owners,
            h // BLOCK,
        )
        if width != 656:
            offsets.mul_(width // 656)

    def fetch():
        runtime.batch_get(addresses, valid, indices, offsets, total, output_indices, output, stream, False)

    def slot_mapping():
        # This is serving's logical-index -> physical-slot gather, not a KV
        # payload gather. Layer interleave multiplication belongs to SFA and is
        # outside both serial and fused replay scopes.
        indices_for_map = state["logical"] + bases
        state["slots"] = torch.gather(forward_map.view(-1), 0, indices_for_map.reshape(-1)).view_as(state["logical"])
        generated_outputs.append(state["slots"])

    stages = (("native_li", native_li), ("lidu_prepare", prepare), ("batch_get", fetch), ("slot_mapping", slot_mapping))

    def forward():
        for _, call in stages:
            call()

    def reset():
        forward_map.copy_(initial_forward)
        reverse_map.copy_(initial_reverse)
        output.copy_(initial_output.view(-1, width))
        cursors.zero_()
        # Native transport queue cursors and registrations must never be reset.

    def verify():
        values = [
            cpu_array(torch, value)
            for value in (state["logical"], state["slots"], forward_map, reverse_map, output, counts, destinations)
        ]
        checked_selection = selection.verify(values[0])
        reads = _verify_serial(
            archive,
            inputs,
            oracle,
            *values,
        )
        selection_comparison.clear()
        selection_comparison.update(checked_selection)
        return reads

    retained.update(state=state, stages=stages)
    return SimpleNamespace(
        forward=forward,
        stage_calls=stages,
        reset=reset,
        verify=verify,
        lease=None,
        library=None,
        retained=retained,
        selection_comparison=selection_comparison,
    )
