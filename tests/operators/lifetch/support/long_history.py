# SPDX-License-Identifier: Apache-2.0
"""Fused eager and graph boundary check with a closed-form score oracle."""

from types import SimpleNamespace

import numpy as np
from reporting import atomic_json
from workloads import ReplayPagePlan, ReplayRound


def max_history_sequence(directory, history):
    batch, resident, heads = 16, 16384, 32
    # Four decode queries share an aligned Index allocation. The serving limit
    # counts the context before the three speculative positions are appended.
    last_end = {524288: 524288, 524544: 524291}[history]
    ends = np.arange(last_end - 3, last_end + 1, dtype=np.int32)
    positions = np.arange(history, dtype=np.int32)
    keys = np.zeros((history, 128), dtype=np.uint8)
    queries = np.zeros((4, heads, 128), dtype=np.uint8)
    scores = np.full((4, history), 0x8000, dtype=np.uint16)  # sortable BF16 zero
    for q, end in enumerate(ends):
        queries[q, :, q] = 0x38  # All 32 heads read just this column, with unit weights and scales.
        hot = np.arange(q * 1536, (q + 1) * 1536)
        # Exercise both sides of each address boundary and the final physical page.
        cold = [np.arange(bound - 64 + q, bound + 64 + q) for bound in (65536, 131072, 262144)]
        # Leave distinct high-address repair requests for the four Qs, plus
        # the last valid token of each Q. The joint end-of-sequence group
        # exceeds one prefetch window's 256-token allowance.
        last_block = np.arange(end - (q + 1) * 128, end - q * 128 - 1)
        wanted = np.concatenate([hot, *cold, last_block, [end - 1]])
        keys[wanted, q] = 0x38
        scores[q, wanted] = 0xC200  # sortable BF16(+32)
        # Promote the upper half of the 2^18 boundary band into the
        # prefetch rank as soon as these scores are visible.
        high = cold[-1][cold[-1] >= 262144]
        keys[high, q] = 0x40
        scores[q, high] = 0xC280  # sortable BF16(+64)
        # Higher-scoring keys beyond each Q's causal end must still be excluded.
        keys[end:, q] = 0x48
        scores[q, end:] = 0xC300  # sortable BF16(+128)

    def rows(value):
        return np.broadcast_to(value, (batch, *value.shape))

    geometry = dict(
        batch=batch, history=history, resident=resident, heads=heads, token_bytes=656, output_stride=resident + 384
    )
    manifest = directory / "max_history_input.json"
    atomic_json(
        manifest,
        dict(geometry=geometry, causal_ends=ends.tolist(), cold_winners_per_q=512, required_maximum_id=last_end - 1),
    )
    page_count = history // 128
    # The last page is private to detect cross-row output errors. Shared prefix pages bound DDR size.
    last_page = (last_end - 1) // 128
    pages = tuple(tuple(None if page == last_page else (0, page) for page in range(page_count)) for _ in range(batch))
    round_input = ReplayRound(
        1,
        1,
        tuple((0, str(i), i) for i in range(batch)),
        geometry,
        rows(keys),
        rows(np.ones(history, np.float32)),
        rows(queries),
        rows(np.full((4, heads), 0x3F80, np.uint16)),
        rows(np.ones((4, heads), np.float32)),
        rows(ends),
        np.zeros((batch, 2), np.int64),
        rows(np.ones(history, np.uint8)),
        pages,
        rows(positions),
        rows(np.ones(resident, np.int32)),
        rows(np.arange(resident - 2048, resident, dtype=np.int32)),
        (str(manifest),),
        rows(scores),
        rows(positions[None, :] < ends[:, None]),
    )

    class Sequence:
        rounds = ((SimpleNamespace(source_round=1, manifest_path=manifest),),)

        def __iter__(self):
            yield round_input

    return Sequence()


def check_max_history(config, directory, run_path):
    sequence = max_history_sequence(directory, config.history)
    inputs = next(iter(sequence))
    required_max_id = int(inputs.query_ends.max()) - 1
    pages = ReplayPagePlan(sequence)
    observed = []
    high_counts = {}

    def check_outputs(state, inputs):
        if config.transport == "urma" and state.forward.terminal is None:
            raise RuntimeError("512K test did not execute the four-query URMA terminal")
        # run_path already checked score equivalence, unique slots/IDs, all KV bytes,
        # the persistent Index columns and (for URMA) each private Get completion.
        ids = state.update.logical_output.cpu().numpy().reshape(state.batch, 4, 2048)
        if not np.array_equal(ids.max(axis=2), inputs.query_ends - 1):
            raise AssertionError("Missing last visible token in the 512K winners")
        if not np.all((ids >= state.resident).sum(axis=2) == 512):
            raise AssertionError("Missing 512 cold winners per Q")
        if len(observed) == 2 and config.transport == "urma":
            # Only the separate diagnostic replay saves source positions that
            # the kernel retires. Use the original order of the recorded pairs.
            # The fixture admits only cold sources, so no displacement chains;
            # each recorded source must exchange its row with exactly one hot destination.
            layout = state.storage.logical_ids[1:].cpu().numpy()
            sources = state.forward.events_src.cpu().numpy()
            destinations = state.forward.events_dst.cpu().numpy()
            counts = state.rotating.change_count.cpu().numpy()
            prefetch_counts = state.rotating.rows[:, 9].cpu().numpy()
            final_counts = state.rotating.final_posts[:, :, 0].cpu().numpy().sum(axis=1)
            if not np.array_equal(counts, prefetch_counts + final_counts):
                raise AssertionError("Index journal does not cover all issued requests")
            high_counts.update(prefetch=0, final=0)
            for row, n in enumerate(counts):
                if not 0 < n <= sources.shape[1]:
                    raise AssertionError(f"row {row}: invalid admission count {n}")
                src, dst = sources[row, :n], destinations[row, :n]
                if (
                    np.any(src < state.resident)
                    or np.any(src >= state.history)
                    or np.any(dst < 0)
                    or np.any(dst >= state.resident)
                    or np.unique(src).size != n
                    or np.unique(dst).size != n
                ):
                    raise AssertionError(f"row {row}: invalid or duplicate admission")
                prefetch = prefetch_counts[row]
                high_counts["prefetch"] += int(np.sum(src[:prefetch] >= 262144))
                high_counts["final"] += int(np.sum(src[prefetch:] >= 262144))
                expected = state.input_layout[row].copy()
                expected[src], expected[dst] = expected[dst], expected[src]
                if not np.array_equal(layout[row], expected):
                    raise AssertionError(f"row {row}: ordered Index differs from admitted exchanges")
        observed.append(int(ids.max()))
        if len(observed) == 2 and config.transport == "urma":
            state.forward.set_diagnostics(True)  # After the uninstrumented graph verification.

    result = run_path(config, directory / "fused", sequence, pages, fused=True, round_check=check_outputs)
    passed = result["correctness"] == "PASS" and observed == [required_max_id] * 3
    detail = dict(
        history=config.history,
        query_ends=inputs.query_ends[0].tolist(),
        resident=16384,
        batch=16,
        queries=4,
        max_token_id=observed,
        ddr_pages=pages.page_count,
        issued_at_or_above_262144=high_counts,
        ordered_index="checked on diagnostic replay" if config.transport == "urma" else "not collected",
        failures=result["failures"],
    )
    atomic_json(directory / "max_history_result.json", dict(passed=passed, **detail))
    atomic_json(
        directory / "checks.json",
        [dict(case=config.check_case, correctness="PASS" if passed else "FAIL", detail=str(detail))],
    )
    return int(not passed)
