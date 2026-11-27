# SPDX-License-Identifier: Apache-2.0
"""Time ten independent states with the archived hot and tail. Trace window clocks afterwards."""
from argparse import Namespace
from math import isclose
from pathlib import Path

import numpy as np

from b24_window_policy import install, policy, trace_lanes


def diagnostic(ex, torch, state, inputs, oracle, mode, *, save_clocks=False):
    from large_batch_pair import check_ordered_index, check_scores, li_intervals, read_selected
    from large_two_cold import attach, build
    from window_density import summarize

    state.reset()
    forward = state.forward
    mhz = forward.calibrate_counter_mhz()
    state.rotating.rank_progress.zero_()
    forward.times.zero_()
    forward.events_src.fill_(-1)
    forward.events_dst.fill_(-1)
    forward.terminal.times.zero_()
    stream = torch.npu.current_stream(state.device)
    stream.synchronize()
    ordinary = state.large_window_launch
    trace = build(*state.large_window_build_args, trace_sibling=True)
    if ordinary.source_fingerprint == trace.source_fingerprint:
        raise ValueError("Diagnostic clocks are present in the timed cold image")
    attach(state, trace)
    try:
        forward.set_diagnostics(True)
        forward()
        stream.synchronize()
        forward.set_diagnostics(False)
    except BaseException as error:
        ex.NativeFetchSession.quarantine("window_study_completion_uncertain", error)
        raise
    finally:
        attach(state, ordinary)
    state.check_completion()
    ids, counts = read_selected(state, inputs, True)
    check_scores(inputs, ids, oracle)
    state.validate(inputs)
    check_ordered_index(state, inputs.captured_layout, inputs.captured_hot_valid)
    ranks, grants = policy(mode)
    raw = state.rotating.rank_progress.cpu().numpy().view(np.uint64)
    ends = state.rotating.query_ends.cpu().numpy()
    scheduled = raw[:, :, :len(ranks)]
    coverage = (scheduled >> 18) & ((1 << 18) - 1)
    if (raw.shape != (state.batch, 4, 32) or ends.shape != (state.batch, 4)
            or not np.all(scheduled >> 36 == np.asarray(ranks))
            or np.any(coverage[:, :, 1:] < coverage[:, :, :-1]) or np.any(coverage > ends[:, :, None])):
        raise ValueError("Ranks actually consumed or covered positions disagree with the window policy")
    spans = trace_lanes(raw, len(grants)-1)
    li = li_intervals(forward.times.cpu().numpy(), state.batch, mhz)
    events = state.rotating.events.cpu().numpy().view(np.uint64).tolist()
    scores = {k: spans[:, lane, :, :2].tolist() for lane, k in enumerate(("post_owner", "sibling"))}
    predecessor = spans[:, 1, :, 2:].tolist()
    density = summarize(events, li, state.token_bytes,
                        winners=state.workspace.topk.cpu().numpy().reshape(state.batch, -1),
                        sources=forward.events_src.cpu().numpy(), sibling_checks=predecessor, score_spans=scores)
    if counts["posted_per_window"] != [e["posted_tokens"] for e in density["epochs"]]:
        raise ValueError("Window doorbells disagree with the row counters")
    clocks = dict(separate_LI_diagnostic=li, separate_score_span_cycles=scores,
                  separate_sibling_predecessor_cycles=predecessor, separate_window_event_u64=events, checked_traffic=counts) if save_clocks else {}
    return dict(**clocks, actual_ranks=list(ranks), checked_rank_queries=state.batch*4, window_density=density,
                diagnostic_cold_build_fingerprint=trace.source_fingerprint)


def run(a):
    import large_batch_pair
    import large_two_cold
    import fixed_q4_packed_worker as packed
    from four_rank_accept_worker import terminal_image
    from reporting import atomic_json
    import json

    original = json.loads((a.ref / f"g{a.group}/bs{a.batch}/block1/fused/packed_result.json").read_bytes())
    install(large_two_cold, a.mode)
    regular_measure = packed.measure_packed

    def measure(ex, proof, serial, torch, states, sequence, *rest, **kwargs):
        if kwargs.get("window_density") or not kwargs["fused"]:
            raise ValueError("Expected a natural timed ten-call graph")
        image = terminal_image(states[0], a.sdk)
        if (image["cold_build"].split("/")[0] != original["cold_build_fingerprint"].split("/")[0]
                or image["cold_build"] == original["cold_build_fingerprint"]):
            raise ValueError("Expected the archived regular library with a distinct isolated LI23 image")
        result = regular_measure(ex, proof, serial, torch, states, sequence, *rest, **kwargs)
        if result["correctness"] == "PASS":
            proof.update(diagnostic(ex, torch, states[0], sequence.inputs, serial[0], a.mode))
            proof["original_library"] = image["cold_build"].split("/")[0]
            proof["tail_text"] = image["tail_text"]
        return result

    packed.measure_packed = measure
    try:
        large_batch_pair.run(Namespace(trace=a.trace, group=a.group, batch=a.batch, round=1, cohort=0,
            prepared=a.bank, arm="fused", kvaf_root=a.sdk, kvaf_server=a.server, output=a.out,
            packed_ten=True, captured_score_oracle=True, large_window_policy="four-ranks",
            compare_packed_candidate=True, carry_credit=False, window_density=False,
            serial_result=None, reuse_serial_archive=None, compare_archived_pair=False, keep_trace=False))
    finally:
        packed.measure_packed = regular_measure

    result = json.loads((a.out / "packed_result.json").read_bytes())
    if (result["identity"] != original["identity"] or result["correctness"] != "PASS"
            or result["checked_queries"] != 40*a.batch or result["checked_rows"] != 10*a.batch
            or result["get_workers"] != 32 or result["calls"] != 10 or result["full_KV_Index_checked_calls"] != 10
            or result["window_grants_per_row"] != list(policy(a.mode)[1])
            or not isclose(10 * result["graph_mean_us"], result["graph_event_us"], rel_tol=1e-8)):
        raise ValueError("Input, geometry, budget, correctness or timing protocol differs from the original")
    atomic_json(a.out / "window-study.json", dict(mode=a.mode, reference=str(a.ref)))
