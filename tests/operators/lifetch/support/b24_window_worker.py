# SPDX-License-Identifier: Apache-2.0
"""Time ten independent states with the archived hot and tail. Trace window clocks afterwards."""
from argparse import Namespace
from math import isclose
from pathlib import Path

import numpy as np

from b24_window_policy import ADVANCE, MEDIUM, WINDOW_CEILINGS, install, policy, trace_lanes


def diagnostic(ex, torch, state, inputs, oracle, mode):
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
    density = summarize(state.rotating.events.cpu().numpy().view(np.uint64).tolist(), li, state.token_bytes,
                        winners=state.workspace.topk.cpu().numpy().reshape(state.batch, -1),
                        sources=forward.events_src.cpu().numpy(), sibling_checks=spans[:, 1, :, 2:].tolist(),
                        score_spans={k: spans[:, lane, :, :2].tolist() for lane, k in enumerate(("post_owner", "sibling"))})
    if counts["posted_per_window"] != [e["posted_tokens"] for e in density["epochs"]]:
        raise ValueError("Window doorbells disagree with the row counters")
    return dict(actual_ranks=list(ranks), checked_rank_queries=state.batch*4, window_density=density,
                diagnostic_cold_build_fingerprint=trace.source_fingerprint)


def run(a):
    import large_batch_pair
    import large_two_cold
    import fixed_q4_packed_worker as packed
    from four_rank_accept_worker import terminal_image
    from reporting import atomic_json
    import json

    import os
    local = os.environ.get("PEER_G7_X4_LOCAL") == "1"
    if local and (a.mode not in ("window-x4-early",ADVANCE,MEDIUM) or a.group != 7 or a.batch not in (28, 32)):
        raise ValueError("Unexpected local window shape")
    original = None if local else json.loads((a.ref / f"g{a.group}/bs{a.batch}/block1/fused/packed_result.json").read_bytes())
    is_medium = a.mode == MEDIUM
    install(large_two_cold, ADVANCE if is_medium else a.mode)
    if is_medium:
        import medium_tail
        medium_tail.install(large_two_cold, a.sdk / "build-npu")
    regular_traffic = large_batch_pair.window_traffic
    if a.mode in (ADVANCE, MEDIUM):
        def earned_traffic(counters, counts, grants=(), *, carry_credit=False):
            traffic = regular_traffic(counters, counts, grants, carry_credit=True)
            if grants and np.any(counters[:,7:] > np.asarray(WINDOW_CEILINGS)):
                raise ValueError("Terminal or intermediate window exceeded its individual ceiling")
            return traffic
        large_batch_pair.window_traffic = earned_traffic
    regular_measure = packed.measure_packed

    def measure(ex, proof, serial, torch, states, sequence, *rest, **kwargs):
        if kwargs.get("window_density") or not kwargs["fused"]:
            raise ValueError("Expected a natural timed ten-call graph")
        image = medium_tail.image(states[0]) if is_medium else terminal_image(states[0], a.sdk)
        if original and (image["cold_build"].split("/")[0] != original["cold_build_fingerprint"].split("/")[0]
                or image["cold_build"] == original["cold_build_fingerprint"]):
            raise ValueError("Expected the archived regular library with a distinct isolated LI23 image")
        result = regular_measure(ex, proof, serial, torch, states, sequence, *rest, **kwargs)
        if result["correctness"] == "PASS":
            if is_medium:
                proof["measured_private_final_CQEs"] = medium_tail.receipts(states)
            proof.update(diagnostic(ex, torch, states[0], sequence.inputs, serial[0], a.mode))
            proof["original_library"] = image["cold_build"].split("/")[0]
            proof["tail_text"] = image["tail_text"]
            if is_medium:
                proof["original_tail_text"] = image["old_tail_text"]
                proof["medium_tail"] = True
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
        large_batch_pair.window_traffic = regular_traffic

    result = json.loads((a.out / "packed_result.json").read_bytes())
    if ((original and result["identity"] != original["identity"]) or result["correctness"] != "PASS"
            or result["checked_queries"] != 40*a.batch or result["checked_rows"] != 10*a.batch
            or result["get_workers"] != 32 or result["calls"] != 10 or result["full_KV_Index_checked_calls"] != 10
            or result["window_grants_per_row"] != list(policy(a.mode)[1])
            or not isclose(10 * result["graph_mean_us"], result["graph_event_us"], rel_tol=1e-8)):
        raise ValueError("Input, geometry, budget, correctness or timing protocol differs from the original")
    atomic_json(a.out / "window-study.json", dict(mode=a.mode, reference=None if local else str(a.ref)))
