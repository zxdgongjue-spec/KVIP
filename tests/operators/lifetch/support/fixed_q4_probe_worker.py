# SPDX-License-Identifier: Apache-2.0
"""Single-card Q4 diagnostics using the original, frozen request and serial oracle."""

from contextlib import contextmanager
from copy import deepcopy
import json

from large_batch_pair import (
    CALLS, WARMUP, check_ordered_index, check_scores, large_account, last_row_windows,
    mutable_snapshots, read_selected, stage_sample,
)


ARCHIVE_NATIVE = "af52a02b97daddadd5d93b9d100f2ff1b03b499d"


def checked_archive(identity, serial_directory):
    base = dict(identity, native=ARCHIVE_NATIVE)
    original = serial_directory.parent / "fused/ARM_RESULT.json"
    record = json.loads(original.read_text())
    rows = identity["batch"]
    if (record["identity"] != base or record["arm"] != "fused"
            or record["checked_queries"] != CALLS * rows * 4 or record["checked_rows"] != CALLS * rows
            or record["ordered_Index_diagnostic_rows"] != rows or len(record["samples"]) != CALLS):
        raise ValueError(f"Archived fused run does not match the requested NPU/input/protocol: {original}")
    return base


@contextmanager
def restored_graph(ex, state, torch):
    stream = torch.npu.current_stream(state.device)
    stream.synchronize()
    state.reset()
    snapshots = mutable_snapshots(state, True)
    policies = deepcopy(state.policies)
    graph = torch.npu.NPUGraph()
    capture = torch.npu.Stream()
    pending = False

    def restore():
        state.policies = deepcopy(policies)
        for target, initial in snapshots:
            target.copy_(initial)
        stream.synchronize()

    def run(start=None, stop=None):
        nonlocal pending
        pending = True
        if start is not None:
            start.record(stream)
        graph.replay()
        if stop is None:
            stream.synchronize()
        else:
            stop.record(stream)
            stop.synchronize()
        pending = False

    try:
        state.forward.set_diagnostics(False)
        stream.synchronize()
        with state.lease.capture_scope(), torch.npu.stream(capture):
            pending = True
            with torch.npu.graph(graph, stream=capture):
                state.forward()
            capture.synchronize()
            pending = False
        if hasattr(state.lease, "retain_graph"):
            state.lease.retain_graph(graph)
        yield restore, run
    except BaseException as error:
        if pending:
            ex.NativeFetchSession.quarantine("fixed_q4_probe_stream_completion_unknown", error)
        raise
    finally:
        if not pending:
            graph.reset()


def check_call(state, inputs, serial, *, full=False, journal=False):
    if full:
        state.validate(inputs)
    selected, traffic = read_selected(state, inputs, True)
    check_scores(inputs, selected, serial)
    if journal:
        check_ordered_index(state, inputs.captured_layout, inputs.captured_hot_valid)
    return traffic


def clear_diagnostic(state):
    state.forward.events_src.fill_(-1)
    state.forward.events_dst.fill_(-1)
    state.forward.times.zero_()
    state.forward.terminal.times.zero_()


def cold(state, inputs, serial, args, restore, run):
    from fixed_q4_cold_profile import collect

    archive = json.loads((args.serial_result.parent / "fused/ARM_RESULT.json").read_text())
    last = last_row_windows(archive["separate_window_event_u64"], archive["separate_LI_diagnostic"])
    if not last["available"]:
        raise ValueError(f"Missing archived last-row timestamps for G{args.group}: {last}")
    mhz = state.forward.calibrate_counter_mhz()
    state.forward.set_diagnostics(True)

    def replay_once():
        restore()
        clear_diagnostic(state)
        run()
        state.check_completion()

    result = collect(state, replay_once, mhz, last["row"])
    traffic = check_call(state, inputs, serial[0], full=True, journal=True)
    result.update(archive_critical_row=last["row"], archive_cold_us=last["segments_us"]["cold_to_window"],
                  measured_traffic=traffic, checked_queries=state.batch * 4,
                  checked_KV_Index_rows=state.batch, checked_ordered_rows=state.batch)
    return result


def post_pair(state, inputs, serial, args, restore, run):
    import torch_npu
    from torch.profiler import record_function
    from fixed_q4_post_pair import PostFixture

    torch = state.torch
    fixture = PostFixture(state)
    slow = args.cohort in (1, 3, 4, 8, 12, 14)
    seed_counts = []
    for _ in range(8):
        restore()
        fixture.record()
        run()
        traffic = check_call(state, inputs, serial[0])
        seed_counts.append(traffic["final"])
        if (traffic["final"] >= 256) == slow:
            state.validate(inputs)
            fixture.freeze()
            break
    else:
        raise ValueError(f"R{args.cohort} never reproduced its archived split class; seed counts={seed_counts}")
    for policy in ("four", "one") * WARMUP:
        restore()
        fixture.set_policy(policy)
        run()
        state.check_completion()
        fixture.verify()

    # Each pair shares both the captured graph and live queues. Reverse the
    # first policy in successive pairs to remove a consistent order advantage.
    pairs = [dict(order=(["four", "one"] if (args.cohort + i) % 2 == 0 else ["one", "four"])) for i in range(CALLS)]
    receipts = []
    profiler = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False, profile_memory=False, with_stack=False)
    with profiler:
        for i, pair in enumerate(pairs):
            for policy in pair["order"]:
                restore()
                fixture.set_policy(policy)
                # Keep flag uploads and state restoration out of this graph's event.
                torch.npu.current_stream(state.device).synchronize()
                start, stop = [torch.npu.Event(enable_timing=True) for _ in range(2)]
                scope = f"fixed_q4_post_{i:02d}_{policy}"
                with record_function(scope):
                    run(start, stop)
                receipt = fixture.verify()
                traffic = check_call(state, inputs, serial[i], full=i == 0)
                if sum(receipt["per_q_tokens"]) != traffic["final"]:
                    raise ValueError("Actual ordered GET list count differs from the reported final token count")
                receipts.append(dict(pair=i, policy=policy, traffic=traffic, **receipt))
                pair[policy] = (scope, start.elapsed_time(stop) * 1000)
    path = args.output / "measured_trace.json"
    profiler.export_chrome_trace(str(path.resolve()))
    timeline = json.loads(path.read_text())
    events = timeline["traceEvents"] if isinstance(timeline, dict) else timeline
    for pair in pairs:
        for policy in pair["order"]:
            scope, elapsed = pair[policy]
            account = large_account(events, scope, elapsed, batch=state.batch)
            pair[policy] = stage_sample(account, True)

    if len({r["sha256"] for r in receipts}) != 1:
        raise ValueError("Accepted measurements do not share a single ordered GET list")
    per_q = receipts[0]["per_q_tokens"]
    threshold = sum(per_q) >= 256
    if slow != threshold:
        raise ValueError(f"R{args.cohort} fixed list has {sum(per_q)} tokens; archived trigger class changed")
    for r in receipts:
        if r["per_q_tokens"] != per_q:
            raise ValueError("Recorded per-Q counts differ between replays")
        if r["policy"] == "one" or not threshold:
            expected = [int(n != 0) for n in per_q]
            if r["per_q_cqes"] != expected:
                raise ValueError("Expected exactly one CQ receipt per nonempty Q")
        elif max(r["per_q_cqes"]) != 4:
            raise ValueError(f"Long-list control did not exercise four batches: {r['per_q_cqes']}")
    state.forward.set_diagnostics(True)
    for policy in ("four", "one"):
        restore()
        fixture.set_policy(policy)
        clear_diagnostic(state)
        run()
        check_call(state, inputs, serial[0], full=True, journal=True)
        fixture.verify()
    result = dict(pairs=pairs, receipts=receipts, worklist=dict(
        sha256=receipts[0]["sha256"], per_q_tokens=per_q, split_row=threshold,
        untimed_capture_final_counts=seed_counts),
        checked_queries_per_policy=CALLS * 4, checked_rows_per_policy=CALLS,
        full_KV_Index_checked_calls_per_policy=1, ordered_diagnostic_calls_per_policy=1)
    if not args.keep_trace:
        path.unlink()
    return result


def measure_probe(ex, proofs, serial, args, torch, states, sequence, config, directory, *, fused):
    if not fused or len(states) != 1 or len(sequence.rounds) != 1 or config.warmup != WARMUP:
        raise ValueError("Expected one fused capture and one reusable request geometry")
    state = states[0]
    terminal = state.forward.terminal
    if terminal is None or len(terminal.queue_views) != 4 * state.batch or len(set(terminal.queue_views)) != 4 * state.batch:
        raise ValueError("Expected four independent terminal queues per request")
    with restored_graph(ex, state, torch) as (restore, run):
        worker = cold if args.fixed_probe == "cold" else post_pair
        proofs.update(worker(state, sequence.inputs, serial, args, restore, run))
    return dict(correctness="PASS", failures=[])
