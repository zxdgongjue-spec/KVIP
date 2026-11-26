# SPDX-License-Identifier: Apache-2.0
"""One restored fixed-Q4 batch per state, ten independent states per measured graph."""

from pathlib import Path
import shutil

import numpy as np

from large_batch_pair import CALLS, WARMUP, check_scores, mutable_buffers, read_selected
from reporting import atomic_json


class FrozenPackedRound:
    """All invocations start with identical contents at distinct writable addresses."""

    def __init__(self, states, inputs, serial, fused):
        self.inputs = inputs
        self.serial = serial
        self.fused = fused
        self.exact = 0
        self.traffic = []
        self.destinations = {}
        seen = set()
        first = states[0]
        # The run_path eager sanity check advanced the first state's map and LRU.
        first.reset()
        for state in states:
            if state.initial is not inputs:
                raise ValueError("Packed states were built from different input captures")
            targets = mutable_buffers(state, fused)
            self.destinations[state] = targets
            persistent = [*targets]
            if fused:
                terminal = state.forward.terminal
                if terminal is None or len(terminal.queue_views) != 4 * state.batch or (
                    np.unique(terminal.queue_views).size != 4 * state.batch
                ):
                    raise ValueError("Missing private per-Q terminal queues")
                workers, jettys = (32, (5 * state.batch + 31) // 32) if state.batch > 16 else (40, 2)
                context = terminal.owner.context
                if (context["worker_count"], context["jettys_per_core"]) != (workers, jettys):
                    raise ValueError("Unexpected packed fused transport geometry")
                if state.lease is not first.lease:
                    raise ValueError("Expected one shared stream-ordered KVAF lease")
                persistent += [state.workspace.topk, state.update.logical_output, terminal.owner.core_status]
            else:
                if state.native.owner.session.runtime is not first.native.owner.session.runtime:
                    raise ValueError("Expected one shared strong-serial GET runtime")
                persistent += [state.native.destinations, state.native.counts]
            pointers = {tensor.data_ptr() for tensor in persistent}
            if seen & pointers:
                raise ValueError("Two packed invocations share mutable device state")
            seen.update(pointers)
        # Save only one copy for all ten destinations. The live queue pointers
        # and registered outputs remain bound; queue cursors must never rewind.
        self.template = [tensor.clone() for tensor in self.destinations[first]]
        first.torch.npu.current_stream(first.device).synchronize()

    def reset(self, state):
        for target, initial in zip(self.destinations[state], self.template, strict=True):
            target.copy_(initial)

    def validate(self, state, inputs, index):
        if inputs is not self.inputs or index != len(self.traffic):
            raise ValueError("Unexpected measured packed invocation")
        selected, counts = read_selected(state, inputs, self.fused)
        self.exact += check_scores(inputs, selected, self.serial[index])
        state.validate(inputs)
        self.traffic.append(counts)


def measure_packed(ex, proofs, serial, torch, states, sequence, config, directory, *, fused, window_density=False):
    from fixed_q4_packed_timing import profile_fixed_graph_batch

    if len(states) != CALLS or len(sequence.rounds) != 1 or config.warmup != WARMUP or config.mode != "bench":
        raise ValueError("Expected a single frozen round, four graph warmups and ten independent measured states")
    frozen = FrozenPackedRound(states, sequence.inputs, serial, fused)

    launch = getattr(states[0], "large_window_launch", None)
    expected_li23 = None
    if launch:
        from large_two_cold import COLD_SYMBOL
        if any(getattr(state, "large_window_launch", None) is not launch for state in states):
            raise ValueError("The packed graph contains different experimental kernels")
        expected_li23 = COLD_SYMBOL[launch.window_policy]
        proofs.update(large_window_policy=launch.window_policy, measured_cold_symbol=expected_li23,
                      cold_build_fingerprint=launch.source_fingerprint)

    def profile(*args, **kwargs):
        result = profile_fixed_graph_batch(*args, batch=states[0].batch, expected_li23=expected_li23, **kwargs)
        proofs.update(result["packed"])
        return result

    result = ex.measure_replay_batch(torch, states, sequence, config, directory,
                                     fused=fused, profile=profile, frozen=frozen)
    if result["correctness"] == "PASS" and len(frozen.traffic) != CALLS:
        raise ValueError("Missing packed-state verification or transport counts")
    proofs.update(exact_ID_sets=frozen.exact, traffic=frozen.traffic)
    if window_density and result["correctness"] == "PASS":
        from large_batch_pair import check_ordered_index, li_intervals
        from window_density import summarize

        # A single extra invocation runs after both events and the measured-state checks.
        # Rebuild the frozen map, but preserve the live SQ/CQ cursors.
        state = states[0]
        inputs = sequence.inputs
        state.reset()
        mhz = state.forward.calibrate_counter_mhz()
        state.forward.set_diagnostics(True)
        state.forward.times.zero_()
        state.forward.events_src.fill_(-1)
        state.forward.events_dst.fill_(-1)
        state.forward.terminal.times.zero_()
        stream = torch.npu.current_stream(state.device)
        stream.synchronize()
        sibling_checks = score_spans = None
        trace_args = getattr(state, "large_window_build_args", None)
        split_trace = launch is not None and launch.window_policy == "split-role"
        if split_trace and not trace_args:
            raise ValueError("Split-role diagnostics require their dedicated trace kernel")
        original_launch = state.forward.terminal_function
        if trace_args:
            from large_two_cold import build, attach
            trace_launch = build(*trace_args, trace_sibling=True)
            if trace_launch.source_fingerprint == launch.source_fingerprint:
                raise ValueError("The timing and diagnostic cold builds must be separate")
            if getattr(launch, "hot_function", None) is not None:
                trace_launch.hot_function = launch.hot_function
            attach(state, trace_launch)
            proofs.update(diagnostic_cold_symbol=expected_li23 + "_trace",
                          diagnostic_cold_build_fingerprint=trace_launch.source_fingerprint)
        try:
            state.forward()
            stream.synchronize()
        except BaseException as error:
            ex.NativeFetchSession.quarantine("window_density_completion_uncertain", error)
            raise
        finally:
            state.forward.terminal_function = original_launch
            if trace_args:
                state.large_window_launch = launch
        state.check_completion()
        ids, traffic = read_selected(state, inputs, True)
        check_scores(inputs, ids, serial[0])
        state.validate(inputs)
        check_ordered_index(state, inputs.captured_layout, inputs.captured_hot_valid)
        events = state.rotating.events.cpu().numpy().view("<u8").tolist()
        clocks = state.forward.times.cpu().numpy()
        li = li_intervals(clocks, state.batch, mhz)
        if trace_args and not split_trace:
            count = len(events) - 1
            words = clocks[state.batch * 64:state.batch * 128].reshape(state.batch, 4, 16)
            lane_times = words[:, :, 4:4 + 4 * count].reshape(state.batch, 4, count, 4)
            sibling_checks = lane_times[:, 3, :, 2:].tolist()
            score_spans = {name: lane_times[:, q, :, :2].tolist() for name, q in (("post_owner", 0), ("sibling", 3))}
        winners = state.workspace.topk.cpu().numpy().reshape(state.batch, -1)
        sources = state.forward.events_src.cpu().numpy()
        density = summarize(events, li, state.token_bytes, winners=winners, sources=sources,
                            sibling_checks=sibling_checks, score_spans=score_spans)
        if split_trace:
            from window_density import split_role_intervals
            density["split_role"] = split_role_intervals(clocks, events, li)
        if sum(e["posted_tokens"] for e in density["epochs"]) != traffic["prefetch"]:
            raise ValueError("Doorbells disagree with the independent row GET counters")
        proofs["window_density"] = density
        state.forward.set_diagnostics(False)
    return result


def save_packed(args, identity, inputs, proofs, workers, scratch_existed):
    checked = CALLS * args.batch * 4
    if proofs["calls"] != CALLS or len(proofs["samples"]) != CALLS:
        raise ValueError("Packed profile did not contain ten complete invocations")
    summary = dict(identity=identity, arm=args.arm, correctness="PASS", get_workers=workers,
                   checked_queries=checked, checked_rows=CALLS * args.batch, full_KV_Index_checked_calls=CALLS,
                   serial_oracle=str(args.serial_result.resolve()),
                   **proofs)
    if not args.keep_trace:
        trace = args.output / "path/diagnose" / f"round-{inputs.source_round}/trace.json"
        trace.unlink(missing_ok=True)
        scratch = args.output / "export_only_prof_dir"
        if args.output.resolve() == Path.cwd() and not scratch_existed and scratch.is_dir() and not scratch.is_symlink():
            shutil.rmtree(scratch)
    atomic_json(args.output / "packed_result.json", summary)
    if "window_density" in proofs:
        density = proofs["window_density"]
        windows = density["epochs"]
        tokens = "/".join(str(v["posted_tokens"]) for v in windows)
        blocked = "/".join(str(v["rows_arrived_before_previous_end"]) for v in windows[1:])
        longest = density["longest_inter_post_us"]
        gap = "NA" if longest is None else f"{longest:.2f}"
        first_useful = density["first_useful_post_us"]
        first_text = "NA" if first_useful is None else f"{first_useful:.2f}"
        useful = "/".join(str(w["used_by_final_topk"]) for w in windows)
        concurrent = "/".join(f"{v:.2f}" for v in density["adjacent_epochs_distinct_row_service_overlap_us"])
        sibling = "/".join(str(w.get("sibling_rows_arrived_before_previous_end", "NA")) for w in windows[1:])
        scored = "/".join(str(w.get("sibling_score_overlaps_previous_service_rows", "NA")) for w in windows[1:])
        detailed = ""
        if all("mean_sibling_exposed_previous_us" in w for w in windows[1:]):
            waits = "/".join(f"{w['mean_sibling_exposed_previous_us']:.2f}" for w in windows[1:])
            scored_time = "/".join(f"{w['sibling_mean_score_overlap_previous_service_us']:.2f}" for w in windows[1:])
            detailed = f" sibling_mean_exposed_wait_us={waits} sibling_mean_score_span_overlap_us={scored_time}"
        if "split_role" in density:
            roles = density["split_role"]
            row = roles["rows"][roles["last_li3_row"]]
            fmt = lambda values: "/".join(f"{v:.2f}" for v in values)
            print(f"G{args.group} BS{args.batch} SPLIT_ROLE_DIAG last_LI3_row={roles['last_li3_row']} "
                  f"last_window_row={roles['last_window_row']} score3_us={fmt(row['score_us'])} "
                  f"wait_ready3_us={fmt(row['wait_ready_us'])} "
                  f"wait_for_unfinished_score3_us={fmt(row['wait_for_unfinished_score_us'])} "
                  f"service4_us={fmt(row['service_us'])} "
                  f"score3_overlap_service_us={fmt(row['score_overlapped_by_service_us'])} "
                  f"post_window_to_LI3_end_us={row['post_window_to_li3_end_us']:.2f}")
        print(f"G{args.group} BS{args.batch} WINDOW_DIAG posted_tokens={tokens} "
              f"useful_tokens={useful} first_useful_post_us={first_text} "
              f"post_owner_rows_arrived_before_previous_end={blocked} "
              f"sibling_rows_arrived_before_previous_end={sibling} sibling_score_span_overlaps_previous_service_rows={scored} "
              f"distinct_row_service_overlap_us={concurrent} "
              f"longest_gap_between_posts_us={gap}{detailed}")
    print(f"CHECK G{args.group} BS{args.batch} {args.arm} TopK_score_vs_strong_serial=PASS "
          f"queries={checked}/{checked} exact_ID_sets={proofs['exact_ID_sets']}/{checked} "
          f"KV_and_Index_rows={CALLS * args.batch}/{CALLS * args.batch} get_workers={workers}")
    print(f"TIMING G{args.group} BS{args.batch} {args.arm} graph_calls={CALLS} "
          f"event_total_us={proofs['graph_event_us']:.2f} graph_mean_us={proofs['graph_mean_us']:.2f} "
          f"LI_mean_us={proofs['mean_us']['LI']:.2f} tail_mean_us={proofs['mean_us']['tail']:.2f}")
