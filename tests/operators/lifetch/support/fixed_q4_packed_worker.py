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


def measure_packed(ex, proofs, serial, torch, states, sequence, config, directory, *, fused):
    from fixed_q4_packed_timing import profile_fixed_graph_batch

    if len(states) != CALLS or len(sequence.rounds) != 1 or config.warmup != WARMUP or config.mode != "bench":
        raise ValueError("Expected a single frozen round, four graph warmups and ten independent measured states")
    frozen = FrozenPackedRound(states, sequence.inputs, serial, fused)

    def profile(*args, **kwargs):
        result = profile_fixed_graph_batch(*args, batch=states[0].batch, **kwargs)
        proofs.update(result["packed"])
        return result

    result = ex.measure_replay_batch(torch, states, sequence, config, directory,
                                     fused=fused, profile=profile, frozen=frozen)
    if result["correctness"] == "PASS" and len(frozen.traffic) != CALLS:
        raise ValueError("Missing packed-state verification or transport counts")
    proofs.update(exact_ID_sets=frozen.exact, traffic=frozen.traffic)
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
    print(f"CHECK G{args.group} BS{args.batch} {args.arm} TopK_score_vs_strong_serial=PASS "
          f"queries={checked}/{checked} exact_ID_sets={proofs['exact_ID_sets']}/{checked} "
          f"KV_and_Index_rows={CALLS * args.batch}/{CALLS * args.batch} get_workers={workers}")
    print(f"TIMING G{args.group} BS{args.batch} {args.arm} graph_calls={CALLS} "
          f"event_total_us={proofs['graph_event_us']:.2f} graph_mean_us={proofs['graph_mean_us']:.2f} "
          f"LI_mean_us={proofs['mean_us']['LI']:.2f} tail_mean_us={proofs['mean_us']['tail']:.2f}")
