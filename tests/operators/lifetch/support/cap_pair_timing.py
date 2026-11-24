# SPDX-License-Identifier: Apache-2.0
"""Time one capacity-bound graph with one restored device state per replay."""
from functools import partial
from pathlib import Path


def install(execution, config):
    # run_path allocates one complete Index, score workspace and registered output
    # per iteration. Keep one allocation for both capacities and repeat the graph.
    calls = config.iterations
    if calls != 20 or config.warmup != 4:
        raise ValueError("Capacity comparison expects 4 warmups and 20 measured calls")
    execution.measure_replay_batch = partial(measure, execution, calls)
    config.iterations = 1


def measure(execution, calls, torch, states, sequence, config, directory, *, fused):
    from performance import profile_graph_batch
    from reporting import atomic_json, timing_summary

    if not fused or len(states) != 1 or len(sequence.rounds) != 1 or config.mode != "diagnose":
        raise ValueError("Capacity timing requires one fused state and one frozen input")
    state, = states
    inputs = next(iter(sequence))
    target = Path(directory) / "diagnose" / f"round-{inputs.source_round}"
    stream = torch.npu.current_stream(state.device)
    capture = torch.npu.Stream()
    graph = torch.npu.NPUGraph()
    checks = execution.CorrectnessRecorder()
    samples = []
    pending = False

    def restore():
        # reset() restores the captured hot rows, Index, physical order, LRU,
        # query and validity. Its uploads and synchronization precede both events.
        state.reset()
        stream.synchronize()

    class TimedGraph:
        def replay(self):
            nonlocal pending
            pending = True
            graph.replay()

        def completed(self):
            nonlocal pending
            pending = False

    timed = TimedGraph()

    def validate(label):
        if not checks.check(label, lambda: state.validate(inputs)):
            raise ValueError(f"{label}: {checks.failures[-1]['error']}")

    try:
        # run_path finished a real eager Get for this state before calling us;
        # its output is registered and transport metadata are ready for capture.
        state.forward.set_diagnostics(False)
        restore()
        with state.lease.capture_scope(), torch.npu.stream(capture):
            pending = True
            with torch.npu.graph(graph, stream=capture):
                state.forward()
            capture.synchronize()
            timed.completed()
        if hasattr(state.lease, "retain_graph"):
            state.lease.retain_graph(graph)

        for i in range(config.warmup):
            restore()
            timed.replay()
            stream.synchronize()
            timed.completed()
            validate(f"warmup{i}")

        for i in range(calls):
            restore()
            profile = profile_graph_batch(torch, timed, calls=1, folder=target / f"call-{i:02d}", stream=stream)
            validate(f"measured/input{i}/round{inputs.source_round}")
            item, = profile["samples"]
            samples.append(dict(round=i, source_round=inputs.source_round, us=item["us"],
                                accounting=dict(item, invocation=i, event_us=profile["accounting"]["event_us"])))

        state.forward.set_diagnostics(True)
        for i in range(calls):
            restore()
            state.forward.times.zero_()
            state.rotating.rank_progress.zero_()
            stream.synchronize()
            timed.replay()
            stream.synchronize()
            timed.completed()
            validate(f"detail/input{i}/round{inputs.source_round}")
    except BaseException as error:
        if pending:
            execution.NativeFetchSession.quarantine("capacity_replay_completion_uncertain", error)
        raise
    finally:
        if not pending:
            state.forward.set_diagnostics(False)
            graph.reset()

    timing = timing_summary([s["accounting"]["event_us"] for s in samples])
    timing.update(graph_calls=calls, measured_graphs=calls, event_mean_us=timing["mean_us"],
                  timing_protocol="one_state_individual_graphs")
    atomic_json(target / "full_attribution.json", dict(graph_calls=calls, samples=samples, timing=timing))
    return dict(rounds={inputs.source_round: timing}, all_rounds=timing, correctness=checks.status,
                failures=checks.failures)
