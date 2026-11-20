# SPDX-License-Identifier: Apache-2.0
"""Independent route checks performed after each completed replay."""

import numpy as np


def expected_mode(previous, owners, ends, token_bytes, generation, step):
    if generation == 0 or step <= 1 or previous is None:
        return 0
    selected = previous[owners]
    if (np.any(selected[:, 0] != generation) or np.any(selected[:, 1] != step - 1)
            or np.any(selected[:, 3] != 4)):
        return 0
    return int(int(selected[:, 2].sum()) * int(token_bytes) * 32 <= int(ends.sum()) * 5)


class RouteAudit:
    def __init__(self, states):
        self.states = states
        self.previous = [None] * len(states)
        self.ends = [s.rotating.query_ends.cpu().numpy().copy().reshape(s.batch, 4) for s in states]
        self.owners = [s.rotating.row_owners.cpu().numpy().astype(int).reshape(-1) for s in states]

    def check(self, states, step, generation, *, low=None):
        modes, counts = [], []
        for i, state in enumerate(states):
            route = state.rotating.adaptive_route
            owners = self.owners[i]
            mode = int(route.mode.cpu().item())
            expected = expected_mode(self.previous[i], owners, self.ends[i], state.token_bytes, generation, step)
            if mode != expected:
                raise ValueError(f'Route mismatch at call {i}, generation {generation}, step {step}: {mode} vs {expected}')
            actual = route.feedback.cpu().numpy().copy()
            reads = route.demand.cpu().numpy().reshape(-1)
            committed = actual[owners]
            if (np.any(committed[:, 0] != generation) or np.any(committed[:, 1] != step)
                    or np.any(committed[:, 3] != 4) or np.any(reads < 0)
                    or not np.array_equal(reads, committed[:, 2])):
                raise ValueError(f'Uncommitted device demand at call {i}, generation {generation}, step {step}')
            self.previous[i] = actual
            modes.append(mode)
            counts.append(int(reads.sum()))
        return dict(generation=generation, step=step, low=modes.count(1), high=modes.count(0),
                    modes=modes, feedback_demand_per_call=counts)

    def lifecycle(self, inputs, point, graph, stream, group):
        from lean_replay import set_step
        states = self.states
        for state in states:
            state.forward.set_diagnostics(False)

        def launch(generation, step):
            for state in states:
                state.reset()
                set_step(state.rotating.adaptive_route, generation=generation, step=step)
            stream.synchronize()
            graph.replay()
            stream.synchronize()
            graph.completed()
            for state in states:
                state.check_completion()

        generation, step = point['generation'], point['step']
        launch(0, step + 1)
        for i, state in enumerate(states):
            route = state.rotating.adaptive_route
            if route.mode.item() != 0 or not np.array_equal(self.previous[i], route.feedback.cpu().numpy()):
                raise ValueError('Invalid request identity changed feedback or used LOW')

        # Force the opposite prediction once. The following decision must learn
        # the completed Get count from the mistaken round at the same addresses.
        false_reads = 512 if group == 2 else 0
        for i, state in enumerate(states):
            state.rotating.adaptive_route.feedback[1:, 2].fill_(false_reads)
            self.previous[i][1:, 2] = false_reads
        observed = []
        measured_digest = states[0].topk_set_digest
        for next_step in (step + 1, step + 2):
            launch(generation, next_step)
            result = self.check(states, next_step, generation)
            states[0].validate(inputs)
            observed.append(result['modes'])
        states[0].topk_set_digest = measured_digest
        if not all(a != b for a, b in zip(*observed)):
            raise ValueError('Fault injection failed to demonstrate opposite routes on consecutive calls')
        return dict(invalid_identity_preserved=True, misprediction_recovered=True,
                    injected_modes=observed[0], recovered_modes=observed[1])
