# SPDX-License-Identifier: Apache-2.0
"""Regression checks for the reference used to audit on-device route history."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'support'))
from ranked_route_replay import expected_mode


def test_last_success_and_current_query_geometry_select_route():
    owners = np.arange(1, 17)
    feedback = np.tile([7, 10, 100, 4], (17, 1))
    ends = np.full((16, 4), 131072)
    assert expected_mode(None, owners, ends, 656, 7, 11) == 0
    assert expected_mode(feedback, owners[::-1], ends, 656, 7, 11) == 1
    assert expected_mode(feedback, owners, ends // 2, 656, 7, 11) == 0
    assert expected_mode(feedback, owners, ends, 2624, 7, 11) == 0
    assert expected_mode(feedback, owners, ends, 656, 8, 11) == 0
    assert expected_mode(feedback, owners, ends, 656, 7, 12) == 0
    feedback[16, 2] = 5000
    assert expected_mode(feedback, owners, ends, 656, 7, 11) == 0


def test_rejection_of_incomplete_predecessor_forces_whole_graph_high():
    owners = np.arange(1, 17)
    ends = np.full((16, 4), 130048)
    feedback = np.tile([3, 12, 20, 4], (17, 1))
    feedback[2, 3] = 1
    assert expected_mode(feedback, owners, ends, 656, 3, 13) == 0



def test_independent_policy_rejects_corrupt_demand_and_out_of_range_queries():
    owners = np.arange(1, 17)
    feedback = np.tile([3, 12, 20, 4], (17, 1))
    ends = np.full((16, 4), 130048)
    feedback[8, 2] = -1
    assert expected_mode(feedback, owners, ends, 656, 3, 13) == 0
    feedback[8, 2] = 20
    ends[8, 2] = 131329
    assert expected_mode(feedback, owners, ends, 656, 3, 13) == 0

def test_fault_injection_invalidates_high_certificate_before_replay():
    from types import SimpleNamespace
    from ranked_route_replay import RouteAudit

    class Buffer(np.ndarray):
        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self)

        def fill_(self, value):
            self.fill(value)

    def buffer(a):
        return np.asarray(a, dtype=np.int64).view(Buffer)

    owners = buffer(np.arange(1, 17))
    ends = buffer(np.full((16, 4), 131072))
    route = SimpleNamespace(requests=buffer(np.zeros((16, 4))),
                            feedback=buffer(np.tile([2, 6, 512, 4], (17, 1))),
                            demand=buffer(np.full(16, 512)), mode=buffer(0),
                            used_high_bound=buffer(0), high_bound=buffer([[8192], [0]]))
    route.set_requests = lambda rows: np.copyto(route.requests, rows)
    route.invalidate_high_bound = lambda: route.high_bound[0].fill(0)
    noop = lambda *args: None
    state = SimpleNamespace(rotating=SimpleNamespace(adaptive_route=route, query_ends=ends, row_owners=owners),
                            batch=16, resident=16384, history=131328, token_bytes=656,
                            forward=SimpleNamespace(set_diagnostics=noop), reset=noop,
                            check_completion=noop, validate=noop, topk_set_digest='measured')
    used = []

    def replay():
        generation, step = route.requests[0, 1:3]
        hit = bool(route.high_bound[0, 0] * 656 * 32 > ends.sum() * 5)
        used.append(hit)
        route.used_high_bound.fill(hit)
        route.mode.fill(0 if hit else expected_mode(route.feedback, owners, ends, 656, generation, step))
        if generation:
            route.feedback[owners] = [generation, step, 512, 4]
            route.high_bound[0, 0] = 8192

    audit = RouteAudit([state])
    audit.previous = [route.feedback.numpy().copy()]
    graph = SimpleNamespace(replay=replay, completed=noop)
    result = audit.lifecycle(None, dict(generation=2, step=6), graph, SimpleNamespace(synchronize=noop), group=7)
    assert used == [True, False, True]
    assert result['injected_modes'] == [1] and result['recovered_modes'] == [0]
