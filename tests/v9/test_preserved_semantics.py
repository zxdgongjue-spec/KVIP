"""Pin V9-specific H0 and retained-extra semantics without changing the oracle."""

import numpy as np
from contract_reference import eligible_victims, frozen_hot, plan_row


def test_victim_hot_and_equal_threshold_are_preserved_as_v9_profile():
    forward = np.full(8192, -1, np.int32)
    reverse = np.full(6144, -1, np.int32)
    forward[9], reverse[0] = 0, 9
    victims = np.arange(2048, dtype=np.int32)
    hot = frozen_hot(forward, reverse, victims, 8192, 6144)
    assert hot[9] == 0  # A legacy bitmap-exclusion hit test would exclude it.
    scores = np.zeros((4, 8192), np.float32)
    assert 0 not in eligible_victims(scores, hot, victims)
    selected = np.full((4, 2048), -1, np.int32)
    selected[:, 0] = 9
    plan = plan_row(selected, hot, victims, 6144, 0, reservations={8191: 1})
    assert not plan.misses
    assert plan.extra == {8191: 1} and not plan.discarded
    assert np.all(plan.destination[:, 0] == 0)


def test_repeated_cold_token_is_one_logical_miss_across_four_queries():
    selected = np.full((4, 2048), -1, np.int32)
    selected[:, 0] = 8000
    plan = plan_row(selected, {}, np.arange(2048, dtype=np.int32), 6144, 0)
    assert len(plan.misses) == 1 and 8000 in plan.misses
    assert len(set(map(int, plan.destination[:, 0]))) == 1
