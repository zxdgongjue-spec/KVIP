"""Reference simultaneous permutation for unique final Index assignments.

All sources denote the input Index, not a previously overwritten physical row.
No timing here is used as a device performance estimate.
"""


def plan(sources, destinations, history, resident):
    if len(sources) != len(destinations):
        raise ValueError('Unequal record lengths')
    if len(set(sources)) != len(sources) or len(set(destinations)) != len(destinations):
        raise ValueError('Duplicate assignment')
    if any(s < 0 or s >= history for s in sources):
        raise ValueError('Source outside history')
    if any(d < 0 or d >= resident for d in destinations):
        raise ValueError('Destination outside resident')
    src_set, dst_set = set(sources), set(destinations)
    # Match the main planner's input-order closure: preserve displaced rows.
    displaced = [d for d in destinations if d not in src_set]
    freed = [s for s in sources if s not in dst_set]
    assert len(displaced) == len(freed)
    moves = list(zip(sources, destinations)) + list(zip(displaced, freed))
    assert {s for s, _ in moves} == {d for _, d in moves}
    return moves


def apply(values, moves):
    result = list(values)
    for source, destination in moves:
        result[destination] = values[source]
    return result
