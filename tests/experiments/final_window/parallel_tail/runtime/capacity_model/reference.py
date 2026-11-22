"""Capacity and parallel Index correctness oracles; no host timing estimates.

Overflow destinations must exclude ALL final hot hits, including later Qs.
An Index permutation reads the old state before any worker writes its partition.
"""


def extend_destinations(lru, required, resident, protected):
    if len(set(lru)) != len(lru) or any(s < 0 or s >= resident for s in lru):
        raise ValueError("Invalid LRU destinations")
    if required < 0:
        raise ValueError("Negative demand")
    if required <= len(lru):
        return list(lru[:required])
    # The prefix is already reserved by the producer. Only new destinations
    # are selected here; never evict a later Q's hot hit to repair an earlier Q.
    result = list(lru)
    unavailable = set(lru) | set(protected)
    for slot in range(resident):
        if slot not in unavailable:
            result.append(slot)
            if len(result) == required:
                return result
    raise ValueError("Insufficient unprotected resident slots")


def parallel_permutation(sources, destinations, workers=4):
    """Keep the production planner's input-order displaced-row closure.

    Each returned worker partition contains (old source, new destination).
    All partitions must gather their payload before ANY partition commits.
    Independent swaps and cycles therefore use the same parallel schedule.
    """
    if workers <= 0 or len(sources) != len(destinations):
        raise ValueError("Invalid geometry")
    src_set, dst_set = set(sources), set(destinations)
    if len(src_set) != len(sources) or len(dst_set) != len(destinations):
        raise ValueError("Duplicate assignment")
    displaced = [d for d in destinations if d not in src_set]
    freed = [s for s in sources if s not in dst_set]
    moves = list(zip(sources, destinations)) + list(zip(displaced, freed))
    return [moves[len(moves)*q//workers:len(moves)*(q+1)//workers]
            for q in range(workers)]


def gather_partitions(values, partitions):
    return [[(destination, values[source]) for source, destination in part]
            for part in partitions]


def commit_partitions(values, snapshots, order):
    result = list(values)
    for worker in order:
        for destination, value in snapshots[worker]:
            result[destination] = value
    return result
