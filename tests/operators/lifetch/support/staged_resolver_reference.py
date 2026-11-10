# SPDX-License-Identifier: Apache-2.0
"""Golden model for a sequential-Q resolver; no device timing or transport."""


def resolve(queries, resident, lru_slots, used, prefetch_map, hot_valid):
    """Return per-Q slots and unique repair records, protecting all Q hits.

    prefetch_map maps logical tokens to physical hot slots. Slots in the first
    ``used`` LRU positions are conservatively treated as occupied by prefetch.
    A reserved hot token without a prefetch mapping needs a fresh repair.
    """
    if len(queries) != 4 or not 0 <= used <= len(lru_slots):
        raise ValueError('requires four queries and a valid prefetch prefix')
    if len(hot_valid) != resident or len(set(lru_slots)) != len(lru_slots):
        raise ValueError('invalid hot validity or repeated LRU slots')
    if any(slot < 0 or slot >= resident for slot in lru_slots):
        raise ValueError('LRU slot outside resident')
    reserved = set(lru_slots)
    protected = set()
    for query in queries:
        for token in query:
            if token < 0:
                continue
            if token in prefetch_map:
                slot = prefetch_map[token]
                if not 0 <= slot < resident:
                    raise ValueError('invalid prefetched physical slot')
                protected.add(slot)
            elif token < resident and hot_valid[token] and token not in reserved:
                protected.add(token)
    available = ([slot for slot in lru_slots[used:] if slot not in protected]
                 + [slot for slot in lru_slots[:used] if slot not in protected]
                 + [slot for slot in range(resident) if slot not in reserved and slot not in protected])
    assigned = dict(prefetch_map)
    records, outputs, phase_counts = [], [], []
    for query in queries:
        out, before = [], len(records)
        for token in query:
            if token < 0:
                out.append(-1)
            elif token in assigned:
                out.append(assigned[token])
            elif token < resident and hot_valid[token] and token not in reserved:
                out.append(token)
            else:
                if len(records) >= len(available):
                    raise ValueError('insufficient unprotected hot slots')
                slot = available[len(records)]
                assigned[token] = slot
                records.append((token, slot))
                out.append(slot)
        outputs.append(out)
        phase_counts.append(len(records) - before)
    return outputs, records, phase_counts
