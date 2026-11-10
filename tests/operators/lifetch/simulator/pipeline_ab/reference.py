"""Host oracle for ordered Index-row exchanges and deferred metadata commit.

Original row IDs are stable only for the lifetime of a single forward. Events
carry original row IDs and current physical destinations. The sparse maps track
both directions; untouched entries are identity. No network/CQ is modeled.
"""
from dataclasses import dataclass, field


@dataclass
class SparsePermutation:
    history: int
    resident: int
    row_at: dict[int, int] = field(default_factory=dict)
    position_of: dict[int, int] = field(default_factory=dict)

    def swap(self, source_original: int, destination: int) -> tuple[int, int]:
        if not 0 <= source_original < self.history:
            raise ValueError("source original row is outside history")
        if not 0 <= destination < self.resident:
            raise ValueError("destination is outside hot")
        current_source = self.position_of.get(source_original, source_original)
        displaced = self.row_at.get(destination, destination)
        if current_source == destination:
            return current_source, displaced
        self.position_of[source_original] = destination
        self.position_of[displaced] = current_source
        self.row_at[destination] = source_original
        self.row_at[current_source] = displaced
        return current_source, displaced

    def commit(self) -> dict[int, int]:
        return {pos: source for pos, source in self.row_at.items() if pos != source}


def materialized_oracle(history, resident, batches):
    rows = list(range(history))
    # Per-round original IDs stand for Index identities, not logical-ID values.
    inverse = list(range(history))
    milestones = []
    for events in batches:
        for token, dst in events:
            src = inverse[token]
            old = rows[dst]
            rows[src], rows[dst] = old, token
            inverse[token], inverse[old] = dst, src
        milestones.append(tuple(rows[:resident]))
    return rows, milestones


def sparse_oracle(history, resident, batches):
    state = SparsePermutation(history, resident)
    milestones = []
    for events in batches:
        for source, dst in events:
            state.swap(source, dst)
        milestones.append(tuple(state.row_at.get(i, i) for i in range(resident)))
    rows = [state.row_at.get(i, i) for i in range(history)]
    return rows, milestones


def partition_ordered(state: SparsePermutation, events, limit=32):
    """Yield maximal ordered prefixes whose current source/destination rows are disjoint.

    A pending prefix is applied before the next prefix is resolved. The greedy
    frontier never skips a conflicting record, preserving observable batch order.
    """
    i = 0
    while i < len(events):
        chosen = []
        occupied = set()
        while i < len(events) and len(chosen) < limit:
            source, destination = events[i]
            if not 0 <= source < state.history or not 0 <= destination < state.resident:
                raise ValueError("invalid update")
            src_pos = state.position_of.get(source, source)
            pair = {src_pos, destination}
            if chosen and pair & occupied:
                break
            chosen.append((source, destination, src_pos))
            occupied.update(pair)
            i += 1
        yield chosen
        for source, destination, expected_src in chosen:
            actual, _ = state.swap(source, destination)
            assert actual == expected_src


def apply_parallel_batch(state: SparsePermutation, events):
    """Model the existing partial-permutation planner on current positions.

    Incoming row identities and hot destinations must each be unique in this
    batch. Snapshotting every source before writes handles cycles. The cold
    rows displaced by this plan need not match ordered-swap cold placement.
    """
    resolved = []
    for source, destination in events:
        if not 0 <= source < state.history or not 0 <= destination < state.resident:
            raise ValueError("invalid update")
        resolved.append((state.position_of.get(source, source), destination))
    sources = {s for s, _ in resolved}
    destinations = {d for _, d in resolved}
    if len(sources) != len(resolved) or len(destinations) != len(resolved):
        raise ValueError("duplicate source or destination in batch")
    outgoing = [d for _, d in resolved if d not in sources]
    freed = [s for s, _ in resolved if s not in destinations]
    moves = [(s, d) for s, d in resolved] + list(zip(outgoing, freed))
    snapshot = [(state.row_at.get(s, s), d) for s, d in moves]
    for row, destination in snapshot:
        state.row_at[destination] = row
        state.position_of[row] = destination
    return len(moves)
