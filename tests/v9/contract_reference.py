"""Independent CPU oracle for identity, allocation, publication and feedback."""

from dataclasses import dataclass

import numpy as np
from kvop.v9.contract import BITMAP_WORDS, BLOCK_TOKENS, MAX_SELECTION, TOKEN_BYTES, TOPK, Status


class ContractError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = Status(status)


def bitmap(slots):
    result = np.zeros(BITMAP_WORDS, np.uint32)
    for slot in slots:
        if not 0 <= int(slot) < 32 * BITMAP_WORDS:
            raise ContractError(Status.INVALID_ARGUMENT, "bitmap slot out of range")
        result[int(slot) // 32] |= np.uint32(1 << (int(slot) % 32))
    return result.view(np.int32)


def frozen_hot(forward, reverse, victims, candidate, budget):
    victims = np.asarray(victims, dtype=np.int64)
    if len(victims) != TOPK or len(set(victims.tolist())) != TOPK:
        raise ContractError(Status.INVALID_ARGUMENT, "victims must contain 2048 unique slots")
    if np.any((victims < 0) | (victims >= budget)):
        raise ContractError(Status.INVALID_ARGUMENT, "victim outside resident budget")
    hot = {}
    for slot, raw in enumerate(reverse[:budget]):
        token = int(raw)
        if 0 <= token < candidate and int(forward[token]) == slot:
            hot[token] = slot
    return hot


def eligible_victims(scores, hot, victims):
    q, c = scores.shape
    thresholds = []
    for i in range(q):
        ids = [t for t in hot if t < c - q + i + 1]
        thresholds.append(np.partition(scores[i, ids], -TOPK)[-TOPK] if len(ids) >= TOPK else -np.inf)
    protected = {
        s for t, s in hot.items() if any(t < c - q + i + 1 and scores[i, t] >= thresholds[i] for i in range(q))
    }
    return [int(s) for s in victims if int(s) not in protected]


def validate_equal_values(tensors, geometry):
    b, q = geometry.batch, geometry.queries
    if tensors["profile"][0] == 2 or np.all(tensors["row_modes"] == 0):
        return
    if not np.array_equal(tensors["cumulative_query"], np.arange(1, b + 1) * q):
        raise ContractError(Status.INVALID_ARGUMENT, "unequal query lengths")
    for name in ("candidate_lens", "logical_lens", "budgets"):
        if np.any(tensors[name] != tensors[name][0]):
            raise ContractError(Status.INVALID_ARGUMENT, f"unequal {name}")
    if tensors["profile"][0] == 2 or np.all(tensors["row_modes"] == 0):
        return
    if not np.all(tensors["row_modes"] == 2):
        raise ContractError(Status.INVALID_ARGUMENT, "only homogeneous sparse calls are supported")
    c, logical, budget = (int(tensors[n][0]) for n in ("candidate_lens", "logical_lens", "budgets"))
    if not 1 <= c <= geometry.candidate_capacity or not c <= logical:
        raise ContractError(Status.INVALID_ARGUMENT, "invalid candidate/logical lengths")
    if not TOPK <= budget <= geometry.resident_capacity:
        raise ContractError(Status.INVALID_ARGUMENT, "invalid resident budget")
    if budget + logical - c > geometry.resident_columns * BLOCK_TOKENS:
        raise ContractError(Status.INVALID_ARGUMENT, "tail exceeds resident table")
    rows = tensors["pool_rows"].tolist()
    if len(set(rows)) != b or any(not 0 <= p < geometry.max_requests for p in rows):
        raise ContractError(Status.INVALID_ARGUMENT, "invalid or repeated pool row")
    for row in range(b):
        if not np.array_equal(bitmap(tensors["victims"][row]), tensors["victim_bitmap"][row]):
            raise ContractError(Status.INVALID_ARGUMENT, "victim bitmap/list mismatch")
        index = tensors["index_table"][row, : (c + 127) // 128]
        resident = tensors["resident_table"][row, : (budget + logical - c + 127) // 128]
        if np.any((index < 0) | (index >= geometry.index_blocks)):
            raise ContractError(Status.INVALID_ARGUMENT, "invalid index physical page")
        if np.any((resident < 0) | (resident >= geometry.resident_blocks)):
            raise ContractError(Status.INVALID_ARGUMENT, "invalid resident physical page")


def address(tensors, geometry, row, token, slot):
    p = int(tensors["pool_rows"][row])
    c, budget = int(tensors["candidate_lens"][row]), int(tensors["budgets"][row])
    if not 0 <= token < c or not 0 <= slot < budget:
        raise ContractError(Status.INVALID_ARGUMENT, "token/slot outside logical bounds")
    directory = (p * geometry.num_layers + geometry.layer_id) * geometry.qcap + token // 128
    if directory >= geometry.table_capacity or not tensors["valid_table"][directory]:
        raise ContractError(Status.DIRECTORY, "invalid remote directory")
    base = int(tensors["address_table"].view(np.uint64)[directory])
    remote = base + (token % 128) * TOKEN_BYTES
    block = int(tensors["resident_table"][row, slot // 128])
    if not 0 <= block < geometry.resident_blocks:
        raise ContractError(Status.INVALID_ARGUMENT, "invalid destination page")
    if base == 0 or remote + TOKEN_BYTES > 2**64:
        raise ContractError(Status.DIRECTORY, "remote address overflow")
    physical = block * 128 + slot % 128
    return directory, (token % 128) * TOKEN_BYTES, physical, remote


@dataclass
class Plan:
    locations: np.ndarray
    destination: np.ndarray
    ordered_tokens: np.ndarray
    misses: dict
    supplement: dict
    discarded: dict
    extra: dict
    cursor: int
    deferred: set


def plan_row(selected, hot, victims, budget, cursor, reservations=None, pending=None):
    """Plan without mutating maps. MTP ordering is stable first occurrence here."""
    selected = np.asarray(selected, np.int32).reshape(-1, TOPK)
    q = selected.shape[0]
    if not 1 <= q <= 4:
        raise ContractError(Status.INVALID_ARGUMENT, "query count outside 1..4")
    reservations = dict(reservations or {})
    pending = set(reservations.values()) if pending is None else set(pending)
    if len(reservations) > TOPK or len(set(reservations.values())) != len(reservations):
        raise ContractError(Status.INVALID_ARGUMENT, "invalid reservations")
    victim_set = set(map(int, victims))
    if any(s not in victim_set for s in reservations.values()):
        raise ContractError(Status.INVALID_ARGUMENT, "reservation must use frozen victim")
    ordered = list(dict.fromkeys(int(t) for t in selected.flat if t >= 0))
    if len(ordered) > budget:
        raise ContractError(Status.CAPACITY, "final unique set exceeds budget")
    for row in selected:
        valid = row[row >= 0]
        if len(np.unique(valid)) != len(valid) or np.any(row < -1):
            raise ContractError(Status.INVALID_ARGUMENT, "duplicate/invalid TopK token")
    miss_tokens = [t for t in ordered if t not in hot]
    if len(miss_tokens) > MAX_SELECTION:
        raise ContractError(Status.CAPACITY, "miss capacity exceeded")
    hit_slots = {hot[t] for t in ordered if t in hot}
    misses = {t: reservations[t] for t in miss_tokens if t in reservations}
    used = hit_slots | set(misses.values())
    free_victims = iter(int(s) for s in victims if int(s) not in used and int(s) not in pending)
    proposed = cursor
    scan = max(0, int(cursor)) % budget
    scanned = 0
    for token in miss_tokens:
        if token in misses:
            continue
        slot = next(free_victims, None)
        if slot is None:
            if q == 1:
                raise ContractError(Status.CAPACITY, "ordinary victim capacity exceeded")
            while scanned < budget and (scan in victim_set or scan in used or scan in pending):
                scan = (scan + 1) % budget
                scanned += 1
            if scanned >= budget:
                slot = next((int(s) for s in victims if int(s) not in used), None)
                if slot is None:
                    raise ContractError(Status.CAPACITY, "no additional eviction slots")
            else:
                slot = scan
                scan = (scan + 1) % budget
                scanned += 1
                proposed = scan
        misses[token] = slot
        used.add(slot)
    supplement = {t: misses[t] for t in miss_tokens if t not in reservations}
    extra = {t: s for t, s in reservations.items() if t not in misses and s not in used}
    discarded = {t: s for t, s in reservations.items() if t not in misses and t not in extra}
    locations = np.full_like(selected, -1)
    destination = np.full_like(selected, -1)
    ordered_tokens = np.full_like(selected, -1)
    for i, row in enumerate(selected):
        valid = [int(t) for t in row if t >= 0]
        order = [t for t in valid if t not in hot] + [t for t in valid if t in hot]
        for j, token in enumerate(row):
            if token >= 0:
                locations[i, j] = hot[int(token)] if int(token) in hot else misses[int(token)]
        for j, token in enumerate(order):
            ordered_tokens[i, j] = token
            destination[i, j] = hot[token] if token in hot else misses[token]
    return Plan(
        locations,
        destination,
        ordered_tokens,
        misses,
        supplement,
        discarded,
        extra,
        proposed,
        set(supplement.values()) & pending,
    )


def commit_plan(forward, reverse, plan, reservations, *, completed=True):
    if not completed:
        raise ContractError(Status.COMMUNICATION, "cannot publish before all payloads complete")
    touched = set(reservations.values()) | set(plan.supplement.values())
    # All clears precede every install, including token moves across victims.
    for slot in touched:
        old = int(reverse[slot])
        if 0 <= old < len(forward) and int(forward[old]) == slot:
            forward[old] = -1
        reverse[slot] = -1
    for token, slot in (plan.misses | plan.extra).items():
        forward[token] = slot
        reverse[slot] = token
    return plan.cursor


def select_topk(scores, q):
    """Tie-stable oracle; hardware may choose any valid cutoff ties."""
    scores = np.asarray(scores)
    c = scores.shape[-1]
    output = np.full((q, TOPK), -1, np.int32)
    for i in range(q):
        visible = max(0, c - q + i + 1)
        count = min(TOPK, visible)
        ids = np.arange(visible)
        order = np.lexsort((ids, -scores[i, :visible]))[:count]
        output[i, :count] = ids[order]
    return output


def verify_selection(selected, scores):
    selected = np.asarray(selected).reshape(-1, TOPK)
    q, c = scores.shape
    for i, row in enumerate(selected):
        visible = max(0, c - q + i + 1)
        ids = row[row >= 0]
        count = min(TOPK, visible)
        if len(ids) != count or len(np.unique(ids)) != count:
            raise AssertionError("wrong number of unique valid selected tokens")
        if np.any(ids >= visible) or np.any(row < -1):
            raise AssertionError("TopK violates real-token causal visibility")
        if count:
            cutoff = np.partition(scores[i, :visible], -count)[-count]
            below = ids[scores[i, ids] < cutoff]
            if len(below):
                sample = below[:8]
                raise AssertionError(
                    f"TopK includes a below-cutoff token: query={i}, visible={visible}, "
                    f"cutoff={float(cutoff)}, below_count={len(below)}, "
                    f"tokens={sample.tolist()}, scores={scores[i, sample].tolist()}"
                )
            required = set(np.flatnonzero(scores[i, :visible] > cutoff).tolist())
            if not required.issubset(set(ids.tolist())):
                raise AssertionError("TopK omitted a strictly higher-scoring token")


def attention_metadata(destination, budget, candidate, logical):
    tail = logical - candidate
    rows = np.asarray(destination).reshape(-1, TOPK)
    result = np.full((len(rows), 1, TOPK + tail), -1, np.int32)
    result[:, 0, :TOPK] = rows
    result[:, 0, TOPK:] = np.arange(budget, budget + tail)
    return result
