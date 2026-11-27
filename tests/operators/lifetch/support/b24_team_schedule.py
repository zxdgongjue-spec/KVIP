# SPDX-License-Identifier: Apache-2.0
"""Read the original terminal's actual four-Q team assignments after one completed invocation."""
import argparse
import hashlib
import math
import struct
from pathlib import Path

import numpy as np

from export_large_rank_evidence import observation
from four_rank_accept import matched, read
from large_batch_placement import field, placement

BASE = "d0fc5963881bd38fa97eff0f6ef40c757a6e8713"
SCHEMA = "original_terminal_team_schedule_v1"
BATCHES = (20, 24, 28, 32)
GROUPS = (10, 4, 17, 7, 2)


def device_words(value, shape, name):
    values = np.asarray(value)
    if values.shape != shape or values.dtype.kind not in ("i", "u") or np.any(values < 0):
        raise ValueError(f"Bad {name} dimensions or values")
    return values


def decode(raw, batch):
    if raw["schema"] != SCHEMA or batch not in BATCHES:
        raise ValueError("Unexpected original scheduler or batch")
    words = device_words(raw["schedule_words"], (4, 32), "TerminalRowSchedule")
    entry = device_words(raw["q_entry_ticks"], (batch, 4), "Q entry")
    exit_ = device_words(raw["q_exit_ticks"], (batch, 4), "Q exit")
    mhz = float(raw["counter_mhz"])
    if not math.isfinite(mhz) or mhz <= 0 or np.any(entry == 0) or np.any(exit_ < entry):
        raise ValueError("Missing or reversed original terminal timestamps")
    if words[0, :3].tolist() != [0, 16 + batch, 16] or not np.all(words[2, :batch] == 4):
        raise ValueError("Missing completed fourth arrival, retirements or final ticket count")
    order = list(map(int, words[1, :batch]))
    if sorted(order) != list(range(batch)):
        raise ValueError("The initial device row order is not a permutation")
    ranks = {r:k for k,r in enumerate(order)}
    succ = [int(n)-1 for n in words[3, :batch]]
    if any(n < 0 or n > batch for n in succ) or sorted(i for i in succ if i != batch) != sorted(order[16:]):
        raise ValueError("Missing or duplicate successor, invalid sentinel or reassigned first-wave row")
    origin = int(entry.min())
    start = [int(min(t)) for t in entry]
    end = [int(max(t)) for t in exit_]
    rows = [None] * batch
    teams = []
    for team, first in enumerate(order[:16]):
        row, pred, chain = first, None, []
        while row != batch:
            rank = ranks[row]
            if rows[row] is not None or (pred is not None and (rank <= ranks[pred] or start[row] < end[pred])):
                raise ValueError("Repeated, out-of-order or overlapping assignments on one four-Q team")
            record = dict(row=row, queue_rank=rank, team=team, q_logical_aiv_blocks=list(range(4*team, 4*team+4)),
                          wave_on_team=len(chain)+1, predecessor=pred, enter_us=(start[row]-origin)/mhz,
                          exit_us=(end[row]-origin)/mhz, q_enter_us=[(int(t)-origin)/mhz for t in entry[row]],
                          q_exit_us=[(int(t)-origin)/mhz for t in exit_[row]])
            if pred is not None:
                # The parent wrote its last Q exit stamp before claiming; the
                # child wrote its first Q entry stamp after receiving the ticket.
                record.update(claim_lower_us=(end[pred]-origin)/mhz, claim_upper_us=(start[row]-origin)/mhz,
                              parent_exit_to_child_entry_us=(start[row]-end[pred])/mhz)
            rows[row] = record
            chain.append(row)
            pred, row = row, succ[row]
        teams.append(chain)
    if any(x is None for x in rows):
        raise ValueError("An input row was not reached from any team")
    # Claims in the global ticket queue are ordered, even if wall-clock stamp
    # intervals from different teams overlap. There must exist a legal ordering.
    first_possible = 0
    for r in order[16:]:
        first_possible = max(first_possible, end[rows[r]["predecessor"]])
        if first_possible > start[r]:
            raise ValueError("Device timestamps cannot enclose the observed global ticket order")
    return dict(teams=teams, rows=rows, queue=order, counter_mhz=mhz, terminal_span_us=(max(end)-origin)/mhz)


def capture(state):
    terminal = state.forward.terminal
    if state.batch not in BATCHES or terminal is None or terminal.row_schedule is None:
        raise ValueError("Missing large-batch row schedule")
    bound = struct.unpack("<6Q2I9Q", terminal.workspace.cpu().numpy().tobytes())
    if (bound[6] != 1 or not bound[7] & 1 or bound[4] != terminal.times.data_ptr()
            or bound[9] != terminal.row_schedule.data_ptr()):
        raise ValueError("Actual profiled terminal binding does not point at its clocks and scheduler")
    words = terminal.row_schedule.cpu().numpy().copy()
    stamps = terminal.times.cpu().numpy()
    raw = dict(schema=SCHEMA, schedule_words=words.tolist(), q_entry_ticks=stamps[:, :, 0].tolist(),
               q_exit_ticks=stamps[:, :, 10].tolist(), counter_mhz=state.forward.calibrate_counter_mhz())
    decode(raw, state.batch)
    return raw


def verified(directory, ref, group, batch):
    root = directory.parents[3]
    rel = Path(f"g{group}/bs{batch}/block1")
    if (group not in GROUPS or batch not in BATCHES or directory.parent != root / rel
            or directory.name not in ("journal", "team-schedule")
            or not (directory / "placement_verified").is_file() or not matched(root, ref)
            or field(root, "audit_reference") != str(ref) or field(root, "native") != BASE
            or field(ref, "native") != BASE or field(ref, "physical_npu") != "4"):
        raise ValueError(f"Invalid original schedule archive path or host identity: {directory}")
    plain = read(ref / rel / "fused/packed_result.json")
    _, identity = observation(ref, ref / rel, group, batch, "fused")
    proof = read(directory / "journal.json")
    if (proof["identity"] != plain["identity"] or proof["arm"] != "journal" or proof["correctness"] != "PASS"
            or proof["get_workers"] != 32 or proof["chunks"] != batch*8
            or not proof.get("tail_text") or not proof.get("compiler_sha256")
            or proof["cold_build"] != plain["cold_build_fingerprint"]
            or placement(directory / "thp.json") != identity[1]):
        raise ValueError(f"Incorrect input, cold library, placement or correctness: {directory}")
    # A pre-existing journal without the actual four device cache lines is
    # still useful for checking GETs, but cannot identify the hardware team.
    if "terminal_team_schedule" not in proof:
        return None
    if (proof.get("measured_calls"), proof.get("diagnostic_calls")) != (0, 1):
        raise ValueError("Team evidence must be from a single unmeasured invocation")
    raw = proof["terminal_team_schedule"]
    h = hashlib.sha256()
    with (directory / "journal.npz").open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            h.update(chunk)
    if h.hexdigest() != proof["journal_sha256"]:
        raise ValueError(f"GET journal checksum mismatch at {directory}")
    decoded = decode(raw, batch)
    if directory.name == "team-schedule":
        doc = read(directory / "team-schedule.json")
        if (doc.get("arm") != "team-schedule" or doc.get("identity") != proof["identity"]
                or doc.get("terminal_team_schedule") != raw or doc.get("journal_sha256") != proof["journal_sha256"]):
            raise ValueError("Schedule and GET evidence refer to different invocations")
    return decoded


def find(directory, ref, group, batch):
    if not matched(directory, ref):
        raise ValueError("Scheduler lookup and reference use different hardware or runtime")
    for type_ in ("team-schedule", "journal"):
        for root in sorted(ref.parent.glob(f"fixed-q4-audit-{type_}.*"), reverse=True):
            arm = root / f"g{group}/bs{batch}/block1/{type_}"
            if (field(root, "audit_reference") != str(ref) or field(root, "native") != BASE or not matched(root, ref)
                    or not (arm / "placement_verified").is_file() or not (arm / "journal.json").is_file()):
                continue
            if type_ == "team-schedule" and not (arm / "team-schedule.json").is_file():
                continue
            if verified(arm, ref, group, batch) is not None:
                return arm
    return None


def show(directory, ref, group, batch):
    data = verified(directory, ref, group, batch)
    if data is None:
        raise ValueError(f"This journal has no preserved row-to-team scheduler binding: {directory}")
    tag = f"G{group} BS{batch}"
    routes = " ".join(f"T{i}=" + ">".join(map(str, rs)) for i, rs in enumerate(data["teams"]))
    print(f"{tag} SCHEDULE diagnostic_calls=1 timed_calls=0 logical_AIV=4*team+q0 {routes} source={directory}")
    for r in data["queue"][16:]:
        row = data["rows"][r]
        print(f"{tag} rank={row['queue_rank']} row={r} team={row['team']} team_visit={row['wave_on_team']} "
              f"Q1_to_Q4_blocks={row['q_logical_aiv_blocks'][0]}..{row['q_logical_aiv_blocks'][-1]} from_row={row['predecessor']} claim_between_us={row['claim_lower_us']:.3f}/{row['claim_upper_us']:.3f} "
              f"parent_exit_to_child_entry_us={row['parent_exit_to_child_entry_us']:.3f} "
              f"child_entry/exit_us={row['enter_us']:.3f}/{row['exit_us']:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("cached", "show"))
    p.add_argument("directory", type=Path)
    p.add_argument("ref", type=Path)
    p.add_argument("group", type=int, choices=GROUPS)
    p.add_argument("batch", type=int, choices=BATCHES)
    p.add_argument("mode", choices=("team-schedule",))
    a = p.parse_args()
    if a.action == "show":
        show(a.directory, a.ref, a.group, a.batch)
    else:
        cached = find(a.directory, a.ref, a.group, a.batch)
        if cached:
            print(cached)
