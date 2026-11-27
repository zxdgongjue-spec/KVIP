# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import b24_team_schedule as scheduler


def snapshot(batch=20):
    # Exercise a queue whose row ids differ from queue ranks and one team
    # collecting several children. Expected teams cannot be inferred via row%16.
    order = list(range(batch-1, -1, -1))
    words = np.zeros((4,32), dtype=np.int32)
    words[0,:3] = [0,16+batch,16]
    words[1,:batch] = order
    words[2,:batch] = 4
    words[3,:batch] = batch+1
    chain = [order[4]] + order[16:]
    for pred,row in zip(chain,chain[1:]):
        words[3,pred] = row+1
    rank = {r:k for k,r in enumerate(order)}
    start = [[100+(rank[r]-15)*20+q if rank[r]>=16 else 100+q for q in range(4)] for r in range(batch)]
    stop = [[t+10 for t in row] for row in start]
    return dict(schema=scheduler.SCHEMA,schedule_words=words.tolist(),q_entry_ticks=start,q_exit_ticks=stop,
                counter_mhz=2.0)


@pytest.mark.parametrize("batch", scheduler.BATCHES)
def test_exact_queue_routes_and_times(batch):
    s = snapshot(batch)
    d = scheduler.decode(s,batch)
    row = d["rows"][d["queue"][17]]
    assert row["team"] == 4 and row["wave_on_team"] == 3 and row["q_logical_aiv_blocks"] == [16,17,18,19]
    assert (row["parent_exit_to_child_entry_us"], row["claim_lower_us"],row["claim_upper_us"]) == (3.5,16.5,20)
    assert sum(len(route) for route in d["teams"]) == batch


@pytest.mark.parametrize("part,index,value",[(0,0,1),(0,1,16),(0,2,15),(1,1,19),(2,2,0),
                                             (3,2,0),(3,15,3),(3,0,22)])
def test_incomplete_or_reset_schedule_is_never_evidence(part,index,value):
    s = snapshot()
    s["schedule_words"][part][index] = value
    with pytest.raises(ValueError):
        scheduler.decode(s,20)


def test_ticket_intervals_from_successive_claims_must_be_feasible():
    s=snapshot()
    s["q_entry_ticks"][2]=[105,106,107,108]
    s["q_exit_ticks"][2]=[115,116,117,118]
    with pytest.raises(ValueError,match="overlapping"):
        scheduler.decode(s,20)
    s=snapshot()
    s["counter_mhz"]=float("nan")
    with pytest.raises(ValueError):
        scheduler.decode(s,20)


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def test_old_journal_without_schedule_is_skipped_and_new_preserves_runtime_and_origin(tmp_path,monkeypatch):
    base=tmp_path/"fixed-q4-four-ranks-one-block.old"
    old=tmp_path/"fixed-q4-audit-journal.one"
    new=tmp_path/"fixed-q4-audit-team-schedule.two"
    rel=Path("g7/bs20/block1")
    for root in (base, old, new):
        root.mkdir()
        for k,v in dict(native=scheduler.BASE,physical_npu="4",kvaf="sdk",kvaf_tree="sdk-tree",host="b24",
                         machine_id="id",audit_reference=str(base)).items():
            (root/(k+".txt")).write_text(v+"\n")
    identity=dict(group=7,batch=20,native=scheduler.BASE)
    write(base/rel/"fused/packed_result.json",dict(identity=identity,cold_build_fingerprint="four"))
    monkeypatch.setattr(scheduler,"observation",lambda *args: (None,(None,"placement")))
    monkeypatch.setattr(scheduler,"placement",lambda *args: "placement")
    j=dict(arm="journal",identity=identity,correctness="PASS",get_workers=32,chunks=160,tail_text=["aiv"],
           compiler_sha256={"cc":"sha"},cold_build="four",journal_sha256=hashlib.sha256(b"journal").hexdigest())
    d=old/rel/"journal"
    write(d/"journal.json",j)
    (d/"placement_verified").touch()
    assert scheduler.find(base,base,7,20) is None
    d=new/rel/"team-schedule"; j.update(terminal_team_schedule=snapshot(), measured_calls=0, diagnostic_calls=1)
    write(d/"journal.json",j)
    write(d/"team-schedule.json",dict(j,arm="team-schedule"))
    (d/"placement_verified").touch()
    (d/"journal.npz").write_bytes(b"journal")
    assert scheduler.find(base,base,7,20) == d
    (d/"journal.npz").write_bytes(b"wrong")
    with pytest.raises(ValueError,match="checksum"):
        scheduler.find(base,base,7,20)
