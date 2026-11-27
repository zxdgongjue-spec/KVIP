# SPDX-License-Identifier: Apache-2.0
"""Pin one diagnostic to the natural ten-call LI result of the same window policy."""
import hashlib
from pathlib import Path

from b24_window_policy import install, policy
from b24_window_report import CONTROL, cached, values

STUDY_READER = "e602ff4a688a733664aaabe84aa30732b9e68062"
CELLS = {"window-tail-x2": (7,10), "window-tail-x4": (7,), "window-tail-x3": (10,), "window-tail-x3-g7": (7,)}


def policy_name(mode):
    return CONTROL if mode == "window-tail-x3-g7" else mode.replace("window-tail-x", "window-x") + "-early"


def supported(mode, group, batch):
    batches = (28,32) if mode == "window-tail-x3-g7" else (32,)
    return group in CELLS.get(mode, ()) and batch in batches


class MissingNaturalPolicy(FileNotFoundError):
    """No completed natural arm of this policy is available on this host."""


def bind(mode, reference, group, batch):
    if not supported(mode, group, batch):
        raise ValueError("Unsupported window-tail diagnostic shape or policy")
    name = policy_name(mode)
    control = reference / f"g{group}/bs{batch}/block1/fused"
    if name == CONTROL:
        arm = control
    else:
        arm = cached(STUDY_READER, reference, group, batch, name)
        if arm is None:
            raise MissingNaturalPolicy(f"No completed, same-card {name} natural archive at {reference}")
    record, original, identity = values(arm, reference, group, batch, name)
    if (record["calls"] != 10 or len(record["traffic"]) != 10 or
            not record.get("cold_build_fingerprint") or
            (name != CONTROL and not record.get("tail_text"))):
        raise ValueError(f"Missing natural timing/provenance at {arm}")
    if (record.get("correctness") != "PASS" or record.get("full_KV_Index_checked_calls") != 10
            or (record.get("checked_queries"), record.get("checked_rows")) != (batch*40,batch*10)):
        raise ValueError(f"Missing checked natural ten-call graph at {arm}")
    ordinary_tail = record.get("tail_text")
    if name == CONTROL:
        # The old four-window archive predates the per-stage ELF digest.
        # Its paired X2 run recorded the same ordinary library and its tail.
        partner = cached(STUDY_READER, reference, group, batch, "window-x2-early")
        if partner is not None:
            alternate, paired, _ = values(partner, reference, group, batch, "window-x2-early")
            if paired["cold_build_fingerprint"] != original["cold_build_fingerprint"]:
                raise ValueError("Natural controls have different ordinary libraries")
            ordinary_tail = alternate["tail_text"]
    packed = arm / "packed_result.json"
    digest = hashlib.sha256(packed.read_bytes()).hexdigest()
    return record, dict(mode=name, packed_result=str(packed), sha256=digest,
                        original_library=original["cold_build_fingerprint"].split("/")[0],
                        tail_text=ordinary_tail)


def select_policy(large_two_cold, name):
    if name != CONTROL:
        install(large_two_cold, name)
    if (tuple(large_two_cold.FOUR_WINDOW_RANKS), tuple(large_two_cold.WINDOW_GRANTS["four-ranks"])) != policy(name):
        raise ValueError("Worker uses different ranks or per-row grants")


def verify_launch(bound, record, proof):
    if (bound["cold_build"] != record["cold_build_fingerprint"] or
            bound["cold_build"].split("/")[0] != proof["original_library"] or
            (proof["tail_text"] is not None and bound["tail_text"] != proof["tail_text"])):
        raise ValueError("Actual LI23 image or uninstrumented terminal .text differs from natural run")


def capture_first(execution, observe):
    """Observe the original eager call. Restore its real forward before launching."""
    from contextlib import contextmanager

    @contextmanager
    def installed():
        original = execution.FusedReplayState
        calls = []

        class FirstState(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if calls:
                    raise ValueError("The tail diagnostic needs only one input state")
                regular = self.forward

                def first():
                    self.forward = regular
                    calls.append(1)
                    return observe(self)

                self.forward = first

        execution.FusedReplayState = FirstState
        try:
            yield
            if calls != [1]:
                raise ValueError("Tail probe did not run at the original eager forward")
        finally:
            execution.FusedReplayState = original

    return installed()


def witness(tail):
    times = tail["separate_terminal_query_stamps"]
    diagnostic = tail["decoded"]
    mhz = tail["clock_mhz"]
    origin = diagnostic["origin"]
    last = diagnostic["last_row"]
    row = tail["original_team_schedule"]["rows"][last]
    q2 = tail["prefetch_cq_checkpoints"][last]
    def us(stamp):
        return round((stamp-origin)/mhz, 3)
    streams = [q for q in diagnostic["qs"] if q["row"]==last]
    q2_obs = q2["observations"]
    if not q2_obs[0]["pending"]:
        interval = None
    else:
        positives = [v for v in q2_obs if v["all_visible"]]
        negatives = [v for v in q2_obs if not v["all_visible"]]
        interval = dict(last_observed_incomplete_us=us(negatives[-1]["begin_tick"]) if negatives else None,
                        first_confirmed_complete_us=us(positives[0]["end_tick"]) if positives
                        else us(q2["after_original_drain_tick"]))
    batches = [(q,item) for q in diagnostic["qs"] for item in q["batches"]]
    last_drain = None
    if batches:
        q, item = max(batches,key=lambda pair:pair[1]["drain_end"])
        last_drain = dict(row=q["row"],query=q["query"],end_us=us(item["drain_end"]),
                          terminal_work_after_software_CQ_us=round(diagnostic["terminal_span_us"]-us(item["drain_end"]),3))
    return dict(row=last,team=row["team"],wave=row["wave_on_team"],row_entry_us=us(min(q[0] for q in times[last])),
                row_exit_us=us(max(q[10] for q in times[last])),q2_prefetch_visible_interval=interval,
                latest_final_cq_in_batch=last_drain,
                final_by_q=[dict(query=q["query"],token_count=q["own"],enter_us=us(times[last][q["query"]-1][0]),
                                 index_complete_us=us(q["index_end"]),
                                 first_post_us=us(q["batches"][0]["post_end"]) if q["batches"] else None,
                                 last_final_cq_us=us(q["batches"][-1]["drain_end"]) if q["batches"] else None)
                            for q in streams])


def origin(area, mode):
    from large_batch_placement import field
    from portable_two_row_audit import local, local_identity
    host=local_identity()
    cells = [(7,28),(7,32)] if mode == "window-tail-x3-g7" else [(7,32),(10,32)]
    for path in sorted(area.glob("fixed-q4-four-ranks-one-block.*"),key=lambda p:p.stat().st_mtime_ns,reverse=True):
        if (local(path,host) and field(path,"native")=="d0fc5963881bd38fa97eff0f6ef40c757a6e8713"
                and field(path,"kvaf")=="813f1750a50b7e6c92d5f5397f2fbc05befef6da"
                and field(path,"kvaf_tree")=="1de18e8315137d7a0227c3ae4c82afd45b5fe743"
                and field(path,"selection_oracle")=="captured_logical_scores" and (path/"packed_summary.json").is_file()
                and all((path/f"g{group}/bs{batch}/block1/fused/placement_verified").is_file() for group,batch in cells)):
            return path
    return "LOCAL"


if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--origin",type=Path,required=True)
    p.add_argument("--mode", required=True, choices=tuple(CELLS))
    args=p.parse_args()
    print(origin(args.origin,args.mode))
