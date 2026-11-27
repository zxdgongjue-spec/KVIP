# SPDX-License-Identifier: Apache-2.0
"""Time the BS32 two-row terminal using ordinary window GETs and ten independent states."""

from argparse import Namespace
import json
from math import isclose, isfinite
from statistics import mean
import sys

BASE = "d0fc5963881bd38fa97eff0f6ef40c757a6e8713"
SYMBOL = "v9_test_two_row_terminal"
STAGES = ("graph", "LI", "tail")


def comparison(old, new):
    return dict(
        baseline_mean_us={key: old[key] for key in STAGES},
        saved_mean_us={key: old[key] - new[key] for key in STAGES},
        graph_gain_pct=100 * (1 - new["graph"] / old["graph"]),
    )


def install_readback(packed, expected_cold, joint_identity):
    measure = packed.measure_packed

    def measure_natural(ex, proof, serial, torch, states, *args, **kwargs):
        if len(states) != 10 or not kwargs.get("fused") or kwargs.get("window_density"):
            raise ValueError("Expected ten natural states and diagnostics off")
        terminals = [state.large_two_row_terminal for state in states]
        if len({binding.workspace.data_ptr() for binding in terminals}) != len(states):
            raise ValueError("The captured graph reuses mutable terminal mailboxes")
        if any(state.large_window_launch.source_fingerprint != expected_cold for state in states):
            raise ValueError("Natural two-row timing changed the archived four-rank LI build")
        image = joint_identity(states[0])
        # The existing worker collects its graph event and device timeline, then validates all ten states.
        result = measure(ex, proof, serial, torch, states, *args, **kwargs)
        if result["correctness"] == "PASS":
            receipts = [binding.readback() for binding in terminals]
            if any(not r["accepted"] for r in receipts):
                raise ValueError("Joint-terminal mailbox rejected a measured state")
            proof.update(
                measured_tail_symbol=SYMBOL, terminal_build=image,
                terminal_decisions=receipts,
                parked_rows_per_call=[r["parked_rows"] for r in receipts],
                calls_with_row_pair=sum(r["parked_rows"] > 0 for r in receipts),
            )
        return result

    packed.measure_packed = measure_natural
    return measure


def run(a):
    if a.mode != "two-row-natural" or a.batch != 32 or a.group not in (7, 2, 10, 4, 17):
        raise ValueError("Expected one of the five fixed Q4 BS32 traces")
    # Timed source owns the ordinary LI and graph runner. The controller supplies only the test terminal.
    sys.path.insert(0, str(a.timed / "tests/operators/lifetch/support"))
    import acl
    import large_batch_pair
    import large_two_cold
    import fixed_q4_packed_worker as packed
    from b24_two_row_attach import identity as joint_identity, install, install_profiler
    from export_large_rank_evidence import observation
    from four_rank_followup import serial_reference
    from reporting import atomic_json

    block = a.ref / f"g{a.group}/bs{a.batch}/block1"
    original = json.loads((block / "fused/packed_result.json").read_bytes())
    control, physical_identity = observation(a.ref, block, a.group, a.batch, "fused")
    if (original["identity"]["native"] != BASE or original["identity"]["physical_npu"] != 4
            or original["measured_cold_symbol"] != "v9_li_large_four_ranks"):
        raise ValueError("Expected the complete four-window B24 NPU4 archive")
    install(large_two_cold, a.timed, a.sdk / "build-npu", acl.get_soc_name())
    install_profiler()
    previous_measure = install_readback(packed, original["cold_build_fingerprint"], joint_identity)
    try:
        large_batch_pair.run(Namespace(
            trace=a.trace, group=a.group, batch=32, round=1, cohort=0, prepared=a.bank,
            arm="fused", kvaf_root=a.sdk, kvaf_server=a.server, output=a.out,
            packed_ten=True, captured_score_oracle=True, large_window_policy="four-ranks",
            compare_packed_candidate=True, carry_credit=False, window_density=False,
            serial_result=None, reuse_serial_archive=None, compare_archived_pair=False, keep_trace=False,
        ))
    finally:
        packed.measure_packed = previous_measure

    result = json.loads((a.out / "packed_result.json").read_bytes())
    if (result["identity"] != original["identity"] or result["correctness"] != "PASS"
            or result["measured_tail_symbol"] != SYMBOL or result["get_workers"] != 32
            or result["cold_build_fingerprint"] != original["cold_build_fingerprint"]
            or result["checked_queries"] != 1280 or result["checked_rows"] != 320
            or result["calls"] != 10 or result["full_KV_Index_checked_calls"] != 10
            or not isclose(result["graph_mean_us"] * 10, result["graph_event_us"], rel_tol=1e-8)):
        raise ValueError("Natural result differs from the archived input, LI, runtime or timing protocol")
    new = {"graph": result["graph_mean_us"], **{k: result["mean_us"][k] for k in ("LI", "tail")}}
    if not all(isfinite(v) and v > 0 for v in new.values()):
        raise ValueError("Invalid measured graph, LI or tail duration")
    serial = serial_reference(a.ref, physical_identity, a.group, a.batch)
    result.update(
        arm=a.mode, reference=str(a.ref), scope="natural ten-call graph with original four-rank LI",
        vs_four_windows=comparison(control, new),
        vs_strong_serial=dict(source=serial["source"], blocks=serial["blocks"], **comparison(serial, new)) if serial else None,
        natural_GET_mean={k: mean(t[k] for t in result["traffic"]) for k in ("prefetch", "final")},
    )
    atomic_json(a.out / "two-row-natural.json", result)
