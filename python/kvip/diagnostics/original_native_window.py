# SPDX-License-Identifier: Apache-2.0
"""Read the original ready-batch Get worker's native-call markers."""
from kvip.diagnostics.counter_clock import calibration_summary
from kvip.diagnostics.read_through_report import require


def original_native_window(sample):
    data = sample["raw"]
    batch, resolvers, workers = data["batch"], data["resolver_cores"], data["workers_per_query"]
    first_worker = batch * resolvers
    cores = data["cores"]
    require(sample["auto_route"]["mode"] == "high" and (batch, resolvers, workers) == (16, 3, 4)
            and len(cores) == first_worker + 4 * workers and all(len(c) == 32 for c in cores),
            "Expected original BS16 HIGH ready-batch roles")
    require(all(0 < c[0] <= c[1] for c in cores), "Missing streaming role timestamps")
    origin = min(c[0] for c in cores)
    active = []
    for lane in cores[first_worker:]:
        entry, leave, begin, call, returned, finished = lane[:6]
        calls, requests = lane[9:11]
        require(entry <= begin <= finished <= leave and calls >= 0 and requests >= 0,
                "Invalid ready-batch worker envelope")
        if requests:
            require(0 < calls <= requests and begin <= call <= returned <= finished,
                    "Missing nonempty native Get timestamp")
            active.append(lane)
        else:
            require((calls, call, returned) == (0, 0, 0), "Stale empty Get markers")
    final = sum(c[10] for c in active)
    require(final == sample["reads"]["final"], "Mismatch between native Get and final request count")
    return dict(
        first_native_call=min((c[3] - origin for c in active), default=None),
        last_native_return=max((c[4] - origin for c in active), default=None),
        last_consumer_exit=max(c[1] for c in cores[1:first_worker:resolvers]) - origin,
        last_role_exit=max(c[1] for c in cores) - origin,
        final_Get=final, active_workers=len(active), native_calls=sum(c[9] for c in active),
    )


def print_original_clock(group, case):
    diag = case["native_diagnostics"]
    clock = diag["counter_calibration"]
    require(not case["high_post_index"] and case["tuning"]["ready_batch_fetch"] and
            not diag["included_in_timing"] and clock["status"] == "measured" and
            not clock["included_in_timing"] and len(diag["samples"]) == 1,
            "Original ready-batch clock requires one separate calibrated diagnostic replay")
    result = original_native_window(diag["samples"][0])
    rate = calibration_summary(clock["samples"])["cycles_per_us"]
    def us(key):
        return "NA" if result[key] is None else f"{result[key] / rate:.2f}"
    print(f"G{group} ORIGINAL_NATIVE_us first_blocking_call={us('first_native_call')}"
          f" last_blocking_return={us('last_native_return')} last_consumer_exit={us('last_consumer_exit')}"
          f" last_role_exit={us('last_role_exit')} final_Get={result['final_Get']}"
          f" active_workers={result['active_workers']} calls={result['native_calls']}"
          " origin=first_streaming_role_entry", flush=True)
