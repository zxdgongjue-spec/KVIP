# SPDX-License-Identifier: Apache-2.0
"""Calibrate the existing device counter using an independent long graph."""

import ctypes
import math
from statistics import median


def calibration_summary(samples):
    if len(samples) != 3 or any(
        not math.isfinite(row["event_us"]) or row["event_us"] <= 0
        or row["cycles"] < 20_000_000 for row in samples
    ):
        raise ValueError("Counter calibration requires three valid 20M-cycle samples")
    rates = [row["cycles"] / row["event_us"] for row in samples]
    return dict(status="measured", cycles_per_us=median(rates), samples=samples,
                rate_min=min(rates), rate_max=max(rates), included_in_timing=False,
                method="Existing kvip_counter_calibration: independent long NPU graph and timing events",
                limitation="Approximate conversion includes graph/event edges; not a fixed hardware clock or a bus-bandwidth measurement")


def calibrate_counter(torch, library, device, stream):
    function = getattr(library, "kvip_counter_calibration", None)
    if function is None:
        raise ValueError("Loaded library has no counter-calibration entry")
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    function.restype = ctypes.c_uint32
    output = torch.empty(4, dtype=torch.int64, device=device)
    capture = torch.npu.Stream(device=device)
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph, stream=capture):
            if function(output.data_ptr(), capture.npu_stream):
                raise RuntimeError("Counter calibration launch failed")
        capture.synchronize()
        samples = []
        for _ in range(3):
            start, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
            start.record(stream)
            graph.replay()
            end.record(stream)
            end.synchronize()
            values = output.cpu().tolist()
            samples.append(dict(cycles=int(values[1]) - int(values[0]),
                                event_us=float(start.elapsed_time(end)) * 1000))
        return calibration_summary(samples)
    finally:
        graph.reset()
