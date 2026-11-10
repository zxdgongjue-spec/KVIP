#!/usr/bin/env python3
"""Verify that actual simulator MTE2 waits for TopK before scratch reuse."""
import json
import re
import sys
from pathlib import Path


def verify(path):
    events = json.loads(path.read_text())["traceEvents"]
    dma = []
    waits = []
    for event in events:
        if event.get("ph") != "X" or str(event.get("pid")) != "60":
            continue
        detail = event.get("args", {}).get("detail", "")
        if event.get("name") == "MOV_SRC_TO_DST_ALIGNv2" and "Dtype:B32" in detail:
            dma.append(event)
        if (event.get("name") == "WAIT_FLAG" and "PIPE:VEC,TRIGGERPIPE:MTE2" in detail
                and re.search(r"FLAGID:(?:0x)?3,", detail)):
            waits.append(event)
    dma.sort(key=lambda event: float(event["ts"]))
    if len(dma) != 4:
        raise ValueError(f"{path}: expected four metadata transfers, found {len(dma)}")
    # The first two B32 transfers bring the first metadata half. A fresh
    # V->MTE2 wait must separate them from the scratch-reusing second half.
    first_half_end = float(dma[1]["ts"])
    second_half_start = float(dma[2]["ts"])
    matching = [event for event in waits if first_half_end <= float(event["ts"])
                and float(event["ts"]) + float(event["dur"]) <= second_half_start + 0.002]
    if not matching:
        raise ValueError(f"{path}: no fresh V->MTE2 event before scratch reuse")
    wait = max(matching, key=lambda event: float(event["ts"]))
    return {"core": path.parent.name, "wait_us": round(float(wait["dur"]), 3)}


def main():
    paths = sorted(Path(sys.argv[1]).glob("OPPROF_*/simulator/core*/trace.json"))
    if len(paths) != 4:
        raise SystemExit(f"expected four core traces, found {len(paths)} in {sys.argv[1]}")
    print("PASS V->MTE2 scratch dependency:", json.dumps([verify(path) for path in paths]))


if __name__ == "__main__":
    main()
