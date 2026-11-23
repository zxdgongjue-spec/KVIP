#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Find executed owner milestones in detailed first-replay C310 instruction traces."""
import argparse
import json
from pathlib import Path
import re
import subprocess


PC = re.compile(r"\[(\d+)\] \(PC: (0x[0-9a-f]+)\)")
DMA = re.compile(r"XD:X\d+=(0x[0-9a-f]+),.* XT:X\d+=(0x[0-9a-f]+)")
SOURCE = re.compile(r"^(.+?):(\d+):\d+$")
COLD = "v9_li_kernel_2_2_mix_aiv"


def cold_trace(path):
    task = 0
    records = []
    with path.open() as trace:
        for line in trace:
            if "done++" in line:
                task += 1
                if task == 3:
                    break
            elif task == 2 and (match := PC.search(line)):
                records.append((int(match[1]), int(match[2], 16)))
    if not records or task != 3:
        raise ValueError(f"Missing full first cold-kernel trace in {path}")
    return records


def decode(obj, traces, symbolizer):
    nm = subprocess.check_output(["nm", "-n", str(obj)], text=True)
    entry, = [int(line.split()[0], 16) for line in nm.splitlines()
              if line.split()[1:] == ["T", COLD]]
    base, = {records[0][1] - entry for records in traces.values()}
    pcs = sorted({pc for records in traces.values() for _, pc in records})
    data = "".join(f"{pc-base:#x}\n" for pc in pcs)
    symbols = subprocess.run([str(symbolizer), "--obj=" + str(obj)],
                             input=data, text=True, capture_output=True, check=True)
    blocks = symbols.stdout.rstrip().split("\n\n")
    if len(blocks) != len(pcs):
        raise ValueError("Incomplete ELF line information")
    frames = {}
    for pc, stack in zip(pcs, blocks):
        located = set()
        for line in stack.splitlines():
            if match := SOURCE.match(line):
                located.add((Path(match[1]).name, int(match[2])))
        frames[pc] = (located, stack)
    return frames


def source_line(path, key, after):
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if after in line)
    index = next(i for i in range(start, len(lines)) if key in lines[i])
    return path.name, index + 1


def collect(records, frames, root, epoch):
    vector = root / "indexer"
    last = vector / "final_window_queries.h"
    is_team = epoch == 3 and last.exists()
    method = last if is_team else vector / "incremental_methods.h"
    start = "inline uint64_t SnapshotLastWindow(" if is_team else "inline bool AdvanceWindow("
    common = root / "prefetch.h"
    begin = source_line(method, "uint64_t begin=trace", start)
    completed_dma = source_line(method, "SetFlag<HardEvent::MTE2_S>(EVENT_ID5)", start)
    markers = {
        "predecessor_released": begin,
        "snapshot_dma_barrier": completed_dma,
        "post_workspace_fill": source_line(common, "postInput.SetValue(64,count)", "inline void FastService("),
        "build_wqe_batch": source_line(common, "Simt::VF_CALL<BatchBuild<", "inline void FastService("),
    }
    if is_team:
        impl = "inline void RunAssignedWindowQuery(" if "RunAssignedWindowQuery(" in last.read_text() else "inline void RunLastWindowQuery("
        markers["owner_reads_own_query_header"] = source_line(last, "DataCopy(incrementalMetadata_,meta,8)", impl)
    pc_groups = {name: {pc for pc, (locations, _) in frames.items() if key in locations}
                 for name, key in markers.items()}
    pc_groups["post_service"] = {pc for pc, (_, stack) in frames.items() if "rotating::FastService<" in stack}
    if is_team:
        pc_groups["publish_notifications"] = {pc for pc, (_, stack) in frames.items()
            if "SnapshotLastWindow(" in stack and ("SendLastWindowMail(" in stack or "PublishLastWindowCommand(" in stack)}
    released = next(tick for tick, pc in records if pc in pc_groups["predecessor_released"])
    result = {}
    for name, pcs in pc_groups.items():
        hits = [(tick, hex(pc)) for tick, pc in records if tick >= released and pc in pcs]
        if not hits:
            if name in ("owner_reads_own_query_header", "publish_notifications"):
                # Inlining may attribute a device instruction to the caller.
                continue
            raise ValueError(f"Missing executed marker {name} in epoch {epoch}")
        result[name] = {"first": hits[0], "last": hits[-1]}
    if result["snapshot_dma_barrier"]["last"][0] >= result["post_workspace_fill"]["first"][0]:
        raise ValueError("Owner DMA and post markers are out of order")
    return result


def query_phases(records, frames):
    groups = (
        ("last_query", lambda stack: "RunLastWindowQuery(" in stack),
        ("incremental_update", lambda stack: "RunLastWindowQuery(" in stack and "AdvanceIncremental(" in stack),
        ("signal_completion", lambda stack: "RunLastWindowQuery(" in stack and "SendLastWindowMail(" in stack),
        ("owner_wait_for_team", lambda stack: "CompleteLastWindow(" in stack and "WaitLastWindowMail(" in stack
         and "RunLastWindowQuery(" not in stack),
    )
    result = {}
    for name, pred in groups:
        pcs = {pc for pc, (_, stack) in frames.items() if pred(stack)}
        hits = [(tick, hex(pc)) for tick, pc in records if pc in pcs]
        if hits:
            result[name] = {"first": hits[0], "last": hits[-1]}
    return result


def topk_writes(paths, frames, root):
    source = root / "indexer/incremental_methods.h"
    stores = {(source.name, number) for number, line in enumerate(source.read_text().splitlines(), 1)
              if "DataCopy(stateScores" in line or "DataCopy(stateIds" in line}
    members = {pc: stack for pc, (locations, stack) in frames.items() if locations & stores}
    windows = {}
    total = 0
    for (role, row), path in paths.items():
        task = 0
        with path.open() as trace:
            for line in trace:
                if "done++" in line:
                    task += 1
                elif task == 2 and "Src:UB, Dst:OUT" in line:
                    match = PC.search(line)
                    pc = int(match[2], 16)
                    if pc not in members:
                        continue
                    transfer = DMA.search(line)
                    if not transfer or "MOV_SRC_TO_DST_ALIGNv2" not in line:
                        raise ValueError(f"Unsupported TopK transfer at {pc:#x}")
                    # Each write is one contiguous byte row on this model.
                    # The low 32 bits of XT give its length in bytes.
                    size = int(transfer[2], 16) & 0xffffffff
                    if not 0 < size <= 8192 or size % 32:
                        raise ValueError(f"Invalid transfer length at {pc:#x}")
                    stack = members[pc]
                    if "RunLastWindowQuery(" in stack:
                        epoch = 3
                    elif "OfferFinishedScorer(" in stack:
                        epoch = 2
                    else:
                        epoch = role
                    windows.setdefault(f"e{epoch}row{row}", []).append(
                        {"tick": int(match[1]), "pc": hex(pc), "destination": transfer[1], "bytes": size})
                    total += size
    if not total:
        raise ValueError("No persisted TopK score or ID write recorded")
    return {"bytes": total, "transactions": windows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("linked_device_object", type=Path)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--symbolizer", type=Path, default="/usr/local/Ascend/cann-9.2.0/tools/bisheng_compiler/bin/llvm-symbolizer")
    parser.add_argument("--queries", action="store_true",
                        help="also resolve final-window worker entry, incremental update and completion")
    parser.add_argument("--writes", action="store_true", help="decode the actual TopK score/ID device writes")
    args = parser.parse_args()
    root = args.source_root / "csrc/operators/lightning_indexer_with_fetch/op_kernel"
    roles = ((0, 0), (1, 2), (2, 1), (3, 3))
    paths = {(role, row): args.run / "model" / f"core{cube}.veccore{row}.instr_log.dump"
             for role, cube in roles if args.queries or args.writes or role >= 2 for row in range(2)}
    traces = {lane: cold_trace(path) for lane, path in paths.items()}
    frames = decode(args.linked_device_object, traces, args.symbolizer)
    result = {f"e{epoch}row{row}": collect(traces[epoch, row], frames, root, epoch)
              for epoch in (2, 3) for row in range(2)}
    if args.queries:
        result["last_window_queries"] = {f"row{row}query{role}": query_phases(records, frames)
                                         for (role, row), records in traces.items()}
    if args.writes:
        result["persisted_topk_writes"] = topk_writes(paths, frames, root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
