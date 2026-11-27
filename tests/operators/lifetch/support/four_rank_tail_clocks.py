# SPDX-License-Identifier: Apache-2.0
"""Isolated terminal diagnostic for the four-rank candidate. Timestamps are observed by the consuming AIV."""
import ctypes
import fcntl
import hashlib
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

from b24_tail_archive import query_spans, require


def build_probe(repo, normal_library, sdk_build, soc):
    from kvip.runtime.build import build_failure, kvaf_build_paths
    cann, headers = kvaf_build_paths(sdk_build)
    op = repo / "csrc/operators/lightning_indexer_with_fetch"
    src = Path(__file__).with_name("terminal_batch_clock_native")
    digest = hashlib.sha256(normal_library.parent.name.encode())
    for file in sorted(src.iterdir()):
        if file.is_file():
            digest.update(file.read_bytes())
    folder = normal_library.parent / ("terminal-private-clock-" + digest.hexdigest()[:16])
    folder.mkdir(exist_ok=True)
    library = folder / "libterminal_clock_bridge.so"
    log = folder / "build.log"
    with (folder / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.is_file():
            with tempfile.TemporaryDirectory(dir=folder) as tmp, log.open("w") as output:
                stage = Path(tmp)
                for name in ("kernel.cpp", "bridge.cpp", "CMakeLists.txt"):
                    shutil.copyfile(src / name, stage / name)
                for name in ("row_kernel.h", "terminal_transport.h"):
                    shutil.copyfile(op / "op_kernel/terminal" / name, stage / name)
                for command in (
                    ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(src / "device-clock.patch")],
                    ["cmake", "-S", ".", "-B", "build", f"-DOP={op}", f"-DSOC_VERSION={soc}",
                     f"-DASCEND_CANN_PACKAGE_PATH={cann}", f"-DKVAF_INCLUDE={headers}"],
                    ["cmake", "--build", "build", "--target", "terminal_clock_bridge", "--parallel", "2"]
                ):
                    status = subprocess.run(command, cwd=stage, stdout=output, stderr=subprocess.STDOUT)
                    if status.returncode:
                        output.flush()
                        raise build_failure(log, "terminal batch clocks", status.returncode)
                (stage / "build/libterminal_clock_bridge.so").replace(library)
    return ctypes.CDLL(str(library))


def expected_final_prefixes(own, total):
    if not own:
        return []
    ends = [(own * n + 15) // 16 for n in (1, 6, 11)] if total >= 256 and own > 1 else []
    return list(dict.fromkeys([*ends, own]))


def decode(stamps, detail, mhz):
    require(math.isfinite(mhz) and mhz > 0 and len(stamps) == len(detail), "Invalid clock calibration/shape")
    origin = min(q[0] for row in stamps for q in row)
    last_row, last_q = max(((r, q) for r in range(len(stamps)) for q in range(4)),
                           key=lambda x: stamps[x[0]][x[1]][10])
    out = []
    for r, (row, clocks) in enumerate(zip(stamps, detail)):
        require(len(row) == len(clocks) == 4, "Invalid Q geometry")
        total = sum(t[20] for t in row)
        for q, (t, extra) in enumerate(zip(row, clocks)):
            stages = query_spans(t, q)
            own = t[20]
            ends = expected_final_prefixes(own, total)
            require(len(extra) == 32 and extra[0] == len(ends) and extra[1:1+len(ends)] == ends,
                    f"G7 R{r} Q{q+1}: evaluated prefix counts do not match")
            windows = []
            last = t[16]
            for b, end in enumerate(ends):
                at = extra[8+4*b:12+4*b]
                if not b:
                    at = [t[15], t[4], at[2], at[3]]
                require(at[0] <= at[1] <= at[2] <= at[3] <= t[24] and at[0] > 0
                        and (last <= at[0] if b else at[1] <= t[26] <= t[16] <= at[2]),
                        f"G7 R{r} Q{q+1}: invalid Post/Index/CQ chronology")
                windows.append(dict(prefix_end=end, post_begin=at[0], post_end=at[1],
                                    drain_begin=at[2], drain_end=at[3]))
                last = at[3]
            out.append(dict(row=r, query=q+1, own=own, phases_us={k:v/mhz for k,v in stages.items()},
                       start_us=(t[0]-origin)/mhz, until_exit_us=(t[10]-origin)/mhz,
                       index_begin=t[26], index_end=t[16], batches=windows,
                       last_final_cq_to_exit_us=(t[10]-last)/mhz if windows else None))
    return dict(last_row=last_row, last_q=last_q+1, qs=out, origin=origin,
                terminal_span_us=(stamps[last_row][last_q][10]-origin)/mhz)


def collect(state, directory, repo):
    import torch
    from large_batch_pair import check_ordered_index
    launch = state.large_window_launch
    _, normal, sdk_build, soc, policy = state.large_window_build_args
    require(policy == "four-ranks" and not launch.trace_only, "Expected original timed LI")
    lib = build_probe(repo, normal, sdk_build, soc)
    tail = ctypes.cast(lib.kvip_test_terminal_clocks, ctypes.c_void_p)
    hot = ctypes.cast(state.rotating.library.aclrtlaunch_v9_li_large_hot, ctypes.c_void_p)
    fwd, terminal = state.forward, state.forward.terminal
    words = torch.zeros((2, state.batch, 4, 32), dtype=torch.int64, device=state.device)
    pointer = torch.tensor(list(struct.pack("<Q", words.data_ptr())), dtype=torch.uint8, device=state.device)
    mhz = fwd.calibrate_counter_mhz()
    stream = torch.npu.current_stream(state.device)
    saved = fwd.terminal_function
    old_pointer = terminal.workspace[32:40].clone()
    require(int.from_bytes(bytes(old_pointer.cpu().tolist()), "little") == terminal.times.data_ptr(),
            "Unexpected terminal timing pointer")
    # These tensor handles stay live until the isolated diagnostic worker exits, including a quarantine path.
    state._terminal_probe = (words, lib, pointer, old_pointer)
    fwd.events_src.fill_(-1)
    fwd.events_dst.fill_(-1)
    terminal.workspace[32:40].copy_(pointer)
    fwd.terminal_function = lambda a,b,s,shape: launch(a,b,s,shape,hot,tail)
    fwd.set_diagnostics(True)
    stream.synchronize()
    fwd()
    stream.synchronize()
    fwd.terminal_function = saved
    terminal.workspace[32:40].copy_(old_pointer)
    fwd.set_diagnostics(False)
    stream.synchronize()
    stamps, detail = words.cpu().tolist()
    report = decode(stamps, detail, mhz)
    check_ordered_index(state, state.initial.captured_layout, state.initial.captured_hot_valid)
    # The original Index journal separates initial window admissions from the final assignments.
    # Keep token/slot identities for later CPU comparisons of this specific diagnostic call.
    tokens = fwd.events_src.cpu().numpy()
    slots = fwd.events_dst.cpu().numpy()
    window_ends = state.rotating.rows[:, 9].cpu().tolist()
    journal_ends = state.rotating.change_count.cpu().tolist()
    final_assignments = [
        dict(tokens=tokens[i, first:end].tolist(), hot_slots=slots[i, first:end].tolist())
        for i, (first, end) in enumerate(zip(window_ends, journal_ends))
    ]
    r, q = report["last_row"], report["last_q"]
    print(f"LAST_TEAM G7 BS{state.batch} row={r} Q={q} diagnostic_span_us={report['terminal_span_us']:.2f}")
    useful = [x for x in report["qs"] if x["batches"]]
    latest = max(useful, key=lambda x:x["batches"][-1]["drain_end"]) if useful else None
    for entry in report["qs"]:
        if entry["row"] != r and entry is not latest:
            continue
        spans = entry["batches"]
        drains = "/".join(f"{(c['drain_end']-c['drain_begin'])/mhz:.2f}" for c in spans) or "-"
        posts = "/".join(f"{(c['post_end']-report['origin'])/mhz:.2f}" for c in spans) or "-"
        after = entry["last_final_cq_to_exit_us"]
        print(f"BATCH_CLOCK G7 BS{state.batch} row={entry['row']} Q={entry['query']} final={entry['own']} "
              f"post_end_from_first_entry_us={posts} Index_join_us={entry['phases_us']['Index_join']:.2f} "
              f"drain_duration_us={drains} last_final_CQ_to_exit_us={after if after is None else round(after,2)}")
    return dict(clock_mhz=mhz, separate_terminal_query_stamps=stamps, prefix_clock_words=detail,
                decoded=report, final_index_assignments_by_row=final_assignments, ordered_Index_rows=state.batch)
