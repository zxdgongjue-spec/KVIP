# SPDX-License-Identifier: Apache-2.0
"""Sample pipe activity for one uninstrumented four-rank invocation using CANN's acl_prof.h ABI."""
import csv
import ctypes as ct
from contextlib import ExitStack
from math import isfinite
import os
from pathlib import Path
import re
import shutil
import subprocess


SYMBOLS = {"LI1": "v9_li_large_hot", "LI23": "v9_li_large_four_ranks"}


def binding(lib, name, args, result=ct.c_int):
    function = getattr(lib, name)
    function.argtypes = args
    function.restype = result
    return function


def checked(function, *args):
    code = function(*args)
    if code:
        raise RuntimeError(f"{function.__name__} failed: {code}")


def capture(directory, invoke):
    # CANN 9.2 acl/acl_prof.h: task-based time and per-task PipeUtilization.
    runtime = ct.CDLL("libascendcl.so")
    prof = ct.CDLL("libmsprofiler.so")
    device = ct.c_int()
    checked(binding(runtime, "aclrtGetDevice", [ct.POINTER(ct.c_int)]), ct.byref(device))
    if device.value != 0:
        raise ValueError(f"Expected the sole visible device at logical index 0, got {device.value}")
    initialize = binding(prof, "aclprofInit", [ct.c_char_p, ct.c_size_t])
    finalize = binding(prof, "aclprofFinalize", [])
    create = binding(prof, "aclprofCreateConfig",
                     [ct.POINTER(ct.c_uint32), ct.c_uint32, ct.c_int, ct.c_void_p, ct.c_uint64], ct.c_void_p)
    destroy = binding(prof, "aclprofDestroyConfig", [ct.c_void_p])
    start = binding(prof, "aclprofStart", [ct.c_void_p])
    stop = binding(prof, "aclprofStop", [ct.c_void_p])
    directory.mkdir()
    encoded = os.fsencode(directory.resolve())
    with ExitStack() as stack:
        checked(initialize, encoded, len(encoded))
        stack.callback(checked, finalize)
        config = create(ct.byref(ct.c_uint32(device.value)), 1, 1, None, 0x0002 | 0x0004)
        if not config:
            raise RuntimeError("aclprofCreateConfig returned a null PipeUtilization config")
        stack.callback(checked, destroy, config)
        checked(start, config)
        stack.callback(checked, stop, config)
        invoke()


def norm(key):
    return re.sub(r"[^a-z0-9]", "", key.lower())


def read_pipe_rows(folder):
    stages = {stage: [] for stage in SYMBOLS}
    names = []
    for source in sorted(folder.rglob("op_summary*.csv")):
        names.append(str(source.relative_to(folder)))
        with source.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                labels = " ".join(str(v) for k, v in row.items() if k and norm(k) in
                                  {"opname", "optype", "kernelname", "taskname", "name", "type"})
                for stage, symbol in SYMBOLS.items():
                    if symbol not in labels or symbol + "_trace" in labels:
                        continue
                    values = {}
                    for key, val in row.items():
                        if key and re.search(r"mte|mac|cube|vec|scalar|fixpipe|icache|totalcycle|totaltime|duration", norm(key)):
                            try:
                                value = float(val.strip().rstrip("%"))
                            except (TypeError, AttributeError, ValueError):
                                continue
                            if isfinite(value):
                                values[key] = val.strip()
                    stages[stage].append(dict(names=labels, source=str(source.relative_to(folder)),
                                              task={k: v for k, v in row.items() if k and norm(k) in
                                                    {"taskid", "streamid", "tasktype", "coretype", "blockdim"}},
                                              counters=values))
    def has_counter(rows, family):
        return any(re.search(family, norm(key)) for row in rows for key in row["counters"])
    coverage = {stage: {family: has_counter(rows, pattern) for family, pattern in
                        (("data_move", "mte2"), ("matrix", "mac|cube"), ("vector", "aivvec|vec(time|ratio|cycle)"))}
                for stage, rows in stages.items()}
    return dict(status="COMPLETE" if all(all(f.values()) for f in coverage.values()) else "PARTIAL",
                stages=stages, coverage=coverage, raw_csv=names)


def collect(state, directory):
    directory = Path(directory)
    folder = directory / "li-pmu"
    if (state.forward.diagnostics_enabled or getattr(state.large_window_launch, "trace_only", True)
            or state.large_window_launch.window_policy != "four-ranks"):
        raise ValueError("LI pipe counters require the normal four-rank cold kernel with diagnostics disabled")
    stream = state.torch.npu.current_stream(state.device)
    stream.synchronize()
    def invoke():
        state.forward()
        stream.synchronize()
    capture(folder, invoke)
    sessions = [p for p in folder.glob("PROF_*") if p.is_dir()]
    if len(sessions) != 1:
        return dict(status="PARTIAL", reason=f"Found {len(sessions)} PROF directories", raw=str(folder))
    tool = shutil.which("msprof")
    if not tool:
        prefix = os.environ.get("ASCEND_CANN_PACKAGE_PATH", "")
        executable = Path(prefix) / "bin/msprof"
        tool = str(executable) if executable.is_file() else None
    if not tool:
        return dict(status="PARTIAL", reason="msprof exporter unavailable", raw=str(folder))
    with (directory / "msprof-export.log").open("w") as log:
        proc = subprocess.run([tool, "--export=on", "--type=text", "--output=" + str(sessions[0])],
                              stdout=log, stderr=subprocess.STDOUT)
    result = read_pipe_rows(folder)
    result.update(metric_group="PipeUtilization", export_rc=proc.returncode, raw=str(folder))
    if proc.returncode:
        result["status"] = "PARTIAL"
    return result
