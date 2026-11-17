# SPDX-License-Identifier: Apache-2.0
"""Startup-only build of the unified LIFetch target."""

import fcntl
import hashlib
import json
import logging
import subprocess
from pathlib import Path


def kvaf_build_paths(build_dir):
    """Use the installed Runtime's actual CANN and source, not guessed paths."""
    values = {}
    for line in (Path(build_dir) / "CMakeCache.txt").read_text().splitlines():
        if line.startswith(("#", "//")) or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.split(":", 1)[0]] = value
    cann = Path(values["ASCEND_CANN_PACKAGE_PATH"])
    headers = Path(values["CMAKE_HOME_DIRECTORY"]) / "access_operator/demo/aiv_batch_get/include"
    for path in (cann / "compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake", headers / "kvaf_batch_get_device.h"):
        if not path.is_file():
            raise FileNotFoundError(f"LIFetch build dependency unavailable: {path}")
    return cann, headers


def build_failure(log, stage, returncode):
    """Expose the first diagnostic, not just recursive make's final exit."""
    lines = log.read_text(errors="replace").splitlines()
    markers = ("error:", "fatal error:", "cmake error", "error generated", "errors generated")
    first = next((i for i, line in enumerate(lines) if any(m in line.lower() for m in markers)), None)
    start = max(0, first - 5) if first is not None else max(0, len(lines) - 35)
    excerpt = "\n".join(lines[start : start + 35])
    return RuntimeError(
        f"LIFetch {stage} failed (exit={returncode}); full log: {log}\n--- build diagnostic ---\n{excerpt}"
    )


def build_libraries(*, cache_dir, cann, headers, soc, reverse_cold=False,
                    rolling_threshold=False, window_refilter=False, rolling_batched=False, rolling_windowed=False,
                    rolling_compact=False, rolling_first_window=False, refilter_safe=False, refilter_refill=False,
                    compact_out_of_place=False, rolling_reuse_seed=False, rolling_fast_path=False, rolling_sparse=False, rolling_boundary=False, incremental_topk=False,
                    incremental_window=False, incremental_register=False, incremental_prefilter=False, incremental_direct=False, incremental_ties=False, incremental_light=False, incremental_inplace=False):
    """Build one shape-independent binary per source/toolchain/SoC."""
    if incremental_inplace and not (incremental_light and incremental_ties):
        raise ValueError("In-place boundary updates require light state and exact tie updates")
    if incremental_light and not incremental_direct:
        raise ValueError("Light incremental state requires direct selection")
    if incremental_ties and not incremental_direct:
        raise ValueError("Boundary tie updates require direct incremental state")
    if incremental_direct and not incremental_prefilter:
        raise ValueError("Direct incremental state requires window prefilter")
    if incremental_prefilter and not incremental_register:
        raise ValueError("Window prefilter requires register window selection")
    if incremental_window and not incremental_topk:
        raise ValueError("Window staging requires incremental TopK")
    if incremental_register and not incremental_window:
        raise ValueError("Register selection requires incremental window staging")
    if incremental_topk and (rolling_threshold or window_refilter):
        raise ValueError("Incremental TopK replaces rolling/refilter")
    if (window_refilter or rolling_batched or rolling_windowed) and not rolling_threshold:
        raise ValueError("Window refilter and batched counts require rolling threshold")
    if rolling_batched and rolling_windowed:
        raise ValueError("Select either batch or window accumulation")
    if (rolling_compact or rolling_first_window) and not rolling_windowed:
        raise ValueError("Compact/first-window options require window accumulation")
    if rolling_first_window and not rolling_compact:
        raise ValueError("First-window correction requires compact admission")
    if rolling_reuse_seed and not rolling_compact:
        raise ValueError("Seed reuse requires compact window accumulation")
    if rolling_boundary and (not rolling_fast_path or rolling_sparse or rolling_first_window or rolling_batched):
        raise ValueError("Boundary readback requires dense fast window counts and every window")
    if rolling_sparse and not rolling_fast_path:
        raise ValueError("Sparse window counts require packed seed reuse")
    if rolling_fast_path and not rolling_reuse_seed:
        raise ValueError("Packed byte accumulation requires seed histogram reuse")
    if refilter_safe and not window_refilter:
        raise ValueError("Visibility-safe admission requires window refilter")
    if refilter_refill and not refilter_safe:
        raise ValueError("Same-window refill requires visibility-safe admission")
    root = Path(__file__).resolve().parents[3] / "csrc/operators"
    options = dict(
        RUN_MODE="npu",
        ASCEND_CANN_PACKAGE_PATH=str(cann),
        KVAF_INCLUDE=str(headers),
        SOC_VERSION=soc,
        KVIP_INCREMENTAL_WINDOW="ON" if incremental_window else "OFF",
        KVIP_INCREMENTAL_REGISTER="ON" if incremental_register else "OFF",
        KVIP_INCREMENTAL_PREFILTER="ON" if incremental_prefilter else "OFF",
        KVIP_INCREMENTAL_INPLACE="ON" if incremental_inplace else "OFF",
        KVIP_INCREMENTAL_LIGHT="ON" if incremental_light else "OFF",
        KVIP_INCREMENTAL_DIRECT="ON" if incremental_direct else "OFF",
        KVIP_INCREMENTAL_TIES="ON" if incremental_ties else "OFF",
        KVIP_INCREMENTAL_TOPK="ON" if incremental_topk else "OFF",
        KVIP_ROLLING_REUSE_SEED="ON" if rolling_reuse_seed else "OFF",
        KVIP_ROLLING_BOUNDARY="ON" if rolling_boundary else "OFF",
        KVIP_ROLLING_SPARSE="ON" if rolling_sparse else "OFF",
        KVIP_ROLLING_FAST_PATH="ON" if rolling_fast_path else "OFF",
        KVIP_COMPACT_OUT_OF_PLACE="ON" if compact_out_of_place else "OFF",
        KVIP_REFILTER_SAFE="ON" if refilter_safe else "OFF",
        KVIP_REFILTER_REFILL="ON" if refilter_refill else "OFF",
        KVIP_REVERSE_COLD="ON" if reverse_cold else "OFF",
        KVIP_ROLLING_THRESHOLD="ON" if rolling_threshold else "OFF",
        KVIP_ROLLING_COMPACT="ON" if rolling_compact else "OFF",
        KVIP_ROLLING_FIRST_WINDOW="ON" if rolling_first_window else "OFF",
        KVIP_ROLLING_WINDOWED="ON" if rolling_windowed else "OFF",
        KVIP_ROLLING_BATCHED="ON" if rolling_batched else "OFF",
        KVIP_WINDOW_REFILTER="ON" if window_refilter else "OFF",
    )
    digest = hashlib.sha256(json.dumps(options, sort_keys=True).encode())
    for relative in (
        "version.info",
        "compiler/ccec_compiler/bin/bisheng",
        "compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake",
    ):
        tool = Path(cann) / relative
        if tool.is_file():
            stat = tool.stat()
            digest.update(f"{tool.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    for directory in (root / "lightning_indexer_with_fetch", root / "common", Path(headers)):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and (path.suffix in (".h", ".cpp", ".py") or path.name == "CMakeLists.txt"):
                digest.update(str(path.relative_to(directory)).encode())
                digest.update(path.read_bytes())
    shared_selector = root / "lidu/op_kernel/hot_slot_selection.h"
    digest.update(str(shared_selector.relative_to(root)).encode())
    digest.update(shared_selector.read_bytes())
    directory = Path(cache_dir) / digest.hexdigest()[:20]
    directory.mkdir(parents=True, exist_ok=True)
    libraries = directory / "libv9_bridge.so",
    log = directory / "build.log"
    logging.getLogger("kvip").info("Universal LIFetch build: soc=%s log=%s", soc, log)
    with (directory / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (directory / "complete.json").is_file() or not all(p.is_file() for p in libraries):
            with log.open("w") as output:
                source = root / "lightning_indexer_with_fetch"
                for command in (
                    ["cmake", "-S", str(source), "-B", str(directory), *[f"-D{k}={v}" for k, v in options.items()]],
                    ["cmake", "--build", str(directory), "--target", "v9_bridge", "--parallel", "2"],
                ):
                    output.write(json.dumps(command) + "\n")
                    output.flush()
                    result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False)
                    if result.returncode:
                        output.flush()
                        stage = "compile/link" if command[1] == "--build" else "configure"
                        raise build_failure(log, stage, result.returncode)
            if not all(p.is_file() for p in libraries):
                raise RuntimeError(f"LIFetch build did not produce the required libraries; see {log}")
            (directory / "complete.json").write_text(json.dumps(options, indent=2) + "\n")
    return libraries
