# SPDX-License-Identifier: Apache-2.0
"""Startup-only build of the unified LIFetch target."""

import fcntl
import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
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


def _cache_complete(directory, libraries, options):
    try:
        recorded = json.loads((directory / "complete.json").read_text())
        return recorded == options and all(path.is_file() and path.stat().st_size > 0 for path in libraries)
    except (OSError, ValueError):
        return False


def _clear_incomplete(directory):
    """Under build.lock, discard only this variant's interrupted build outputs."""
    for path in directory.iterdir():
        # Keep the inode: another builder may already be waiting on this lock.
        if path.name == "build.lock":
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def _publish_complete(directory, options):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, prefix=".complete-", delete=False) as output:
            temporary = Path(output.name)
            output.write(json.dumps(options, indent=2) + "\n")
        temporary.replace(directory / "complete.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_libraries(
    *,
    cache_dir,
    cann,
    headers,
    soc,
    final_topk_prune=False,
    selected_metadata=False,
    early_fetch_ready=False,
    compact_init=False,
    ready_batch_fetch=False,
    ready_direct_ranges=False,
    adaptive_admission=False,
    low_miss_skip=False,
    low_miss_tail=False,
    low_miss_sparse_init=False,
    auto_route_fast_probe=False,
    auto_route_row_publish=False,
    auto_route_simt=False,
    low_miss_direct_index=False,
    low_miss_compact_plan=False,
    auto_route=False,
    prefetch_gate=False,
    prefetch_rank=0,
    skip_exhausted_prefetch=False,
    early_prefetch_drain=False,
    inverse_mapping=False,
    topk_prune_diagnostics=False,
):
    """Build one shape-independent binary per source/toolchain/SoC."""
    root = Path(__file__).resolve().parents[3] / "csrc/operators"
    options = dict(
        RUN_MODE="npu",
        ASCEND_CANN_PACKAGE_PATH=str(cann),
        KVAF_INCLUDE=str(headers),
        SOC_VERSION=soc,
    )
    variants = dict(
        FINAL_TOPK_PRUNE=final_topk_prune,
        SELECTED_METADATA=selected_metadata,
        EARLY_FETCH_READY=early_fetch_ready,
        COMPACT_INIT=compact_init,
        READY_BATCH_FETCH=ready_batch_fetch,
        READY_DIRECT_RANGES=ready_direct_ranges,
        ADAPTIVE_ADMISSION=adaptive_admission,
        LOW_MISS_SKIP=low_miss_skip,
        LOW_MISS_TAIL=low_miss_tail,
        LOW_MISS_SPARSE_INIT=low_miss_sparse_init,
        AUTO_ROUTE_FAST_PROBE=auto_route_fast_probe,
        AUTO_ROUTE_ROW_PUBLISH=auto_route_row_publish,
        AUTO_ROUTE_SIMT=auto_route_simt,
        LOW_MISS_DIRECT_INDEX=low_miss_direct_index,
        LOW_MISS_COMPACT_PLAN=low_miss_compact_plan,
        AUTO_ROUTE=auto_route,
        PREFETCH_GATE=prefetch_gate,
        SKIP_EXHAUSTED_PREFETCH=skip_exhausted_prefetch,
        EARLY_PREFETCH_DRAIN=early_prefetch_drain,
        INVERSE_MAPPING=inverse_mapping,
        TOPK_PRUNE_DIAGNOSTICS=topk_prune_diagnostics,
    )
    if any(type(value) is not bool for value in variants.values()):
        raise ValueError("LIFetch pipeline build options must be bool")
    if prefetch_gate and not adaptive_admission:
        raise ValueError("prefetch_gate requires adaptive_admission=True")
    if low_miss_skip and not adaptive_admission:
        raise ValueError("low_miss_skip requires adaptive_admission=True")
    if low_miss_tail and not low_miss_skip:
        raise ValueError("low_miss_tail requires low_miss_skip=True")
    if auto_route and not (low_miss_skip and ready_batch_fetch and early_fetch_ready):
        raise ValueError("auto_route requires low_miss_skip, ready_batch_fetch and early_fetch_ready")
    if low_miss_sparse_init and not (auto_route and low_miss_tail and selected_metadata):
        raise ValueError("low_miss_sparse_init requires auto_route, low_miss_tail and selected_metadata")
    if low_miss_sparse_init and inverse_mapping:
        raise ValueError("low_miss_sparse_init and inverse_mapping are separate initialization policies")
    if auto_route_fast_probe and not low_miss_sparse_init:
        raise ValueError("auto_route_fast_probe requires low_miss_sparse_init")
    if auto_route_row_publish and not low_miss_sparse_init:
        raise ValueError("auto_route_row_publish requires low_miss_sparse_init")
    if auto_route_row_publish and auto_route_fast_probe:
        raise ValueError("row publication requires complete evaluation, not fast probes")
    if auto_route_simt and not auto_route_row_publish:
        raise ValueError("auto_route_simt requires auto_route_row_publish")
    if low_miss_direct_index and not (low_miss_tail and selected_metadata and early_fetch_ready):
        raise ValueError("low_miss_direct_index requires low_miss_tail, selected_metadata and early_fetch_ready")
    if low_miss_compact_plan and not (auto_route and low_miss_tail and selected_metadata and early_fetch_ready):
        raise ValueError("low_miss_compact_plan requires auto_route, low_miss_tail, selected_metadata and early_fetch_ready")
    if type(prefetch_rank) is not int or not 0 <= prefetch_rank <= 2048:
        raise ValueError("prefetch_rank must be an integer in 0..2048")
    if prefetch_rank and not prefetch_gate:
        raise ValueError("prefetch_rank requires prefetch_gate=True")
    options["KVIP_PREFETCH_RANK"] = str(prefetch_rank)
    options.update({f"KVIP_{name}": "ON" if value else "OFF" for name, value in variants.items()})
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
    libraries = (directory / "libv9_bridge.so",)
    log = directory / "build.log"
    logging.getLogger("kvip").info("Universal LIFetch build: soc=%s log=%s", soc, log)
    with (directory / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not _cache_complete(directory, libraries, options):
            _clear_incomplete(directory)
            with log.open("w") as output:
                source = root / "lightning_indexer_with_fetch"
                for command in (
                    ["cmake", "-S", str(source), "-B", str(directory), *[f"-D{k}={v}" for k, v in options.items()]],
                    ["cmake", "--build", str(directory), "--target", "v9_bridge", "--parallel", "2"],
                ):
                    output.write(json.dumps(command) + "\n")
                    output.flush()
                    # Keep the same lock alive if the Python parent is killed
                    # while CMake is still writing this variant's artifacts.
                    result = subprocess.run(
                        command, stdout=output, stderr=subprocess.STDOUT, check=False, pass_fds=(lock.fileno(),),
                    )
                    if result.returncode:
                        output.flush()
                        stage = "compile/link" if command[1] == "--build" else "configure"
                        raise build_failure(log, stage, result.returncode)
            if not all(p.is_file() and p.stat().st_size > 0 for p in libraries):
                raise RuntimeError(f"LIFetch build did not produce the required libraries; see {log}")
            _publish_complete(directory, options)
    return libraries
