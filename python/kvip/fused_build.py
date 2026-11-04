# SPDX-License-Identifier: Apache-2.0
"""Startup-only builds of the original LIFetch and Index-update targets."""

import fcntl
import hashlib
import importlib
import json
import logging
import subprocess
from pathlib import Path


def load_original(name):
    """Import the original host modules without fixture paths or name clashes."""
    return importlib.import_module("kvip._lifetch." + name)


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


def build_libraries(
    *, cache_dir, cann, headers, soc, batch, history, resident, aics, windows, token_bytes, index_heads=64
):
    """Compile before capture; concurrent workers share a locked build cache."""
    root = Path(__file__).resolve().parents[2] / "csrc/operators"
    geometry = load_original("geometry")
    shape = geometry.li_geometry(batch, "auto", windows, aics)
    if shape["logical_aic_tasks"] > aics or not shape["intermediate_fetches"]:
        raise ValueError("LIFetch requires concurrently resident LI tasks and at least one Fetch window")
    options = dict(
        RUN_MODE="npu",
        ASCEND_CANN_PACKAGE_PATH=str(cann),
        KVAF_INCLUDE=str(headers),
        SOC_VERSION=soc,
        INDEX_HEADS=index_heads,
        SIM_BATCH=batch,
        SIM_HISTORY=history,
        SIM_RESIDENT=resident,
        AIC_COUNT=aics,
        TOKEN_BYTES=token_bytes,
        LI_TILING="auto",
        INTERMEDIATE_FETCHES=windows,
        INTERMEDIATE_PROFILING=0,
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
    for directory in (root / "lightning_indexer_with_fetch", Path(headers)):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and (path.suffix in (".h", ".cpp", ".py") or path.name == "CMakeLists.txt"):
                digest.update(str(path.relative_to(directory)).encode())
                digest.update(path.read_bytes())
    directory = Path(cache_dir) / digest.hexdigest()[:20]
    directory.mkdir(parents=True, exist_ok=True)
    libraries = directory / "libv9_bridge.so", directory / "libkvop_cache_update.so"
    log = directory / "build.log"
    logging.getLogger("vllm").info("KVAF LIFetch build: batch=%s token_bytes=%s log=%s", batch, token_bytes, log)
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
                        raise RuntimeError(f"LIFetch build failed ({result.returncode}); see {log}")
            if not all(p.is_file() for p in libraries):
                raise RuntimeError(f"LIFetch build did not produce the required libraries; see {log}")
            (directory / "complete.json").write_text(json.dumps(options, indent=2) + "\n")
    return libraries
