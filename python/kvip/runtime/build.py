# SPDX-License-Identifier: Apache-2.0
"""Startup-only build of the unified LIFetch target."""

import fcntl
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
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
    objects = []
    if 'unknown file type' in '\n'.join(lines).lower():
        for path in sorted(log.parent.rglob('*.cpp.o')):
            if not any(part in ('aiv_obj.dir', 'aic_obj.dir') for part in path.parts):
                continue
            try:
                with path.open('rb') as source:
                    header = source.read(32)
                kind = 'empty' if not header else 'non-ELF'
                if header.startswith(b'\x7fELF') and len(header) >= 20:
                    endian = 'little' if header[5] == 1 else 'big'
                    elf_type = int.from_bytes(header[16:18], endian)
                    machine = int.from_bytes(header[18:20], endian)
                    kind = f"ELF type={elf_type} ({'REL' if elf_type == 1 else 'EXEC/already linked' if elf_type == 2 else 'other'}), machine=0x{machine:x}"
                objects.append(f'{path}: {path.stat().st_size} bytes; {kind}')
            except OSError as error:
                objects.append(f'{path}: unreadable ({error})')
            if len(objects) >= 8:
                break
    details = '\n--- preprocess object diagnostics ---\n' + '\n'.join(objects) if objects else ''
    return RuntimeError(
        f"LIFetch {stage} failed (exit={returncode}); full log: {log}\n--- build diagnostic ---\n{excerpt}{details}"
    )


def build_libraries(*, cache_dir, cann, headers, soc):
    """Build one shape-independent binary per source/toolchain/SoC."""
    root = Path(__file__).resolve().parents[3] / "csrc/operators"
    options = dict(
        RUN_MODE="npu",
        ASCEND_CANN_PACKAGE_PATH=str(cann),
        KVAF_INCLUDE=str(headers),
        SOC_VERSION=soc,
    )
    digest = hashlib.sha256(json.dumps(options, sort_keys=True).encode())
    # CANN merge_obj_text.sh replaces REL objects in place with EXEC objects.
    # An incomplete CMake tree must never be reused for a later attempt.
    digest.update(b'kvip-fresh-build-attempt-v1')
    for relative in (
        "version.info",
        "compiler/ccec_compiler/bin/bisheng",
        "aarch64-linux/ccec_compiler/bin/bisheng",
        "aarch64-linux/ccec_compiler/bin/ld.lld",
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
            attempt = Path(tempfile.mkdtemp(prefix='attempt-', dir=directory))
            # Keep the lock at the stable cache root. Each attempt has its own
            # CMake/ExternalProject stamps, objects and log; failed evidence stays.
            public_log = directory / 'build.log'
            if public_log.is_symlink():
                public_log.unlink()
            elif public_log.exists():
                public_log.rename(directory / f'legacy-build-{uuid.uuid4().hex}.log')
            log = attempt / 'build.log'
            public_log.symlink_to(log.relative_to(directory))
            logging.getLogger('kvip').info('Fresh LIFetch build attempt: %s', attempt)
            with log.open("w") as output:
                source = root / "lightning_indexer_with_fetch"
                for command in (
                    ["cmake", "-S", str(source), "-B", str(attempt), *[f"-D{k}={v}" for k, v in options.items()]],
                    ["cmake", "--build", str(attempt), "--target", "v9_bridge", "--parallel", "2"],
                ):
                    output.write(json.dumps(command) + "\n")
                    output.flush()
                    result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False)
                    if result.returncode:
                        output.flush()
                        stage = "compile/link" if command[1] == "--build" else "configure"
                        raise build_failure(log, stage, result.returncode)
            if not all((attempt / p.name).is_file() for p in libraries):
                raise RuntimeError(f"LIFetch build did not produce the required libraries; see {log}")
            for library in libraries:
                pending = library.with_suffix(library.suffix + '.pending')
                shutil.copy2(attempt / library.name, pending)
                os.replace(pending, library)
            complete = directory / 'complete.pending.json'
            complete.write_text(json.dumps({**options, 'build_directory': str(attempt)}, indent=2) + '\n')
            os.replace(complete, directory / 'complete.json')
    return libraries
