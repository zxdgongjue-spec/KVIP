# SPDX-License-Identifier: Apache-2.0
"""Startup-only, content-addressed builds for the static UBmem pipeline."""

import ctypes
import fcntl
import hashlib
import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from kvip.runtime.build import build_failure


@dataclass(frozen=True)
class Specialization:
    batch: int
    history: int
    resident: int
    heads: int
    token_bytes: int
    aic_count: int
    intermediate_fetches: int
    soc: str

    def __post_init__(self):
        if not 1 <= self.batch <= 16 or not 1 <= self.aic_count <= 32:
            raise ValueError("UBmem LIFetch requires 1..16 requests and 1..32 AICs")
        if self.batch > self.aic_count:
            raise ValueError("UBmem LIFetch requests must fit concurrently resident LI cores")
        if self.history % 256 or self.resident % 256 or not 8192 <= self.resident < self.history <= 210944:
            raise ValueError("UBmem history/resident must align to 256, with 8192 <= resident < history <= 210944")
        if self.heads not in (32, 64) or self.token_bytes not in (656, 2624):
            raise ValueError("UBmem supports 32/64 Indexer heads and 656/2624-byte KV rows")
        if not 1 <= self.intermediate_fetches <= 32 or not self.soc.startswith("Ascend950"):
            raise ValueError("UBmem requires 1..32 prefetch windows and an explicit Ascend950 SoC")
        cores = self.batch * (2 if 2 * self.batch <= self.aic_count else 1)
        if (self.history - self.resident) // 128 <= min(self.intermediate_fetches, cores):
            raise ValueError("UBmem requires more cold tiles than active prefetch windows")

    def options(self, cann):
        return dict(
            RUN_MODE="npu",
            ASCEND_CANN_PACKAGE_PATH=str(Path(cann).resolve()),
            SOC_VERSION=self.soc,
            SIM_BATCH=self.batch,
            SIM_HISTORY=self.history,
            SIM_RESIDENT=self.resident,
            INDEX_HEADS=self.heads,
            TOKEN_BYTES=self.token_bytes,
            AIC_COUNT=self.aic_count,
            INTERMEDIATE_FETCHES=self.intermediate_fetches,
        )


def kvaf_cann(build_dir):
    """Resolve CANN from this runtime's build without URMA header dependencies."""
    for line in (Path(build_dir) / "CMakeCache.txt").read_text().splitlines():
        if line.startswith("ASCEND_CANN_PACKAGE_PATH:"):
            return Path(line.split("=", 1)[1])
    raise ValueError(f"CANN path is missing from {build_dir}/CMakeCache.txt")


def build_specialization(*, cache_dir, cann, shape, jobs=2):
    """Compile once before profiling/capture; never called by forward."""
    if not isinstance(shape, Specialization) or jobs < 1:
        raise ValueError("A validated UBmem specialization and positive jobs are required")
    source = Path(__file__).resolve().parents[4] / "csrc/operators/lightning_indexer_with_fetch_ubmem"
    if not (source / "CMakeLists.txt").is_file():
        raise FileNotFoundError(f"UBmem fusion sources unavailable: {source}")
    options = shape.options(cann)
    digest = hashlib.sha256(json.dumps(options, sort_keys=True).encode())
    for directory in (source, source.parent / "common"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and (path.suffix in (".h", ".cpp", ".asc", ".cmake") or path.name == "CMakeLists.txt"):
                digest.update(str(path.relative_to(source.parent)).encode())
                digest.update(path.read_bytes())
    for relative in (
        "version.info",
        "compiler/ccec_compiler/bin/bisheng",
        "compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake",
        "tools/ccec_compiler/bin/ccec",
    ):
        tool = Path(cann) / relative
        if tool.is_file():
            stat = tool.stat()
            digest.update(f"{tool.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    directory = Path(cache_dir) / digest.hexdigest()[:20]
    directory.mkdir(parents=True, exist_ok=True)
    library, log = directory / "libkvip_lifetch_ubmem.so", directory / "build.log"
    logging.getLogger("kvip").info("UBmem LIFetch specialization=%s log=%s", asdict(shape), log)
    with (directory / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (directory / "complete.json").is_file() or not library.is_file():
            with log.open("w") as output:
                for command in (
                    ["cmake", "-S", str(source), "-B", str(directory), *[f"-D{k}={v}" for k, v in options.items()]],
                    ["cmake", "--build", str(directory), "--target", "kvip_lifetch_ubmem", "--parallel", str(jobs)],
                ):
                    output.write(json.dumps(command) + "\n")
                    output.flush()
                    result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False)
                    if result.returncode:
                        output.flush()
                        raise build_failure(log, "UBmem " + command[1], result.returncode)
            if not library.is_file():
                raise RuntimeError(f"UBmem build did not produce {library}; see {log}")
            (directory / "complete.json").write_text(json.dumps(options, indent=2) + "\n")
    return library


def query(library, name):
    function = getattr(library, "v9_" + name)
    function.argtypes, function.restype = [], ctypes.c_uint32
    return int(function())
