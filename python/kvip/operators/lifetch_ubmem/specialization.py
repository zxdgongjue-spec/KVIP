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
    parallel_init: bool = False
    balanced_index: bool = False
    skip_inactive: bool = True
    compact_reserved: bool = False
    grouped_copy: bool = False
    tagged_map: bool = False
    final_copy_stages: int = 4
    final_copy_rows: int = 2
    merge_outputs: bool = False
    ordered_pair: bool = False
    cold_only_prefetch: bool = False
    threshold_only: bool = False
    simt_init: bool = False
    single_plan: bool = False
    staged_outputs: bool = False
    rolling_threshold: bool = False
    reserved_ub: bool = False
    cached_buckets: bool = False
    fused_init: bool = False

    def __post_init__(self):
        if any(
            type(getattr(self, name)) is not bool
            for name in (
                "parallel_init",
                "balanced_index",
                "skip_inactive",
                "compact_reserved",
                "grouped_copy",
                "tagged_map",
                "merge_outputs",
                "ordered_pair",
                "cold_only_prefetch",
                "threshold_only",
                "simt_init",
                "single_plan",
                "staged_outputs",
                "rolling_threshold",
                "reserved_ub",
                "cached_buckets",
                "fused_init",
            )
        ):
            raise ValueError("UBmem optimization switches must be booleans")
        if self.fused_init and self.parallel_init:
            raise ValueError("fused_init requires one init worker per request")
        if self.tagged_map and self.parallel_init:
            raise ValueError("Tagged mapping requires one init worker per request")
        if self.simt_init and self.parallel_init:
            raise ValueError("SIMT init requires one init worker per request")
        if self.reserved_ub and (
            self.resident != 16384 or not self.compact_reserved or self.simt_init or self.parallel_init
        ):
            raise ValueError("reserved_ub requires resident=16384, compact_reserved, and scalar single-worker init")
        if self.rolling_threshold and (
            self.resident != 16384 or 2 * self.batch > self.aic_count or self.threshold_only
        ):
            raise ValueError("rolling_threshold requires resident=16384, split-Q2, and threshold_only=False")
        if self.ordered_pair and not (self.grouped_copy and self.token_bytes == 2624):
            raise ValueError("Ordered pair copy requires grouped four-layer copy")
        if (
            type(self.final_copy_stages) is not int
            or type(self.final_copy_rows) is not int
            or (self.final_copy_stages, self.final_copy_rows) not in ((4, 2), (8, 2), (6, 4))
        ):
            raise ValueError("UBmem final copy requires (stages, rows) = (4, 2), (8, 2), or (6, 4)")
        if self.final_copy_stages != 4 and not (self.grouped_copy and self.token_bytes == 2624):
            raise ValueError("Extended final copy requires grouped four-layer copy")
        if self.final_copy_rows == 4 and self.ordered_pair:
            raise ValueError("Four-row final copy requires ordered_pair=False")
        if not 1 <= self.batch <= 16 or not 1 <= self.aic_count <= 32:
            raise ValueError("UBmem LIFetch requires 1..16 requests and 1..32 AICs")
        if self.batch > self.aic_count:
            raise ValueError("UBmem LIFetch requests must fit concurrently resident LI cores")
        if self.history % 256 or self.resident % 256 or not 8192 <= self.resident < self.history <= 210944:
            raise ValueError("UBmem history/resident must align to 256, with 8192 <= resident < history <= 210944")
        if self.resident > 131072:
            raise ValueError("LIFetch hot bitmap supports resident <= 131072")
        if self.heads not in (32, 64) or self.token_bytes not in (656, 2624):
            raise ValueError("UBmem supports 32/64 Indexer heads and 656/2624-byte KV rows")
        if self.grouped_copy and self.token_bytes != 2624:
            raise ValueError("Opt-in paired copy requires the four-layer 2624-byte layout")
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
            UBMEM_PARALLEL_INIT=int(self.parallel_init),
            UBMEM_BALANCED_INDEX=int(self.balanced_index),
            SKIP_INACTIVE=int(self.skip_inactive),
            COMPACT_RESERVED=int(self.compact_reserved),
            UBMEM_GROUPED_COPY=int(self.grouped_copy),
            UBMEM_TAGGED_MAP=int(self.tagged_map),
            UBMEM_FINAL_COPY_STAGES=self.final_copy_stages,
            UBMEM_FINAL_COPY_ROWS=self.final_copy_rows,
            UBMEM_MERGE_OUTPUTS=int(self.merge_outputs),
            UBMEM_ORDERED_PAIR=int(self.ordered_pair),
            COLD_ONLY_PREFETCH=int(self.cold_only_prefetch),
            THRESHOLD_ONLY=int(self.threshold_only),
            SIMT_INIT=int(self.simt_init),
            UBMEM_SINGLE_PLAN=int(self.single_plan),
            STAGED_OUTPUTS=int(self.staged_outputs),
            ROLLING_THRESHOLD=int(self.rolling_threshold),
            RESERVED_UB=int(self.reserved_ub),
            CACHED_BUCKETS=int(self.cached_buckets),
            UBMEM_FUSED_INIT=int(self.fused_init),
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
    shared_selector = source.parent / "lidu/op_kernel/hot_slot_selection.h"
    digest.update(str(shared_selector.relative_to(source.parent)).encode())
    digest.update(shared_selector.read_bytes())
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
