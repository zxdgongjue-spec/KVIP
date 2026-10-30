"""Explicit source-fingerprinted CANN build, outside graph capture."""

import argparse
import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


def native_source():
    here = Path(__file__).resolve().parent
    packaged = here / "_native"
    source = packaged if packaged.is_dir() else here.parents[1] / "operators" / "v9"
    if not (source / "CMakeLists.txt").is_file():
        raise RuntimeError("KVOP device sources are missing; reinstall the complete KVOP wheel")
    return source


@dataclass(frozen=True)
class BuildConfig:
    cann_path: str = "/usr/local/Ascend/cann"
    kvaf_include: str = ""
    build_dir: str = "build/v9"
    soc: str = "Ascend950DT_9574"
    intermediate_fetches: int = 4
    intermediate_profiling: bool = False


def build_library(config, g, aics):
    import fcntl

    g.validate()
    source = native_source()
    if not (Path(config.kvaf_include) / "kvaf_batch_get_device.h").is_file():
        raise ValueError("kvaf_include must contain the matching KVAF device headers")
    key = (g.batch, g.queries, g.candidate_capacity, g.resident_capacity)
    digest = hashlib.sha256()
    files = list(source.glob("*"))
    for subdir in ("base", "reference"):
        files.extend((source / subdir).rglob("*"))
    files.extend(Path(config.kvaf_include).rglob("*.h"))
    for path in sorted(p for p in files if p.is_file() and p.suffix in (".h", ".cpp", ".inc", ".py", ".txt")):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    digest.update(
        json.dumps(
            dict(
                aics=aics,
                cann=str(Path(config.cann_path).resolve()),
                soc=config.soc,
                kvaf_include=str(Path(config.kvaf_include).resolve()),
                windows=config.intermediate_fetches,
                profiling=bool(config.intermediate_profiling),
            ),
            sort_keys=True,
        ).encode()
    )
    for version in Path(config.cann_path).glob("*version*"):
        if version.is_file():
            digest.update(version.read_bytes())
    root = Path(config.build_dir).resolve() / digest.hexdigest()[:16]
    build = root / f"b{key[0]}-q{key[1]}-c{key[2]}-s{key[3]}"
    root.mkdir(parents=True, exist_ok=True)
    library = build / "libv9_bridge.so"
    with (root / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.is_file() or not (build / "complete.json").is_file():
            build.mkdir(exist_ok=True)
            options = dict(
                V9_CONTRACT_MODE="ON",
                RUN_MODE="npu",
                SIM_BATCH=g.batch,
                SIM_QUERIES=g.queries,
                SIM_HISTORY=g.candidate_capacity,
                SIM_RESIDENT=g.resident_capacity,
                AIC_COUNT=aics,
                INTERMEDIATE_FETCHES=config.intermediate_fetches,
                INTERMEDIATE_PROFILING=int(config.intermediate_profiling),
                SOC_VERSION=config.soc,
                ASCEND_CANN_PACKAGE_PATH=config.cann_path,
                KVAF_INCLUDE=config.kvaf_include,
            )
            logger.info(
                "[DSA_V9] compiling %s windows_requested=%d intermediate_profiling=%s",
                build,
                config.intermediate_fetches,
                config.intermediate_profiling,
            )
            with (build / "configure.log").open("w") as log:
                subprocess.run(
                    [
                        "cmake",
                        "-S",
                        str(source),
                        "-B",
                        str(build),
                        *[f"-D{k}={v}" for k, v in options.items()],
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            with (build / "build.log").open("w") as log:
                subprocess.run(
                    ["cmake", "--build", str(build), "--target", "v9_bridge", "-j2"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            (build / "complete.json").write_text(json.dumps(options, indent=2))
    return library


def main():
    from types import SimpleNamespace

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--history", type=int, default=131072)
    p.add_argument("--resident", type=int, default=12288)
    p.add_argument("--aic-count", type=int, required=True)
    p.add_argument("--kvaf-include", required=True)
    p.add_argument("--cann", default="/usr/local/Ascend/cann")
    p.add_argument("--soc", default="Ascend950DT_9574")
    p.add_argument("--build-dir", default="build/v9")
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--profiling", action="store_true")
    a = p.parse_args()
    # Compile-time geometry only; allocation-specific validation occurs at launch.
    g = SimpleNamespace(
        batch=a.batch, queries=4, candidate_capacity=a.history, resident_capacity=a.resident, validate=lambda: None
    )
    config = BuildConfig(a.cann, a.kvaf_include, a.build_dir, a.soc, a.windows, a.profiling)
    print(build_library(config, g, a.aic_count))


if __name__ == "__main__":
    main()
