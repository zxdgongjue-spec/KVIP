#!/usr/bin/env python3
"""Compile/run actual UBmem ring and ABI host models; no NPU validation claim."""

from pathlib import Path
import os
import subprocess
import tempfile

root = Path(__file__).resolve().parents[5] / "csrc/operators/lightning_indexer_with_fetch_ubmem"
tests = Path(__file__).resolve().parent
compiler = os.environ.get("CXX", "c++")
common = {
    "SIM_HISTORY": 33024,
    "SIM_RESIDENT": 8192,
    "INDEX_HEADS": 64,
    "SIM_Q": 4,
    "V9_AIC_COUNT": 32,
    "V9_SPLIT_Q": 1,
    "V9_INTERMEDIATE_FETCHES": 8,
    "V9_FINAL_AIVS": 0,
    "V9_SERVICE_AICS": 0,
    "V9_FINAL_REPLICATED": 1,
    "V9_DEDICATED_SERVICE": 0,
    "V9_UBPIPE_DEPTH": 8,
    "V9_UBPIPE_ROWS": 4,
    "V9_WINDOW_UBPIPE_DEPTH": 8,
    "V9_USER_UB_BYTES": 223232,
}
with tempfile.TemporaryDirectory(prefix="kvip-ubmem-model-") as directory:
    for batch, resident, width in (
        (1, 8192, 656),
        (1, 8192, 2624),
        (2, 8192, 656),
        (2, 8192, 2624),
        (1, 2048, 2624),
        (32, 2048, 656),
    ):
        definitions = common | {
            "SIM_BATCH": batch,
            "SIM_RESIDENT": resident,
            "TOKEN_BYTES": width,
            "V9_SPLIT_Q": int(2 * batch <= common["V9_AIC_COUNT"]),
        }
        options = [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            *[f"-D{k}={v}" for k, v in definitions.items()],
            f"-I{root / 'include'}",
            f"-I{root / 'op_kernel/copy'}",
        ]
        for name, sources in (
            ("copy", [tests / "copy_model.cpp"]),
            ("abi", [tests / "abi_test.cpp", root / "op_host/launch.cpp"]),
        ):
            binary = Path(directory) / f"{name}-{batch}-{width}"
            subprocess.run([*options, *map(str, sources), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)
print("Static ABI and real copy-header models passed B1/B2 and B32; 2048/8192 hot slots, 656/2624-byte rows")
