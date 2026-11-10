#!/usr/bin/env python3
"""Compile/run actual UBmem ring and ABI host models; no NPU validation claim."""

from pathlib import Path
import os
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
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
    init_binary = Path(directory) / "init_partition_test"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            f"-I{root / 'include'}",
            str(root / "tests/init_partition_test.cpp"),
            "-o",
            str(init_binary),
        ],
        check=True,
    )
    subprocess.run([str(init_binary)], check=True)
    for batch in (1, 2):
        for width in (656, 2624):
            definitions = common | {"SIM_BATCH": batch, "TOKEN_BYTES": width}
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
                ("copy", [root / "tests/copy_model.cpp"]),
                ("abi", [root / "tests/abi_test.cpp", root / "op_host/launch.cpp"]),
            ):
                binary = Path(directory) / f"{name}-{batch}-{width}"
                subprocess.run([*options, *map(str, sources), "-o", str(binary)], check=True)
                subprocess.run([str(binary)], check=True)
    # The opted-in B16 launch uses all 64 AIVs; verify the exported stage
    # wrappers still issue the same plan/output/apply calls and block counts.
    definitions = common | {
        "SIM_BATCH": 16,
        "TOKEN_BYTES": 2624,
        "V9_UBMEM_PARALLEL_INIT": 1,
        "V9_UBMEM_BALANCED_INDEX": 1,
    }
    binary = Path(directory) / "abi-optimized-b16"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            *[f"-D{k}={v}" for k, v in definitions.items()],
            f"-I{root / 'include'}",
            str(root / "tests/abi_test.cpp"),
            str(root / "op_host/launch.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)
print("Init partition model and ABI/copy-header models passed all B1/B2 × 656/2624 specializations")
