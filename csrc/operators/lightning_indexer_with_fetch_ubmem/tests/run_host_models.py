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
        for width, grouped, stages, ordered in (
            (656, 0, 4, 0),
            (2624, 0, 4, 0),
            (2624, 1, 4, 0),
            (2624, 1, 8, 0),
            (2624, 1, 4, 1),
            (2624, 1, 8, 1),
        ):
            definitions = common | {
                "SIM_BATCH": batch,
                "TOKEN_BYTES": width,
                "V9_UBMEM_GROUPED_COPY": grouped,
                "V9_UBMEM_FINAL_COPY_STAGES": stages,
                "V9_UBMEM_ORDERED_PAIR": ordered,
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
                ("copy", [root / "tests/copy_model.cpp"]),
                ("abi", [root / "tests/abi_test.cpp", root / "op_host/launch.cpp"]),
            ):
                binary = Path(directory) / f"{name}-{batch}-{width}-{grouped}-{stages}-{ordered}"
                subprocess.run([*options, *map(str, sources), "-o", str(binary)], check=True)
                subprocess.run([str(binary)], check=True)
    # Parallel init uses all 64 AIVs; tagged mapping requires the serial init
    # owner. Both keep the same complete-plan/output/apply sequence and ABI.
    for parallel, tagged, merged, single_plan in ((1, 0, 0, 0), (0, 1, 0, 0), (0, 1, 1, 0), (0, 1, 0, 1)):
        definitions = common | {
            "SIM_BATCH": 16,
            "TOKEN_BYTES": 2624,
            "V9_UBMEM_PARALLEL_INIT": parallel,
            "V9_UBMEM_TAGGED_MAP": tagged,
            "V9_UBMEM_MERGE_OUTPUTS": merged,
            "V9_UBMEM_SINGLE_PLAN": single_plan,
            "V9_UBMEM_BALANCED_INDEX": 1,
            "V9_UBMEM_GROUPED_COPY": 1,
            "V9_UBMEM_FINAL_COPY_STAGES": 8,
        }
        binary = Path(directory) / f"abi-optimized-b16-{parallel}-{tagged}-{merged}-{single_plan}"
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
print(
    "Init and ABI/copy models passed B1/B2 × 656/2624, including paired final 4/8-stage and ordered-pair DMA/event checks"
)
