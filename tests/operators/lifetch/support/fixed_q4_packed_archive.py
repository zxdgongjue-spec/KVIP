# SPDX-License-Identifier: Apache-2.0
"""Reuse verified native TopK IDs for a fresh serial/fused ten-call graph comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

NATIVE = "af52a02b97daddadd5d93b9d100f2ff1b03b499d"
SINGLE = "one_restored_graph_per_device_event_with_profiler"
PACKED = "ten_independent_states_one_graph_one_event_v1"
IDENTITY_KEYS = (
    "group", "batch", "sequence_sha256", "physical_npu", "host", "calls", "warmup", "hardware",
    "torch", "torch_npu", "cann", "kvaf_get_kernel_sha256", "physical_history", "token_bytes",
    "registered_pages", "query_end_minmax", "donor_rows",
)


def read(path):
    return json.loads(path.read_text())


def require(ok, error):
    if not ok:
        raise ValueError(str(error))


def same_production(repo, archived, current, *, candidate=False):
    # This measurement only changes the harness. Check the deployed C++, Python,
    # KVAF integration patches and build scripts, including source changes on a branch.
    paths = ["csrc", "python", "patches", "scripts", "pyproject.toml"]
    if candidate:
        paths += [":(exclude)csrc/operators/lightning_indexer_with_fetch/**", ":(exclude)python/kvip/operators/lifetch/**",
                  ":(exclude)csrc/operators/common/lifetch/sequence_schedule.h"]
    completed = subprocess.run(
        ["git", "-C", str(repo), "diff", "--quiet", archived, current, "--",
         *paths],
        check=False,
    )
    require(completed.returncode == 0, f"Production sources/build differ between {archived} and {current}")


def oracle_dir(archive, group, batch, cohort):
    pair = archive / f"g{group}/bs{batch}"
    n = max(16 // batch, 1)
    require(0 <= cohort < n, "Invalid donor cohort")
    if n > 1:
        pair = pair / f"cohort{cohort}"
    arm = read(pair / "fused/ARM_RESULT.json")
    require(arm["arm"] == "fused" and arm["identity"]["group"] == group and arm["identity"]["batch"] == batch,
            f"Unexpected archived candidate at {pair}")
    return Path(arm.get("comparison_archive", pair)) / "serial"


def load_archived_oracle(directory, identity, repo, *, candidate=False):
    """Only return logical token IDs. Single-call event timings never leave this function."""
    directory = Path(directory)
    record = read(directory / "ARM_RESULT.json")
    old = record["identity"]
    require(identity["protocol"] == PACKED and old["protocol"] == SINGLE, "Unexpected TopK oracle protocol")
    require(identity["calls"] == 10 and identity["warmup"] == 4, "Unexpected replay size")
    mismatched = [key for key in IDENTITY_KEYS if old[key] != identity[key]]
    require(not mismatched, f"TopK archive differs for {mismatched} at {directory}")
    # A compute-only or harness-only change may precede the currently measured operator.
    # A previous single-event serial result is usable as an oracle only if its native path is unchanged.
    old_rev = old["native"]
    if old_rev == NATIVE:
        same_production(repo, NATIVE, identity["native"], candidate=candidate)
    else:
        # The complete old fixed matrix could refer to an earlier large-BS archive. Its independent
        # strong serial uses the stock native LI, LIDU prepare and KVAF GET with two workers.
        subprocess.run(["git", "-C", str(repo), "diff", "--exit-code", "--quiet", old_rev, NATIVE,
                        "--", "csrc/operators/lidu", "python/kvip/operators/lidu",
                        "scripts/build.py", "patches"], check=True)
        same_production(repo, NATIVE, identity["native"], candidate=candidate)
    proof = read(directory / "path/path_result.json")
    batch = identity["batch"]
    require(record["arm"] == "serial" and record["get_workers"] == 2
            and record["checked_queries"] == 40 * batch and record["checked_rows"] == 10 * batch
            and record["full_KV_Index_checked_calls"] == 1 and record["full_KV_Index_checked_rows"] == batch
            and proof["path"] == "serial" and proof["transport"] == "urma" and proof["runtime_workers"] == 2
            and proof["correctness"] == "PASS" and not proof["failures"] and proof["geometry"]["batch"] == batch,
            f"Incomplete strong serial TopK proof at {directory}")
    path = directory / "selected.npy"
    # Maximum at BS32 is 10 MiB; compute the checksum on the saved .npy stream, including its header.
    require(hashlib.sha256(path.read_bytes()).hexdigest() == record["selection_sha256"],
            f"Strong serial selection checksum mismatch at {path}")
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    require(data.shape == (10, batch, 4, 2048) and data.dtype == np.int32, f"Invalid strong serial IDs in {path}")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    oracle = parser.add_subparsers(dest="action", required=True).add_parser("oracle-dir")
    oracle.add_argument("archive", type=Path)
    oracle.add_argument("group", type=int)
    oracle.add_argument("batch", type=int, choices=(1, 2, 4, 8, 16, 20, 24, 28, 32))
    oracle.add_argument("cohort", type=int)
    args = parser.parse_args()
    print(oracle_dir(args.archive, args.group, args.batch, args.cohort))
