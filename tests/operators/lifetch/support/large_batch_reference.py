# SPDX-License-Identifier: Apache-2.0
"""Verify archived serial and fused results before comparing live measurements."""

import hashlib
import json
import math
import shutil
from statistics import mean, median
import subprocess

import numpy as np

from large_batch_placement import placement

SOURCE_REV = "e783914bc96f8aefde1043a2127cfdbadbc3d777"
G10_BS20_SHA = "80b7b26e7bb7a663025c4c4f99a41c5ac4f58ce50b1a438c79154e240f0a6035"
COMPUTE_BASELINE = "d2d9d482759ca852decf2ef2899b0604a4afcdb7"


FOUR_WINDOW_REV = "d8c302883f5df58c6442dedbf224ff3c7dc03d03"


def check_archive_root(old, new, revision):
    for name in ("host.txt", "machine_id.txt", "physical_npu.txt"):
        value = (old / name).read_text().strip()
        if not value or value != (new / name).read_text().strip():
            raise ValueError(f"Archived comparison differs in {name}")
    if (old / "native.txt").read_text().strip() != revision:
        raise ValueError(f"Expected archived native commit {revision} at {old}")


def compute_reference(directory, identity, output, four_window=None):
    new = output.parents[2]
    check_archive_root(directory.parents[1], new, COMPUTE_BASELINE)
    fused_path = directory / "fused" if four_window is None else four_window
    fused_revision = COMPUTE_BASELINE if four_window is None else FOUR_WINDOW_REV
    if four_window is not None:
        check_archive_root(four_window.parents[2], new, fused_revision)
    base = dict(identity, native=COMPUTE_BASELINE)
    for arm, workers in (("serial", 2), ("fused", 32)):
        path = directory / arm if arm == "serial" else fused_path
        record = json.loads((path / "ARM_RESULT.json").read_text())
        proof = json.loads((path / "path/path_result.json").read_text())
        if (
            record["arm"] != arm
            or proof["path"] != arm
            or proof["transport"] != "urma"
            or proof["correctness"] != "PASS"
            or proof["failures"]
            or record["get_workers"] != workers
            or proof["runtime_workers"] != workers
            or record["checked_queries"] != identity["calls"] * identity["batch"] * 4
            or record["checked_rows"] != identity["calls"] * identity["batch"]
            or len(record["samples"]) != identity["calls"]
        ):
            raise ValueError(f"Incomplete {arm} archive at {path}")
        keys = ("graph", "LI", "tail", "LI1", "LI23") if arm == "fused" else ("graph", "LI", "tail")
        for key in keys:
            v = [s[key] for s in record["samples"]]
            if any(not math.isfinite(t) or t <= 0 for t in v) or [mean(v), median(v)] != record["mean_p50_us"][key]:
                raise ValueError(f"Archived {arm} {key} timing disagrees with samples")
        if arm == "fused" and record["identity"] != dict(identity, native=fused_revision):
            raise ValueError("Previous fused input, runtime or protocol differs")
    if len(record["traffic"]) != identity["calls"]:
        raise ValueError("Incomplete previous fused traffic measurements")
    return base, record


def serial_identity(identity, repo):
    # Only permit the measured predecessor of this specific retry. Ordinary
    # serial/fused comparisons elsewhere continue to require identical commits.
    parent = subprocess.check_output(["git", "-C", repo, "rev-parse", identity["native"] + "^"], text=True).strip()
    if (
        parent != SOURCE_REV
        or (identity["group"], identity["batch"], identity["physical_npu"]) != (10, 20, 4)
        or identity["sequence_sha256"] != G10_BS20_SHA
        or identity["calls"] != 10
        or identity["warmup"] != 4
    ):
        raise ValueError("Serial reference applies only to the G10 BS20 NPU4 provider-budget retry")
    return dict(identity, native=SOURCE_REV)


def stage_serial(root, destination, identity, repo):
    old = root.resolve(strict=True)
    new = destination.parents[2]
    expected = serial_identity(identity, repo)
    for name in ("host.txt", "physical_npu.txt", "machine_id.txt"):
        if not (new / name).is_file() or not (old / name).is_file():
            raise ValueError(f"Missing {name} for same-host serial reuse")
        actual = (new / name).read_text().strip()
        if not actual or (old / name).read_text().strip() != actual:
            raise ValueError(f"Serial reference differs in {name}")
    if (old / "native.txt").read_text().strip() != SOURCE_REV:
        raise ValueError("Unexpected archived KVIP revision")
    src = old / "g10/bs20/serial"
    node, adapter = placement(src / "thp.json")
    raw = (src / "ARM_RESULT.json").read_bytes()
    result = json.loads(raw)
    proof = json.loads((src / "path/path_result.json").read_bytes())
    if (
        result["identity"] != expected
        or result["arm"] != "serial"
        or result["get_workers"] != 2
        or result["checked_queries"] != 800
        or result["checked_rows"] != 200
        or result["full_KV_Index_checked_calls"] != 1
        or result["full_KV_Index_checked_rows"] != 20
        or proof["correctness"] != "PASS"
        or proof["failures"]
        or proof["path"] != "serial"
        or proof["runtime_workers"] != 2
        or proof["transport"] != "urma"
        or proof["geometry"]["batch"] != 20
        or len(result["samples"]) != 10
    ):
        raise ValueError("Missing or incompatible strong-serial timing, correctness or runtime metadata")
    for key in ("graph", "LI", "tail"):
        v = [s[key] for s in result["samples"]]
        if any(not math.isfinite(x) or x <= 0 for x in v) or [mean(v), median(v)] != result["mean_p50_us"][key]:
            raise ValueError(f"Archived {key} timing does not match its ten raw samples")
    ids = src / "selected.npy"
    selected = np.load(ids, mmap_mode="r", allow_pickle=False)
    if selected.shape != (10, 20, 4, 2048) or selected.dtype != np.int32:
        raise ValueError("Incomplete archived serial TopK")
    if hashlib.sha256(ids.read_bytes()).hexdigest() != result["selection_sha256"]:
        raise ValueError("Archived serial TopK checksum mismatch")
    # Use private copies so the new result can be moved independently and
    # cleaning up successful worker logs cannot remove anything from the source.
    for filename in ("selected.npy", "thp.json", "ARM_RESULT.json"):
        shutil.copyfile(src / filename, destination / filename)
    (destination / "path").mkdir()
    shutil.copyfile(src / "path/path_result.json", destination / "path/path_result.json")
    reference = dict(
        source=str(src),
        native=SOURCE_REV,
        compared_with=identity["native"],
        result_sha256=hashlib.sha256(raw).hexdigest(),
        node=node,
        adapter=adapter,
    )
    (destination / "serial_reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    print(
        "CHECK G10 BS20 serial TopK_score=PASS queries=800/800 Index_slots=PASS rows=200/200 "
        f"full_KV_Index=PASS full_KV_rows=20/20 reused_from={src}"
    )
    print(
        "TIMING G10 BS20 serial",
        *(
            f"{key}_mean/P50_us={a:.2f}/{b:.2f}"
            for key in ("graph", "LI", "tail")
            for a, b in (result["mean_p50_us"][key],)
        ),
    )


def reference_identity(directory, identity, repo):
    marker = directory / "serial_reference.json"
    if not marker.is_file():
        return identity, None
    reference = json.loads(marker.read_text())
    expected = serial_identity(identity, repo)
    if (
        reference["compared_with"] != identity["native"]
        or reference["native"] != SOURCE_REV
        or hashlib.sha256((directory / "ARM_RESULT.json").read_bytes()).hexdigest() != reference["result_sha256"]
    ):
        raise ValueError("Staged serial archive does not match the verified result")
    return expected, reference
