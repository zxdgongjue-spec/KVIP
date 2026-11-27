# SPDX-License-Identifier: Apache-2.0
"""Collect completed four-rank clocks and compare any matching individual serial blocks (CPU only)."""
import argparse
import io
import json
from pathlib import Path
from statistics import mean
import tarfile
import tempfile

from fixed_q4_packed_archive import IDENTITY_KEYS, PACKED
from fixed_q4_packed_report import REFERENCE, read_arm
from large_batch_placement import field, same_card


GROUPS = (7, 2, 10, 4, 17)
BATCHES = (32, 28, 24, 20)
METADATA = ("native", "host", "physical_npu", "machine_id", "boot_id", "kvaf", "kvaf_tree")


def observation(base, block, group, batch, arm):
    directory = block / arm
    path = directory / "packed_result.json"
    record = json.loads(path.read_bytes())
    times, (identity, placement) = read_arm(block, group, batch, 0, base, arm)
    if (identity["protocol"] != PACKED or identity["warmup"] != 4 or identity["calls"] != 10
            or record["full_KV_Index_checked_calls"] != 10 or len(record["samples"]) != 10):
        raise ValueError(f"Wrong timing or correctness protocol at {directory}")
    key = {k: identity[k] for k in IDENTITY_KEYS}
    return times, (key, placement)


def collect(root, output_dir):
    if field(root, "physical_npu") != "4" or field(root, "selection_oracle") != "captured_logical_scores":
        raise ValueError(f"Expected the candidate-only physical NPU4 archive, got {root}")
    files = set()

    def include(base, names):
        for filename in names:
            path = base / filename
            if path.is_file():
                files.add(path)

    def include_root(base):
        include(base, [v + ".txt" for v in METADATA])

    def include_arm(directory):
        include(directory, ("packed_result.json", "placement_verified", "thp.json"))

    include_root(root)
    include(root, ("selection_oracle.txt", "packed_source.txt", "candidate_blocks.txt",
                   "packed_summary.json", "li_phase_comparison.json"))
    candidates = {}
    for g in GROUPS:
        for b in BATCHES:
            block = root / f"g{g}/bs{b}/block1"
            candidates[g, b] = observation(root, block, g, b, "fused")
            include_arm(block / "fused")
    comparisons = []
    rejected = []
    for base in sorted(root.parent.iterdir()):
        if (base == root or not base.is_dir() or not same_card(base, root)
                or any(not field(base, k) or field(base, k) != field(root, k) for k in ("kvaf", "kvaf_tree"))):
            continue
        for (g, b), (current, candidate_identity) in candidates.items():
            by_block = []
            for path in sorted((base / f"g{g}/bs{b}").glob("block[12]/serial/packed_result.json")):
                try:
                    serial, serial_identity = observation(base, path.parents[1], g, b, "serial")
                    if serial_identity != candidate_identity:
                        raise ValueError("input, runtime or NUMA/UB differs")
                except (OSError, ValueError, TypeError, KeyError) as error:
                    rejected.append(dict(file=str(path), reason=str(error)))
                    continue
                include_root(base)
                include_arm(path.parent)
                by_block.append(dict(block=path.parents[1].name, **serial))
            if not by_block:
                continue
            # Report each archive independently. Copies or repeated runs never earn extra weight.
            canonical = field(base, "native") == REFERENCE
            serial = mean(item["graph"] for item in by_block)
            gain = 100 * (1 - current["graph"] / serial)
            comparisons.append(dict(group=g, batch=b, source=str(base), canonical_serial=canonical,
                                    serial_blocks=by_block, candidate=current,
                                    gain_pct=gain if canonical else None))
            if canonical:
                print(f"G{g} BS{b} MATCHED_SERIAL blocks={len(by_block)} "
                      f"graph_serial/candidate_us={serial:.2f}/{current['graph']:.2f} "
                      f"gain_pct={gain:.2f} source={base.name}")
            else:
                print(f"G{g} BS{b} CHECK_SERIAL_REVISION source={base.name} blocks={len(by_block)}")
    counts = {(r["group"], r["batch"]) for r in comparisons if r["canonical_serial"]}
    manifest = dict(candidate=str(root), compared=comparisons, rejected=rejected,
                    missing_serial_cells=[(g, b) for g, b in candidates if (g, b) not in counts])
    text = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
    output = Path(tempfile.mkdtemp(prefix="b24-npu4-rank-evidence.", dir=output_dir)) / "rank-evidence.tar.gz"
    with tarfile.open(output, "w:gz", compresslevel=3, dereference=True) as tar:
        info = tarfile.TarInfo("comparison.json")
        info.size = len(text)
        tar.addfile(info, io.BytesIO(text))
        for path in sorted(files):
            tar.add(path, arcname=str(path.relative_to(root.parent)), recursive=False)
    print(f"CANDIDATE=20/20 MATCHED_SERIAL={len(counts)}/20")
    print(f"RETURN_FILE={output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    collect(root, root.parent.parent)
