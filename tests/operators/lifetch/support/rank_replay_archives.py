# SPDX-License-Identifier: Apache-2.0
"""Find usable comparisons on the running host. Timings from another card never qualify."""
import argparse
import json
from pathlib import Path

from large_batch_placement import field, same_card

CONTROL_REV = "10442f1ffa314606c23f54b73e49ecfc372f1f9a"
SOURCE_REV = "af52a02b97daddadd5d93b9d100f2ff1b03b499d"
CARRY_REV = "983734aa8a94310f8090055d916d9a49ab9b4645"
TWO_COLD_REV = "4792819f8b4ed30d8eade0c400de57e2b0b5cbbc"
GROUPS = (7, 2, 10, 4, 17)
BATCHES = (32, 28, 24, 20)
PROTOCOL = "ten_independent_calls_one_device_event"


def newest(area, pattern, revision, identity):
    for path in sorted(area.glob(pattern), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        if path.is_dir() and field(path, "native") == revision and same_card(path, identity):
            yield path.resolve()


def has_cells(root, batches, blocks):
    try:
        summary = json.loads((root / "packed_summary.json").read_text())
        rows = {(row["group"], row["batch"]) for row in summary["cells"]}
        wanted = {(group, batch) for group in GROUPS for batch in batches}
        return (summary.get("protocol") == PROTOCOL and summary.get("blocks") == blocks
                and wanted == rows and len(summary["cells"]) == len(wanted))
    except (OSError, ValueError, TypeError, KeyError):
        return False


def input_archive(control):
    source = field(control, "packed_source")
    if not source:
        return None
    root = Path(source)
    if not root.is_absolute() or field(root, "native") != SOURCE_REV or not same_card(root, control):
        return None
    if any(not field(control, key) or field(root, key) != field(control, key) for key in ("kvaf", "kvaf_tree")):
        return None
    # A summary can outlive its input archive or its per-case strong-serial oracle.
    for batch in BATCHES:
        for group in GROUPS:
            pair = root / f"g{group}/bs{batch}"
            try:
                arm = json.loads((pair / "fused/ARM_RESULT.json").read_text())
                directory = Path(arm.get("comparison_archive", pair)) / "serial"
                for name in ("selected.npy", "ARM_RESULT.json", "path/path_result.json", "thp.json"):
                    if not (directory / name).is_file():
                        return None
                for block in (1, 2):
                    for mode in ("serial", "fused"):
                        measurement = control / f"g{group}/bs{batch}/block{block}/{mode}"
                        if not all((measurement / item).is_file() for item in
                                   ("placement_verified", "thp.json", "packed_result.json")):
                            return None
            except (OSError, TypeError, ValueError):
                return None
    return root


def discover(area, identity):
    for root in newest(area, "fixed-q4-packed.*", CONTROL_REV, identity):
        if has_cells(root, (1, 2, 4, 8, 16, *BATCHES), 2):
            source = input_archive(root)
            if source:
                return root, source
    return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("area", type=Path)
    parser.add_argument("identity", type=Path)
    args = parser.parse_args()
    control, source = discover(args.area, args.identity)
    carry = next((p for p in newest(args.area, "fixed-q4-four-window-carry*.*", CARRY_REV, args.identity)
                  if (p / "packed_summary.json").is_file() and field(p, "candidate_blocks") in ("1", "2")
                  and all((p / f"g{g}/bs{b}/block1/fused/packed_result.json").is_file()
                          for b in BATCHES for g in GROUPS)), None)
    for path in (control, source, carry):
        print(path if path else "-")
