# SPDX-License-Identifier: Apache-2.0
"""Select local four-rank results and export available window evidence."""
import argparse
import gzip
import json
import os
from pathlib import Path

from large_batch_placement import field, placement

BASE = "d0fc5963881bd38fa97eff0f6ef40c757a6e8713"
SDK = "813f1750a50b7e6c92d5f5397f2fbc05befef6da"
TREE = "1de18e8315137d7a0227c3ae4c82afd45b5fe743"


# These readers ran the same window clocks. Their differences are outside the
# device path (host selection, summaries, or the offline CQ report).
PORTABLE_READERS = ("128295e8533270b537b7c64128087f3505800209",
                    "b70ad72e642a0d93e09f6937702525d2b9e3d753",
                    "69f6805757fb03c44bb9f793aa88e25e19098806")
EARLIER_PAIRED = "fc67beee1c357c2d0140dd8b1dfd2601d5ec445a"


def paired_clock(ref, reader, b):
    from b24_window_evidence import clock_hit
    for version in (reader, *PORTABLE_READERS, EARLIER_PAIRED):
        arm = clock_hit(ref, version, b)
        if arm:
            return arm
    return None


def machine():
    def read(path):
        p = Path(path)
        return p.read_text().strip() if p.is_file() else ""
    return os.uname().nodename, read("/etc/machine-id"), read("/proc/sys/kernel/random/boot_id")


def local(root, current):
    host, mid, boot = current
    if field(root, "host") != host or field(root, "physical_npu") != "4":
        return False
    other = field(root, "machine_id")
    if mid and other:
        return mid == other
    return bool(boot and field(root, "boot_id") == boot)


def origin(area, current):
    from b24_window_evidence import CELLS, READER
    from b24_window_report import cached
    options = []
    for root in area.glob("fixed-q4-four-ranks-one-block.*"):
        if not (local(root, current) and field(root, "native") == BASE and field(root, "kvaf") == SDK
                and field(root, "kvaf_tree") == TREE and field(root, "selection_oracle") == "captured_logical_scores"
                and field(root, "candidate_blocks") == "1"):
            continue
        cells = [(g, b, x) for g, b, x in CELLS if
                 (root / f"g{g}/bs{b}/block1/fused/placement_verified").is_file() and
                 (root / f"g{g}/bs{b}/block1/fused/packed_result.json").is_file()]
        if cells:
            hits = sum(bool(cached(READER, root, g, b, f"window-x{x}-early")) for g, b, x in cells)
            options.append((hits, len(cells), root.stat().st_mtime_ns, root))
    return max(options)[-1] if options else None


def portable_clock(area, reader, b, current):
    # The capture manifests and row selectors for these two shapes are pinned by the reader.
    for root in sorted(area.glob("local-q4-window-clocks.*"), key=lambda p:p.stat().st_mtime_ns, reverse=True):
        arm = root / f"g2/bs{b}/block1/clocks"
        if not (local(root,current) and field(root,"native") == BASE and field(root,"kvaf") == SDK
                and field(root,"kvaf_tree") == TREE and (arm/"local_placement_measured").is_file()):
            continue
        try:
            d = json.loads((arm/"single_diagnostic.json").read_bytes())
            ident = d["identity"]
            placement(arm/"thp.json")
            if (d["controller_commit"] in (reader, *PORTABLE_READERS) and d["correctness"] == "PASS" and d["reference_oracle"] == "captured_logical_scores"
                    and d["diagnostic_calls"] == 1 and d["measured_calls"] == 0 and ident["native"] == BASE
                    and ident["physical_npu"] == 4 and ident["host"] == current[0] and ident["group"] == 2
                    and ident["batch"] == b and len(ident["sequence_sha256"]) == 64
                    and d["li_windows"]["checked_rank_queries"] == 4*b):
                return arm
        except (OSError, TypeError, ValueError, KeyError):
            continue
    return None


def write_package(directory, ref, reader):
    from b24_window_evidence import CELLS, READER, lines, recorded, show_clocks, saved_frequency_fields
    from b24_window_report import cached
    from export_large_rank_evidence import observation
    data = dict(host=os.uname().nodename, physical_npu=4, reference=str(ref) if ref else None,
                archived={}, local_diagnostics={})
    if ref:
        for g, b, x in CELLS:
            if not cached(READER, ref, g, b, f"window-x{x}-early"):
                continue
            arm, new, old = recorded(ref, g, b, x)
            slot = data["archived"].setdefault(f"G{g}-BS{b}", {})
            for n, result, folder in ((3, old, ref/f"g{g}/bs{b}/block1/fused"), (x, new, arm)):
                if f"1+{n}" not in slot:
                    slot[f"1+{n}"] = dict(**lines(g, b, n, result), source=str(folder),
                                            frequency=saved_frequency_fields(folder))
    for b in (28, 32):
        index = directory/f"g2-bs{b}.txt"
        if index.is_file():
            arm = Path(index.read_text().strip())
            result = json.loads((arm/"single_diagnostic.json").read_bytes())
            root = arm.parents[3]
            if (not local(root, machine()) or field(root, "native") != BASE
                    or field(root, "kvaf") != SDK or field(root, "kvaf_tree") != TREE
                    or result["controller_commit"] not in (reader, *PORTABLE_READERS) or result["correctness"] != "PASS"
                    or result["identity"]["physical_npu"] != 4 or result["identity"]["native"] != BASE
                    or result["identity"]["batch"] != b or result["identity"]["group"] != 2
                    or not (arm/"local_placement_measured").is_file()
                    or result["measured_calls"] != 0 or result["diagnostic_calls"] != 1):
                raise ValueError(f"Invalid local G2 BS{b} diagnostic {arm}")
            observed = placement(arm/"thp.json")
            item = dict(source=str(arm), placement=observed, single_diagnostic=result)
            if ref and (ref/f"g2/bs{b}/block1/fused/placement_verified").is_file():
                _, ident = observation(ref, ref/f"g2/bs{b}/block1", 2, b, "fused")
                baseline = json.loads((ref/f"g2/bs{b}/block1/fused/packed_result.json").read_bytes())
                if observed == ident[1] and all(baseline["identity"].get(k) == v
                                                for k,v in result["identity"].items()):
                    item["original_four_window_diagnostic"] = {k: baseline[k] for k in (
                        "separate_LI_diagnostic", "separate_score_span_cycles",
                        "separate_sibling_predecessor_cycles", "separate_window_event_u64")}
            data["local_diagnostics"][str(b)] = item
        else:
            arm = paired_clock(ref, reader, b) if ref else None
            if not arm:
                raise ValueError(f"Missing expected G2 BS{b} diagnostic")
            data["local_diagnostics"][str(b)] = show_clocks(arm, ref, b)
    dest = directory/"window-evidence.json.gz"
    with gzip.open(dest.with_suffix(".tmp"), "wt", encoding="utf-8") as fp:
        json.dump(data, fp, separators=(",", ":"), allow_nan=False)
    dest.with_suffix(".tmp").replace(dest)
    print(f"RETURN_FILE={dest} bytes={dest.stat().st_size}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["origin","paired","paired-clock","clock","package"])
    p.add_argument("path", type=Path)
    p.add_argument("extra", nargs="*")
    a = p.parse_args()
    if a.command == "origin":
        print(origin(a.path.resolve(), machine()) or "")
    elif a.command == "paired-clock":
        print(paired_clock(a.path, a.extra[0], int(a.extra[1])) or "")
    elif a.command == "clock":
        print(portable_clock(a.path, a.extra[0], int(a.extra[1]), machine()) or "")
    elif a.command == "paired":
        from b24_window_evidence import READER
        from b24_window_report import cached
        print(cached(READER, a.path, 2, int(a.extra[0]), "window-x2-early") or "")
    else:
        reference = Path(a.extra[0]) if a.extra[0] else None
        write_package(a.path, reference, a.extra[1])
