# SPDX-License-Identifier: Apache-2.0
"""Export three complete LI input rows selected from the B24 four-window critical paths."""
import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from math import prod
from pathlib import Path

import numpy as np

from export_four_rank_inputs import BATCHES, DEFAULT_ARCHIVE, ORIGIN, find_bank, identities, physical_arrays
from export_large_rank_evidence import observation
from four_rank_followup import CELLS, serial_reference

# Largest-batch low-miss row, then rows covering three and two observed last finishers.
PRIORITY = ((2, 10), (17, 12), (7, 12))
LIMIT = 48_000_000
FIELDS = ("rows", "layout", "keys", "key_scale", "address_valid", "score_keys", "score_valid",
          "query", "weights", "query_scale", "query_ends", "tail_bounds", "hot_valid", "lru")


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_row(member, index, count):
    version = np.lib.format.read_magic(member)
    read_header = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0}[version]
    shape, fortran, dtype = read_header(member)
    if fortran or not shape or shape[0] != count or dtype.hasobject or not 0 <= index < count:
        raise ValueError(f"Invalid donor axis or dtype: {shape}, {dtype}")
    size = prod(shape[1:]) * dtype.itemsize
    member.seek(index * size, 1)
    data = member.read(size)
    if len(data) != size:
        raise ValueError("Truncated donor slice")
    return np.frombuffer(data, dtype).reshape((1, *shape[1:]))


def archive_digests(expected):
    return {str(batch): row["sequence_sha256"] for batch, row in expected.items()}


def cache_for_row(work, group, donor, expected):
    for path in sorted(work.glob(f"four-rank-full-li.*/g{group}-full-li.npz"),
                       key=lambda p: p.stat().st_mtime_ns, reverse=True):
        receipt = path.parent / "SHA256SUMS"
        if not receipt.is_file():
            continue
        with np.load(path, allow_pickle=False) as src:
            manifest = json.loads(str(src["manifest"]))
            rows = src["rows"].tolist()
        if (manifest["group"] != group or manifest["native"] != ORIGIN or manifest["rows"] != rows
                or manifest["sequence_sha256"] != archive_digests(expected)
                or manifest["donor_rows"] != {str(b): expected[b]["donor_rows"] for b in BATCHES}
                or rows.count(donor) != 1):
            continue
        if f"{sha256(path)}  {path.name}" not in receipt.read_text().splitlines():
            raise ValueError(f"SHA256SUMS mismatch for {path}")
        return path, rows.index(donor), len(rows)
    return None


def extract(root, group, donor, dest):
    expected = identities(root, group)
    cached = cache_for_row(root.parents[1], group, donor, expected)
    meta = dict(group=group, rows=[donor], native=ORIGIN, history=expected[32]["physical_history"],
                sequence_sha256=archive_digests(expected),
                donor_rows={str(b): expected[b]["donor_rows"] for b in BATCHES})
    # Each inner file is a standalone NPZ with full FP8 operands and all reference arrays.
    with zipfile.ZipFile(dest, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        def put(name, arr):
            with target.open(name + ".npy", "w", force_zip64=True) as handle:
                np.lib.format.write_array(handle, arr, allow_pickle=False)
        if cached is not None:
            path, index, count = cached
            meta["source"] = str(path)
            with zipfile.ZipFile(path) as src:
                for key in FIELDS:
                    with src.open(key + ".npy") as member:
                        put(key, read_row(member, index, count))
        else:
            source, bank = find_bank(root, group, expected)
            meta["source"] = bank
            for key, arr in physical_arrays(source, [donor]):
                put(key, arr)
        put("manifest", np.array(json.dumps(meta)))
    with zipfile.ZipFile(dest) as check:
        if check.testzip() is not None:
            raise ValueError(f"Corrupt export {dest}")
    with np.load(dest, allow_pickle=False) as check:
        if check["rows"].tolist() != [donor] or set(check.files) != {*FIELDS, "manifest"}:
            raise ValueError(f"Incorrect contents for G{group} donor {donor}")
    return dict(group=group, donor=donor, name=dest.name, sha256=sha256(dest), bytes=dest.stat().st_size)


def serial_results(root):
    out = []
    for group, batch in CELLS:
        _, identity = observation(root, root / f"g{group}/bs{batch}/block1", group, batch, "fused")
        out.append(dict(group=group, batch=batch, serial=serial_reference(root, identity, group, batch)))
    return out


def assemble(destination, serial, produce, priority=PRIORITY, limit=LIMIT):
    target = destination / "li-critical-rows.zip"
    with tempfile.TemporaryDirectory(dir=destination) as tmp:
        stage = Path(tmp)
        accepted, omitted = [], []
        package = stage / "package.zip"

        def write(rows, skipped):
            manifest = dict(native=ORIGIN, maximum_bytes=limit, rows=rows, omitted=skipped, serial=serial)
            with zipfile.ZipFile(package, "w", zipfile.ZIP_STORED) as archive:
                archive.writestr("summary.json", json.dumps(manifest, indent=2).encode())
                for row in rows:
                    archive.write(stage / row["name"], row["name"])

        for group, donor in priority:
            path = stage / f"g{group}-donor{donor}.npz"
            row = produce(group, donor, path)
            write([*accepted, row], omitted)
            # Reserve headroom for the small omission records in the final manifest.
            if package.stat().st_size <= limit - 1024:
                accepted.append(row)
            else:
                omitted.append(dict(group=group, donor=donor, reason="size_limit"))
        write(accepted, omitted)
        if not accepted or package.stat().st_size > limit:
            raise ValueError("Unable to publish a useful bundle within the size limit")
        with zipfile.ZipFile(package) as archive:
            if archive.testzip() is not None:
                raise ValueError("Bundle verification failed")
        package.rename(target)
    return target, accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    root = parser.parse_args().archive.resolve(strict=True)
    if ((root / "physical_npu.txt").read_text().strip() != "4" or (root / "native.txt").read_text().strip() != ORIGIN
            or (root / "host.txt").read_text().strip() != os.uname().nodename):
        raise SystemExit("Expected the B24 four-window archive from physical NPU4")
    serial = serial_results(root)
    dest = Path(tempfile.mkdtemp(prefix="li-critical-rows.", dir=root.parents[1]))
    target, rows = assemble(dest, serial, lambda g, r, p: extract(root, g, r, p))
    for row in rows:
        print(f"G{row['group']} original_donor={row['donor']} bytes={row['bytes']}")
    print(f"ARCHIVED_SERIAL={sum(r['serial'] is not None for r in serial)}/{len(serial)}")
    print(f"RETURN_FILE={target} bytes={target.stat().st_size} sha256={sha256(target)}")


if __name__ == "__main__":
    main()
