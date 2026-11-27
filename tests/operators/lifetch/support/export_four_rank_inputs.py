# SPDX-License-Identifier: Apache-2.0
"""Export the original NPU4 replay's five complete fixed-length LI donor banks on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from export_large_rank_evidence import observation
from large_batch_input import FrozenRound, expand, input_digest, mapped_source, read_source
from workloads import ReplayPagePlan


GROUPS = (10, 4, 17, 7, 2)
BATCHES = (20, 24, 28, 32)
ORIGIN = "d0fc5963881bd38fa97eff0f6ef40c757a6e8713"
# SHA-256 over all arrays in the two full rows already returned to the workstation.
EXISTING = {
    10: ((0, 13), "0cb29d5a8f539eb741d3bc682491f2e24344da56ddd3455881defbd8fcc58ceb"),
    4: ((0, 12), "4072ed8e36bdb5ed34b81c4e4ab2b4846291b8c765b05e91bd9e771a1daacf41"),
    17: ((2, 8), "30d64aea0c02012ea9d37786c9788eb5654b3bbb5ac36619a6f7e2115490d5a3"),
    7: ((2, 14), "62bd62db3ec9a13f4ee53dc065dd7e2ed57fd53868b6397fe95e660b3a8d48a1"),
    2: ((0, 14), "f400832fa0b870de0917e257468bcc82c487e1e329331422caf8c4e64d664d0b"),
}
DEFAULT_ARCHIVE = Path("/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold/peer-baseline/"
                       "fixed-q4-four-ranks-one-block.MRDDaH")


def physical_arrays(inputs, rows):
    rows = np.asarray(rows, dtype=np.int32)
    layout = inputs.captured_layout[rows]
    if not np.all(np.sort(layout, axis=1) == np.arange(inputs.geometry["history"])):
        raise ValueError("Invalid captured physical-to-logical map")
    yield "rows", rows
    yield "layout", layout
    for dest, src in (("keys", "logical_keys"), ("key_scale", "logical_scales"),
                      ("address_valid", "logical_address_valid")):
        table = getattr(inputs, src)
        yield dest, np.stack([table[r][ids] for r, ids in zip(rows, layout)])
    for dest, src in (("score_keys", "logical_score_keys"), ("score_valid", "logical_score_valid")):
        table = getattr(inputs, src)
        yield dest, np.stack([np.take(table[r], ids, axis=1) for r, ids in zip(rows, layout)])
    for dest, src in (("query", "query"), ("weights", "weights"), ("query_scale", "query_scale"),
                      ("query_ends", "query_ends"), ("tail_bounds", "tail_bounds"),
                      ("hot_valid", "captured_hot_valid"), ("lru", "captured_lru")):
        yield dest, getattr(inputs, src)[rows]


def array_digest(arrays):
    digest = hashlib.sha256()
    for key, arr in sorted(arrays, key=lambda item: item[0]):
        digest.update(json.dumps([key, list(arr.shape), arr.dtype.str]).encode() + b"\0")
        digest.update(memoryview(np.ascontiguousarray(arr)).cast("B"))
    return digest.hexdigest()


def identities(root, group):
    rows = {}
    for b in BATCHES:
        block = root / f"g{group}/bs{b}/block1"
        _, (identity, _) = observation(root, block, group, b, "fused")
        rows[b] = identity
    return rows


def verify(source, refs, expected):
    if len(refs) != 8 or [r.dp_rank for r in refs] != list(range(8)) or len(source.identities) != 16:
        raise ValueError("Expected sixteen requests from eight captured DPs")
    for b in BATCHES:
        inputs, ids = expand(source, b)
        pages = ReplayPagePlan(FrozenRound(inputs, refs))
        if (ids != expected[b]["donor_rows"] or pages.page_count != expected[b]["registered_pages"]
                or input_digest(inputs, pages, ids) != expected[b]["sequence_sha256"]):
            raise ValueError(f"BS{b} input differs from completed four-rank replay")
        del inputs, pages


def find_bank(root, group, expected):
    direct = root / f"g{group}/donors/source.json"
    indexes = [direct, *sorted(root.parent.glob(f"*/g{group}/donors/source.json"))]
    traces = []
    errors = []
    for index in dict.fromkeys(indexes):
        if not index.is_file():
            continue
        try:
            trace = Path(json.loads(index.read_bytes())["trace"])
            traces.append(trace)
            source, refs = mapped_source(index.parent, trace, group, 1)
            verify(source, refs, expected)
            return source, str(index.parent)
        except (KeyError, OSError, ValueError) as error:
            errors.append(f"{index}: {error}")
    work = root.parents[1]
    suffix = f"r{51 if group == 10 else 2}-g{group}"
    for area in (work / "peer-inputs", work / "lifetch-real-suite", work.parent / "lifetch-real-suite",
                 Path("/home/x00965276/KVIP/lifetch-real-suite")):
        traces.extend(sorted(area.glob(f"case*-{suffix}")))
    for trace in dict.fromkeys(traces):
        if not trace.is_dir():
            continue
        try:
            source, refs = read_source(trace, group, 1)
            verify(source, refs, expected)
            return source, str(trace)
        except (KeyError, OSError, ValueError) as error:
            errors.append(f"{trace}: {error}")
    raise ValueError(f"G{group} no matching local bank or original trace: {errors}")


def export(root, group, dest):
    expected = identities(root, group)
    source, bank = find_bank(root, group, expected)
    keep, sha = EXISTING[group]
    same = array_digest(physical_arrays(source, keep)) == sha
    rows = [i for i in range(16) if not same or i not in keep]
    manifest = dict(format=1, archive=str(root), source=bank, native=ORIGIN, group=group,
                    history=source.geometry["history"], rows=rows, already_on_A=list(keep) if same else [],
                    existing_array_sha256=sha if same else None, identities=source.identities,
                    logical_page_keys=source.page_keys, geometry=source.geometry,
                    sequence_sha256={b: expected[b]["sequence_sha256"] for b in BATCHES},
                    donor_rows={b: expected[b]["donor_rows"] for b in BATCHES})
    output = dest / f"g{group}-full-li.npz"
    tmp = output.with_suffix(".tmp")
    try:
        with zipfile.ZipFile(tmp, "x", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
            for key, arr in physical_arrays(source, rows):
                with z.open(key + ".npy", "w", force_zip64=True) as f:
                    np.lib.format.write_array(f, arr, allow_pickle=False)
            with z.open("manifest.npy", "w") as f:
                np.lib.format.write_array(f, np.array(json.dumps(manifest)), allow_pickle=False)
        with zipfile.ZipFile(tmp) as z:
            bad = z.testzip()
            if bad:
                raise ValueError(f"{tmp}: invalid compressed array {bad}")
        with tmp.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        tmp.rename(output)
    finally:
        tmp.unlink(missing_ok=True)
    with (dest / "SHA256SUMS").open("a") as f:
        f.write(f"{digest}  {output.name}\n")
    print(f"RETURN_FILE={output} G{group} rows={len(rows)} bytes={output.stat().st_size} sha256={digest}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    p.add_argument("--groups", nargs="+", type=int, choices=GROUPS, default=GROUPS)
    a = p.parse_args()
    r = a.archive.resolve(strict=True)
    if (len(set(a.groups)) != len(a.groups) or (r / "native.txt").read_text().strip() != ORIGIN
            or (r / "physical_npu.txt").read_text().strip() != "4"
            or (r / "host.txt").read_text().strip() != os.uname().nodename):
        raise SystemExit(f"Unexpected archive, card, host, or repeated groups: {r}")
    destination = Path(tempfile.mkdtemp(prefix="four-rank-full-li.", dir=r.parents[1]))
    print(f"EXPORT_DIR={destination}", flush=True)
    for group in a.groups:
        export(r, group, destination)
    print(f"CHECKSUMS={destination / 'SHA256SUMS'}", flush=True)
