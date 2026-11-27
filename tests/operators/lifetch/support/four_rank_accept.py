# SPDX-License-Identifier: Apache-2.0
"""Verify same-card serial and independent ordered GET and tail-binary evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from export_large_rank_evidence import observation
from four_rank_followup import CELLS, serial_reference
from large_batch_placement import field, placement, same_card
from reporting import atomic_json


def read(path):
    return json.loads(path.read_bytes())


def matched(root, ref):
    return same_card(root, ref) and all(
        field(root, k) and field(root, k) == field(ref, k) for k in ("kvaf", "kvaf_tree")
    )


def cached(root, ref, group, batch, mode):
    base = ref / f"g{group}/bs{batch}/block1"
    _, identity = observation(ref, base, group, batch, "fused")
    if not matched(root, ref):
        raise ValueError("Audit and measured candidate differ in card, host, or runtime")
    if mode == "serial":
        serial = serial_reference(root, identity, group, batch)
        return Path(serial["source"]) if serial else None
    expected = read(base / "fused/packed_result.json")
    for path in sorted(
        root.parent.glob(f"fixed-q4-audit-journal.*/g{group}/bs{batch}/block1/journal/journal.json")
    ):
        folder, archive = path.parent, path.parents[4]
        if not matched(archive, ref) or not (folder / "placement_verified").is_file():
            continue
        rec = read(path)
        if (
            rec.get("correctness") != "PASS"
            or rec.get("identity") != expected["identity"]
            or rec.get("cold_build") != expected["cold_build_fingerprint"]
            or rec.get("chunks") != batch * 8
            or rec.get("get_workers") != 32
            or not rec.get("tail_text")
            or not rec.get("compiler_sha256")
            or placement(folder / "thp.json") != identity[1]
        ):
            continue
        with (folder / "journal.npz").open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != rec["journal_sha256"]:
                raise ValueError(f"Journal checksum mismatch at {folder}")
        return folder
    return None


def compare_counts(a, b):
    if len(a) != 10 or len(b) != 10:
        raise ValueError("Exactly ten measured calls required")

    def item(c):
        # All four Q counts and all four epochs for every request, not only aggregate traffic.
        return json.dumps([c["row_window_prefetch"], c["row_query_final"]], separators=(",", ":"))

    left, right = list(map(item, a)), list(map(item, b))
    return sum(x == y for x, y in zip(left, right)), sorted(left) == sorted(right)


def compare_journals(a, b):
    left, right = read(a / "journal.json"), read(b / "journal.json")
    code = left["tail_text"] == right["tail_text"] and left["compiler_sha256"] == right["compiler_sha256"]
    failures = []
    total = 0
    with (
        np.load(a / "journal.npz", allow_pickle=False) as baseline,
        np.load(b / "journal.npz", allow_pickle=False) as candidate,
    ):
        if set(baseline.files) != set(candidate.files) or len(baseline.files) != left["chunks"]:
            raise ValueError("Malformed per-row four-window/four-Q archive")
        for name in baseline.files:
            total += 1
            x, y = baseline[name], candidate[name]
            if np.array_equal(x, y):
                continue
            k = min(len(x), len(y))
            idx = np.flatnonzero((x[:k] != y[:k]).any(axis=1))
            n = int(idx[0]) if idx.size else k
            failures.append(
                dict(
                    segment=name,
                    ordinal=n,
                    baseline=x[n].tolist() if n < len(x) else None,
                    candidate=y[n].tolist() if n < len(y) else None,
                )
            )
    return dict(
        tail_text_and_compiler_match=code,
        matched_segments=total - len(failures),
        total_segments=total,
        differences=failures,
    )


def report(root, ref):
    result = []
    for g, b in CELLS:
        p = f"g{g}/bs{b}/block1"
        old, identity = observation(ref, ref / p, g, b, "fused")
        new, other = observation(root, root / p, g, b, "fused")
        if other != identity or not matched(ref, root):
            raise ValueError(f"G{g} BS{b} candidate and reference differ in physical inputs or runtime")
        src = serial_reference(root, identity, g, b)
        a, bfile = cached(root, ref, g, b, "journal"), cached(root, root, g, b, "journal")
        if src is None or a is None or bfile is None:
            raise ValueError(f"G{g} BS{b} missing verified serial or journal")
        n, reordered = compare_counts(
            read(ref / p / "fused/packed_result.json")["traffic"],
            read(root / p / "fused/packed_result.json")["traffic"],
        )
        journal = compare_journals(a, bfile)
        passed = n == 10 and journal["tail_text_and_compiler_match"] and not journal["differences"]
        gain = 100 * (1 - new["graph"] / src["graph"])
        record = dict(
            group=g,
            batch=b,
            serial=src,
            previous=old,
            candidate=new,
            graph_gain_vs_serial_pct=gain,
            ten_call_counts_equal=n,
            counts_multiset_equal=reordered,
            journal_check=journal,
            strict_equivalence_pass=passed,
            target20_graph=gain >= 20,
        )
        result.append(record)
        print(
            f"G{g} BS{b} serial/candidate_graph_us={src['graph']:.2f}/{new['graph']:.2f} gain_pct={gain:+.2f} "
            f"serial_LI/tail_us={src['LI']:.2f}/{src['tail']:.2f} candidate_LI/tail_us={new['LI']:.2f}/{new['tail']:.2f} "
            f"saved_vs_previous_graph/LI/tail_us={old['graph'] - new['graph']:+.2f}/{old['LI'] - new['LI']:+.2f}/{old['tail'] - new['tail']:+.2f} "
            f"matched_counts={n}/10 "
            f"ordered_GET_segments={journal['matched_segments']}/{journal['total_segments']} "
            f"tail_text_compiler={'PASS' if journal['tail_text_and_compiler_match'] else 'FAIL'} "
            f"strict={'PASS' if passed else 'FAIL'}"
        )
    path = root / "strict_acceptance.json"
    atomic_json(
        path,
        dict(
            protocol="ten_timed_calls_plus_one_untimed_ordered_GET_journal_per_arm",
            diag_columns=[
                "source_physical",
                "logical_token",
                "registered_page",
                "byte_offset",
                "destination_slot",
            ],
            cells=result,
        ),
    )
    print(
        f"STRICT_REPORT={path} strict_pass_cells={sum(r['strict_equivalence_pass'] for r in result)}/{len(result)} "
        f"graph20_cells={sum(r['target20_graph'] for r in result)}/{len(result)}"
    )
    return all(r["strict_equivalence_pass"] for r in result)


def timed(ref, native):
    for root in sorted(
        ref.parent.glob("fixed-q4-rank-followup.*"), key=lambda p: p.stat().st_mtime_ns, reverse=True
    ):
        if (
            field(root, "native") != native
            or field(root, "followup_reference") != str(ref)
            or not matched(root, ref)
        ):
            continue
        summary = root / "followup_summary.json"
        if not summary.is_file():
            continue
        report = read(summary)
        if report.get("scope") != "all" or len(report.get("candidates", [])) != len(CELLS):
            continue
        if all(
            observation(root, root / f"g{g}/bs{b}/block1", g, b, "fused")[1]
            == observation(ref, ref / f"g{g}/bs{b}/block1", g, b, "fused")[1]
            for g, b in CELLS
        ):
            return root
    return None



def serial_summary(ref):
    matches = []
    missing = []
    for g, b in CELLS:
        _, identity = observation(ref, ref / f"g{g}/bs{b}/block1", g, b, "fused")
        src = serial_reference(ref, identity, g, b)
        if src is None:
            missing.append(f"G{g}/BS{b}")
            continue
        print(f"G{g} BS{b} strong_serial graph/LI/tail_mean_us={src['graph']:.2f}/{src['LI']:.2f}/{src['tail']:.2f} "
              f"blocks={src['blocks']} source={src['source']}")
        matches.append(dict(group=g, batch=b, **src))
    print(f"SERIAL_COVERAGE={len(matches)}/{len(CELLS)} missing={','.join(missing) or 'none'}")
    if missing:
        return False
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("placement").add_argument("path", type=Path)
    sub.add_parser("serial-summary").add_argument("ref", type=Path)
    t = sub.add_parser("timed")
    t.add_argument("ref", type=Path)
    t.add_argument("native")
    for action in ("cached", "check", "report"):
        q = sub.add_parser(action)
        q.add_argument("root", type=Path)
        q.add_argument("ref", type=Path)
        if action != "report":
            q.add_argument("group", type=int)
            q.add_argument("batch", type=int)
            q.add_argument("mode", choices=["serial", "journal"])
    a = p.parse_args()
    if a.action == "serial-summary":
        raise SystemExit(0 if serial_summary(a.ref) else 3)
    if a.action == "timed":
        hit = timed(a.ref, a.native)
        if hit:
            print(hit)
    elif a.action == "placement":
        print(*placement(a.path))
    elif a.action == "report":
        raise SystemExit(0 if report(a.root, a.ref) else 3)
    else:
        hit = cached(a.root, a.ref, a.group, a.batch, a.mode)
        if a.action == "check" and not hit:
            raise ValueError(f"Unverified {a.mode} result at {a.root}")
        if hit:
            print(hit)
