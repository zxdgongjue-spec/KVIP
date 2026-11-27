# SPDX-License-Identifier: Apache-2.0
"""Verify same-card serial and independent ordered GET and tail-binary evidence."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from export_large_rank_evidence import observation
from four_rank_controlled import GUARD
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
    if mode == "frozen":
        origin = Path(os.environ["PEER_AUDIT_ORIGIN"])
        pinned = cached(root, origin, group, batch, "journal")
        if not pinned:
            raise ValueError("The untimed original GET journal has not been collected")
        reference = read(pinned / "journal.json")
        for path in sorted(root.parent.glob(
            f"fixed-q4-audit-frozen.*/g{group}/bs{batch}/block1/frozen/frozen.json"
        )):
            d, trial = path.parent, path.parents[4]
            if (not (d / "placement_verified").is_file() or not matched(trial, ref)
                    or field(trial, "native") != expected["identity"]["native"]):
                continue
            a = read(path)
            if (a.get("identity") == expected["identity"] and a.get("correctness") == "PASS"
                    and a.get("arm") == "frozen" and a.get("get_workers") == 32
                    and a.get("guard_sha256") == hashlib.sha256(GUARD.read_bytes()).hexdigest()
                    and a.get("frozen_journal_sha256") == reference["journal_sha256"]
                    and a.get("tail_text") == reference["tail_text"]
                    and a.get("compiler_sha256") == reference["compiler_sha256"]
                    and a.get("calls") == len(a.get("timed_journal_checks", [])) == 10
                    and a.get("full_KV_Index_checked_calls") == 10
                    and a.get("expected_window_sha256") and a.get("expected_final_sha256")
                    and placement(d / "thp.json") == identity[1]):
                return d
        return None
    # A team-schedule sample also captures the ordinary terminal journal in the same eager call.
    regular = f"fixed-q4-audit-journal.*/g{group}/bs{batch}/block1/journal/journal.json"
    scheduled = f"fixed-q4-audit-team-schedule.*/g{group}/bs{batch}/block1/team-schedule/journal.json"
    local_journals = [
        f"{prefix}.*/g{group}/bs{batch}/block1/stagger-journal/journal.json"
        for prefix in ("fixed-q4-audit-stagger-pair", "fixed-q4-audit-first-receipt-pair")
    ]
    for path in (p for pattern in (regular, scheduled, *local_journals) for p in sorted(root.parent.glob(pattern))):
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
            f"one_untimed_ordered_GET_segments={journal['matched_segments']}/{journal['total_segments']} "
            f"tail_text_compiler={'PASS' if journal['tail_text_and_compiler_match'] else 'FAIL'} "
            f"natural_diagnostic={'PASS' if passed else 'FAIL'}"
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
        f"natural_raw_speed_graph20_cells={sum(r['target20_graph'] for r in result)}/{len(result)}"
    )
    return all(r["strict_equivalence_pass"] for r in result)


def frozen_report(root, ref):
    result = []
    for g, b in CELLS:
        old = cached(root, ref, g, b, "frozen")
        new = cached(root, root, g, b, "frozen")
        if old is None or new is None:
            raise ValueError(f"Missing a controlled arm for G{g} BS{b}")
        left, right = (read(d / "frozen.json") for d in (old, new))
        if left["frozen_journal_sha256"] != right["frozen_journal_sha256"]:
            raise ValueError("Two arms used different original GET journals")
        if any(left[k] != right[k] for k in ("expected_window_sha256", "expected_final_sha256",
                                              "tail_text", "compiler_sha256", "guard_sha256")):
            raise ValueError("Controlled arms used different list, guard, compiler or terminal")
        def verified(a):
            return sum(c["ordered_match"] and c["actual_window_sha256"] == a["expected_window_sha256"]
                       and c["actual_final_sha256"] == a["expected_final_sha256"] for c in a["timed_journal_checks"])
        checked = [verified(a) for a in (left, right)]
        passed = checked == [10, 10]
        delta = None
        if passed:
            delta = {s: left["mean_us"][s] - right["mean_us"][s] for s in ("LI", "tail")}
            delta["graph"] = left["graph_mean_us"] - right["graph_mean_us"]
        rec = dict(group=g, batch=b, original=str(old), candidate=str(new), verified_graph_calls=checked,
                   guarded_pass=passed, controlled_savings_us=delta)
        result.append(rec)
        gains = "/".join(f"{delta[k]:+.2f}" for k in ("graph", "LI", "tail")) if passed else "UNAVAILABLE"
        print(f"G{g} BS{b} CONTROLLED old/new_ordered_calls={checked[0]}/10,{checked[1]}/10 "
              f"graph/LI/tail_saved_mean_us={gains}")
    path = root / "controlled_acceptance.json"
    atomic_json(path, dict(scope="controlled only; both arms include identical guard and terminal journaling",
                           cells=result))
    print(f"CONTROLLED_REPORT={path} controlled_pass_cells={sum(c['guarded_pass'] for c in result)}/{len(result)}")
    return all(c["guarded_pass"] for c in result)


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



def completed_natural_cells(root, ref, native):
    """Authenticate every completed cell of an interrupted natural candidate against the pinned archive."""
    if field(root, "native") != native or field(root, "followup_reference") != str(ref) or not matched(root, ref):
        raise ValueError(f"Wrong native, same-card comparison archive, or runtime at {root}")
    allowed = {root / f"g{g}/bs{b}/block1/fused/placement_verified": (g,b) for g,b in CELLS}
    markers = list(root.glob("g*/bs*/block*/fused/placement_verified"))
    if any(m not in allowed or m.resolve() != root.resolve() / m.relative_to(root) for m in markers):
        raise ValueError(f"Unexpected completed state or batch at {root}")
    for m in markers:
        g,b = allowed[m]
        relative = m.parent.parent.relative_to(root)
        _, old = observation(ref, ref / relative, g, b, "fused")
        _, fresh = observation(root, root / relative, g, b, "fused")
        from fixed_q4_candidate_only import cell
        r = cell(root, g, b)
        if fresh != old or r["ranks"]["ranks"] != [1024,1568,1792,2048]:
            raise ValueError(f"Completed candidate cell differs in input, placement, or ranks at {m}")
    return len(markers)


def partial_natural(ref, native):
    options = []
    for root in ref.parent.glob("fixed-q4-rank-followup.*"):
        if field(root, "native") == native and field(root, "followup_reference") == str(ref) and matched(root, ref):
            n = completed_natural_cells(root, ref, native)
            if n:
                options.append((n, root.stat().st_mtime_ns, root))
    return max(options)[2] if options else None



def natural_summary(root, ref):
    counts = [0, 0, 0]
    for g, b in CELLS:
        block = f"g{g}/bs{b}/block1"
        old, key = observation(ref, ref / block, g, b, "fused")
        new, live = observation(root, root / block, g, b, "fused")
        if live != key or not matched(root, ref):
            raise ValueError(f"G{g} BS{b} has a different input, card, runtime or placement")
        serial = serial_reference(ref, live, g, b)
        gain_text = "UNAVAILABLE"
        if serial is not None:
            gain = 100 * (1 - new["graph"] / serial["graph"])
            gain_text = f"{gain:+.2f}%"
            counts[0] += 1
            counts[1] += gain >= 15
            counts[2] += gain >= 20
        stages = ("graph", "LI", "tail")
        values = "/".join(f"{new[k]:.2f}" for k in stages)
        savings = "/".join(f"{old[k]-new[k]:+.2f}" for k in stages)
        print(f"G{g} BS{b} NATURAL_LI graph/LI/tail_mean_us={values} "
              f"vs_original_graph/LI/tail_saved_us={savings} vs_same_card_serial={gain_text}")
    print(f"NATURAL_LI serial_coverage={counts[0]}/{len(CELLS)} "
          f"at_least_15pct={counts[1]} at_least_20pct={counts[2]}")


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
    q = sub.add_parser("coverage")
    q.add_argument("ref", type=Path)
    q.add_argument("mode", choices=["serial", "journal", "frozen"])
    q.add_argument("--batches", nargs="+", type=int, choices=[20, 24, 28, 32])
    for action in ("timed", "partial"):
        t = sub.add_parser(action)
        t.add_argument("ref", type=Path)
        t.add_argument("native")
    validate = sub.add_parser("validate-partial")
    validate.add_argument("path", type=Path)
    validate.add_argument("ref", type=Path)
    validate.add_argument("native")
    for action in ("cached", "check", "report", "frozen-report", "natural-summary"):
        q = sub.add_parser(action)
        q.add_argument("root", type=Path)
        q.add_argument("ref", type=Path)
        if action in ("cached", "check"):
            q.add_argument("group", type=int)
            q.add_argument("batch", type=int)
            q.add_argument("mode", choices=["serial", "journal", "frozen"])
    a = p.parse_args()
    if a.action == "coverage":
        selected = [(g, b) for g, b in CELLS if not a.batches or b in a.batches]
        missing = [(g, b) for g, b in selected if not cached(a.ref, a.ref, g, b, a.mode)]
        print(f"ARCHIVE mode={a.mode} complete={len(selected)-len(missing)}/{len(selected)}")
        raise SystemExit(3 if missing else 0)
    if a.action == "serial-summary":
        raise SystemExit(0 if serial_summary(a.ref) else 3)
    if a.action == "validate-partial":
        print(f"RESUME verified_cells={completed_natural_cells(a.path, a.ref, a.native)} root={a.path}")
        raise SystemExit(0)
    if a.action in ("timed", "partial"):
        hit = (timed if a.action == "timed" else partial_natural)(a.ref, a.native)
        if hit:
            print(hit)
    elif a.action == "placement":
        print(*placement(a.path))
    elif a.action == "natural-summary":
        natural_summary(a.root, a.ref)
    elif a.action == "frozen-report":
        raise SystemExit(0 if frozen_report(a.root, a.ref) else 3)
    elif a.action == "report":
        raise SystemExit(0 if report(a.root, a.ref) else 3)
    else:
        hit = cached(a.root, a.ref, a.group, a.batch, a.mode)
        if a.action == "check" and not hit:
            raise ValueError(f"Unverified {a.mode} result at {a.root}")
        if hit and a.action == "check" and a.mode == "frozen":
            data = read(hit / "frozen.json")
            checks = data["timed_journal_checks"]
            matched_calls = sum(item["ordered_match"] for item in checks)
            print(f"G{a.group} BS{a.batch} CONTROLLED ordered_GET_calls={matched_calls}/10 "
                  f"graph/LI/tail_mean_us={data['graph_mean_us']:.2f}/{data['mean_us']['LI']:.2f}/{data['mean_us']['tail']:.2f} "
                  f"source={hit}")
        elif hit:
            print(hit)
