# SPDX-License-Identifier: Apache-2.0
"""Check multi-round capture coverage or extract one independently replayable snapshot."""

import argparse
import copy
import json
from pathlib import Path

from kvip.diagnostics.archive import TensorArchive, checked_path, load_snapshot


def inspect_series(root, *, expected_dp=None, expected_rounds=None, expected_layers=None, expected_content=None):
    """Check every manifest, without rereading terabytes of saved tensor payloads."""
    root = Path(root)
    paths = sorted(root.glob("dp[0-9]*/series.json"))
    if not paths:
        raise ValueError(f"No DP series manifests found under {root}")
    reports, ranks, expected_ranks = [], set(), None
    contents, real_plans = set(), set()
    for path in paths:
        series = json.loads(path.read_text())
        rank = series.get("dp_rank")
        if (
            series.get("schema_version") != 1
            or series.get("kind") != "kvaf_decode_snapshot_series"
            or series.get("scope") != "target"
            or type(rank) is not int
            or rank < 0
            or path.parent.name != f"dp{rank}"
            or rank in ranks
        ):
            raise ValueError(f"Invalid DP series identity: {path}")
        ranks.add(rank)
        content = series.get("content", "full")
        contents.add(content)
        if content not in ("full", "token_ids", "scores") or (
            expected_content is not None and content != expected_content
        ):
            raise ValueError(f"DP{rank} content is {content}, expected {expected_content}")
        configured = set(series["dp_ranks"])
        if expected_ranks is not None and configured != expected_ranks:
            raise ValueError("DP series disagree about the selected ranks")
        expected_ranks = configured
        if series.get("status") != "complete":
            raise ValueError(
                f"DP{rank} capture is {series.get('status')}: "
                f"{series.get('completed_rounds', 0)}/{series.get('expected_rounds')} rounds"
            )
        total = series["expected_rounds"]
        rounds = series["rounds"]
        if series["completed_rounds"] != total or len(rounds) != total:
            raise ValueError(f"DP{rank} round count differs from its completion marker")
        if expected_rounds is not None and total != expected_rounds:
            raise ValueError(f"DP{rank} has {total} rounds, expected {expected_rounds}")
        if content == "scores" and (
            series.get("seed_rounds") != 1 or series.get("consecutive_required") is not True or total < 2
        ):
            raise ValueError(f"DP{rank} scores require a seed and consecutive analysis rounds")
        real_rounds = set(series.get("real_input_rounds", []))
        real_groups = set(series.get("real_input_groups", []))
        real_count = 0
        groups = {entry["group"]: entry["layer_names"] for entry in series["groups"]}
        if content == "scores":
            if (
                bool(real_rounds) != bool(real_groups)
                or not real_rounds.issubset(range(1, total + 1))
                or not real_groups.issubset(groups)
            ):
                raise ValueError(f"DP{rank} real-input sampling plan exceeds the captured rounds/groups")
            real_plans.add((tuple(sorted(real_rounds)), tuple(sorted(real_groups))))
        layers = [name for names in groups.values() for name in names]
        if (
            len(groups) != len(series["groups"])
            or len(layers) != len(set(layers))
            or set(layers) != set(series["target_layer_names"])
            or len(layers) != series["expected_target_layer_count"]
        ):
            raise ValueError(f"DP{rank} target layer coverage is incomplete or duplicated")
        if expected_layers is not None and len(layers) != expected_layers:
            raise ValueError(f"DP{rank} covers {len(layers)} layers, expected {expected_layers}")
        previous_scheduler_round = -1
        for number, round_entry in enumerate(rounds, 1):
            if round_entry["round_index"] != number:
                raise ValueError(f"DP{rank} capture round sequence is incomplete")
            schedule = round_entry["scheduler_round"]
            if type(schedule) is not int or schedule <= previous_scheduler_round:
                raise ValueError(f"DP{rank} scheduler round sequence is not increasing")
            if content == "scores" and number > 1 and schedule != previous_scheduler_round + 1:
                raise ValueError(f"DP{rank} round {number} has no consecutive captured predecessor")
            previous_scheduler_round = schedule
            seen = set()
            for entry in round_entry["groups"]:
                group = entry["group"]
                if group not in groups or group in seen:
                    raise ValueError(f"DP{rank} round {number}: unexpected or duplicate group")
                seen.add(group)
                snapshot_path = checked_path(path.parent, entry["manifest"])
                manifest = json.loads(snapshot_path.read_text())
                source = manifest.get("metadata", {})
                context = source.get("context", {})
                requests = source.get("requests", [])
                signature = [{"request_id": item["request_id"], "owner": item["owner"]} for item in requests]
                if content in ("token_ids", "scores"):
                    is_scores = content == "scores"
                    valid_record = (
                        manifest.get("schema_version") == (4 if is_scores else 3)
                        and manifest.get("kind")
                        == ("kvaf_decode_score_trace" if is_scores else "kvaf_decode_token_trace")
                        and manifest.get("content") == content
                        and manifest.get("validation", {}).get("passed") is True
                        and manifest.get("validation", {}).get("kind")
                        == ("score_trace_integrity" if is_scores else "token_id_integrity")
                    )
                    if is_scores:
                        expected_real = number in real_rounds and group in real_groups
                        valid_record = (
                            valid_record
                            and manifest.get("real_inputs_saved") is expected_real
                            and source.get("capture_real_inputs") is expected_real
                        )
                        real_count += int(expected_real)
                    archive = manifest.get("archive", {})
                    token_file = checked_path(snapshot_path.parent, archive.get("file", ""))
                    valid_record = (
                        valid_record and token_file.is_file() and token_file.stat().st_size == archive.get("bytes")
                    )
                else:
                    valid_record = (
                        manifest.get("schema_version") in (1, 2) and manifest.get("oracle", {}).get("passed") is True
                    )
                if (
                    not valid_record
                    or manifest.get("status") != "complete"
                    or source.get("dp_rank") != rank
                    or source.get("group") != group
                    or source.get("layer_names") != groups[group]
                    or context.get("global_decode_confirmed") is not True
                    or context.get("round") != schedule
                    or manifest.get("geometry", {}).get("batch") != series["batch"]
                    or manifest.get("geometry", {}).get("token_count") != series["batch"] * series["queries"]
                    or len(requests) != series["batch"]
                    or signature != series["request_owner_signature"]
                    or any(item["query_stop"] - item["query_begin"] != series["queries"] for item in requests)
                ):
                    raise ValueError(f"DP{rank} round {number} group {group}: snapshot contract differs")
            if seen != set(groups):
                raise ValueError(f"DP{rank} round {number}: missing target groups")
        reports.append(
            dict(
                dp_rank=rank,
                rounds=total,
                layers=len(layers),
                groups=len(groups),
                snapshots=total * len(groups),
                batch=series["batch"],
                queries=series["queries"],
                content=content,
                analysis_rounds=total - 1 if content == "scores" else total,
                real_input_snapshots=real_count,
                scheduler_rounds=[entry["scheduler_round"] for entry in rounds],
            )
        )
    if ranks != expected_ranks or expected_dp is not None and ranks != set(expected_dp):
        raise ValueError(f"Missing/unexpected DP series: found {sorted(ranks)}, configured {sorted(expected_ranks)}")
    if len(contents) != 1:
        raise ValueError("DP series disagree about capture content")
    if len(real_plans) > 1:
        raise ValueError("DP series disagree about real-input sampling")
    if len({tuple(report["scheduler_rounds"]) for report in reports}) != 1:
        raise ValueError("Selected DPs did not capture the same coordinated scheduler rounds")
    return reports


def materialize_snapshot(source, destination):
    """Expand a selected packed snapshot to the original self-contained file layout."""
    from kvip.diagnostics.ddr import materialize_pages

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Extraction destination already exists: {destination}")
    archive = load_snapshot(source)
    try:
        manifest = copy.deepcopy(archive.manifest)
        manifest["schema_version"] = 1
        manifest.pop("object_store", None)
        descriptions = manifest.pop("tensors")
        manifest["tensors"] = {}
        destination.mkdir(parents=True)
        (destination / "INCOMPLETE").write_text("Extraction is not complete until manifest.json is published.\n")
        output = TensorArchive(destination, manifest)
        for name, description in descriptions.items():
            output.save(name, archive.array(name), dtype=description["dtype"])
        manifest["ddr"] = materialize_pages(archive.directory / "ddr", manifest["ddr"], destination / "ddr")
        pending = destination / "manifest.json.tmp"
        pending.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        pending.replace(destination / "manifest.json")
        (destination / "INCOMPLETE").unlink()
    finally:
        archive.close()
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--expect-dp", type=int, nargs="+")
    parser.add_argument("--expect-rounds", type=int)
    parser.add_argument("--expect-layers", type=int)
    parser.add_argument("--expect-content", choices=("full", "token_ids", "scores"))
    parser.add_argument("--extract", type=Path, help="Expand one snapshot to a new standalone directory")
    parser.add_argument("--dp-rank", type=int)
    parser.add_argument("--round", dest="round_index", type=int)
    parser.add_argument("--group", type=int)
    args = parser.parse_args(argv)
    selection = (args.dp_rank, args.round_index, args.group)
    if args.extract is not None and any(value is None for value in selection):
        parser.error("--extract requires --dp-rank, --round and --group")
    if args.extract is None and any(value is not None for value in selection):
        parser.error("--dp-rank, --round and --group require --extract")
    try:
        reports = inspect_series(
            args.root,
            expected_dp=args.expect_dp,
            expected_rounds=args.expect_rounds,
            expected_layers=args.expect_layers,
            expected_content=args.expect_content,
        )
        for report in reports:
            print(
                f"TRACE_DP_PASS dp={report['dp_rank']} batch={report['batch']} queries={report['queries']} "
                f"rounds={report['rounds']} layers={report['layers']} groups={report['groups']} "
                f"snapshots={report['snapshots']} content={report['content']} all_pure_decode=true"
            )
            if report["content"] == "scores":
                print(
                    f"TRACE_SCORE_PASS dp={report['dp_rank']} seed_rounds=1 "
                    f"analysis_rounds={report['analysis_rounds']} real_input_snapshots={report['real_input_snapshots']}"
                )
        if reports[0]["content"] in ("token_ids", "scores"):
            print(
                f"TRACE_SERIES_PASS coverage=all_target_layers content={reports[0]['content']} "
                "real_payload=not_collected"
            )
        else:
            print("TRACE_SERIES_PASS coverage=all_target_layers payload_check=at_capture_and_replay")
        if args.extract is not None:
            if args.dp_rank not in [report["dp_rank"] for report in reports] or args.round_index < 1:
                raise ValueError("Extraction DP/round is outside the captured series")
            series_path = args.root / f"dp{args.dp_rank}" / "series.json"
            series = json.loads(series_path.read_text())
            rounds = [entry for entry in series["rounds"] if entry["round_index"] == args.round_index]
            if len(rounds) != 1:
                raise ValueError("Extraction round is outside the captured series")
            groups = [entry for entry in rounds[0]["groups"] if entry["group"] == args.group]
            if len(groups) != 1:
                raise ValueError("Extraction group is outside the captured series")
            source = checked_path(series_path.parent, groups[0]["manifest"]).parent
            if series.get("content") in ("token_ids", "scores"):
                command = "score_replay" if series.get("content") == "scores" else "token_replay"
                raise ValueError(
                    f"Use python3 -m kvip.diagnostics.{command} with the selected token trace directory "
                    f"to construct standalone operator inputs: {source}"
                )
            materialize_snapshot(source, args.extract)
            print(f"TRACE_EXTRACT_PASS snapshot={args.extract}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"TRACE_SERIES_FAILED: {error}") from error


if __name__ == "__main__":
    main()
