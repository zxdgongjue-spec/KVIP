# SPDX-License-Identifier: Apache-2.0
"""Run a single diagnostic from the archived executable's worker. No ten-call capture or baseline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


def run(args):
    sys.path.insert(0, str(args.timed / "tests/operators/lifetch/support"))
    import torch
    import torch_npu
    import kvaf
    import kvip
    import execution
    from fixed_q4_candidate_only import cell
    from large_batch_input import load, input_digest
    from large_batch_pair import captured_score_oracle, check_scores, read_selected, replay_configuration
    from reporting import atomic_json
    from workloads import ReplayPagePlan

    row = cell(args.archive, args.group, args.batch)
    path = args.archive / f"g{args.group}/bs{args.batch}/block1/fused/packed_result.json"
    previous = json.loads(path.read_text())
    identity = previous["identity"]
    if not (Path(kvaf.__file__).resolve().is_relative_to(args.kvaf_root / "build-npu/python")
            and Path(kvip.__file__).resolve().is_relative_to(args.timed / "python")):
        raise ValueError("KVIP/KVAF must be loaded from the archived timing checkout and SDK")
    revision = subprocess.check_output(["git", "-C", args.timed, "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", args.timed, "status", "--porcelain", "-uno"], text=True).strip():
        raise ValueError(f"Tracked files changed in {args.timed}")
    with (args.kvaf_root / "build-npu/bin/kvaf_execute_batch_get_kernels.o").open("rb") as f:
        sdk_digest = hashlib.file_digest(f, "sha256").hexdigest()
    sequence, donors = load(args.trace, args.group, 1, args.batch, 0, args.prepared)
    inputs, pages = sequence.inputs, ReplayPagePlan(sequence)
    wanted = dict(native=revision, sequence_sha256=input_digest(inputs, pages, donors), donor_rows=donors,
                  physical_npu=4, host=os.uname().nodename, hardware=torch.npu.get_device_name(torch.device("npu:0")),
                  torch=str(torch.__version__), torch_npu=str(torch_npu.__version__),
                  cann=str(getattr(torch.version, "cann", "")), kvaf_get_kernel_sha256=sdk_digest)
    if any(identity[k] != v for k,v in wanted.items()) or os.environ["ASCEND_RT_VISIBLE_DEVICES"] != "4":
        raise ValueError("Diagnostic differs from the archived input, binary, host, software or physical card")
    reference = captured_score_oracle(inputs)[0]
    args.packed_ten = True
    cfg = replay_configuration(args, "four-ranks")
    cfg.warmup, cfg.iterations = 0, 1
    proof = {}

    def benchmark(torch_mod, states, seq, config, directory, *, fused):
        if not fused or len(states) != 1 or states[0].large_window_launch.source_fingerprint != previous["cold_build_fingerprint"]:
            raise ValueError("Expected one state using the same measured cold executable")
        state = states[0]
        state.reset()
        state.forward.set_diagnostics(False)
        torch_mod.npu.current_stream(state.device).synchronize()
        try:
            if args.group == 7:
                from four_rank_tail_clocks import collect
                proof["tail"] = collect(state, args.output, args.timed)
            else:
                from four_rank_li_counters import collect
                proof["li_counters"] = collect(state, args.output)
                counters = proof["li_counters"]
                print(f"LI_COUNTER G{args.group} BS{args.batch} status={counters['status']} "
                      f"coverage={json.dumps(counters.get('coverage', {}), separators=(',', ':'))}")
        except BaseException as error:
            execution.NativeFetchSession.quarantine("single_diagnostic_completion_uncertain", error)
            raise
        selected, traffic = read_selected(state, inputs, True)
        check_scores(inputs, selected, reference)
        state.validate(inputs)
        proof["traffic"] = traffic
        compare_keys = ("posted_per_window", "row_window_prefetch", "row_query_final")
        same = sum(all(sample[k] == traffic[k] for k in compare_keys) for sample in previous["traffic"])
        proof["archive_matches_window_and_final_counts"] = same
        nearest = min(previous["traffic"], key=lambda sample: sum(
            abs(a-b) for key in ("row_window_prefetch", "row_query_final")
            for left, right in zip(sample[key], traffic[key]) for a,b in zip(left,right)))
        proof["diagnostic_minus_nearest_archived_counts"] = {
            key: [[a-b for a,b in zip(left,right)] for left,right in zip(traffic[key],nearest[key])]
            for key in ("row_window_prefetch","row_query_final")}
        proof["posted_per_window"] = traffic["posted_per_window"]
        if not same:
            print("SINGLE_DIAGNOSTIC counts differ; inspect per-row deltas before attributing original tail")
        print(f"SINGLE_DIAGNOSTIC G{args.group} BS{args.batch} TopK_score/KV/Index/private_CQ=PASS "
              f"queries={4*args.batch} calls=1 archived_row_Q_count_matches={same}/10 "
              f"prefetch/final={traffic['prefetch']}/{traffic['final']}")
        return dict(correctness="PASS", failures=[])

    result = execution.run_path(cfg, args.output/"path", sequence, pages, fused=True, measure=benchmark,
                                large_window_policy="four-ranks")
    if result["correctness"] != "PASS" or result["failures"] or result["runtime_workers"] != 32:
        raise ValueError("Original eager certification or diagnostic failed")
    atomic_json(args.output/"single_diagnostic.json", dict(input_archive=str(path), identity=identity,
                controller_commit=subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[4]),
                "rev-parse", "HEAD"], text=True).strip(), correctness="PASS", measured_calls=0,
                diagnostic_calls=1, reference_oracle=row["oracle"], **proof))


def cached(source, group, batch, controller):
    from large_batch_placement import placement
    if not ((int(group) == 7 and int(batch) in (28,32)) or (int(group) in (2,10) and int(batch) == 32)):
        raise ValueError("Unexpected diagnostic cell")
    ref = source / f"g{group}/bs{batch}/block1/fused"
    expected = json.loads((ref/"packed_result.json").read_text())
    revision = subprocess.check_output(["git", "-C", controller, "rev-parse", "HEAD"], text=True).strip()
    for path in sorted(source.parent.glob(f"four-rank-diagnostic.*/g{group}/bs{batch}/fused/single_diagnostic.json"),
                       key=lambda item: item.stat().st_mtime_ns, reverse=True):
        try:
            observed = json.loads(path.read_text())
            if ((path.parent/"placement_verified").is_file() and (path.parent/"worker.log").is_file()
                    and observed.get("input_archive") == str(ref/"packed_result.json")
                    and observed["identity"] == expected["identity"] and observed["correctness"] == "PASS"
                    and observed["controller_commit"] == revision and observed["measured_calls"] == 0
                    and observed["diagnostic_calls"] == 1
                    and placement(path.parent/"thp.json") == placement(ref/"thp.json")):
                return path.parent
        except (OSError, KeyError, TypeError, ValueError):
            continue
    return None


def package(root):
    root = root.resolve(strict=True)
    destination = root / "four-rank-diagnostics.tar.gz"
    pending = destination.with_suffix(".tmp")
    with tarfile.open(pending, "w:gz") as out:
        out.add(root / "summary.txt", arcname="summary.txt")
        for index in sorted(root.glob("g*/bs*/diagnostic_source.txt")):
            group, batch, _ = index.relative_to(root).parts
            source = Path(index.read_text().strip()).resolve(strict=True)
            if source.parents[3] != root.parent or not (source/"placement_verified").is_file():
                raise ValueError(f"Invalid diagnostic source {source}")
            names = [source/"single_diagnostic.json", source/"thp.json", source/"msprof-export.log"]
            names.extend(sorted((source/"li-pmu").rglob("op_summary*.csv")))
            for file in names:
                if file.is_file():
                    out.add(file, arcname=f"{group}/{batch}/{file.relative_to(source)}")
    pending.replace(destination)
    print(f"RETURN_FILE={destination} bytes={destination.stat().st_size}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--package"]:
        package(Path(sys.argv[2]).resolve())
        sys.exit(0)
    if sys.argv[1:2] == ["--cached"]:
        hit = cached(Path(sys.argv[2]),sys.argv[3],sys.argv[4],Path(sys.argv[5]))
        if hit:
            print(hit)
        sys.exit(0)
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("timed","archive"):
        p.add_argument(name, type=Path)
    p.add_argument("group", choices=(2,10,7), type=int)
    p.add_argument("batch", choices=(28,32), type=int)
    for name in ("trace","kvaf_root"):
        p.add_argument(name, type=Path)
    p.add_argument("kvaf_server")
    p.add_argument("prepared", type=Path)
    p.add_argument("output", type=Path)
    run(p.parse_args())
