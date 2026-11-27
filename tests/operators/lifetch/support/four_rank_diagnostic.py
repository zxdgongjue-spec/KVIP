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

    if os.getenv("PEER_RANK_DIAG_MODE") == "prefetch-ready" and (args.group,args.batch) != (7,28):
        raise ValueError("Window-prefetch CQ diagnosis requires G7 BS28")
    mode = os.getenv("PEER_RANK_DIAG_MODE", "")
    window_mode = mode.startswith("window-tail-x")
    local_only = window_mode and args.archive == Path("LOCAL")
    binding, previous, path = None, None, None
    if window_mode:
        from b24_window_tail_diagnostic import CELLS, MissingNaturalPolicy, bind, select_policy
        import large_two_cold
        if args.group not in CELLS.get(mode, ()) or args.batch != 32:
            raise ValueError("Unexpected local window policy or group")
        name=mode.replace("window-tail-x","window-x")+"-early"
        oracle_name="captured_logical_scores"
        if not local_only:
            try:
                previous,binding = bind(mode, args.archive, args.group, args.batch)
            except MissingNaturalPolicy:
                local_only=True
            else:
                path = Path(binding["packed_result"])
        select_policy(large_two_cold, name)
    else:
        oracle_name=cell(args.archive, args.group, args.batch)["oracle"]
        path=args.archive / f"g{args.group}/bs{args.batch}/block1/fused/packed_result.json"
        previous=json.loads(path.read_text())
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
    identity=previous["identity"] if previous is not None else dict(wanted,group=args.group,batch=args.batch)
    if window_mode and revision!="d0fc5963881bd38fa97eff0f6ef40c757a6e8713":
        raise ValueError("Unexpected ordinary hot/LI/terminal source")
    if any(identity[k] != v for k,v in wanted.items()) or os.environ["ASCEND_RT_VISIBLE_DEVICES"] != "4":
        raise ValueError("Diagnostic differs from the archived input, binary, host, software or physical card")
    reference = captured_score_oracle(inputs)[0]
    args.packed_ten = True
    cfg = replay_configuration(args, "four-ranks")
    cfg.warmup, cfg.iterations = 0, 1
    proof = {"window_policy":name, "window_policy_archive":binding} if window_mode else {}

    def observe_window(state):
        from b24_window_tail_diagnostic import verify_launch
        from four_rank_accept_worker import terminal_image
        from four_rank_tail_clocks import collect
        state.reset()
        state.forward.set_diagnostics(False)
        torch.npu.current_stream(state.device).synchronize()
        bound = terminal_image(state, args.kvaf_root)
        if previous is not None:
            verify_launch(bound, previous, binding)
        elif not bound.get("tail_text") or not bound.get("cold_build"):
            raise ValueError("Original LI/terminal code missing in local unpaired sample")
        proof["ordinary_images"] = bound
        proof["tail"] = collect(state, args.output, args.timed, prefetch_ready=True, profile_only_terminal=True)

    def benchmark(torch_mod, states, seq, config, directory, *, fused):
        if not fused or len(states) != 1 or (previous is not None and
                states[0].large_window_launch.source_fingerprint != previous["cold_build_fingerprint"]):
            raise ValueError("Expected one state using the same measured cold executable")
        state = states[0]
        if not window_mode:
            state.reset()
            state.forward.set_diagnostics(False)
            torch_mod.npu.current_stream(state.device).synchronize()
        try:
            if window_mode:
                if not proof.get("tail") or "ordinary_images" not in proof:
                    raise ValueError("Missed the natural eager-forward probe")
            elif args.group == 7:
                from four_rank_tail_clocks import collect
                proof["tail"] = collect(state, args.output, args.timed, os.getenv("PEER_RANK_DIAG_MODE") == "prefetch-ready")
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
        same = None
        if previous is not None:
            compare_keys = ("posted_per_window", "row_window_prefetch", "row_query_final")
            same = sum(all(sample[k] == traffic[k] for k in compare_keys) for sample in previous["traffic"])
            nearest = min(previous["traffic"], key=lambda sample: sum(
                abs(a-b) for key in ("row_window_prefetch", "row_query_final")
                for left, right in zip(sample[key], traffic[key]) for a,b in zip(left,right)))
            proof["diagnostic_minus_nearest_archived_counts"] = {
                key: [[a-b for a,b in zip(left,right)] for left,right in zip(traffic[key],nearest[key])]
                for key in ("row_window_prefetch","row_query_final")}
        proof["archive_matches_window_and_final_counts"] = same
        proof["posted_per_window"] = traffic["posted_per_window"]
        receipt = f"{same}/10" if same is not None else "UNAVAILABLE"
        if window_mode:
            from b24_window_tail_diagnostic import witness
            proof["critical_tail"]=witness(proof["tail"])
            print(f"WINDOW_TAIL G{args.group} BS{args.batch} mode={name} "
                  f"source={path or 'LOCAL_UNPAIRED'} LI_image={proof['ordinary_images']['cold_build']} "
                  f"archived_row_Q_counts_match={receipt}")
            print("CRITICAL_TAIL "+json.dumps(proof["critical_tail"],separators=(",",":")))
        if same==0:
            print("SINGLE_DIAGNOSTIC counts differ; inspect per-row deltas before attributing original tail")
        print(f"SINGLE_DIAGNOSTIC G{args.group} BS{args.batch} TopK_score/KV/Index/private_CQ=PASS "
              f"queries={4*args.batch} calls=1 archived_row_Q_count_matches={receipt} "
              f"prefetch/final={traffic['prefetch']}/{traffic['final']}")
        return dict(correctness="PASS", failures=[])

    from contextlib import nullcontext
    context = nullcontext()
    if window_mode:
        from b24_window_tail_diagnostic import capture_first
        context = capture_first(execution, observe_window)
    with context:
        result = execution.run_path(cfg, args.output/"path", sequence, pages, fused=True, measure=benchmark,
                                    large_window_policy="four-ranks")
    if result["correctness"] != "PASS" or result["failures"] or result["runtime_workers"] != 32:
        raise ValueError("Original eager certification or diagnostic failed")
    atomic_json(args.output/"single_diagnostic.json", dict(input_archive=str(path) if path is not None else None, identity=identity,
                controller_commit=subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[4]),
                "rev-parse", "HEAD"], text=True).strip(), correctness="PASS", measured_calls=0,
                diagnostic_calls=1, reference_oracle=oracle_name, **proof))


def cached(source, group, batch, controller):
    from large_batch_placement import placement
    if not ((int(group) == 7 and int(batch) in (28,32)) or (int(group) in (2,10) and int(batch) == 32)):
        raise ValueError("Unexpected diagnostic cell")
    mode = os.getenv("PEER_RANK_DIAG_MODE", "")
    window_mode=mode.startswith("window-tail-x")
    local_sample=window_mode and source==Path("LOCAL")
    area=Path(os.environ["PEER_G2_WORK"])/"peer-baseline" if local_sample else source.parent
    binding,packed_path,expected=None,None,None
    if not local_sample:
        ref=source / f"g{group}/bs{batch}/block1/fused"
        if window_mode:
            from b24_window_tail_diagnostic import MissingNaturalPolicy, bind
            try:
                expected,binding=bind(mode,source,int(group),int(batch))
            except MissingNaturalPolicy:
                local_sample=True
            else:
                packed_path=Path(binding["packed_result"])
        else:
            packed_path=ref/"packed_result.json"
            expected=json.loads(packed_path.read_bytes())
    revision=subprocess.check_output(["git","-C",controller,"rev-parse","HEAD"],text=True).strip()
    for path in sorted(area.glob(f"four-rank-diagnostic.*/g{group}/bs{batch}/fused/single_diagnostic.json"),
                       key=lambda item: item.stat().st_mtime_ns, reverse=True):
        try:
            observed = json.loads(path.read_text())
            if local_sample:
                from portable_two_row_audit import local, local_identity
                from large_batch_placement import field
                root=path.parents[3]
                if (not local(root,local_identity()) or
                        field(root,"kvaf")!="813f1750a50b7e6c92d5f5397f2fbc05befef6da" or
                        field(root,"kvaf_tree")!="1de18e8315137d7a0227c3ae4c82afd45b5fe743"):
                    continue
                if (observed["identity"]["native"]!="d0fc5963881bd38fa97eff0f6ef40c757a6e8713"
                        or observed["identity"]["group"]!=int(group) or observed["identity"]["batch"]!=int(batch)):
                    continue
                placement(path.parent/"thp.json")
            if ((path.parent/"placement_verified").is_file() and (path.parent/"worker.log").is_file()
                    and (observed.get("window_policy_archive") == binding)
                    and (not window_mode or observed.get("window_policy")==mode.replace("window-tail-x","window-x")+"-early")
                    and ((mode == "prefetch-ready" or window_mode) ==
                         ("prefetch_cq_checkpoints" in observed.get("tail", {})))
                    and observed.get("input_archive") == (str(packed_path) if packed_path is not None else None)
                    and (expected is None or observed["identity"] == expected["identity"]) and observed["correctness"] == "PASS"
                    and observed["controller_commit"] == revision and observed["measured_calls"] == 0
                    and observed["diagnostic_calls"] == 1
                    and (local_sample or placement(path.parent/"thp.json") == placement(ref/"thp.json"))):
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
            counters = json.loads((source/"single_diagnostic.json").read_text()).get("li_counters", {})
            profiler = source/"li-pmu"
            if counters.get("status") == "PARTIAL":
                # Preserve the one native capture for offline re-export on a different CANN installation.
                names.extend(sorted(p for p in profiler.rglob("*") if p.is_file() and not p.is_symlink()))
            else:
                names.extend(sorted(profiler.rglob("op_summary*.csv")))
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
