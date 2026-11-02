#!/usr/bin/env python3
"""Compare local MTP kernel snapshots without a vLLM checkout or Git history."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KERNEL = "a5_lidu_fused_batch_get_prepare_mtp"
SOURCE_FILES = (KERNEL + ".cpp", KERNEL + "_tiling.h", KERNEL + "_vector.h")
DEFAULT_IMAGE = "quay.io/ascend/cann:9.1.0-950-ubuntu22.04-py3.12"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_metrics():
    path = ROOT / "tools" / "legacy_metrics.py"
    spec = importlib.util.spec_from_file_location("mtp_bundle_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.HARNESS = ROOT / "harness"
    module.ORACLE_PATH = ROOT / "oracle" / "reference.py"
    return module


def select(items: list[dict], names: list[str] | None) -> list[dict]:
    if not names:
        return items
    unknown = set(names) - {item["name"] for item in items}
    if unknown:
        raise ValueError(f"unknown names: {sorted(unknown)}")
    return [item for item in items if item["name"] in names]


def normalize_case(item: dict) -> dict:
    item = dict(item)
    if item.get("mutation"):
        raise ValueError(
            f"layout mutations require a dedicated contract oracle; unsupported mutation: {item['mutation']}"
        )
    if Path(item["name"]).name != item["name"] or item["name"] in (".", ".."):
        raise ValueError("case name must be a plain directory name")
    batch = int(item["batch"])
    if batch < 1:
        raise ValueError("batch must be positive")
    item.setdefault("source_capacity", 131072)
    item.setdefault("cache_tokens", 12288)
    item.setdefault("seed", 20261029)
    queries = list(item.get("query_counts", [int(item.get("query_count", 4))] * batch))
    misses = list(item.get("miss_counts", [int(item.get("unique_misses", 300))] * batch))
    if len(queries) != batch or len(misses) != batch:
        raise ValueError(f"{item['name']}: per-request arrays must contain B values")
    if any(q not in (2, 3, 4) for q in queries):
        raise ValueError("native MTP supports 2, 3 or 4 packed queries per request")
    if any(m < 0 or m > q * 2048 for q, m in zip(queries, misses)):
        raise ValueError("miss count exceeds query capacity")
    item.update(query_counts=queries, miss_counts=misses)
    item["query_count"] = max(queries)
    item["unique_misses"] = max(misses)
    return item


def core_geometry(batch: int) -> tuple[int, int]:
    if batch > 32:
        return 32, 1
    per_request = next(n for n in (8, 4, 2, 1) if n <= 32 // batch)
    return batch * per_request, per_request


def make_case(oracle, spec: dict):
    return oracle.build_case(
        batch_size=spec["batch"],
        query_counts=spec["query_counts"],
        miss_counts=spec["miss_counts"],
        source_capacity=spec["source_capacity"],
        cache_tokens=spec["cache_tokens"],
        seed=spec["seed"],
    )


def snapshot_sources(output: Path, variants: list[dict]) -> list[dict]:
    records = []
    for variant in variants:
        local = ROOT / variant["path"]
        if not local.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("variant source must be inside the bundle")
        source = local / "op_kernel"
        target = output / "sources" / variant["name"]
        target.mkdir(parents=True)
        hashes = {}
        for filename in SOURCE_FILES:
            path = source / filename
            shutil.copyfile(path, target / filename)
            hashes[filename] = sha256(path)
        content = (target / SOURCE_FILES[0]).read_bytes()
        marker = b'\nextern "C" __global__ __aicore__ void '
        if marker not in content:
            raise ValueError(f"{variant['name']}: kernel entry marker absent")
        implementation = target / (KERNEL + "_implementation.inc")
        implementation.write_bytes(content.rsplit(marker, 1)[0] + b"\n")
        records.append({**variant, "source_sha256": hashes, "implementation_sha256": sha256(implementation)})
    return records


def input_arrays(case) -> dict:
    return {
        name: getattr(case, name)
        for name in (
            "topk",
            "actual_seq_lengths_query",
            "req_pool_entries",
            "token_to_slot_pool",
            "slot_to_token_pool",
            "eviction_cursors",
            "cache_tokens",
            "candidate_lens",
            "miss_slots",
            "miss_slot_bitmap",
            "resident_block_table",
            "logical_seq_lengths",
        )
    }


def prepare_cases(output: Path, cases: list[dict], warmup: int, active: int, profile_enabled: int) -> list[dict]:
    oracle = load_metrics().load_oracle()
    records = []
    for spec in cases:
        case = make_case(oracle, spec)
        directory = output / "cases" / spec["name"]
        directory.mkdir(parents=True)
        for name, value in input_arrays(case).items():
            np.ascontiguousarray(value).tofile(directory / f"{name}.bin")
        used, per_request = core_geometry(spec["batch"])
        pool_size = int(case.token_to_slot_pool.shape[0])
        config = (
            spec["batch"],
            case.topk.shape[0],
            pool_size,
            spec["source_capacity"],
            spec["cache_tokens"],
            case.resident_block_table.shape[1],
            used,
            per_request,
            2,
            4,
            pool_size,
            spec["source_capacity"] // 128,
            warmup,
            active,
            profile_enabled,
        )
        (directory / "config.txt").write_text(" ".join(map(str, config)) + "\n")
        records.append(
            {
                **spec,
                "used_cores": used,
                "cores_per_request": per_request,
                "packed_queries": int(case.topk.shape[0]),
                "pool_size": pool_size,
                "input_sha256": {p.name: sha256(p) for p in sorted(directory.iterdir())},
            }
        )
    return records


def verify_input_hashes(directory: Path, expected: dict) -> None:
    for filename, digest in expected.items():
        if sha256(directory / filename) != digest:
            raise AssertionError(f"input changed: {directory / filename}")


def canonical_reference(case: dict) -> str:
    return "p0_extended_reference" if max(case.get("miss_counts", [case["unique_misses"]])) > 2048 else "v0_base"


def add_comparisons(rows: list[dict], cases: list[dict], profile_enabled: bool = True) -> dict:
    indexed = {(r["variant"], r["case"]): r for r in rows}
    winners = {}
    for case in cases:
        reference_name = canonical_reference(case)
        base = indexed.get((reference_name, case["name"]))
        base_cycles = (
            base.get("median_total_cycles")
            if base and base["correctness"] == "pass" and base.get("ranking_eligible", True)
            else None
        )
        eligible = []
        for row in rows:
            if row["case"] != case["name"]:
                continue
            row["reference_variant"] = reference_name
            row["improvement_vs_reference_pct"] = None
            if row["correctness"] != "pass" or not profile_enabled or not row.get("ranking_eligible", True):
                continue
            cycles = row.get("median_total_cycles")
            if cycles is not None and cycles > 0:
                eligible.append(row)
                if base_cycles and base_cycles > 0:
                    row["improvement_vs_reference_pct"] = 100 * (1 - cycles / base_cycles)
        if eligible:
            winners[case["name"]] = min(eligible, key=lambda r: r["median_total_cycles"])["variant"]
    return winners


def run_command(argv: list[str], log: Path, timeout: int, env=None, cwd=None) -> None:
    with log.open("w") as stream:
        subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=timeout, env=env, cwd=cwd)


def simulator_environment(cann: Path, disposable: bool) -> tuple[dict, dict]:
    candidates = [
        cann / "aarch64-linux/simulator/dav_3510/lib",
        cann / "x86_64-linux/simulator/dav_3510/lib",
        cann / "tools/simulator/Ascend950PR_9599/lib",
    ]
    simulator = next((p for p in candidates if (p / "config.json").is_file()), None)
    if simulator is None:
        raise RuntimeError(f"Ascend950 simulator configuration not found under {cann}")
    config_path = simulator / "config.json"
    config = read_json(config_path)
    log = config.get("LOG", config.get("log", {}))
    if disposable:
        log["flush_level"] = 2
        log["core_enable_mask"] = ["0x0"]
        write_json(config_path, config)
    elif log.get("flush_level") != 2:
        raise RuntimeError("native simulator requires flush_level=2; this runner does not edit native CANN")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(simulator) + ":" + env.get("LD_LIBRARY_PATH", "")
    return env, {
        "cann": str(cann),
        "simulator_lib": str(simulator),
        "configuration": config,
        "platform": platform.platform(),
    }


def execute(output: Path, cann: Path, jobs: int, timeout: int, disposable: bool) -> None:
    manifest = read_json(output / "manifest.json")
    env, environment_record = simulator_environment(cann, disposable)
    write_json(output / "environment.json", environment_record)
    builds = {}
    failures = {}
    for variant in manifest["variants"]:
        name = variant["name"]
        directory = output / "build" / name
        directory.mkdir(parents=True)
        try:
            run_command(
                [
                    "cmake",
                    "-S",
                    str(ROOT / "harness"),
                    "-B",
                    str(directory),
                    "-DRUN_MODE=sim",
                    f"-DASCEND_CANN_PACKAGE_PATH={cann}",
                    f"-DMTP_VARIANT_DIR={output / 'sources' / name}",
                ],
                directory / "configure.log",
                600,
                env,
            )
            run_command(["cmake", "--build", str(directory), "-j2"], directory / "build.log", 1800, env)
            builds[name] = directory / "mtp_prepare_sim"
            variant["binary_sha256"] = sha256(builds[name])
        except (OSError, subprocess.SubprocessError) as error:
            failures[name] = str(error)

    def one(variant, case):
        name, case_name = variant["name"], case["name"]
        directory = output / "runs" / name / case_name
        directory.mkdir(parents=True)
        try:
            if name in failures:
                raise RuntimeError("build failed: " + failures[name])
            source = output / "cases" / case_name
            verify_input_hashes(source, case["input_sha256"])
            for path in source.iterdir():
                shutil.copyfile(path, directory / path.name)
            started = time.monotonic()
            run_command([str(builds[name]), str(directory)], directory / "simulator.log", timeout, env, directory)
            diagnostics = (directory / "simulator.log").read_text(errors="replace").lower()
            if any(marker in diagnostics for marker in ("[error]", "assertion failed", "vec_err_instr")):
                raise RuntimeError("simulator reported an instruction/runtime diagnostic")
            write_json(
                directory / "execution.json",
                {
                    "status": "simulated",
                    "elapsed_seconds": time.monotonic() - started,
                    "binary_sha256": variant["binary_sha256"],
                },
            )
        except Exception as error:
            write_json(directory / "execution.json", {"status": "error", "error": str(error)})
        print(f"finished {name}/{case_name}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(one, variant, case) for variant in manifest["variants"] for case in manifest["cases"]]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    write_json(output / "manifest.json", manifest)


def sample_directories(run: Path, active: int) -> tuple[list[Path], str]:
    samples = run / "samples"
    if samples.is_dir():
        expected = [samples / str(n) for n in range(active)]
        if any(not p.is_dir() for p in expected):
            raise AssertionError("missing active output sample")
        actual = {p.name for p in samples.iterdir() if p.is_dir()}
        if actual != {str(n) for n in range(active)}:
            raise AssertionError("active sample count mismatch")
        return expected, "all_active_samples"
    if active == 1:
        return [run], "last_sample_only_active_1"
    raise AssertionError("legacy run contains only last output; all active outputs cannot be verified")


def verify(output: Path, run_root: Path | None = None) -> dict:
    manifest = read_json(output / "manifest.json")
    metrics = load_metrics()
    oracle = metrics.load_oracle()
    active = manifest["active"]
    rows = []
    traces = output / "traces"
    traces.mkdir(exist_ok=True)
    for case in manifest["cases"]:
        generated = make_case(oracle, normalize_case(case))
        reference = oracle.reference_update(generated, layer_id=2, num_layers=4, qcap=case["source_capacity"] // 128)
        for variant in manifest["variants"]:
            directory = (run_root or output) / "runs" / variant.get("run_variant", variant["name"]) / case["name"]
            row = {
                "variant": variant["name"],
                "revision": variant["revision"],
                "case": case["name"],
                "correctness": "fail",
                "error": "",
                "batch": case["batch"],
                "query_counts": case["query_counts"],
                "miss_counts": case["miss_counts"],
                "ranking_eligible": variant.get("ranking_eligible", True),
            }
            try:
                verify_input_hashes(directory, case["input_sha256"])
                samples, coverage = sample_directories(directory, active)
                for sample in samples:
                    metrics.assert_unordered_result(oracle, generated, reference, sample)
                row.update(correctness="pass", verified_samples=len(samples), sample_coverage=coverage)
                if manifest.get("profile_enabled", 1):
                    summary = metrics.summarize_profile(directory, case["batch"], active)
                    if len(summary["active_total_cycles"]) != active or any(
                        value <= 0 for value in summary["active_total_cycles"]
                    ):
                        raise AssertionError("missing positive profiling cycle count in an active sample")
                    row.update(summary)
                    trace = traces / f"{variant['name']}__{case['name']}.json"
                    metrics.write_perfetto_trace(trace, variant["name"], case["name"], directory, case["batch"], active)
                    row["trace"] = str(trace.relative_to(output))
            except Exception as error:
                row.update(correctness="fail", error=f"{type(error).__name__}: {error}")
            rows.append(row)
    winners = add_comparisons(rows, manifest["cases"], bool(manifest.get("profile_enabled", 1)))
    result = {
        "schema_version": 1,
        "status": "pass" if all(r["correctness"] == "pass" for r in rows) else "fail",
        "scope": "CANN_Ascend950_simulator_only",
        "physical_npu_executed": False,
        "network": False,
        "device_CQ_executed": False,
        "warmup": manifest["warmup"],
        "active": active,
        "best_variant_by_case": winners,
        "results": rows,
    }
    write_json(output / "results.json", result)
    fields = [
        "variant",
        "revision",
        "case",
        "batch",
        "query_counts",
        "miss_counts",
        "correctness",
        "ranking_eligible",
        "verified_samples",
        "sample_coverage",
        "median_total_cycles",
        "reference_variant",
        "improvement_vs_reference_pct",
        *(f"median_{p}_cycles" for p in metrics.TIMED_PHASES),
        "trace",
        "error",
    ]
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# MTP local kernel matrix",
        "",
        "Metric: Ascend950 simulator device cycles. network=false; device_CQ_executed=false.",
        "Perfetto phase positions are reconstructed from duration fields, not instruction timestamps.",
        "",
        f"Warmup={manifest['warmup']}; active={active}; "
        f"PASS={sum(r['correctness'] == 'pass' for r in rows)}; "
        f"FAIL={sum(r['correctness'] != 'pass' for r in rows)}.",
        "",
        "Each comparison uses the same generated input hashes. Positive percentages mean fewer cycles.",
        "Failed baselines produce no percentage; failed kernels are excluded from winners.",
        "",
        "| Case | Variant | Oracle | Cycles | Reference | Reduction |",
        "|---|---|---|---:|---|---:|",
    ]
    for row in rows:
        gain = row.get("improvement_vs_reference_pct")
        lines.append(
            f"| {row['case']} | {row['variant']} | {row['correctness']} | "
            f"{row.get('median_total_cycles', '')} | {row['reference_variant']} | "
            f"{'' if gain is None else f'{gain:.2f}%'} |"
        )
    errors = [r for r in rows if r["error"]]
    if errors:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {r['variant']}/{r['case']}: {r['error'].replace(chr(10), ' ')}" for r in errors)
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def import_manifest(
    existing: Path,
    variants: list[dict],
    cases: list[dict],
    output: Path,
    names: list[str] | None,
    case_names: list[str] | None,
) -> dict:
    original = read_json(existing / "manifest.json")
    by_hash = {
        tuple(sha256(ROOT / v["path"] / "op_kernel" / filename) for filename in SOURCE_FILES): v for v in variants
    }
    imported = []
    for old in original["variants"]:
        old_name = old["name"]
        if Path(old_name).name != old_name or old_name in (".", ".."):
            raise ValueError("import variant name must be a plain directory name")
        snapshot = existing / "sources" / old_name
        if not all((snapshot / filename).is_file() for filename in SOURCE_FILES):
            raise ValueError(f"{old_name}: import requires all three archived kernel source files")
        actual_hashes = tuple(sha256(snapshot / filename) for filename in SOURCE_FILES)
        expected_hashes = tuple(old.get("source_sha256", {}).get(filename) for filename in SOURCE_FILES)
        if actual_hashes != expected_hashes:
            raise ValueError(f"{old_name}: archived kernel source hashes do not match the import manifest")
        known = by_hash.get(actual_hashes)
        record = {**old, "run_variant": old_name, "ranking_eligible": known is not None}
        if known:
            record.update(name=known["name"], role=known.get("role"))
        else:
            record.update(role="unrecognized_source", source_identity="not_in_packaged_variants")
        imported.append(record)
    if len({v["name"] for v in imported}) != len(imported):
        raise ValueError("import contains duplicate canonical kernel snapshots; select one matrix per import")
    imported = select(imported, names)
    selected_cases = select([normalize_case(c) for c in original["cases"]], case_names)
    # Imported manifests often did not hash config.txt; all binary inputs must still match.
    manifest = {
        **original,
        "variants": imported,
        "cases": selected_cases,
        "profile_enabled": original.get("profile_enabled", 1),
        "imported_from": str(existing),
        "imported_manifest_sha256": sha256(existing / "manifest.json"),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="print the Cartesian matrix without running")
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--docker", action="store_true")
    mode.add_argument("--native", action="store_true")
    mode.add_argument("--verify-existing", type=Path, metavar="RESULT_DIRECTORY")
    parser.add_argument("--inside-container", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--case-file", type=Path, default=ROOT / "cases/matrix.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--active", type=int, default=3)
    parser.add_argument("--profile-enabled", type=int, choices=(0, 1), default=1)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--cann", type=Path)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.active < 1 or args.jobs < 1 or args.timeout < 1:
        parser.error("warmup >= 0; active, jobs and timeout must be positive")
    if args.inside_container:
        if not args.output or not args.cann:
            parser.error("container worker requires output and cann")
        execute(args.output.resolve(), args.cann, args.jobs, args.timeout, True)
        return 0
    variants = read_json(ROOT / "variants.json")["variants"]
    cases = [normalize_case(c) for c in read_json(args.case_file)["cases"]]
    selected_variants = select(variants, args.variants)
    selected_cases = select(cases, args.cases)
    if not any((args.prepare_only, args.docker, args.native, args.verify_existing)):
        print(
            json.dumps(
                {
                    "variants": [v["name"] for v in selected_variants],
                    "cases": [c["name"] for c in selected_cases],
                    "cells": len(selected_variants) * len(selected_cases),
                    "warmup": args.warmup,
                    "active": args.active,
                    "references": {c["name"]: canonical_reference(c) for c in selected_cases},
                },
                indent=2,
            )
        )
        return 0
    output = (args.output or ROOT / ".artifacts" / time.strftime("%Y%m%d-%H%M%S", time.gmtime())).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.verify_existing:
        existing = args.verify_existing.resolve()
        import_manifest(existing, variants, cases, output, args.variants, args.cases)
        result = verify(output, existing)
    else:
        if args.native and (platform.system() != "Linux" or not args.cann):
            parser.error("--native requires Linux and an explicit --cann installation")
        manifest = {
            "schema_version": 1,
            "scope": "CANN_Ascend950_simulator_only",
            "warmup": args.warmup,
            "active": args.active,
            "profile_enabled": args.profile_enabled,
            "variants": snapshot_sources(output, selected_variants),
            "cases": prepare_cases(output, selected_cases, args.warmup, args.active, args.profile_enabled),
            "bundle_manifest_sha256": sha256(ROOT / "variants.json"),
            "runner_sha256": sha256(Path(__file__)),
            "case_file_sha256": sha256(args.case_file),
            "harness_sha256": {p.name: sha256(p) for p in (ROOT / "harness").iterdir() if p.is_file()},
            "oracle_sha256": sha256(ROOT / "oracle/reference.py"),
        }
        write_json(output / "manifest.json", manifest)
        if args.prepare_only:
            print(f"prepared {len(selected_variants) * len(selected_cases)} cells: {output}")
            return 0
        if args.docker:
            image = subprocess.check_output(
                ["docker", "image", "inspect", args.image, "--format", "{{json .RepoDigests}}"], text=True
            )
            manifest["container_image"] = args.image
            manifest["container_repo_digests"] = json.loads(image)
            write_json(output / "manifest.json", manifest)
            cann = args.cann or Path("/usr/local/Ascend/cann-9.1.0")
            subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{ROOT}:/bundle:ro",
                    "-v",
                    f"{output}:/output",
                    "-w",
                    "/bundle",
                    args.image,
                    "python3",
                    "/bundle/tools/compare_kernels.py",
                    "--inside-container",
                    "--output",
                    "/output",
                    "--cann",
                    str(cann),
                    "--jobs",
                    str(args.jobs),
                    "--timeout",
                    str(args.timeout),
                ],
                check=True,
            )
        else:
            execute(output, args.cann.resolve(), args.jobs, args.timeout, False)
        result = verify(output)
    print(f"{result['status'].upper()}: report={output / 'report.md'}")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
