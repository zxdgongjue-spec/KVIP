#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P0-A2.3 strict, sequential Active-K OFF/ON ABBA graph benchmark.

A=OFF, B=ON. Each run starts a separate worker, creates its own build,
and must pass full native/V9/SFA checks. This is a test harness, not a
production change. Results are paired P50 estimates, not significance tests.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

METRICS = (
    "round_init", "native_chain", "prototype_core",
    "prototype_with_update", "aiv_cache_update",
)
ORDER = ("OFF", "ON", "ON", "OFF")
FIXTURE_FILES = (
    "logical_scores.npy", "physical_to_logical.npy",
    "arg1.bin", "arg4.bin",
)
K_PROFILES = {
    "short": (32768, 32768, 32768, 32768),
    "mixed": (32768, 65536, 98304, 131072),
    "full": (131072, 131072, 131072, 131072),
}
DEFAULT_K = ",".join(map(str, K_PROFILES["mixed"]))


def profile_lengths(name):
    if name not in K_PROFILES:
        raise ValueError(f"unknown Active-K profile {name!r}")
    return list(K_PROFILES[name])


def mapping_clear_geometry(name, history=131072):
    lengths = profile_lengths(name)
    if len(lengths) != 4 or any(k > history or k < 8192 or k % 128 for k in lengths):
        raise ValueError("Active-K profile geometry invalid")
    base = history * len(lengths)
    active = sum(lengths)
    return {
        "actual_key_lengths": lengths,
        "mapping_tokens_off": base,
        "mapping_tokens_on": active,
        "mapping_bytes_saved_theoretical": 4 * (base - active),
        "mapping_clear_reduction_pct_theoretical": 100 * (base - active) / base,
        "mapping_copy_chunks_off": sum((history + 4095) // 4096 for _ in lengths),
        "mapping_copy_chunks_on": sum((k + 4095) // 4096 for k in lengths),
    }


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def base_command(args, run_output, enabled):
    script = Path(__file__).with_name("test_lifetch.py")
    command = [
        sys.executable, str(script),
        "--transport", "urma",
        "--device", str(args.device),
        "--kvaf-server", args.kvaf_server,
        "--kvaf-root", str(args.kvaf_root),
        "--cann", args.cann,
        "--soc", args.soc,
        "--batches", "4",
        "--history", "131072",
        "--resident", "8192",
        "--index-heads", "32",
        "--miss-percent", "10",
        "--prefetch-tokens", "512",
        "--intermediate-fetches", "8",
        "--native-union-fit",
        "--actual-key-lengths", ",".join(map(str, profile_lengths(args.k_profile))),
        "--warmup", str(args.warmup),
        "--iterations", str(args.iterations),
        "--trace-rounds", "0",
        "--output", str(run_output),
    ]
    if enabled:
        command.append("--active-k-map-clear")
    return command


def validate_completed_run(output, label, iterations, k_profile="mixed"):
    """Fail closed: a run without all checks, or with wrong build, is invalid."""
    manifests = list(Path(output).glob("*/case0-b4-prototype-graph/report.json"))
    if len(manifests) != 1:
        raise RuntimeError(f"{label}: expected one full-chain report, found {len(manifests)}")
    report_path = manifests[0]
    run_dir = report_path.parent.parent
    summary_path = run_dir / "summary.log"
    summary = summary_path.read_text() if summary_path.is_file() else ""
    if not summary.startswith("PASS: exit_code=0"):
        raise RuntimeError(f"{label}: full-chain summary did not pass: {summary_path}")
    r = json.loads(report_path.read_text())
    if r.get("status") != "pass" or r.get("stateful_rounds") != 3:
        raise RuntimeError(f"{label}: correctness/stateful verification not PASS")
    if r.get("actual_key_lengths") != profile_lengths(k_profile):
        raise RuntimeError(f"{label}: actual-K geometry changed")
    if r.get("dynamic_k_transition_report") is not None:
        raise RuntimeError(f"{label}: benchmark must exclude optional dynamic-transition checks")
    switch = str(r.get("build_options", {}).get("V9_ACTIVE_K_MAP_CLEAR", "missing"))
    expected = "1" if label == "ON" else "0"
    if switch != expected:
        raise RuntimeError(f"{label}: build switch is {switch}, expected {expected}")
    cmake = run_dir / "universal-build/CMakeCache.txt"
    if not cmake.is_file() or f"V9_ACTIVE_K_MAP_CLEAR:STRING={expected}" not in cmake.read_text():
        raise RuntimeError(f"{label}: CMakeCache does not confirm opt-in switch")
    admission = r.get("fixture", {}).get("actual_key_admission") or {}
    if admission.get("key_lengths") != profile_lengths(k_profile):
        raise RuntimeError(f"{label}: independent actual-K oracle missing")
    if len(admission.get("requests", [])) != 4:
        raise RuntimeError(f"{label}: per-request oracle missing")
    checks = r.get("native_misses_per_request")
    if checks is None or len(checks) != 4:
        raise RuntimeError(f"{label}: Native baseline misses missing")
    metrics = {}
    for name in METRICS:
        timing = r.get("timing", {}).get(name, {})
        if int(timing.get("samples", 0)) != iterations:
            raise RuntimeError(f"{label}: {name} captured wrong sample count")
        value = float(timing["p50_us"])
        if not 0 < value < 1000000:
            raise RuntimeError(f"{label}: invalid timing {name}={value}")
        metrics[name] = value
    fixture_dir = report_path.parent / "fixture"
    fingerprints = {
        name: sha256_file(fixture_dir / name)
        for name in FIXTURE_FILES
    }
    return {
        "switch": label,
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "fixture_sha256": fingerprints,
        "native_unique_misses": checks,
        "metrics_p50_us": metrics,
        "sfa_max_abs": float(r["sfa"]["max_abs"]),
    }


def paired_analysis(runs):
    if [r["switch"] for r in runs] != list(ORDER):
        raise ValueError("ABBA must use OFF,ON,ON,OFF")
    reference = runs[0]
    for run in runs[1:]:
        for key in ("fixture_sha256", "native_unique_misses"):
            if run[key] != reference[key]:
                raise ValueError(f"ABBA mismatch {key}: {run['switch']} vs baseline")
    pairs = []
    for off, on in ((runs[0], runs[1]), (runs[3], runs[2])):
        delta = {}
        for metric in METRICS:
            off_us = off["metrics_p50_us"][metric]
            on_us = on["metrics_p50_us"][metric]
            delta[metric] = {
                "off_us": off_us,
                "on_us": on_us,
                "saved_us": off_us - on_us,
                "saved_pct": 100 * (off_us - on_us) / off_us,
            }
        pairs.append(delta)
    summary = {}
    for metric in METRICS:
        values = [p[metric]["saved_pct"] for p in pairs]
        us = [p[metric]["saved_us"] for p in pairs]
        summary[metric] = {
            "paired_saved_pct": values,
            "paired_saved_us": us,
            "median_saved_pct": statistics.median(values),
            "median_saved_us": statistics.median(us),
            "both_pairs_positive": all(x > 0 for x in values),
        }
    return {
        "order": list(ORDER),
        "comparison": "A=OFF, B=ON, fresh process and build for each run",
        "fixture_sha256": reference["fixture_sha256"],
        "pairs": pairs,
        "metrics": summary,
        "runs": runs,
        "limitations": (
            "P50 from separate per-stage graphs; 2 local pairs are exploratory, "
            "not statistical significance. Device trace is a separate profiled pass. "
            "No real GLM-5.3 serving or persistent cross-K Index/KV proven."
        ),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kvaf-server", default="127.0.0.1:28999")
    p.add_argument("--kvaf-root", type=Path, required=True)
    p.add_argument("--cann", default="/usr/local/Ascend/cann-9.2.1")
    p.add_argument("--soc", default="Ascend950DT_9582")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument(
        "--k-profile", choices=tuple(K_PROFILES),
        default="mixed",
        help="short:75%%, mixed:37.5%%, full:0%% theoretical Mapping-clear reduction",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true", help="print commands without launching workers")
    args = p.parse_args(argv)
    if args.warmup < 1 or args.iterations < 5:
        p.error("ABBA requires at least 1 warmup and 5 timed iterations")
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        p.error(f"output directory must be fresh: {root}")
    commands = [
        base_command(args, root / f"{i+1}-{label.lower()}", label == "ON")
        for i, label in enumerate(ORDER)
    ]
    if args.dry_run:
        for label, cmd in zip(ORDER, commands):
            print(label, " ".join(cmd))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    (root / "commands.json").write_text(json.dumps(commands, indent=2))
    runs = []
    for i, (label, cmd) in enumerate(zip(ORDER, commands), 1):
        print(f"ABBA {i}/4 {label}: running complete correctness + graph timing", flush=True)
        log_path = root / f"{i}-{label.lower()}.log"
        with log_path.open("w") as log:
            finished = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
        if finished.returncode != 0:
            raise RuntimeError(f"ABBA {label} run {i} failed, see {log_path}")
        runs.append(validate_completed_run(
            root / f"{i}-{label.lower()}", label, args.iterations,
            args.k_profile,
        ))
        print(f"ABBA {i}/4 {label}: PASS", flush=True)
    result = paired_analysis(runs)
    result["k_profile"] = args.k_profile
    result["k_geometry"] = mapping_clear_geometry(args.k_profile)
    (root / "abba_report.json").write_text(json.dumps(result, indent=2))
    print(
        f"K_PROFILE={args.k_profile} "
        f"Mapping-clear theoretical reduction="
        f"{result['k_geometry']['mapping_clear_reduction_pct_theoretical']:.1f}%"
    )
    for metric, data in result["metrics"].items():
        print(
            f"{metric}: median_saved_us={data['median_saved_us']:.3f} "
            f"median_saved_pct={data['median_saved_pct']:.3f}% "
            f"both_pairs_positive={data['both_pairs_positive']}",
            flush=True,
        )
    print("ABBA_REPORT", root / "abba_report.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
