# SPDX-License-Identifier: Apache-2.0
"""Bound variant resource lifetime without changing the timed device graph."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from kvip.diagnostics.console import emit, saved_failure
from kvip.diagnostics.replay_measure import verify_case_equivalence


def command(args, name, output):
    values = dict(
        snapshot=args.snapshot, build_dir=args.build_dir, transport=args.transport,
        mode=args.mode, threshold_policy=args.threshold_policy, study=args.study,
        upstream_root=args.upstream_root, upstream_revision=args.upstream_revision,
        kvaf_server=args.kvaf_server, device=args.device, soc=args.soc,
        warmup=args.warmup, iterations=args.iterations, diagnostic_replays=args.diagnostic_replays,
        cold_windows=args.cold_windows, output=output,
    )
    result = [sys.executable, "-B", "-X", "faulthandler", "-m", "kvip.diagnostics.replay"]
    for key, value in values.items():
        if value is not None:
            result.extend(("--" + key.replace("_", "-"), str(value)))
    # Never forward isolate-variants: each child owns exactly one device context.
    result.extend(("--scan-variants", name, "--quiet"))
    return result


def run(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f".{args.output.stem}-variants.", dir=args.output.parent))
    cases, memory, name = {}, [], "unassigned"
    child = {}
    log = directory / "startup.log"
    try:
        for name in args.scan_variants:
            child = {}
            output, log = directory / f"{name}.json", directory / f"{name}.log"
            emit(f"ISOLATED_CASE_BEGIN {name}")
            with log.open("w") as stream:
                process = subprocess.run(command(args, name, output), stdout=stream, stderr=subprocess.STDOUT)
            if output.is_file():
                child = json.loads(output.read_text())
            if process.returncode or child.get("status") != "pass":
                saved_failure(output, log, returncode=process.returncode)
                raise RuntimeError(f"{name} failed in isolated process; retained evidence: {directory}")
            if child.get("execution_order") != [name] or set(child.get("cases", {})) != {name}:
                raise ValueError(f"Unexpected isolated variant output: {name}")
            cases[name] = child["cases"][name]
            memory.extend(child.get("memory_samples", []))
            timing = cases[name]["timing"]
            emit(f"ISOLATED_CASE_PASS {name} mean_us={timing['mean_us']:.3f} p50_us={timing['p50_us']:.3f}")
        verify_case_equivalence(cases, same_threshold=True)
        report = dict(
            status="pass", transport=args.transport, mode=args.mode, study=args.study,
            cases=cases, execution_order=list(cases), memory_samples=memory,
            process_model="one_fresh_process_per_variant_sequential",
            allocation_note="Identical snapshot restored per variant; process/runtime initialization is outside device timing. Physical allocations and queue endpoints can differ between processes.",
        )
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
    except BaseException as error:
        report = dict(child) if child.get("status") == "failed" else {}
        report.update(status="failed", completed_cases=cases, isolated_directory=str(directory))
        report.setdefault("progress", dict(variant=name, phase="isolated_variant"))
        report.setdefault("error", dict(type=type(error).__name__, message=str(error)))
        report.setdefault("native_log", str(log))
        report.setdefault("memory_samples", memory)
        if not args.output.exists():
            with args.output.open("x") as stream:
                json.dump(report, stream, indent=2)
        emit(f"ISOLATED_REPLAY_FAILED {error}\nFAILURE_JSON {args.output}", file=sys.stderr)
        raise SystemExit(130 if isinstance(error, KeyboardInterrupt) else 70) from error
    shutil.rmtree(directory)
    emit(f"ISOLATED_VARIANTS_PASS cases={len(cases)}")
