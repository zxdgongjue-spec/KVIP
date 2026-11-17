# SPDX-License-Identifier: Apache-2.0
"""Independent torch_npu system traces; never part of headline timing."""

import csv
from decimal import Decimal, InvalidOperation
import importlib
import json
import math
from pathlib import Path
import platform
import re

from kvip.diagnostics.console import emit

PROFILE_METRICS = (
    "available-pipeline",
    "none",
    "PipeUtilization",
    "ArithmeticUtilization",
    "Memory",
    "MemoryL0",
    "MemoryUB",
    "ResourceConflictRatio",
    "L2Cache",
    "MemoryAccess",
)


def profile_configurations(npu, metrics):
    """Reject unavailable metrics and silent experimental-config fallbacks."""
    api = npu.profiler
    experimental = getattr(api, "experimental_config", None)
    if experimental is None:
        experimental = importlib.import_module("torch_npu.profiler.experimental_config")
    supported = experimental.supported_ai_core_metrics()
    if "available-pipeline" in metrics:
        if list(metrics) != ["available-pipeline"]:
            raise ValueError("available-pipeline must be the only metric selector")
        wanted = ("PipeUtilization", "ResourceConflictRatio")
        chosen = [name for name in wanted
                  if getattr(api.AiCMetrics, name, None) is not None
                  and getattr(api.AiCMetrics, name) in supported]
        emit("PROFILE_METRICS selected=" + ",".join(["none", *chosen])
             + " unavailable=" + (",".join(name for name in wanted if name not in chosen) or "none"))
        metrics = ["none", *chosen]
    configurations = []
    for name in metrics:
        if name not in PROFILE_METRICS:
            raise ValueError(f"Unknown profiler metric: {name}")
        enum_name = "AiCoreNone" if name == "none" else name
        metric = getattr(api.AiCMetrics, enum_name, None)
        if metric is None or (name != "none" and metric not in supported):
            raise ValueError(f"Installed torch_npu does not support profiler metric {name}")
        # Some versions replace AiCoreNone with PipeUtilization at Level1.
        level = api.ProfilerLevel.Level0 if name == "none" else api.ProfilerLevel.Level1
        config = api._ExperimentalConfig(
            profiler_level=level,
            aic_metrics=metric,
            export_type=api.ExportType.Text,
            data_simplification=False,
        )
        exports = config.export_type
        if not isinstance(exports, (tuple, list)):
            exports = (exports,)
        if (
            getattr(config, "_aic_metrics", None) != metric
            or getattr(config, "_profiler_level", None) != level
            or getattr(config, "_data_simplification", None) is not False
            or api.ExportType.Text not in exports
        ):
            raise RuntimeError(f"torch_npu changed the requested {name}/Text/raw-data configuration")
        configurations.append((name, config))
    return configurations


def summarize_timeline(directory, artifacts):
    """Read one canonical device export, avoiding duplicate CANN exports.

    Chrome trace ts/dur values are microseconds (displayTimeUnit is a viewer
    preference). Intervals may overlap: neither kernel sums nor spans are an
    uninstrumented graph duration. Retain names/times to audit this distinction.
    """
    candidates = [row for row in artifacts["trace_files"] if row.get("device_events")]
    canonical = [row for row in candidates if Path(row["path"]).name == "trace_view.json"]
    if canonical:
        candidates = canonical
    if not candidates:
        return _csv_timeline(Path(directory), artifacts)
    if len(candidates) != 1:
        return dict(status="unavailable", reason="Expected one unambiguous device trace", candidates=candidates)
    path = directory / candidates[0]["path"]
    try:
        data = json.loads(path.read_text())
        events = data.get("traceEvents", []) if isinstance(data, dict) else data
        kernels = []
        for event in events:
            if not isinstance(event, dict) or event.get("ph") != "X":
                continue
            if not any(word in str(event.get("cat", "")).lower() for word in ("kernel", "aicore", "ai_core")):
                continue
            start, duration = float(event["ts"]), float(event["dur"])
            if not (math.isfinite(start) and math.isfinite(duration) and duration > 0):
                continue
            kernels.append(dict(name=str(event.get("name", "unknown")), start_us=start, duration_us=duration,
                                pid=event.get("pid"), tid=event.get("tid")))
    except (OSError, ValueError, TypeError, KeyError) as error:
        return dict(status="unavailable", reason=str(error), trace=str(path))
    if not kernels:
        return dict(status="unavailable", reason="No complete device kernel intervals", trace=str(path))
    return _timeline_summary(path, kernels, "chrome_trace")


def _csv_timeline(directory, artifacts):
    candidates = [row for row in artifacts.get("csv_files", [])
                  if Path(row["path"]).name == "kernel_details.csv"]
    canonical = [row for row in candidates if "ASCEND_PROFILER_OUTPUT" in Path(row["path"]).parts]
    candidates = canonical or candidates
    if len(candidates) != 1:
        return dict(status="unavailable", reason="Expected one unambiguous kernel_details.csv",
                    candidates=candidates)
    path = directory / candidates[0]["path"]
    try:
        kernels = []
        with path.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            required = {"Name", "Start Time(us)", "Duration(us)"}
            if not required <= set(reader.fieldnames or []):
                raise ValueError("Unrecognized kernel_details.csv columns")
            for row in reader:
                start, duration = Decimal(row["Start Time(us)"].strip()), Decimal(row["Duration(us)"].strip())
                if not (start.is_finite() and duration.is_finite() and duration > 0):
                    continue
                kernels.append(dict(name=row["Name"].strip(), start_us=start, duration_us=float(duration),
                                    device=row.get("Device_id", "").strip(),
                                    wait_us=row.get("Wait Time(us)", "").strip(),
                                    blocks=row.get("Block Num", "").strip()))
        if not kernels:
            raise ValueError("No complete device kernel intervals")
        origin = min(row["start_us"] for row in kernels)
        # Epoch microseconds can exceed 1e15. Subtract before float conversion;
        # otherwise sub-microsecond launch gaps lose precision.
        for row in kernels:
            row["start_us"] = float(row["start_us"] - origin)
    except (OSError, ValueError, TypeError, KeyError, InvalidOperation, csv.Error) as error:
        return dict(status="unavailable", reason=str(error), trace=str(path))
    result = _timeline_summary(path, kernels, "kernel_details.csv")
    result["origin_us"] = str(origin)
    return result


def _timeline_summary(path, kernels, source_format):

    def envelope(rows):
        if not rows:
            return None
        return dict(count=len(rows), sum_us=sum(row["duration_us"] for row in rows),
                    span_us=max(row["start_us"] + row["duration_us"] for row in rows)
                    - min(row["start_us"] for row in rows))

    return dict(status="available", trace=str(path), source_format=source_format,
                units="us", included_in_headline_timing=False,
                scope="Independent profiler replay; overlapping kernel intervals must not be added",
                device=envelope(kernels),
                init=envelope([row for row in kernels if "v9_init_kernel" in row["name"]]),
                li=envelope([row for row in kernels if "v9_li_kernel" in row["name"]]),
                streaming=envelope([row for row in kernels if "v9_streaming_resolve_update" in row["name"]]),
                kernels=kernels)


def inspect_artifacts(directory, metric):
    """Index exports without assuming identical CSV schemas across CANN versions.

    This is an availability check, not validation of hardware counter accuracy.
    Zero-valued counters are valid. Missing/unknown columns remain explicit.
    """
    files = sorted(path for path in directory.rglob("*") if path.is_file() and path.name != "profile_manifest.json")
    traces, tables = [], []
    tokens = {
        "PipeUtilization": ("ratio", "utilization"),
        "ArithmeticUtilization": ("ratio", "utilization"),
        "Memory": ("bandwidth", "read", "write"),
        "MemoryL0": ("l0",),
        "MemoryUB": ("ub_", "ub "),
        "ResourceConflictRatio": ("conflict",),
        "L2Cache": ("l2",),
        "MemoryAccess": ("read", "write", "memory"),
    }.get(metric, ())
    for path in files:
        relative = str(path.relative_to(directory))
        if path.suffix == ".json" and ("trace" in path.name.lower() or "timeline" in path.name.lower()):
            entry = dict(path=relative, bytes=path.stat().st_size, device_events=None)
            if path.stat().st_size <= 64 << 20:
                try:
                    data = json.loads(path.read_text())
                    events = data.get("traceEvents", []) if isinstance(data, dict) else data
                    entry["device_events"] = sum(
                        isinstance(event, dict)
                        and any(word in str(event.get("cat", "")).lower() for word in ("kernel", "aicore", "ai_core"))
                        for event in events
                    )
                except (OSError, ValueError, TypeError) as error:
                    entry["inspection_error"] = str(error)
            else:
                entry["inspection_note"] = "Trace exceeds 64 MiB; retained for external inspection"
            traces.append(entry)
        if path.suffix == ".csv":
            entry = dict(path=relative, counter_columns_with_values=[])
            try:
                with path.open(newline="", encoding="utf-8-sig") as source:
                    reader = csv.DictReader(source)
                    entry["kernel_timing_schema"] = (
                        path.name == "kernel_details.csv"
                        and {"Name", "Start Time(us)", "Duration(us)"} <= set(reader.fieldnames or ())
                    )
                    columns = [
                        name for name in reader.fieldnames or () if any(token in name.lower() for token in tokens)
                    ]
                    found = set()
                    for row in reader:
                        for name in columns:
                            if str(row.get(name, "")).strip().lower() not in (
                                "",
                                "n/a",
                                "na",
                                "none",
                                "nan",
                                "-",
                                "--",
                            ):
                                found.add(name)
                        if len(found) == len(columns):
                            break
                    entry["counter_columns_with_values"] = sorted(found)
            except (OSError, ValueError, csv.Error) as error:
                entry["inspection_error"] = str(error)
            tables.append(entry)
    device_events = (any(entry.get("device_events", 0) for entry in traces)
                     or any(entry.get("kernel_timing_schema", False) for entry in tables))
    counters = metric == "none" or any(entry["counter_columns_with_values"] for entry in tables)
    return dict(
        status="exports_present_review_required"
        if device_events and tables and counters
        else "incomplete_or_unrecognized_exports",
        trace_files=traces,
        csv_files=tables,
        raw_directories=[
            str(path.relative_to(directory))
            for path in sorted(directory.rglob("*"))
            if path.is_dir() and path.name in ("data", "FRAMEWORK", "PROFILER")
        ],
        hardware_metrics_validated=False,
        note="Column-name availability only; inspect raw data/CSV for device support and counter meaning. No peak-utilization claim.",
    )


def collect_system_profiles(
    *, torch, reset, replay, synchronize, verify, directory, case, metrics, metadata, retained, checkpoint, npu=None
):
    """Restore/fence outside collection; profile one existing graph per metric."""
    if npu is None:
        npu = importlib.import_module("torch_npu")
    configurations = profile_configurations(npu, metrics)
    case = re.sub(r"[^A-Za-z0-9_.-]", "_", case)
    root = Path(directory).resolve() / case
    if any((root / name).exists() for name, _ in configurations):
        raise FileExistsError(f"Profiler case/metric output already exists below {root}")
    results = []
    for step, (name, config) in enumerate(configurations):
        target = root / name
        target.mkdir(parents=True, exist_ok=False)
        manifest = target / "profile_manifest.json"
        report = dict(
            schema_version=1,
            status="prepared",
            metric=name,
            directory=str(target),
            included_in_headline_timing=False,
            graph_replays=1,
            cpu_range=f"KVIP_GRAPH_REPLAY_{case}",
            cpu_range_note="Host enqueue and completion fence range; not device execution duration",
            scope="One complete existing graph replay and its completion fence; excludes restore/import, capture, warmup, and correctness readback",
            versions=dict(
                python=platform.python_version(), torch=str(torch.__version__), torch_npu=str(npu.__version__)
            ),
            configuration=dict(
                activities=["CPU", "NPU"],
                profiler_level="Level0" if name == "none" else "Level1",
                aic_metrics="AiCoreNone" if name == "none" else name,
                export_type="Text",
                data_simplification=False,
                analyse_flag=True,
                async_mode=False,
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                with_modules=False,
            ),
            experiment=metadata,
        )

        def save():
            manifest.write_text(json.dumps(report, indent=2) + "\n")

        save()
        # Keep the path visible before entering native code, including fatal exits.
        emit(f"PROFILE_BEGIN {case} metric={name} directory={target} headline_timing=false")
        retained.setdefault("system_profile_manifests", []).append(str(manifest))
        profiler = npu.profiler.profile(
            activities=[npu.profiler.ProfilerActivity.CPU, npu.profiler.ProfilerActivity.NPU],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
            experimental_config=config,
            on_trace_ready=npu.profiler.tensorboard_trace_handler(
                str(target),
                worker_name=case + "_" + name,
                analyse_flag=True,
                async_mode=False,
            ),
        )
        retained.setdefault("system_profilers", []).append(profiler)
        started, stop_attempted = False, False
        try:
            checkpoint("profile_reset", step)
            reset()
            synchronize()
            checkpoint("profile_start", step)
            profiler.start()
            started = True
            checkpoint("profile_replay_and_sync", step)
            with torch.profiler.record_function(report["cpu_range"]):
                replay()
                synchronize()
            checkpoint("profile_stop_and_export", step)
            stop_attempted = True
            profiler.stop()  # Synchronous Text analysis/export callback.
            checkpoint("profile_verify", step)
            verify()
            report.update(status="captured", correctness="passed", artifacts=inspect_artifacts(target, name))
            if name == "none":
                report["timeline"] = summarize_timeline(target, report["artifacts"])
            save()
        except BaseException as error:
            report.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
            if started and not stop_attempted:
                try:
                    profiler.stop()
                except BaseException as stop_error:
                    report["stop_error"] = dict(type=type(stop_error).__name__, message=str(stop_error))
            try:
                save()
            except OSError:
                pass  # Preserve the first device/export failure.
            raise
        results.append(report)
        emit(f"PROFILE_CAPTURED {case} metric={name} artifacts={report['artifacts']['status']} manifest={manifest}")
    return dict(included_in_headline_timing=False, passes=results)
