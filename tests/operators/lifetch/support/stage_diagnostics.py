"""Durable stage markers for isolating asynchronous NPU test failures."""

import json
import os
import time
from pathlib import Path


def record_stage(log_path, scenario, stage, event, **details):
    """Append and fsync one structured marker before proceeding."""
    entry = dict(details)
    entry.update(
        {
            "timestamp_monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "scenario": str(scenario),
            "stage": str(stage),
            "event": str(event),
        }
    )
    path = Path(log_path)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(entry, sort_keys=True, default=str) + "\\n")
        output.flush()
        os.fsync(output.fileno())
    return entry


def run_synchronized_stage(log_path, scenario, stage, submit, synchronize):
    """Log submit and sync separately; re-raise the original stage exception."""
    record_stage(log_path, scenario, stage, "begin")
    try:
        result = submit()
    except BaseException as error:
        record_stage(
            log_path,
            scenario,
            stage,
            "failed",
            phase="submit",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise

    record_stage(log_path, scenario, stage, "submitted")
    record_stage(log_path, scenario, stage, "sync_begin")
    try:
        synchronize()
    except BaseException as error:
        record_stage(
            log_path,
            scenario,
            stage,
            "failed",
            phase="synchronize",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise

    record_stage(log_path, scenario, stage, "completed")
    return result
