# SPDX-License-Identifier: Apache-2.0
"""CLI-only native output capture and saved-result summaries; never timed."""

from collections import deque
import ctypes
import os
from pathlib import Path
import sys
import tempfile
import traceback


_ACTIVE = None


def emit(message, *, file=None, end="\n", flush=True):
    """Keep explicit progress visible while native fd 1/2 chatter is spooled."""
    target = file or sys.stdout
    if _ACTIVE is not None:
        target = _ACTIVE.stderr if target is sys.stderr else _ACTIVE.stdout
    print(message, file=target, end=end, flush=flush)


def flush_output():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    # Native printf may be buffered independently of Python's text wrappers.
    try:
        ctypes.CDLL(None).fflush(None)
    except (OSError, AttributeError):
        pass


def failure_tail(path, *, message, lines=80):
    """Read bounded memory, retain the complete spool, and expose its location."""
    try:
        with Path(path).open(errors="replace") as source:
            tail = "".join(deque(source, maxlen=lines))
    except OSError as error:
        tail = f"Could not read retained log: {error}\n"
    emit(f"FAILED: {message}\nFailure log: {Path(path).resolve()}\n--- last {lines} log lines ---", file=sys.stderr)
    if tail:
        emit(tail, file=sys.stderr, end="" if tail.endswith("\n") else "\n")


def print_replay_failure(report, *, result_path, log_path=None):
    """Report the failed operation first, without repeating per-core errors."""
    where = report.get("progress", {})
    error = report.get("error", {})
    message = next((line.strip() for line in error.get("message", "").splitlines() if line.strip()), "unknown")
    emit(
        f"REPLAY_FAILED variant={where.get('variant', 'unassigned')} phase={where.get('phase', 'unknown')} "
        f"step={where.get('step')} {error.get('type', 'Error')}: {message[:400]}",
        file=sys.stderr,
    )
    # The native builder already selects the first compiler error and context.
    # Preserve it in concise mode instead of hiding it behind the first line.
    diagnostic = error.get("message", "").partition("--- build diagnostic ---")[2]
    if diagnostic:
        emit("BUILD_DIAGNOSTIC_BEGIN", file=sys.stderr)
        for line in diagnostic.strip().splitlines()[:35]:
            emit(line[:1200], file=sys.stderr)
        emit("BUILD_DIAGNOSTIC_END", file=sys.stderr)
    for name, case in report.get("completed_cases", {}).items():
        timing = case.get("timing", {})
        emit(
            f"COMPLETED {name} status={case.get('status')} p50_us={timing.get('p50_us')} "
            f"mean_us={timing.get('mean_us')}",
            file=sys.stderr,
        )
    buffers = report.get("buffers", {})
    if buffers:
        emit(
            "BUFFERS "
            + " ".join(
                f"{name}={buffers[name]['address']}+{buffers[name]['bytes']}B"
                for name in ("output", "scoreGM", "prefix", "keys", "context", "rows")
                if name in buffers
            ),
            file=sys.stderr,
        )
    # The full first exception, traceback and native diagnostics remain saved.
    for marker in ("Page fault info:", "SIMT accesses an invalid GM address"):
        line = next((line.strip() for line in error.get("message", "").splitlines() if marker in line), None)
        if line:
            emit(line[:700], file=sys.stderr)
    emit(f"FAILURE_JSON {Path(result_path).resolve()}", file=sys.stderr)
    if log_path:
        emit(f"FAILURE_LOG {Path(log_path).resolve()}", file=sys.stderr)


class NativeLog:
    """Restore fds before an exception reaches the CLI's retained-owner _exit."""

    def __init__(self, enabled, *, report_failure=True):
        self.enabled = enabled
        self.report_failure = report_failure
        self.failure_path = None
        self.path = None

    def __enter__(self):
        global _ACTIVE
        if not self.enabled:
            return self
        if _ACTIVE is not None:
            raise RuntimeError("Native CLI logging cannot be nested")
        flush_output()
        fd, path = tempfile.mkstemp(prefix="kvip-replay-", suffix=".log")
        self.path = Path(path)
        self.stdout = self.stderr = None
        try:
            self.stdout = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
            self.stderr = os.fdopen(os.dup(2), "w", encoding="utf-8", buffering=1)
            os.dup2(fd, 1)
            os.dup2(fd, 2)
        except BaseException:
            for target, original in ((1, self.stdout), (2, self.stderr)):
                if original is not None:
                    os.dup2(original.fileno(), target)
                    original.close()
            self.path.unlink(missing_ok=True)
            raise
        finally:
            os.close(fd)
        _ACTIVE = self
        emit(f"RUN_LOG {self.path.resolve()} (temporary; removed on success)")
        return self

    def __exit__(self, kind, error, tb):
        global _ACTIVE
        if not self.enabled:
            return False
        try:
            if error is not None:
                traceback.print_exception(kind, error, tb)
        finally:
            flush_output()
            os.dup2(self.stdout.fileno(), 1)
            os.dup2(self.stderr.fileno(), 2)
            _ACTIVE = None
        try:
            if error is None:
                self.path.unlink(missing_ok=True)
            else:
                self.failure_path = self.path
                if self.report_failure:
                    failure_tail(self.path, message=f"{kind.__name__}: {error}")
        finally:
            self.stdout.close()
            self.stderr.close()
        return False
