# SPDX-License-Identifier: Apache-2.0
"""CLI-only native output capture and saved-result summaries; never timed."""

from collections import deque
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
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
    for sample in report.get("memory_samples", [])[-12:]:
        label = f"MEMORY {sample['checkpoint']} variant={sample.get('variant')}"
        if "free_bytes" in sample:
            emit(label + " " + " ".join(
                f"{key}_GiB={sample[key + '_bytes'] / 2**30:.3f}"
                for key in ("free", "allocated", "reserved")
            ), file=sys.stderr)
        else:
            emit(label + f" unavailable={sample.get('error')}", file=sys.stderr)
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


def saved_failure(result_path, outer_log, *, returncode=None):
    """Inspect a saved exit, including native exits that bypass Python handlers.

    Quiet replay writes its RUN_LOG path before importing the NPU runtime.
    Follow that recorded path instead of guessing a build hash or displaying
    only the two progress lines in the outer shell log. This never reruns work.
    """
    emit("FAILURE_DETAIL_BEGIN", file=sys.stderr)
    if returncode is not None:
        emit(f"CHILD_EXIT_CODE={returncode}", file=sys.stderr)
    native_logs = []
    try:
        report = json.loads(Path(result_path).read_text())
    except (OSError, ValueError) as error:
        emit(f"FAILURE_JSON_UNAVAILABLE {result_path}: {error}", file=sys.stderr)
    else:
        if isinstance(report, dict) and report.get("status") == "failed":
            print_replay_failure(report, result_path=result_path, log_path=report.get("native_log"))
            if report.get("native_log"):
                native_logs.append(Path(report["native_log"]))
        else:
            emit(f"RESULT_STATUS={report.get('status') if isinstance(report, dict) else 'invalid'}", file=sys.stderr)
    outer_tail = deque(maxlen=12)
    try:
        with Path(outer_log).open(errors="replace") as stream:
            for line in stream:
                outer_tail.append(line)
                match = re.match(r"^RUN_LOG (.+) \(temporary; removed on success\)\s*$", line)
                if match:
                    native_logs.append(Path(match.group(1)))
    except OSError as error:
        emit(f"OUTER_LOG_UNAVAILABLE {outer_log}: {error}", file=sys.stderr)
    if outer_tail:
        emit(f"OUTER_LOG_TAIL {outer_log}\n" + "".join(outer_tail), file=sys.stderr, end="")
    for path in dict.fromkeys(native_logs):
        emit(f"NATIVE_LOG={path}", file=sys.stderr)
        tail, before, first = deque(maxlen=20), deque(maxlen=5), []
        remaining = 0
        try:
            with path.open(errors="replace") as stream:
                for line in stream:
                    line = line.rstrip("\n")[:1200]
                    tail.append(line)
                    if not first and re.search(r"error|fatal|traceback|exception|aborted|segmentation|killed|terminate called|bad_alloc", line, re.I):
                        first = [*before, line]
                        remaining = 15
                    elif remaining:
                        first.append(line)
                        remaining -= 1
                    before.append(line)
        except OSError as error:
            emit(f"NATIVE_LOG_UNAVAILABLE: {error}", file=sys.stderr)
            continue
        if first:
            emit("FIRST_NATIVE_ERROR_CONTEXT\n" + "\n".join(first), file=sys.stderr)
        emit("NATIVE_LOG_TAIL\n" + "\n".join(tail), file=sys.stderr)
    if not native_logs:
        emit("NO_RECORDED_NATIVE_LOG", file=sys.stderr)
    emit("FAILURE_DETAIL_END", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Read retained replay failure evidence; no NPU use")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--returncode", type=int)
    args = parser.parse_args()
    saved_failure(args.result, args.log, returncode=args.returncode)


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


if __name__ == "__main__":
    main()
