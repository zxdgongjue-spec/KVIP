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


class NativeLog:
    """Restore fds before an exception reaches the CLI's retained-owner _exit."""

    def __init__(self, enabled):
        self.enabled = enabled
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
                failure_tail(self.path, message=f"{kind.__name__}: {error}")
        finally:
            self.stdout.close()
            self.stderr.close()
        return False
