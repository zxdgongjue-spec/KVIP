# SPDX-License-Identifier: Apache-2.0
"""Keep test status visible even when a native dependency terminates Python."""

import json
import os
import signal
import subprocess
import time
from collections import deque
from pathlib import Path


def progress(run, stage):
    (run / "stage.log").write_text(stage + "\n")
    print(stage, flush=True)


def log_tail(path, lines=12):
    # Bound reads even if native transport logging grows very large.
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 8192))
        return stream.read().decode(errors="replace").splitlines()[-lines:]


def compiler_error(path, before=4, after=10):
    """Keep the first compiler diagnostic instead of only recursive make exits."""
    context = deque(maxlen=before)
    with path.open(errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            entry = f"{number}: {line.rstrip()}"
            if "error:" in line or "undefined reference" in line or "CMake Error" in line:
                result = [*context, entry]
                for offset, following in enumerate(stream, 1):
                    result.append(f"{number + offset}: {following.rstrip()}")
                    if offset >= after:
                        break
                return result
            context.append(entry)
    return []


def supervise(command, run, heartbeat_seconds=20, timeout_seconds=300):
    """Only a checked worker completion marker plus exit code zero means PASS."""
    run = Path(run)
    summary = run / "summary.log"
    started = time.monotonic()

    def status(message):
        text = f"{message}\nResults: {run}\n"
        summary.write_text(text)
        print(message, flush=True)

    status("RUNNING: starting worker")
    print(f"Summary: {summary}", flush=True)
    interrupted = False
    timed_out = False
    with (run / "worker.log").open("w") as output:
        try:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as error:
            status(f"FAIL: cannot start worker: {error}")
            return 1
        try:
            while True:
                try:
                    code = child.wait(timeout=heartbeat_seconds)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() - started >= timeout_seconds:
                        timed_out = True
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            code = child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            code = child.wait()
                        break
                    stage = (
                        (run / "stage.log").read_text().strip() if (run / "stage.log").exists() else "imports/startup"
                    )
                    logs = [p for p in run.rglob("*.log") if p.name not in ("summary.log", "stage.log")]
                    latest = max(logs, key=lambda p: p.stat().st_mtime, default=None)
                    evidence = f"; latest={latest.relative_to(run)} bytes={latest.stat().st_size}" if latest else ""
                    status(f"RUNNING +{time.monotonic() - started:.0f}s: {stage}{evidence}")
        except KeyboardInterrupt:
            interrupted = True
            # Also stop compiler children, without killing unrelated processes.
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
    marker = run / "worker-result.json"
    try:
        result = json.loads(marker.read_text()) if marker.exists() else {}
    except (OSError, ValueError):
        result = {}
    passed = code == 0 and result.get("exit_code") == 0 and not interrupted
    detail = f"exit_code={code}"
    if timed_out:
        detail += f"; timeout={timeout_seconds}s"
    if code < 0:
        detail += f" signal={signal.Signals(-code).name}"
    if not marker.exists():
        detail += "; worker did not reach checked completion"
    stage = (run / "stage.log").read_text().strip() if (run / "stage.log").exists() else "startup"
    lines = [f"{'PASS' if passed else 'FAIL'}: {detail}; elapsed={time.monotonic() - started:.1f}s; stage={stage}"]
    cases = run / "cases.log"
    if cases.exists():
        lines.extend(cases.read_text().splitlines())
    if not passed:
        for audit in run.rglob("streaming_fetch_kv_mismatch.json"):
            try:
                data = json.loads(audit.read_text())
                compact = {key: data[key] for key in ("batch", "mismatch_positions", "posts")}
                compact["examples"] = data.get("examples", [])[:3]
                lines.append(f"STREAMING FETCH KV AUDIT: {json.dumps(compact)}; details={audit}")
            except (OSError, ValueError, KeyError) as error:
                lines.append(f"Could not read streaming fetch audit {audit}: {error}")
        # At most four short tails, including exceptions before native logging.
        logs = [p for p in run.rglob("*.log") if p.name in ("native.log", "error.log", "build.log", "configure.log")]
        logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for path in logs:
            if path.name in ("build.log", "configure.log"):
                diagnostic = compiler_error(path)
                if diagnostic:
                    lines.append(f"--- {path.relative_to(run)} (first compiler error) ---")
                    lines.extend(diagnostic)
                    break
        for path in [run / "worker.log", *logs[:3]]:
            lines.append(f"--- {path.relative_to(run)} (tail) ---")
            lines.extend(log_tail(path))
    lines.append(f"Full logs: {run}")
    summary.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0 if passed else 1
