# SPDX-License-Identifier: Apache-2.0
"""Read-only replay snapshots. Compare memory counters only within the same boot and cgroup."""
import json
import os
import re
import sys
import time
from pathlib import Path


def _read(path):
    try:
        return path.read_text(errors="replace").strip()
    except OSError:
        return ""


def _mtime(path):
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _log_tail(path):
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, os.fstat(stream.fileno()).st_size - 65536))
            lines = stream.read().decode(errors="replace").splitlines()
    except OSError:
        return {}
    markers = re.compile(r"Terminated|\bKilled\b|out of memory|oom.kill|Traceback|"
                         r"received signal|timed out|FAILED rc=", re.I)
    keep = [i for i, line in enumerate(lines) if markers.search(line)][-3:]
    keep.extend(range(max(0, len(lines) - 3), len(lines)))
    return {"mtime_ns": _mtime(path), "lines": [lines[i][:400] for i in sorted(set(keep))]}


def _memory():
    groups = [s.split(":", 2) for s in _read(Path("/proc/self/cgroup")).splitlines()]
    for line in _read(Path("/proc/self/mountinfo")).splitlines():
        left, _, right = line.partition(" - ")
        mount, fs = left.split(), right.split()
        if len(mount) < 5 or len(fs) < 3:
            continue
        v2 = fs[0] == "cgroup2"
        if not v2 and not (fs[0] == "cgroup" and "memory" in fs[2].split(",")):
            continue
        root, target = [Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), s))
                        for s in mount[3:5]]
        for entry in groups:
            if len(entry) != 3:
                continue
            _, controllers, group = entry
            if (v2 and controllers) or (not v2 and "memory" not in controllers.split(",")):
                continue
            relative = Path(group)
            if ".." in relative.parts:
                continue
            directories = [target / group.lstrip("/")]
            if relative.is_relative_to(root):
                directories.insert(0, target / relative.relative_to(root))
            keys = ("memory.events.local", "memory.events", "memory.current", "memory.max") if v2 else (
                "memory.oom_control", "memory.failcnt", "memory.usage_in_bytes", "memory.limit_in_bytes")
            for directory in dict.fromkeys(directories):
                values = {k: v for k in keys if (v := _read(directory / k))}
                if values:
                    return {"version": 2 if v2 else 1, "group": group, "path": str(directory), **values}
    return {}


def _card_lock(path):
    info = {"path": str(path)}
    try:
        key = path.stat()
    except OSError:
        return info
    procs = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            fd = (proc / "fd/9").stat()
            if (fd.st_dev, fd.st_ino) != (key.st_dev, key.st_ino):
                continue
            pid = int(proc.name)
            claims = [s for s in _read(proc / "fdinfo/9").splitlines() if s.startswith("lock:")]
            procs.append({"pid": pid, "pgid": os.getpgid(pid), "sid": os.getsid(pid),
                          "comm": _read(proc / "comm"), "fdinfo_lock": claims,
                          "cgroup": _read(proc / "cgroup")})
        except OSError:
            continue
    info["visible_fd9"] = procs
    return info


def snapshot(root):
    """Return a JSON-serializable observation even when run.log was removed on success."""
    root = Path(root)
    boot = _read(Path("/proc/sys/kernel/random/boot_id"))
    card = _read(root / "physical_npu.txt")
    paths = [root / "run.log"]
    worker = max(root.glob("g*/bs*/block1/fused/worker.log"), key=_mtime, default=None)
    if worker:
        paths.extend([worker, worker.with_name("cpu.log")])
    logs = {str(p.relative_to(root)): v for p in paths if (v := _log_tail(p))}
    result = {"time_ns": time.time_ns(), "root": str(root), "host": os.uname().nodename,
              "boot_id": boot, "archived_boot_id": _read(root / "boot_id.txt"),
              "physical_npu": card, "logs": logs, "memory_cgroup": _memory()}
    if boot and card in ("3", "4"):
        lock = root.parent.parent / "fused-scan-sources" / f"largebs-{boot}-npu{card}.lock"
        result["card_lock"] = _card_lock(lock)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: replay_interrupt_evidence.py RESULT_ROOT")
    print(json.dumps(snapshot(sys.argv[1]), ensure_ascii=False))
