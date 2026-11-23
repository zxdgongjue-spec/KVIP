"""Resume artifact-checked LI rank model timing in bounded chunks inside the CANN container."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

IN_BYTES = 7184 * 4
OUT_BYTES = 72 * 4
GROUPS = (10, 4, 17, 7, 2)
LINE = re.compile(r"^PASS row=(\d+) pass=(\d+) variant=(\d+) count=(\d+) ticks=(\d+)$", re.M)
CGROUP = Path("/sys/fs/cgroup")


def json_read(path):
    return json.loads(path.read_text())


def atomic_text(path, data):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(data)
    temp.replace(path)


def checksum(path):
    with path.open("rb") as f:
        digest = hashlib.sha256()
        for block in iter(lambda: f.read(1024**2), b""):
            digest.update(block)
        return digest.hexdigest()


def load_cases(samples, g):
    prefix = samples / f"g{g}"
    records, expected = (prefix.with_suffix(s).read_bytes() for s in (".bin", ".want"))
    meta = json_read(prefix.with_suffix(".json"))
    if len(records) != len(meta) * IN_BYTES or len(expected) != len(meta) * OUT_BYTES or not meta:
        raise ValueError(f"Invalid G{g} input sizes")
    if len({(row["window"], row["row"], row["q"]) for row in meta}) != len(meta):
        raise ValueError(f"Duplicate window, row and Q in G{g}")
    for i, row in enumerate(meta):
        if row["group"] != g or row["posted"] != struct.unpack_from("<I", expected, i * OUT_BYTES)[0]:
            raise ValueError(f"G{g} index {i} has inconsistent identity or expected output")
    return meta, records, expected


def sample_bytes(data, i):
    _, raw, want = data
    return raw[i * IN_BYTES : (i + 1) * IN_BYTES], want[i * OUT_BYTES : (i + 1) * OUT_BYTES]


def measurements(log, meta, indices, variants, abba=False):
    found = {}
    for match in LINE.finditer(log):
        k, run, v, n, tick = map(int, match.groups())
        if k >= len(indices) or v not in variants or n != meta[indices[k]]["posted"] or tick <= 0:
            raise ValueError(f"Invalid output line: {match[0]}")
        order = (0, 1, 0, 1, 1, 0) if abba else (0, 1)
        if run >= len(order) or order[run] != v:
            raise ValueError(f"Invalid measurement order: {match[0]}")
        key = (indices[k], v, run)
        if key in found:
            raise ValueError(f"Duplicate output line: {match[0]}")
        found[key] = tick
    return found


def get_proc(pid):
    try:
        parts = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        cmd = [os.fsdecode(x) for x in parts if x]
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if not cmd or state[0] == "Z":
            return None
        return cmd, state[0], state[19], Path(f"/proc/{pid}/exe").resolve()
    except (OSError, IndexError):
        return None


def external(args, groups, ledger, host):
    """Register a live legacy run once, then re-import complete lines while it finishes."""
    held = {}
    previous = {
        int(item["group"]): f"{item['group']}:{item['pid']}:{item['log']}"
        for path in args.out.glob("external-g*.json")
        for item in [json_read(path)]
    }
    for mapping in args.existing:
        g = int(mapping.split(":", 1)[0])
        if g in previous and previous[g] != mapping:
            raise ValueError(f"Existing G{g} legacy registration disagrees")
        previous[g] = mapping
    for mapping in previous.values():
        g, pid, filename = mapping.split(":", 2)
        g, pid = int(g), int(pid)
        if g not in groups:
            continue
        log = Path(filename).resolve()
        registration = args.out / f"external-g{g}.json"
        proc = get_proc(pid)
        old = json_read(registration) if registration.exists() else None
        if old is None:
            if proc is None:
                raise ValueError(f"Register G{g} while its legacy process is still identifiable")
            cmd, state, birth, exe = proc
            if (
                len(cmd) != 3
                or cmd[2] != "--once"
                or Path(cmd[1]).resolve() != args.samples / f"g{g}"
                or checksum(exe) != host["previous_executable_sha256"]
            ):
                raise ValueError(f"Unexpected legacy G{g} process {pid}")
            old = dict(group=g, pid=pid, start=birth, log=str(log), kernel=host["device_sha256"])
            atomic_text(registration, json.dumps(old))
        if old["pid"] != pid or old["log"] != str(log) or old["kernel"] != host["device_sha256"]:
            raise ValueError(f"Existing G{g} receipt does not match")
        rows = groups[g][0]
        values = measurements(log.read_text(errors="replace"), rows, list(range(len(rows))), {0, 1})
        for (i, v, run), ticks in values.items():
            ledger[g, i, v] = (ticks, "legacy-full", i == 0)
        if proc and proc[2] == old["start"]:
            held[g] = (pid, proc[1])
        if "Mismatch for input" in log.read_text(errors="replace"):
            raise ValueError(f"Legacy correctness failure in {log}")
    return held


def import_representatives(args, groups, ledger):
    if not args.prior_samples:
        return
    for g, data in groups.items():
        samples = load_cases(args.prior_samples, g)
        meta = data[0]
        index = {(m["row"], m["q"], m["window"]): i for i, m in enumerate(meta)}
        ids = [index[m["row"], m["q"], m["window"]] for m in samples[0]]
        for k, i in enumerate(ids):
            if sample_bytes(samples, k) != sample_bytes(data, i):
                raise ValueError(f"Prior G{g}/{k} does not contain this exact input and oracle")
        path = args.prior_logs / f"model-g{g}.log"
        text = path.read_text(errors="replace")
        matches = measurements(text, meta, ids, {0, 1}, True)
        if len(matches) != 6 * len(ids) or "Mismatch for input" in text:
            raise ValueError(f"Incomplete prior ABBA: {path}")
        for k, i in enumerate(ids):
            for v, passes in enumerate(((2, 5), (3, 4))):
                ledger[g, i, v] = (mean(matches[i, v, p] for p in passes), "legacy-warm-abba", k == 0)


def worker(directory, lease):
    job = json_read(directory / "job.json")
    if job.get("idle_cpu"):
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    env = os.environ.copy()
    env["CAMODEL_LOG_PATH"] = str(directory)
    env["CAMODEL_CONFIG_PATH"] = job["config"]
    with (directory / "run.log").open("w") as log:
        rc = subprocess.call(
            [job["runner"], str(directory / "input"), job["mode"]],
            cwd=directory,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            pass_fds=(lease,),
        )
    atomic_text(directory / "exit.json", json.dumps(dict(rc=rc)))
    return 0


def owned(directory):
    with (directory / "lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def read_jobs(args, groups, ledger):
    running = set()
    retries = {}
    jobs = list(sorted((args.out / "jobs").glob("*/job.json")))
    for name in jobs:
        folder = name.parent
        job = json_read(name)
        g, ids, variants = job["group"], job["indices"], job["variants"]
        if g not in groups:
            continue
        meta = groups[g][0]
        log = folder / "run.log"
        text = log.read_text(errors="replace") if log.exists() else ""
        allkeys = {(g, i, v) for i in ids for v in variants}
        if "Mismatch for input" in text:
            raise ValueError(f"Correctness failure: {log}")
        readings = measurements(text, meta, ids, set(variants))
        for (i, v, _), ticks in readings.items():
            key = g, i, v
            if key in ledger:
                raise ValueError(f"Repeated scheduled measurement {key}: {log}")
            ledger[key] = (ticks, folder.name, i == ids[0])
        if owned(folder):
            running |= allkeys
            continue
        status = folder / "exit.json"
        if status.exists() and json_read(status)["rc"] == 0 and len(readings) != len(allkeys):
            raise ValueError(f"Runner reported success with missing output: {log}")
        for key in allkeys:
            retries[key] = retries.get(key, 0) + 1
    return running, retries


def cpu_usage():
    stats = (CGROUP / "cpu.stat").read_text().splitlines()
    return int(next(line.split()[1] for line in stats if line.startswith("usage_usec ")))


def allowed(args, last):
    now = time.monotonic()
    used = cpu_usage()
    previous, previous_used = last
    available = len(os.sched_getaffinity(0))
    quota, period = (CGROUP / "cpu.max").read_text().split()
    if quota != "max":
        available = min(available, int(quota) / int(period))
    free_cpu = available - (used - previous_used) / max((now - previous) * 1e6, 1)
    limit = (CGROUP / "memory.max").read_text().strip()
    if limit == "max":
        available_kb = next(
            s.split()[1]
            for s in Path("/proc/meminfo").read_text().splitlines()
            if s.startswith("MemAvailable:")
        )
        spare_mib = int(available_kb) / 1024
    else:
        spare_mib = (int(limit) - int((CGROUP / "memory.current").read_text())) / 1024**2
    protected = False
    if args.wait_executable_prefix:
        for pidfile in Path("/proc").glob("[0-9]*/exe"):
            try:
                if any(str(pidfile.readlink()).startswith(x) for x in args.wait_executable_prefix):
                    protected = True
                    break
            except (FileNotFoundError, PermissionError):
                continue
    return (
        (
            not protected
            and free_cpu >= args.reserve_cores + args.job_cores
            and spare_mib >= args.reserve_mib + args.job_mib
        ),
        (now, used),
        free_cpu,
        spare_mib,
    )


def next_chunk(args, groups, missing, held):
    for g in args.groups:
        if g in held:
            continue

        def variants_at(i):
            return tuple(v for v in (0, 1) if (g, i, v) in missing)

        for first in range(len(groups[g][0])):
            variants = variants_at(first)
            if not variants:
                continue
            end = first + 1
            while end < min(len(groups[g][0]), first + args.chunk) and variants_at(end) == variants:
                end += 1
            return g, list(range(first, end)), list(variants)
    return None


def start_job(args, groups, group, ids, variants):
    root = args.out / "jobs"
    count = sum(1 for _ in root.glob("*/job.json"))
    folder = root / f"{count + 1:05d}-g{group}-{ids[0]}-{ids[-1]}"
    folder.mkdir()
    for position, suffix in enumerate((".bin", ".want")):
        (folder / ("input" + suffix)).write_bytes(
            b"".join(sample_bytes(groups[group], i)[position] for i in ids)
        )
    mode = "--once" if variants == [0, 1] else ("--only-new" if variants == [1] else "--only-old")
    job = dict(
        group=group,
        indices=ids,
        variants=variants,
        runner=str(args.host / "probe"),
        mode=mode,
        config=str(args.config),
        idle_cpu=args.idle_cpu,
    )
    atomic_text(folder / "job.json", json.dumps(job))
    # The open-file-description lock survives the scheduler and is released on worker exit.
    with (folder / "lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proc = subprocess.Popen(
            [sys.executable, __file__, "_worker", str(folder), str(lock.fileno())],
            pass_fds=(lock.fileno(),),
            start_new_session=True,
        )
    print(f"START {folder.name} variants={variants} pid={proc.pid}", flush=True)
    return proc


def save_ledger(out, groups, ledger):
    prefix = out / "per-query.csv"
    temp = prefix.with_suffix(".tmp")
    with temp.open("w", newline="") as f:
        rows = csv.writer(f)
        rows.writerow(
            ["group", "index", "window", "row", "q", "variant", "ticks", "source", "first_in_launch"]
        )
        for g, packed in groups.items():
            for i, m in enumerate(packed[0]):
                for v in (0, 1):
                    if (g, i, v) in ledger:
                        ticks, source, first = ledger[g, i, v]
                        rows.writerow((g, i, m["window"], m["row"], m["q"], v, ticks, source, int(first)))
    temp.replace(prefix)


def schedule(args):
    for key in ("samples", "out", "host", "config", "prior_samples", "prior_logs"):
        path = getattr(args, key)
        if path:
            setattr(args, key, path.resolve())
    prior_record = args.out / "prior.json"
    if prior_record.exists():
        prior = json_read(prior_record)
        for name in ("prior_samples", "prior_logs"):
            recorded = Path(prior[name])
            selected = getattr(args, name)
            if selected and selected != recorded:
                raise ValueError("Prior sample source changed")
            setattr(args, name, recorded)
    if bool(args.prior_logs) != bool(args.prior_samples):
        raise ValueError("Provide both prior ABBA paths")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "jobs").mkdir(exist_ok=True)
    if not (args.config / "config.json").is_file():
        raise ValueError("Select the same CANN model configuration")
    groups = {g: load_cases(args.samples, g) for g in args.groups}
    host = json_read(args.host / "kernel.json")
    if (
        checksum(args.host / "probe") != host["runner_sha256"]
        or checksum(Path(host["device_archive"])) != host["device_sha256"]
    ):
        raise ValueError("Compiled host/device identity changed")
    if args.prior_samples:
        selected = json_read(args.config / "config.json")
        archive_cfg = json_read(args.prior_logs / "config.json")
        for configuration in (selected, archive_cfg):
            configuration.get("LOG", {}).pop("core_enable_mask", None)
        if selected != archive_cfg:
            raise ValueError("The prior ABBA model settings differ beyond the trace log mask")
        prior = dict(
            prior_samples=str(args.prior_samples),
            prior_logs=str(args.prior_logs),
            logs={str(g): checksum(args.prior_logs / f"model-g{g}.log") for g in args.groups},
        )
        if prior_record.exists() and json_read(prior_record) != prior:
            raise ValueError("Prior ABBA provenance changed")
        atomic_text(prior_record, json.dumps(prior))
    fingerprint = dict(
        device=host["device_sha256"],
        runner=host["runner_sha256"],
        config=checksum(args.config / "config.json"),
        inputs={
            str(g): [
                checksum(args.samples / f"g{g}.json"),
                *[hashlib.sha256(data).hexdigest() for data in packed[1:]],
            ]
            for g, packed in groups.items()
        },
    )
    manifest = args.out / "inputs.json"
    if manifest.exists() and json_read(manifest) != fingerprint:
        raise ValueError("Output directory was used for different fixtures, kernel, host or model config")
    atomic_text(manifest, json.dumps(fingerprint, indent=2))
    historical = {}
    import_representatives(args, groups, historical)
    keys = {(g, i, v) for g, pack in groups.items() for i in range(len(pack[0])) for v in (0, 1)}
    last = time.monotonic(), cpu_usage()
    clear = 0
    handles = []
    while True:
        handles = [p for p in handles if p.poll() is None]
        ledger = historical.copy()
        held = external(args, groups, ledger, host)
        running, retries = read_jobs(args, groups, ledger)
        missing = keys - ledger.keys() - running
        save_ledger(args.out, groups, ledger)
        if args.plan or len(ledger) == len(keys):
            print(
                json.dumps(
                    dict(
                        checked=len(ledger),
                        total=len(keys),
                        in_flight=len(running),
                        pending=len(missing),
                        external=held,
                        next_chunk=next_chunk(args, groups, missing, held),
                    )
                ),
                flush=True,
            )
            return
        exhausted = [key for key in missing if retries.get(key, 0) >= args.attempts]
        if exhausted:
            raise RuntimeError(f"More than {args.attempts} attempts for {exhausted[:5]}; see logs")
        time.sleep(args.poll)
        room, last, free_cpu, free_mib = allowed(args, last)
        clear = clear + 1 if room else 0
        active_external = sum(state not in ("T", "t") for _, state in held.values())
        active_jobs = sum(owned(p.parent) for p in (args.out / "jobs").glob("*/job.json"))
        cap = args.max_jobs
        if args.cap_file:
            cap = int(args.cap_file.read_text().strip())
            if not 1 <= cap <= args.max_jobs:
                raise ValueError(f"Requested cap {cap} outside [1, {args.max_jobs}]")
        if args.telemetry:
            with (args.out / "utilization.jsonl").open("a") as out:
                pairs = {
                    g: sum((g, i, 0) in ledger and (g, i, 1) in ledger for i in range(len(groups[g][0])))
                    for g in args.groups
                }
                print(
                    json.dumps(
                        dict(
                            time=time.time(),
                            active=active_jobs + active_external,
                            cap=cap,
                            free_cores=round(free_cpu, 2),
                            free_mib=round(free_mib),
                            pairs=pairs,
                        )
                    ),
                    file=out,
                )
        if clear < args.clear_polls or active_jobs + active_external >= cap:
            continue
        suspended = next(((g, pid) for g, (pid, state) in held.items() if state in ("T", "t")), None)
        if suspended and args.resume_existing:
            clear = 0
            os.kill(suspended[1], signal.SIGCONT)
            print(f"RESUME legacy G{suspended[0]} pid={suspended[1]}", flush=True)
            continue
        chunk = next_chunk(args, groups, missing, held)
        if chunk:
            clear = 0
            handles.append(start_job(args, groups, *chunk))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "_worker":
        sys.exit(worker(Path(sys.argv[2]), int(sys.argv[3])))
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("samples", "out", "host", "config"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--groups", nargs="+", type=int, choices=GROUPS, default=list(GROUPS))
    p.add_argument("--prior-samples", type=Path)
    p.add_argument("--prior-logs", type=Path)
    p.add_argument("--existing", action="append", default=[], metavar="G:PID:LOG")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--resume-existing", action="store_true")
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--max-jobs", type=int, default=2)
    p.add_argument("--cap-file", type=Path, help="Optional live concurrency cap, 1..max-jobs")
    p.add_argument("--idle-cpu", action="store_true", help="Use Linux SCHED_IDLE for model worker threads")
    p.add_argument(
        "--telemetry", action="store_true", help="Log headroom and checked pairs by group each poll"
    )
    p.add_argument("--reserve-cores", type=float, default=3)
    p.add_argument("--job-cores", type=float, default=2)
    p.add_argument("--reserve-mib", type=int, default=1024)
    p.add_argument("--job-mib", type=int, default=768)
    p.add_argument("--poll", type=int, default=8)
    p.add_argument("--clear-polls", type=int, default=3)
    p.add_argument("--attempts", type=int, default=2)
    p.add_argument("--wait-executable-prefix", action="append", default=[])
    cfg = p.parse_args()
    if (
        min(cfg.chunk, cfg.max_jobs, cfg.poll, cfg.clear_polls, cfg.attempts, cfg.job_mib, cfg.job_cores) <= 0
        or min(cfg.reserve_mib, cfg.reserve_cores) < 0
        or len(set(cfg.groups)) != len(cfg.groups)
    ):
        p.error("Invalid resource or cohort options")
    cfg.out.mkdir(parents=True, exist_ok=True)
    with (cfg.out / "scheduler.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.exit(2, f"Another scheduler owns {cfg.out}. Read per-query.csv for its last checkpoint.\n")
        schedule(cfg)
