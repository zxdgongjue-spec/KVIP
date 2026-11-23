"""Exercise interrupted model scheduling without launching a simulator."""

import csv
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixture(samples, group):
    meta = []
    inputs = bytearray()
    wants = bytearray()
    for i in range(6):
        posted = (group + i) % 5
        meta.append(dict(group=group, row=i // 4, q=i % 4, window=1, posted=posted))
        inputs += struct.pack("<2I", group, i) + bytes(7184 * 4 - 8)
        wants += struct.pack("<3I", posted, group, i) + bytes(72 * 4 - 12)
    (samples / f"g{group}.json").write_text(json.dumps(meta))
    (samples / f"g{group}.bin").write_bytes(inputs)
    (samples / f"g{group}.want").write_bytes(wants)


def mock_runner(prefix, mode):
    # Different group, index and variant produce different known ticks. An incorrect
    # slice, ordering error or accidental retry is visible in the final ledger.
    source = prefix.with_suffix(".bin").read_bytes()
    expected = prefix.with_suffix(".want").read_bytes()
    for v in {"--once": [0, 1], "--only-old": [0], "--only-new": [1]}[mode]:
        for k in range(len(source) // (7184 * 4)):
            g, i = struct.unpack_from("<2I", source, k * 7184 * 4)
            n, g2, i2 = struct.unpack_from("<3I", expected, k * 72 * 4)
            assert (g, i) == (g2, i2)
            print(f"PASS row={k} pass={v} variant={v} count={n} ticks={1000 * g + 10 * i + v}", flush=True)
        if g == 10 and i == 2 and mode == "--once" and v == 0:
            time.sleep(3)
            return 13
    return 0


def exercise(root):
    samples, host, config, out = (root / k for k in ("samples", "host", "config", "results"))
    for folder in (samples, host, config):
        folder.mkdir()
    for g in (10, 4):
        write_fixture(samples, g)
    (config / "config.json").write_text("{}")
    runner = host / "probe"
    runner.write_text("#!" + sys.executable + "\n" + Path(__file__).read_text())
    runner.chmod(0o700)
    device = host / "unused-simulated-device-lib"
    device.write_bytes(b"mock")
    (host / "kernel.json").write_text(
        json.dumps(
            dict(runner_sha256=digest(runner), device_sha256=digest(device), device_archive=str(device))
        )
    )
    script = Path(__file__).with_name("schedule_chunks.py")
    command = [
        sys.executable,
        str(script),
        "--samples",
        str(samples),
        "--host",
        str(host),
        "--config",
        str(config),
        "--out",
        str(out),
        "--groups",
        "10",
        "4",
        "--max-jobs",
        "2",
        "--chunk",
        "3",
        "--poll",
        "1",
        "--clear-polls",
        "1",
        "--reserve-cores",
        "0",
        "--job-cores",
        ".01",
        "--reserve-mib",
        "0",
        "--job-mib",
        "1",
    ]
    first = subprocess.Popen(command)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            logs = list((out / "jobs").glob("*/run.log"))
            if any("variant=0" in path.read_text() for path in logs):
                break
            time.sleep(0.1)
        else:
            raise TimeoutError("Mock runner did not start")
    finally:
        first.terminate()
        first.wait(timeout=5)
    subprocess.run(command, text=True, capture_output=True, check=True, timeout=45)
    for path in (out / "jobs").glob("*/exit.json"):
        assert json.loads(path.read_text())["rc"] in (0, 13)
    jobs = sorted((out / "jobs").glob("*/job.json"))
    assert len(jobs) == 5
    assert any(json.loads(job.read_text())["mode"] == "--only-new" for job in jobs)
    rows = list(csv.DictReader((out / "per-query.csv").open()))
    assert len(rows) == len({(r["group"], r["index"], r["variant"]) for r in rows}) == 24
    for row in rows:
        g, i, v = (int(row[k]) for k in ("group", "index", "variant"))
        assert float(row["ticks"]) == 1000 * g + 10 * i + v
        assert int(row["first_in_launch"]) == (i % 3 == 0)
    again = subprocess.run(command + ["--plan"], text=True, capture_output=True, check=True)
    assert json.loads(again.stdout)["checked"] == 24
    assert len(list((out / "jobs").glob("*/job.json"))) == len(jobs)
    print(
        f"PASS recovered {len(rows)} original-index measurements in {len(jobs)} launches; "
        "one manager interruption and one partial runner failure"
    )


if __name__ == "__main__":
    if len(sys.argv) == 3:
        sys.exit(mock_runner(Path(sys.argv[1]), sys.argv[2]))
    with tempfile.TemporaryDirectory(prefix="check-li-resume-") as temp:
        exercise(Path(temp))
