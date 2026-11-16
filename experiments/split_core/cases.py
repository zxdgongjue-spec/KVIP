"""Current score/staged validation cases and an exclusive Model lease."""

import contextlib
import itertools
import json
import os
from pathlib import Path
import uuid

BATCHES = (1, 2, 4, 8, 16, 24, 32)
MISSES = (3, 5, 10, 15, 20, 50)


def cases(group="all"):
    batches = {"small": (1, 2), "medium": (4, 8), "large": (16, 24, 32), "all": BATCHES}[group]
    for batch, miss in itertools.product(batches, MISSES):
        count = round(2048 * miss / 100)
        yield dict(
            case=f"b{batch}-h100-m{miss}",
            batch=batch,
            queries_per_request=4,
            total_queries=4 * batch,
            history_percent=100,
            valid_history=131072,
            history=131072,
            resident=16384,
            miss_percent=miss,
            miss_per_query=count,
            actual_miss_percent=100 * count / 2048,
        )


@contextlib.contextmanager
def lease(backend, device, lease_file=None):
    path = Path(
        lease_file or ("/tmp/kvip-cann91-simulator.lease" if backend == "model" else f"/tmp/kvip-npu-{device}.lease")
    )
    owner = dict(task="split-core-validation", pid=os.getpid(), lease=uuid.uuid4().hex)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(owner, f)
    try:
        yield
    finally:
        if path.exists() and json.loads(path.read_text()).get("lease") == owner["lease"]:
            path.unlink()
