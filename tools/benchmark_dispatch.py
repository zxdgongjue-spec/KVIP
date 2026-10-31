"""CPU-only session-dispatch evidence; not an NPU or end-to-end performance test."""

import json
import statistics
import timeit
from types import SimpleNamespace

import numpy as np

from kvop.testing.owner import StubQueueOwner
from kvop.v9 import PreparedSession
from kvop.v9.access import OUTPUT_FIELDS


def main():
    stream = SimpleNamespace(device="npu:0", npu_stream=10)
    owner = StubQueueOwner(np.zeros((1, 656), np.uint8), stream)
    torch_stub = SimpleNamespace(npu=SimpleNamespace(current_stream=lambda _: stream))
    status = np.zeros(1, np.int32)

    def complete():
        owner.mark_completed(completion_verified=True)
        return status

    launcher = SimpleNamespace(
        owner=owner,
        torch=torch_stub,
        device="npu:0",
        launch=lambda: None,
        completion=complete,
        outputs=dict.fromkeys(OUTPUT_FIELDS, owner.output),
    )
    session = PreparedSession(launcher)

    def baseline():
        owner.in_flight = True
        launcher.launch()
        return bool(np.all(launcher.completion() == 0))

    def guarded():
        session.enqueue()
        return session.complete().success

    number = 20000
    low = (
        statistics.median(timeit.repeat(baseline, number=number, repeat=5))
        / number
        * 1e6
    )
    high = (
        statistics.median(timeit.repeat(guarded, number=number, repeat=5))
        / number
        * 1e6
    )
    print(
        json.dumps(
            {
                "scope": "CPU Python dispatch only; no tensor/kernel/queue work",
                "baseline_us": low,
                "session_us": high,
                "additional_us": high - low,
                "production_performance_equivalence": "requires NPU/KVAF/model A-B validation",
            },
            indent=2,
        )
    )
    session.close()


if __name__ == "__main__":
    main()
