# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from kvip.diagnostics.graph_accounting import queue_stats
from kvip.diagnostics.graph_queue_probe import COPIES, FENCED, PROTOCOL, probe


@pytest.mark.parametrize("separated", [True, False])
def test_forced_begin_and_prequeued_begin_with_same_empty_controls(separated):
    ops = []
    stream = SimpleNamespace(device="npu:0", synchronize=lambda: ops.append("sync"))

    def query():
        waited = "wait-start" in ops[len(ops) - 4 :]
        ops.append("query")
        return waited or not separated

    begin = SimpleNamespace(
        record=lambda s: ops.append("begin"),
        query=query,
        synchronize=lambda: ops.append("wait-start"),
        elapsed_time=lambda other: 0.01,
    )
    end = SimpleNamespace(record=lambda s: ops.append("end"), synchronize=lambda: ops.append("wait"))
    target = SimpleNamespace(copy_=lambda data, **kwargs: ops.append("copy"))
    torch = SimpleNamespace(
        npu=SimpleNamespace(stream=lambda s: nullcontext()),
        uint8=object(),
        empty_like=lambda source: target,
        zeros=lambda *args, **kwargs: object(),
    )
    results = probe(
        torch,
        stream,
        begin,
        end,
        lambda: ops.append("replay"),
        lambda step: ops.append(step),
        lambda step: ops.append("verify") or 20,
        8,
        20,
    )
    for i, row in enumerate(results["samples"]):
        start = ops.index(8 + i)
        measured = ["copy"] * COPIES + ["begin"] + (["wait-start"] if row["fenced"] else [])
        after = ["end", "query", "wait"]
        graph, blank = measured + ["replay"] + after, measured + after
        expected = [8 + i, "sync"] + (blank + graph if i < 2 else graph + blank) + ["verify"]
        assert ops[start : start + len(expected)] == expected
    assert results["status"] == ("measured" if separated else "queue_states_unresolved")
    stats = queue_stats(results, 20)
    assert ("sensitivity_after_empty" in stats) == separated
    assert "empty_started" in stats


def test_empty_and_graph_are_accounted_separately():
    samples = []
    for fenced in FENCED:
        samples.append(
            dict(
                fenced=fenced,
                final_get=2,
                elapsed_us=27 if fenced else 17,
                begin_pending_after_submit=not fenced,
                host_call_us=10 if fenced else 6,
                empty=dict(elapsed_us=8 if fenced else 1, begin_pending_after_submit=not fenced, host_call_us=0.1),
            )
        )
    payload = dict(status="measured", protocol=PROTOCOL, samples=samples)
    stats = queue_stats(payload, 2)
    assert stats["queue_sensitivity"] == 10 and stats["empty_sensitivity"] == 7
    assert stats["sensitivity_after_empty"] == 3 and stats["queued_host_call"] == 6
    samples[1]["empty"]["begin_pending_after_submit"] = False
    assert "sensitivity_after_empty" not in queue_stats(payload, 2)
    del payload["protocol"]
    assert queue_stats(payload, 2)["queue_sensitivity"] == 10


def test_missing_query_keeps_headline_and_profile():
    assert probe(None, None, object(), None, None, None, None, 1, 0)["status"] == "unavailable"
