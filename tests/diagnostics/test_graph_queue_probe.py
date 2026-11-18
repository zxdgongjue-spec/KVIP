# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from kvip.diagnostics.graph_queue_probe import COPIES, probe


@pytest.mark.parametrize("separated", [True, False])
def test_equal_traffic_around_both_stream_states(separated):
    ops = []
    stream = SimpleNamespace(device="npu:0", synchronize=lambda: ops.append("sync"))
    begin = SimpleNamespace(
        record=lambda s: ops.append("begin"),
        query=lambda: ops.append("query") or (ops[-5] == "sync" and separated),
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
        expected = [
            8 + i,
            "sync",
            *(["copy"] * COPIES),
            *(["sync"] if row["fenced"] else []),
            "begin",
            "replay",
            "end",
            "query",
            "wait",
            "verify",
        ]
        assert ops[start : start + len(expected)] == expected
    assert results["status"] == ("measured" if separated else "queue_states_unresolved")


def test_missing_query_keeps_headline_and_profile():
    assert probe(None, None, object(), None, None, None, None, 1, 0)["status"] == "unavailable"
