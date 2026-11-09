"""Host tests for the single-run diagnostic orchestration."""

import compare_transports as compare


def test_baseline_requires_matching_geometry_and_successful_full_graph(tmp_path):
    import pytest

    summary = tmp_path / "summary.log"
    summary.write_text(
        "batch=16 history=131072 resident=16384 heads=32 Q=4 topk=2048 windows=8 budget=512 graph=True\n"
        "urma-miss10: PASS exit=0\n"
        "graph_p50_us=native_chain:605.794, prototype_with_update:518.138\n"
        "ubmem-miss10: PASS exit=0\n"
        "UBMEM_GRAPH_EVENT_US p50=499.474 p99=508.989\n"
        "urma-miss30: FAIL exit=1\n"
        "graph_p50_us=prototype_with_update:1\n"
    )
    assert compare.baseline_timings(summary, 16, 131072, 16384, 32) == {
        ("urma", 10): 518.138, ("ubmem", 10): 499.474,
    }
    with pytest.raises(ValueError, match="geometry differs"):
        compare.baseline_timings(summary, 8, 131072, 16384, 32)


def test_runner_requires_checked_completion_for_both_backends(tmp_path, monkeypatch):
    root = tmp_path / "KVAF"
    (root / "build-npu").mkdir(parents=True)
    (root / "build-npu/CMakeCache.txt").touch()
    commands = []

    def fake_execute(command, log, timeout, label):
        commands.append((label, command))
        if "--comparison-sfa" in command:
            import numpy as np
            np.save(command[command.index("--comparison-sfa") + 1], np.zeros((1,)))
        if label == "ubmem-runtime-shapes":
            log.write_text("UBMEM_RUNTIME_BINARY path=fixture\n" +
                           "UBMEM_LIFETCH_PASS\nUBMEM_SFA_PASS\n" * 5)
        elif label.startswith("urma"):
            log.write_text("PASS: exit_code=0\ngraph_p50_us=native_chain:100\nSFA_max_abs=0.001\n")
        else:
            log.write_text("UBMEM_LIFETCH_PASS\nUBMEM_GRAPH_EVENT_US p50=100\nUBMEM_SFA_PASS\n"
                           "UBMEM_WINDOW_TRIGGERS tokens=[16384]\n"
                           + "".join(f"UBMEM_TRACE round={i} resolve_us=10 fetch_us=20\n" for i in range(3)))
        return 0

    monkeypatch.setattr(compare, "execute", fake_execute)
    monkeypatch.setattr(compare, "compare_snapshots", lambda *args: ["logical_topk=identical selected_kv_bytes=identical"])
    output = tmp_path / "results"
    args = [
        "--device",
        "5",
        "--kvaf-root",
        str(root),
        "--ubmem-server",
        "127.0.0.1:28885",
        "--urma-server",
        "127.0.0.1:28885",
        "--soc",
        "Ascend950PR_9599",
        "--miss-percent",
        "10",
        "--output",
        str(output),
    ]
    assert compare.main(args) == 0
    assert [label for label, _ in commands] == ["ubmem-runtime-shapes", "urma-miss10", "ubmem-miss10"]
    for _, command in commands:
        for flag, value in (
            ("--history", "131072"),
            ("--resident", "16384"),
            ("--index-heads", "32"),
            ("--miss-percent", "10"),
        ):
            assert command[command.index(flag) + 1] == value
    summary = next(output.glob("*/summary.log")).read_text()
    assert "urma-miss10: PASS" in summary and "ubmem-miss10: PASS" in summary


def test_runner_does_not_treat_zero_exit_without_correctness_as_pass(tmp_path, monkeypatch):
    root = tmp_path / "KVAF"
    (root / "build-npu").mkdir(parents=True)
    (root / "build-npu/CMakeCache.txt").touch()

    def incomplete(command, log, timeout, label):
        log.write_text("PASS: exit_code=0\ngraph_p50_us=native_chain:100\n")
        return 0

    monkeypatch.setattr(compare, "execute", incomplete)
    output = tmp_path / "results"
    args = [
        "--device",
        "5",
        "--kvaf-root",
        str(root),
        "--ubmem-server",
        "x:1",
        "--urma-server",
        "x:1",
        "--soc",
        "Ascend950PR_9599",
        "--miss-percent",
        "10",
        "--output",
        str(output),
    ]
    assert compare.main(args) == 1
    assert "ubmem-runtime-shapes: FAIL" in next(output.glob("*/summary.log")).read_text()


def test_snapshot_compares_kv_and_reports_prefetch_differences(tmp_path):
    import numpy as np
    import pytest
    from transport_snapshot import save, compare as compare_snapshot

    ids = np.tile(np.arange(2048, dtype=np.int32), (1, 4, 1))
    slots = ids.copy()
    output = np.arange(4096 * 4, dtype=np.uint8).reshape(1, 4096, 4)
    mapping = np.full((1, 4096), -1, dtype=np.int32)
    mapping[0, 8] = 4096
    lru = np.arange(2048, dtype=np.int32)[None, :]
    rows = np.zeros((1, 256), dtype=np.uint32)
    rows[0, 16] = 1
    rows[0, 214] = 1
    addresses = np.arange(4096, dtype=np.int64)[None, :]
    a, b = tmp_path / 'a.npz', tmp_path / 'b.npz'
    save(a, ids, slots, output, mapping, lru, rows, addresses)
    mapping[0, 9] = 4097
    rows[0, 17] = 1
    rows[0, 215] = 2
    lru[0, 1] = 17
    save(b, ids, slots, output, mapping, lru, rows, addresses)
    result = compare_snapshot(a, b, 4096)
    assert 'logical_topk=identical selected_kv_bytes=identical' in result[0]
    assert 'lru_candidate_order_different_rows=1/1 differing_positions=1' in result
    assert any('ubmem_only=1' in line for line in result)
    output[0, 0, 0] ^= 1
    save(b, ids, slots, output, mapping, lru, rows, addresses)
    with pytest.raises(AssertionError, match='KV bytes differ'):
        compare_snapshot(a, b, 4096)
