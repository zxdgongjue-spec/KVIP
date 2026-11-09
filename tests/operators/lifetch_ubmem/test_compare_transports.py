"""Host tests for the single-run diagnostic orchestration."""

import compare_transports as compare


def test_runner_requires_checked_completion_for_both_backends(tmp_path, monkeypatch):
    root = tmp_path / "KVAF"
    (root / "build-npu").mkdir(parents=True)
    (root / "build-npu/CMakeCache.txt").touch()
    commands = []

    def fake_execute(command, log, timeout, label):
        commands.append((label, command))
        if label.startswith("urma"):
            log.write_text("PASS: exit_code=0\ngraph_p50_us=native_chain:100\nSFA_max_abs=0.001\n")
        else:
            log.write_text("UBMEM_LIFETCH_PASS\nUBMEM_GRAPH_EVENT_US p50=100\nUBMEM_SFA_PASS\nUBMEM_WINDOW_TRIGGERS tokens=[16384]\n")
        return 0

    monkeypatch.setattr(compare, "execute", fake_execute)
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
    assert [label for label, _ in commands] == ["urma-miss10", "ubmem-miss10"]
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
    assert "urma-miss10: FAIL" in next(output.glob("*/summary.log")).read_text()
