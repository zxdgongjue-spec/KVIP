# SPDX-License-Identifier: Apache-2.0
"""Failure checkpoints must survive native errors without changing timing."""

import json
import gc
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest

from kvip.diagnostics import compare, replay, replay_measure


@pytest.mark.parametrize("close_fails", [False, True])
def test_fused_bundle_released_without_waiting_for_gc(close_fails):
    class Allocation:
        pass

    def make():
        allocation = Allocation()
        reference = weakref.ref(allocation)
        bundle = SimpleNamespace(output=allocation)

        def close():
            if close_fails:
                raise RuntimeError("fence failed")

        bundle.lease = SimpleNamespace(close=close)
        bundle.verify = lambda: bundle.output
        bundle.read_counts = lambda: bundle.output
        bundle.reset = lambda: allocation
        bundle.seeds = [allocation]
        bundle.oracle = object()
        return bundle, reference

    enabled = gc.isenabled()
    gc.disable()
    try:
        bundle, reference = make()
        retained = {"active_case": {"bundle": bundle, "graphs": []}}
        if close_fails:
            with pytest.raises(RuntimeError, match="fence failed"):
                replay_measure.release_case(bundle, retained)
            assert retained["active_case"]["bundle"] is bundle
            assert bundle.verify() is reference()
            # The real CLI retains owners until exit when completion is unknown.
            bundle.lease.close = lambda: None
        replay_measure.release_case(bundle, retained)
        assert "active_case" not in retained
        del bundle
        assert reference() is None, "Completed case still retains its allocation"
    finally:
        if enabled:
            gc.enable()


def test_memory_checkpoints_keep_only_numbers_and_do_not_synchronize(monkeypatch, capsys):
    from kvip.diagnostics.console import print_replay_failure

    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4")
    torch = SimpleNamespace(npu=SimpleNamespace(
        mem_get_info=lambda device: (20 << 30, 92 << 30),
        memory_allocated=lambda device: 6 << 30,
        memory_reserved=lambda device: 7 << 30,
    ))
    retained = {"progress": {"variant": "reverse"}}
    row = replay_measure.record_memory(torch, "npu:0", retained, "before_bundle")
    assert row["visible_devices"] == "4" and row["device"] == "npu:0"
    assert row["free_bytes"] == 20 << 30
    json.dumps(retained)
    print_replay_failure(retained, result_path="failed.json")
    assert "free_GiB=20.000 allocated_GiB=6.000 reserved_GiB=7.000" in capsys.readouterr().err

    def fail(device):
        raise RuntimeError("device unavailable")

    torch.npu.mem_get_info = fail
    row = replay_measure.record_memory(torch, 0, retained, "failure")
    assert row["error"] == "RuntimeError: device unavailable"


def test_native_exit_without_json_follows_recorded_spool(tmp_path, capsys):
    from kvip.diagnostics.console import saved_failure

    native = tmp_path / "native log with spaces.log"
    native.write_text("startup\n" * 50 + "FATAL allocation failed\n" + "context\n" * 20 + "last native line\n")
    outer = tmp_path / "outer.log"
    outer.write_text(f"RUN_LOG {native} (temporary; removed on success)\nBUILD_VARIANTS_READY unique=6\n")
    saved_failure(tmp_path / "absent.json", outer, returncode=137)
    output = capsys.readouterr().err
    assert "CHILD_EXIT_CODE=137" in output
    assert "FATAL allocation failed" in output and "last native line" in output
    assert "FAILURE_JSON_UNAVAILABLE" in output and "NATIVE_LOG_UNAVAILABLE" not in output
    assert len(output.splitlines()) < 65


def test_saved_failure_keeps_structured_phase_and_reads_native_log_once(tmp_path, capsys):
    from kvip.diagnostics.console import saved_failure

    native = tmp_path / "native.log"
    native.write_text("ERROR runtime link failed\n")
    result = tmp_path / "failed.json"
    result.write_text(json.dumps(dict(status="failed", progress=dict(variant="unassigned", phase="runtime_link"),
        error=dict(type="RuntimeError", message="link failure"), native_log=str(native))))
    outer = tmp_path / "outer.log"
    outer.write_text(f"RUN_LOG {native} (temporary; removed on success)\n")
    saved_failure(result, outer)
    output = capsys.readouterr().err
    assert "phase=runtime_link" in output
    assert output.count(f"NATIVE_LOG={native}\n") == 1


def test_quiet_failure_preserves_first_compiler_diagnostic(tmp_path, capsys):
    from kvip.diagnostics.console import print_replay_failure

    message = ("LIFetch compile/link failed; full log: /build/build.log\n"
               "--- build diagnostic ---\n"
               "rolling_threshold_vector.h:29: error: no matching function for call to 'MaskOr'\n"
               "note: requires 4 arguments, but 3 were provided\n" + "trailing\n" * 100)
    print_replay_failure(dict(error=dict(type="RuntimeError", message=message)), result_path=tmp_path / "failed.json")
    output = capsys.readouterr().err
    assert "BUILD_DIAGNOSTIC_BEGIN" in output and "BUILD_DIAGNOSTIC_END" in output
    assert "requires 4 arguments, but 3 were provided" in output
    assert output.count("trailing\n") == 33


def _case():
    return {"status": "pass", "timing": {"mean_us": 12.5, "p50_us": 12.0}, "complete_oracle": True}


def test_completed_before_survives_after_failure_without_releasing_runtime(monkeypatch, capsys):
    args = SimpleNamespace(study="skip", mode="fused", transport="urma", threshold_policy="both")
    retained, released = {}, []
    monkeypatch.setattr(replay_measure, "allocate_context", lambda *args: object())
    monkeypatch.setattr(replay_measure, "release_context", released.append)

    def measure(args, archive, context, retained):
        if not args.fused_tuning["recent_skip"]:
            return _case()
        retained["active_case"] = {"buffers": {"output": {"address": "0x1234", "bytes": 128}}}
        replay_measure.phase(retained, "first_replay_and_sync")
        raise RuntimeError("invalid GM")

    monkeypatch.setattr(replay_measure, "measure_case", measure)
    with pytest.raises(RuntimeError, match="invalid GM"):
        replay_measure.run(args, object(), retained)
    assert retained["completed_cases"] == {"before": _case()}
    assert retained["progress"]["variant"] == "after"
    assert retained["progress"]["phase"] == "first_replay_and_sync"
    assert retained["active_case"]["buffers"]["output"]["address"] == "0x1234"
    assert not released
    output = capsys.readouterr().out
    assert "CASE_PASS urma-fused-before" in output and "CASE_PASS urma-fused-after" not in output


@pytest.mark.parametrize("fail_timed", [False, True])
def test_checkpoints_stay_outside_events_and_preserve_existing_synchronizations(monkeypatch, fail_timed):
    state = dict(active=False, stream_sync=0, event_sync=0, replays=0, event_count=0)
    phases, original = [], replay_measure.phase

    def checked(retained, name, step=None):
        assert not state["active"], "Failure metadata entered the timed device interval"
        phases.append((name, step))
        original(retained, name, step)

    monkeypatch.setattr(replay_measure, "phase", checked)

    class Event:
        def __init__(self, **kwargs):
            self.begin = state["event_count"] == 0
            state["event_count"] += 1

        def record(self, stream):
            if self.begin:
                state["active"] = True

        def synchronize(self):
            state["event_sync"] += 1
            state["active"] = False

        def elapsed_time(self, other):
            return 0.01

    class Graph:
        def capture_begin(self, stream):
            pass

        def capture_end(self):
            pass

        def replay(self, stream):
            state["replays"] += 1
            if fail_timed and state["replays"] == 3:
                raise RuntimeError("timed native failure")

        def reset(self):
            assert not state["active"]

    def synchronize():
        state["stream_sync"] += 1

    bundle = SimpleNamespace(
        reset=lambda: None, verify=lambda: {}, read_counts=lambda: {}, forward=lambda: None, lease=None
    )
    args = SimpleNamespace(mode="fused", warmup=1, iterations=2)
    retained = {}

    def invocation():
        return replay_measure.measure(
            SimpleNamespace(npu=SimpleNamespace(Event=Event)),
            SimpleNamespace(AclGraph=Graph),
            SimpleNamespace(synchronize=synchronize),
            bundle,
            args,
            retained,
        )

    if fail_timed:
        with pytest.raises(RuntimeError, match="timed native failure"):
            invocation()
        assert retained["progress"] == {"phase": "timed_replay", "step": 1}
        assert state["stream_sync"] == 5 and state["event_sync"] == 1
    else:
        assert invocation()["timing"]["count"] == 2
        assert state["stream_sync"] == 6 and state["event_sync"] == 3
        assert retained["progress"]["phase"] == "graph_reset"


def test_buffer_metadata_never_reads_tensor_values_or_syncs():
    class Tensor:
        dtype = "torch.uint8"

        def __init__(self, address, shape):
            self.address, self.shape = address, shape

        def data_ptr(self):
            return self.address

        def numel(self):
            import math

            return math.prod(self.shape)

        def element_size(self):
            return 1

    compute, prefix = Tensor(0x200000, (1 << 20,)), Tensor(0x400000, (4096,))
    arguments = [None] * 14
    arguments[0], arguments[9], arguments[12], arguments[13] = (
        Tensor(0x100, (4, 32, 128)),
        Tensor(0x200, (8192,)),
        compute,
        prefix,
    )
    bundle = SimpleNamespace(
        output=Tensor(0x1200D7A00000, (17, 16896, 656)),
        storage=SimpleNamespace(keys=Tensor(0x600000, (17, 131328, 128))),
        rotating=SimpleNamespace(arguments=arguments, runtime_shape=SimpleNamespace(batch=1, history=4096)),
    )
    result = replay_measure.buffer_addresses(bundle)
    assert result["output"]["bytes"] == 188424192
    assert result["output"]["shape"] == [17, 16896, 656]
    assert result["scoreGM"]["address"] == result["compute"]["address"]
    assert result["scoreGM"]["bytes"] == 1 * 4 * 4096 * 2
    json.dumps(result)


def test_cli_saves_failure_before_retained_owner_exit(tmp_path, monkeypatch, capsys):
    args = SimpleNamespace(
        quiet=True,
        isolate_variants=False,
        output=tmp_path / "failed.json",
        snapshot=tmp_path / "input",
        transport="urma",
        mode="fused",
        study="skip",
        upstream_root=None,
    )
    log = SimpleNamespace(failure_path=tmp_path / "native.log")
    monkeypatch.setattr(replay, "arguments", lambda argv: args)

    class Log:
        failure_path = log.failure_path

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(replay, "NativeLog", lambda *args, **kwargs: Log())
    archive = SimpleNamespace(directory=tmp_path, manifest={"content": "real_li_synthetic_kv", "ddr": {}})
    monkeypatch.setattr(replay, "load_snapshot", lambda path: archive)
    from kvip.diagnostics import ddr, snapshot

    monkeypatch.setattr(ddr, "validate_pages", lambda *args: None)
    monkeypatch.setattr(snapshot, "validate_capture_inputs", lambda *args: None)

    def fail(args, archive, retained):
        retained.update(
            progress={"variant": "after", "phase": "timed_replay", "step": 3},
            completed_cases={"before": _case()},
            active_case={"buffers": {"output": {"address": "0x1200d7a00000", "bytes": 188424192}}},
        )
        raise RuntimeError("first native cause\n" + "core timeout duplicate\n" * 100)

    monkeypatch.setattr(replay_measure, "run", fail)

    def exit(code):
        assert code == 70
        report = json.loads(args.output.read_text())
        assert report["status"] == "failed"
        assert report["completed_cases"]["before"] == _case()
        assert report["buffers"]["output"]["bytes"] == 188424192
        assert report["error"]["message"].count("core timeout duplicate") == 100
        raise SystemExit(code)

    monkeypatch.setattr(replay.os, "_exit", exit)
    with pytest.raises(SystemExit, match="70"):
        replay.main([])
    output = capsys.readouterr().err
    assert output.splitlines()[0].startswith("REPLAY_FAILED variant=after phase=timed_replay step=3")
    assert "first native cause" in output and "core timeout duplicate" not in output
    assert "COMPLETED before" in output and "FAILURE_JSON" in output


def test_compare_uses_failed_json_instead_of_repeating_device_error_tail(tmp_path, monkeypatch, capsys):
    args = SimpleNamespace(
        result_dir=tmp_path / "results",
        build_dir=tmp_path / "build",
        snapshot=tmp_path / "input",
        kvaf_server="127.0.0.1:29094",
        device=0,
        soc="A5",
        serial_workers=32,
        warmup=2,
        iterations=10,
        study="skip",
        transports="urma",
    )

    def child(command, stdout, **kwargs):
        output = Path(command[command.index("--output") + 1])
        if command[command.index("--mode") + 1] == "serial":
            output.write_text(json.dumps(_case()))
            return SimpleNamespace(returncode=0)
        stdout.write("duplicated low level message\n" * 100)
        output.write_text(
            json.dumps(
                dict(
                    status="failed",
                    progress={"variant": "after", "phase": "first_replay_and_sync", "step": None},
                    error={"type": "RuntimeError", "message": "first native cause"},
                    completed_cases={"before": _case()},
                )
            )
        )
        return SimpleNamespace(returncode=70)

    monkeypatch.setattr(compare.subprocess, "run", child)
    with pytest.raises(SystemExit, match="1"):
        compare.run(args)
    output = capsys.readouterr().err
    assert output.splitlines()[0].startswith("REPLAY_FAILED variant=after phase=first_replay_and_sync")
    assert "COMPLETED before" in output and "duplicated low level" not in output
    assert (args.result_dir / ".urma-fused.log").exists()
    assert (args.result_dir / "urma-serial.json").exists()
    failure = json.loads((args.result_dir / "urma-fused.json").read_text())
    assert failure["child_exit_code"] == 70
    assert Path(failure["comparison_log"]).is_file()
