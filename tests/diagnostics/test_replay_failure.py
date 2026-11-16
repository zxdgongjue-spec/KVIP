# SPDX-License-Identifier: Apache-2.0
"""Failure checkpoints must survive native errors without changing timing."""

import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest

from kvip.diagnostics import compare, replay, replay_measure
from kvip.diagnostics.replay_journal import ReplayJournal


def _case():
    return {"status": "pass", "timing": {"mean_us": 12.5, "p50_us": 12.0}, "complete_oracle": True}


def test_completed_before_survives_after_failure_without_releasing_runtime(tmp_path, monkeypatch, capsys):
    args = SimpleNamespace(study="skip", mode="fused", transport="urma", threshold_policy="both")
    retained, released = {}, []
    retained["journal"] = ReplayJournal(tmp_path / "result.json")
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
    assert json.loads((tmp_path / "result.cases/before.json").read_text()) == _case()
    assert not (tmp_path / "result.cases/after.json").exists()
    assert not (tmp_path / "result.json").exists()
    checkpoint = json.loads((tmp_path / "result.progress.json").read_text())
    assert checkpoint["completed_cases"] == ["before"]
    assert checkpoint["progress"]["phase"] == "first_replay_and_sync"
    assert checkpoint["progress"]["variant"] == "after"
    output = capsys.readouterr().out
    assert "CASE_PASS urma-fused-before" in output and "CASE_PASS urma-fused-after" not in output


@pytest.mark.parametrize("fail_timed", [False, True])
def test_checkpoints_stay_outside_events_and_preserve_existing_synchronizations(tmp_path, monkeypatch, fail_timed):
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
    from kvip.diagnostics import replay_journal

    write_json = replay_journal._write_json
    written = []

    def write(path, value, **kwargs):
        assert not state["active"], "Disk checkpoint entered the timed device interval"
        assert value["progress"]["step"] is None
        written.append(value["progress"]["phase"])
        write_json(path, value, **kwargs)

    monkeypatch.setattr(replay_journal, "_write_json", write)
    retained = {"journal": ReplayJournal(tmp_path / "result.json")}

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
    assert written.count("timed_measurement") == 1
    assert not set(written) & {"iteration_reset", "warmup_replay", "timed_replay", "iteration_read_counts"}


@pytest.mark.parametrize("fail_close", [False, True])
def test_success_breaks_bundle_cycles_but_failed_close_preserves_owners(fail_close):
    class Bundle:
        pass

    closed = []

    def close():
        closed.append(True)
        if fail_close:
            raise RuntimeError("lease still active")

    def make_bundle():
        bundle = Bundle()
        bundle.lease = SimpleNamespace(close=close)
        bundle.reset = lambda: bundle
        bundle.verify = lambda: bundle
        bundle.read_counts = lambda: bundle
        bundle.auto_replay = SimpleNamespace(bundle=bundle)
        return bundle

    bundle = make_bundle()
    reference = weakref.ref(bundle)
    retained = {"active_case": {"bundle": bundle}}
    if fail_close:
        with pytest.raises(RuntimeError, match="lease still active"):
            replay_measure.release_case(bundle, retained)
        assert retained["active_case"]["bundle"] is bundle
        assert bundle.verify() is bundle and bundle.auto_replay.bundle is bundle
    else:
        replay_measure.release_case(bundle, retained)
        assert "active_case" not in retained
        del bundle
        # Refcounting must suffice: waiting for cyclic GC accumulated prior
        # variants' large device tensors in the real multi-case replay.
        assert reference() is None
    assert closed == [True]


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


@pytest.mark.parametrize("journal_failure", [False, True])
def test_cli_saves_failure_before_retained_owner_exit(tmp_path, monkeypatch, capsys, journal_failure):
    args = SimpleNamespace(
        quiet=True,
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
    if journal_failure:
        def fail_finish(*args, **kwargs):
            raise OSError("journal disk error")

        monkeypatch.setattr(ReplayJournal, "finish", fail_finish)

    def exit(code):
        assert code == 70
        report = json.loads(args.output.read_text())
        assert report["status"] == "failed"
        assert report["completed_cases"]["before"] == _case()
        assert report["buffers"]["output"]["bytes"] == 188424192
        assert report["error"]["message"].count("core timeout duplicate") == 100
        progress = json.loads(args.output.with_suffix(".progress.json").read_text())
        if journal_failure:
            assert progress["status"] == "in_progress"
        else:
            assert progress["status"] == "failed"
            assert progress["report"] == str(args.output)
            assert progress["progress"] == report["progress"]
        raise SystemExit(code)

    monkeypatch.setattr(replay.os, "_exit", exit)
    with pytest.raises(SystemExit, match="70"):
        replay.main([])
    captured = capsys.readouterr()
    output = captured.err
    assert output.splitlines()[0].startswith("REPLAY_FAILED variant=after phase=timed_replay step=3")
    assert "first native cause" in output and "core timeout duplicate" not in output
    assert "COMPLETED before" in output and "FAILURE_JSON" in output
    if journal_failure:
        assert "Replay completion checkpoint could not be saved: journal disk error" in captured.out


def test_cli_marks_complete_only_after_final_report_is_written(tmp_path, monkeypatch):
    args = SimpleNamespace(
        quiet=False, output=tmp_path / "complete.json", snapshot=tmp_path / "input",
        transport="urma", mode="fused", study="skip", upstream_root=None, threshold_policy="recent",
    )
    monkeypatch.setattr(replay, "arguments", lambda argv: args)
    archive = SimpleNamespace(directory=tmp_path, manifest={"content": "real_li_synthetic_kv", "ddr": {}})
    monkeypatch.setattr(replay, "load_snapshot", lambda path: archive)
    from kvip.diagnostics import ddr, snapshot

    monkeypatch.setattr(ddr, "validate_pages", lambda *args: None)
    monkeypatch.setattr(snapshot, "validate_capture_inputs", lambda *args: None)
    monkeypatch.setattr(replay_measure, "run", lambda *args: _case())
    finish = ReplayJournal.finish

    def checked(self, progress, **kwargs):
        assert json.loads(args.output.read_text()) == _case()
        assert kwargs["status"] == "complete"
        finish(self, progress, **kwargs)

    monkeypatch.setattr(ReplayJournal, "finish", checked)
    replay.main([])
    progress = json.loads(args.output.with_suffix(".progress.json").read_text())
    assert progress["status"] == "complete"
    assert progress["report"] == str(args.output)


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
