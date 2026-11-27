# SPDX-License-Identifier: Apache-2.0
"""Strong-serial timing, a single natural GET journal, or an instrumented fixed-list timing block."""

import argparse
import ctypes
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from four_rank_tail_image import tail_text


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def canonical_journal(inputs, pages, addresses, sources, slots, epochs, final, starts, history, width):
    """One stable remote address = registered donor-page ID and byte offset within that page."""
    result = {}
    for r, (by_window, by_q) in enumerate(zip(epochs, final)):
        end = sum(by_window) + sum(by_q)
        if end > sources.shape[1]:
            raise ValueError("Journal capacity exceeded")
        spans = []
        at = 0
        for window, n in enumerate(by_window):
            spans.append((f"w{window}", at, n))
            at += n
        spans.extend((f"q{q}", int(starts[r, q]), n) for q, n in enumerate(by_q))
        cover = np.zeros(end, dtype=bool)
        for tag, begin, n in spans:
            if n == 0:
                result[f"r{r}_{tag}"] = np.empty((0, 5), dtype="<i4")
                continue
            if n < 0 or begin < 0 or begin + n > end or cover[begin : begin + n].any():
                raise ValueError(f"Overlapping or incomplete terminal journal R{r} {tag}")
            cover[begin : begin + n] = True
            pos, slot = sources[r, begin : begin + n], slots[r, begin : begin + n]
            if (
                np.any(pos < 0)
                or np.any(pos >= history)
                or np.any(slot < 0)
                or np.any(slot >= inputs.geometry["resident"])
            ):
                raise ValueError("Invalid source or destination physical slot")
            token = inputs.captured_layout[r, pos]
            offset = token % 128 * width
            if not np.all(inputs.logical_address_valid[r, token]) or not np.array_equal(
                addresses[r, token], addresses[r, token // 128 * 128] + offset.astype(np.uint64)
            ):
                raise ValueError("Journal source does not resolve to a valid original registered address")
            page = pages.page_indices[r, token // 128]
            result[f"r{r}_{tag}"] = np.column_stack((pos, token, page, offset, slot)).astype("<i4")
        if not np.all(cover):
            raise ValueError("GET journal does not cover every admission")
    return result


def compiler_identity(sdk):
    from kvip.runtime.build import kvaf_build_paths

    cann, _ = kvaf_build_paths(sdk / "build-npu")
    compilers = [
        cann / p
        for p in (
            "compiler/ccec_compiler/bin/bisheng",
            "aarch64-linux/ccec_compiler/bin/bisheng",
            "compiler/bin/bisheng",
            "bin/bisheng",
        )
        if (cann / p).is_file()
    ]
    if not compilers:
        raise ValueError(f"Cannot identify the installed device compiler in {cann}")
    tools = {str(p.resolve()): sha(p) for p in dict.fromkeys(p.resolve() for p in compilers)}
    rules = cann / "compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake"
    tools[str(rules.resolve())] = sha(rules)
    return tools


def terminal_image(s, sdk):
    fwd = s.forward
    library = Path(s.rotating.library._name).resolve()
    fun = fwd.terminal_function
    bound = dict(zip(fun.__code__.co_freevars, (v.cell_contents for v in fun.__closure__)))
    expected = ctypes.cast(s.rotating.library.aclrtlaunch_v9_parallel_terminal_large, ctypes.c_void_p)
    if getattr(bound.get("tail"), "value", None) != expected.value:
        raise ValueError("Cannot verify the bound four-Q terminal launch")
    return dict(
        tail_text=tail_text(library),
        compiler_sha256=compiler_identity(sdk),
        cold_build=s.large_window_launch.source_fingerprint,
    )


def device_journal(s, inputs, pages, counts, map_digests=None):
    from large_batch_pair import check_ordered_index

    fwd = s.forward
    check_ordered_index(s, inputs.captured_layout, inputs.captured_hot_valid)
    stamp = fwd.terminal.times.cpu().numpy()
    final = counts["row_query_final"]
    if not np.array_equal(stamp[:, :, 20], final):
        raise ValueError("Terminal count and per-Q completed GETs disagree")
    src = fwd.events_src.cpu().numpy()
    dst = fwd.events_dst.cpu().numpy()
    if map_digests is not None:
        from four_rank_final_map import row_assignment
        for r, (w, q) in enumerate(zip(counts["row_window_prefetch"], final, strict=True)):
            n = int(sum(w) + sum(q))
            map_digests.append(row_assignment(src[r, :n], dst[r, :n]))
    return canonical_journal(
        inputs,
        pages,
        s.logical_addresses,
        src,
        dst,
        counts["row_window_prefetch"],
        final,
        stamp[:, :, 22],
        s.history,
        s.token_bytes,
    )


def run(a):
    sys.path.insert(0, str(a.timed / "tests/operators/lifetch/support"))
    if a.mode.startswith("window-x"):
        from b24_window_worker import run as run_windows
        return run_windows(a)
    if a.mode == "two-row-natural":
        if a.batch != 32:
            raise ValueError("The two-row tail requires BS32")
        from b24_two_row_natural import run as run_natural
        return run_natural(a)
    two_row = a.mode == "two-row-frozen"
    stagger = a.mode == "stagger-pair"
    if stagger:
        import b24_two_row_attach
        import terminal_stagger_attach
        flag = os.environ["PEER_STAGGER_ENABLE"]
        if flag not in ("0", "1"):
            raise ValueError("Invalid terminal gate setting")
        enabled = flag == "1"
        b24_two_row_attach.install_profiler("v9_terminal_stagger")
    if two_row:
        if a.batch != 32:
            raise ValueError("The two-row tail requires BS32")
        import b24_two_row_attach
        b24_two_row_attach.install_profiler()
    import torch
    import torch_npu
    import execution as ex
    import kvaf
    import kvip
    from large_batch_input import load, input_digest
    from fixed_q4_packed_worker import measure_packed
    from large_batch_pair import read_selected
    from reporting import atomic_json
    from workloads import configuration, ReplayPagePlan

    ref_path = a.ref / f"g{a.group}/bs{a.batch}/block1/fused/packed_result.json"
    local_journal = a.mode in ("stagger-journal", "stagger-pair") and not ref_path.is_file()
    rec = {} if local_journal else json.loads(ref_path.read_bytes())
    old = {"native":"d0fc5963881bd38fa97eff0f6ef40c757a6e8713"} if local_journal else rec["identity"]
    rev = subprocess.check_output(["git", "-C", a.timed, "rev-parse", "HEAD"], text=True).strip()
    if (
        subprocess.check_output(["git", "-C", a.timed, "status", "--porcelain", "-uno"], text=True).strip()
        or not Path(kvip.__file__).resolve().is_relative_to(a.timed / "python")
        or not Path(kvaf.__file__).resolve().is_relative_to(a.sdk / "build-npu/python")
        or (a.mode == "serial" and rev != "10442f1ffa314606c23f54b73e49ecfc372f1f9a")
        or (a.mode != "serial" and rev != old["native"])
    ):
        raise ValueError("Wrong timed source, runtime or archive")
    seq, donors = load(a.trace, a.group, 1, a.batch, 0, a.bank)
    inputs, pages = seq.inputs, ReplayPagePlan(seq)
    wanted = dict(
        sequence_sha256=input_digest(inputs, pages, donors),
        registered_pages=pages.page_count,
        donor_rows=donors,
        batch=a.batch,
        group=a.group,
        calls=10,
        warmup=4,
        physical_npu=int(os.environ["ASCEND_RT_VISIBLE_DEVICES"]),
        host=os.uname().nodename,
        hardware=torch.npu.get_device_name(torch.device("npu:0")),
        torch=str(torch.__version__),
        torch_npu=str(torch_npu.__version__),
        cann=str(getattr(torch.version, "cann", "")),
        kvaf_get_kernel_sha256=sha(a.sdk / "build-npu/bin/kvaf_execute_batch_get_kernels.o"),
    )
    if local_journal:
        old.update(wanted)
    if old["physical_npu"] != 4 or any(old[k] != v for k, v in wanted.items()):
        raise ValueError("Archive and live run differ in input, device, SDK or hardware")
    fused = a.mode != "serial"
    journal = a.mode in ("journal", "team-schedule", "stagger-journal")
    timed = not journal
    if a.mode == "team-schedule" and rev != "d0fc5963881bd38fa97eff0f6ef40c757a6e8713":
        raise ValueError("A scheduler capture must use the original fixed four-window source")

    cfg = configuration(
        [
            "bench",
            "--transport",
            "urma",
            "--input",
            "real",
            "--trace-root",
            str(a.trace),
            "--group",
            str(a.group),
            "--round-range",
            "1:1",
            "--device",
            "0",
            "--kvaf-root",
            str(a.sdk),
            "--kvaf-server",
            a.server,
            "--warmup",
            "4" if timed else "0",
            "--iterations",
            "10" if timed else "1",
            "--prefetch-rank-start",
            "1024",
            "--prefetch-tokens",
            "1024",
            "--intermediate-fetches",
            "4",
            "--output",
            str(a.out),
        ]
    )
    kwargs = {}
    proof = {}
    if timed:
        ref = np.empty((a.batch, 4, 2048), dtype=np.int32)
        for r in range(a.batch):
            for q in range(4):
                keys = inputs.logical_score_keys[r, q]
                ids = np.flatnonzero(inputs.logical_score_valid[r, q, : int(inputs.query_ends[r, q])])
                if ids.size < 2048:
                    raise ValueError(f"Missing valid causal scores for row {r}, Q{q + 1}")
                top = np.argpartition(keys[ids], -2048)[-2048:]
                ref[r, q] = np.sort(ids[top])
        inputs.validate_topk(ref)
        worker = partial(measure_packed, ex, proof, np.broadcast_to(ref, (10, *ref.shape)))
    if journal:

        kwargs = dict(large_window_policy="four-ranks")

        # run_path always executes one eager call as its correctness gate. Arm
        # diagnostics before that call and consume that very journal below.
        class JournalState(ex.FusedReplayState):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.forward.events_src.fill_(-1)
                self.forward.events_dst.fill_(-1)
                self.forward.terminal.times.zero_()
                self.forward.set_diagnostics(True)

        ex.FusedReplayState = JournalState  # Private subprocess for this one cell.

        def worker(torch, states, sequence, config, directory, *, fused):
            (s,) = states
            if rec and rec["cold_build_fingerprint"] != s.large_window_launch.source_fingerprint:
                raise ValueError("Diagnostic and measured LI executable differ")
            proof.update(terminal_image(s, a.sdk))
            if rev == "d0fc5963881bd38fa97eff0f6ef40c757a6e8713":
                from b24_team_schedule import capture
                proof["terminal_team_schedule"] = capture(s)
            proof.update(measured_calls=0, diagnostic_calls=1,
                         final_starts=s.forward.terminal.times[:, :, 22].cpu().tolist())
            s.forward.set_diagnostics(False)
            _, counts = read_selected(s, inputs, True)
            records = device_journal(s, inputs, pages, counts)
            path = a.out / "journal.npz"
            np.savez_compressed(path, **records)
            proof.update(
                journal_sha256=sha(path),
                chunks=len(records),
                requests=sum(map(len, records.values())),
                counts=counts,
            )
            return dict(correctness="PASS", failures=[])

    if a.mode == "frozen" or two_row or stagger:
        from four_rank_controlled import (GUARD, clear_evidence, controlled_build, evidence, extend_state,
                                          list_digest, validate_bound_addresses, validate_physical_operands, window_inputs)
        import large_two_cold
        from fixed_q4_packed_worker import FrozenPackedRound

        folder = Path(os.environ["PEER_FROZEN_LIST"])
        manifest = json.loads((folder / "journal.json").read_bytes())
        pinned = manifest["identity"]
        if (manifest["correctness"] != "PASS" or pinned["native"] != "d0fc5963881bd38fa97eff0f6ef40c757a6e8713"
                or not (folder / "placement_verified").is_file() or any(pinned[k] != v for k, v in wanted.items())
                or sha(folder / "journal.npz") != manifest["journal_sha256"]):
            raise ValueError("Incorrect, unmatched, or corrupted original ordered GET list")
        with np.load(folder / "journal.npz", allow_pickle=False) as j:
            records = {k: j[k] for k in j.files}
        ids, counts = window_inputs(records, a.batch, inputs.geometry["history"])
        validate_physical_operands(records, inputs, pages)
        large_two_cold.build = controlled_build(large_two_cold.build)
        if two_row:
            import acl
            from four_rank_final_map import archived_assignments
            control_folder = Path(os.environ["PEER_FROZEN_CONTROL"])
            control = json.loads((control_folder / "frozen.json").read_bytes())
            if (not (control_folder / "placement_verified").is_file()
                    or control.get("identity") != old or old["native"] != pinned["native"]
                    or control.get("arm") != "frozen" or control.get("correctness") != "PASS"
                    or control.get("calls") != 10 or control.get("get_workers") != 32
                    or control.get("checked_rows") != 10 * a.batch or control.get("full_KV_Index_checked_calls") != 10
                    or len(control.get("traffic", [])) != 10 or len(control.get("timed_journal_checks", [])) != 10
                    or control.get("frozen_journal_sha256") != manifest["journal_sha256"]
                    or control.get("guard_sha256") != sha(GUARD) or control.get("tail_text") != manifest["tail_text"]
                    or control.get("compiler_sha256") != manifest["compiler_sha256"]
                    or control.get("expected_window_sha256") != list_digest(records, "w")
                    or control.get("expected_final_sha256") != list_digest(records, "q")):
                raise ValueError("The original controlled arm and ordered GET list do not match this cell")
            recovered_maps = archived_assignments(records, a.batch)
            b24_two_row_attach.install(large_two_cold, a.timed, a.sdk / "build-npu", acl.get_soc_name())
            proof["original_control"] = str(control_folder)
            proof["original_control_sha256"] = sha(control_folder / "frozen.json")
        if stagger:
            import acl
            terminal_stagger_attach.install(large_two_cold, a.timed, a.sdk / "build-npu", acl.get_soc_name(), enabled)
        kwargs = dict(large_window_policy="four-ranks")

        class ControlledState(ex.FusedReplayState):
            def __init__(self, *args, **options):
                super().__init__(*args, **options)
                validate_bound_addresses(self, records)
                extend_state(self, ids, counts)
                if stagger:
                    terminal_stagger_attach.bind_final_plan(self, records, manifest.get("final_starts"))
                clear_evidence(self)

        ex.FusedReplayState = ControlledState
        base_reset, base_validate = FrozenPackedRound.reset, FrozenPackedRound.validate
        checks = []

        def reset(frozen, state):
            base_reset(frozen, state)
            clear_evidence(state)

        def validate(frozen, state, inp, index):
            base_validate(frozen, state, inp, index)
            count = frozen.traffic[-1]
            row_maps = []
            actual = device_journal(state, inp, pages, count, row_maps)
            seen, status = state.frozen_binding[2], state.frozen_binding[4]
            checks.append(evidence(records, actual, np.asarray(count["row_window_prefetch"]),
                                   seen.cpu().numpy(), status[:, :, 0].cpu().numpy()))
            checks[-1]["index_assignment_by_row"] = row_maps
            if two_row:
                checks[-1]["two_row_receipt"] = state.large_two_row_terminal.readback()
                if not checks[-1]["two_row_receipt"]["accepted"]:
                    raise ValueError("Measured device rejected a two-row binding")
                image = b24_two_row_attach.identity(state)
                if index == 0:
                    proof.update(image, compiler_sha256=compiler_identity(a.sdk))
                elif any(proof[k] != value for k, value in image.items()):
                    raise ValueError("The ten-call graph used different LI or terminal images")
            elif stagger:
                codes = state.captured_final_plan[:, :, 2].cpu().numpy()
                checks[-1]["native_final_guard_failures"] = int(np.count_nonzero(codes != 1))
                final_keys = [k for k in records if "_q" in k]
                checks[-1]["final_source_order_bad"] = sum(
                    not np.array_equal(actual[k][:,:4], records[k][:,:4]) for k in final_keys)
                checks[-1]["final_destinations_bad"] = sum(
                    not np.array_equal(actual[k][:,4], records[k][:,4]) for k in final_keys)
                image = terminal_stagger_attach.identity(state, enabled)
                if index == 0:
                    proof.update(image, compiler_sha256=compiler_identity(a.sdk))
                elif any(proof[k] != v for k, v in image.items()):
                    raise ValueError("Frozen graph mixed different gate settings or binaries")
            elif index == 0:
                proof.update(terminal_image(state, a.sdk))

        FrozenPackedRound.reset, FrozenPackedRound.validate = reset, validate
        proof.update(frozen_journal_sha256=manifest["journal_sha256"], guard_sha256=sha(GUARD),
                     original_journal=str(folder), timed_journal_checks=checks,
                     expected_window_sha256=list_digest(records, "w"), expected_final_sha256=list_digest(records, "q"))
    outcome = ex.run_path(cfg, a.out / "path", seq, pages, fused=fused, measure=worker, **kwargs)
    workers = 32 if fused else 2
    if outcome["correctness"] != "PASS" or outcome["failures"] or outcome["runtime_workers"] != workers:
        raise ValueError("Incorrect replay result or execution resource allocation")
    summary = dict(
        identity=dict(old, native=rev), arm=a.mode, correctness="PASS", get_workers=workers, **proof
    )
    if timed:
        summary.update(
            checked_queries=40 * a.batch,
            checked_rows=10 * a.batch,
            full_KV_Index_checked_calls=10,
            selection_oracle="captured_logical_scores",
        )
    if a.mode == "frozen":
        if len(checks) != 10 or proof["tail_text"] != manifest["tail_text"] or proof["compiler_sha256"] != manifest["compiler_sha256"]:
            raise ValueError("Expected ten measured journals and identical original tail image and compiler")
        summary["guarded_pass"] = all(c["ordered_match"] for c in checks)
        summary["scope"] = "controlled ten-call graph with guarded GET and tail diagnostics enabled"
    if two_row:
        from four_rank_final_map import compare
        if (proof["cold_build"] != control["cold_build"] or proof["cold_build_fingerprint"] != control["cold_build_fingerprint"]
                or proof["compiler_sha256"] != control["compiler_sha256"] or proof["original_tail_text"] != control["tail_text"]
                or proof["tail_text"] == control["tail_text"]):
            raise ValueError("The controlled pair must share the LI binary and compiler and exercise distinct tail images")
        summary.update(compare(control, summary, recovered_maps))
        summary["scope"] = "controlled ten-call graph with guarded GET and terminal journaling on both arms"
    if a.mode == "team-schedule":
        atomic_json(a.out / "journal.json", dict(summary, arm="journal"))
    path = {"team-schedule": "team-schedule.json", "serial": "packed_result.json", "journal": "journal.json", "frozen": "frozen.json",
            "two-row-frozen": "two-row-frozen.json", "stagger-pair": "stagger.json", "stagger-journal": "journal.json"}[a.mode]
    atomic_json(a.out / path, summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    from b24_window_policy import MODES
    p.add_argument("mode", choices=[*MODES, "journal", "serial", "frozen", "two-row-natural", "two-row-frozen", "stagger-pair", "stagger-journal", "team-schedule"])
    for name in ("timed", "ref"):
        p.add_argument(name, type=Path)
    for name in ("group", "batch"):
        p.add_argument(name, type=int)
    for name in ("trace", "sdk"):
        p.add_argument(name, type=Path)
    p.add_argument("server")
    for name in ("bank", "out"):
        p.add_argument(name, type=Path)
    run(p.parse_args())
