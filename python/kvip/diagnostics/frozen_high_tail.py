# SPDX-License-Identifier: Apache-2.0
"""Three counterfactual URMA tail orders on one live, verified HIGH event stream."""

import ctypes
from itertools import permutations
import json
from pathlib import Path
from statistics import mean, median

import numpy as np

from kvip.diagnostics.archive import cpu_array
from kvip.diagnostics.high_index_profile import (
    decode_high_index,
    descriptor_fingerprint,
    event_work,
    verify_permutation,
)
from kvip.diagnostics.snapshot import persistent_tensors


ARMS = ("get_only", "overlap", "wait")
ORDERS = tuple(permutations(ARMS))


def compact_statistics(values):
    return dict(
        mean=mean(values),
        p50=median(values),
        minimum=min(values),
        maximum=max(values),
        second_half_minus_first=median(values[len(values) // 2 :]) - median(values[: len(values) // 2]),
    )


def paired_values(samples, key):
    if len(samples) != len(ORDERS) or any(tuple(row["order"]) != expected for row, expected in zip(samples, ORDERS)):
        raise ValueError("Expected all six counterbalanced permutations of one frozen workload")

    def times(arm):
        return [float(row["arms"][arm][key]) for row in samples]

    values = {name: times(name) for name in ARMS}
    if any(not np.isfinite(v).all() or min(v) <= 0 for v in values.values()):
        raise ValueError("Missing positive measured time")
    for name, (a, b) in {"exposed": ("overlap", "get_only"), "saved": ("wait", "overlap")}.items():
        values[name] = [x - y for x, y in zip(values[a], values[b])]
    return {name: compact_statistics(samples) for name, samples in values.items()}


def descriptor_slots(posts, destinations, output_rows):
    chunks = [destinations[int(v[4]) : int(v[4] + v[0])] for v in posts.reshape(-1, 32) if v[0]]
    if not chunks:
        raise ValueError("An overlap measurement requires real final Gets")
    slots = np.unique(np.concatenate(chunks))
    if slots.min() < 0 or slots.max() >= output_rows:
        raise ValueError("Final KV destination outside output")
    return slots.astype(np.int64)


def device_spans(marks, arm, rate):
    order = {"get_only": (3, 7, 13), "overlap": (3, 4, 5, 6, 7, 13), "wait": (3, 7, 4, 5, 6, 13)}[arm]
    if (
        not np.isfinite(rate)
        or rate <= 0
        or marks.shape != (64, 32)
        or np.any(marks[:, order] <= 0)
        or np.any(np.diff(marks[:, order].astype(np.int64), axis=1) < 0)
    ):
        raise ValueError("Invalid final Post/Index/CQ phase markers")

    def span(begin, end):
        return (int(marks[:, end].max()) - int(marks[:, begin].min())) / rate

    return dict(
        joined_us=span(3, 13),
        index_us=0 if arm == "get_only" else span(4, 5),
        cq_us=span(6 if arm == "overlap" else 3, 7),
    )


def format_frozen(report):
    def fields(key):
        data = report[key]
        diff = data["exposed"]
        return (
            f"{key}_get/overlap/wait_p50="
            + "/".join(f"{data[k]['p50']:.2f}" for k in ARMS)
            + f" exposed_mean/p50/min/max/drift={diff['mean']:.2f}/{diff['p50']:.2f}/"
            f"{diff['minimum']:.2f}/{diff['maximum']:.2f}/{diff['second_half_minus_first']:.2f}"
            f" saved_p50={data['saved']['p50']:.2f}"
        )

    return (
        f"FROZEN_TAIL final_Get={report['final_get']} events={report['event_count']} "
        f"pairs={len(report['probed'])} units=us "
        + fields("joined")
        + " "
        + fields("event")
        + " check=identical_events_descriptors_KV_Index"
    )


def measure_frozen(torch, stream, bundle, library_path, revision, reference, rate):
    library_path = Path(library_path)
    # Preserve the declared revision next to the public symlink.
    if json.loads((library_path.parent / "revision.json").read_text())["revision"] != revision:
        raise ValueError("Frozen replay library and production plan revisions differ")
    lib = ctypes.CDLL(str(library_path.absolute()))
    launch = lib.kvip_replay_fixed_tail
    launch.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]
    launch.restype = ctypes.c_uint32
    forward, rot = bundle.forward, bundle.rotating
    persistent = persistent_tensors(bundle.selector)

    def host(tensor):
        return cpu_array(torch, tensor)

    flags, src, dst = (host(v) for v in (forward.flags, forward.events_src, forward.events_dst))
    posts = host(rot.final_posts)
    count = sum(reference["final_get_by_core"])

    def fingerprint():
        work = event_work(
            host(forward.flags), host(forward.events_src), host(forward.events_dst), history=forward.history
        )
        work["descriptor_sha256"] = descriptor_fingerprint(
            host(rot.final_posts),
            [host(v) for v in (rot.blocks, rot.offsets, rot.destinations)],
            host(rot.fallback_addresses),
            final_get=count,
            arena_base=bundle.restored["new_base"],
        )
        return work

    work = fingerprint()
    if any(reference[k] != v for k, v in work.items()) or reference["fallback"]:
        raise ValueError("Frozen replay requires the captured nonblocking final Post descriptors")
    golden = {k: t.clone() for k, t in persistent.items()}
    scratch = forward._high_index_storage[: forward.batch * 20480 * 160]
    private_plan, headers = scratch.clone(), forward.worker_status.clone()
    # Verify that the host reads the production private header before it is reused by Post.
    preplan = reference["preplan"]
    h = host(headers)
    if (
        np.any(h[:, 0])
        or not np.array_equal(h[:, 3], preplan["row_pairs"])
        or not np.array_equal(2 * h[:, 3] + h[:, 4], preplan["row_moves"])
    ):
        raise ValueError("Frozen private plan differs from the verified original launch")

    width = int(rot.runtime_shape.tokenBytes)
    slots = descriptor_slots(posts, host(rot.destinations), golden["output"].numel() // width)
    output = persistent["output"].view(torch.uint8).reshape(-1, width)
    device_slots = torch.from_numpy(slots).to(output.device)
    golden_rows = golden["output"].view(torch.uint8).reshape_as(output)
    poison = torch.bitwise_xor(golden_rows.index_select(0, device_slots), 255)

    # Find the final state of hot_valid at the Index join, before production publishes new residency.
    bundle.reset()
    seeds = {t.data_ptr(): seed for t, seed in bundle.seeds}
    before = {k: seeds[t.data_ptr()] for k, t in persistent.items()}
    valid = host(before["hot_valid"]).copy()
    owners = host(rot.row_owners).ravel()
    invalid = []
    verify_permutation(
        flags,
        src,
        dst,
        host(before["logical_ids"]),
        host(golden["logical_ids"]),
        owners,
        resident=forward.resident,
        invalidated_slots=invalid,
    )
    for owner, cleared in zip(owners, invalid):
        valid[owner, cleared] = 0
    golden["hot_valid"] = torch.from_numpy(valid).to(output.device)
    args_blob = host(forward.workspace).copy()
    options = args_blob[96:100].view("<u4")
    common = int(options[0]) & ~(0x02000000 | 0x80000000 | 0x00800000)
    args_device = {}
    for arm in ARMS:
        for profile in (False, True):
            options[0] = (
                common
                | (0x80000000 if profile else 0)
                | {"get_only": 0x00800000, "overlap": 0, "wait": 0x02000000}[arm]
            )
            args_device[arm, profile] = torch.from_numpy(args_blob.copy()).to(output.device)

    def run(arm, profile):
        # Only restore the five Index tensors and private plan. No queue cursor,
        # completion, entity, or external transport binding is ever rewound.
        for name, value in before.items():
            if name != "output":
                persistent[name].copy_(value)
        persistent["output"].copy_(golden["output"])
        output.index_copy_(0, device_slots, poison)
        scratch.copy_(private_plan)
        forward.worker_status.copy_(headers)
        forward.times.zero_()
        forward.post_index.zero_()
        stream.synchronize()
        start, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
        start.record(stream)
        rc = launch(64, stream.npu_stream, args_device[arm, profile].data_ptr())
        if rc:
            raise RuntimeError(f"Fixed HIGH tail launch failed rc={rc}")
        end.record(stream)
        end.synchronize()
        measured = dict(event_us=float(start.elapsed_time(end)) * 1000)
        result = decode_high_index(host(forward.post_index), None, wait_first=arm == "wait", final_get=count, work=work)
        if result["fallback"] or result["final_get_by_core"] != reference["final_get_by_core"]:
            raise ValueError("Frozen HIGH submit/completion or per-core load changed")
        for name, actual in persistent.items():
            target = before[name] if arm == "get_only" and name != "output" else golden[name]
            if not torch.equal(actual.view(torch.uint8), target.view(torch.uint8)):
                raise ValueError(f"Frozen arm={arm} {name} differs after transfer")
        if profile:
            raw = host(forward.times)[forward.batch * 256 :].reshape(64, 32)
            measured.update(device_spans(raw, arm, rate))
            if arm != "get_only" and not np.array_equal(raw[:, 10], reference["moved"]):
                raise ValueError("Frozen per-core physical Index moves changed")
        return measured

    for arm in ARMS:
        run(arm, False)

    def sweep(profile):
        return [dict(order=list(order), arms={arm: run(arm, profile) for arm in order}) for order in ORDERS]

    uninstrumented, probed = sweep(False), sweep(True)
    if fingerprint() != work:
        raise ValueError("Frozen descriptor storage changed during the paired measurement")
    # Future optional lifecycle tests restore their own state explicitly.
    return dict(
        scope="Private plan ready; production packing, final Post, Index, CQ and last join. "
        "Earlier speculative CQs drained; residency and feedback publication outside.",
        **work,
        final_get=count,
        moved=sum(reference["moved"]),
        uninstrumented=uninstrumented,
        probed=probed,
        joined=paired_values(probed, "joined_us"),
        event=paired_values(uninstrumented, "event_us"),
    )
