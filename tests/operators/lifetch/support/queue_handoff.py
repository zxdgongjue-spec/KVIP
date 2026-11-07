# SPDX-License-Identifier: Apache-2.0
"""Untimed native/fused queue handoff audit using owned device allocations."""

import ctypes
import json
import struct

from kvip.runtime.loader import load_native_get


def decode_queues(raw):
    if len(raw) % 64:
        raise ValueError("KVAF queue snapshot must contain complete 64-byte records")
    names = ("magic", "initialized", "sq", "cq", "completed", "error", "poisoned", "reserved0", "reserved1")
    return [dict(zip(names, values)) for values in struct.iter_unpack("<II7Q", raw)]


def check_handoff(torch, owner, folder):
    import acl

    stream = torch.npu.current_stream(owner.device)
    stream.synchronize()
    lease = owner.lease
    workers = lease["worker_count"]
    if owner.abi_sizes["queue"] != 64:
        raise ValueError("Unexpected KVAF queue ABI; refusing raw snapshot")
    records = []

    def snapshot(label):
        stream.synchronize()
        size = workers * lease["jettys_per_core"] * 64
        host = ctypes.create_string_buffer(size)
        # The source is the live lease's device allocation, never dereference it on CPU.
        code = acl.rt.memcpy(ctypes.addressof(host), size, lease["device_queue_state"], size, 2)
        if code:
            raise RuntimeError(f"Queue D2H failed: {code}")
        queues = decode_queues(host.raw)
        records.append({"phase": label, "queues": queues})
        (folder / "queue_handoff.json").write_text(json.dumps(records, indent=2))
        bad = [
            (i, q)
            for i, q in enumerate(queues)
            if q["initialized"] and (q["magic"] != 0x4B564151 or q["error"] or q["poisoned"])
        ]
        print(f"QUEUE HANDOFF {label}: queues={len(queues)} unhealthy={len(bad)} first={bad[:2]}", flush=True)
        return bad

    if snapshot("after_fused_before_native"):
        raise AssertionError("Queue already unhealthy before native Get; see queue_handoff.json")
    library, launch = load_native_get(owner.session.runtime)
    fields = list(struct.unpack("=10Q8I", bytes(owner.device_args[0].cpu().tolist())))
    fields[15], fields[16] = workers, lease["jettys_per_core"]
    blob = ctypes.create_string_buffer(struct.pack("=10Q8I", *fields))
    for count in (0, 128, 0):
        owner.descriptor_scratch.zero_()
        owner.descriptor_scratch[1, :count].copy_(torch.arange(count, dtype=torch.int32, device=owner.device) * 656)
        owner.descriptor_scratch[2, 0] = count
        owner.descriptor_scratch[3, :count].copy_(torch.arange(count, dtype=torch.int32, device=owner.device))
        owner.mark_launched()
        owner.core_status.zero_()
        code = launch(blob, stream.npu_stream)
        if code:
            raise RuntimeError(f"Native handoff launch failed: {code}")
        bad = snapshot(f"native_count_{count}")
        status = owner.core_status[:workers].cpu().numpy().copy().view("<u8")
        # KVAF ABI: magic, processed count, result. Preserve raw status on failure.
        errors = [(i, int(row[2])) for i, row in enumerate(status) if row[2]]
        records[-1]["native_errors"] = errors
        (folder / "queue_handoff.json").write_text(json.dumps(records, indent=2))
        print(f"QUEUE HANDOFF native_count={count}: errors={errors[:8]}", flush=True)
        if int(status[:, 1].sum()) != count or not (status[:, 0] == 0x4B5641464247444F).all():
            raise AssertionError("Native Get completion record missing or count differs")
        if bad or errors:
            raise AssertionError("Native/fused queue handoff failed; see queue_handoff.json")
        owner.mark_completed(completion_verified=True)
        if count:
            torch.testing.assert_close(owner.output[0, :count].cpu(), owner.session.cpu_payload[0, :count])
    del library
