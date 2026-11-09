# SPDX-License-Identifier: Apache-2.0
"""Own the KVAF CPU-service DDR mapping lease for UBmem fusion and captured graphs."""

from contextlib import contextmanager

import numpy as np


class UbmemLease:
    def __init__(self, torch, runtime, output, stream):
        if runtime.transport != "ubmem":
            raise ValueError("UBmem fusion requires a UBmem KVAF runtime")
        self.torch, self.runtime, self.device, self.stream = torch, runtime, output.device, stream
        self.context = runtime.acquire_ubmem_fused_context(output, stream)
        self.closed = self.capturing_startup = self.capture_fence_failed = False
        self.outputs = {self.output_key(output): (output, self.context)}
        self.bindings = []
        self.graphs = []
        self._validate(self.context, output)

    @staticmethod
    def output_key(output):
        return output.data_ptr(), output.numel() * output.element_size()

    def _validate(self, context, output):
        if context["abi_version"] != 1 or context["device_id"] != self.device.index:
            raise ValueError("KVAF UBmem lease ABI/device mismatch")
        if context["output_address"] != output.data_ptr() or context["output_bytes"] != self.output_key(output)[1]:
            raise ValueError("KVAF UBmem lease output differs from its retained tensor")
        if (
            context["slot_bytes"] < context["token_bytes"] * context["tokens_per_block"]
            or context["region_bytes"] % context["slot_bytes"]
        ):
            raise ValueError("KVAF UBmem lease slot geometry is inconsistent")
        for base, size in (
            (context["region_base"], context["region_bytes"]),
            (context["mapped_base"], context["region_bytes"]),
            self.output_key(output),
        ):
            if base <= 0 or size <= 0 or base + size > (1 << 64) - 1:
                raise ValueError("KVAF UBmem lease address span is invalid")
        if not context["lease_id"]:
            raise ValueError("KVAF UBmem requires a live nonzero lease")

    def check_stream(self):
        if self.closed or self.capture_fence_failed:
            raise RuntimeError("UBmem lease is released or its capture completion is unknown")
        stream = self.torch.npu.current_stream(self.device)
        if stream.npu_stream != self.stream.npu_stream and not self.capturing_startup:
            raise RuntimeError("UBmem fusion must use the lease execution stream")
        return stream

    def register_output(self, output):
        self.check_stream()
        key = self.output_key(output)
        if key not in self.outputs:
            context = self.runtime.register_ubmem_fused_output(self.context, output)
            self._validate(context, output)
            self.outputs[key] = output, context
        return self.outputs[key][1]

    def register_table(self, table):
        self.check_stream()
        self.runtime.register_ubmem_fused_access_table(self.context, table)

    def retain_graph(self, graph):
        self.check_stream()
        self.runtime.retain_ubmem_fused_graph(self.context, graph)
        self.graphs.append(graph)

    @contextmanager
    def capture_scope(self):
        self.check_stream()
        if self.capturing_startup:
            raise RuntimeError("Nested UBmem capture scopes are unsupported")
        self.torch.npu.synchronize(self.device)
        self.capturing_startup = True
        try:
            yield
        finally:
            try:
                self.torch.npu.synchronize(self.device)
            except BaseException:
                self.capture_fence_failed = True
                raise
            finally:
                self.capturing_startup = False

    def region(self, output):
        context = self.register_output(output)
        # Access table entries are CPU service VAs; translate by offset into
        # the same KVAF-owned NPU mapping. Both remain valid for this lease.
        base = context["region_base"]
        fields = [
            0x55424D454D563931,
            1,
            base,
            context["mapped_base"],
            context["region_bytes"],
            context["lease_id"],
            output.data_ptr(),
            context["output_bytes"],
            context["device_id"],
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        return self.torch.from_numpy(np.asarray(fields, dtype="<u8").view("u1").copy()).to(self.device)

    def close(self):
        if not self.closed:
            if self.capture_fence_failed:
                raise RuntimeError("UBmem completion is unknown; retain the lease until process exit")
            self.runtime.release_ubmem_fused_context(self.context, self.stream)
            self.closed = True
            self.bindings.clear()
            self.outputs.clear()
            self.graphs.clear()


class UbmemLeasedRuntime:
    """Keep ordinary Put/Get on KVAF; the wrapper owns fusion lifetime only."""

    def __init__(self, lease):
        self.lease = lease

    def __getattr__(self, name):
        return getattr(self.lease.runtime, name)

    def shutdown(self):
        self.lease.close()
        self.lease.runtime.shutdown()
