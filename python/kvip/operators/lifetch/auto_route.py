# SPDX-License-Identifier: Apache-2.0
"""Persistent, device-owned feedback for one group's fixed captured graph.

The scheduler supplies request identity and step metadata, never a chosen mode
or device counters. All decision/feedback work executes inside existing kernels.
"""

import numpy as np


class AutoRouteState:
    """Allocate once; do not share this state across independent layer groups.

Budgets compare predicted necessary Get bytes * 1024 with the sum of active
query history lengths. The default band is an experimental calibration for
the current A5 replay, not a portable optimum. No state is reset by LIFetch's
per-call scratch reset. Submission, metadata copies and graph replay must use
the same ordered stream; do not execute this state on concurrent streams.
"""

    def __init__(self, torch, device, batch, owners, *, low_budget=128,
                 high_budget=192, full_controls=(512, 64, 0, 0), max_step_gap=1):
        if type(batch) is not int or not 1 <= batch <= 16:
            raise ValueError("Auto route requires BS1..16")
        if type(owners) is not int or not batch <= owners <= 65536:
            raise ValueError("Auto route owners must cover the batch and fit 1..65536")
        if (type(low_budget) is not int or type(high_budget) is not int or
                not 1 <= low_budget < high_budget <= 65535):
            raise ValueError("Auto route budgets require 1 <= low < high <= 65535")
        if type(max_step_gap) is not int or not 1 <= max_step_gap <= 1024:
            raise ValueError("Auto route max_step_gap must be in 1..1024")
        bounds = ((1, 2048), (1, 480), (0, 65535), (0, 1))
        if (len(full_controls) != 4 or
                any(type(v) is not int or not lo <= v <= hi
                    for v, (lo, hi) in zip(full_controls, bounds))):
            raise ValueError("Invalid auto route full admission controls")
        if getattr(device, "type", None) != "npu":
            raise ValueError("Auto route storage must be on an NPU")

        self.torch, self.device = torch, device
        self.batch, self.owners = batch, owners
        self.full_controls = tuple(full_controls)
        self.requests = torch.zeros((batch, 4), dtype=torch.int64, device=device)
        self.feedback = torch.zeros((owners, 16), dtype=torch.int64, device=device)
        self.pending = torch.zeros((batch, 16), dtype=torch.int64, device=device)
        words = [1, owners, self.requests.data_ptr(), self.feedback.data_ptr(),
                 self.pending.data_ptr(), low_budget, high_budget, max_step_gap,
                 0, *full_controls, 0, 0, 0]
        values = np.asarray(words, dtype="<u8").view("<i8")
        self.config = torch.from_numpy(values).to(device)

    def _outside_capture(self):
        capturing = getattr(getattr(self.torch, "npu", None), "is_current_stream_capturing", None)
        if callable(capturing) and capturing():
            raise RuntimeError("Queue auto route metadata before replay, outside capture")

    def set_requests(self, values):
        """Queue CPU int64[B,4]: stable owner, generation, step, active queries.

An owner generation must increase when its slot is assigned to a new request;
step must increase only for real decode work. Padding uses active queries=0.
Generation/step repetition and rollback are also checked on device. This
method validates host metadata only, without reading or waiting on the NPU.
"""
        self._outside_capture()
        if getattr(getattr(values, "device", None), "type", "cpu") != "cpu":
            raise ValueError("Auto route requests must be a CPU int64 table")
        values = np.asarray(values)
        if values.dtype != np.int64 or values.shape != (self.batch, 4) or not values.flags.c_contiguous:
            raise ValueError("Auto route requests must be contiguous int64[B,4]")
        if (np.any(values < 0) or np.any(values[:, 0] >= self.owners) or
                np.any(values[:, 3] > 4)):
            raise ValueError("Invalid auto route owner/generation/step/query count")
        active = values[values[:, 3] > 0, 0]
        if len(set(active.tolist())) != active.size:
            raise ValueError("Active auto route owners must be unique")
        self.requests.copy_(self.torch.from_numpy(values.copy()), non_blocking=True)

    def set_learning(self, enabled):
        """Disable while capturing, profiling warmup or replaying fake steps."""
        self._outside_capture()
        if type(enabled) is not bool:
            raise ValueError("Auto route learning flag must be bool")
        value = self.torch.from_numpy(np.array([int(enabled)], dtype=np.int64))
        self.config[8:9].copy_(value, non_blocking=True)

    def validate_binding(self, batch, owners, device):
        if self.batch != batch or self.owners != owners or self.device != device:
            raise ValueError("Auto route state must match this group's device, batch and stable owners")
