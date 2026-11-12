# SPDX-License-Identifier: Apache-2.0
"""One public LIFetch pipeline for URMA and UBmem."""

from kvip.operators.lifetch.streaming import StreamingLIFetchForward


class LIFetchForward(StreamingLIFetchForward):
    def __init__(self, rotating, update, *, q_workers=2, aiv_budget=None):
        super().__init__(
            rotating, update, rotating.torch, rotating.prefetch_limit, fused_final=True, q_workers=q_workers, aiv_budget=aiv_budget
        )
        # Prepare address/logical metadata as each Q arrives. Finalize ordering
        # remains the measured baseline; transport-specific resources stay leased.
        self.prepare_metadata = True
        self.ubmem_overlap_fetch = hasattr(rotating, "lease")
        self.parallel_finalize = self.ubmem_overlap_fetch
        self.set_options()
