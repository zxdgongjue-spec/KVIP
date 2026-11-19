# SPDX-License-Identifier: Apache-2.0
"""One public LIFetch pipeline for URMA and UBmem."""

from kvip.operators.lifetch.streaming import StreamingLIFetchForward


class LIFetchForward(StreamingLIFetchForward):
    def __init__(self, rotating, update, *, q_workers=2, fetch_jettys_per_aiv=2):
        super().__init__(
            rotating, update, rotating.torch, rotating.prefetch_limit, fused_final=True, q_workers=q_workers,
            fetch_jettys_per_aiv=fetch_jettys_per_aiv,
        )
        # Prepare metadata as each Q arrives and publish final residency on its
        # Fetch workers. Transport-specific resources stay leased.
        self.prepare_metadata = True
        self.ubmem_overlap_fetch = hasattr(rotating, "lease")
        self.parallel_finalize = True
        self.set_options()
