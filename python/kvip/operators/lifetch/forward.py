# SPDX-License-Identifier: Apache-2.0
"""One public LIFetch pipeline for URMA and UBmem."""

from kvip.operators.lifetch.streaming import StreamingLIFetchForward
from kvip.operators.lifetch.staged import StagedLIFetchForward


class LIFetchForward(StagedLIFetchForward):
    def __init__(self, rotating, update, *, q_workers=2, aiv_budget=None, score_route=None,
                 score_post=None, score_prepare=None):
        self.staged_score = getattr(rotating, "staged", False)
        route = (
            getattr(rotating, "score_route", "auto" if self.staged_score else "legacy")
            if score_route is None
            else score_route
        )
        post = getattr(rotating, "score_post", "separate") if score_post is None else score_post
        prepare = getattr(rotating, "score_prepare", "wave") if score_prepare is None else score_prepare
        if self.staged_score:
            super().__init__(rotating, update, q_workers=q_workers, aiv_budget=aiv_budget,
                             route=route, score_post=post, score_prepare=prepare)
            return
        if route != "legacy" or post != "separate" or prepare != "wave":
            raise ValueError("Prepare LIFetchBinding(score_route=...) before enabling split score")
        StreamingLIFetchForward.__init__(
            self,
            rotating,
            update,
            rotating.torch,
            rotating.prefetch_limit,
            fused_final=True,
            q_workers=q_workers,
            aiv_budget=aiv_budget,
        )
        # Prepare address/logical metadata as each Q arrives. Finalize ordering
        # remains the measured baseline; transport-specific resources stay leased.
        self.prepare_metadata = True
        self.ubmem_overlap_fetch = hasattr(rotating, "lease")
        self.parallel_finalize = self.ubmem_overlap_fetch
        self.set_options()

    def __call__(self):
        if self.staged_score:
            return StagedLIFetchForward.__call__(self)
        return StreamingLIFetchForward.__call__(self)
