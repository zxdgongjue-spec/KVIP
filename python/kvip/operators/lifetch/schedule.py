"""Independent physical budgets for the current mixed LI and AIV-only tail."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamingSchedule:
    batch: int
    aic_budget: int
    aiv_budget: int
    li_aic: int
    fetch_workers_per_query: int = 2

    def __post_init__(self):
        values = (self.batch, self.aic_budget, self.aiv_budget, self.li_aic, self.fetch_workers_per_query)
        if any(type(x) is not int for x in values):
            raise ValueError("Core budgets and worker counts must be integers")
        if not (1 <= self.batch <= 16 and 1 <= self.aic_budget <= 32 and 1 <= self.aiv_budget <= 64):
            raise ValueError("Unsupported batch or physical AIC/AIV budget")
        if not 1 <= self.li_aic <= self.aic_budget or 2 * self.li_aic > self.aiv_budget:
            raise ValueError("Current mixed LI implementation exceeds its AIC/AIV budget")
        if self.fetch_workers_per_query not in (1, 2, 4) or self.tail_aiv > self.aiv_budget:
            raise ValueError("Streaming resolver/Fetch workers exceed the independent AIV budget")

    @property
    def tail_aiv(self):
        return 3 * self.batch + 4 * self.fetch_workers_per_query

    @property
    def urma_queues(self):
        return 2 * self.batch + 8 * self.fetch_workers_per_query
