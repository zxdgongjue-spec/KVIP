"""Independent physical budgets for the current mixed LI and AIV-only tail."""

from dataclasses import dataclass


def split_score_tasks(batch, begin, end, aics, aivs, route="auto"):
    """Enumerate exact Q/H coverage under the independent MIX1:2 budget."""
    if not (
        1 <= batch <= 32
        and 0 <= begin < end
        and begin % 128 == end % 128 == 0
        and 1 <= aics <= 32
        and 2 <= aivs <= 64
        and route in ("auto", "history", "bsq")
    ):
        raise ValueError("Unsupported split score geometry or route")
    capacity = min(aics, aivs // 2)
    qg = 4 if route == "history" or (route == "auto" and 2 * batch <= capacity) else 2
    tiles = (end - begin) // 128
    producers = max(1, min(capacity // batch, tiles)) if qg == 4 else 1
    tasks = [
        (b, q, begin + tiles * p // producers * 128, begin + tiles * (p + 1) // producers * 128)
        for b in range(batch)
        for q in range(0, 4, qg)
        for p in range(producers)
    ]
    return qg, producers, tasks


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


@dataclass(frozen=True)
class StagedStreamingSchedule:
    batch: int
    aic_budget: int
    aiv_budget: int
    li_aic: int
    fetch_workers_per_query: int = 2

    def __post_init__(self):
        if not (
            1 <= self.batch <= 32
            and self.batch <= self.aiv_budget <= 64
            and 1 <= self.li_aic <= self.aic_budget <= 32
            and 2 * self.li_aic <= self.aiv_budget
            and self.fetch_workers_per_query in (1, 2, 4)
            and self.owner_capacity > 0
        ):
            raise ValueError("Unsupported staged owner/core budget")

    @property
    def owner_capacity(self):
        return min(16, (self.aiv_budget - 4 * self.fetch_workers_per_query) // 3)

    @property
    def urma_queues(self):
        return 2 * self.batch + 8 * self.fetch_workers_per_query
