"""Native Q4/Q2 geometry scaled to the active logical device's AIC budget."""

MAX_SEQUENCE_TOKENS = 512 * 1024
# vLLM reserves three speculative positions after max_model_len for a Q4 step.
MAX_HISTORY_TOKENS = ((MAX_SEQUENCE_TOKENS + 3 + 255) // 256) * 256
MAX_RESIDENT_TOKENS = 128 * 1024
LARGE_BATCHES = (20, 24, 28, 32)


def validate_history(history, resident=None):
    """Check the full sequence capacity and the separately sized hot bitmap."""
    if not 0 < history <= MAX_HISTORY_TOKENS or history % 256:
        raise ValueError(f"LIFetch history must align to 256 and be in 1..{MAX_HISTORY_TOKENS}")
    if resident is not None and (
        not 2048 <= resident <= MAX_RESIDENT_TOKENS or resident >= history or resident % 256
    ):
        raise ValueError(f"LIFetch resident must align to 256, be in 2048..{MAX_RESIDENT_TOKENS} and below history")


def prefetch_budget(prefetch_limit=512, window_limit=None, intermediate_fetches=8):
    """Limit speculative fetches only; final TopK repair has its own capacity."""
    if not isinstance(intermediate_fetches, int) or intermediate_fetches < 1:
        raise ValueError("Prefetch requires a positive window count")
    if window_limit is None:
        window_limit = max(1, prefetch_limit // intermediate_fetches)
    if not 0 <= prefetch_limit <= 2048 or not 1 <= window_limit <= 480:
        raise ValueError("Invalid prefetch budgets: total must be 0..2048 and window 1..480")
    return prefetch_limit, window_limit


def li_geometry(batch, tiling="auto", intermediate_fetches=32, aic_count=32, service_aics=0):
    if not 1 <= batch <= 32 or tiling not in ("auto", "q4"):
        raise ValueError("Requires B1..32 and LI tiling auto/q4")
    if not isinstance(intermediate_fetches, int) or intermediate_fetches < 0:
        raise ValueError("intermediate_fetches must be a nonnegative integer")
    if not isinstance(aic_count, int) or not 1 <= aic_count <= 32:
        raise ValueError("aic_count must be in 1..32")
    if batch in LARGE_BATCHES and tiling == "auto" and aic_count == 32:
        rounds = min(intermediate_fetches, aic_count)
        # Each row's primary group publishes one prefetch stream in every window.
        if service_aics not in (0, batch // 2):
            raise ValueError("Large Q4 batches require one prefetch lane per request")
        primaries = list(range(batch))
        return dict(
            service_aics_per_window=batch // 2, service_aivs_per_window=batch,
            service_groups=[primaries.copy() for _ in range(rounds)],
            li_tiling=tiling, available_aic=aic_count, logical_aic_tasks=aic_count,
            cores_per_batch=1, queries_per_aic=4, queries_per_aiv=2,
            rotating_aic=aic_count, rotating_aiv=2 * aic_count,
            intermediate_fetches_requested=intermediate_fetches, intermediate_fetches=rounds,
            service_aic=primaries,
        )
    cores_per_batch = 2 if tiling == "auto" and 2 * batch <= aic_count else 1
    cores = min(aic_count, batch * cores_per_batch)
    rounds = min(intermediate_fetches, cores)
    maximum = cores // rounds if rounds else 1
    if not isinstance(service_aics, int) or service_aics < 0 or (rounds and service_aics > maximum):
        raise ValueError("service_aics exceeds active AICs/windows")
    group = (service_aics or maximum) if rounds else 1
    order = list(range(0, cores, 2)) + list(range(1, cores, 2)) if cores_per_batch == 2 else list(range(cores))
    groups = [order[e * group : (e + 1) * group] for e in range(rounds)]
    return dict(
        service_aics_per_window=group,
        service_aivs_per_window=2 * group,
        service_groups=groups,
        li_tiling=tiling,
        available_aic=aic_count,
        logical_aic_tasks=batch * cores_per_batch,
        cores_per_batch=cores_per_batch,
        queries_per_aic=4 // cores_per_batch,
        queries_per_aiv=2 // cores_per_batch,
        rotating_aic=cores,
        rotating_aiv=2 * cores,
        intermediate_fetches_requested=intermediate_fetches,
        intermediate_fetches=min(intermediate_fetches, cores),
        service_aic=[core for group in groups for core in group],
    )


def trigger_tokens(batch, history, resident, tiling="auto", intermediate_fetches=32, aic_count=32):
    """Hot-ready first window, followed by evenly spaced cold boundaries."""
    rounds = li_geometry(batch, tiling, intermediate_fetches, aic_count)["intermediate_fetches"]
    cold_tiles = (history - resident) // 128
    validate_history(history, resident)
    if cold_tiles <= rounds:
        raise ValueError("LIFetch requires more cold tiles than windows")
    return [resident + 128 * (cold_tiles * e // max(1, rounds - 1)) for e in range(rounds)]


def final_aivs(batch, tiling="auto", aic_count=32, requested=0):
    if batch in LARGE_BATCHES and tiling == "auto" and aic_count == 32:
        if requested not in (0, 4):
            raise ValueError("Large Q4 batches use four final workers per request")
        return 4
    cores = li_geometry(batch, tiling, 0, aic_count)["rotating_aiv"]
    workers = requested or (4 if cores >= batch * 4 else (2 if cores >= batch * 2 else 1))
    if workers not in (1, 2, 4) or workers * batch > cores:
        raise ValueError("Final AIV count exceeds active LI budget")
    return workers
