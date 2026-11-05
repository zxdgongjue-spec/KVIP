"""Native Q4/Q2 geometry scaled to the active logical device's AIC budget."""


def li_geometry(batch, tiling="auto", intermediate_fetches=32, aic_count=32, service_aics=0):
    if not 1 <= batch <= 32 or tiling not in ("auto", "q4"):
        raise ValueError("Requires B1..32 and LI tiling auto/q4")
    if not isinstance(intermediate_fetches, int) or intermediate_fetches < 0:
        raise ValueError("intermediate_fetches must be a nonnegative integer")
    if not isinstance(aic_count, int) or not 1 <= aic_count <= 32:
        raise ValueError("aic_count must be in 1..32")
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
    if history % 256 or resident % 256 or resident < 2048 or cold_tiles <= rounds:
        raise ValueError("Requires aligned H>R>=2048 and more cold tiles than windows")
    return [resident + 128 * (cold_tiles * e // max(1, rounds - 1)) for e in range(rounds)]


def final_aivs(batch, tiling="auto", aic_count=32, requested=0):
    cores = li_geometry(batch, tiling, 0, aic_count)["rotating_aiv"]
    workers = requested or (4 if cores >= batch * 4 else (2 if cores >= batch * 2 else 1))
    if workers not in (1, 2, 4) or workers * batch > cores:
        raise ValueError("Final AIV count exceeds active LI budget")
    return workers
