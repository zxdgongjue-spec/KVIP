"""Native Q4/Q2 geometry scaled to the active logical device's AIC budget."""

import struct


def checkpoint_topk_options(initial_hot_rank=2048, topk_checkpoints=(), reuse_checkpoint_topk=False,
                            reverse_cold=False):
    windows = tuple(topk_checkpoints)
    if not isinstance(initial_hot_rank, int) or not 1 <= initial_hot_rank <= 2048:
        raise ValueError("initial_hot_rank must be in 1..2048")
    if (len(windows) > 6 or any(not isinstance(window, int) or not 1 <= window <= 30 for window in windows)
            or windows != tuple(sorted(set(windows)))):
        raise ValueError("topk_checkpoints requires up to six distinct increasing cold window indices in 1..30")
    if reuse_checkpoint_topk and not windows:
        raise ValueError("reuse_checkpoint_topk requires at least one checkpoint")
    if reverse_cold and not reuse_checkpoint_topk:
        raise ValueError("reverse_cold requires reuse_checkpoint_topk")
    return struct.pack("<4I", initial_hot_rank, sum(1 << window for window in windows),
                       int(bool(reuse_checkpoint_topk)), int(bool(reverse_cold)))


def check_checkpoint_windows(options, rounds):
    _, mask, _, _ = struct.unpack("<4I", options)
    if mask and mask >> max(0, rounds - 1):
        raise ValueError(f"TopK checkpoints must precede the final effective window ({rounds - 1})")


def check_reverse_cold(shape):
    cold = shape.history - shape.resident
    blocks = cold // 512
    windows = shape.rounds - 1
    if (windows < 1 or not shape.splitQ or shape.indexHeads != 32 or shape.resident % 512
            or cold % 512 or blocks < windows
            or ((blocks + windows - 1) // windows) * (2 * windows - 1) >= 2 * blocks):
        raise ValueError("reverse_cold requires sequence-split LI geometry")


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
