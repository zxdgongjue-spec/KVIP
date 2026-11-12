"""P0-A2.2a same-captured-Graph actual-K transitions; no persistent Index/KV across K.

The Index/KV fixture is restored to baseline before each stage, while the
same NPU Graph and the entire V9 Mapping workspace are reused.
"""


def transition_schedule(initial_k, history, resident):
    initial = tuple(int(k) for k in initial_k)
    if not initial or min(initial) < resident or max(initial) > history:
        raise ValueError("dynamic_actual_k_invalid_geometry")
    if any(k % 128 for k in initial) or history % 128:
        raise ValueError("dynamic_actual_k_requires_128_alignment")
    if min(initial) == history:
        raise ValueError("dynamic_actual_k_requires_short_request")
    full = (history,) * len(initial)
    short = (min(initial),) * len(initial)
    return [
        ("long_before_shrink", full),
        ("short_after_long", short),
        ("long_after_shrink", full),
        ("restore_mixed_K", initial),
    ]
