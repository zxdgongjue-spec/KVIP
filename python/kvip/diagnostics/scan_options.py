"""Independent URMA scan-order/admission experiment; no graph-route changes."""

OPTIONS = ("reverse_cold", "rolling_threshold", "window_refilter", "rolling_batched", "rolling_windowed", "rolling_compact", "rolling_first_window", "refilter_safe", "refilter_refill", "compact_out_of_place", "rolling_reuse_seed", "rolling_fast_path")
DEFAULT_VARIANTS = ("forward", "reverse", "forward-wide", "reverse-wide", "forward-end")
VARIANTS = {name: ("reverse_cold",) if name.startswith("reverse") else () for name in DEFAULT_VARIANTS}
ROLLING_VARIANTS = ("forward", "reverse", "reverse-roll", "reverse-refilter", "forward-refilter", "forward-end")
VARIANTS.update({
    "reverse-roll": ("reverse_cold", "rolling_threshold"),
    "reverse-refilter": ("reverse_cold", "rolling_threshold", "window_refilter"),
    "forward-refilter": ("rolling_threshold", "window_refilter"),
})

BATCHED_VARIANTS = ("forward", "reverse", "reverse-refilter", "reverse-batch",
                    "reverse-batch-ref", "forward-batch-ref", "forward-end")
VARIANTS.update({
    "reverse-batch": ("reverse_cold", "rolling_threshold", "rolling_batched"),
    "reverse-batch-ref": ("reverse_cold", "rolling_threshold", "window_refilter", "rolling_batched"),
    "forward-batch-ref": ("rolling_threshold", "window_refilter", "rolling_batched"),
})

WINDOWED_VARIANTS = ("forward", "reverse-refilter", "reverse-batch-ref", "reverse-window",
                    "reverse-window-ref", "forward-window-ref", "forward-end")
VARIANTS.update({
    "reverse-window": ("reverse_cold", "rolling_threshold", "rolling_windowed"),
    "reverse-window-ref": ("reverse_cold", "rolling_threshold", "window_refilter", "rolling_windowed"),
    "forward-window-ref": ("rolling_threshold", "window_refilter", "rolling_windowed"),
})


LEAN_VARIANTS = ("forward", "reverse", "reverse-window-ref", "reverse-compact-ref",
                 "reverse-first-ref", "forward-end")
VARIANTS.update({
    "reverse-compact-ref": ("reverse_cold", "rolling_threshold", "window_refilter", "rolling_windowed", "rolling_compact"),
    "reverse-first-ref": ("reverse_cold", "rolling_threshold", "window_refilter", "rolling_windowed", "rolling_compact", "rolling_first_window"),
})


SAFE_VARIANTS = ("forward", "reverse", "reverse-window-ref", "reverse-safe-ref",
                 "reverse-refill-ref", "forward-end")
VARIANTS.update({
    "reverse-safe-ref": ("reverse_cold", "rolling_threshold", "window_refilter", "rolling_windowed", "refilter_safe"),
    "reverse-refill-ref": ("reverse_cold", "rolling_threshold", "window_refilter", "rolling_windowed", "refilter_safe", "refilter_refill"),
})


ECONOMY_VARIANTS = ("forward", "reverse", "reverse-out", "reverse-compact-safe",
                    "reverse-first", "reverse-first-safe", "reverse-end", "forward-end")
VARIANTS.update({
    "reverse-out": ("reverse_cold", "compact_out_of_place"),
    "reverse-compact-safe": ("reverse_cold", "compact_out_of_place", "rolling_threshold", "rolling_windowed", "rolling_compact", "window_refilter", "refilter_safe"),
    "reverse-first": ("reverse_cold", "compact_out_of_place", "rolling_threshold", "rolling_windowed", "rolling_compact", "rolling_first_window"),
    "reverse-first-safe": ("reverse_cold", "compact_out_of_place", "rolling_threshold", "rolling_windowed", "rolling_compact", "rolling_first_window", "window_refilter", "refilter_safe"),
    "reverse-end": ("reverse_cold",),
})


SEED_VARIANTS = ("forward", "reverse", "reverse-first-in", "reverse-first-reuse",
                 "reverse-window-in", "reverse-window-reuse", "reverse-end", "forward-end")
for _name in SEED_VARIANTS[2:6]:
    _flags = ("reverse_cold", "rolling_threshold", "rolling_windowed", "rolling_compact",
              "window_refilter", "refilter_safe")
    if "first" in _name:
        _flags += ("rolling_first_window",)
    if _name.endswith("reuse"):
        _flags += ("rolling_reuse_seed",)
    VARIANTS[_name] = _flags

FAST_VARIANTS = ("forward", "reverse", "reverse-window-in", "reverse-window-reuse",
                 "reverse-window-fast", "reverse-end", "forward-end")
VARIANTS["reverse-window-fast"] = VARIANTS["reverse-window-reuse"] + ("rolling_fast_path",)


def options_for(name):
    return {key: key in VARIANTS[name] for key in OPTIONS}


def budgets_for(name):
    if name not in VARIANTS:
        raise ValueError(f"Unknown scan variant: {name}")
    return (1024, 128) if name.endswith("-wide") else (512, 64)


def q_workers_for(name):
    if name not in VARIANTS:
        raise ValueError(f"Unknown scan variant: {name}")
    return 2


def validate_options(options):
    if not isinstance(options, dict) or set(options) != set(OPTIONS):
        raise ValueError("Scan experiment options do not match the selected native interface")
    if any(type(value) is not bool for value in options.values()):
        raise ValueError("Scan build options must be bool")
    if (options["window_refilter"] or options["rolling_batched"] or options["rolling_windowed"]) and not options["rolling_threshold"]:
        raise ValueError("Window refilter requires rolling threshold")
    if options["rolling_batched"] and options["rolling_windowed"]:
        raise ValueError("Select either batch or window accumulation")
    if (options["rolling_compact"] or options["rolling_first_window"]) and not options["rolling_windowed"]:
        raise ValueError("Compact/first-window options require window accumulation")
    if options["rolling_first_window"] and not options["rolling_compact"]:
        raise ValueError("First-window correction requires compact admission")
    if options["rolling_reuse_seed"] and not options["rolling_compact"]:
        raise ValueError("Seed reuse requires compact window accumulation")
    if options["rolling_fast_path"] and not options["rolling_reuse_seed"]:
        raise ValueError("Packed byte accumulation requires seed histogram reuse")
    if options["refilter_safe"] and not options["window_refilter"]:
        raise ValueError("Visibility-safe admission requires window refilter")
    if options["refilter_refill"] and not options["refilter_safe"]:
        raise ValueError("Same-window refill requires visibility-safe admission")
    return dict(options)


def prebuild(args, retained):
    """Use the production builder before allocating/restoring any NPU arena."""
    from pathlib import Path

    from kvip.runtime.build import build_libraries, kvaf_build_paths
    from kvip.diagnostics.console import emit

    cann, headers = kvaf_build_paths(args.build_dir)
    seen = set()
    for name in args.scan_variants:
        options = options_for(name)
        key = tuple(options.values())
        if key in seen:
            continue
        retained.setdefault("progress", {}).update(variant=name, phase="compile_preflight", step=None)
        build_libraries(cache_dir=Path(args.build_dir) / "lifetch", cann=cann, headers=headers,
                        soc=args.soc, **options)
        seen.add(key)
    emit(f"BUILD_VARIANTS_READY unique={len(seen)}")
