"""Independent URMA scan-order/admission experiment; no graph-route changes."""

OPTIONS = ("reverse_cold", "rolling_threshold", "window_refilter")
DEFAULT_VARIANTS = ("forward", "reverse", "forward-wide", "reverse-wide", "forward-end")
VARIANTS = {name: ("reverse_cold",) if name.startswith("reverse") else () for name in DEFAULT_VARIANTS}
ROLLING_VARIANTS = ("forward", "reverse", "reverse-roll", "reverse-refilter", "forward-refilter", "forward-end")
VARIANTS.update({
    "reverse-roll": ("reverse_cold", "rolling_threshold"),
    "reverse-refilter": OPTIONS,
    "forward-refilter": ("rolling_threshold", "window_refilter"),
})


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
        raise ValueError("Scan experiment requires exactly reverse_cold/rolling_threshold/window_refilter")
    if any(type(value) is not bool for value in options.values()):
        raise ValueError("Scan build options must be bool")
    if options["window_refilter"] and not options["rolling_threshold"]:
        raise ValueError("Window refilter requires rolling threshold")
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
