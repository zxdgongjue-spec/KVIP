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
