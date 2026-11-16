"""Independent URMA scan-order/admission experiment; no graph-route changes."""

OPTIONS = ("reverse_cold",)
DEFAULT_VARIANTS = ("forward", "reverse", "forward-wide", "reverse-wide", "forward-end")
VARIANTS = {name: ("reverse_cold",) if name.startswith("reverse") else () for name in DEFAULT_VARIANTS}


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
        raise ValueError("Scan experiment requires exactly reverse_cold")
    if any(type(value) is not bool for value in options.values()):
        raise ValueError("Scan build options must be bool")
    return dict(options)
