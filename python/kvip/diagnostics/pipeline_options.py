"""Independent, opt-in variants on top of the consolidated recent-index PR."""

OPTIONS = ("final_topk_prune", "selected_metadata", "early_fetch_ready", "compact_init")
VARIANTS = {
    "control": (),
    "topk": ("final_topk_prune",),
    "metadata": ("selected_metadata",),
    "ready": ("early_fetch_ready",),
    "init": ("compact_init",),
    "combined": OPTIONS,
}


def options_for(name):
    return {key: key in VARIANTS[name] for key in OPTIONS}


def validate_options(options):
    if not isinstance(options, dict) or set(options) != set(OPTIONS):
        raise ValueError("Pipeline variants must specify exactly the four independent options")
    if any(type(value) is not bool for value in options.values()):
        raise ValueError("Pipeline options must be bool")
    return dict(options)
