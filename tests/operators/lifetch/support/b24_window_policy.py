# SPDX-License-Identifier: Apache-2.0
"""Build the isolated LI23 image for the B24 window count and rank experiment."""
from fractions import Fraction
import hashlib
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

ADVANCE = "window-x4-earned"
MODERATE = "window-x4-224"
WINDOW_CEILINGS = (256, 320, 384, 448, 192)
CREDIT_CEILINGS = {ADVANCE: WINDOW_CEILINGS, MODERATE: (256, 224, 224, 224, 192)}
MODES = tuple(CREDIT_CEILINGS) + tuple(f"window-x{x}-{curve}" for x in (2, 3, 4, 8) for curve in ("linear", "early"))


def policy(mode):
    if mode in CREDIT_CEILINGS:
        return policy("window-x4-early")
    if mode not in MODES:
        raise ValueError(f"Unknown LI window policy {mode}")
    _, windows, curve = mode.split("-")
    x = int(windows[1:])
    knots = (1024, 1568, 1792, 2048) if curve == "early" else (1024, 2048)
    ranks = []
    for epoch in range(x + 1):
        position = Fraction(epoch * (len(knots) - 1), x)
        i = min(int(position), len(knots) - 2)
        rank = knots[i] + (position - i) * (knots[i + 1] - knots[i])
        ranks.append(int(rank + Fraction(1, 2)))
    return tuple(ranks), (256,) + (768 // x,) * x


def replace_once(path, old, new):
    source = path.read_text()
    if source.count(old) != 1:
        raise ValueError(f"Unexpected archived source in {path.name}: {old.splitlines()[0]}")
    path.write_text(source.replace(old, new))


def stage(source, tree, mode):
    """Only the cold/threshold test image is rebuilt. Keep its relative includes intact."""
    ranks, grants = policy(mode)
    source = source.resolve()
    active = source.parents[4]
    op = Path("csrc/operators/lightning_indexer_with_fetch")
    for name in ("include", "op_kernel"):
        shutil.copytree(active / op / name, tree / op / name)
    for sibling in (active / op.parent).iterdir():
        if sibling.name != op.name:
            (tree / op.parent / sibling.name).symlink_to(sibling, target_is_directory=sibling.is_dir())
    experiment = tree / source.relative_to(active)
    shutil.copytree(source, experiment)
    bridge = experiment / "bridge.cpp"
    old = "#elif defined(KVIP_TEST_WINDOW_RANK1)\nconstexpr uint32_t kWindows = 4;"
    replace_once(bridge, old, old.replace("= 4;", f"= {len(grants)};"))
    kernel = tree / op / "op_kernel"
    incremental = kernel / "indexer/incremental_methods.h"
    old = """    if (rankedEpoch_ == 1) return KVIP_TEST_WINDOW_RANK1;
    if (rankedEpoch_ == 2) return KVIP_TEST_WINDOW_RANK2;"""
    replace_once(incremental, old, "\n".join(
        f"    if (rankedEpoch_ == {e}) return {r};" for e, r in enumerate(ranks[1:-1], 1)))
    prefetch = kernel / "prefetch.h"
    limit = f"rankedEpoch_ == 0 ? 64U : {grants[1] // 4}U"
    grant = f"epoch == 0 ? 256U : {grants[1]}U"
    if mode in CREDIT_CEILINGS:
        # Save the grant before the hot/score ranking functions reuse incrementalOld_.
        # Both Qs on this lane must use the same committed row count.
        replace_once(incremental,"uint32_t rankedEpoch_=0;bool windowCoveredEmpty_=false;",
                     "uint32_t rankedEpoch_=0, earnedQueryLimit_=0;bool windowCoveredEmpty_=false;")
        pair_native = kernel / "indexer/large_window_pair.h"
        snapshot = "windowCoveredEmpty_ = incrementalOld_.GetValue(offsetof(rotating::Row, requests) / sizeof(uint32_t)) == 0;"
        replace_once(pair_native, snapshot,
                     "const uint32_t used = incrementalOld_.GetValue(offsetof(rotating::Row, requests) / sizeof(uint32_t));\n"
                     "    windowCoveredEmpty_ = used == 0;\n"
                     "    earnedQueryLimit_ = lifetch_earned_window::RowLimit(epoch, used) / 4U;")
        limit = "rankedEpoch_ == 0 ? 64U : earnedQueryLimit_"
        grant = "lifetch_earned_window::RowLimit(epoch, requests)"
        shutil.copyfile(Path(__file__).with_name("earned_window_quota.h"),kernel / "earned_window_quota.h")
        if mode == MODERATE:
            replace_once(kernel / "earned_window_quota.h",
                         "epoch == 4 ? 192U : 256U + 64U * epoch",
                         "epoch == 4 ? 192U : (epoch == 0 ? 256U : 224U)")
        include = '#include "../../common/lifetch/prefetch_budget.h"'
        replace_once(prefetch, include, include + '\n#include "earned_window_quota.h"')
    replace_once(incremental, "limit=ctx->windowLimit/4, minimum=limit/2",
                 f"limit={limit}, minimum=Min(limit,32U)")
    replace_once(prefetch, "if(available>ctx->windowLimit)available=ctx->windowLimit;",
                 f"const uint32_t grant = {grant};\n"
                 "        if (available > grant) available = grant;")

    # Separate diagnostic image. Rank records use words [0,9) of each 32-word Q.
    # Lane 0 writes Q0/Q1 and lane 1 writes Q2/Q3. Words [16,32) are on their
    # own 128-byte line and hold four groups of four clocks per Q.
    pair = kernel / "indexer/large_window_pair.h"
    old = """        auto pipeline = reinterpret_cast<__gm__ pipeline_ab::Args*>(streamingWorkspace_);
        auto line = reinterpret_cast<__gm__ uint64_t*>(pipeline->times)
                  + runtimeShape_.batch * 64 + (row * 4 + (first ? 3 : 0)) * 16;
        const uint32_t slot = 4 + 4 * (epoch - 1);"""
    replace_once(pair, old, """        auto ranks = reinterpret_cast<__gm__ uint64_t*>(ctx->incremental +
            runtimeShape_.batch * (512 + 4*2048*6 + 1024 + (runtimeShape_.history+255)/256*32));
        auto line = ranks + (row * 4 + first + (epoch - 1) / 4) * 32 + 16;
        const uint32_t slot = 4 * ((epoch - 1) % 4);""")
    digest = hashlib.sha256()
    for header in (incremental, prefetch, pair):
        digest.update(header.read_bytes())
    if mode in CREDIT_CEILINGS:
        digest.update((kernel / "earned_window_quota.h").read_bytes())
    (experiment / "policy.sha256").write_text(f"{mode}\n{digest.hexdigest()}\n")
    return experiment


def install(module, mode):
    ranks, grants = policy(mode)
    module.FOUR_WINDOW_RANKS = ranks
    module.WINDOW_GRANTS["four-ranks"] = grants
    original = module.build

    def build(source, normal, sdk, soc, policy_name, *, trace_sibling=False):
        if policy_name != "four-ranks":
            raise ValueError("The window experiment requires the archived ranked LI source")
        with TemporaryDirectory(prefix="kvip-li-windows-") as folder:
            cold = stage(source, Path(folder), mode)
            launch = original(cold, normal, sdk, soc, policy_name, trace_sibling=trace_sibling)
        if mode in CREDIT_CEILINGS:
            launch.window_limit = max(CREDIT_CEILINGS[mode])
        launch.density_mode = mode
        return launch

    module.build = build


def trace_lanes(records, epochs):
    import numpy as np
    words = np.asarray(records).view(np.uint64)
    if words.ndim != 3 or words.shape[1:] != (4, 32) or epochs not in (2, 3, 4, 8):
        raise ValueError("Unexpected variable-window clock storage")
    return np.stack([words[:, first:first+2, 16:].reshape(len(words), 8, 4)[:, :epochs, :]
                     for first in (0, 2)], axis=1)
