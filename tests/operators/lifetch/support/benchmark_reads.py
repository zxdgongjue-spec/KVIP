# SPDX-License-Identifier: Apache-2.0
"""Fixed Fetch work, independent of which stage fetches a cold token."""

READ_FIELDS = ("prefetch_hot", "prefetch_cold", "unused_hot", "unused_cold", "final", "total", "unused")


def fixed_read_workload(reads):
    """Validate counters and omit only the prefetch-cold/final partition.

    A window may fetch a requested cold token, or leave that same token to the
    final stage. Both executions do equal work.
    Preserve hot prefetches, unused reads and the total transfer count so this
    allowance cannot hide changed speculative traffic or a warmed-cache run.
    """
    if not isinstance(reads, dict):
        raise ValueError("Fetch counters must be a dictionary")
    for name in READ_FIELDS:
        if type(reads.get(name)) is not int or reads[name] < 0:
            raise ValueError(f"Fetch counter {name} must be a nonnegative integer")
    if reads["total"] != reads["prefetch_hot"] + reads["prefetch_cold"] + reads["final"]:
        raise ValueError("Fetch total is inconsistent")
    if reads["unused"] != reads["unused_hot"] + reads["unused_cold"]:
        raise ValueError("Unused Fetch total is inconsistent")
    if reads["unused_hot"] > reads["prefetch_hot"] or reads["unused_cold"] > reads["prefetch_cold"]:
        raise ValueError("Unused reads exceed prefetches")
    return dict(
        prefetch_hot=reads["prefetch_hot"],
        cold_or_final=reads["prefetch_cold"] + reads["final"],
        unused_hot=reads["unused_hot"],
        unused_cold=reads["unused_cold"],
        total=reads["total"],
        unused=reads["unused"],
    )
