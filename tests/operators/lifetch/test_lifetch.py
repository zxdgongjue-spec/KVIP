#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unified LIFetch correctness, benchmark and diagnostic entry point."""
import sys
from pathlib import Path

SUPPORT = Path(__file__).resolve().parent / 'support'
sys.path[:0] = [str(SUPPORT), str(Path(__file__).resolve().parents[3] / 'python')]
from workloads import configuration  # noqa: E402


def main(argv=None):
    config = configuration(argv)
    if getattr(config, 'compare_routes', None):
        from route_comparison import run
        return run(config)
    from execution import dispatch

    return dispatch(config)


if __name__ == '__main__':
    raise SystemExit(main())
