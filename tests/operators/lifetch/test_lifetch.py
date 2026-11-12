#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unified LIFetch correctness, benchmark and diagnostic entry point."""
import sys
from pathlib import Path

SUPPORT = Path(__file__).resolve().parent / 'support'
sys.path.insert(0, str(SUPPORT))
from workloads import configuration  # noqa: E402
from execution import dispatch  # noqa: E402


def main(argv=None):
    return dispatch(configuration(argv))


if __name__ == '__main__':
    raise SystemExit(main())
