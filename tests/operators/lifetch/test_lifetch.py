#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unified LIFetch correctness, benchmark and diagnostic entry point."""
import os
import sys
from pathlib import Path

SUPPORT = Path(__file__).resolve().parent / 'support'
OPERATOR_ROOT = Path(os.environ.get('KVIP_TEST_OPERATOR_ROOT', Path(__file__).resolve().parents[3])).resolve()
sys.path[:0] = [str(SUPPORT), str(OPERATOR_ROOT / 'python')]
from workloads import configuration  # noqa: E402


def main(argv=None):
    config = configuration(argv)
    from execution import dispatch

    return dispatch(config)


if __name__ == '__main__':
    raise SystemExit(main())
