#!/usr/bin/env python3
"""Compatibility entry for the shared bench/diagnose operand generator."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'support'))
from compact_generator import generate, main  # noqa: E402,F401

if __name__ == '__main__':
    main()
