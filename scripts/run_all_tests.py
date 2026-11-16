#!/usr/bin/env python3
"""Shim forwarding to tests/run_all_tests.py."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

target = Path(__file__).resolve().parent.parent / "tests" / "run_all_tests.py"
runpy.run_path(str(target), run_name="__main__")
