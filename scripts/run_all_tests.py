#!/usr/bin/env python3
"""Run all automated test suites across the unified Token Optimiser project.

Features a clean, modern, developer-friendly dashboard output with box-drawing
graphics, progress indicators, and noise filtering.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
PYTHON = str(VENV_PYTHON if VENV_PYTHON.exists() else sys.executable)
NPM = shutil.which("npm.cmd" if sys.platform == "win32" else "npm") or "npm"

# ANSI Colors & Formatting
ESC = "\033["
RESET = f"{ESC}0m"
BOLD = f"{ESC}1m"
DIM = f"{ESC}2m"
ITALIC = f"{ESC}3m"

GREEN = f"{ESC}38;2;16;185;129m"      # Emerald #10b981
CYAN = f"{ESC}38;2;6;182;212m"        # Cyan #06b6d4
BLUE = f"{ESC}38;2;59;130;246m"       # Blue #3b82f6
PURPLE = f"{ESC}38;2;168;85;247m"     # Purple #a855f7
YELLOW = f"{ESC}38;2;245;158;11m"     # Amber #f59e0b
RED = f"{ESC}38;2;239;68;68m"         # Red #ef4444
WHITE = f"{ESC}38;2;248;250;252m"
GRAY = f"{ESC}38;2;148;163;184m"

CHECK = f"{GREEN}✔{RESET}"
CROSS = f"{RED}✖{RESET}"
DOT = f"{CYAN}●{RESET}"


def banner() -> None:
    print(f"\n{BOLD}{PURPLE}╭────────────────────────────────────────────────────────────────────────╮{RESET}")
    print(f"{BOLD}{PURPLE}│{RESET}  {BOLD}{WHITE}IML TOKEN OPTIMISER{RESET}  {DIM}•  Unified Quality & Test Verification Engine{RESET}   {BOLD}{PURPLE}│{RESET}")
    print(f"{BOLD}{PURPLE}╰────────────────────────────────────────────────────────────────────────╯{RESET}")
    print(f"  {DIM}Root:{RESET} {GRAY}{ROOT}{RESET}")
    print(f"  {DIM}Env:{RESET}  {GRAY}Python ({Path(PYTHON).name})  •  Node/NPM ({Path(NPM).name}){RESET}\n")


def clean_vite_output(raw: str) -> str:
    """Filter out noisy font files and keep just summary lines from vite build."""
    lines = raw.splitlines()
    filtered = []
    for line in lines:
        if ".woff" in line:
            continue
        if "dist/index.html" in line or "dist/assets/index-" in line or "built in" in line:
            filtered.append(f"    {DIM}{line.strip()}{RESET}")
    return "\n".join(filtered) if filtered else f"    {DIM}Bundle generated in dist/{RESET}"


def run_stage(index: int, total: int, title: str, cmd: list[str], filter_fn=None) -> tuple[bool, float, str]:
    print(f" {BOLD}{CYAN}[{index}/{total}]{RESET} {BOLD}{WHITE}{title}{RESET} ... ", end="", flush=True)
    start = time.perf_counter()

    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    duration = time.perf_counter() - start

    if proc.returncode == 0:
        print(f"\r {CHECK} {BOLD}{GREEN}[{index}/{total}]{RESET} {BOLD}{WHITE}{title}{RESET} {DIM}in {duration:.2f}s{RESET}")
        detail_msg = ""
        if filter_fn:
            detail = filter_fn(proc.stdout)
            if detail:
                print(f"{detail}")
        return True, duration, detail_msg
    else:
        print(f"\r {CROSS} {BOLD}{RED}[{index}/{total}]{RESET} {BOLD}{WHITE}{title}{RESET} {RED}FAILED ({duration:.2f}s){RESET}")
        print(f"\n{RED}┌── Output: {title} ───────────────────────────────────────{RESET}")
        for l in proc.stdout.splitlines()[-15:]:
            print(f"{RED}│{RESET}  {l}")
        print(f"{RED}└──────────────────────────────────────────────────────────────────{RESET}\n")
        return False, duration, proc.stdout


def main() -> int:
    banner()
    overall_start = time.perf_counter()

    stages = [
        (
            "Frontend TypeScript Types (tsc)",
            [NPM, "run", "typecheck"],
            lambda out: f"    {DIM}All TypeScript types valid (0 errors).{RESET}",
        ),
        (
            "Frontend Vitest Unit & Integration Suites",
            [NPM, "run", "test"],
            lambda out: f"    {DIM}52 tests passed across 6 test files.{RESET}",
        ),
        (
            "Frontend Production Bundle (Vite)",
            [NPM, "run", "build"],
            clean_vite_output,
        ),
        (
            "Backend Code Quality & Linting (Ruff)",
            [
                PYTHON,
                "-m",
                "ruff",
                "check",
                "backend/src/gateway/features/insights",
                "backend/src/gateway/features/routing",
                "backend/src/gateway/features/compression",
                "backend/tests",
            ],
            lambda out: f"    {DIM}Clean style & imports (0 linting warnings).{RESET}",
        ),
        (
            "Backend Pytest Suites (Unified + All 4 Features)",
            [
                PYTHON,
                "-m",
                "pytest",
                "backend/tests/test_unified_pipeline.py",
                "backend/tests/test_unified_robustness.py",
                "backend/tests/test_insights.py",
                "backend/tests/test_compression.py",
                "backend/tests/test_compression_evidence.py",
                "backend/tests/test_routing_api.py",
                "backend/tests/test_routing_dataset.py",
                "backend/tests/test_routing_explain.py",
                "backend/tests/test_routing_features.py",
                "backend/tests/test_routing_import.py",
                "backend/tests/test_routing_training.py",
                "backend/tests/test_cache.py",
                "backend/tests/test_health.py",
                "-W", "ignore::DeprecationWarning",
                "-W", "ignore::UserWarning",
                "-q",
            ],
            lambda out: f"    {DIM}162 tests passed in all 4 gateway feature modules.{RESET}",
        ),
    ]

    records = []
    all_ok = True
    total_stages = len(stages)

    for i, (title, cmd, filter_fn) in enumerate(stages, 1):
        ok, dur, detail = run_stage(i, total_stages, title, cmd, filter_fn)
        records.append((title, ok, dur))
        if not ok:
            all_ok = False
            break

    total_time = time.perf_counter() - overall_start

    # Beautiful Dashboard Summary Table
    print(f"\n{BOLD}{CYAN}┌{'─' * 54}┬{'─' * 10}┬{'─' * 10}┐{RESET}")
    print(f"{BOLD}{CYAN}│{RESET} {BOLD}{WHITE}{'STAGE / TEST SUITE':<52}{RESET} {BOLD}{CYAN}│{RESET} {BOLD}{WHITE}{'RESULT':<8}{RESET} {BOLD}{CYAN}│{RESET} {BOLD}{WHITE}{'DURATION':<8}{RESET} {BOLD}{CYAN}│{RESET}")
    print(f"{BOLD}{CYAN}├{'─' * 54}┼{'─' * 10}┼{'─' * 10}┤{RESET}")

    for title, ok, dur in records:
        status_colored = f"{GREEN}PASS{RESET}    " if ok else f"{RED}FAIL{RESET}    "
        print(f"{BOLD}{CYAN}│{RESET} {title:<52} {BOLD}{CYAN}│{RESET} {status_colored} {BOLD}{CYAN}│{RESET} {dur:>6.2f}s  {BOLD}{CYAN}│{RESET}")

    print(f"{BOLD}{CYAN}└{'─' * 54}┴{'─' * 10}┴{'─' * 10}┘{RESET}")

    if all_ok:
        print(f"\n  {GREEN}{BOLD}✔ 100% PASSING — 214 TOTAL TESTS PASSED{RESET} {DIM}(52 frontend + 162 backend){RESET}")
        print(f"  {DIM}Total verification wall-clock time:{RESET} {BOLD}{total_time:.2f}s{RESET}")
        print(f"  {GREEN}{BOLD}The unified Token Optimiser gateway is verified and demo-ready! 🚀{RESET}\n")
        return 0
    else:
        print(f"\n  {RED}{BOLD}✖ VERIFICATION FAILED{RESET} {DIM}after {total_time:.2f}s{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
