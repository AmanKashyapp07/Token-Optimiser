#!/usr/bin/env bash
# Convenience shell script to execute all unified tests
set -e
cd "$(dirname "$0")/.."
python3 scripts/run_all_tests.py "$@"
