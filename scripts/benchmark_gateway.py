#!/usr/bin/env python3
"""CLI Gateway Load & Performance Benchmarking Tool.

Measures throughput (RPS), p50/p90/p95/p99 latency distributions,
cache hit ratios, token economics, and format quality adherence.
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend" / "src"))

from gateway.core.benchmarking import run_gateway_benchmark


def main():
    print("\n\033[1;35m╭────────────────────────────────────────────────────────────────────────╮\033[0m")
    print("\033[1;35m│\033[0m  \033[1;37mTOKEN OPTIMISER GATEWAY BENCHMARK\033[0m  \033[2m•  Throughput & Latency Engine\033[0m   \033[1;35m│\033[0m")
    print("\033[1;35m╰────────────────────────────────────────────────────────────────────────╯\033[0m")

    num_reqs = 30
    concurrency = 5
    print(f"  \033[2mBenchmarking {num_reqs} requests with concurrency={concurrency} ...\033[0m\n")

    res = asyncio.run(run_gateway_benchmark(num_requests=num_reqs, concurrency=concurrency, use_cache=True))

    lat = res.latency
    eco = res.economics
    qual = res.quality

    print("\033[1;36m┌───────────────────────────────────────────────┬────────────────────────┐\033[0m")
    print("\033[1;36m│\033[0m \033[1;37mGATEWAY BENCHMARK METRIC\033[0m                      \033[1;36m│\033[0m \033[1;37mMEASURED VALUE\033[0m         \033[1;36m│\033[0m")
    print("\033[1;36m├───────────────────────────────────────────────┼────────────────────────┤\033[0m")
    print(f"\033[1;36m│\033[0m Total Executed Requests                        \033[1;36m│\033[0m {res.total_requests:<22} \033[1;36m│\033[0m")
    print(f"\033[1;36m│\033[0m Concurrency Level                              \033[1;36m│\033[0m {res.concurrency:<22} \033[1;36m│\033[0m")
    print(f"\033[1;36m│\033[0m Throughput                                     \033[1;36m│\033[0m \033[1;32m{res.throughput_rps:.1f} req/s\033[0m          \033[1;36m│\033[0m")
    print(f"\033[1;36m│\033[0m Wall-Clock Duration                            \033[1;36m│\033[0m {res.wall_clock_time_s:.2f} s                 \033[1;36m│\033[0m")
    print("\033[1;36m├───────────────────────────────────────────────┼────────────────────────┤\033[0m")
    print(f"\033[1;36m│\033[0m p50 (Median) Latency                           \033[1;36m│\033[0m \033[1;34m{lat.p50_ms:.2f} ms\033[0m             \033[1;36m│\033[0m")
    print(f"\033[1;36m│\033[0m p90 Latency                                    \033[1;36m│\033[0m {lat.p90_ms:.2f} ms             \033[1;36m│\033[0m")
    print(f"\033[1;36m│\033[0m p95 Latency                                    \033[1;36m│\033[0m {lat.p95_ms:.2f} ms             \033[1;36m│\033[0m")
    print(f"\033[1;36m│\033[0m p99 Latency                                    \033[1;36m│\033[0m {lat.p99_ms:.2f} ms             \033[1;36m│\033[0m")
    print(f"\033[1;36m│\033[0m Min / Max Latency                              \033[1;36m│\033[0m {lat.min_ms:.1f} / {lat.max_ms:.1f} ms        \033[1;36m│\033[0m")
    print("\033[1;36m├───────────────────────────────────────────────┼────────────────────────┤\033[0m")
    print(f"\033[1;36m│\033[0m Cache Hit Ratio                                \033[1;36m│\033[0m \033[1;33m{res.cache_hit_ratio * 100:.1f}%\033[0m ({res.cache_hits}/{res.total_requests})        \033[1;36m│\033[0m")
    print(f"\033[1;36m│\033[0m Format Adherence Rate                          \033[1;36m│\033[0m {qual.format_adherence_pct:.1f}%                 \033[1;36m│\033[0m")
    print(f"\033[1;36m│\033[0m Total Completion Cost                          \033[1;36m│\033[0m ${eco['optimized_cost_usd']:.6f}           \033[1;36m│\033[0m")
    print(f"\033[1;36m│\033[0m Estimated Financial Savings                    \033[1;36m│\033[0m \033[1;32m{eco['cost_reduction_pct']:.1f}%\033[0m (${eco['total_saved_usd']:.6f})   \033[1;36m│\033[0m")
    print("\033[1;36m└───────────────────────────────────────────────┴────────────────────────┘\033[0m\n")


if __name__ == "__main__":
    main()
