# Testing & Benchmarking Directory

This directory provides a consolidated entrypoint for all automated tests, quality evaluations, and performance benchmarks across the Token Optimiser gateway.

---

## Directory Organization

```
tests/
├── README.md                      # This guide
├── run_all_tests.py               # Master unified verification engine (258+ tests)
├── run_tests.sh                   # Fast bash test runner
│
├── backend/                       # Backend Pytest test suites (symlink to backend/tests)
│   ├── test_advanced_gateway_capabilities.py
│   ├── test_budgeting_v2.py       # ML adaptive token budgeting tests
│   ├── test_cache.py              # Semantic cache core tests
│   ├── test_cache_v2.py           # Context-aware compound key & FPR tests
│   ├── test_chat_completions_and_observability.py
│   ├── test_compression.py        # Lexical compression tests
│   ├── test_eval_v2.py            # Quality evaluation harness tests
│   ├── test_health.py             # Health check endpoints
│   ├── test_insights.py           # Observability & telemetry tests
│   ├── test_production_resilience.py
│   ├── test_resilience_v2.py      # Singleflight, 503 isolation, and distributed coalescing tests
│   ├── test_routing_api.py        # Model router API tests
│   ├── test_unified_pipeline.py   # Multi-stage gateway pipeline tests
│   └── test_unified_robustness.py
│
├── frontend/                      # Frontend Vitest test suites (located in src/)
│   ├── src/app/smoke.test.ts
│   ├── src/app/unified_integration.test.ts
│   ├── src/features/insights/insights.test.ts
│   ├── src/features/compression/client.test.ts
│   ├── src/features/routing/helpers.test.ts
│   └── src/features/routing/client.test.ts
│
└── benchmarks/                    # Dedicated benchmark & evaluation runners
    ├── run_eval_v2.py             # 250-prompt multi-task quality & cost evaluation harness
    ├── evaluate_cache_fpr.py      # Semantic cache false positive rate & precision curve
    ├── benchmark_adaptive_budgeting.py # Ridge regression adaptive budgeting vs static capping
    ├── benchmark_indic_tokenizers.py   # Empirical multi-tokenizer Indic Token Tax benchmark
    ├── benchmark_gateway.py       # Gateway CPU overhead latency & throughput benchmark
    ├── evaluate_quality.py        # 120-prompt legacy quality comparison
    └── build_eval_dataset.py      # Eval dataset generator
```

---

## Quick Commands

### 1. Run Complete Verification (Frontend + Backend + Types + Linting)
```bash
python tests/run_all_tests.py
# Or via npm shortcut:
npm run test:all
```

### 2. Run Backend Tests Only
```bash
pytest
# Or targeting specific modules:
pytest tests/backend/test_resilience_v2.py -v
```

### 3. Run Frontend Tests Only
```bash
npm test
```

### 4. Run Quality & Performance Benchmarks
```bash
# 250-prompt quality evaluation benchmark
python tests/benchmarks/run_eval_v2.py

# Semantic cache False Positive Rate (FPR) curve
python tests/benchmarks/evaluate_cache_fpr.py

# ML Adaptive Budgeting comparison
python tests/benchmarks/benchmark_adaptive_budgeting.py

# Indic Tokenizer inflation benchmark
python tests/benchmarks/benchmark_indic_tokenizers.py

# Gateway throughput & latency benchmark
python tests/benchmarks/benchmark_gateway.py
```
