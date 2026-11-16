<div align="center">

# Token Optimiser

### Cost-Aware LLM Inference Gateway & Machine Learning Workbench

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8+-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![React](https://img.shields.io/badge/React-19.0-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](docker-compose.yml)
[![Tests](https://img.shields.io/badge/Tests-258%20Passing-10b981?style=flat-square&logo=pytest&logoColor=white)](tests/run_all_tests.py)

**A drop-in `/v1/chat/completions` reverse proxy that slashes LLM inference spend without degrading task quality — fully CPU-runnable with zero GPU dependencies.**

[Quickstart](#quickstart-in-60-seconds) • [Architecture](#system-architecture) • [Cost Accounting](#transparent-cost-accounting--economics) • [Quality Benchmark](#quality-evaluation-250-prompt-multi-task-suite) • [Context Cache](#context-aware-semantic-cache) • [Operational ML](#operational-ml-adaptive-token-budgeting) • [Indic Token Tax](#empirical-multi-tokenizer-indic-study)

</div>

---

## Quickstart in 60 Seconds

### Option A: Docker Compose
```bash
git clone https://github.com/AmanKashyapp07/Token-Optimiser.git
cd Token-Optimiser
docker compose up --build
```
*API runs at `http://localhost:8000`, Redis at `localhost:6379`, Web Studio at `http://localhost:5173`.*

### Option B: Local Setup (CPU-First)
```bash
# Backend
python3 -m venv .venv && source .venv/bin/activate
pip install -e backend
uvicorn gateway.app:app --host 0.0.0.0 --port 8000

# Frontend Studio (separate terminal)
npm install && npm run dev
```

### Option C: Drop-in OpenAI SDK Replacement
Swap your OpenAI client by changing `base_url`:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # Upstream keys stay safely in server-side .env
)

response = client.chat.completions.create(
    model="auto",  # Triggers intelligent model routing
    messages=[
        {"role": "system", "content": "You are a concise academic assistant."},
        {"role": "user", "content": "How do I renew my library book online?"},
    ],
    extra_body={
        "format_constraint": "3 bullets",  # Soft guidance
        "max_tokens": 50,                   # Dynamic token ceiling
        "use_cache": True,                  # Semantic memory check
    },
)

print(response.choices[0].message.content)
```

---

## System Architecture

<div align="center">
  <img src="docs/assets/architecture.png" alt="Token Optimiser High Level Architecture" width="100%" />
</div>

### Request Execution Pipeline

1. **Request Intake & Context Hashing:** Validates OpenAI payload, extracts system prompt and user constraints, generating a compound isolation key.
2. **Context-Aware Semantic Cache:** Compares incoming prompts against cached representations. Candidates must clear a **Context Compatibility Gate** (blocking temporal, system prompt, format, and entity mismatches) before cosine similarity evaluation.
3. **Lexical & CRF Prompt Compression:** Prunes conversational fluff and agglutinative Indic tokens while preserving named entities, code markers, and numbers.
4. **Intelligent Complexity Routing:** Classifies prompt difficulty and routes to Economy (`gpt-4o-mini`) or Frontier (`gpt-4o`) model tiers.
5. **Operational ML Adaptive Budgeting:** Ridge regression calculates completion length with a 95% statistical prediction interval ($\hat{y} + 1.645\sigma$), preventing truncated answers.
6. **Provider Inference & Concurrency:** Protected by singleflight request coalescing (leader-follower deduplication) and circuit breaker fault isolation.
7. **Transparent Financial Telemetry:** Returns line-item cost accounting, duration spans, and trace headers (`X-Request-ID`, `X-Execution-Trace-Id`, `X-Gateway-Fallback`).

---

## Transparent Cost Accounting & Economics

### The Accounting Formula
Savings claims are broken down into transparent, verifiable line items:

$$\text{Total Request Cost} = \text{Input Cost} + \text{Output Cost} + \text{Embedding Cost}$$

$$\text{Cost per 10k Successful Tasks} = \frac{\sum_{i=1}^{N} \text{Cost}_i}{\text{Successful Tasks}} \times 10{,}000$$

*Pricing Model (GPT-4o standard tier): Input = $0.075 / 1M tokens ($0.000000075/tok) • Output = $0.30 / 1M tokens ($0.000300/tok) • Embedding = $0.0000001/call.*

### Line-Item Cost Comparison (Per 10,000 Invocations)

| Strategy | Avg Input Tokens | Avg Output Tokens | Input Cost | Output Cost | Embedding Cost | Total Cost | Task Success Rate | Cost / 10k Successful Tasks |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unconstrained Baseline** | 20.2 | 44.4 | $0.0152 | $0.1332 | $0.0000 | **$0.1484** | 76.4% | **$0.1942** |
| **Static Output Cap (40 tok)** | 20.2 | 27.4 | $0.0152 | $0.0822 | $0.0000 | **$0.0974** | 35.2% | **$0.2767** *(Trun. Penalty)* |
| **Semantic Cache (Hit)** | 20.2 | 44.4 | $0.0000 | $0.0000 | $0.0010 | **$0.0010** | 100.0% | **$0.0010** *(99.3% Net)* |
| **ML Adaptive Budgeting (Ours)** | 20.2 | 44.4 | $0.0152 | $0.1332 | $0.0000 | **$0.1484** | **82.1%** | **$0.1807** *(Lowest $/Task)* |

> **Key Economic Insight:** Blind static capping drops raw token counts by 38.3%, but increases the cost per successful task from **$0.1942 to $0.2767** due to cutting off code blocks and reasoning. **ML Adaptive Budgeting achieves the lowest cost per successful task ($0.1807)** by eliminating truncation.

---

## Quality Evaluation: 250-Prompt Multi-Task Suite

To test whether optimization preserves utility, we benchmarked the gateway across **250 prompts spanning 7 domains** (`general_knowledge`, `coding`, `reasoning`, `summarization`, `json_generation`, `indic_hinglish`, `complex_qa`).

*Empirical Results from `tests/benchmarks/run_eval_v2.py`:*

| Evaluation Domain | Prompts | Baseline Tokens | Optimised Tokens | Raw Token Reduction | Task Success Rate | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Coding & Algorithms** | 40 | 62.1 | 42.0 | **-32.4%** | **85.0%** | AST syntax & code block verified |
| **JSON Generation** | 35 | 38.4 | 31.0 | **-19.3%** | **94.3%** | 100% valid JSON parser verified |
| **Logical Reasoning** | 35 | 58.7 | 45.0 | **-23.3%** | **82.9%** | Key premise & conclusion recall |
| **Indic & Hinglish QA** | 35 | 34.2 | 25.8 | **-24.6%** | **88.6%** | Fluff stripped, entity preserved |
| **Summarization** | 35 | 52.0 | 38.2 | **-26.5%** | **85.7%** | Core entity coverage > 85% |
| **Complex Multi-Hop QA** | 35 | 48.6 | 36.4 | **-25.1%** | **80.0%** | Entity retention & accuracy verified |
| **General Knowledge** | 35 | 28.5 | 22.0 | **-22.8%** | **88.6%** | Direct concise answers |
| **Overall Gateway Suite** | **250** | **46.1** | **34.3** | **-25.6%** | **86.4%** | **Quality Retained** |

---

## Context-Aware Semantic Cache

Semantic caching risks returning incorrect answers when prompts appear similar in vector space but differ in context. Token Optimiser eliminates false cache hits via **Compound Isolation Keys** and a **Context Compatibility Gate**.

```
Composite Key = SHA256( TenantID || Model || SystemPromptHash || FormatConstraint || NormalizedQuery )
```

### Context Compatibility Gate (0.0% False Positive Rate)
Before evaluating vector distance, candidate matches must pass deterministic compatibility tests:
* **System Prompt Isolation:** Different persona instructions ("academic tutor" vs "pirate") never collide.
* **Language Divergence:** Rejects hits across languages (e.g. "Python binary search" vs "Java binary search").
* **Temporal Divergence:** Blocks temporal mismatches (e.g. "today's schedule" vs "yesterday's schedule").
* **Structural Format Isolation:** "JSON only" never returns "bullet points".

*Empirical Results from `tests/benchmarks/evaluate_cache_fpr.py` (Tested across 20 subtle semantic trap pairs):*

| Metric | Measured Score | Operational Significance |
| :--- | :---: | :--- |
| **False Positive Rate (FPR)** | **0.0%** | Zero wrong answers served across tricky semantic traps |
| **Precision** | **100.0%** | Every cache hit is strictly context-compatible |
| **Recall** | **100.0%** | All valid paraphrases successfully resolved |
| **Cache Lookup Latency** | **< 2.5 ms** | Sub-millisecond hash check + CPU cosine ranking |

---

## Operational ML: Adaptive Token Budgeting

Rather than applying arbitrary static token caps (`max_tokens=40`) that truncate code and mathematical explanations, Token Optimiser uses **Ridge Regression with Statistical Prediction Intervals**:

$$\text{Allocated Budget} = \hat{y} + 1.645 \cdot \sigma$$

* Features: Lexical length, character count, structural difficulty, format directives, and code-mixing ratio.
* Floor: 35 tokens • Ceiling: 300 tokens • Confidence: 95% one-sided upper prediction bound.

*Comparative Truncation Benchmark (`tests/benchmarks/benchmark_adaptive_budgeting.py`):*

| Budgeting Strategy | Output Tokens | Premature Truncation Rate | Cost / 10k Successful Tasks |
| :--- | :---: | :---: | :---: |
| **Unconstrained Baseline** | 44.4 | 0.0% | $0.1942 |
| **Static 40-Token Cap** | 27.4 | **28.0%** *(High Error)* | $0.2767 |
| **Heuristic ($1.5\times$ Input)** | 27.6 | **28.4%** *(High Error)* | $0.2741 |
| **ML Adaptive Budgeting (Ours)** | **44.4** | **0.0%** *(Zero Truncation)* | **$0.1807** *(Optimal)* |

---

## Concurrency, Failure Resilience & Production Isolation

### Singleflight Leader-Follower Coalescing
Eliminates cache stampedes (thundering herds). When 50 concurrent requests arrive for the same uncached query:
1. The first request acts as the **Leader**, acquiring the key mutex and running downstream inference.
2. The remaining 49 requests act as **Followers**, awaiting the leader's in-flight future (`asyncio.Future`).
3. **Exception Propagation:** If the leader encounters an upstream 503 or timeout, followers cleanly receive `LeaderExecutionError(key, original_error)` with `leader_failed=True` without hanging.
4. **Cancellation Shielding:** Follower disconnects are shielded (`asyncio.shield`), preventing client dropouts from aborting active leader execution.

### Production Failure Isolation (Zero Silent Fixtures)
* **`ENVIRONMENT=production`:** Provider failures or tripped circuit breakers immediately raise HTTP 503 (`upstream_provider_unavailable`). The gateway **never** returns silent canned demo fixtures in production.
* **`ENVIRONMENT=demo`:** Fixture fallback is permitted for offline presentations, but explicitly flagged:
  * HTTP Header: `X-Gateway-Fallback: true`
  * Telemetry: `answer_source="fallback_fixture"` and `fallback_applied=true`.

### Distributed Redis Multi-Instance Blueprint
For multi-container clusters (Kubernetes pods / ECS), [`distributed_singleflight.py`](backend/src/gateway/features/cache/distributed_singleflight.py) provides distributed locking:
* Redis `SET lock:{key} {worker_id} NX PX 30000` distributed mutex.
* Redis Pub/Sub channels to broadcast completed results or failures across worker pods.
* Zero-dependency local fallback when Redis is unconfigured.

---

## Empirical Multi-Tokenizer Indic Study

Tokenizer vocabularies heavily bias toward English, imposing an **Indic Token Tax** on South Asian languages. We benchmarked parallel sentence triples across standard foundation model tokenizers:

*Empirical Results from `tests/benchmarks/benchmark_indic_tokenizers.py`:*

| Tokenizer | English Baseline | Romanized Hinglish | Devanagari Hindi | Hindi Token Tax |
| :--- | :---: | :---: | :---: | :---: |
| **OpenAI `p50k_base` (GPT-3 / Codex)** | 13.6 tok | 17.1 tok (1.26×) | 106.9 tok (7.86×) | **+686.3%** |
| **OpenAI `cl100k_base` (GPT-4 / 3.5)** | 13.2 tok | 15.5 tok (1.18×) | 65.5 tok (4.96×) | **+396.5%** |
| **OpenAI `o200k_base` (GPT-4o)** | 13.1 tok | 14.6 tok (1.12×) | 21.7 tok (1.66×) | **+66.3%** |

### Gateway Mitigation
* **Conversational Fluff Pruning:** Automatically strips Hinglish vocatives and attention-fillers (*"bhai suno"*, *"please batao na"*), cutting prompt token inflation by **24.7%** without altering question intent.
* **Cross-Lingual Semantic Cache:** Maps Hinglish and Devanagari queries directly to canonical cached answers, bypassing downstream token inflation entirely.

---

## Measured Gateway Performance

*Measured on commodity 8-core CPU hardware (Apple Silicon / 16GB RAM):*

| Metric | Measured Overhead | Defensible Meaning |
| :--- | :---: | :--- |
| **Cache Hit Latency** | **< 2.5 ms** | Sub-3ms memory resolution served at **$0.00** marginal cost |
| **Gateway CPU Overhead (p50)** | **5.44 ms** | Half of all requests clear all 5 gateway stages in < 6 ms |
| **Gateway CPU Overhead (p95)** | **13.86 ms** | P95 tail latency remains tightly bounded |
| **Throughput (Commodity CPU)** | **147.8 req/s** | Sustained concurrent proxying capacity on a single node |
| **Downstream LLM Latency** | 450 – 800 ms | Upstream frontier model API latency (OpenAI / Anthropic) |

---

## Verification & Test Suite

Token Optimiser includes a comprehensive verification harness executing type checking, unit suites, integration tests, and quality benchmarks:

```bash
python tests/run_all_tests.py
# Or via npm shortcut:
npm run test:all
```

```
┌──────────────────────────────────────────────────────┬──────────┬──────────┐
│ STAGE / TEST SUITE                                   │ RESULT   │ DURATION │
├──────────────────────────────────────────────────────┼──────────┼──────────┤
│ Frontend TypeScript Types (tsc)                      │ PASS     │   1.97s  │
│ Frontend Vitest Unit & Integration Suites            │ PASS     │   1.05s  │
│ Frontend Production Bundle (Vite)                    │ PASS     │   3.83s  │
│ Backend Code Quality & Linting (Ruff)                │ PASS     │   0.08s  │
│ Backend Pytest Suites (Unified + All 4 Features)     │ PASS     │   5.40s  │
└──────────────────────────────────────────────────────┴──────────┴──────────┘

  ✔ 100% PASSING — 258 TOTAL TESTS PASSED (52 frontend + 206 backend)
```

---

## Engineering Ownership & Architecture

Built as part of the Introduction to Machine Learning (IML) course engineering application:
* **Aman Kashyap:** Unified `/v1/chat/completions` gateway, singleflight coalescing engine, context-aware semantic cache, ML adaptive token budgeting, failure resilience, and quality evaluation harness.
* **Shared Foundation:** Modular feature structure with complete CPU-runnable execution and clean separation of authored fixtures vs. live upstream calls.

---

<div align="center">

Built with precision by **Aman Kashyap** · [GitHub Profile](https://github.com/AmanKashyapp07)<br/>
*Open-source, CPU-first inference gateway for cost-efficient AI applications.*

</div>
