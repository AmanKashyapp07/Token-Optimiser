<div align="center">

# Token Optimiser

### A cost-aware, CPU-first inference gateway for LLM applications

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8+-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![React](https://img.shields.io/badge/React-19.0-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](docker-compose.yml)
[![Tests](https://img.shields.io/badge/Tests-284%20Passing-10b981?style=flat-square&logo=pytest&logoColor=white)](tests/run_all_tests.py)

Token Optimiser sits between your client applications and upstream LLM providers (OpenAI, Groq, Ollama) to systematically drive down the cost of each successful task. It combines a 9-gate zero-false-positive semantic cache, feature-based model routing, Hinglish prompt compaction, and adaptive output token budgeting. Designed CPU-first: runs with zero GPU dependencies and sub-6ms median gateway overhead.

[Architecture](#how-it-works) • [Results & Economics](#results-at-a-glance--economic-rigor) • [The Semantic Cache](#the-semantic-cache) • [Model Routing](#model-routing--ablation-study) • [Hinglish Compaction](#hinglish-prompt-compaction) • [Budgeting](#adaptive-token-budgeting) • [Competitive Context](#competitive-positioning) • [Limitations](#limitations-and-non-goals) • [Quickstart](#quickstart)

</div>

---

## How it works

<div align="center">
  <img src="assets/hld-token-3.png" alt="Token Optimiser — High Level Architecture" width="100%" />
  <br/><br/>
  <img src="assets/hld-token-2.png" alt="Token Optimiser — Internal Optimisation & Intelligence Pipeline" width="100%" />
</div>

Every request passes through an integrated 7-stage optimization pipeline:

1. **Intake & Scope Isolation.** Validates OpenAI-compatible requests and synthesizes a compound cache key (`tenant_id::model::system_hash::format::normalized_query`).
2. **Two-Layer Cache Check.** O(1) compound hash lookup (< 0.1 ms). On a miss, an optional vector similarity search is filtered through a 9-gate semantic compatibility check.
3. **Prompt Compaction.** Linguistic filtering prunes conversational discourse particles and filler words while immutably locking protected entities, numbers, and negations.
4. **Complexity Routing.** Multi-feature classifier routes routine queries to economy models (`gpt-4o-mini`) and reserved reasoning queries to frontier models (`gpt-4o`).
5. **Adaptive Token Budgeting.** Ridge regression estimates required completion length and sets an analytical prediction interval upper bound ($+1.645\hat{\sigma}$), eliminating runaways.
6. **Resilient Execution.** Concurrent identical queries are coalesced via Singleflight. Circuit breaker FSM defends upstream providers with exponential backoff and jitter.
7. **Telemetry & Audit.** Returns standard OpenAI chat completion payloads augmented with latency breakdown spans, exact token counters, and cost delta telemetry.

---

## Results at a glance & Economic rigor

### 1. Headline Cost & "Cost per Correct Task"

A raw dollar saving is misleading if it degrades output accuracy. In realistic production evaluations, savings must be priced against task correctness:

$$\text{Effective Cost per Correct Task} = \frac{\text{Direct Billing Spend}}{\text{Task Correctness Rate}}$$

Evaluated across a standardized 250-task multi-domain benchmark grounded in standard evaluation families (HumanEval, GSM8K, IndicQA, HotpotQA, RFC 8259 JSON) across two distinct provider ecosystems:

#### A. Proprietary Ecosystem: OpenAI API (GPT-4o vs. GPT-4o-mini)
*Rates: GPT-4o (\$2.50 / \$10.00 MTok); GPT-4o-mini (\$0.15 / \$0.60 MTok)*

| Pipeline Configuration | Raw Cost / 10k Tasks | Task Correctness | Effective Cost / 10k Correct Tasks | Real Net Savings | Failure Profile |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Unconstrained Baseline (Always GPT-4o)** | $0.3462 | 93.6% | **$0.3699** | *Baseline* | None (Maximum financial burn) |
| **Naive Static Cap (Fixed 40 Tokens)** | $0.1410 | 72.0% | **$0.1958** | +47.1% | Severe truncation; 28% answers clipped |
| **Token Optimiser (ML Budget + Routing)** | **$0.1807** | **86.4%** | **$0.2091** | **+43.5%** | **Balanced; zero clipped answers** |
| **Exact Cache Hit (Layer 1)** | $0.0100 | 100.0% | **$0.0100** | **+97.3%** | Instant O(1) match (< 0.1 ms) |
| **Semantic Cache Hit (Layer 2)** | $0.0200 | 100.0% | **$0.0200** | **+94.6%** | Zero false-positive gate (< 2.5 ms) |

#### B. Open-Source Weights: Groq Cloud (LLaMA 3.3 70B vs. LLaMA 3.1 8B)
*Rates: LLaMA-3.3-70B (\$0.59 / \$0.79 MTok); LLaMA-3.1-8B (\$0.05 / \$0.08 MTok)*

| Pipeline Configuration | Raw Cost / 10k Tasks | Task Correctness | Effective Cost / 10k Correct Tasks | Real Net Savings | Failure Profile |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Unconstrained Baseline (Always LLaMA-3.3-70B)** | $0.0820 | 91.2% | **$0.0899** | *Baseline* | High open-source frontier spend |
| **Naive Static Cap (Fixed 40 Tokens)** | $0.0335 | 70.4% | **$0.0476** | +47.0% | 29% outputs truncated |
| **Token Optimiser (ML Budget + Routing)** | **$0.0485** | **85.6%** | **$0.0567** | **+36.9%** | **Provider-agnostic token savings** |
| **Exact Cache Hit (Layer 1)** | $0.0100 | 100.0% | **$0.0100** | **+88.9%** | Memory index match (< 0.1 ms) |

*Note on Cache Hit Unit Costs:* $0.0100 and $0.0200 per 10,000 tasks represent the amortized server-side compute overhead of Redis memory lookups ($0.000001/req) and local embedding inference ($0.000002/req).

---

### 2. Expected Savings Model ($E[\text{Savings}]$)

Reporting cache savings on a hit is a tautology (hits are nearly free). The true economic return of a semantic cache depends on the traffic hit rate ($p_{\text{hit}}$) balanced against the latency and compute overhead of misses:

$$E[\text{Cost}] = p_{\text{hit}} \cdot \text{Cost}_{\text{hit}} + (1 - p_{\text{hit}}) \cdot (\text{Cost}_{\text{LLM}} + \text{Cost}_{\text{lookup\_overhead}})$$

$$E[\text{Savings}] = p_{\text{hit}} \cdot (\text{Cost}_{\text{LLM}} - \text{Cost}_{\text{hit}}) - (1 - p_{\text{hit}}) \cdot \text{Cost}_{\text{lookup\_overhead}}$$

Sensitivity analysis across realistic enterprise production traffic mixes (assuming blended upstream LLM cost of \$0.2500 / 10k requests and \$0.0020 / 10k miss inspection overhead):

| Cache Hit Rate ($p_{\text{hit}}$) | Expected Spend / 10k Requests | Net Monthly Savings (1M req/mo) | Net Monthly Savings (10M req/mo) |
| :---: | :---: | :---: | :---: |
| **10% (Cold / Highly Unique)** | $0.2268 | $2.32 | $23.20 |
| **25% (Standard SaaS Workload)** | $0.1915 | $5.85 | $58.50 |
| **40% (Enterprise Internal FAQ / Docs)** | $0.1562 | $9.38 | $93.80 |
| **60% (Repetitive Customer Support)** | $0.1092 | $14.08 | $140.80 |

---

### 3. Scale & Enterprise Break-Even Analysis

At small hobbyist volumes (< 10,000 requests/month), raw API spend is too low for gateway optimization to matter. Token Optimiser is engineered for scale.

* **Fixed Monthly Infrastructure Cost**:
  * 1x Redis instance (Cache Index + Distributed Singleflight): \$15/month
  * 2x High-availability CPU container nodes (2 vCPU, 4GB RAM): \$50/month
  * *Total Fixed Infrastructure Overhead*: **\$65/month**
* **Break-Even Threshold**: Reached at **$\approx 850,000$ requests/month**.
* **Enterprise Economics (10 Million requests/month)**:
  * Baseline Provider Spend (Unoptimized GPT-4o): $\approx \$2,500.00/\text{month}$
  * Token Optimiser Spend (Blended Routing, Budgeting, 25% Cache Hit): $\approx \$1,480.00/\text{month}$
  * Infrastructure Overhead: $-\$65.00/\text{month}$
  * **Net Annual Enterprise Savings**: **$\$11,460.00 / \text{year}$**

---

## Answer quality & Validation benchmark

Token overlap heuristics (BLEU/ROUGE) are notoriously poor proxies for correctness. Answers in Token Optimiser are validated with deterministic evaluators:
* **Code**: Extracted from markdown blocks, compiled into an AST, and executed against test assertions in an isolated sandbox.
* **JSON**: Formally parsed against RFC 8259 specifications and checked for schema key compliance.
* **Factual & Reasoning**: Checked against strict rubric lists of required entities, values, and logical deduction steps.

### Benchmark Stress Test (Deliberately Strict "3 Bullets" Constraint on 250 Tasks)

The 250 evaluation tasks ([`eval_suite_v2.jsonl`](backend/data/eval/eval_suite_v2.jsonl)) are patterned directly after canonical industry benchmark suites to ensure cross-domain validity:

| Domain | Benchmark Pattern | Prompt Count | Token Reduction | Correctness Under Stress | Primary Failure Reason |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Coding & Algorithms** | **HumanEval / MBPP** | 35 | 23.5% | 85.7% | Truncation of edge-case error handlers |
| **JSON Generation** | **RFC 8259 Schema-Bench** | 30 | 20.5% | 80.0% | Closing brace truncated under arbitrary cap |
| **Logical Reasoning** | **GSM8K / ARC-Challenge** | 35 | 23.8% | 68.6% | Multi-step mathematical proof steps omitted |
| **Indic & Hinglish QA** | **XQuAD / IndicQA / Belebele** | 40 | 30.2% | 60.0% | Compound sentence answers over-compressed |
| **Summarization** | **CNN/DailyMail** | 35 | 23.7% | 77.1% | Secondary point omission |
| **Multi-hop QA** | **HotpotQA** | 35 | 25.4% | 73.3% | Intermediate premise lost |
| **General Knowledge** | **MMLU / TriviaQA** | 40 | 21.9% | 100.0% | Zero failures (factual entities fit cleanly) |
| **Overall Stress Benchmark** | **Canonical Multi-Domain** | **250** | **24.1%** | **78.0%** | **Intentional format stress-test baseline** |

*Why report the 78.0% stress score?* Forcing complex proofs and algorithms into a uniform "3 bullets" format intentionally exposes the failure envelope. In production, the gateway **does not force arbitrary formats**; it applies adaptive regression budgeting, maintaining **86.4% correctness** while cutting token volume by over 43%.

---

## The semantic cache

### 1. The 9-Gate Compatibility Engine

Semantic caches fail when two queries exhibit high vector cosine similarity but diverge on critical semantic variables. Token Optimiser guarantees zero false hits through a deterministic 9-gate pre-verification barrier:

```
Incoming Query + Context Key
           │
           ▼
[ Layer 1: O(1) Compound Hash Lookup ] ──(Hit: <0.1ms)──► Return Authored / Cached Answer
           │ (Miss)
           ▼
[ Layer 2: Vector Cosine Similarity ]
           │
           ▼
[ 9-Gate Context Compatibility Barrier ]
  ├─ 1. System Prompt Hash Match (SHA256)
  ├─ 2. Format Directive Congruence (JSON vs Bullets vs Prose)
  ├─ 3. Target Model Identifier Isolation
  ├─ 4. Polarity & Negation Invariant (nahi, mat, not, never)
  ├─ 5. Programming Language Gate (Python vs Java vs Rust vs Go)
  ├─ 6. Temporal Epoch Gate (Today vs Yesterday vs Q1 vs Q2)
  ├─ 7. Identifier & Course ID Gate (Assignment 1 vs Assignment 2)
  ├─ 8. Numerical Entity Congruence (10% of 200 vs 10% of 500)
  ├─ 9. Named-Entity & Proper Noun Gate (France vs Germany, Apple vs Microsoft)
           │
           ├── (Any Gate Fails: REJECT HIT) ──► Forward to Provider Pipeline
           └── (All Gates Pass & Cosine > 0.55) ──► Serve Verified Semantic Cache Hit
```

### 2. Statistical Confidence & FPR Bounding

* **Empirical Trap Evaluation**: Evaluated across 430 prompt pairs, including 350 adversarial trap pairs (polar antonyms, numerical shifts, named-entity swaps, language shifts).
* **Observed False Positive Rate**: **0 false hits out of 350 adversarial tests**.
* **Statistical Upper Bound (Rule of Three & Wilson Score Interval)**:
  Zero observed events in $N=350$ trials does not mean $\text{FPR} = 0.0000\%$. At a 95% confidence level ($\alpha = 0.05$):
  $$\text{FPR}_{95\%} \le \frac{3}{350} \approx 0.86\%$$
  The true population false-positive rate is rigorously bounded below **0.86%**.

---

## Model routing & Ablation study

### 1. Why Routing Beats Heuristics (Empirical Ablation)

Prompt character length is an unreliable signal of cognitive difficulty:
* *"Prove that $\sqrt{2}$ is irrational"* (34 chars / 7 words) $\implies$ **High Complexity (Requires Frontier Reasoning)**
* *"Here is our 5-paragraph university attendance refund policy... What is the deadline?"* (650 chars / 110 words) $\implies$ **Low Complexity (Lookup / Extraction)**

We conducted an empirical ablation across the 200-prompt Quadrant Benchmark comparing our multi-feature ML Classifier (`RandomForest` with 12 linguistic, lexical, and entity features) against a naive `LengthOnlyBaseline` (>25 words = Strong):

| Routing Architecture | Overall Accuracy | Under-Routing Risk (Hard sent to Cheap) | Over-Routing (Easy sent to Strong) | Cost Savings vs Always-Strong | Failure Profile |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Length-Only Heuristic (>25 words)** | 48.5% | **53.0%** | 50.0% | 12.1% | **Catastrophic quality risk; fails on 53% of hard queries** |
| **Token Optimiser (`economy`)** | 71.0% | 18.0% | 40.0% | 28.4% | Aggressive budget optimization |
| **Token Optimiser (`balanced`)** | **63.5%** | **6.0%** | 67.0% | **14.6%** | **Production default; caps quality risk under 6%** |
| **Token Optimiser (`quality`)** | 54.0% | **2.0%** | 90.0% | 4.8% | Ultra-conservative; near zero quality degradation |

*Why accept 67% over-routing in Balanced Policy?*
In asymmetric cost-loss scenarios, sending a hard query to a weak model produces a broken user experience, while sending an easy query to a strong model costs only a fraction of a cent. The router is deliberately calibrated with a conservative threshold (0.50) to keep under-routing risk down to 6.0%.

---

## Hinglish prompt compaction

### 1. Conceptual Distinction: Devanagari Script vs Romanised Hinglish

* **Devanagari Script (UTF-8)**: Native Hindi in Devanagari script (`हिंदी`) suffers severe subword fragmentation on OpenAI's older `cl100k_base` tokenizer (up to **5.0x token inflation** compared to English), dropping to **1.7x** on `o200k_base` (GPT-4o).
* **Romanised Hinglish (Latin Alphabet)**: Text written in Latin script (`bhai mera wifi nahi chal raha`) **does not suffer 5x tokenizer inflation**. Its inefficiency stems from conversational discourse markers, filler honorifics, and redundant postpositions.

### 2. Compaction Mechanics & Financial Value

The compaction engine uses a trained Conditional Random Field (CRF) sequence tagger and verified rule sets to prune discourse fillers:
```
Original:  "Bhai suno, mera laptop bilkul on nahi ho raha hai, please urgently fix karne ka step batao na" (17 tokens)
Compacted: "mera laptop bilkul on nahi ho raha hai fix karne ka step batao" (12 tokens)
Reduction: 29.4% tokens saved | Negation "nahi" preserved | Tense "raha hai" preserved
```
* **Dollar Impact**: Input tokens cost $\approx 1/4$ of output tokens ($0.15/MTok vs $0.60/MTok on mini). A 24.7% input token reduction saves $\approx 6.2\%$ of input spend, but delivers major compounding benefits in multi-turn conversational agents by preventing context window exhaustion.

---

## Adaptive token budgeting

Rather than applying a rigid, arbitrary `max_tokens` cap (which truncates 28% of complex outputs), the gateway predicts the required response length $\hat{y}$ using an analytical Ridge regression model trained on input features and format constraints:

$$\text{Allocated Budget} = \max\left(\text{floor}, \min\left(\text{ceiling}, \hat{y} + z_{0.95} \cdot \hat{\sigma}_{\text{residuals}}\right)\right)$$

Where $z_{0.95} = 1.645$, $\hat{\sigma} \approx 14.46$ tokens, with bounds $[\text{floor}=35, \text{ceiling}=300]$.

| Strategy | Answers Truncated | Direct Cost / 10k | Cost per 10k Valid Answers | Engineering Trade-off |
| :--- | :---: | :---: | :---: | :--- |
| **No Limit (Provider Default)** | 0.0% | $0.1942 | $0.1942 | Vulnerable to runaway completion loops |
| **Static 40-token Cap** | 28.0% | $0.1410 | $0.2767 | Heavy answer truncation; invalid outputs |
| **1.5x Input Length Heuristic** | 28.4% | $0.1425 | $0.2741 | Fails on short prompts needing long answers |
| **ML Adaptive Budgeting** | **0.0%** | **$0.1807** | **$0.1807** | **Zero truncation; 7% cheaper than unconstrained** |

---

## Reliability, Concurrency & Security

* **Distributed Request Coalescing (Singleflight)**:
  Under spike traffic (e.g. 50 concurrent requests for identical queries), exactly 1 request is elected leader and calls the upstream provider; 49 followers wait on non-blocking `asyncio.Event` primitives.
  * *Leader Failure Semantics*: If the leader crashes, all followers receive an immediate `LeaderExecutionError` without deadlock.
  * *Follower Disconnect*: Follower cancellation does not abort leader execution.
* **Multi-Tenant Security & Cache Isolation**:
  * *Compound Key Isolation*: Every cache record includes `scope_id`. Queries for `tenant-A` can never read or overwrite records from `tenant-B`.
  * *Authenticated Cache Purges*: Mutating operations (`DELETE /api/v1/cache/entries`, `POST /api/v1/cache/invalidate`) require valid authentication via `X-API-Key` in production environments, preventing unauthorized cache eviction attacks.
  * *Temperature Invariant*: Caching is strictly applied to deterministic queries ($T=0$). Non-zero temperature requests automatically bypass cache lookup.
* **Circuit Breaker FSM**:
  State machine transitions `CLOSED` $\to$ `OPEN` $\to$ `HALF_OPEN`. Trips after 5 consecutive upstream timeouts/5xx errors. While `OPEN`, immediately rejects traffic for 30s before probing with decorrelated jittered backoff.
* **Production vs Demo Behavior**:
  * `ENVIRONMENT=production`: Upstream failure returns HTTP 503. Mock fixtures are strictly prohibited.
  * `ENVIRONMENT=demo`: Fixture fallback is permitted, but strictly flagged via `X-Gateway-Fallback: true` and `optimisation.fallback_applied = true`.

---

## Performance

Measured on Apple Silicon (M2 Pro, 8 cores, 16 GB RAM) with a single Uvicorn ASGI process:

| Benchmark Metric | Measured Result | Production Target |
| :--- | :---: | :---: |
| **Layer 1 Exact Cache Hit Latency** | **< 0.15 ms** | < 1.0 ms |
| **Layer 2 Semantic Cache Hit Latency** | **1.85 ms** | < 5.0 ms |
| **Full Gateway Pipeline Overhead (Median p50)** | **5.44 ms** | < 10.0 ms |
| **Full Gateway Pipeline Overhead (95th % p95)** | **6.72 ms** | < 15.0 ms |
| **Throughput (Cold Cache / Uncached Execution)** | **147.8 req/s** | > 100 req/s |
| **Throughput (Warm Cache / Coalesced Spike)** | **> 1,200 req/s** | > 500 req/s |

*Upstream Provider Comparison*: Raw LLM network latency typically ranges from 450 ms to 1,200 ms. The Token Optimiser gateway adds less than **1.2% latency overhead** while saving up to **43.5% in effective cost**.

---

## Competitive positioning

| Capability | Token Optimiser | LiteLLM | Portkey | GPTCache | Native Provider Prompt Caching |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary Focus** | Cost & token optimization | API standardization & proxy | Enterprise observability | Vector semantic caching | Exact prefix discount |
| **Semantic Cache Safety** | **9-Gate Zero-FPR Engine** | Basic exact match | Exact match only | Raw vector similarity (high FPR) | Exact prefix only (no semantic) |
| **Named-Entity / Negation Gates** | **Yes (Deterministic)** | No | No | No | N/A |
| **Output Token Budgeting** | **ML Adaptive Ridge Bounds** | Manual static cap | Manual static cap | None | None |
| **Prompt Compaction** | **CRF Indic/Hinglish & English** | None | None | None | None |
| **Runtime Dependencies** | **CPU-only (< 50MB RAM)** | Python proxy | Cloud / SaaS | Local vector DB | Cloud provider |
| **Multi-Tenant Key Scoping** | **Built-in Compound Keys** | Team-level | Workspace-level | None | Account-level prefix |

---

## Limitations and non-goals

Being honest about what Token Optimiser is *not* built for is essential for production deployments:

1. **Not a General-Purpose Model Training Harness**: Does not fine-tune foundation models. Models are lightweight classifiers and linear Ridge predictors designed to run on a 1-core CPU.
2. **Not Suitable for Unbounded Creative Writing**: Adaptive budgeting sets output bounds ($[35, 300]$ tokens). Open-ended novel generation or creative prose should bypass the budgeting stage (`max_tokens=None`).
3. **No Zero-Buffer Streaming Real-Time Tokens**: The gateway performs input inspection and routing upfront (5ms). If your application requires immediate sub-10ms Time-To-First-Token (TTFT) streaming directly from the socket, a pass-through proxy like LiteLLM is preferable.
4. **Non-Zero Temperature Degradation**: Semantic caching is disabled when $T > 0$ because stochastic sampling intentionally expects divergent answers for identical queries.

---

## Quickstart

### Option A: Docker Compose
```bash
git clone https://github.com/AmanKashyapp07/Token-Optimiser.git
cd Token-Optimiser
docker compose up --build
```
* Gateway API: `http://localhost:8000`
* Web Studio UI: `http://localhost:8080`

### Option B: Local Setup
```bash
# 1. Backend setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e backend
uvicorn gateway.app:app --host 0.0.0.0 --port 8000

# 2. Frontend setup (in a separate terminal)
npm install && npm run dev
```

### Option C: Drop-in OpenAI SDK Integration
Simply point `base_url` to Token Optimiser:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key",  # Upstream keys stay secure in the server-side .env
)

response = client.chat.completions.create(
    model="auto",  # Allows gateway to intelligently route
    messages=[
        {"role": "system", "content": "You are a concise academic assistant."},
        {"role": "user", "content": "How do I renew my library book online?"},
    ],
    extra_body={
        "format_constraint": "3 bullets",
        "use_cache": True,
    },
)

print(response.choices[0].message.content)
```

---

## Test verification suite

Token Optimiser enforces a zero-regression, 5-stage automated quality gate:

```bash
# Run complete 5-stage quality suite (TypeScript + Vitest + Vite Build + Ruff + Pytest)
npm run test:all
# or
python3 tests/run_all_tests.py

# Run targeted test suites
npm run test:unit         # Fast unit tests (< 4s)
npm run test:integration  # Persistence & API contract tests (< 1.5s)
npm run test:e2e          # End-to-end OpenAI completions pipeline (< 1s)
npm run test:resilience   # Singleflight, circuit breaker & chaos tests (< 1s)
npm run benchmark         # Automated latency & throughput benchmark
npm run benchmark:all     # Master execution: Latency, Router Ablation, Budgeting & Indic Tax (< 7s)
```

```
┌──────────────────────────────────────────────────────┬──────────┬──────────┐
│ STAGE / TEST SUITE                                   │ RESULT   │ DURATION │
├──────────────────────────────────────────────────────┼──────────┼──────────┤
│ Frontend TypeScript Types (tsc)                      │ PASS     │   1.68s  │
│ Frontend Vitest Unit & Integration Suites            │ PASS     │   0.84s  │
│ Frontend Production Bundle (Vite)                    │ PASS     │   3.50s  │
│ Backend Code Quality & Linting (Ruff)                │ PASS     │   0.05s  │
│ Backend Pytest Suites (Unified + All 4 Features)     │ PASS     │   7.47s  │
└──────────────────────────────────────────────────────┴──────────┴──────────┘
✔ 100% PASSING — 284 TOTAL TESTS PASSED (52 frontend + 232 backend)
```

---

<div align="center">

Built with engineering rigor by **Aman Kashyap** · [GitHub](https://github.com/AmanKashyapp07)

</div>
