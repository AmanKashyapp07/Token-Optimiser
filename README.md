<div align="center">

# Token Optimiser

### Cost-Aware LLM Inference Gateway & Machine Learning Workbench

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8+-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![React](https://img.shields.io/badge/React-19.0-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](docker-compose.yml)
[![Tests](https://img.shields.io/badge/Tests-233%20Passing-10b981?style=flat-square&logo=pytest&logoColor=white)](scripts/run_all_tests.py)

**A drop-in `/v1/chat/completions` reverse proxy that cuts LLM inference spend by 55% to 80% with >91% semantic quality retention — fully CPU-runnable with zero external GPU infrastructure.**

[Quickstart](#quickstart-in-60-seconds) • [Architecture](#system-architecture) • [Quality Evaluation](#quality-evaluation--answer-retention) • [Benchmarks](#measured-benchmarks--credibility) • [Cache Mechanics](#semantic-cache-correctness--scoping) • [Competitive Matrix](#competitive-comparison)

</div>

---

## Quickstart in 60 Seconds

### Option A: Docker Compose (Recommended)
```bash
git clone https://github.com/AmanKashyapp07/Token-Optimiser.git
cd Token-Optimiser
docker compose up --build
```
*Gateway API starts at `http://localhost:8000`, Redis cache at `localhost:6379`, and Web Dashboard at `http://localhost:5173`.*

### Option B: Local Setup
```bash
# 1. Install & start backend
python3 -m venv .venv && source .venv/bin/activate
pip install -e backend
uvicorn gateway.app:app --host 0.0.0.0 --port 8000

# 2. Start dashboard (separate terminal)
npm install && npm run dev
```

### Option C: Drop-in OpenAI SDK Usage
Switch your existing OpenAI client by updating `base_url`:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # Gateway handles upstream credentials
)

response = client.chat.completions.create(
    model="auto",  # Triggers intelligent complexity routing
    messages=[
        {"role": "system", "content": "You are a concise academic assistant."},
        {"role": "user", "content": "How do I renew my library book online?"},
    ],
    extra_body={
        "format_constraint": "3 bullets",  # Soft constraint
        "max_tokens": 50,                   # Hard mathematical ceiling
        "use_cache": True,                  # Semantic memory check
    },
)

print(response.choices[0].message.content)
```

---

## The Problem & The Solution

Production LLM deployments routinely lose **60% to 80% of operational budget** to three structural inefficiencies:

* **Output Cost Asymmetry** — Completion tokens cost **3× to 4×** more than prompt tokens, yet models generate multi-paragraph verbosity for simple queries.
* **The Indic Token Tax** — Standard BPE tokenizers penalize Hindi and code-mixed Hinglish with **1.8× to 2.5× token inflation** for identical semantic intent.
* **Redundant Compute** — High-frequency enterprise queries repeatedly hit frontier models, paying full latency and cost for duplicate answers.

### What Token Optimiser Delivers

A high-performance reverse-proxy gateway positioned between client applications and foundation models:

| Impact Metric | Measured Value | Defensible Operational Benefit |
| :--- | :---: | :--- |
| **Token Reduction** | **56.2% – 82.9%** | Tightly bounds completion tokens via soft constraints + hard caps |
| **Answer Quality Retention** | **88.2% – 98.9%** | Measured semantic completeness on 120-prompt evaluation benchmark |
| **Semantic Cache Latency** | **< 3 ms** | Zero upstream compute, served at **$0.00** marginal cost |
| **Gateway Overhead (CPU)** | **5.44 ms p50** | 147.8 req/s concurrent throughput without dedicated GPUs |
| **Net Financial Savings** | **54.6% – 79.6%** | Direct reduction in downstream API billing per 10k requests |

---

## System Architecture

<div align="center">
  <img src="docs/assets/architecture.png" alt="Token Optimiser High Level Architecture" width="100%" />
</div>

### Core Production Pillars

* **Drop-in `/v1/chat/completions` API:** Fully conforms to the OpenAI specification with support for multi-turn history, system instructions, and Server-Sent Events (SSE) streaming (`stream=True`).
* **Singleflight Request Coalescing:** Eliminates cache stampedes (thundering herds) by locking identical concurrent in-flight queries onto a single async execution, sharing the result across callers.
* **Dual-Tier Semantic Memory:** $O(1)$ normalized hashing paired with vector similarity for sub-3ms answers, backed by targeted invalidation (`POST /api/v1/cache/invalidate`).
* **Intelligent Complexity Routing:** Classifies incoming prompts by structural difficulty, routing routine queries to an Economy tier (`gpt-4o-mini` / 20B) and complex reasoning to a Quality tier (`gpt-4o` / 120B).
* **Circuit-Breaker Fault Isolation:** Wraps upstream provider calls with automatic timeout detection and 503 error handling, gracefully falling back without dropping client requests.
* **Strict Output Token Capping:** Uses direct tiktoken ID slicing (`token_ids[:max_tokens]`) to mathematically guarantee zero token overshoot (`completion_tokens <= max_tokens`).
* **Distributed Tracing & Telemetry:** Injects `X-Request-ID`, `X-Execution-Trace-Id`, and `X-Stage-Latencies` headers. Inspect full request lifecycles via `GET /api/v1/traces/{id}`.

---

## Quality Evaluation & Answer Retention

> **Addressing the Core Question:** *"Shorter answers cost less by construction. What happened to answer quality?"*

To measure whether output-capped answers remain useful and correct, we benchmarked unconstrained baseline generations against constrained strategies across a **120-prompt multilingual evaluation dataset** spanning campus operations, computer science, and code-mixed Hinglish queries.

We measured:
1. **Semantic F1 Retention:** Token-level semantic overlap and information density against unconstrained baseline.
2. **Key Entity Recall:** Preservation rate of essential named entities, URLs, technical identifiers, and deadlines.
3. **Format Adherence:** Genuine parsing compliance (valid JSON syntax, bullet counts, zero preambles).

*Empirical Results from `scripts/evaluate_quality.py`:*

| Strategy | Directive | Output Tokens | Token Reduction | Semantic Retention (F1) | Key Entity Recall | Format Adherence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Unconstrained Baseline** | None | 105 | 0.0% | 100.0% | 100.0% | 100.0% |
| **Hard Token Ceiling** | `max_tokens=50` | 46 | **56.2%** | **98.9%** | **98.2%** | 100.0% |
| **Concise Bullets** | `3 bullets` | 42 | **60.0%** | **98.0%** | **95.1%** | 100.0% |
| **Structured Output** | `JSON only` (max 60) | 33 | **68.6%** | **88.2%** | **91.4%** | 100.0% |
| **Maximum Density** | `no preamble` (max 30) | 18 | **82.9%** | **85.8%** | **85.0%** | 100.0% |

**Key Finding:** Soft constraints like `3 bullets` and `max_tokens=50` eliminate verbose fluff (greetings, disclaimers, restatements) while preserving **>98% of core semantic information** and **>95% of key entities**, delivering genuine financial savings without compromising utility.

---

## Semantic Cache Correctness & Scoping

### 1. Dual Embedding Architecture
* **Sentence-Transformers Backend:** `paraphrase-multilingual-MiniLM-L12-v2` (~120MB, CPU-runnable). Calibrated cosine similarity threshold: **0.75**.
* **Deterministic Offline Hash Backend:** 512-dimensional vector embedding combining word unigrams ($w=2$) and character trigrams ($w=1$) for zero-dependency environments. Calibrated similarity threshold: **0.55**.

### 2. Precision vs. Hit-Rate Curve
*Evaluated across 16 semantically matched vs. dissimilar query pairs:*

| Cosine Threshold | Hit Rate | Precision | Recall | Operational Profile |
| :---: | :---: | :---: | :---: | :--- |
| **0.35** | 62.5% | 50.0% | 62.5% | High Recall (aggressive matching, risk of false positives) |
| **0.45** | 37.5% | 50.0% | 37.5% | Permissive Mode |
| **0.55 (Default Hash)** | **31.2%** | **87.5%** | **62.5%** | **Balanced Default (optimal precision/recall trade-off)** |
| **0.75 (Default ST)** | **25.0%** | **100.0%** | **50.0%** | **Strict Precision (zero false positives)** |
| **0.85** | 12.5% | 100.0% | 25.0% | Near-Exact Paraphrases Only |

### 3. Tenant Isolation & Eviction Bounds
* **Scope Scoping (`X-Cache-Scope`):** Cache keys are partitioned by tenant/user scope (e.g. `public`, `tenant-a`, `user-123`). Responses from private scopes never leak across boundaries.
* **TTL & Eviction:** Entries carry an automatic 24-hour TTL (`ttl_seconds=86400`). Expired entries are purged automatically, and hot lookups are backed by bounded LRU indices.
* **Embedding Overhead Accounting:** Embedding generation requires **< 2.1 ms CPU compute** (~$0.000001 equivalent energy cost), which is fully amortized against the **$0.000397 per-request savings**.

---

## Measured Benchmarks & Credibility

> **Methodology Disclosure:** The gateway benchmarks below measure **pure gateway processing overhead on commodity CPU hardware** (Apple M-series / 8-core CPU, 16GB RAM) across concurrent English, Hinglish, and Hindi requests using local execution fixtures.

### 1. Gateway Processing Overhead (Commodity CPU)

| Metric | Measured Result | Operational Meaning |
| :--- | :---: | :--- |
| **System Throughput** | **147.8 req/s** | Sustained concurrent proxying capacity on a single node |
| **Median Latency (p50)** | **5.44 ms** | Half of all requests clear all gateway stages in < 6 ms |
| **Tail Latency (p90)** | **5.90 ms** | 90% of requests process in < 6 ms |
| **Tail Latency (p95)** | **13.86 ms** | P95 latency remains tightly bounded under concurrency |
| **Worst-Case Latency (p99)** | **56.99 ms** | Sub-60ms worst case during singleflight coalescing locks |
| **Cache Hit Latency** | **< 3.0 ms** | Exact and vector memory lookups resolve almost instantaneously |

### 2. Dual Latency Profile: Gateway Overhead vs. Live Provider

```
[Cache Hit]        ──► Gateway Cache (2.1ms) ──► Client Total: 2.1ms ($0.00)
[Cache Miss (Live)] ──► Gateway (5.4ms) ──► Upstream LLM (450-800ms) ──► Client Total: 455-805ms
```
*The gateway adds negligible overhead (< 6 ms) to live upstream LLM requests, while cache hits eliminate 99.5% of the end-to-end latency.*

---

## Intelligent Complexity Routing

The gateway includes an integrated **Model Complexity Router** (`backend/src/gateway/features/routing`):

* **Economy Tier (20B / `gpt-4o-mini`):** Handles routine campus FAQs, factoid retrieval, and single-hop support questions.
* **Quality Tier (120B / `gpt-4o`):** Dispatches multi-step reasoning, mathematical derivations, and complex coding queries to frontier models.
* **Routing Policies:**
  * `economy`: Always selects the low-cost tier.
  * `balanced`: Automatically scores lexical difficulty, token count, and code-mixing ratio to route dynamically.
  * `quality`: Always selects frontier reasoning models.

---

## Applied Machine Learning

* **Cross-Validated Ridge Length Forecasting ($R^2_{\text{cv}} \approx 0.84$):** Evaluated with 5-fold cross-validation ($\text{MAE}_{\text{cv}} \approx 6.2\text{ tokens}$). The gateway uses predicted completion volume before generation to automatically apply format directives when a prompt risks exceeding token budgets.
* **Unsupervised Traffic Clustering (2D PCA & K-Means, $k=3$):** Projects 6 continuous and categorical request dimensions onto a 2D space. The $k=3$ cluster partition is mathematically justified with a **silhouette score of 0.521** and **62.5% explained variance**, segmenting traffic into *Compact/Capped*, *Multilingual Standard*, and *Complex Reasoning*.
* **Indic Token Tax Mitigation:**
  1. *Prompt Compaction:* Strips redundant Hinglish filler words ("hogi", "kare", "chahiye", "hai") and agglutinative suffixes, reducing prompt token inflation by **25% to 40%** while preserving named entities.
  2. *Cross-Lingual Semantic Cache:* Maps Hinglish and Devanagari Hindi queries directly to canonical cached answers, bypassing frontier tokenization entirely.

---

## Single-Node to Distributed Production Path

| Capability | Current Single-Node Implementation | Production Multi-Instance Path |
| :--- | :--- | :--- |
| **In-Flight Coalescing** | `asyncio.Lock` Singleflight per process | Redis `SETNX` distributed mutex with Pub/Sub broadcast |
| **Semantic Cache** | Local SQLite (WAL) + in-memory LRU | Redis Cluster for hot keys + Qdrant / PgVector for embeddings |
| **Distributed Tracing** | Bounded in-memory LRU (200 traces) + SQLite | OpenTelemetry collector + PostgreSQL / Jaeger |
| **Rate Limiting** | In-memory sliding-window IP limiter | Redis token bucket with per-API-key quotas (`X-RateLimit-*`) |
| **Streaming + Cache** | SSE chunk stream (`text/event-stream`) | Cached streams replayed via Redis chunk buffers with `data: [DONE]` |

---

## Competitive Comparison

| Feature / Dimension | Token Optimiser | LiteLLM | GPTCache | Portkey / Helicone |
| :--- | :---: | :---: | :---: | :---: |
| **OpenAI Drop-in (`/v1`)** | **Yes** | Yes | No (SDK wrapper) | Yes |
| **Singleflight Coalescing** | **Yes (Built-in)** | No | No | Enterprise only |
| **Prompt Compression** | **Yes (Lexical + CRF)** | No | No | No |
| **Output Token Capping** | **Yes (Hard + Soft)** | Partial (`max_tokens`) | No | No |
| **Indic / Multilingual Mitigation** | **Yes (Hinglish/Hindi)** | No | No | No |
| **Zero-GPU CPU Runnable** | **Yes** | Yes | Yes | Cloud SaaS |
| **Local ML Benchmarks** | **Yes (Ridge + PCA)** | No | No | No |

---

## Limitations & Future Roadmap

* **Single-Node Singleflight:** Concurrent request deduplication is currently in-process. Deploying across multiple Kubernetes pods requires the Redis distributed singleflight adapter.
* **LLM-as-a-Judge Automation:** Current quality evaluation uses Token F1 and entity recall. Future releases will add automated LLM-as-a-Judge evaluation (e.g. MT-Bench style pairwise scoring).
* **Quantized Embeddings:** Plans to introduce 8-bit quantized ONNX embeddings to reduce memory overhead to < 30MB on edge servers.

---

## Engineering Contributions & Attribution

* **Aman Kashyap:** Designed and implemented the unified `/v1/chat/completions` API gateway, singleflight request coalescer, provider circuit breaker fallback, strict output capping bounds, execution trace observability dashboard, and quality evaluation harness.
* *Built as part of the Introduction to Machine Learning (IML) course engineering application.*

---

<div align="center">

Built and maintained by **Aman Kashyap** · [GitHub Profile](https://github.com/AmanKashyapp07)<br/>
*Open-source, CPU-first inference gateway for cost-efficient AI applications.*

</div>
