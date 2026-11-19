<div align="center">

# Token Optimiser

### A cost-aware, CPU-first inference gateway for LLM applications

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8+-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![React](https://img.shields.io/badge/React-19.0-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](docker-compose.yml)
[![Tests](https://img.shields.io/badge/Tests-284%20Passing-10b981?style=flat-square&logo=pytest&logoColor=white)](tests/run_all_tests.py)

Token Optimiser sits between client applications and upstream LLM providers (OpenAI, Groq, Ollama) to systematically reduce the cost per successful task. It combines a 9-gate semantic cache with adversarial trap validation, feature-based model complexity routing, Hinglish conversational prompt compaction, and adaptive output token budgeting. Designed CPU-first: runs with zero mandatory GPU dependencies and sub-6ms median gateway overhead.

[Architecture](#how-it-works) • [Economics & Baselines](#results-at-a-glance--economic-rigor) • [The Semantic Cache](#the-semantic-cache) • [Model Routing](#model-routing--ablation-study) • [Hinglish Compaction](#hinglish-prompt-compaction) • [Budgeting](#adaptive-token-budgeting) • [Reliability & Security](#reliability-concurrency--security) • [Competitive Matrix](#competitive-positioning) • [Limitations](#limitations-and-non-goals) • [Quickstart](#quickstart)

</div>

---

## How it works

<div align="center">
  <img src="assets/hld-token-3.png" alt="Token Optimiser — High Level Architecture" width="100%" />
  <br/><br/>
  <img src="assets/hld-token-2.png" alt="Token Optimiser — Internal Optimisation & Intelligence Pipeline" width="100%" />
</div>

Every request passes through an integrated 7-stage optimization pipeline:

1. **Intake & Multi-Tenant Isolation.** Validates OpenAI-compatible requests and synthesizes a scoped compound cache key (`tenant_id::model::system_hash::format::normalized_query`). Multi-turn conversations incorporate a cryptographic digest of historical turns.
2. **Two-Layer Cache Check.** O(1) compound hash lookup (< 0.15 ms). On a miss, an offline CPU subword hash vector lookup (< 1.85 ms) is verified against a deterministic 9-gate contextual compatibility barrier before cache reuse.
3. **Prompt Compaction.** Linguistic filtering prunes conversational discourse particles and filler phrases while immutably locking protected named entities, numbers, and polarity markers.
4. **Complexity Routing.** A 16-feature classifier evaluates linguistic, syntactic, and difficulty markers, routing routine tasks to economy tiers (`gpt-4o-mini`) and reserving reasoning-heavy tasks for frontier models (`gpt-4o`).
5. **Adaptive Token Budgeting.** Ridge regression estimates required completion length $\hat{y}$ from prompt features and enforces an analytical upper bound ($\hat{y} + 1.645\hat{\sigma}$), eliminating runaways without truncating valid outputs.
6. **Resilient Execution & Coalescing.** Concurrent identical queries coalesce via Singleflight. Circuit breaker state machine monitors upstream health and fails over on provider outages.
7. **Telemetry & Audit.** Returns standard OpenAI chat completion payloads augmented with latency breakdown spans, exact token counters, and cost delta telemetry.

---

## Results at a glance & Economic rigor

### 1. Headline Cost & "Cost per Correct Task"

Raw dollar savings are misleading if they degrade answer quality. To maintain scientific honesty, savings must be evaluated against ground-truth task correctness:

$$\text{Effective Cost per 10k Correct Tasks} = \frac{\text{Normalized Cost / 10k Tasks}}{\text{Task Correctness Rate}}$$

Evaluated across a standardized 250-task multi-domain benchmark grounded in canonical industry evaluation families (HumanEval, GSM8K, IndicQA, HotpotQA, RFC 8259 JSON) across two distinct provider ecosystems:

#### A. Proprietary Ecosystem: OpenAI API (GPT-4o vs. GPT-4o-mini)
*Provider Rates: GPT-4o (\$2.50 / \$10.00 MTok); GPT-4o-mini (\$0.15 / \$0.60 MTok)*

| Pipeline Configuration | Raw Cost (250 Tasks) | Normalized Cost / 10k Tasks | Task Correctness | Effective Cost / 10k Correct Tasks | Real Net Savings | Failure Profile |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Unconstrained Baseline (Always GPT-4o)** | $0.3462 | $13.85 | 93.6% | **$14.80** | *Baseline* | None (Maximum financial burn) |
| **Naive Static Cap (Fixed 40 Tokens)** | $0.1410 | $5.64 | 72.0% | **$7.83** | +47.1% | Severe truncation; 28% answers clipped |
| **Token Optimiser (ML Budget + Routing)** | **$0.1807** | **$7.23** | **86.4%** | **$8.37** | **+43.5%** | **Balanced; zero clipped answers** |
| **Exact Cache Hit (Layer 1)** | $0.0002 | $0.01 | 100.0%* | **$0.01** | **+99.9%** | Instant O(1) match (< 0.15 ms) |
| **Semantic Cache Hit (Layer 2)** | $0.0005 | $0.02 | 100.0%* | **$0.02** | **+99.8%** | 9-gate verified match (< 2.5 ms) |

*\*Note on Cache Correctness & Hit Costs:* Cache hits assume upstream authoring/validation of stored answers. $0.01 and $0.02 per 10,000 tasks represent the amortized server-side compute overhead of Redis memory lookups ($0.000001/req) and local embedding inference ($0.000002/req).

#### B. Open-Source Weights: Groq Cloud (LLaMA 3.3 70B vs. LLaMA 3.1 8B)
*Provider Rates: LLaMA-3.3-70B (\$0.59 / \$0.79 MTok); LLaMA-3.1-8B (\$0.05 / \$0.08 MTok)*

| Pipeline Configuration | Raw Cost (250 Tasks) | Normalized Cost / 10k Tasks | Task Correctness | Effective Cost / 10k Correct Tasks | Real Net Savings | Failure Profile |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Unconstrained Baseline (Always LLaMA-3.3-70B)** | $0.0820 | $3.28 | 91.2% | **$3.60** | *Baseline* | High open-source frontier spend |
| **Naive Static Cap (Fixed 40 Tokens)** | $0.0335 | $1.34 | 70.4% | **$1.90** | +47.2% | 29.6% outputs truncated |
| **Token Optimiser (ML Budget + Routing)** | **$0.0485** | **$1.94** | **85.6%** | **$2.27** | **+36.9%** | **Provider-agnostic token savings** |
| **Exact Cache Hit (Layer 1)** | $0.0002 | $0.01 | 100.0%* | **$0.01** | **+99.7%** | Memory index match (< 0.15 ms) |

---

### 2. Expected Savings Model ($E[\text{Savings}]$)

Reporting cache savings purely on hits is a tautology (hits cost almost nothing). The true economic return of a caching gateway depends on the traffic hit rate ($p_{\text{hit}}$) balanced against the latency and compute overhead of misses:

$$E[\text{Cost}] = p_{\text{hit}} \cdot \text{Cost}_{\text{hit}} + (1 - p_{\text{hit}}) \cdot (\text{Cost}_{\text{LLM}} + \text{Cost}_{\text{miss\_overhead}})$$

$$E[\text{Savings}] = p_{\text{hit}} \cdot (\text{Cost}_{\text{LLM}} - \text{Cost}_{\text{hit}}) - (1 - p_{\text{hit}}) \cdot \text{Cost}_{\text{miss\_overhead}}$$

Sensitivity analysis across realistic enterprise production traffic mixes (assuming blended upstream LLM baseline of \$13.85 / 10k requests [\$1,385.00 / 1M requests], cache hit cost of \$0.01 / 10k requests, and \$0.0020 / 10k miss lookup overhead):

| Cache Hit Rate ($p_{\text{hit}}$) | Blended Spend / 10k Requests | Net Monthly Savings (1M req/mo) | Net Monthly Savings (10M req/mo) | Primary Workload Type |
| :---: | :---: | :---: | :---: | :--- |
| **10%** | $12.47 | **$138.22** | **$1,382.20** | Cold / Highly Unique Long-Tail Queries |
| **25%** | $10.39 | **$345.85** | **$3,458.50** | Standard Enterprise SaaS Applications |
| **40%** | $8.31 | **$553.48** | **$5,534.80** | Internal Documentation & Employee FAQ |
| **60%** | $5.55 | **$830.32** | **$8,303.20** | Repetitive Customer Support Automation |

---

### 3. Scale & Enterprise Break-Even Analysis

At small developer volumes (< 10,000 requests/month), raw API spend is too low for gateway optimization to offset hosting. Token Optimiser is engineered for production scale.

* **Fixed Monthly Infrastructure Cost**:
  * 1x Redis instance (Cache Index + Distributed Singleflight): \$15/month
  * 2x High-availability CPU container nodes (2 vCPU, 4GB RAM): \$50/month
  * *Total Fixed Infrastructure Overhead*: **\$65.00/month**
* **Combined Pipeline Savings**: At 25% cache hit rate + 14.6% routing & budgeting savings on remaining misses, net savings are **$497.55 per 1 Million requests**.
* **Break-Even Threshold**: Reached at **$\approx 130,000$ requests/month** ($\approx 4,300$ requests/day).
* **Enterprise Economics (10 Million requests/month)**:
  * Baseline Provider Spend (Unoptimized GPT-4o): $\$13,850.00/\text{month}$
  * Token Optimiser Spend (Blended Cache + Routing + Budgeting): $\$8,874.50/\text{month}$
  * Infrastructure Overhead: $-\$65.00/\text{month}$
  * **Net Monthly Savings**: **$\$4,910.50 / \text{month}$**
  * **Net Annual Enterprise Savings**: **$\$58,926.00 / \text{year}$**

---

## Answer quality & Validation benchmark

Token overlap heuristics (BLEU/ROUGE) are poor proxies for task correctness. Answers in Token Optimiser are validated with deterministic programmatic evaluators:
* **Code**: Extracted from markdown fences, parsed into an Abstract Syntax Tree (AST), and executed against test assertions in an isolated sandbox.
* **JSON**: Formally parsed against RFC 8259 specifications and checked for schema key compliance.
* **Factual & Reasoning**: Checked against strict rubric lists of required entities, values, and logical deduction steps.

### Benchmark Stress Test (Deliberately Strict "3 Bullets" Constraint on 250 Tasks)

The 250 evaluation tasks ([`eval_suite_v2.jsonl`](backend/data/eval/eval_suite_v2.jsonl)) are patterned directly after canonical industry benchmark suites:

| Domain | Canonical Pattern | Prompt Count | Token Reduction | Correctness Under Stress | Primary Failure Reason |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Coding & Algorithms** | **HumanEval / MBPP** | 35 | 23.5% | 85.7% (30/35) | Truncation of edge-case error handlers |
| **JSON Generation** | **RFC 8259 Schema-Bench** | 35 | 20.5% | 80.0% (28/35) | Closing brace truncated under arbitrary cap |
| **Logical Reasoning** | **GSM8K / ARC-Challenge** | 35 | 23.8% | 68.6% (24/35) | Multi-step mathematical proof steps omitted |
| **Indic & Hinglish QA** | **XQuAD / IndicQA / Belebele** | 40 | 30.2% | 60.0% (24/40) | Compound sentence answers over-compressed |
| **Summarization** | **CNN/DailyMail** | 35 | 23.7% | 77.1% (27/35) | Secondary point omission |
| **Multi-hop QA** | **HotpotQA** | 30 | 25.4% | 73.3% (22/30) | Intermediate premise lost |
| **General Knowledge** | **MMLU / TriviaQA** | 40 | 21.9% | 100.0% (40/40) | Zero failures (factual entities fit cleanly) |
| **Overall Stress Benchmark** | **Canonical Multi-Domain** | **250** | **24.1%** | **78.0% (195/250)** | **Intentional format stress-test baseline** |

*Why report the 78.0% stress score and 60.0% Indic score?*
Forcing complex multi-hop proofs and compound Indic sentences into a rigid "3 bullets" format intentionally stress-tests the failure envelope. In production, the gateway **does not force arbitrary formats**; it applies adaptive regression budgeting, maintaining **86.4% correctness** across the exact same suite while cutting token costs by 43.5%.

---

## The semantic cache

### 1. Dual-Backend Architecture

Token Optimiser supports two embedding engines, selected via configuration:
1. **Offline CPU Subword Hash Vectorizer (`hash`, default)**: Constructs 512-dimensional L2-normalized float64 vectors by hashing word unigrams and character trigrams. Requires 0 external downloads, 0 GPU memory, and executes in **< 1.85 ms** on CPU. Operates with a baseline cosine threshold of 0.55.
2. **Neural SentenceTransformer (`st`, optional)**: Loads `paraphrase-multilingual-MiniLM-L12-v2` (~120MB CPU-runnable checkpoint) covering 50+ languages. Operates with a tighter cosine threshold of 0.75.

### 2. Top-K Vector Retrieval & Pre-Filter Compatibility Gates

Vector cosine similarity serves solely as candidate retrieval. To avoid false-hit rejections prematurely collapsing cache recall, Layer 2 retrieves the top $K=5$ nearest vector candidates above the similarity threshold and applies 9 deterministic contextual pre-filters in rank order:

```
Incoming Query + Scoped Context Key
           │
           ▼
[ Layer 1: O(1) Compound Hash Lookup ] ──(Hit: <0.15ms)──► Return Authored / Cached Answer
           │ (Miss)
           ▼
[ Layer 2: Vector Top-K Candidates (K=5) ]
           │
           ▼
[ 9 Deterministic Context Compatibility Gates ]
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
           ├── (Candidate Fails Any Gate) ──► Inspect Candidate K+1 (Up to K=5)
           └── (First Candidate Passing All Gates) ──► Serve Verified Semantic Cache Hit
           │ (All K Candidates Exhausted/Rejected)
           ▼
Forward to Upstream Provider Pipeline
```

* **Quality-Gated Cache Admission (Defending Against Cheap-Model Poisoning)**:
  A critical failure mode in multi-tier caching is "cheap-model cache poisoning": if the router delegates an ambiguous prompt to an economy model (`gpt-4o-mini`) and it hallucinates, caching that output permanently degrades future tenant traffic.
  Token Optimiser enforces a strict **admission firewall**:
  1. **Frontier Model Responses (`gpt-4o`)**: Admitted to cache automatically upon successful upstream completion.
  2. **Economy Tier Responses (`gpt-4o-mini`)**: Admitted **only** if they pass deterministic validation (AST test suite execution for code, RFC 8259 syntax & schema validation for JSON, or exact entity rubric checks).
  3. **Unvalidated Economy Outputs**: Served directly to the caller with `Cache-Control: no-store` and discarded from the shared cache index.
  4. **Natural Language & FAQ Admission Policy**: Routine FAQ queries (e.g. campus policies, library hours) without programmatic AST/JSON schemas avoid cache starvation via structured promotion:
     * *Authored Seed Ingestion (Active)*: Enterprise FAQ and policy documents are pre-seeded into the cache via verified admin ingestion (`POST /api/v1/cache/entries`).
     * *Consensus Promotion ($N=2$ Agreement - Architecture Roadmap)*: For zero-temperature ($T=0$) FAQ queries, when 2 independent identical responses agree or match an authored entity pattern, the answer graduates to persistent cache.
     * *Async Shadow-Verification (Architecture Roadmap)*: High-frequency economy responses are enqueued for asynchronous background verification with the frontier model before cache promotion.

### 3. Empirical Evaluation & Statistical Confidence

* **Adversarial Evaluation Suite**: Evaluated across 430 prompt pairs, including 350 adversarial trap pairs spanning polar antonyms, numerical shifts, named-entity swaps, and language shifts.
* **Observed False Positive Rate**: **0 false hits out of 350 adversarial trap tests**.
* **Statistical Upper Bound (Rule of Three & Wilson Score Interval)**:
  Zero observed errors in $N=350$ independent adversarial trials yields a 95% one-sided upper bound via the Rule of Three:
  $$\text{FPR}_{95\%} \le \frac{3}{350} \approx 0.86\%$$
  Under a two-sided 95% Wilson score interval with continuity correction, the true population false-positive rate on adversarial prompts is bounded at $\mathbf{\le 1.05\%}$.
* **Precision vs. Recall Trade-off**:
  The compatibility gates intentionally prioritize precision over recall. Scanning top $K=5$ candidates improves paraphrase recall to **64.2%** (up from 52.0% with top-1 lookup) while maintaining 100% precision across all 350 adversarial trap pairs.

---

## Model routing & Ablation study

### 1. Why Routing Beats Heuristics (Empirical Ablation)

Prompt length is an unreliable proxy for reasoning difficulty:
* *"Prove that $\sqrt{2}$ is irrational."* (34 chars / 7 words) $\implies$ **High Complexity (Requires Frontier Reasoning)**
* *"Here is the official university attendance policy [500 words]... What time does the library close on Sundays?"* (650 chars / 110 words) $\implies$ **Low Complexity (Basic Extraction)**

We conducted an empirical ablation across the 200-prompt Quadrant Benchmark ([`routing_benchmark_200.jsonl`](backend/data/eval/routing_benchmark_200.jsonl)) comparing our multi-feature ML Classifier (`RandomForest` with 16 linguistic, lexical, and entity features) against heuristic baselines:

| Routing Architecture | Overall Accuracy | Under-Routing Risk (Hard sent to Cheap) | Over-Routing (Easy sent to Strong) | Cost / 10k Tasks | Savings vs Always-Strong | Failure Profile |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Always-Strong Baseline** | 50.0%* | 0.0% | 100.0% | $1.0800 | 0.0% | 100% financial waste on routine queries |
| **Always-Cheap Baseline** | 50.0%* | 100.0% | 0.0% | $0.2700 | 75.0% | Fails 100% of complex reasoning tasks |
| **Length-Only Heuristic (>25 words)** | 48.5% | **53.0%** | 50.0% | $0.6628 | 38.6% | **Disastrous quality risk; fails 53% of hard queries** |
| **Token Optimiser (`economy`)** | **78.0%** | 16.0% | 28.0% | $0.7236 | **33.0%** | Maximum cost reduction policy |
| **Token Optimiser (`balanced`)** | **63.5%** | **6.0%** | 67.0% | $0.9220 | **14.6%** | **Production default; caps quality risk under 6%** |
| **Token Optimiser (`quality`)** | 62.0% | **6.0%** | 70.0% | $0.9342 | 13.5% | Ultra-conservative reasoning protection |

*\*Note on Always-Strong/Always-Cheap Accuracy:* Across a balanced benchmark with 100 hard and 100 easy prompts, routing all tasks to one tier scores 100/200 = 50.0%.

### 2. Router Training & Asymmetric Cost Rationale

* **Feature Vector (16 Features)**: Includes `char_length`, `word_count`, `devanagari_ratio`, `latin_ratio`, `code_mix_ratio`, `roman_hindi_ratio`, language one-hots, `difficulty_entity_density`, `difficulty_math_marker_count`, and boolean markers for code fences, formulas, emails, and amounts.
* **Validation Methodology & Confidence Bounds**: Evaluated across the 200-prompt Quadrant Benchmark using 5-fold Stratified Cross-Validation (`random_state=42`). Across the 5 folds ($n=40$ per fold, $N=200$ total), the router achieves a mean cross-validation accuracy of **$63.5\% \pm 6.7\%$** ($95\%$ binomial confidence interval, $\text{SE} \approx 3.4\%$). In production, continuous shadow-routing expands this evaluation across 1,000+ stratified enterprise prompts.
* **Why Accept 67% Over-Routing in Balanced Policy?**
  In enterprise inference gateways, under-routing (sending a hard query to a cheap model) breaks the user experience. Over-routing (sending an easy query to a strong model) incurs only a fraction-of-a-cent penalty. The router is calibrated with a conservative decision threshold (0.50) to cap under-routing risk at 6.0%.

---

## Hinglish prompt compaction

### 1. Conceptual Distinction: Devanagari Script vs Romanised Hinglish

* **Devanagari Script (UTF-8)**: Native Hindi in Devanagari script (`हिंदी`) suffers severe subword fragmentation on OpenAI's older `cl100k_base` tokenizer (**4.96x token inflation** compared to English), decreasing to **1.66x** on `o200k_base` (GPT-4o).
* **Romanised Hinglish (Latin Alphabet)**: Text written in Latin script (`bhai mera wifi nahi chal raha`) **does not suffer script-level tokenizer inflation** (1.12x–1.18x inflation). Its bloat stems from conversational discourse particles, honorific fillers, and redundant conversational framing.

### 2. Compaction Mechanics & Financial Value

The compaction engine uses linguistic rule sets and entity masks to prune discourse fillers:
```
Original:  "Bhai suno, mera laptop bilkul on nahi ho raha hai, please urgently fix karne ka step batao na" (17 tokens)
Compacted: "mera laptop bilkul on nahi ho raha hai fix karne ka step batao" (12 tokens)
Reduction: 29.4% tokens saved | Negation "nahi" preserved | Tense "raha hai" preserved
```
* **Dollar Impact**: Input tokens cost $\approx 1/4$ of output tokens ($0.15/MTok vs $0.60/MTok on mini). A 24.7% input token reduction saves **24.7% of input token billing**, translating to $\approx 6.2\%$ of blended total request spend. Its primary enterprise value is preventing context window bloat in multi-turn conversational agents.

### 3. Upstream Provider KV Prefix Caching Invariant

Modern foundation model APIs (OpenAI, Anthropic, Groq) offer automated 50%–90% discounts for prompts sharing identical prefixes ($>1024$ tokens).
* **Critical Architecture Invariant**: Compaction is strictly applied to **dynamic user turns and runtime query strings**. The gateway **never mutates the static system prompt or few-shot exemplars**. Mutating prefix tokens invalidates hardware KV-cache hit buffers on provider GPUs, causing financial penalties that outweigh token reduction savings. Token Optimiser guarantees prefix immutability.

---

## Adaptive token budgeting

Rather than applying a rigid, arbitrary `max_tokens` cap (which truncates 28% of complex outputs), the gateway predicts the required response length $\hat{y}$ using an analytical Ridge regression model trained on input length, language, and format directives:

$$\text{Allocated Budget} = \max\left(\text{floor}, \min\left(\text{ceiling}, \hat{y} + z_{0.95} \cdot \hat{\sigma}_{\text{residuals}}\right)\right)$$

Where $z_{0.95} = 1.645$, $\hat{\sigma} \approx 14.46$ tokens, with operational bounds $[\text{floor}=35, \text{ceiling}=300]$.

| Strategy | Direct Cost / 10k | Answers Truncated | Task Correctness | Effective Cost / 10k Correct Tasks | Incremental vs Routing Only | Engineering Trade-off |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Unconstrained (GPT-4o Baseline)** | $13.85 | 0.0% | 93.6% | **$14.80** | +$5.81 (+64.6%) | Baseline financial burn; tail latency runaway risk |
| **Routing Only (No Token Cap)** | $7.77 | 0.0% | 86.4% | **$8.99** | *Reference* | Routes to cheap tier; vulnerable to output loops |
| **Static 40-token Cap (with Routing)** | $5.64 | 28.0% | 72.0% | **$7.83** | -$1.16 (-12.9%) | Severe answer truncation; invalid code/JSON |
| **1.5x Input Length Heuristic** | $5.70 | 30.4% | 65.4% | **$8.71** | -$0.28 (-3.1%) | Fails on concise prompts needing long answers |
| **Token Optimiser (ML Budgeting + Routing)** | **$7.23** | **0.0%** | **86.4%** | **$8.37** | **-$0.62 (-6.9%)** | **Zero truncation; -43.5% net vs Unconstrained** |

*Clean Variable Isolation & Statistical Caveat:*
1. **Routing Alone**: Shifting routine traffic to the economy tier cuts direct spend from **$13.85** down to **$7.77 / 10k** ($8.99 effective, a 39.3% reduction vs baseline).
2. **Budgeting Incremental Impact**: Holding the router constant, ML Adaptive Budgeting further trims direct spend from **$7.77** to **$7.23 / 10k** (saving an incremental **7.0%** direct spend, or **$0.62 / 10k effective**) with **0.0% truncation error**, eliminating tail latency and token generation runaways without degrading task correctness.
3. *Distribution Caveat*: Response lengths are right-skewed and heteroscedastic. The $+1.645\hat{\sigma}$ margin functions as an operational upper tolerance ceiling to prevent runaway expansion, not a pure Gaussian confidence interval. For structured formats (JSON/Code), the gateway monitors `finish_reason == "length"` to trigger automated retry fallback if a response is prematurely cut off.

---

## Reliability, Concurrency & Security

* **Stateless Pod Architecture (Redis Unified Cluster State)**:
  In multi-node deployments, gateway replicas operate **100% stateless**. SQLite is strictly an isolated local development fallback. Production multi-replica deployments rely exclusively on Redis for:
  * Distributed Singleflight lock leases (`SETNX` with TTL) and pub/sub broadcast (`X-Gateway-Leader`).
  * Multi-tenant cache index and vector similarity search (Redis Stack / RediSearch HNSW).
  * Cluster-wide sliding-window circuit breaker failure counters.
  Replicas can be horizontally autoscaled or restarted with zero state divergence or lock contention.
* **Distributed Request Coalescing (Singleflight)**:
  Under spike traffic (e.g. 50 concurrent requests for identical queries), exactly 1 request is elected leader and calls the upstream provider; 49 followers wait on non-blocking async primitives.
  * *In-Process Coalescing*: Implemented via `asyncio.Event` within a single gateway process.
  * *Distributed Cluster Coalescing*: Utilizes Redis distributed locks (`SETNX` with TTL) and pub/sub channels (`X-Gateway-Leader`) to elect a cluster-wide leader across multi-worker deployments.
  * *Leader Failure Semantics*: If the leader crashes, all followers receive an immediate `LeaderExecutionError` without deadlock or hanging.
  * *Follower Disconnect*: Follower cancellation does not abort leader execution.
* **Multi-Tenant Security & Cache Isolation**:
  * *Credential-Derived Isolation*: `tenant_id` is extracted from verified `X-API-Key` headers or authenticated JWT claims, never trusted from arbitrary client payload parameters.
  * *Compound Key Scoping*: Records are keyed as `tenant_id::model::system_hash::format::normalized_query`.
  * *Multi-Turn Conversation Isolation*: Multi-turn chat cache keys hash historical turns: `sha256(turns[:-1]) :: last_turn`, preventing cross-turn collisions.
  * *Temperature Policy*: Deterministic caching strictly applies when $T=0$. Non-zero temperature requests automatically bypass cache lookup.
  * *Authenticated Cache Purges*: Mutating operations (`DELETE /api/v1/cache/entries`, `POST /api/v1/cache/invalidate`) require admin authentication via `X-API-Key`.
  * *Cache Eviction Policy*: Configurable TTL expiration (`CACHE_TTL_SECONDS`, default 24h) with LRU eviction in Redis.
* **Circuit Breaker FSM & Provider Failover**:
  State machine transitions `CLOSED` $\to$ `OPEN` $\to$ `HALF_OPEN`. Trips after 5 consecutive upstream timeouts or 5xx errors. While `OPEN`, immediately fails over to configured secondary providers (e.g. Groq $\to$ Ollama, or OpenAI $\to$ Groq) before probing upstream recovery with decorrelated jittered backoff.
* **Streaming Support (SSE) & Retry Trade-off**:
  Fully supports OpenAI-compatible Server-Sent Events (`stream=True`).
  * *Trade-off*: When streaming over SSE, response tokens are piped directly to the socket as they arrive to keep Time-To-First-Token (TTFT) sub-10ms. Consequently, `finish_reason == "length"` automatic retries **cannot unsend already-streamed bytes** and are disabled for streaming connections (the final chunk carries `"finish_reason": "length"`).
  * *Deferred Cache Admission*: Cache writes on streaming queries are deferred until the terminal `[DONE]` event is reached, where the full stream is reassembled and verified against quality admission gates.
* **Production vs Demo Behavior**:
  * `ENVIRONMENT=production`: Upstream failure returns HTTP 503. Mock fixtures are strictly prohibited.
  * `ENVIRONMENT=demo`: Fixture fallback is permitted, but strictly flagged via `X-Gateway-Fallback: true` and `optimisation.fallback_applied = true`.

---

## Performance

Measured on Apple Silicon (M2 Pro, 8 cores, 16 GB RAM) with a single Uvicorn ASGI process:

| Benchmark Metric | Measured Result | Production Target |
| :--- | :---: | :---: |
| **Layer 1 Exact Cache Hit Latency** | **< 0.15 ms** | < 1.0 ms |
| **Layer 2 Semantic Cache Hit Latency (Hash Backend)** | **1.85 ms** | < 5.0 ms |
| **Full Gateway Pipeline Overhead (Median p50)** | **5.44 ms** | < 10.0 ms |
| **Full Gateway Pipeline Overhead (95th % p95)** | **6.72 ms** | < 15.0 ms |
| **Throughput (Cold Cache / Uncached Execution)** | **147.8 req/s** | > 100 req/s |
| **Throughput (Warm Cache / Coalesced Spike)** | **> 1,200 req/s** | > 500 req/s |

*Upstream Provider Latency Context:* Raw upstream LLM network latency typically ranges from 450 ms to 1,200 ms. The Token Optimiser gateway adds less than **1.2% latency overhead** while saving up to **43.5% in effective cost**.

---

## Competitive positioning

| Capability | Token Optimiser | LiteLLM | Portkey | GPTCache | Native Provider Prompt Caching |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary Architecture** | Cost-optimization gateway | API standardization proxy | Enterprise observability | Vector semantic caching | Exact prefix discount |
| **Semantic Cache Safety** | **9-Gate Verification Engine** | Exact match (Vector opt-in) | Exact match (Vector opt-in) | Raw vector similarity (high FPR) | Exact prefix only (no semantic) |
| **Named-Entity & Negation Gates** | **Yes (Deterministic Pre-filter)** | No | No | No | N/A |
| **Output Token Budgeting** | **ML Adaptive Ridge Bounds** | Manual static cap | Manual static cap | None | None |
| **Prompt Compaction** | **Linguistic Indic/Hinglish** | None | None | None | None |
| **Runtime Dependencies** | **Gateway: < 50MB RAM (CPU) + Redis Stack** | Python proxy | Cloud / SaaS | Local vector DB | Cloud provider |
| **Multi-Tenant Key Scoping** | **Built-in Compound Keys** | Team-level | Workspace-level | None | Account-level prefix |

---

## Limitations and non-goals

Being explicit about where Token Optimiser should *not* be used is essential for production deployments:

1. **Not a General-Purpose Model Training Harness**: Does not train or fine-tune foundation models. Models are lightweight classifiers and linear Ridge predictors designed to run on a 1-core CPU.
2. **Not Suitable for Unbounded Creative Writing**: Adaptive budgeting sets output bounds ($[35, 300]$ tokens). Open-ended novel generation or creative prose should bypass the budgeting stage (`max_tokens=None`).
3. **Stochastic Sampling Bypasses Cache**: Semantic caching is automatically disabled when $T > 0$ because non-zero temperature explicitly requests varied, non-deterministic responses.
4. **Out-of-Domain Semantic Similarity**: For abstract domain queries without explicit lexical anchors (e.g. nuanced philosophical analogies), pure subword hash caching has lower recall; the neural SentenceTransformer backend (`st`) with a 0.75+ threshold should be configured.

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
    api_key="your-tenant-api-key",  # Upstream keys stay secure in the server-side .env
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
npm run benchmark:all     # Master execution: Latency, Router Ablation, Budgeting & Indic Tax (< 8s)
```

```
┌──────────────────────────────────────────────────────┬──────────┬──────────┐
│ STAGE / TEST SUITE                                   │ RESULT   │ DURATION │
├──────────────────────────────────────────────────────┼──────────┼──────────┤
│ Frontend TypeScript Types (tsc)                      │ PASS     │   1.89s  │
│ Frontend Vitest Unit & Integration Suites            │ PASS     │   1.04s  │
│ Frontend Production Bundle (Vite)                    │ PASS     │   3.77s  │
│ Backend Code Quality & Linting (Ruff)                │ PASS     │   0.10s  │
│ Backend Pytest Suites (Unified + All 4 Features)     │ PASS     │   8.58s  │
└──────────────────────────────────────────────────────┴──────────┴──────────┘
✔ 100% PASSING — 284 TOTAL TESTS PASSED (52 frontend + 232 backend)
```

---

<div align="center">

Built with engineering rigor by **Aman Kashyap** · [GitHub](https://github.com/AmanKashyapp07)

</div>
