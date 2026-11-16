<div align="center">

# Token Optimiser

### A cost-aware gateway for LLM requests

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8+-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![React](https://img.shields.io/badge/React-19.0-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](docker-compose.yml)
[![Tests](https://img.shields.io/badge/Tests-284%20Passing-10b981?style=flat-square&logo=pytest&logoColor=white)](tests/run_all_tests.py)

Token Optimiser sits between your app and your LLM provider and cuts the cost of each successful task. It does this with smart caching, adaptive token limits, and model routing. It runs on a plain CPU, so no GPU is needed.

[How it works](#how-it-works) • [Results](#results-at-a-glance) • [Cache](#the-semantic-cache) • [Routing](#model-routing) • [Hinglish](#hinglish-prompt-compaction) • [Budgeting](#adaptive-token-budgeting) • [Quickstart](#quickstart)

</div>

---

## How it works

<div align="center">
  <img src="hld-token-3.png" alt="Token Optimiser — High Level Architecture" width="100%" />
  <br/><br/>
  <img src="hld-token-2.png" alt="Token Optimiser — Internal Optimisation & Intelligence Pipeline" width="100%" />
</div>

Every request passes through a short pipeline:

1. **Intake.** The gateway validates the OpenAI-style request and builds a cache key from it.
2. **Cache check.** If a safe, matching answer already exists, it is returned right away.
3. **Prompt compression.** Filler words are trimmed, while negations, numbers, and tense are protected.
4. **Model routing.** Simple prompts go to a cheaper model (`gpt-4o-mini`), and hard ones go to a stronger one (`gpt-4o`).
5. **Token budgeting.** A small ML model picks a sensible output limit for each prompt.
6. **Provider call.** Duplicate in-flight requests are merged, and a circuit breaker protects against outages.
7. **Telemetry.** Each response reports its cost, timings, and trace headers.

---

## Results at a glance

Costs below are per 10,000 successful tasks, using standard provider pricing.

| Strategy | Cost | Savings | Task correctness |
| :--- | :---: | :---: | :---: |
| No optimisation (baseline) | $0.3462 | - | 93.6% |
| ML adaptive budgeting | $0.1718 | 50.4% | 86.4% |
| Exact cache hit | $0.0100 | 97.1% | 100% |
| Semantic cache hit | $0.0200 | 94.2% | 100% |

Actual savings depend on how often queries repeat, how long answers are, and which provider pricing you use. Cache hits are cheap but not free, since they still cost a little gateway compute and, for semantic hits, one embedding call.

---

## Answer quality

Token overlap scores can be misleading, so answers are checked with real validators:

- **Code** is parsed and run against unit tests in a sandbox.
- **JSON** is checked against RFC 8259 and the expected schema.
- **Facts** are checked for required names, dates, and notations.

Results on 250 prompts, using a deliberately strict "3 bullets" constraint on everything:

| Domain | Token reduction | Correctness |
| :--- | :---: | :---: |
| Coding and algorithms | 23.5% | 85.7% |
| JSON generation | 20.5% | 80.0% |
| Logical reasoning | 23.8% | 68.6% |
| Indic and Hinglish QA | 30.2% | 60.0% |
| Summarization | 23.7% | 77.1% |
| Multi-hop QA | 25.4% | 73.3% |
| General knowledge | 21.9% | 100.0% |
| **Overall** | **24.1%** | **78.0%** |

This 78.0% is a stress test, not the real-world figure. Forcing every answer into the same short format hurts harder tasks. In normal operation the gateway does not force one format. It sets a per-prompt budget instead, and correctness reaches **86.4%**, compared with 93.6% for the unconstrained baseline.

---

## The semantic cache

A semantic cache can go wrong when two prompts look similar but mean different things, such as "10% of 200" and "10% of 500". Token Optimiser avoids this in two ways.

**Isolated keys.** Each entry is keyed by tenant, model, system prompt, format, and the normalised query. Changing any of these gives a separate entry.

**A compatibility gate.** Before any similarity score is computed, a candidate must pass checks for:

- opposite meaning (successful vs unsuccessful, "nahi jaana" vs "jaana")
- different numbers
- different languages or APIs (Python vs Java)
- different time references (today vs yesterday, Q1 vs Q2)
- different tenants or output formats

**Two lookup layers.** The first layer is an exact hash match, which takes under 0.1 ms and skips embeddings entirely. If that misses, the second layer runs a vector similarity search within the same tenant and model.

**Freshness.** Stable answers can be cached for 1 to 7 days. Prompts that mention things like "today", "latest", or "stock price" skip the cache. Caches can also be cleared per tenant with `DELETE /v1/cache?tenant_id=...`.

**Test results.** The gate was tested on 430 prompt pairs, and none of the 350 trap pairs produced a false hit. At the production threshold (0.55), the cache catches 50% of true paraphrases with 100% precision. Lowering the threshold to 0.40 raises paraphrase recall to 75% and still produces zero false hits. The full report is in [`backend/data/eval/cache_fpr_benchmark_400.json`](backend/data/eval/cache_fpr_benchmark_400.json).

---

## Model routing

Prompt length says little about difficulty. "Prove that the square root of 2 is irrational" is short but hard, while a long FAQ question can be easy. The router looks at several features instead of length.

On 200 labelled prompts, the `balanced` policy gave these results:

| Measure | Result |
| :--- | :---: |
| Hard prompts sent to the cheap model (quality risk) | 6% |
| Easy prompts sent to the strong model (wasted spend) | 67% |
| Cost savings vs always using the strong model | 14.6% |

The policy is intentionally cautious. Sending a hard question to a weak model produces bad answers, while sending an easy one to a strong model only costs a little extra. So the router accepts some wasted spend to keep quality risk low. Cheaper `economy` and stricter `quality` policies are also available.

---

## Hinglish prompt compaction

Trimming words in Hinglish can reverse meaning. "Mujhe jaana hai" and "Mujhe nahi jaana hai" differ by one word. The compactor is built to avoid this.

1. Mask protected spans such as amounts, codes, and emails.
2. Remove filler words like "bhai", "yaar", and "suno".
3. Use a trained sequence tagger (CRF) to drop redundant postpositions.
4. Check that no negation, tense, or number was lost.
5. If any check fails, return the original prompt unchanged.

**Example**

- Before: `Bhai suno, mera laptop bilkul on nahi ho raha hai, please urgently fix karne ka step batao na` (17 tokens)
- After: `mera laptop bilkul on nahi ho raha hai fix karne ka step batao` (12 tokens, 29.4% shorter)

Words such as `nahi`, `mat`, `not`, `never`, `no`, and `without` are never removed. Tense markers such as `tha`, `hai`, and `hoga` are also kept. On the 40-prompt Indic test set, every negation and named entity survived.

**Why it matters.** Tokenizers favour English, so Hindi costs more tokens. In Devanagari, the older `cl100k_base` tokenizer uses about 5 times as many tokens as English, while `o200k_base` (GPT-4o) uses about 1.7 times as many. Trimming filler cuts prompt tokens by about 24.7%.

---

## Adaptive token budgeting

A fixed output cap such as 40 tokens cuts off longer answers. Instead, a Ridge regression model predicts how long each answer needs to be, then adds a safety margin:

`budget = prediction + 1.645 x standard deviation`

The budget stays between 35 and 300 tokens. On the 250-prompt test set, 98.8% of reference answers fit inside their budget.

| Strategy | Answers cut off | Cost per 10k valid tasks |
| :--- | :---: | :---: |
| No limit | 0.0% | $0.1942 |
| Fixed 40-token cap | 28.0% | $0.2767 |
| 1.5x input length | 28.4% | $0.2741 |
| **ML adaptive budget** | **0.0%** | **$0.1807** |

Fixed caps look cheap, but truncated answers count as failures, so each valid answer ends up costing more.

---

## Reliability

**Request merging.** If 50 identical requests arrive at once, one becomes the leader and calls the provider. The other 49 wait for its result. If the leader fails, the followers receive a clear error and do not hang. If a follower disconnects, the leader keeps running. The code is in `backend/src/gateway/core/singleflight.py` and has its own tests.

**Production behaviour.** With `ENVIRONMENT=production`, a provider failure returns HTTP 503. The gateway never serves fake demo answers. With `ENVIRONMENT=demo`, fixture fallback is allowed but always flagged with the `X-Gateway-Fallback: true` header.

**Multi-instance setups.** [`distributed_singleflight.py`](backend/src/gateway/features/cache/distributed_singleflight.py) sketches how to coordinate leaders across nodes with Redis locks and Pub/Sub. It falls back to local merging if Redis is unavailable.

---

## Performance

Measured on an Apple M2 Pro (8 cores, 16 GB) with one Uvicorn worker, 20 concurrent clients, and a 60-second load test.

| Metric | Result |
| :--- | :---: |
| Exact cache hit | under 0.5 ms |
| Semantic cache hit | under 2.5 ms |
| Gateway overhead, median | 5.44 ms |
| Gateway overhead, 95th percentile | 13.86 ms |
| Throughput, cold cache | 147.8 req/s |
| Throughput, warm cache | over 1,200 req/s |

For comparison, the upstream LLM call itself typically takes 450 to 800 ms.

---

## Tests

```bash
python tests/run_all_tests.py
# or
npm run test:all
```

The suite runs TypeScript checks, frontend tests, a production build, Ruff linting, and backend tests. All 284 tests pass (52 frontend and 232 backend).

---

## Quickstart

### Option A: Docker Compose
```bash
git clone https://github.com/AmanKashyapp07/Token-Optimiser.git
cd Token-Optimiser
docker compose up --build
```
The API runs at `http://localhost:8000`, Redis at `localhost:6379`, and the web studio at `http://localhost:5173`.

### Option B: Local setup
```bash
# Backend
python3 -m venv .venv && source .venv/bin/activate
pip install -e backend
uvicorn gateway.app:app --host 0.0.0.0 --port 8000

# Frontend (in a separate terminal)
npm install && npm run dev
```

### Option C: Use it with the OpenAI SDK
Change only the `base_url` in your existing client:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # upstream keys stay in the server-side .env
)

response = client.chat.completions.create(
    model="auto",  # lets the gateway pick the model
    messages=[
        {"role": "system", "content": "You are a concise academic assistant."},
        {"role": "user", "content": "How do I renew my library book online?"},
    ],
    extra_body={
        "format_constraint": "3 bullets",  # soft formatting hint
        "max_tokens": 50,                   # output ceiling
        "use_cache": True,                  # check the cache first
    },
)

print(response.choices[0].message.content)
```

---

<div align="center">

Built by **Aman Kashyap** · [GitHub](https://github.com/AmanKashyapp07)

</div>