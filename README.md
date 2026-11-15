<div align="center">

# Token Optimiser

### Cost-Aware LLM Inference Gateway & Machine Learning Workbench

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8+-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![React](https://img.shields.io/badge/React-19.0-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-6.3-646CFF?style=flat-square&logo=vite&logoColor=white)](https://vitejs.dev)
[![Tests](https://img.shields.io/badge/Tests-227%20Passing-10b981?style=flat-square&logo=pytest&logoColor=white)](scripts/run_all_tests.py)

<p align="center">
  <b>Semantic Caching</b> •
  <b>Prompt Compression</b> •
  <b>Difficulty-Aware Routing</b> •
  <b>Output Capping</b> •
  <b>Multilingual Token Analytics</b>
</p>

<p align="center">
  <a href="https://github.com/AmanKashyapp07/Token-Optimiser">View Repository</a> ·
  <a href="https://github.com/AmanKashyapp07/Token-Optimiser/issues">Report Issue</a>
</p>

</div>

---

## Executive Summary

Enterprise adoption of Generative AI is severely bottlenecked by inference economics. Production workloads routinely waste **60% to 80% of their operational spend** on three structural inefficiencies:

1. **The Output Cost Asymmetry:** Downstream LLM providers (OpenAI, Anthropic, Groq) charge **3× to 4× higher rates** for completion tokens than prompt tokens, while unconstrained models produce multi-paragraph verbosity for simple queries.
2. **The Indic Token Tax:** Standard Byte-Pair Encoding (BPE) vocabularies are heavily biased toward English, penalizing Hindi (Devanagari) and code-mixed Hinglish with **1.8× to 2.5× token inflation** for identical semantic intent.
3. **Redundant Inference Waste:** Enterprise and campus applications repeatedly send identical institutional queries to frontier models, incurring unnecessary latency and recurring financial cost.

**Token Optimiser** solves this as a lightweight, CPU-first inference gateway and analytics workbench. By orchestrating a 4-stage deterministic pipeline—semantic caching, prompt compression, difficulty-aware routing, and output capping—it delivers **up to 82.9% token reduction** and **< 10 ms cache responses at $0.00 compute spend**, all runnable without dedicated GPU hardware.

---

## The Value Proposition

| Stakeholder | Strategic Impact | Quantifiable Return |
| :--- | :--- | :--- |
| **Investors & CTOs** | Drastically slashes cloud inference bills and protects gross margins | **55% to 80% net cost savings** per 10k requests with guaranteed output bounds |
| **Machine Learning Researchers** | Transparent multilingual token density metrics and CPU-based statistical modeling | Supervised Ridge length forecasting ($R^2 \approx 0.84$) and unsupervised 2D PCA clustering ($k=3$) |
| **Enterprise Operations** | Zero GPU cluster requirements; sub-10ms response times for institutional queries | **$0.00 marginal cost** on semantic cache hits; ACID-compliant SQLite audit trail |

---

## System Architecture

Token Optimiser acts as an intelligent reverse-proxy gateway positioned between client applications and downstream foundation models:

```mermaid
flowchart LR
    Client([Client Application]) --> Gateway[Token Optimiser Gateway]

    subgraph Pipeline["4-Stage Optimization Pipeline"]
        direction TB
        S1["1. Semantic Memory Cache<br/>Exact normalized lookup + vector cosine similarity"]
        S2["2. Lexical & CRF Compression<br/>Prompt compaction with protected entity preservation"]
        S3["3. Complexity Router<br/>ML classifiers dispatching to 20B Economy vs 120B Quality"]
        S4["4. Output Capping & Analytics<br/>Layered format constraints, hard max_tokens ceilings"]
        S1 -- "Cache Miss" --> S2 --> S3 --> S4
    end

    Gateway --> S1
    S1 -- "Cache Hit (< 10ms, $0)" --> Out([Optimized Response])
    S4 --> Store[(SQLite Transaction Log)]
    S4 --> Dispatch[Provider Inference Dispatch]
    Dispatch --> Out
```

### The 4 Core Innovation Pillars

- **Zero-Cost Semantic Memory:** Dual-tier retrieval combines $O(1)$ normalized string hashing with vector embedding cosine similarity, resolving frequent queries in under 10 ms without dispatching LLM calls.
- **Safety-Preserving Compression:** Compresses prompts via lexical and CRF techniques while strictly preserving protected literals (URLs, emails, monetary figures, code identifiers).
- **Difficulty-Aware Model Routing:** Evaluates query linguistic complexity using trained classifiers (Logistic Regression, Random Forest, MLP) to route requests between an **Economy Tier** (20B, $0.075/MTok) and a **Quality Tier** (120B, $0.15/MTok) under SLA-driven policies.
- **Constraint Engineering & Output Capping:** Pairs soft structural constraints (`3 bullets`, `JSON only`, `no preamble`) with hard `max_tokens` ceilings to curb the most expensive token class (completions).

---

## Quantified Economics & Benchmarks

*Empirical results measured across 10,000 standard institutional requests using live provider catalog pricing ($0.075/MTok prompt, $0.30/MTok completion for 20B tier).*

| Inference Strategy | Format Directive | Output Tokens | Cost / 10k Reqs (USD) | Token Reduction | Cost Reduction |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Unconstrained Baseline** | None | 105 | $0.324 | 0% | 0% |
| **Hard Token Ceiling** | `max_tokens=50` | 46 | $0.147 | **56.2%** | **54.6%** |
| **Concise Formatting** | `3 bullets` | 42 | $0.138 | **60.0%** | **57.4%** |
| **Structured Output** | `JSON only` (max 60) | 33 | $0.117 | **68.6%** | **63.9%** |
| **Maximum Density** | `no preamble` (max 30) | 18 | $0.066 | **82.9%** | **79.6%** |

> **Key Takeaway:** Because completion tokens carry a 4× cost premium, capping generation length provides the single highest financial leverage in modern LLM system design.

---

## Scientific Innovation & Applied Machine Learning

Designed for both production deployment and academic inspection, the platform incorporates three core ML innovations:

### 1. Supervised Output Length Prediction
A regularized **Ridge Regression** model ($R^2 \approx 0.84, \text{MAE} \approx 6.2\text{ tokens}$) forecasts expected completion volume before token generation begins, utilizing prompt token counts, language classification, structural constraint flags, and ceiling parameters. Feature coefficients mathematically demonstrate that structured directives systematically suppress output inflation.

### 2. Unsupervised Traffic Clustering via 2D PCA & K-Means ($k=3$)
To understand real-world query distributions without manual tagging, the engine reduces 6 continuous and categorical dimensions (input tokens, output tokens, code-mixing ratio, cache status, compression state, and routing tier) onto a 2D principal component space, automatically grouping traffic into:
- **Cluster 0: Compact & Capped Requests** (low-latency, tightly bounded completions).
- **Cluster 1: Multilingual Standard Traffic** (Indic/Hinglish exploratory queries).
- **Cluster 2: High-Complexity Quality Traffic** (reasoning-intensive, uncompressed queries routed to frontier models).

### 3. Empirical Indic "Token Tax" Inspector
Quantifies tokenizer bias across English, Hinglish, and Hindi in real time. By calculating character density, token-to-character ratios, and Romanized-to-Devanagari code-mix ratios, the workbench makes token disparities transparent, providing the feature inputs necessary for equitable multilingual billing.

---

## Technology Stack

| Layer | Technologies | Architectural Role |
| :--- | :--- | :--- |
| **Gateway Core** | Python 3.11, FastAPI, Pydantic v2 | High-throughput asynchronous REST reverse proxy |
| **Statistical ML** | scikit-learn, NumPy, tiktoken | Ridge regression, 2D PCA, K-Means clustering, BPE tokenization |
| **Persistence** | SQLite3 (WAL mode), JSON fixtures | ACID transactional request logging and curated benchmark datasets |
| **Frontend UI** | React 19, TypeScript 5.8, Vite 6 | Strict type-safe interactive analytics and studio workbench |
| **Data Visualization**| Tailwind CSS v4, Recharts 2.15, Lucide | Real-time reactive telemetry, cost curves, and PCA scatter plots |

---

<div align="center">

Built and maintained by **Aman Kashyap** · [GitHub Profile](https://github.com/AmanKashyapp07)<br/>
*Developed for the Introduction to Machine Learning (IML) application workbench.*

</div>
