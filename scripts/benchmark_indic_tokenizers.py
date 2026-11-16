#!/usr/bin/env python3
"""Empirical Multi-Tokenizer Indic Benchmark.

Measures the 'Indic Token Tax' (token inflation) across parallel English, Romanized Hinglish,
and Devanagari Hindi sentence triples on standard industry foundation model tokenizers:
- OpenAI o200k_base (GPT-4o)
- OpenAI cl100k_base (GPT-4 / GPT-3.5)
- OpenAI p50k_base (GPT-3 / Codex)

Also evaluates the mitigation effect of Token Optimiser's prompt compression engine.
"""
from __future__ import annotations

import json
from pathlib import Path

import tiktoken
from gateway.features.compression.service import get_service as get_compression_service

ROOT = Path(__file__).resolve().parents[1]
OUT_FILE = ROOT / "backend" / "data" / "eval" / "indic_tokenizer_benchmark.json"

PARALLEL_DATASET = [
    {
        "id": "indic-001",
        "category": "administrative",
        "en": "How do I renew my library book online before the due date?",
        "hi_latin": "Due date se pehle online library book renew kaise karun?",
        "hi_deva": "अंतिम तिथि से पहले ऑनलाइन लाइब्रेरी की किताब कैसे रिन्यू करें?",
    },
    {
        "id": "indic-002",
        "category": "academics",
        "en": "Where can I find the exam schedule and timetable for this semester?",
        "hi_latin": "Is semester ka exam schedule aur timetable kahan milega?",
        "hi_deva": "इस सेमेस्टर का परीक्षा कार्यक्रम और समय सारणी कहाँ मिलेगी?",
    },
    {
        "id": "indic-003",
        "category": "technical",
        "en": "My campus Wi-Fi connection keeps dropping on the second floor.",
        "hi_latin": "Second floor pe campus Wi-Fi bar bar disconnect ho raha hai.",
        "hi_deva": "दूसरी मंजिल पर कैंपस वाई-फाई बार-बार डिस्कनेक्ट हो रहा है।",
    },
    {
        "id": "indic-004",
        "category": "coursework",
        "en": "What is the deadline for submitting the machine learning assignment?",
        "hi_latin": "Machine learning assignment submit karne ki last date kya hai?",
        "hi_deva": "मशीन लर्निंग असाइनमेंट जमा करने की अंतिम तिथि क्या है?",
    },
    {
        "id": "indic-005",
        "category": "finance",
        "en": "How can I pay the semester tuition fee using net banking or UPI?",
        "hi_latin": "Net banking ya UPI se semester tuition fee kaise bharein?",
        "hi_deva": "नेट बैंकिंग या यूपीआई का उपयोग करके सेमेस्टर ट्यूशन फीस कैसे भरें?",
    },
    {
        "id": "indic-006",
        "category": "housing",
        "en": "What is the procedure to request a room change in the student hostel?",
        "hi_latin": "Student hostel mein room change apply karne ka process kya hai?",
        "hi_deva": "छात्र छात्रावास में कमरा बदलने का अनुरोध करने की प्रक्रिया क्या है?",
    },
    {
        "id": "indic-007",
        "category": "career",
        "en": "When do campus placement drives start for final year engineering students?",
        "hi_latin": "Final year engineering students ke liye placement kab shuru honge?",
        "hi_deva": "अंतिम वर्ष के इंजीनियरिंग छात्रों के लिए कैंपस प्लेसमेंट कब शुरू होंगे?",
    },
    {
        "id": "indic-008",
        "category": "health",
        "en": "Is the campus medical dispensary open on weekends for emergency cases?",
        "hi_latin": "Kya campus ka medical dispensary weekends pe emergency ke liye khula rehta hai?",
        "hi_deva": "क्या सप्ताहांत में आपातकालीन मामलों के लिए कैंपस चिकित्सा औषधालय खुला रहता है?",
    },
    {
        "id": "indic-009",
        "category": "scholarship",
        "en": "What are the eligibility criteria for the merit-cum-means scholarship?",
        "hi_latin": "Merit-cum-means scholarship ke liye kya criteria aur eligibility hai?",
        "hi_deva": "मेरिट-कम-मीन्स छात्रवृत्ति के लिए क्या पात्रता मानदंड हैं?",
    },
    {
        "id": "indic-010",
        "category": "sports",
        "en": "How do I register for the annual inter-college badminton tournament?",
        "hi_latin": "Annual inter-college badminton tournament mein registration kaise hoga?",
        "hi_deva": "वार्षिक अंतर-कॉलेज बैडमिंटन टूर्नामेंट के लिए पंजीकरण कैसे करें?",
    },
    {
        "id": "indic-011",
        "category": "academics",
        "en": "Can I audit this advanced algorithms course without taking the final exam?",
        "hi_latin": "Kya main bina final exam diye ye advanced algorithms course audit kar sakta hun?",
        "hi_deva": "क्या मैं अंतिम परीक्षा दिए बिना इस उन्नत एल्गोरिदम पाठ्यक्रम का ऑडिट कर सकता हूँ?",
    },
    {
        "id": "indic-012",
        "category": "administrative",
        "en": "Where should I submit the medical certificate for attendance condonation?",
        "hi_latin": "Attendance condonation ke liye medical certificate kahan submit karna hoga?",
        "hi_deva": "उपस्थिति में छूट के लिए मेडिकल सर्टिफिकेट कहाँ जमा करना होगा?",
    },
    {
        "id": "indic-013",
        "category": "transport",
        "en": "What are the timings for the campus shuttle bus between north and south campus?",
        "hi_latin": "North aur south campus ke beech chalne wali shuttle bus ki timings kya hain?",
        "hi_deva": "उत्तरी और दक्षिणी परिसर के बीच कैंपस शटल बस का समय क्या है?",
    },
    {
        "id": "indic-014",
        "category": "coding",
        "en": "Why does this recursive depth-first search throw a maximum recursion depth error?",
        "hi_latin": "Ye recursive depth-first search maximum recursion depth error kyun de raha hai?",
        "hi_deva": "यह पुनरावर्ती डेप्थ-फर्स्ट सर्च अधिकतम पुनरावृत्ति गहराई त्रुटि क्यों दे रहा है?",
    },
    {
        "id": "indic-015",
        "category": "general",
        "en": "How can I obtain an official transcript for international university applications?",
        "hi_latin": "International university applications ke liye official transcript kaise milegi?",
        "hi_deva": "अंतर्राष्ट्रीय विश्वविद्यालय आवेदनों के लिए आधिकारिक प्रतिलेख कैसे प्राप्त करें?",
    },
]


def run_indic_benchmark() -> dict:
    encodings = {
        "cl100k_base (GPT-4 / 3.5)": tiktoken.get_encoding("cl100k_base"),
        "o200k_base (GPT-4o)": tiktoken.get_encoding("o200k_base"),
        "p50k_base (GPT-3 / Codex)": tiktoken.get_encoding("p50k_base"),
    }

    comp_svc = get_compression_service()
    results = {}

    print("\n" + "=" * 80)
    print("  EMPIRICAL MULTI-TOKENIZER INDIC BENCHMARK")
    print("=" * 80)

    for enc_name, enc in encodings.items():
        total_en = 0
        total_latin = 0
        total_deva = 0

        for item in PARALLEL_DATASET:
            total_en += len(enc.encode(item["en"]))
            total_latin += len(enc.encode(item["hi_latin"]))
            total_deva += len(enc.encode(item["hi_deva"]))

        n = len(PARALLEL_DATASET)
        avg_en = total_en / n
        avg_latin = total_latin / n
        avg_deva = total_deva / n

        tax_latin = ((avg_latin - avg_en) / avg_en) * 100
        tax_deva = ((avg_deva - avg_en) / avg_en) * 100
        ratio_latin = avg_latin / avg_en
        ratio_deva = avg_deva / avg_en

        results[enc_name] = {
            "avg_tokens_english": round(avg_en, 2),
            "avg_tokens_hinglish": round(avg_latin, 2),
            "avg_tokens_devanagari": round(avg_deva, 2),
            "inflation_ratio_hinglish": round(ratio_latin, 2),
            "inflation_ratio_devanagari": round(ratio_deva, 2),
            "tax_pct_hinglish": round(tax_latin, 1),
            "tax_pct_devanagari": round(tax_deva, 1),
        }

        print(f"\nTokenizer: {enc_name}")
        print(f"  • English Baseline:       {avg_en:.2f} tokens/query (1.00x)")
        print(f"  • Romanized Hinglish:     {avg_latin:.2f} tokens/query ({ratio_latin:.2f}x inflation, +{tax_latin:.1f}%)")
        print(f"  • Devanagari Hindi:       {avg_deva:.2f} tokens/query ({ratio_deva:.2f}x inflation, +{tax_deva:.1f}%)")

    # Evaluate Token Optimiser Prompt Compression on Indic Texts
    print("\n" + "-" * 80)
    print("  MITIGATION: TOKEN OPTIMISER PROMPT COMPRESSION ON CONVERSATIONAL HINGLISH")
    print("-" * 80)

    conversational_hinglish = [
        "Bhai suno, online library book renew kaise karun please jaldi batao na.",
        "Yaar, is semester ka exam timetable aur schedule kahan milega bata do.",
        "Dekho bhai, second floor pe campus Wi-Fi bar bar disconnect ho raha hai.",
        "Suno, machine learning assignment submit karne ki last date kya hai bhai?",
        "Achha, net banking ya UPI se semester tuition fee kaise bharein?",
        "Bhai, student hostel mein room change apply karne ka process kya hai?",
        "Suno yaar, final year engineering students ke liye placement kab shuru honge?",
        "Dekho, kya campus dispensary weekends pe emergency ke liye khuli rehti hai na?",
        "Bhai please, merit-cum-means scholarship ke liye kya eligibility hai?",
        "Yaar suno, annual badminton tournament mein registration kaise hoga?",
    ]

    comp_results = []
    enc_eval = encodings["cl100k_base (GPT-4 / 3.5)"]

    for raw_query in conversational_hinglish:
        res = comp_svc.analyze(raw_query, method="heuristic", budget=0.7)
        raw_tokens = len(enc_eval.encode(raw_query))
        comp_text = res.get("compressed", raw_query)
        comp_tokens = len(enc_eval.encode(comp_text))
        savings = (raw_tokens - comp_tokens) / max(1, raw_tokens)

        comp_results.append({
            "raw_text": raw_query,
            "compressed_text": comp_text,
            "raw_tokens": raw_tokens,
            "compressed_tokens": comp_tokens,
            "savings_ratio": round(savings, 4),
        })

    avg_raw = sum(r["raw_tokens"] for r in comp_results) / len(comp_results)
    avg_comp = sum(r["compressed_tokens"] for r in comp_results) / len(comp_results)
    avg_savings_pct = ((avg_raw - avg_comp) / avg_raw) * 100

    print(f"  • Average Raw Conversational Tokens:        {avg_raw:.2f}")
    print(f"  • Average Compressed Tokens (Fluff Pruned): {avg_comp:.2f}")
    print(f"  • Net Token Inflation Reduction:            {avg_savings_pct:.1f}%")

    output_payload = {
        "dataset_size": len(PARALLEL_DATASET),
        "tokenizer_benchmarks": results,
        "compression_mitigation": {
            "avg_raw_tokens": round(avg_raw, 2),
            "avg_compressed_tokens": round(avg_comp, 2),
            "avg_token_reduction_pct": round(avg_savings_pct, 1),
        },
        "sample_triples": PARALLEL_DATASET[:3],
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print(f"\nBenchmark results saved to: {OUT_FILE}")
    print("=" * 80 + "\n")
    return output_payload


if __name__ == "__main__":
    run_indic_benchmark()
