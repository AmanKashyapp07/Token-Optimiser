#!/usr/bin/env python3
"""Generates the 200-prompt Model Complexity Router Benchmark across 4 quadrants."""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_FILE = ROOT_DIR / "backend" / "data" / "eval" / "routing_benchmark_200.jsonl"

SHORT_HARD_SEEDS = [
    ("Prove that sqrt(2) is irrational.", "mathematical_proof"),
    ("Prove that the halting problem is undecidable in Turing machines.", "computability_theory"),
    ("Derive the time complexity of the Master Theorem case 2 with proof.", "algorithm_analysis"),
    ("Explain why QuickSort worst-case time is O(n^2) and how randomized pivot prevents it.", "algorithm_analysis"),
    ("Explain the ABA problem in lock-free CAS and how hazard pointers solve it.", "concurrency_systems"),
    ("Can two-phase commit guarantee liveness under arbitrary network partitions?", "distributed_systems"),
    ("Prove that any comparison-based sorting algorithm requires Omega(n log n) comparisons.", "complexity_proof"),
    ("What is the quorum intersection invariant in Paxos consensus?", "distributed_systems"),
    ("Explain cache coherence under MESI protocol during concurrent read-modify-write.", "computer_architecture"),
    ("Derive the closed-form solution for the Fibonacci recurrence using generating functions.", "discrete_math"),
]

LONG_EASY_SEEDS = [
    (
        "Here is the official institution student handbook document. The campus library is located in the north academic wing and is open Monday to Friday from 8:00 AM to 10:00 PM, on Saturdays from 9:00 AM to 6:00 PM, and on Sundays from 10:00 AM to 4:00 PM. Book renewal must be completed online 24 hours prior to due date to avoid a late fine of Rs 10 per day. Question: What time does the library close on Sundays?",
        "campus_faq_extraction",
    ),
    (
        "Attached below is the full transcript from the quarterly facilities management committee meeting. Present members included Dr. Verma, Prof. Sen, Mr. Thomas, and Ms. Rao. The committee reviewed the cafeteria renovations, central air conditioning maintenance schedule, and parking lot repaving project. Question: Who among the committee members was present at the meeting?",
        "meeting_notes_extraction",
    ),
    (
        "Please review the following complete university hostel regulations text. Students are strictly required to return to the hostel premises before 10:30 PM on weekdays and 11:00 PM on weekends. Visitors are permitted only in the common lounge between 4:00 PM and 7:00 PM. Cooking appliances exceeding 500W are not permitted in residential rooms. Question: What is the weekday hostel curfew time?",
        "hostel_rules_extraction",
    ),
    (
        "Welcome to the technical support portal. If you are experiencing difficulty connecting to the campus eduroam Wi-Fi network, please ensure that your device has 802.1x authentication enabled, set the EAP method to PEAP, set phase 2 authentication to MSCHAPv2, and enter your campus student email as the username. Question: What should the EAP method be set to for campus Wi-Fi?",
        "tech_support_extraction",
    ),
    (
        "Below is the complete university academic calendar for the monsoon semester. Classes commence on August 1st, mid-semester examinations will take place from October 10th to October 18th, the annual cultural festival is scheduled for November 5th, and end-semester examinations conclude on December 15th. Question: On what date do classes commence?",
        "academic_calendar_extraction",
    ),
]

STANDARD_ROUTINE_SEEDS = [
    ("What is the capital of France and what river flows through it?", "general_knowledge"),
    ("How do I convert a string to lowercase in Python?", "basic_programming"),
    ("When is the university health center open on weekdays?", "campus_routine"),
    ("What is the definition of photosynthesis in plants?", "basic_science"),
    ("Who wrote the play Romeo and Juliet?", "literature_fact"),
    ("What is the chemical symbol for gold in the periodic table?", "basic_chemistry"),
    ("What is the distance between the Earth and the Moon in kilometers?", "astronomy_fact"),
    ("How to print hello world in Java?", "basic_programming"),
    ("What are the days of the week in French?", "language_translation"),
    ("Where is the campus bookstore located?", "campus_routine"),
]

STANDARD_COMPLEX_SEEDS = [
    (
        "Design a high-throughput, low-latency distributed rate limiter supporting 100,000 RPS using Redis sliding window log and token bucket algorithm, detailing race conditions and memory optimizations.",
        "distributed_system_design",
    ),
    (
        "Compare and contrast B-Tree and Log-Structured Merge (LSM) tree storage engines for high-write database workloads, analyzing write amplification, compaction strategies, and point lookup latencies.",
        "database_internals",
    ),
    (
        "Design an end-to-end event-driven architecture for a real-time financial trading settlement engine with idempotent event deduplication, out-of-order message handling, and dead letter queue reprocessing.",
        "system_architecture",
    ),
    (
        "Explain how the Linux Virtual Memory Subsystem manages demand paging, copy-on-write page faults, translation lookaside buffer (TLB) shootdowns in symmetric multiprocessing, and page replacement under memory pressure.",
        "operating_systems",
    ),
    (
        "Design a multi-region distributed consensus protocol based on Raft with pipelined log replication, joint consensus cluster membership changes, and leader leases to serve linearizable read requests locally.",
        "distributed_consensus",
    ),
]


def generate_routing_benchmark():
    dataset = []
    item_id = 1

    # 1. Short-Hard: 50 items (Target: strong / Quality Tier)
    for i in range(50):
        prompt, subcat = SHORT_HARD_SEEDS[i % len(SHORT_HARD_SEEDS)]
        variation = f" (Variant {i // len(SHORT_HARD_SEEDS) + 1})" if i >= len(SHORT_HARD_SEEDS) else ""
        dataset.append({
            "id": f"route-sh-{item_id:03d}",
            "quadrant": "short_hard",
            "prompt": prompt + variation,
            "target_tier": "strong",
            "subcat": subcat,
            "complexity_level": "high",
            "length_bucket": "short",
        })
        item_id += 1

    # 2. Long-Easy: 50 items (Target: cheap / Economy Tier)
    for i in range(50):
        prompt, subcat = LONG_EASY_SEEDS[i % len(LONG_EASY_SEEDS)]
        variation = f" [Ref-ID #{i + 100}]" if i >= len(LONG_EASY_SEEDS) else ""
        dataset.append({
            "id": f"route-le-{item_id:03d}",
            "quadrant": "long_easy",
            "prompt": prompt + variation,
            "target_tier": "cheap",
            "subcat": subcat,
            "complexity_level": "low",
            "length_bucket": "long",
        })
        item_id += 1

    # 3. Standard Routine: 50 items (Target: cheap / Economy Tier)
    for i in range(50):
        prompt, subcat = STANDARD_ROUTINE_SEEDS[i % len(STANDARD_ROUTINE_SEEDS)]
        variation = f" (Query #{i + 1})" if i >= len(STANDARD_ROUTINE_SEEDS) else ""
        dataset.append({
            "id": f"route-sr-{item_id:03d}",
            "quadrant": "standard_routine",
            "prompt": prompt + variation,
            "target_tier": "cheap",
            "subcat": subcat,
            "complexity_level": "low",
            "length_bucket": "medium",
        })
        item_id += 1

    # 4. Standard Complex: 50 items (Target: strong / Quality Tier)
    for i in range(50):
        prompt, subcat = STANDARD_COMPLEX_SEEDS[i % len(STANDARD_COMPLEX_SEEDS)]
        variation = f" Consider edge cases in scenario {i + 1}." if i >= len(STANDARD_COMPLEX_SEEDS) else ""
        dataset.append({
            "id": f"route-sc-{item_id:03d}",
            "quadrant": "standard_complex",
            "prompt": prompt + variation,
            "target_tier": "strong",
            "subcat": subcat,
            "complexity_level": "high",
            "length_bucket": "long",
        })
        item_id += 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Generated {len(dataset)} routing benchmark prompts at {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_routing_benchmark()
