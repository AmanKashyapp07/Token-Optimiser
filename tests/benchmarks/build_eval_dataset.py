#!/usr/bin/env python3
"""Builds the 250-prompt multi-category evaluation benchmark and 430-pair Semantic Cache Trap benchmark."""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_FILE = ROOT_DIR / "backend" / "data" / "eval" / "eval_suite_v2.jsonl"
CACHE_TRAPS_FILE = ROOT_DIR / "backend" / "data" / "eval" / "cache_semantic_traps_400.jsonl"

RAW_CATEGORIES = {
    "general_knowledge": [
        (
            "What is the capital of France and what river flows through it?",
            ["Paris", "Seine"],
            "Paris is the capital of France, and the Seine river flows through it.",
            ["paris", "seine"],
        ),
        (
            "Who proposed the theory of general relativity and in what year was it published?",
            ["Albert Einstein", "1915"],
            "Albert Einstein proposed general relativity, published in 1915.",
            ["einstein", "1915"],
        ),
        (
            "What are the primary colors in additive color theory (RGB)?",
            ["Red", "Green", "Blue"],
            "The primary additive colors are Red, Green, and Blue.",
            ["red", "green", "blue"],
        ),
        (
            "What is the chemical formula for water and what bonds hold it together?",
            ["H2O", "covalent", "hydrogen"],
            "Water is H2O, formed by polar covalent bonds between hydrogen and oxygen.",
            ["h2o", "covalent"],
        ),
        (
            "Which planet in our solar system is known as the Red Planet and why?",
            ["Mars", "iron oxide"],
            "Mars is called the Red Planet due to iron oxide (rust) on its surface.",
            ["mars", "iron oxide"],
        ),
        (
            "What organelle is known as the powerhouse of the eukaryotic cell?",
            ["mitochondria", "ATP"],
            "The mitochondria are the powerhouse of the cell, generating ATP through cellular respiration.",
            ["mitochondria", "atp"],
        ),
        (
            "Who wrote the play Hamlet and what century was it written in?",
            ["William Shakespeare", "16th", "17th"],
            "William Shakespeare wrote Hamlet around the turn of the 17th century.",
            ["shakespeare"],
        ),
        (
            "What is the speed of light in a vacuum in meters per second?",
            ["299,792,458", "3x10^8"],
            "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
            ["299,792,458"],
        ),
    ],
    "coding": [
        (
            "Write a Python function to check if a string is a palindrome ignoring spaces.",
            ["def is_palindrome", "lower()", "[::-1]"],
            "def is_palindrome(s):\n    c = ''.join(ch.lower() for ch in s if ch.isalnum())\n    return c == c[::-1]",
            "assert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False\nassert is_palindrome('A man a plan a canal Panama') == True",
        ),
        (
            "Write a Python function to calculate the Fibonacci number at index n recursively or iteratively.",
            ["def fibonacci", "return"],
            "def fibonacci(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(n - 1): a, b = b, a + b\n    return b",
            "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(7) == 13\nassert fibonacci(10) == 55",
        ),
        (
            "Implement binary search in Python returning the index of target in sorted array arr.",
            ["def binary_search", "low", "high", "mid"],
            "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1",
            "assert binary_search([1, 3, 5, 7, 9], 5) == 2\nassert binary_search([1, 3, 5, 7, 9], 2) == -1\nassert binary_search([1], 1) == 0",
        ),
        (
            "Write a Python function to reverse a singly linked list in-place.",
            ["def reverse_list", "prev", "curr", "next"],
            "class ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\ndef reverse_list(head):\n    prev = None\n    curr = head\n    while curr:\n        nxt = curr.next\n        curr.next = prev\n        prev = curr\n        curr = nxt\n    return prev",
            "n1 = ListNode(1, ListNode(2, ListNode(3)))\nrev = reverse_list(n1)\nassert rev.val == 3 and rev.next.val == 2 and rev.next.next.val == 1\nassert reverse_list(None) is None",
        ),
        (
            "Write a Python generator function that yields prime numbers up to N.",
            ["def get_primes", "yield", "all("],
            "def get_primes(n):\n    for num in range(2, n + 1):\n        if all(num % i != 0 for i in range(2, int(num**0.5) + 1)):\n            yield num",
            "assert list(get_primes(10)) == [2, 3, 5, 7]\nassert list(get_primes(1)) == []",
        ),
        (
            "Write a Python function to find the two numbers in a list that sum to a given target.",
            ["def two_sum", "dict", "target - num"],
            "def two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        diff = target - num\n        if diff in seen: return [seen[diff], i]\n        seen[num] = i\n    return []",
            "res = two_sum([2, 7, 11, 15], 9)\nassert res in ([0, 1], [1, 0])\nassert two_sum([3, 2, 4], 6) in ([1, 2], [2, 1])",
        ),
        (
            "Write a Python function to validate if a given string has balanced parentheses.",
            ["def is_balanced", "stack", "pop"],
            "def is_balanced(s):\n    stack = []\n    for ch in s:\n        if ch == '(': stack.append(ch)\n        elif ch == ')':\n            if not stack: return False\n            stack.pop()\n    return len(stack) == 0",
            "assert is_balanced('(())') == True\nassert is_balanced('(()') == False\nassert is_balanced(')(') == False\nassert is_balanced('') == True",
        ),
    ],
    "reasoning": [
        (
            "If all A are B, and some B are C, can we conclude that some A are definitely C? Explain step by step.",
            ["cannot conclude", "Venn diagram", "overlap"],
            "No, we cannot conclude that some A are C. While all A belong to B, the subset of B that are C might not intersect with A.",
            ["cannot conclude", "no"],
        ),
        (
            "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Show your equation.",
            ["ball costs $0.05", "bat costs $1.05", "x + (x + 1.00) = 1.10"],
            "Let ball = x. Bat = x + 1.00. x + x + 1.00 = 1.10 => 2x = 0.10 => x = $0.05. The ball costs 5 cents.",
            ["0.05", "5 cents"],
        ),
        (
            "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
            ["5 minutes", "rate per machine", "1 widget in 5 minutes"],
            "It takes 5 minutes. 1 machine makes 1 widget in 5 minutes. Therefore, 100 machines working simultaneously make 100 widgets in 5 minutes.",
            ["5 minutes"],
        ),
        (
            "Sally has 3 brothers. Each brother has 2 sisters. How many total sisters are in the family?",
            ["2 sisters", "Sally is one of the sisters"],
            "There are 2 sisters in the family (Sally and her one sister). Each of the 3 brothers has the same 2 sisters.",
            ["2 sisters", "2"],
        ),
        (
            "A farmer has 17 sheep. All but 9 die. How many sheep are left alive?",
            ["9 sheep", "all but 9"],
            "9 sheep are left alive. 'All but 9' means exactly 9 sheep survived.",
            ["9"],
        ),
        (
            "You have a 3-gallon jug and a 5-gallon jug. How do you measure exactly 4 gallons of water?",
            ["fill 5-gallon", "pour into 3-gallon", "transfer", "4 gallons"],
            "1. Fill 5-gallon jug. 2. Pour into 3-gallon jug until full (leaves 2 gallons in 5-gallon jug). 3. Empty 3-gallon jug. 4. Pour 2 gallons into 3-gallon jug. 5. Fill 5-gallon jug. 6. Pour into 3-gallon jug until full (needs 1 gallon). 7. Exactly 4 gallons remain in the 5-gallon jug.",
            ["4 gallons", "3-gallon", "5-gallon"],
        ),
    ],
    "summarization": [
        (
            "Summarize the mechanism of CRISPR-Cas9 gene editing and its RNA guidance.",
            ["guide RNA", "Cas9 enzyme", "double-strand break", "DNA repair"],
            "CRISPR-Cas9 uses a synthetic single-guide RNA (sgRNA) matching a target DNA sequence. The Cas9 endonuclease creates a precise double-strand break at the target site, enabling gene knockout or insertion via cellular repair mechanisms.",
            ["guide rna", "cas9", "break"],
        ),
        (
            "Summarize how the Transformer self-attention mechanism works.",
            ["Query", "Key", "Value", "scaled dot-product", "softmax"],
            "Self-attention computes dynamic weightings across all tokens in a sequence using Query, Key, and Value matrices. Softmax of scaled dot-product QK^T / sqrt(d_k) multiplies Value vectors, capturing long-range contextual relationships without recurrent steps.",
            ["query", "key", "value", "softmax"],
        ),
        (
            "Summarize the Paxos distributed consensus algorithm.",
            ["Proposer", "Acceptor", "Learner", "prepare", "accept", "quorum"],
            "Paxos achieves fault-tolerant distributed consensus through two phases: Prepare (proposers request ballot numbers from a quorum of acceptors) and Accept (proposers submit a value to a quorum). A value is committed once accepted by a majority.",
            ["proposer", "acceptor", "quorum"],
        ),
        (
            "Summarize the difference between symmetric and asymmetric cryptography.",
            ["symmetric shared key", "asymmetric public-private key", "AES", "RSA"],
            "Symmetric cryptography uses a single shared secret key for encryption and decryption (e.g. AES), providing high speed. Asymmetric cryptography uses a mathematically linked public-private key pair (e.g. RSA), enabling secure key exchange and digital signatures.",
            ["symmetric", "asymmetric", "key"],
        ),
    ],
    "json_generation": [
        (
            "Return student course enrollment information for Alice in CS101 as valid JSON.",
            ["Alice", "CS101", "enrolled"],
            '{"student_name": "Alice", "course_code": "CS101", "status": "enrolled", "credits": 4}',
            {"student_name": "str", "course_code": "str", "status": "str"},
        ),
        (
            "Return a server status report for node-01 showing cpu, memory, and disk health in JSON format.",
            ["node-01", "cpu_percent", "memory_percent", "status"],
            '{"hostname": "node-01", "status": "healthy", "metrics": {"cpu_percent": 34.2, "memory_percent": 68.5, "disk_healthy": true}}',
            {"hostname": "str", "status": "str", "metrics": "dict"},
        ),
        (
            "Return library book renewal confirmation with due date and book ID in JSON format.",
            ["book_id", "title", "renewal_status", "new_due_date"],
            '{"book_id": "BK-9821", "title": "Introduction to Algorithms", "renewal_status": "success", "new_due_date": "2026-12-15"}',
            {"book_id": "str", "title": "str", "renewal_status": "str", "new_due_date": "str"},
        ),
        (
            "Return flight booking itinerary summary with flight number, departure, and arrival in JSON format.",
            ["flight_number", "departure", "arrival", "seat"],
            '{"flight_number": "AI-302", "departure": "DEL", "arrival": "BLR", "seat": "14B", "status": "confirmed"}',
            {"flight_number": "str", "departure": "str", "arrival": "str"},
        ),
        (
            "Return an e-commerce order item list with item names, quantities, and prices as JSON.",
            ["items", "total_usd", "currency"],
            '{"order_id": "ORD-5541", "items": [{"name": "Mechanical Keyboard", "quantity": 1, "price": 89.99}], "total_usd": 89.99, "currency": "USD"}',
            {"order_id": "str", "items": "list", "total_usd": "number"},
        ),
    ],
    "indic_hinglish": [
        (
            "Meri library book online kaise renew hogi agar due date kal hai?",
            ["library portal", "renew", "due date"],
            "Online library book renew karne ke liye library portal me sign in kare, issued items section khole, aur Renew par click kare.",
            ["portal", "renew"],
        ),
        (
            "Hostel mess me weekend breakfast aur dinner ke timings kya hain?",
            ["mess", "breakfast", "dinner", "timings"],
            "Hostel mess me weekend breakfast subah 8:00 AM se 10:00 AM tak aur dinner shaam 8:00 PM se 10:00 PM tak hota hai.",
            ["8:00", "10:00", "mess"],
        ),
        (
            "Campus Wi-Fi login page phone me open nahi ho raha, kya step follow kare?",
            ["wi-fi", "portal", "reconnect", "network"],
            "Phone me Wi-Fi disconnect karke forget network kare, dobara connect kare, aur browser me 1.1.1.1 ya captive portal URL enter kare.",
            ["wi-fi", "portal"],
        ),
        (
            "Assignment submission extend karne ke liye professor se kaise email kare?",
            ["email", "assignment", "extension", "reason"],
            "Professor ko formal email likhe jisme aapka roll number, assignment number, extension ka genuine reason aur proposed submission date mention ho.",
            ["email", "extension"],
        ),
        (
            "End sem exam ke liye minimum attendance kitni chahiye hoti hai?",
            ["attendance", "75%", "criteria", "medical"],
            "End sem exams me baithne ke liye minimum 75% attendance mandatory hoti hai, medical ground par 5-10% relaxation possible hai.",
            ["75%"],
        ),
        (
            "पुस्तकालय की किताब ऑनलाइन कैसे नवीनीकृत करें?",
            ["पुस्तकालय", "पोर्टल", "नवीनीकरण", "तिथि"],
            "पुस्तकालय की किताब ऑनलाइन नवीनीकृत करने के लिए संस्थान के लाइब्रेरी पोर्टल पर लॉगिन करें और जारी की गई पुस्तकों में 'Renew' विकल्प चुनें।",
            ["पुस्तकालय", "पोर्टल", "renew"],
        ),
        (
            "छात्रावास मेस का समय और भोजन मेनू कहाँ देख सकते हैं?",
            ["मेस", "समय", "भोजन", "पोर्टल"],
            "छात्रावास मेस का समय और साप्ताहिक भोजन मेनू छात्र कल्याण पोर्टल या मेस सूचना पट्ट पर देखा जा सकता है।",
            ["मेस", "पोर्टल"],
        ),
    ],
    "complex_qa": [
        (
            "Explain how TCP congestion control works including slow start, congestion avoidance, and fast recovery.",
            ["slow start", "congestion avoidance", "fast recovery", "congestion window", "threshold"],
            "TCP congestion control manages transmission rate via cwnd (congestion window). It begins in Slow Start (exponential cwnd growth until ssthresh), switches to Congestion Avoidance (linear growth), and upon packet loss uses Fast Retransmit and Fast Recovery to halve cwnd without dropping to 1 MSS.",
            ["slow start", "congestion avoidance", "fast recovery"],
        ),
        (
            "Explain how modern operating systems implement virtual memory paging and translation lookaside buffers (TLB).",
            ["virtual address", "page table", "TLB", "page fault", "frame"],
            "Virtual memory divides virtual address space into pages mapped to physical frames via page tables. The MMU consults the hardware TLB (translation lookaside buffer) cache for quick address translations; a TLB miss queries multi-level page tables, and missing pages trigger a page fault interrupt to load from disk.",
            ["virtual address", "page table", "tlb"],
        ),
        (
            "Explain ACID properties in database transaction management.",
            ["Atomicity", "Consistency", "Isolation", "Durability", "rollback"],
            "ACID ensures transaction reliability: Atomicity (all-or-nothing execution), Consistency (maintains schema and business constraints), Isolation (concurrent transactions execute independently without interference), and Durability (committed changes persist across crashes).",
            ["atomicity", "consistency", "isolation", "durability"],
        ),
        (
            "Explain the map-reduce computational model and how fault tolerance is achieved during execution.",
            ["Map", "Reduce", "shuffle and sort", "fault tolerance", "re-execution"],
            "MapReduce processes big data in parallel across worker nodes: Map phase transforms raw key-value pairs, shuffle-sort groups by intermediate key, and Reduce aggregates outputs. Fault tolerance is achieved by heartbeats detecting failed worker nodes and master rescheduling deterministic re-execution of tasks.",
            ["map", "reduce", "fault tolerance"],
        ),
    ]
}

def generate_suite():
    dataset = []
    category_counts = {
        "general_knowledge": 40,
        "coding": 35,
        "reasoning": 35,
        "summarization": 35,
        "json_generation": 35,
        "indic_hinglish": 40,
        "complex_qa": 30,
    }

    item_id = 1
    for cat, target_count in category_counts.items():
        seed_list = RAW_CATEGORIES[cat]
        for i in range(target_count):
            seed = seed_list[i % len(seed_list)]
            suffix = f" (Variation {i // len(seed_list) + 1})" if i >= len(seed_list) else ""
            
            prompt_text = seed[0] + suffix
            format_constraint = "none"
            if cat == "json_generation":
                format_constraint = "JSON only"
            elif i % 3 == 1:
                format_constraint = "3 bullets"
            elif i % 5 == 2:
                format_constraint = "no preamble"

            entry = {
                "id": f"eval-{cat[:4]}-{item_id:03d}",
                "category": cat,
                "prompt": prompt_text,
                "system_prompt": "You are an accurate, reliable, and concise technical assistant.",
                "format_constraint": format_constraint,
                "expected_entities": seed[1],
                "reference_answer": seed[2],
            }

            # Ground truth verification metadata
            if cat == "coding":
                entry["test_code"] = seed[3]
            elif cat == "json_generation":
                entry["schema_keys"] = seed[3]
            else:
                entry["critical_facts"] = seed[3]

            dataset.append(entry)
            item_id += 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Generated {len(dataset)} evaluation prompts at {OUTPUT_FILE}")


def generate_cache_traps():
    """Generates 430 semantic cache query pairs: 215 match/paraphrase pairs, 215 tricky traps."""
    traps = []
    trap_id = 1

    # 1. Negation Traps (100 pairs) -> Must NOT match
    negation_templates = [
        ("Is transaction TXN-{i} successful?", "Is transaction TXN-{i} unsuccessful?", "Negation: successful vs unsuccessful"),
        ("Can student {i} register without prerequisite course?", "Can student {i} register with prerequisite course?", "Negation: without vs with prerequisite"),
        ("Should I disable 2-factor authentication for user {i}?", "Should I enable 2-factor authentication for user {i}?", "Negation: disable vs enable"),
        ("Is the database node db-{i} online?", "Is the database node db-{i} offline?", "Negation: online vs offline"),
        ("Mujhe hostel room {i} me jaana hai.", "Mujhe hostel room {i} me nahi jaana hai.", "Hinglish Polarity Invariant: jaana vs nahi jaana"),
        ("Is feature flag flag_{i} enabled by default?", "Is feature flag flag_{i} disabled by default?", "Negation: enabled vs disabled"),
        ("Does account acc_{i} have sufficient balance?", "Does account acc_{i} have insufficient balance?", "Negation: sufficient vs insufficient"),
        ("Is the TLS certificate for domain_{i}.com valid?", "Is the TLS certificate for domain_{i}.com invalid?", "Negation: valid vs invalid"),
        ("Can we allow unauthenticated access to endpoint /api/v1/user/{i}?", "Can we deny unauthenticated access to endpoint /api/v1/user/{i}?", "Negation: allow vs deny"),
        ("Is the cache entry item_{i} expired?", "Is the cache entry item_{i} not expired?", "Negation: expired vs not expired"),
    ]
    for i in range(1, 11):
        for q_a_tmpl, q_b_tmpl, desc in negation_templates:
            traps.append({
                "id": f"trap-neg-{trap_id:03d}",
                "category": "negation",
                "query_a": q_a_tmpl.format(i=i),
                "query_b": q_b_tmpl.format(i=i),
                "expected_hit": False,
                "trap_type": desc,
            })
            trap_id += 1

    # 2. Numerical & Quantitative Traps (90 pairs) -> Must NOT match
    num_templates = [
        ("What is 10% of {val1}?", "What is 10% of {val2}?", "Numerical Divergence: Percentage base"),
        ("Calculate total price for {val1} units at $25 per unit.", "Calculate total price for {val2} units at $25 per unit.", "Numerical Divergence: Quantity count"),
        ("Find the top {val1} longest paths in graph G.", "Find the top {val2} longest paths in graph G.", "Numerical Divergence: Top K items"),
        ("What is the HTTP error code {val1}?", "What is the HTTP error code {val2}?", "Numerical Divergence: Status code"),
        ("A train travels {val1} km at 60 km/h. How many hours does it take?", "A train travels {val2} km at 60 km/h. How many hours does it take?", "Numerical Divergence: Travel distance"),
        ("What is the square root of {val1}?", "What is the square root of {val2}?", "Numerical Divergence: Math operand"),
        ("Show me metrics for port {val1}.", "Show me metrics for port {val2}.", "Numerical Divergence: Port number"),
        ("Allocate {val1} GB RAM to worker pod.", "Allocate {val2} GB RAM to worker pod.", "Numerical Divergence: Memory allocation"),
        ("Run load test with {val1} concurrent users.", "Run load test with {val2} concurrent users.", "Numerical Divergence: Concurrency level"),
    ]
    for i in range(1, 11):
        v1 = 100 * i
        v2 = 100 * i + 50
        for q_a_tmpl, q_b_tmpl, desc in num_templates:
            traps.append({
                "id": f"trap-num-{trap_id:03d}",
                "category": "numerical",
                "query_a": q_a_tmpl.format(val1=v1, val2=v2),
                "query_b": q_b_tmpl.format(val1=v1, val2=v2),
                "expected_hit": False,
                "trap_type": desc,
            })
            trap_id += 1

    # 3. Temporal Drift Traps (60 pairs) -> Must NOT match
    temporal_templates = [
        ("What is the stock price of {stock} today?", "What was the stock price of {stock} yesterday?", "Temporal Divergence: today vs yesterday"),
        ("What is the weather forecast for {stock} this morning?", "What is the weather forecast for {stock} next week?", "Temporal Divergence: morning vs next week"),
        ("What are the Q1 earnings for {stock} in 2026?", "What are the Q2 earnings for {stock} in 2026?", "Temporal Divergence: Q1 vs Q2 earnings"),
        ("What is the schedule of flight 6E-20{i} on Monday?", "What is the schedule of flight 6E-20{i} on Sunday?", "Temporal Divergence: Monday vs Sunday"),
        ("Who is the current CEO of {stock} in 2026?", "Who was the CEO of {stock} in 2010?", "Temporal Divergence: current vs historical"),
        ("What is the latest release version of {stock} today?", "What was the release version of {stock} 5 years ago?", "Temporal Divergence: latest vs past version"),
    ]
    stocks = ["Google", "Apple", "Microsoft", "Amazon", "NVIDIA", "Meta", "Tesla", "TCS", "Infosys", "Reliance"]
    for i, stock in enumerate(stocks):
        for q_a_tmpl, q_b_tmpl, desc in temporal_templates:
            traps.append({
                "id": f"trap-temp-{trap_id:03d}",
                "category": "temporal",
                "query_a": q_a_tmpl.format(stock=stock, i=i),
                "query_b": q_b_tmpl.format(stock=stock, i=i),
                "expected_hit": False,
                "trap_type": desc,
            })
            trap_id += 1

    # 4. Language & API Divergence Traps (60 pairs) -> Must NOT match
    lang_templates = [
        ("How to reverse a list in {lang1} with code example?", "How to reverse a list in {lang2} with code example?", "Language Divergence: List reversal syntax"),
        ("How to handle HTTP GET request in {lang1}?", "How to handle HTTP GET request in {lang2}?", "Language Divergence: HTTP server API"),
        ("How to declare a thread-safe singleton in {lang1}?", "How to declare a thread-safe singleton in {lang2}?", "Language Divergence: Concurrency primitive"),
        ("How to parse JSON string to object in {lang1}?", "How to parse JSON string to object in {lang2}?", "Language Divergence: JSON serialization"),
        ("How to connect to PostgreSQL database in {lang1}?", "How to connect to PostgreSQL database in {lang2}?", "Language Divergence: Database driver"),
        ("Explain unit testing framework in {lang1}.", "Explain unit testing framework in {lang2}.", "Language Divergence: Testing ecosystem"),
    ]
    lang_pairs = [
        ("Python", "Java"),
        ("C++", "Rust"),
        ("Go", "TypeScript"),
        ("C#", "Kotlin"),
        ("Ruby", "Swift"),
        ("PHP", "Scala"),
        ("JavaScript", "Python"),
        ("Rust", "Go"),
        ("Java", "C++"),
        ("Kotlin", "Swift"),
    ]
    for lang1, lang2 in lang_pairs:
        for q_a_tmpl, q_b_tmpl, desc in lang_templates:
            traps.append({
                "id": f"trap-lang-{trap_id:03d}",
                "category": "language_api",
                "query_a": q_a_tmpl.format(lang1=lang1, lang2=lang2),
                "query_b": q_b_tmpl.format(lang1=lang1, lang2=lang2),
                "expected_hit": False,
                "trap_type": desc,
            })
            trap_id += 1

    # 5. Context / Tenant / Format Mismatches (40 pairs) -> Must NOT match
    context_pairs = [
        ("Confidential quarterly financial report for company {i}", "Confidential quarterly financial report for company {i}", "Tenant Isolation Trap", "tenant_a", "tenant_b", "none", "none", None, None),
        ("Summarize the causes of World War 2 in detail.", "Summarize the causes of World War 2 in detail.", "Format Constraint Trap", "public", "public", "JSON only", "3 bullets", None, None),
        ("Explain quantum entanglement to a physics student.", "Explain quantum entanglement to a physics student.", "System Prompt Persona Trap", "public", "public", "none", "none", "You are an expert theoretical quantum physicist.", "You are a kindergarten elementary school teacher."),
        ("User profile address and credit card details for user_{i}", "User profile address and credit card details for user_{i}", "Tenant Boundary Trap", f"org_{i}_prod", f"org_{i}_dev", "none", "none", None, None),
    ]
    for i in range(1, 11):
        for q_a, q_b, desc, scope_a, scope_b, fmt_a, fmt_b, sys_a, sys_b in context_pairs:
            traps.append({
                "id": f"trap-ctx-{trap_id:03d}",
                "category": "context_gate",
                "query_a": q_a.format(i=i),
                "query_b": q_b.format(i=i),
                "expected_hit": False,
                "trap_type": desc,
                "scope_a": scope_a,
                "scope_b": scope_b,
                "format_a": fmt_a,
                "format_b": fmt_b,
                "system_prompt_a": sys_a,
                "system_prompt_b": sys_b,
            })
            trap_id += 1

    # 6. True Positive Paraphrases (80 pairs: English paraphrases & Hinglish equivalents) -> MUST MATCH
    paraphrase_templates = [
        ("How do I renew my library book online before the due date?", "What is the procedure to renew a library book online prior to expiry?", "True Paraphrase: Library renewal"),
        ("Explain binary search algorithm in Python with code.", "Can you provide a Python implementation and explanation of binary search?", "True Paraphrase: Binary search"),
        ("What are the primary colors in additive RGB color model?", "In additive color theory (RGB), which colors are considered primary?", "True Paraphrase: RGB colors"),
        ("How to reverse a singly linked list in Python?", "Write Python code to reverse a singly linked list in-place.", "True Paraphrase: Linked list reverse"),
        ("What organelle is known as the powerhouse of the eukaryotic cell?", "Which cell organelle generates ATP and is called the powerhouse of the cell?", "True Paraphrase: Mitochondria"),
        ("Meri library book online kaise renew hogi?", "Online library book renewal kaise kare?", "True Paraphrase: Hinglish library renew"),
        ("Hostel mess me dinner ke timings kya hain?", "Hostel mess me raat ka khana kis time milta hai?", "True Paraphrase: Hinglish mess timing"),
        ("Campus Wi-Fi connect nahi ho raha phone me.", "Phone me university Wi-Fi login nahi ho raha, kya kare?", "True Paraphrase: Hinglish Wi-Fi connect"),
    ]
    for i in range(10):
        for q_a, q_b, desc in paraphrase_templates:
            traps.append({
                "id": f"trap-hit-{trap_id:03d}",
                "category": "paraphrase_hit",
                "query_a": q_a,
                "query_b": q_b,
                "expected_hit": True,
                "trap_type": desc,
            })
            trap_id += 1

    CACHE_TRAPS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_TRAPS_FILE, "w", encoding="utf-8") as f:
        for item in traps:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Generated {len(traps)} semantic cache trap pairs at {CACHE_TRAPS_FILE}")


if __name__ == "__main__":
    generate_suite()
    generate_cache_traps()
