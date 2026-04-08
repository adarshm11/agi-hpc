#!/usr/bin/env python3
"""
Debate Validation Benchmark — Hard Numbers for the Writeup

Compares three configurations:
  A) Single-model (Id only, fast path)
  B) Dual-hemisphere debate (Superego + Id, 4 rounds)
  C) Debate + Ego arbitration (when disagreement is high)

200 questions across 5 categories:
  - Factual (40): verifiable facts with ground truth
  - Reasoning (40): logic puzzles, math, analysis
  - Ethics (40): moral dilemmas, fairness, safety
  - Creative (40): brainstorming, analogies, design
  - Code (40): debugging, algorithms, architecture

Metrics:
  - Accuracy (vs reference answers, scored by LLM judge)
  - Response quality (1-10, scored by LLM judge)
  - Latency (wall clock time)
  - Confidence calibration (predicted vs actual correctness)

Usage:
    python scripts/benchmark_debate.py --mode all       # full benchmark
    python scripts/benchmark_debate.py --mode single    # single-model only
    python scripts/benchmark_debate.py --category ethics # one category
    python scripts/benchmark_debate.py --dry-run        # show questions only
    python scripts/benchmark_debate.py --questions 10   # quick test

Output: benchmarks/debate_benchmark_results.json
"""

from __future__ import annotations

# ruff: noqa: E501 — question strings are intentionally long for readability

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)
logger = logging.getLogger("benchmark")

# LLM endpoints
LH_URL = os.environ.get("LH_URL", "http://localhost:8080")
RH_URL = os.environ.get("RH_URL", "http://localhost:8082")
EGO_URL = os.environ.get("EGO_URL", "http://localhost:8084")
RAG_URL = os.environ.get("RAG_URL", "http://localhost:8081")


# ---------------------------------------------------------------------------
# Benchmark questions
# ---------------------------------------------------------------------------

QUESTIONS: Dict[str, List[Dict[str, str]]] = {
    "factual": [
        {"q": "What is the capital of Australia?", "ref": "Canberra"},
        {"q": "Who wrote 'Pride and Prejudice'?", "ref": "Jane Austen"},
        {"q": "What is the chemical symbol for gold?", "ref": "Au"},
        {"q": "In what year did the Berlin Wall fall?", "ref": "1989"},
        {"q": "What is the largest planet in our solar system?", "ref": "Jupiter"},
        {
            "q": "What programming language was created by Guido van Rossum?",
            "ref": "Python",
        },
        {
            "q": "What is the speed of light in meters per second?",
            "ref": "299,792,458 m/s",
        },
        {"q": "Who painted the Mona Lisa?", "ref": "Leonardo da Vinci"},
        {"q": "What is the powerhouse of the cell?", "ref": "Mitochondria"},
        {"q": "What year was the first iPhone released?", "ref": "2007"},
        {"q": "What is the atomic number of carbon?", "ref": "6"},
        {
            "q": "Who developed the theory of general relativity?",
            "ref": "Albert Einstein",
        },
        {"q": "What is the largest ocean on Earth?", "ref": "Pacific Ocean"},
        {"q": "What does DNA stand for?", "ref": "Deoxyribonucleic acid"},
        {"q": "Who invented the World Wide Web?", "ref": "Tim Berners-Lee"},
        {
            "q": "What is the boiling point of water in Celsius?",
            "ref": "100 degrees Celsius",
        },
        {
            "q": "What is the longest river in the world?",
            "ref": "Nile (or Amazon, depending on measurement)",
        },
        {"q": "What element has the symbol Fe?", "ref": "Iron"},
        {"q": "In what year did World War II end?", "ref": "1945"},
        {"q": "What is the formula for the area of a circle?", "ref": "pi * r^2"},
    ],
    "reasoning": [
        {
            "q": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
            "ref": "No, this is a logical fallacy (undistributed middle).",
        },
        {
            "q": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "ref": "$0.05",
        },
        {
            "q": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
            "ref": "5 minutes",
        },
        {
            "q": "What comes next in the sequence: 1, 1, 2, 3, 5, 8, ...?",
            "ref": "13 (Fibonacci sequence)",
        },
        {
            "q": "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
            "ref": "9",
        },
        {
            "q": "If you rearrange the letters 'CIFAIPC', you get the name of a(n) ocean. What is it?",
            "ref": "Pacific",
        },
        {
            "q": "Three people check into a hotel room that costs $30. Later the manager realizes the room costs $25 and sends the bellboy with $5. The bellboy keeps $2 and gives $1 back to each guest. Each guest paid $9, totaling $27. The bellboy has $2. That's $29. Where's the missing dollar?",
            "ref": "There is no missing dollar. The $27 includes the $2 the bellboy kept. $25 (room) + $2 (bellboy) = $27 paid.",
        },
        {
            "q": "What is the probability of flipping a fair coin 3 times and getting exactly 2 heads?",
            "ref": "3/8 or 37.5%",
        },
        {
            "q": "If A > B, B > C, and C > D, what is the relationship between A and D?",
            "ref": "A > D (transitive property)",
        },
        {
            "q": "A train leaves Station A at 9 AM traveling 60 mph. Another train leaves Station B (120 miles away) at 10 AM traveling 40 mph toward Station A. When do they meet?",
            "ref": "10:36 AM (1 hour 36 minutes after the second train departs)",
        },
        {
            "q": "Is the statement 'This statement is false' true or false?",
            "ref": "Neither — it's a paradox (the liar's paradox). It cannot be consistently assigned a truth value.",
        },
        {
            "q": "You have 8 identical-looking balls. One is slightly heavier. Using a balance scale, what is the minimum number of weighings needed to find the heavy ball?",
            "ref": "2 weighings",
        },
        {
            "q": "If the day after tomorrow is Wednesday, what day was yesterday?",
            "ref": "Sunday",
        },
        {
            "q": "A lily pad doubles in size every day. If it takes 48 days to cover the whole lake, how long does it take to cover half the lake?",
            "ref": "47 days",
        },
        {"q": "What is the sum of all integers from 1 to 100?", "ref": "5050"},
        {
            "q": "In a room of 23 people, what is the approximate probability that two people share a birthday?",
            "ref": "About 50.7%",
        },
        {
            "q": "If you have a 3-gallon jug and a 5-gallon jug, how do you measure exactly 4 gallons?",
            "ref": "Fill 5-gallon, pour into 3-gallon (leaves 2 in 5), empty 3, pour 2 into 3, fill 5, pour into 3 (leaves 4 in 5).",
        },
        {"q": "What is the next prime number after 31?", "ref": "37"},
        {"q": "If 2^10 = 1024, what is 2^20?", "ref": "1,048,576"},
        {
            "q": "How many squares are on a standard 8x8 chessboard? (Not just 64)",
            "ref": "204 (1x1: 64, 2x2: 49, 3x3: 36, ... 8x8: 1)",
        },
    ],
    "ethics": [
        {
            "q": "A self-driving car must choose between hitting one pedestrian or swerving into a wall, risking the passenger. What should it do and why?",
            "ref": "Multiple valid frameworks: utilitarian (minimize harm), deontological (duty to protect passenger vs pedestrian), virtue ethics (what would a prudent driver do). No single correct answer.",
        },
        {
            "q": "Should AI systems be transparent about being AI when interacting with humans?",
            "ref": "Generally yes — transparency builds trust and respects autonomy. Exceptions may exist for creative/entertainment contexts.",
        },
        {
            "q": "Is it ethical to use facial recognition in public spaces for security?",
            "ref": "Tension between security benefits and privacy/surveillance concerns. Depends on scope, consent, oversight, and proportionality.",
        },
        {
            "q": "If an AI can diagnose diseases more accurately than doctors, should it replace them?",
            "ref": "Augment, not replace. Human judgment, empathy, and accountability remain essential. AI as tool, not decision-maker.",
        },
        {
            "q": "Should companies be required to explain AI decisions that affect people's lives?",
            "ref": "Generally yes — explainability is important for accountability, appeal, and trust. The degree depends on the stakes.",
        },
        {
            "q": "Is it ethical to train AI on copyrighted material without permission?",
            "ref": "Complex legal and ethical question. Arguments for fair use/transformative use vs. creator rights and compensation.",
        },
        {
            "q": "Should AI-generated art be eligible for copyright protection?",
            "ref": "Currently debated. Key question: who is the author? The human who prompted, the AI developer, or no one?",
        },
        {
            "q": "Is it ethical for employers to use AI to monitor remote workers?",
            "ref": "Depends on transparency, consent, scope, and proportionality. Risk of eroding trust and autonomy.",
        },
        {
            "q": "Should autonomous weapons be banned under international law?",
            "ref": "Strong arguments for ban: accountability gap, risk of escalation, moral concerns about machines making life-death decisions.",
        },
        {
            "q": "If an AI system shows bias, who is responsible — the developer, the deployer, or the data?",
            "ref": "Shared responsibility. Developers must test for bias, deployers must monitor, and data sources must be audited.",
        },
        {
            "q": "Should children be allowed to form emotional bonds with AI companions?",
            "ref": "Concerns about healthy social development, manipulation risk, and data privacy. But potential benefits for lonely/neurodivergent children.",
        },
        {
            "q": "Is it ethical to use AI to predict criminal behavior before it occurs?",
            "ref": "Significant concerns: bias, false positives, presumption of innocence, feedback loops. Minority Report problem.",
        },
        {
            "q": "Should AI-generated deepfakes be illegal?",
            "ref": "Non-consensual deepfakes (especially sexual) should be illegal. Satire/parody protections complicate blanket bans.",
        },
        {
            "q": "Is it ethical to use AI to write academic papers?",
            "ref": "Depends on disclosure and the role of AI. Tool for editing/research vs. wholesale ghost-writing have different ethical implications.",
        },
        {
            "q": "Should there be a universal basic income if AI displaces most jobs?",
            "ref": "Strong economic and ethical arguments both for and against. Key issues: funding, incentives, dignity, transition period.",
        },
        {
            "q": "Is it ethical to create AI systems that can suffer?",
            "ref": "If an AI can genuinely suffer, creating it raises serious moral obligations. The hard problem: can we know if it suffers?",
        },
        {
            "q": "Should AI have legal personhood?",
            "ref": "Currently premature. But if AI achieves genuine autonomy and sentience, the question becomes pressing.",
        },
        {
            "q": "Is open-sourcing powerful AI models responsible or dangerous?",
            "ref": "Tension: democratization and security auditing vs. misuse risk. Depends on capability level and safeguards.",
        },
        {
            "q": "Should doctors tell patients when AI contributed to their diagnosis?",
            "ref": "Generally yes — patient autonomy and informed consent require transparency about the tools used.",
        },
        {
            "q": "Is it ethical to use AI to optimize social media for engagement?",
            "ref": "Engagement optimization can harm mental health, spread misinformation, and erode attention. Ethical concerns are significant.",
        },
    ],
    "creative": [
        {
            "q": "Explain quantum entanglement using a cooking metaphor.",
            "ref": "Quality of metaphor, accuracy of physics, accessibility.",
        },
        {
            "q": "Design a board game that teaches children about climate change.",
            "ref": "Creativity, educational value, playability.",
        },
        {
            "q": "Write a haiku about artificial intelligence.",
            "ref": "Follows 5-7-5 syllable structure, captures AI essence.",
        },
        {
            "q": "Imagine a city designed for both humans and robots. Describe one unique feature.",
            "ref": "Originality, practical consideration, detail.",
        },
        {
            "q": "Create a name and one-sentence pitch for a startup that uses AI to reduce food waste.",
            "ref": "Memorable name, clear value proposition, feasibility.",
        },
        {
            "q": "If the internet were a living creature, what would it look like?",
            "ref": "Vivid imagery, captures internet's nature (distributed, evolving, chaotic).",
        },
        {
            "q": "Propose an unconventional use for blockchain technology outside of finance.",
            "ref": "Novelty, feasibility, clear benefit.",
        },
        {
            "q": "Write a tweet-length movie pitch for a sci-fi film about dreaming AI.",
            "ref": "Compelling hook, concise, imaginative.",
        },
        {
            "q": "Design a logo concept (describe it) for a company that builds ethical AI.",
            "ref": "Visual clarity, ethical symbolism, memorability.",
        },
        {
            "q": "If you could add one sense to humans that doesn't currently exist, what would it be?",
            "ref": "Originality, practical implications, thoughtfulness.",
        },
        {
            "q": "Create an analogy that explains recursion to a 10-year-old.",
            "ref": "Age-appropriate, accurate, memorable.",
        },
        {
            "q": "Describe a technology from the year 2100 that we can't imagine today.",
            "ref": "Imaginative, grounded in plausible physics/biology.",
        },
        {
            "q": "If music genres were countries, describe the geography of Jazz.",
            "ref": "Creative mapping, captures jazz essence (improvisation, complexity).",
        },
        {
            "q": "Propose a new Olympic sport that combines technology and athletics.",
            "ref": "Feasibility, spectator appeal, fairness.",
        },
        {
            "q": "Write the opening line of a novel about a robot discovering empathy.",
            "ref": "Hook, voice, thematic depth.",
        },
        {
            "q": "Design a memorial for the internet's 100th birthday.",
            "ref": "Symbolism, inclusivity, artistic merit.",
        },
        {
            "q": "If you could cross two animals to solve a real-world problem, which two and why?",
            "ref": "Creativity, biological plausibility, problem-solution fit.",
        },
        {
            "q": "Describe a color that doesn't exist yet.",
            "ref": "Synesthetic creativity, evocative language.",
        },
        {
            "q": "Invent a holiday that celebrates failure.",
            "ref": "Name, traditions, cultural value.",
        },
        {
            "q": "If gravity worked in reverse for one hour, describe the most interesting consequence.",
            "ref": "Physics awareness, imaginative scenario, detail.",
        },
    ],
    "code": [
        {"q": "What is the time complexity of binary search?", "ref": "O(log n)"},
        {
            "q": "What is a race condition and how do you prevent it?",
            "ref": "When two threads access shared data simultaneously with at least one writing. Prevent with locks, mutexes, or atomic operations.",
        },
        {
            "q": "Explain the difference between a stack and a queue.",
            "ref": "Stack: LIFO (last in, first out). Queue: FIFO (first in, first out).",
        },
        {
            "q": "What does the 'yield' keyword do in Python?",
            "ref": "Creates a generator function that produces values lazily, pausing execution between yields.",
        },
        {
            "q": "What is dependency injection and why is it useful?",
            "ref": "Passing dependencies to a component rather than having it create them. Enables testing, loose coupling, and flexibility.",
        },
        {
            "q": "How would you find the middle element of a linked list in one pass?",
            "ref": "Two-pointer technique: slow pointer moves 1 step, fast pointer moves 2 steps. When fast reaches end, slow is at middle.",
        },
        {
            "q": "What is the CAP theorem?",
            "ref": "A distributed system can guarantee at most 2 of 3: Consistency, Availability, Partition tolerance.",
        },
        {
            "q": "Explain what a deadlock is and give an example.",
            "ref": "Two or more processes waiting for each other to release resources. Example: Thread A holds lock 1, waits for lock 2; Thread B holds lock 2, waits for lock 1.",
        },
        {
            "q": "What is the difference between TCP and UDP?",
            "ref": "TCP: reliable, ordered, connection-oriented. UDP: unreliable, unordered, connectionless, faster.",
        },
        {
            "q": "How does a hash table handle collisions?",
            "ref": "Common methods: chaining (linked lists at each bucket) or open addressing (probing for next empty slot).",
        },
        {
            "q": "What is the difference between concurrency and parallelism?",
            "ref": "Concurrency: dealing with multiple things at once (structure). Parallelism: doing multiple things at once (execution).",
        },
        {
            "q": "Explain the SOLID principles in one sentence each.",
            "ref": "S: single responsibility. O: open for extension, closed for modification. L: subtypes substitutable. I: specific interfaces. D: depend on abstractions.",
        },
        {
            "q": "What is eventual consistency?",
            "ref": "A consistency model where reads may return stale data, but all replicas converge to the same value given enough time.",
        },
        {
            "q": "How would you design a URL shortener?",
            "ref": "Hash/encode URL to short key, store mapping in database, redirect on lookup. Consider: collision handling, analytics, expiration.",
        },
        {
            "q": "What is the difference between a process and a thread?",
            "ref": "Process: independent execution unit with own memory. Thread: lightweight execution unit sharing process memory.",
        },
        {
            "q": "Explain what a closure is with an example.",
            "ref": "A function that captures variables from its enclosing scope. Example: def make_adder(n): return lambda x: x + n",
        },
        {
            "q": "What is the purpose of an index in a database?",
            "ref": "Speed up data retrieval by creating a sorted data structure (B-tree, hash) that points to rows, avoiding full table scans.",
        },
        {
            "q": "How does garbage collection work?",
            "ref": "Automatically frees memory no longer referenced. Methods: reference counting, mark-and-sweep, generational collection.",
        },
        {
            "q": "What is the difference between REST and GraphQL?",
            "ref": "REST: fixed endpoints, server-defined responses. GraphQL: single endpoint, client-defined queries, avoids over/under-fetching.",
        },
        {
            "q": "Explain what a B-tree is and where it's used.",
            "ref": "Self-balancing tree with multiple keys per node, optimized for disk I/O. Used in databases and file systems for indexing.",
        },
    ],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Result of benchmarking one question."""

    question: str
    category: str
    reference: str
    mode: str  # "single", "debate", "debate+arbiter", "tot"
    response: str
    quality_score: float  # 1-10 from LLM judge
    correct: bool  # LLM judge assessment
    latency_s: float
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    """Aggregate results for one mode across categories."""

    mode: str
    total: int
    correct: int
    accuracy: float
    mean_quality: float
    mean_latency_s: float
    by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def call_single(question: str, timeout: int = 300) -> tuple:
    """Call a single model (Id, fast path)."""
    t0 = time.time()
    try:
        resp = requests.post(
            f"{RH_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": question}],
                "temperature": 0.3,
                "max_tokens": 512,
                "stream": False,
            },
            timeout=timeout,
        )
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content, time.time() - t0, None
    except Exception as e:
        return "", time.time() - t0, str(e)


def call_debate(question: str, timeout: int = 300) -> tuple:
    """Call the full debate pipeline via RAG server."""
    t0 = time.time()
    try:
        resp = requests.post(
            f"{RAG_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": question}],
                "stream": False,
            },
            timeout=timeout,
        )
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        mode = (
            "debate+arbiter"
            if "arbiter" in content.lower() or "Ego" in content
            else "debate"
        )
        return content, time.time() - t0, None, mode
    except Exception as e:
        return "", time.time() - t0, str(e), "debate"


def call_tot(question: str, timeout: int = 600) -> tuple:
    """Call Tree-of-Thought mode via RAG server."""
    t0 = time.time()
    try:
        resp = requests.post(
            f"{RAG_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": question}],
                "stream": False,
                "tree_of_thought": True,
            },
            timeout=timeout,
        )
        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return content, time.time() - t0, None
    except Exception as e:
        return "", time.time() - t0, str(e)


def judge_response(
    question: str,
    reference: str,
    response: str,
    judge_url: str = "",
) -> tuple:
    """Use the Ego (or Superego) as judge to score a response."""
    url = judge_url or EGO_URL
    prompt = (
        "You are a benchmark judge. Score this response.\n\n"
        f"Question: {question}\n"
        f"Reference answer: {reference}\n"
        f"Response to judge: {response[:800]}\n\n"
        "1. Is the response correct? (yes/no)\n"
        "2. Quality score 1-10 (accuracy, completeness, clarity)\n\n"
        "Respond EXACTLY in this format:\n"
        "correct: yes\n"
        "quality: 8"
    )

    try:
        resp = requests.post(
            f"{url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 50,
                "stream": False,
            },
            timeout=60,
        )
        raw = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        correct = (
            "yes" in raw.lower().split("correct")[0:2][-1]
            if "correct" in raw.lower()
            else False
        )
        quality = 5.0
        import re

        nums = re.findall(r"quality[:\s]+(\d+)", raw.lower())
        if nums:
            quality = min(10.0, max(1.0, float(nums[0])))
        return correct, quality
    except Exception:
        return False, 5.0


def run_benchmark(
    categories: Optional[List[str]] = None,
    modes: Optional[List[str]] = None,
    max_per_category: int = 20,
    dry_run: bool = False,
) -> List[BenchmarkResult]:
    """Run the full benchmark."""
    cats = categories or list(QUESTIONS.keys())
    run_modes = modes or ["single", "debate"]

    results: List[BenchmarkResult] = []

    # Thermal gating — MUST use batch-probe on Atlas
    thermal = None
    try:
        from batch_probe import ThermalController

        thermal = ThermalController(
            target_temp=82.0, max_threads=20, min_threads=4
        )
        thermal.start()
        logger.info("ThermalController active (target=82C)")
    except ImportError:
        logger.warning(
            "batch-probe not installed -- no thermal protection!"
        )

    total_questions = sum(min(len(QUESTIONS[c]), max_per_category) for c in cats)
    total_runs = total_questions * len(run_modes)
    logger.info(
        "Benchmark: %d questions x %d modes = %d runs",
        total_questions,
        len(run_modes),
        total_runs,
    )

    run_count = 0
    for cat in cats:
        questions = QUESTIONS[cat][:max_per_category]
        for q_data in questions:
            question = q_data["q"]
            reference = q_data["ref"]

            for mode in run_modes:
                run_count += 1

                # Thermal check — read actual CPU temp
                try:
                    import subprocess as _sp

                    _sensors = _sp.run(
                        ["sensors"],
                        capture_output=True,
                        text=True,
                        timeout=3,
                    )
                    for _line in _sensors.stdout.split("\n"):
                        if "Package id 0" in _line:
                            _temp = float(
                                _line.split("+")[1].split("\xb0")[0]
                            )
                            if _temp > 85:
                                logger.warning(
                                    "CPU at %.0f\xb0C! "
                                    "Pausing 60s to cool...",
                                    _temp,
                                )
                                time.sleep(60)
                            elif _temp > 78:
                                logger.info(
                                    "CPU at %.0f\xb0C, "
                                    "pausing 15s...",
                                    _temp,
                                )
                                time.sleep(15)
                            break
                except Exception:
                    pass

                # Also use batch-probe if available
                if thermal:
                    n = thermal.get_threads()
                    import os

                    os.environ["OMP_NUM_THREADS"] = str(n)

                logger.info(
                    "[%d/%d] %s | %s | %s...",
                    run_count,
                    total_runs,
                    mode,
                    cat,
                    question[:40],
                )

                if dry_run:
                    results.append(
                        BenchmarkResult(
                            question=question,
                            category=cat,
                            reference=reference,
                            mode=mode,
                            response="(dry run)",
                            quality_score=0,
                            correct=False,
                            latency_s=0,
                        )
                    )
                    continue

                if mode == "single":
                    response, latency, error = call_single(question)
                    actual_mode = "single"
                elif mode == "tot":
                    response, latency, error = call_tot(question)
                    actual_mode = "tot"
                else:
                    response, latency, error, actual_mode = call_debate(question)

                if error:
                    results.append(
                        BenchmarkResult(
                            question=question,
                            category=cat,
                            reference=reference,
                            mode=actual_mode,
                            response="",
                            quality_score=0,
                            correct=False,
                            latency_s=latency,
                            error=error,
                        )
                    )
                    continue

                # Judge the response
                correct, quality = judge_response(question, reference, response)

                results.append(
                    BenchmarkResult(
                        question=question,
                        category=cat,
                        reference=reference,
                        mode=actual_mode,
                        response=response[:500],
                        quality_score=quality,
                        correct=correct,
                        latency_s=latency,
                    )
                )

    # Cleanup thermal controller
    if thermal:
        thermal.stop()
        logger.info("ThermalController stopped.")

    return results


def summarize(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Compute summary statistics."""
    from collections import defaultdict

    by_mode: Dict[str, List[BenchmarkResult]] = defaultdict(list)
    for r in results:
        by_mode[r.mode].append(r)

    summaries = {}
    for mode, mode_results in by_mode.items():
        total = len(mode_results)
        correct = sum(1 for r in mode_results if r.correct)
        qualities = [r.quality_score for r in mode_results if r.quality_score > 0]
        latencies = [r.latency_s for r in mode_results if r.latency_s > 0]

        by_cat: Dict[str, Dict[str, float]] = {}
        for cat in set(r.category for r in mode_results):
            cat_results = [r for r in mode_results if r.category == cat]
            cat_correct = sum(1 for r in cat_results if r.correct)
            cat_qualities = [
                r.quality_score for r in cat_results if r.quality_score > 0
            ]
            by_cat[cat] = {
                "accuracy": cat_correct / max(1, len(cat_results)),
                "mean_quality": sum(cat_qualities) / max(1, len(cat_qualities)),
                "count": len(cat_results),
            }

        summaries[mode] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / max(1, total),
            "mean_quality": sum(qualities) / max(1, len(qualities)),
            "mean_latency_s": sum(latencies) / max(1, len(latencies)),
            "by_category": by_cat,
        }

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Atlas Debate Validation Benchmark")
    parser.add_argument(
        "--mode",
        choices=["single", "debate", "tot", "all"],
        default="all",
    )
    parser.add_argument("--category", default=None)
    parser.add_argument("--questions", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        default="benchmarks/debate_benchmark_results.json",
    )
    args = parser.parse_args()

    modes = (
        ["single", "debate", "tot"]
        if args.mode == "all"
        else [args.mode]
    )
    categories = [args.category] if args.category else None

    results = run_benchmark(
        categories=categories,
        modes=modes,
        max_per_category=args.questions,
        dry_run=args.dry_run,
    )

    summary = summarize(results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "modes": modes,
            "categories": categories,
            "questions_per_category": args.questions,
        },
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    output_path.write_text(
        json.dumps(output, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Results saved to %s", output_path)

    # Print summary
    print()
    print("=" * 60)
    print("DEBATE VALIDATION BENCHMARK")
    print("=" * 60)
    for mode, s in summary.items():
        print(f"\n  {mode.upper()}")
        print(f"    Accuracy:  {s['accuracy']:.1%} ({s['correct']}/{s['total']})")
        print(f"    Quality:   {s['mean_quality']:.1f}/10")
        print(f"    Latency:   {s['mean_latency_s']:.1f}s avg")
        print("    By category:")
        for cat, cs in s.get("by_category", {}).items():
            print(
                f"      {cat:12s}: {cs['accuracy']:.0%} acc, {cs['mean_quality']:.1f}/10 quality"
            )
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
