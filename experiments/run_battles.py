#!/usr/bin/env python3
"""
sinc-llm Head-to-Head Battles: Compares sinc-prompt against 5 other techniques.

Tests sinc-prompt format against raw prompts, chain-of-thought, few-shot,
RISEN, and CO-STAR on 5 diverse tasks. Supports Anthropic, OpenAI, and Ollama.

Formula: x(t) = Sigma x(nT) * sinc((t - nT) / T)

Usage:
    python run_battles.py --provider ollama --model llama3
    python run_battles.py --provider anthropic --model claude-sonnet-4-20250514
    python run_battles.py --provider openai --model gpt-4o
    python run_battles.py --provider ollama --model llama3 --output results.json

Author: Mario Alexandre
License: MIT
DOI: 10.5281/zenodo.19152668
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for sinc_llm imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sinc_llm.core import compute_snr, build_sinc_json, estimate_tokens

# ---------------------------------------------------------------------------
# Provider abstraction (inline -- no external SDK required)
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, model: str, base_url: str = "http://localhost:11434") -> str:
    """Call Ollama HTTP API."""
    import urllib.request
    import urllib.error

    url = f"{base_url}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 2048},
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")
    except Exception as e:
        return f"[OLLAMA_ERROR] {e}"


def _call_anthropic(prompt: str, model: str, api_key: str) -> str:
    """Call Anthropic Messages API via urllib."""
    import urllib.request

    url = "https://api.anthropic.com/v1/messages"
    payload = json.dumps({
        "model": model,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    req = urllib.request.Request(url, data=payload, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            blocks = data.get("content", [])
            return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    except Exception as e:
        return f"[ANTHROPIC_ERROR] {e}"


def _call_openai(prompt: str, model: str, api_key: str) -> str:
    """Call OpenAI Chat Completions API via urllib."""
    import urllib.request

    url = "https://api.openai.com/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.3,
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    req = urllib.request.Request(url, data=payload, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""
    except Exception as e:
        return f"[OPENAI_ERROR] {e}"


def generate(prompt: str, provider: str, model: str) -> str:
    """Unified generation function."""
    if provider == "ollama":
        return _call_ollama(prompt, model)
    elif provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            return "[ERROR] ANTHROPIC_API_KEY not set"
        return _call_anthropic(prompt, model, key)
    elif provider == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            return "[ERROR] OPENAI_API_KEY not set"
        return _call_openai(prompt, model, key)
    else:
        return f"[ERROR] Unknown provider: {provider}"


# ---------------------------------------------------------------------------
# Quality metrics (same as ablation)
# ---------------------------------------------------------------------------

HEDGE_PHRASES: List[str] = [
    "i think", "probably", "perhaps", "might", "may ", "it seems",
    "it appears", "possibly", "could be", "likely", "unlikely",
    "in my opinion", "i believe", "generally", "typically",
    "it depends", "more or less", "sort of", "kind of",
    "to some extent", "arguably", "presumably", "supposedly",
]

SPECIFICITY_PATTERNS: List[str] = [
    r"\d+\.?\d*%",
    r"\$\d+",
    r"\d{4}",
    r"\d+\.\d+",
    r"(?:API|SDK|HTTP|REST|SQL|JSON|CSV|XML|HTML|CSS)",
    r"(?:function|class|method|variable|parameter|argument)",
    r"(?:step \d|phase \d|stage \d)",
]


def compute_hedging_density(text: str) -> float:
    """Hedging phrases per 100 words."""
    lower = text.lower()
    words = text.split()
    if not words:
        return 0.0
    count = sum(1 for phrase in HEDGE_PHRASES if phrase in lower)
    return (count / len(words)) * 100


def compute_specificity(text: str) -> float:
    """Specificity score (0-1)."""
    if not text:
        return 0.0
    matches = sum(1 for pat in SPECIFICITY_PATTERNS if re.search(pat, text))
    return min(1.0, matches / len(SPECIFICITY_PATTERNS))


def compute_quality_score(text: str) -> float:
    """Overall quality score (0-1): 40% specificity + 30% structure + 30% directness."""
    if not text or (text.startswith("[") and "ERROR" in text):
        return 0.0

    specificity = compute_specificity(text)

    structure_signals = [
        bool(re.search(r"^#{1,3}\s", text, re.MULTILINE)),
        bool(re.search(r"^\d+\.", text, re.MULTILINE)),
        bool(re.search(r"^[-*]\s", text, re.MULTILINE)),
        bool(re.search(r"\|.*\|.*\|", text)),
        len(text.split("\n\n")) >= 3,
    ]
    structure = sum(structure_signals) / len(structure_signals)

    hedging = compute_hedging_density(text)
    directness = max(0.0, 1.0 - hedging / 5.0)

    return round(0.4 * specificity + 0.3 * structure + 0.3 * directness, 4)


# ---------------------------------------------------------------------------
# Test tasks
# ---------------------------------------------------------------------------

BATTLE_TASKS: List[Dict[str, str]] = [
    {
        "id": "architecture",
        "raw": "Design a caching layer for a high-traffic e-commerce API",
        "persona": "You are a principal systems architect with 12 years of experience designing distributed caching systems at Amazon-scale. You have personally designed caching layers serving 500M+ requests/day.",
        "context": "An e-commerce platform serves 45M DAU with a product catalog of 8M SKUs. Current API response time is p95=800ms. The business requires p95<200ms. Peak traffic occurs during flash sales (15x normal). The existing stack is: Go microservices, PostgreSQL (3 read replicas), no caching layer.",
        "data": "Read:write ratio is 95:5. Hot product set: top 5,000 SKUs account for 72% of reads. Product data changes on average every 4.2 hours. Price data changes up to 200 times/day during sales. Session data: 12M concurrent sessions, average 340 bytes each. API endpoints: /products/{id} (45% traffic), /search (25%), /cart (15%), /recommendations (10%), /checkout (5%).",
        "constraints": "Justify every technology choice with a quantitative comparison to at least one alternative. State exact TTL values for every cache tier, not ranges. Never recommend Redis without comparing to Memcached and stating why Redis wins for this use case. Address cache invalidation explicitly for every data type. State the exact memory requirement calculation for each cache tier. Never design a cache that can serve stale prices during a flash sale. Every component must have a stated failure mode and recovery procedure. Address thundering herd, cache stampede, and hot key problems with specific solutions. State the total infrastructure cost estimate with line items.",
        "format": "Start with an architecture diagram (ASCII). Then a cache tier table (Tier, Technology, Data, TTL, Memory, Cost/month). Then detailed design for each tier. Cache invalidation strategy section. Failure modes table. Performance projection (before/after for each endpoint).",
    },
    {
        "id": "ml_pipeline",
        "raw": "Build a real-time fraud detection system for payment processing",
        "persona": "You are a machine learning engineer specializing in real-time fraud detection with 10 years of experience at PayPal and Stripe. You have built systems that process $100B+ in annual transaction volume with <0.01% false positive rates.",
        "context": "A payment processor handles 3,200 transactions per second, $45B annual volume. Current rule-based system catches 67% of fraud (industry: 85-92%). False positive rate is 2.1% (target: <0.5%). Average fraud amount: $847. Total fraud losses: $12M/year. Chargebacks cost an additional $4.2M/year.",
        "data": "Features available: transaction_amount, merchant_category, card_present/not_present, velocity_1h/24h/7d, geo_distance_from_last_txn, device_fingerprint_match, billing_shipping_match, time_since_last_txn, account_age_days, historical_chargeback_rate. Training data: 18 months, 1.2B transactions, 847K confirmed fraud cases (0.07% base rate). Latency budget: model inference must complete in <50ms p99.",
        "constraints": "Every model recommendation must include exact hyperparameters, not just algorithm names. State precision, recall, and F1 at the exact decision threshold you recommend. Never recommend a model without comparing it to at least 2 alternatives on identical metrics. Address the class imbalance problem with at least 3 techniques and state why you chose one. The system must produce an interpretable fraud score, not just a binary decision. Address concept drift explicitly with a retraining strategy and drift detection method. State the exact dollar impact of each 0.1% improvement in detection rate. Never recommend a feature without stating its information gain or SHAP importance.",
        "format": "Start with executive summary (expected fraud reduction in dollars). Architecture diagram (ASCII). Model comparison table (Model, Precision, Recall, F1, Latency, Training Time). Feature importance ranking. Real-time serving design. Monitoring and retraining pipeline. Cost-benefit analysis table.",
    },
    {
        "id": "database",
        "raw": "Migrate a 2TB PostgreSQL database to a sharded architecture with zero downtime",
        "persona": "You are a database architect with 15 years of experience in large-scale PostgreSQL deployments. You have performed zero-downtime migrations for databases serving 50,000+ queries/second at companies including Uber, Instagram, and Notion.",
        "context": "A SaaS application stores multi-tenant data in a single PostgreSQL 15 instance. The database has grown to 2TB with 450 tables. Write throughput is approaching the single-node limit. The largest table (events) has 3.2B rows and grows by 8M rows/day. Query latency has degraded from p95=45ms to p95=280ms over the past 6 months.",
        "data": "Current setup: RDS db.r6g.4xlarge (16 vCPU, 128GB RAM), 2TB gp3 storage (12,000 IOPS). QPS: 32,000 read, 4,500 write. Top tables by size: events (1.1TB, 3.2B rows), audit_logs (340GB, 890M rows), user_activities (180GB, 420M rows). Tenant distribution: 12,000 tenants, top 50 tenants account for 65% of data. Connection count: 800 (pool of 200 per app server x 4). Vacuum takes 4.2 hours on the events table.",
        "constraints": "Zero downtime means zero -- not 'near-zero', not 'minimal'. State the exact shard key selection criteria and why alternatives were rejected. Never recommend application-level sharding without comparing to Citus, Vitess, and native partitioning. Every migration step must have an explicit rollback procedure with expected rollback time. State exact replication lag expectations during each phase. Never assume the application can be modified without stating exactly which queries need rewriting. Address cross-shard queries explicitly with performance implications. State the exact timeline for each phase in days. Provide the exact pg_dump/pg_restore commands or logical replication setup.",
        "format": "Start with migration strategy summary (1 paragraph). Shard key analysis table (Candidate, Distribution, Cross-shard %, Verdict). Phase table (Phase, Duration, Risk, Downtime, Rollback Time). Detailed steps for each phase with exact commands. Monitoring checklist. Post-migration validation queries.",
    },
    {
        "id": "incident_response",
        "raw": "Write an incident response runbook for a production database outage",
        "persona": "You are a site reliability engineering lead with 10 years of experience managing production incidents at scale. You have led incident response for 200+ P1 incidents at companies serving 100M+ users and hold the GIAC Certified Incident Handler (GCIH) certification.",
        "context": "An e-commerce platform with $340M annual revenue experiences a complete database outage during peak hours (Black Friday). The primary PostgreSQL instance becomes unresponsive. Application error rate jumps to 100%. 45,000 concurrent users are affected. The on-call engineer has been paged and needs a step-by-step runbook.",
        "data": "Infrastructure: Primary PostgreSQL on RDS (db.r6g.8xlarge), 2 read replicas, pgBouncer (4 instances), application (24 pods on EKS). RPO: 1 minute (continuous WAL archiving to S3). RTO: 15 minutes. Last backup: 6 hours ago (automated daily snapshot). Monitoring: Datadog (APM, infrastructure), PagerDuty (alerting), Statuspage (customer communication). Escalation path: On-call SRE -> SRE Lead -> VP Engineering -> CTO. Revenue impact: $38,000/minute during peak.",
        "constraints": "Every step must have an exact command or action, not a vague instruction. State the expected outcome of each step and what to do if the outcome differs. Never skip the communication steps -- customers, stakeholders, and team coordination are as important as technical recovery. Time-box every diagnostic step: if a step takes longer than its time box, escalate immediately. State the exact monitoring queries to verify recovery. Address data integrity verification explicitly. Never assume the on-call engineer has deep database expertise -- write for a competent generalist. Every decision point must have a clear if/then branch.",
        "format": "Start with a severity assessment checklist (30 seconds). Then triage flowchart (text-based decision tree). Then: Phase 1 Immediate (0-5 min), Phase 2 Diagnosis (5-15 min), Phase 3 Recovery (15-45 min), Phase 4 Verification (45-60 min), Phase 5 Post-incident. Each phase has numbered steps with exact commands. Communication templates for each phase. Post-incident review template.",
    },
    {
        "id": "testing_strategy",
        "raw": "Design a comprehensive testing strategy for a microservices payment system",
        "persona": "You are a quality engineering architect with 12 years of experience in testing distributed systems. You designed the testing infrastructure for payment systems processing $50B+ annual volume at Stripe and Square.",
        "context": "A payment processing system consists of 14 microservices handling authorization, capture, settlement, refunds, disputes, and reporting. The system processes 2,800 transactions/second with 99.99% uptime SLA. Current test coverage is 62% (target: 90%). Release cycle: weekly deployments. Bug escape rate: 3.2 bugs/release reaching production (target: <0.5).",
        "data": "Services: API Gateway, Auth Service, Payment Router, Card Processor (Visa/MC/Amex), Bank Connector, Fraud Engine, Ledger Service, Settlement Engine, Refund Service, Dispute Manager, Notification Service, Webhook Delivery, Reporting Service, Admin Dashboard. Inter-service communication: gRPC (sync) and Kafka (async). External integrations: 4 card networks, 12 bank partners, 3 fraud vendors. Test infrastructure: GitHub Actions (CI), Kubernetes (staging), Testcontainers, WireMock. Current test counts: 4,200 unit, 340 integration, 45 E2E, 0 contract, 0 chaos.",
        "constraints": "State the exact test count target for each test type and each service. Never recommend a testing tool without stating why it was chosen over the top alternative. Every test type must have a stated execution time budget and CI pipeline stage. Address data consistency testing across services explicitly. State the exact metrics that determine release readiness. Never recommend 'more tests' without stating which specific scenarios are untested. Contract testing must cover every service boundary. Chaos testing must target the 5 highest-risk failure modes. State the team effort in engineer-weeks for each implementation phase.",
        "format": "Start with testing pyramid diagram (ASCII). Test strategy matrix (Service x Test Type, with counts). CI pipeline stages table (Stage, Tests, Duration, Blocking?). Per-test-type section with: tool, coverage targets, example test, execution time. Chaos testing scenarios table. Implementation roadmap (phases, effort, milestones). Success metrics dashboard specification.",
    },
]


# ---------------------------------------------------------------------------
# Prompt technique builders
# ---------------------------------------------------------------------------

def build_raw_prompt(task: Dict[str, str]) -> str:
    """Technique 1: Raw unstructured prompt."""
    return task["raw"]


def build_chain_of_thought(task: Dict[str, str]) -> str:
    """Technique 2: Chain-of-thought prompting."""
    return (
        f"{task['raw']}\n\n"
        "Let's think step by step. First, consider the key requirements. "
        "Then, analyze the constraints and trade-offs. "
        "Finally, provide a detailed, well-structured solution."
    )


def build_few_shot(task: Dict[str, str]) -> str:
    """Technique 3: Few-shot with examples."""
    return (
        "Here are examples of high-quality technical responses:\n\n"
        "Example 1:\n"
        "Q: How should I implement rate limiting?\n"
        "A: Use a token bucket algorithm with Redis. Configure: 100 requests/minute per API key, "
        "burst of 20. Implementation: MULTI/EXEC with TTL-based token replenishment. "
        "Cost: 2 Redis commands per request (~0.1ms). Monitor: track 429 response rate, "
        "alert at >5% rejection rate.\n\n"
        "Example 2:\n"
        "Q: What database should I use for time-series data?\n"
        "A: TimescaleDB (PostgreSQL extension). Benchmarks: 3.2x faster than vanilla PG for "
        "time-range queries on 1B+ rows. Compression: 95% space reduction. Supports continuous "
        "aggregates for real-time dashboards. Alternative considered: InfluxDB -- rejected due to "
        "limited JOIN support and higher operational complexity.\n\n"
        f"Now answer this question with the same level of specificity:\n{task['raw']}"
    )


def build_risen(task: Dict[str, str]) -> str:
    """Technique 4: RISEN framework (Role, Instructions, Steps, End goal, Narrowing)."""
    return (
        f"**Role:** {task['persona']}\n\n"
        f"**Instructions:** {task['raw']}. Provide a comprehensive, actionable solution "
        f"based on the following context: {task['context']}\n\n"
        f"**Steps:**\n"
        "1. Analyze the current state and identify key challenges\n"
        "2. Evaluate possible solutions with trade-offs\n"
        "3. Recommend the best approach with justification\n"
        "4. Provide implementation details\n"
        "5. Address risks and mitigations\n\n"
        "**End goal:** A production-ready solution design with specific implementation guidance.\n\n"
        "**Narrowing:** Focus on practical, battle-tested approaches. Use exact numbers for "
        "all claims. Avoid vague recommendations."
    )


def build_costar(task: Dict[str, str]) -> str:
    """Technique 5: CO-STAR framework (Context, Objective, Style, Tone, Audience, Response)."""
    return (
        f"**Context:** {task['context']}\n\n"
        f"**Objective:** {task['raw']}\n\n"
        "**Style:** Technical and precise, like a senior engineer's design document.\n\n"
        "**Tone:** Authoritative and direct. No hedging.\n\n"
        "**Audience:** Senior software engineers who will implement this solution.\n\n"
        "**Response format:** Structured with headers, tables, and code examples. "
        "Lead with the recommendation, then provide supporting analysis."
    )


def build_sinc_prompt(task: Dict[str, str]) -> Tuple[str, Dict[str, Any]]:
    """Technique 6: sinc-prompt (6-band Nyquist-Shannon decomposition)."""
    sinc_json = build_sinc_json(
        persona=task["persona"],
        context=task["context"],
        data=task["data"],
        constraints=task["constraints"],
        fmt=task["format"],
        task=task["raw"],
    )

    parts = [
        f"[PERSONA]\n{task['persona']}",
        f"[CONTEXT]\n{task['context']}",
        f"[DATA]\n{task['data']}",
        f"[CONSTRAINTS]\n{task['constraints']}",
        f"[FORMAT]\n{task['format']}",
        f"[TASK]\n{task['raw']}",
    ]
    return "\n\n".join(parts), sinc_json


TECHNIQUES = [
    ("raw", "Raw Prompt", build_raw_prompt),
    ("cot", "Chain-of-Thought", build_chain_of_thought),
    ("few_shot", "Few-Shot", build_few_shot),
    ("risen", "RISEN", build_risen),
    ("costar", "CO-STAR", build_costar),
]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_battles(provider: str, model: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """Run head-to-head battles."""
    print("=" * 78)
    print("SINC-LLM HEAD-TO-HEAD BATTLES")
    print(f"Provider: {provider} | Model: {model}")
    print(f"Tasks: {len(BATTLE_TASKS)} | Techniques: 6 (5 baselines + sinc-prompt)")
    print(f"Total trials: {len(BATTLE_TASKS) * 6}")
    print("=" * 78)
    print()

    results: List[Dict[str, Any]] = []
    total_trials = len(BATTLE_TASKS) * 6
    trial_num = 0

    for task in BATTLE_TASKS:
        task_id = task["id"]
        print(f"--- Task: {task_id} ---")

        # Run all baseline techniques
        for tech_id, tech_name, builder in TECHNIQUES:
            trial_num += 1
            prompt = builder(task)
            token_count = estimate_tokens(prompt)

            print(f"  [{trial_num}/{total_trials}] {tech_name:>16} ({token_count} tokens) ...", end=" ", flush=True)

            t0 = time.time()
            response = generate(prompt, provider, model)
            elapsed = time.time() - t0

            quality = compute_quality_score(response)
            hedging = compute_hedging_density(response)
            specificity = compute_specificity(response)
            is_error = response.startswith("[") and "ERROR" in response

            result = {
                "task_id": task_id,
                "technique": tech_id,
                "technique_name": tech_name,
                "quality_score": quality,
                "hedging_density": round(hedging, 4),
                "specificity": round(specificity, 4),
                "word_count": len(response.split()),
                "prompt_tokens": token_count,
                "response_tokens": estimate_tokens(response),
                "elapsed_s": round(elapsed, 2),
                "error": is_error,
            }
            results.append(result)

            status = "ERR" if is_error else f"Q={quality:.2f}"
            print(f"{status} | hedge={hedging:.2f} | spec={specificity:.2f} | {elapsed:.1f}s")

        # Run sinc-prompt
        trial_num += 1
        sinc_text, sinc_json = build_sinc_prompt(task)
        sinc_token_count = estimate_tokens(sinc_text)
        snr_data = compute_snr(sinc_json)

        print(f"  [{trial_num}/{total_trials}] {'sinc-prompt':>16} ({sinc_token_count} tokens, SNR={snr_data['snr']}) ...", end=" ", flush=True)

        t0 = time.time()
        response = generate(sinc_text, provider, model)
        elapsed = time.time() - t0

        quality = compute_quality_score(response)
        hedging = compute_hedging_density(response)
        specificity = compute_specificity(response)
        is_error = response.startswith("[") and "ERROR" in response

        result = {
            "task_id": task_id,
            "technique": "sinc",
            "technique_name": "sinc-prompt",
            "quality_score": quality,
            "hedging_density": round(hedging, 4),
            "specificity": round(specificity, 4),
            "word_count": len(response.split()),
            "prompt_tokens": sinc_token_count,
            "response_tokens": estimate_tokens(response),
            "elapsed_s": round(elapsed, 2),
            "snr": snr_data["snr"],
            "snr_grade": snr_data["grade"],
            "error": is_error,
        }
        results.append(result)

        status = "ERR" if is_error else f"Q={quality:.2f}"
        print(f"{status} | hedge={hedging:.2f} | spec={specificity:.2f} | {elapsed:.1f}s")

        print()

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("=" * 78)
    print("BATTLE RESULTS SUMMARY")
    print("=" * 78)
    print()

    # Aggregate by technique
    tech_stats: Dict[str, Dict[str, List[float]]] = {}
    for r in results:
        if r["error"]:
            continue
        tech = r["technique_name"]
        if tech not in tech_stats:
            tech_stats[tech] = {"quality": [], "hedging": [], "specificity": [], "tokens": []}
        tech_stats[tech]["quality"].append(r["quality_score"])
        tech_stats[tech]["hedging"].append(r["hedging_density"])
        tech_stats[tech]["specificity"].append(r["specificity"])
        tech_stats[tech]["tokens"].append(r["prompt_tokens"])

    print(f"{'Technique':>16} | {'Avg Quality':>11} | {'Avg Hedging':>11} | {'Avg Specificity':>15} | {'Avg Tokens':>10}")
    print("-" * 78)

    for tech_name in ["Raw Prompt", "Chain-of-Thought", "Few-Shot", "RISEN", "CO-STAR", "sinc-prompt"]:
        if tech_name not in tech_stats:
            continue
        stats = tech_stats[tech_name]
        avg_q = sum(stats["quality"]) / len(stats["quality"])
        avg_h = sum(stats["hedging"]) / len(stats["hedging"])
        avg_s = sum(stats["specificity"]) / len(stats["specificity"])
        avg_t = sum(stats["tokens"]) / len(stats["tokens"])
        marker = " <--" if tech_name == "sinc-prompt" else ""
        print(f"{tech_name:>16} | {avg_q:>11.4f} | {avg_h:>11.4f} | {avg_s:>15.4f} | {avg_t:>10.0f}{marker}")

    # Win/loss counts
    print()
    print("HEAD-TO-HEAD: sinc-prompt vs each technique")
    print("-" * 78)

    sinc_results = {r["task_id"]: r for r in results if r["technique"] == "sinc" and not r["error"]}

    for tech_id, tech_name, _ in TECHNIQUES:
        wins = 0
        losses = 0
        ties = 0
        for r in results:
            if r["technique"] != tech_id or r["error"]:
                continue
            sinc_r = sinc_results.get(r["task_id"])
            if not sinc_r:
                continue
            if sinc_r["quality_score"] > r["quality_score"] + 0.01:
                wins += 1
            elif r["quality_score"] > sinc_r["quality_score"] + 0.01:
                losses += 1
            else:
                ties += 1

        total = wins + losses + ties
        win_rate = (wins / total * 100) if total > 0 else 0
        print(f"  vs {tech_name:>16}: sinc wins {wins}, loses {losses}, ties {ties}  ({win_rate:.0f}% win rate)")

    # Overall stats
    if "sinc-prompt" in tech_stats and "Raw Prompt" in tech_stats:
        sinc_avg = sum(tech_stats["sinc-prompt"]["quality"]) / len(tech_stats["sinc-prompt"]["quality"])
        raw_avg = sum(tech_stats["Raw Prompt"]["quality"]) / len(tech_stats["Raw Prompt"]["quality"])
        improvement = ((sinc_avg - raw_avg) / max(raw_avg, 0.001)) * 100
        print(f"\nsinc-prompt quality improvement over raw: {improvement:.1f}%")

    if "sinc-prompt" in tech_stats:
        sinc_hedge = sum(tech_stats["sinc-prompt"]["hedging"]) / len(tech_stats["sinc-prompt"]["hedging"])
        other_hedges = []
        for tn, ts in tech_stats.items():
            if tn != "sinc-prompt":
                other_hedges.extend(ts["hedging"])
        if other_hedges:
            avg_other_hedge = sum(other_hedges) / len(other_hedges)
            hedge_reduction = ((avg_other_hedge - sinc_hedge) / max(avg_other_hedge, 0.001)) * 100
            print(f"sinc-prompt hedging reduction vs average: {hedge_reduction:.1f}%")

    # Save results
    output = {
        "experiment": "battles",
        "provider": provider,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_trials": len(results),
        "results": results,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_battles",
        description=(
            "sinc-llm Head-to-Head Battles: Compare sinc-prompt against 5 other techniques.\n"
            "Tests raw, chain-of-thought, few-shot, RISEN, CO-STAR, and sinc-prompt."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_battles.py --provider ollama --model llama3\n"
            "  python run_battles.py --provider anthropic --model claude-sonnet-4-20250514\n"
            "  python run_battles.py --provider openai --model gpt-4o\n"
            "\n"
            "DOI: 10.5281/zenodo.19152668"
        ),
    )
    parser.add_argument(
        "--provider", required=True, choices=["anthropic", "openai", "ollama"],
        help="LLM provider (ollama requires no API key)"
    )
    parser.add_argument(
        "--model", required=True,
        help="Model name (e.g., llama3, claude-sonnet-4-20250514, gpt-4o)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save results as JSON"
    )

    args = parser.parse_args()
    run_battles(args.provider, args.model, args.output)


if __name__ == "__main__":
    main()
