#!/usr/bin/env python3
"""
sinc-llm Ablation Study: Tests prompt quality at 1-band through 6-band levels.

Measures how response quality degrades as frequency bands are removed from
sinc-formatted prompts. Supports Anthropic, OpenAI, and Ollama providers.

Formula: x(t) = Sigma x(nT) * sinc((t - nT) / T)

Usage:
    python run_ablation.py --provider ollama --model llama3
    python run_ablation.py --provider anthropic --model claude-haiku-4-20250514
    python run_ablation.py --provider openai --model gpt-4o-mini
    python run_ablation.py --provider ollama --model llama3 --output results.json

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

from sinc_llm.core import (
    compute_snr,
    build_sinc_json,
    estimate_tokens,
    grade_snr,
    BETA0,
    AMPLITUDE,
    CEILING,
)

# ---------------------------------------------------------------------------
# Provider abstraction (inline -- no external SDK required)
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, model: str, base_url: str = "http://localhost:11434") -> str:
    """Call Ollama HTTP API. Zero external dependencies."""
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
    except urllib.error.URLError as e:
        return f"[OLLAMA_ERROR] {e}"
    except Exception as e:
        return f"[OLLAMA_ERROR] {e}"


def _call_anthropic(prompt: str, model: str, api_key: str) -> str:
    """Call Anthropic Messages API via urllib. No SDK required."""
    import urllib.request
    import urllib.error

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
    except urllib.error.URLError as e:
        return f"[ANTHROPIC_ERROR] {e}"
    except Exception as e:
        return f"[ANTHROPIC_ERROR] {e}"


def _call_openai(prompt: str, model: str, api_key: str) -> str:
    """Call OpenAI Chat Completions API via urllib. No SDK required."""
    import urllib.request
    import urllib.error

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
    except urllib.error.URLError as e:
        return f"[OPENAI_ERROR] {e}"
    except Exception as e:
        return f"[OPENAI_ERROR] {e}"


def generate(prompt: str, provider: str, model: str) -> str:
    """Unified generation function across providers."""
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
# Quality metrics
# ---------------------------------------------------------------------------

HEDGE_PHRASES: List[str] = [
    "i think", "probably", "perhaps", "might", "may ", "it seems",
    "it appears", "possibly", "could be", "likely", "unlikely",
    "in my opinion", "i believe", "generally", "typically",
    "it depends", "more or less", "sort of", "kind of",
    "to some extent", "arguably", "presumably", "supposedly",
]

SPECIFICITY_PATTERNS: List[str] = [
    r"\d+\.?\d*%",           # percentages
    r"\$\d+",                # dollar amounts
    r"\d{4}",                # years/4-digit numbers
    r"\d+\.\d+",             # decimal numbers
    r"(?:API|SDK|HTTP|REST|SQL|JSON|CSV|XML|HTML|CSS)",  # tech terms
    r"(?:function|class|method|variable|parameter|argument)",  # code terms
    r"(?:step \d|phase \d|stage \d)",  # numbered steps
]


def compute_hedging_density(text: str) -> float:
    """Compute hedging phrases per 100 words."""
    lower = text.lower()
    words = text.split()
    if len(words) == 0:
        return 0.0
    count = sum(1 for phrase in HEDGE_PHRASES if phrase in lower)
    return (count / len(words)) * 100


def compute_specificity(text: str) -> float:
    """Compute specificity score (0-1) based on concrete content markers."""
    if not text:
        return 0.0
    matches = sum(1 for pat in SPECIFICITY_PATTERNS if re.search(pat, text))
    # Normalize: 7 patterns max, score = matches/7
    return min(1.0, matches / len(SPECIFICITY_PATTERNS))


def compute_quality_score(text: str) -> float:
    """Compute overall quality score (0-1) from response text.

    Combines:
    - Specificity (40%): concrete numbers, technical terms, named entities
    - Structure (30%): headers, lists, organized sections
    - Directness (30%): inverse of hedging density
    """
    if not text or text.startswith("[") and "ERROR" in text:
        return 0.0

    # Specificity component (0-1)
    specificity = compute_specificity(text)

    # Structure component (0-1)
    structure_signals = [
        bool(re.search(r"^#{1,3}\s", text, re.MULTILINE)),  # markdown headers
        bool(re.search(r"^\d+\.", text, re.MULTILINE)),      # numbered lists
        bool(re.search(r"^[-*]\s", text, re.MULTILINE)),     # bullet lists
        bool(re.search(r"\|.*\|.*\|", text)),                # tables
        len(text.split("\n\n")) >= 3,                        # multiple paragraphs
    ]
    structure = sum(structure_signals) / len(structure_signals)

    # Directness component (0-1): inverse of hedging
    hedging = compute_hedging_density(text)
    directness = max(0.0, 1.0 - hedging / 5.0)  # 5+ hedges/100w = 0 directness

    return round(0.4 * specificity + 0.3 * structure + 0.3 * directness, 4)


# ---------------------------------------------------------------------------
# Test tasks
# ---------------------------------------------------------------------------

ABLATION_TASKS: List[Dict[str, str]] = [
    {
        "id": "code_review",
        "task": "Review this Python function for bugs and performance issues",
        "persona": "You are a senior Python engineer with 15 years of experience in code review, performance optimization, and security auditing.",
        "context": "The function processes user-uploaded CSV files in a Django web application serving 50,000 daily active users. Average file size is 2MB with 10,000-50,000 rows. The application runs on AWS EC2 t3.large instances behind an ALB.",
        "data": "Current function: def process_csv(file): reader = csv.reader(open(file.name)); data = list(reader); results = []; for row in data: results.append(transform(row)); return results. Average response time: 4.2 seconds. Memory usage spikes to 1.8GB during processing. Error rate: 3.2% on files with unicode characters.",
        "constraints": "State facts directly -- never hedge with 'I think' or 'probably'. Identify every bug, not just the obvious ones. Rank issues by severity (CRITICAL/HIGH/MEDIUM/LOW). For each issue, provide the exact fix as a code diff. Never suggest refactoring that changes the public API. Address memory, security, and correctness independently. Always explain WHY a pattern is problematic, not just WHAT to change. Never recommend a library without stating its exact version and license.",
        "format": "Lead with a severity summary table (columns: Issue, Severity, Line, Fix). Then provide each fix as a before/after code block. End with a performance impact estimate (expected speedup and memory reduction in concrete numbers).",
    },
    {
        "id": "market_analysis",
        "task": "Analyze the competitive landscape for AI-powered code review tools in 2025",
        "persona": "You are a senior technology market analyst at Gartner with expertise in developer tools, AI/ML markets, and B2B SaaS valuation.",
        "context": "The AI code review market reached $1.2B in 2024 and is projected to grow 34% CAGR through 2028. Major players include GitHub Copilot (Microsoft), CodeRabbit, Sourcery, and Snyk Code. The market shifted from rule-based static analysis to LLM-powered semantic review in 2023-2024.",
        "data": "GitHub Copilot: $10/month individual, $19/month business, estimated 1.8M paying subscribers. CodeRabbit: $15/month, est. 45K users. Sourcery: freemium, est. 120K free users, 8K paid. Snyk Code: enterprise pricing, est. $50K ACV, 2,400 enterprise accounts. Total addressable market: 27M professional developers worldwide.",
        "constraints": "Use exact revenue figures, user counts, and growth rates for every claim. Never use vague quantities like 'several' or 'many'. Compare products on identical dimensions. State competitive advantages as measurable differentiators, not subjective opinions. Never predict market outcomes without stating the confidence interval and the model used. Always cite the data source for each number. Distinguish between disclosed financials and analyst estimates.",
        "format": "Open with a market size table. Then a competitive matrix (rows: products, columns: pricing, users, features, strengths, weaknesses). Follow with a threats/opportunities section using numbered lists. Close with a 3-year forecast table with confidence intervals.",
    },
    {
        "id": "creative_writing",
        "task": "Write the opening scene of a science fiction story set on a generation ship 200 years into its journey",
        "persona": "You are a published science fiction author whose work combines hard science with literary prose. Your style draws from Ursula K. Le Guin's sociological depth and Kim Stanley Robinson's technical precision.",
        "context": "The generation ship Meridian launched from Earth orbit in 2186 carrying 12,000 colonists toward Proxima Centauri b. It is now year 200 of a 280-year journey. The ship has developed a rigid caste system based on original crew roles. A mysterious signal was detected 3 months ago from within the ship's dead zone -- a sealed section damaged in a micrometeorite impact 140 years ago.",
        "data": "Ship dimensions: 4.2km long, 800m diameter. Population: 8,400 (down from 12,000 due to a plague in year 87). Rotation provides 0.8g. Power: deuterium fusion. Speed: 0.12c. The dead zone is a 200m section of Deck 7 through Deck 12, sealed since year 60. Oxygen recycling operates at 94.3% efficiency (design spec: 99.1%). Food production covers 96% of caloric needs.",
        "constraints": "Show, never tell -- convey worldbuilding through character action and sensory detail, not exposition dumps. Every technical detail must be physically plausible. The opening scene must introduce exactly one character in a specific moment of change. Use present tense. Keep sentences under 25 words on average. No dialogue tags other than 'said'. The first sentence must be under 10 words. Never use the word 'suddenly'. Ground every abstract concept in a concrete sensory image.",
        "format": "Prose only. 500-700 words. No chapter heading. No author notes. Single scene, single POV, single timeline. Three section breaks maximum (marked with a blank line).",
    },
    {
        "id": "system_design",
        "task": "Design a real-time notification system for a social media platform with 50M DAU",
        "persona": "You are a principal systems architect at a FAANG-scale company with 12 years of experience designing distributed systems that serve billions of requests per day.",
        "context": "The social media platform has 50 million daily active users, 200 million monthly active users, and processes 15,000 notifications per second at peak. Current system uses polling (30-second intervals) which causes 40% of notifications to arrive 15-60 seconds late. The company wants sub-second delivery for 99th percentile.",
        "data": "Current infrastructure: AWS us-east-1 and us-west-2, PostgreSQL (8 read replicas), Redis cluster (128 nodes), Kafka (24 brokers). Current notification types: likes (45%), comments (25%), follows (15%), DMs (10%), system (5%). Average payload: 340 bytes. Peak traffic: weekdays 6-9 PM EST. Mobile: 72% of traffic (iOS 45%, Android 55%). Budget for new infrastructure: $2.4M/year.",
        "constraints": "Design for exactly the stated scale -- do not over-engineer for 10x growth without being asked. Every component must have a stated latency SLA. Justify every technology choice with a quantitative comparison to at least one alternative. Never recommend a technology without stating its operational complexity (low/medium/high) and team expertise requirements. Address failure modes explicitly: what happens when each component fails? State the exact cost estimate for each infrastructure component. Never use 'microservices' without specifying exactly which services and their boundaries.",
        "format": "Start with a system diagram in ASCII art. Then a components table (Component, Technology, Purpose, Latency SLA, Cost/month). Then detailed design for each component. End with a failure modes table and a capacity planning section with exact numbers.",
    },
    {
        "id": "data_science",
        "task": "Build a churn prediction model for a SaaS product and explain the methodology",
        "persona": "You are a senior data scientist with 10 years of experience in predictive modeling, specializing in customer lifecycle analytics for B2B SaaS companies.",
        "context": "The SaaS product is a project management tool with 15,000 active accounts. Monthly churn rate is 4.2% (industry average: 5.8%). The company has 18 months of historical data including usage logs, support tickets, billing events, and NPS survey responses.",
        "data": "Features available: daily_active_users (per account), feature_adoption_score (0-100), support_tickets_last_30d, nps_score (-100 to 100), contract_value_monthly ($50-$5000), account_age_months, seats_purchased, seats_active, integrations_enabled (0-15), last_login_days_ago, training_sessions_attended. Historical churn events: 2,847 churned accounts out of 22,400 total (12.7% base rate over 18 months). Class imbalance ratio: 7.9:1.",
        "constraints": "Use exact metric values for every model evaluation claim. Never recommend an algorithm without comparing it to at least 2 alternatives on the same metrics. Address class imbalance explicitly with at least 2 techniques. State all hyperparameter values, not just algorithm names. Validate using time-based splits only -- never random cross-validation for time-series churn data. Report precision, recall, F1, and AUC-ROC for every model. Never claim a model 'works well' without stating the exact threshold and its business impact in dollar terms.",
        "format": "Start with an executive summary (3 sentences). Then methodology section with numbered steps. Model comparison table (Algorithm, Precision, Recall, F1, AUC-ROC, Training Time). Feature importance chart as a ranked list. Close with deployment recommendations and expected business impact in dollars.",
    },
    {
        "id": "api_design",
        "task": "Design a RESTful API for a multi-tenant expense management system",
        "persona": "You are an API architect with 8 years of experience designing APIs used by 500+ third-party integrators. You authored your company's API design guidelines and have shipped APIs processing $2B+ in annual transaction volume.",
        "context": "The expense management system serves 3,200 companies (tenants) with 180,000 total users. It handles expense report creation, approval workflows, receipt OCR, policy enforcement, and accounting system integration. The API will be used by internal mobile/web apps and 50+ third-party accounting integrations.",
        "data": "Transaction volume: 2.4M expense reports/month, 8.7M line items/month. Average API response time target: <200ms p95. Concurrent users at peak: 12,000. Authentication: OAuth 2.0 with tenant-scoped tokens. Rate limits: 1,000 req/min per tenant (standard), 5,000 req/min (enterprise). Data retention: 7 years for compliance.",
        "constraints": "Follow REST conventions exactly -- proper HTTP methods, status codes, and idempotency. Every endpoint must have a stated rate limit and authentication requirement. Never design an endpoint that returns unbounded results -- always paginate. Use consistent naming (snake_case for fields, kebab-case for URLs). Every write operation must be idempotent. Never expose internal IDs -- use UUIDs. Address multi-tenancy isolation at every layer. State the exact HTTP status code for every error condition. Never design a breaking change without a versioning strategy.",
        "format": "Start with base URL and authentication overview. Then endpoint table (Method, Path, Description, Auth, Rate Limit). Then 3 detailed endpoint examples with full request/response JSON. Error response format specification. Pagination specification. Versioning strategy.",
    },
    {
        "id": "security_audit",
        "task": "Perform a security audit of this authentication flow and identify all vulnerabilities",
        "persona": "You are a senior application security engineer with OSCP and CISSP certifications, 10 years of experience in penetration testing, and deep expertise in OAuth 2.0, JWT, and session management vulnerabilities.",
        "context": "A fintech application handling $50M in monthly transactions uses a custom authentication system. The system was built by a 3-person team in 6 months without a security review. It serves 25,000 users across web and mobile platforms. The application is subject to PCI-DSS and SOC 2 Type II compliance requirements.",
        "data": "Auth flow: (1) User submits email + password via POST /api/login. (2) Server validates against bcrypt hash (cost factor 10). (3) Server generates JWT with user_id, email, role, exp=24h, signed with HS256 using a static secret stored in .env. (4) JWT returned in response body. (5) Client stores JWT in localStorage. (6) All API requests include JWT in Authorization header. (7) Password reset: generates 6-digit code, sends via email, code valid for 1 hour, no rate limit on verification endpoint.",
        "constraints": "Classify every vulnerability using CVSS 3.1 base score. Map each finding to the relevant OWASP Top 10 category. Never state a finding without a specific, implementable remediation. Rank findings by risk (CRITICAL/HIGH/MEDIUM/LOW). Never recommend a security control without stating its implementation complexity and timeline. Address each component of the auth flow independently. State the exact attack vector for each vulnerability -- not just the category. Never assume a vulnerability is 'theoretical' if an attack path exists.",
        "format": "Executive summary with total finding count by severity. Findings table (ID, Title, CVSS, OWASP Category, Component). Then each finding as: Description, Attack Vector, Impact, Remediation, Implementation Effort. Close with a prioritized remediation roadmap.",
    },
    {
        "id": "devops_migration",
        "task": "Plan the migration of a monolithic application to Kubernetes",
        "persona": "You are a DevOps architect with 8 years of experience in container orchestration, having led 6 monolith-to-Kubernetes migrations for applications serving 10M+ users.",
        "context": "A Ruby on Rails monolith serves an e-commerce platform with $120M annual GMV. Current deployment: 4 EC2 c5.4xlarge instances behind an ALB, with a single PostgreSQL RDS instance (db.r5.2xlarge). CI/CD uses Jenkins. Deployments happen twice weekly with 15-minute downtime windows. Team size: 12 engineers, 2 with Kubernetes experience.",
        "data": "Application metrics: 3,200 req/sec average, 8,500 req/sec peak (Black Friday: 22,000). Response time: p50=120ms, p95=450ms, p99=1.2s. Database: 340GB, 5,000 queries/sec. Background jobs: 45,000/hour via Sidekiq (Redis-backed). Memory per instance: 14GB RSS. Boot time: 45 seconds. Test suite: 4,200 tests, 22-minute runtime. Monthly infrastructure cost: $18,400.",
        "constraints": "Never recommend a tool without stating why it was chosen over the top alternative. Provide exact resource requests/limits for every container. State the expected cost difference (current vs. Kubernetes) with line items. Every migration phase must have a rollback procedure. Never recommend running stateful workloads in Kubernetes without justifying it against managed alternatives. Address the team skill gap explicitly with training timeline and cost. State exact downtime expectations for each phase. Never use 'shift left' or 'cloud native' without defining what you mean concretely.",
        "format": "Start with a migration phases table (Phase, Duration, Risk, Rollback). Then architecture diagram (ASCII). Then per-phase detailed plan with exact commands/configs. Resource allocation table. Cost comparison table (current vs. projected). Risk matrix. Training plan.",
    },
    {
        "id": "technical_writing",
        "task": "Write API documentation for a webhook delivery system",
        "persona": "You are a senior technical writer with 8 years of experience writing API documentation for developer platforms. You have documented APIs used by 100,000+ developers at Stripe, Twilio, and Shopify.",
        "context": "The webhook delivery system allows customers to receive real-time HTTP callbacks when events occur in the platform. The system handles 2M webhook deliveries per day across 8,000 registered endpoints. It supports retry logic, signature verification, and delivery status tracking.",
        "data": "Supported events: 24 event types across 4 categories (orders, payments, inventory, customers). Delivery SLA: 99.5% within 30 seconds. Retry schedule: 5 retries at 1min, 5min, 30min, 2hr, 24hr. Payload format: JSON, average 1.2KB. Signature: HMAC-SHA256 with per-endpoint secret. Rate limit: 10,000 deliveries/hour per endpoint. Max payload: 64KB. Timeout: 10 seconds per delivery attempt.",
        "constraints": "Every code example must be complete and copy-pasteable -- no pseudocode, no ellipsis, no 'your-value-here' without a realistic example. Include examples in exactly 3 languages: Python, Node.js, and Go. Never describe a parameter without stating its type, whether it is required, and its default value. Every error response must include the HTTP status code and a realistic error body. Never assume the reader knows what a webhook is -- define it in the first paragraph. Test every code example mentally for syntax errors. Never use 'simply' or 'just' -- these words dismiss complexity.",
        "format": "Start with a 2-sentence overview. Then: Quick Start (working example in <20 lines), Concepts, Event Types table, Endpoint Configuration, Signature Verification (with code in 3 languages), Retry Logic, Error Handling, Best Practices (numbered list). Each section has a clear heading.",
    },
    {
        "id": "debugging",
        "task": "Diagnose why this distributed system experiences cascading failures under load",
        "persona": "You are a site reliability engineer with 10 years of experience debugging distributed systems at scale. You have diagnosed and resolved cascading failures in systems serving 100M+ users at Netflix, Google, and Amazon.",
        "context": "A microservices architecture serving a food delivery platform experiences cascading failures when order volume exceeds 5,000/minute. The system recovers only after manual intervention (restarting the order-processing service). The issue started 3 weeks ago after a deployment that added a new payment provider integration.",
        "data": "Architecture: API Gateway -> Order Service -> [Payment Service, Restaurant Service, Delivery Service]. Order Service: 8 pods, 2 CPU / 4GB each, Go. Payment Service: 4 pods, 1 CPU / 2GB each, Java. Connection pool: 50 connections per pod to PostgreSQL. Timeout: Payment Service has 30s timeout to payment providers. Circuit breaker: none configured. During failure: Order Service CPU hits 100%, response time goes from 50ms to 30s, error rate jumps from 0.1% to 45%. Payment Service logs show 'connection pool exhausted' errors. PostgreSQL connections hit max (200) during incidents.",
        "constraints": "Trace the exact failure chain from trigger to impact -- every link must be stated. Never diagnose without stating the evidence that supports the diagnosis. Provide the exact configuration change for every fix (not just 'add a circuit breaker' -- state the exact thresholds, timeout values, and retry counts). Distinguish between immediate mitigation (stop the bleeding) and permanent fix (prevent recurrence). Every recommendation must have a measurable success criterion. Never recommend more than 3 changes in a single deployment. State the expected recovery time for each mitigation step.",
        "format": "Start with root cause in one sentence. Then: Failure Chain (numbered sequence diagram), Evidence (table mapping symptom to cause), Immediate Mitigation (3 steps with exact configs), Permanent Fixes (prioritized list with implementation effort), Monitoring Additions (exact metrics and alert thresholds).",
    },
]


# ---------------------------------------------------------------------------
# Band ablation logic
# ---------------------------------------------------------------------------

BAND_ORDER = ["persona", "context", "data", "constraints", "format", "task"]
BAND_NAMES = ["PERSONA", "CONTEXT", "DATA", "CONSTRAINTS", "FORMAT", "TASK"]


def build_prompt_at_level(task: Dict[str, str], num_bands: int) -> Tuple[str, Dict[str, Any]]:
    """Build a prompt using only the first N bands (by importance-weighted order).

    Band inclusion priority (by importance weight):
    1. TASK (always included -- the objective)
    2. CONSTRAINTS (42.7% importance)
    3. FORMAT (26.3% importance)
    4. PERSONA (7.0% importance)
    5. CONTEXT (6.3% importance)
    6. DATA (3.8% importance)

    Returns (prompt_text, sinc_json).
    """
    # Priority order for inclusion (most to least important for SNR)
    priority = ["task", "constraints", "format", "persona", "context", "data"]
    included = priority[:num_bands]

    # Build band contents
    bands = {
        "persona": task.get("persona", "") if "persona" in included else "",
        "context": task.get("context", "") if "context" in included else "",
        "data": task.get("data", "") if "data" in included else "",
        "constraints": task.get("constraints", "") if "constraints" in included else "",
        "fmt": task.get("format", "") if "format" in included else "",
        "task": task.get("task", "") if "task" in included else task.get("task", ""),
    }

    sinc_json = build_sinc_json(**bands)

    # Build a flat prompt from included bands
    parts = []
    if bands["persona"]:
        parts.append(f"[PERSONA]\n{bands['persona']}")
    if bands["context"]:
        parts.append(f"[CONTEXT]\n{bands['context']}")
    if bands["data"]:
        parts.append(f"[DATA]\n{bands['data']}")
    if bands["constraints"]:
        parts.append(f"[CONSTRAINTS]\n{bands['constraints']}")
    if bands["fmt"]:
        parts.append(f"[FORMAT]\n{bands['fmt']}")
    if bands["task"]:
        parts.append(f"[TASK]\n{bands['task']}")

    prompt_text = "\n\n".join(parts)
    return prompt_text, sinc_json


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_ablation(provider: str, model: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """Run the full ablation study."""
    print("=" * 72)
    print("SINC-LLM ABLATION STUDY")
    print(f"Provider: {provider} | Model: {model}")
    print(f"Tasks: {len(ABLATION_TASKS)} | Band levels: 1-6")
    print(f"Total trials: {len(ABLATION_TASKS) * 6}")
    print("=" * 72)
    print()

    results: List[Dict[str, Any]] = []
    total_trials = len(ABLATION_TASKS) * 6
    trial_num = 0

    for task in ABLATION_TASKS:
        task_id = task["id"]
        print(f"--- Task: {task_id} ---")

        for num_bands in range(1, 7):
            trial_num += 1
            prompt_text, sinc_json = build_prompt_at_level(task, num_bands)
            snr_data = compute_snr(sinc_json)

            print(f"  [{trial_num}/{total_trials}] {num_bands}-band (SNR={snr_data['snr']}) ...", end=" ", flush=True)

            t0 = time.time()
            response = generate(prompt_text, provider, model)
            elapsed = time.time() - t0

            # Compute metrics
            quality = compute_quality_score(response)
            hedging = compute_hedging_density(response)
            specificity = compute_specificity(response)
            word_count = len(response.split())

            is_error = response.startswith("[") and "ERROR" in response

            result = {
                "task_id": task_id,
                "num_bands": num_bands,
                "snr": snr_data["snr"],
                "snr_grade": snr_data["grade"],
                "quality_score": quality,
                "word_count": word_count,
                "hedging_density": round(hedging, 4),
                "specificity": round(specificity, 4),
                "elapsed_s": round(elapsed, 2),
                "prompt_tokens": estimate_tokens(prompt_text),
                "response_tokens": estimate_tokens(response),
                "error": is_error,
                "provider": provider,
                "model": model,
            }
            results.append(result)

            status = "ERR" if is_error else f"Q={quality:.2f}"
            print(f"{status} | {word_count}w | hedge={hedging:.2f} | spec={specificity:.2f} | {elapsed:.1f}s")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print()
    print("=" * 72)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 72)
    print()

    # Aggregate by band count
    band_stats: Dict[int, Dict[str, List[float]]] = {}
    for r in results:
        if r["error"]:
            continue
        nb = r["num_bands"]
        if nb not in band_stats:
            band_stats[nb] = {"quality": [], "hedging": [], "specificity": [], "words": []}
        band_stats[nb]["quality"].append(r["quality_score"])
        band_stats[nb]["hedging"].append(r["hedging_density"])
        band_stats[nb]["specificity"].append(r["specificity"])
        band_stats[nb]["words"].append(r["word_count"])

    print(f"{'Bands':>5} | {'SNR Grade':>10} | {'Quality':>8} | {'Hedging':>8} | {'Specificity':>11} | {'Avg Words':>9}")
    print("-" * 72)

    for nb in sorted(band_stats.keys()):
        stats = band_stats[nb]
        avg_q = sum(stats["quality"]) / len(stats["quality"])
        avg_h = sum(stats["hedging"]) / len(stats["hedging"])
        avg_s = sum(stats["specificity"]) / len(stats["specificity"])
        avg_w = sum(stats["words"]) / len(stats["words"])

        # Compute SNR for a representative prompt at this band level
        _, sinc_json = build_prompt_at_level(ABLATION_TASKS[0], nb)
        snr_data = compute_snr(sinc_json)

        print(f"{nb:>5} | {snr_data['grade']:>10} | {avg_q:>8.4f} | {avg_h:>8.4f} | {avg_s:>11.4f} | {avg_w:>9.0f}")

    print()
    print(f"Total trials: {len(results)}")
    error_count = sum(1 for r in results if r["error"])
    if error_count:
        print(f"Errors: {error_count}")

    # Key finding
    if band_stats.get(6) and band_stats.get(1):
        q6 = sum(band_stats[6]["quality"]) / len(band_stats[6]["quality"])
        q1 = sum(band_stats[1]["quality"]) / len(band_stats[1]["quality"])
        improvement = ((q6 - q1) / max(q1, 0.001)) * 100
        print(f"\n6-band vs 1-band quality improvement: {improvement:.1f}%")

    # Save results
    output = {
        "experiment": "ablation",
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
        prog="run_ablation",
        description=(
            "sinc-llm Ablation Study: Test prompt quality at 1-band through 6-band levels.\n"
            "Measures SNR, response quality, hedging density, and specificity."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_ablation.py --provider ollama --model llama3\n"
            "  python run_ablation.py --provider anthropic --model claude-haiku-4-20250514\n"
            "  python run_ablation.py --provider openai --model gpt-4o-mini\n"
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
        help="Model name (e.g., llama3, claude-haiku-4-20250514, gpt-4o-mini)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save results as JSON (default: print to stdout only)"
    )

    args = parser.parse_args()
    run_ablation(args.provider, args.model, args.output)


if __name__ == "__main__":
    main()
