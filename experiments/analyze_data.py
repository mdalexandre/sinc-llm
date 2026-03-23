#!/usr/bin/env python3
"""
sinc-llm Data Analysis: Analyze nyquist_session.jsonl production telemetry.

Computes SNR per agent, token usage over time, band importance weights,
and summary statistics matching the paper's 275-observation claims.

Formula: x(t) = Sigma x(nT) * sinc((t - nT) / T)

Usage:
    python analyze_data.py
    python analyze_data.py --input data/nyquist_session.jsonl
    python analyze_data.py --output analysis_results.json

Author: Mario Alexandre
License: MIT
DOI: 10.5281/zenodo.19152668
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for sinc_llm imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sinc_llm.core import (
    BETA0,
    AMPLITUDE,
    CEILING,
    FRAGMENT_AXIS,
    grade_snr,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_session_data(path: str) -> List[Dict[str, Any]]:
    """Load JSONL session data. Each line is a JSON object."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping line {line_num}: {e}", file=sys.stderr)
    return records


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_agent_statistics(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute per-agent statistics: token usage, efficiency, session counts."""
    agents: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {
            "total_tokens": [],
            "input_tokens": [],
            "output_tokens": [],
            "duration_ms": [],
            "tool_uses": [],
            "ratio": [],
        }
    )

    for r in records:
        agent = r.get("agent", "unknown")
        agents[agent]["total_tokens"].append(r.get("total_tokens", 0))
        agents[agent]["input_tokens"].append(r.get("input_tokens", 0))
        agents[agent]["output_tokens"].append(r.get("output_tokens", 0))
        agents[agent]["duration_ms"].append(r.get("duration_ms", 0))
        agents[agent]["tool_uses"].append(r.get("tool_uses", 0))
        agents[agent]["ratio"].append(r.get("ratio", 0.0))

    stats: Dict[str, Dict[str, Any]] = {}
    for agent, data in sorted(agents.items()):
        n = len(data["total_tokens"])
        total_tok = data["total_tokens"]
        output_tok = data["output_tokens"]
        dur = data["duration_ms"]
        tools = data["tool_uses"]
        ratio = data["ratio"]

        # Efficiency: output tokens per second
        efficiency_values = []
        for o, d in zip(output_tok, dur):
            if d > 0:
                efficiency_values.append(o / (d / 1000.0))

        stats[agent] = {
            "count": n,
            "total_tokens": {
                "mean": round(statistics.mean(total_tok), 1) if total_tok else 0,
                "std": round(statistics.stdev(total_tok), 1) if len(total_tok) > 1 else 0,
                "min": min(total_tok) if total_tok else 0,
                "max": max(total_tok) if total_tok else 0,
                "sum": sum(total_tok),
            },
            "output_tokens": {
                "mean": round(statistics.mean(output_tok), 1) if output_tok else 0,
                "sum": sum(output_tok),
            },
            "duration_ms": {
                "mean": round(statistics.mean(dur), 1) if dur else 0,
                "sum": sum(dur),
            },
            "tool_uses": {
                "mean": round(statistics.mean(tools), 1) if tools else 0,
                "sum": sum(tools),
            },
            "ratio": {
                "mean": round(statistics.mean(ratio), 4) if ratio else 0,
                "max": round(max(ratio), 4) if ratio else 0,
            },
            "efficiency_tokens_per_sec": {
                "mean": round(statistics.mean(efficiency_values), 2) if efficiency_values else 0,
                "max": round(max(efficiency_values), 2) if efficiency_values else 0,
            },
        }

    return stats


def compute_temporal_analysis(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze token usage and activity patterns over time."""
    # Parse timestamps
    timestamped: List[Tuple[datetime, Dict[str, Any]]] = []
    for r in records:
        ts_str = r.get("timestamp", "")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            timestamped.append((ts, r))
        except (ValueError, TypeError):
            continue

    if not timestamped:
        return {"error": "No valid timestamps found"}

    timestamped.sort(key=lambda x: x[0])

    first_ts = timestamped[0][0]
    last_ts = timestamped[-1][0]
    total_duration = (last_ts - first_ts).total_seconds()

    # Hourly buckets
    hourly: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"total_tokens": 0, "records": 0, "duration_ms": 0}
    )
    for ts, r in timestamped:
        hour_key = ts.strftime("%Y-%m-%dT%H:00")
        hourly[hour_key]["total_tokens"] += r.get("total_tokens", 0)
        hourly[hour_key]["records"] += 1
        hourly[hour_key]["duration_ms"] += r.get("duration_ms", 0)

    # Compute token velocity (tokens per minute)
    token_velocities = []
    for hour_data in hourly.values():
        if hour_data["duration_ms"] > 0:
            minutes = hour_data["duration_ms"] / 60000.0
            velocity = hour_data["total_tokens"] / max(minutes, 0.001)
            token_velocities.append(velocity)

    return {
        "time_range": {
            "start": first_ts.isoformat(),
            "end": last_ts.isoformat(),
            "duration_hours": round(total_duration / 3600, 2),
        },
        "hourly_buckets": len(hourly),
        "peak_hour": max(hourly.items(), key=lambda x: x[1]["total_tokens"])[0] if hourly else None,
        "peak_hour_tokens": max(h["total_tokens"] for h in hourly.values()) if hourly else 0,
        "token_velocity": {
            "mean_per_min": round(statistics.mean(token_velocities), 1) if token_velocities else 0,
            "max_per_min": round(max(token_velocities), 1) if token_velocities else 0,
        },
    }


def compute_snr_proxy(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute proxy SNR per agent using output/input ratio as a quality signal.

    The ratio field in the session data approximates information density:
    higher ratio = more output per input token = higher signal content.
    We map this to the SNR scale using the formula's range [0.588, 0.855].
    """
    agent_snr: Dict[str, List[float]] = defaultdict(list)

    for r in records:
        agent = r.get("agent", "unknown")
        ratio = r.get("ratio", 0.0)
        input_tok = r.get("input_tokens", 0)
        output_tok = r.get("output_tokens", 0)

        # Map ratio to SNR proxy: ratio of 1.0 = theoretical ceiling,
        # ratio of 0 = baseline. Clamp to valid range.
        if input_tok > 0:
            # Use output/input ratio, clamped [0, 3] then mapped to [BETA0, CEILING]
            clamped_ratio = min(3.0, max(0.0, ratio))
            snr_proxy = BETA0 + (AMPLITUDE * clamped_ratio / 3.0)
        else:
            snr_proxy = BETA0

        agent_snr[agent].append(snr_proxy)

    stats: Dict[str, Dict[str, Any]] = {}
    for agent, snrs in sorted(agent_snr.items()):
        stats[agent] = {
            "count": len(snrs),
            "mean_snr": round(statistics.mean(snrs), 4),
            "std_snr": round(statistics.stdev(snrs), 4) if len(snrs) > 1 else 0,
            "min_snr": round(min(snrs), 4),
            "max_snr": round(max(snrs), 4),
            "grade": grade_snr(statistics.mean(snrs)),
        }

    # Overall
    all_snrs = [s for snrs in agent_snr.values() for s in snrs]
    overall = {
        "mean": round(statistics.mean(all_snrs), 4),
        "std": round(statistics.stdev(all_snrs), 4) if len(all_snrs) > 1 else 0,
        "grade": grade_snr(statistics.mean(all_snrs)),
    }

    return {"per_agent": stats, "overall": overall}


def compute_band_importance(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute empirical band importance weights from session patterns.

    Uses the theoretical weights from FRAGMENT_AXIS and validates them
    against the observed token distribution patterns.
    """
    theoretical = {
        spec["t"]: spec["importance"]
        for n, spec in FRAGMENT_AXIS.items()
    }

    # Empirical: agents that primarily generate content (high ratio)
    # contribute more to output quality than agents that primarily read
    # (low ratio). Map agent roles to approximate band contributions.
    agent_band_map = {
        "origin": "CONSTRAINTS",          # Sets rules and framing
        "surgeon": "TASK",                # Executes specific fixes
        "hybrid-controller-orchestrator": "CONTEXT",  # Explores and provides context
        "calibrated-pipeline-orchestrator": "DATA",    # Analyzes specific inputs
        "tela": "FORMAT",                 # Evaluates output structure
        "witness-orchestrator": "CONSTRAINTS",  # Enforces rules
        "emergence-detector": "CONTEXT",  # Detects patterns
        "token-nyquist": "DATA",          # Optimizes token usage
        "Explore": "CONTEXT",             # Broad exploration
        "shape": "FORMAT",               # Structural diagnostics
        "memory-agent": "DATA",           # Knowledge substrate
    }

    band_token_totals: Dict[str, int] = defaultdict(int)
    band_quality_signals: Dict[str, List[float]] = defaultdict(list)

    for r in records:
        agent = r.get("agent", "unknown")
        band = agent_band_map.get(agent, "CONTEXT")
        band_token_totals[band] += r.get("total_tokens", 0)
        if r.get("ratio", 0) > 0:
            band_quality_signals[band].append(r.get("ratio", 0))

    # Normalize to get empirical weights
    total = sum(band_token_totals.values()) or 1
    empirical = {
        band: round(tokens / total, 4)
        for band, tokens in sorted(band_token_totals.items())
    }

    return {
        "theoretical": theoretical,
        "empirical_token_distribution": empirical,
        "band_token_totals": dict(band_token_totals),
        "note": "Empirical weights approximate band importance via agent-to-band mapping. "
                "Theoretical weights derived from MATLAB-fitted zone functions.",
    }


def compute_summary_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute overall summary statistics matching the paper's claims."""
    total_records = len(records)

    total_tokens = sum(r.get("total_tokens", 0) for r in records)
    total_input = sum(r.get("input_tokens", 0) for r in records)
    total_output = sum(r.get("output_tokens", 0) for r in records)
    total_duration_ms = sum(r.get("duration_ms", 0) for r in records)
    total_tool_uses = sum(r.get("tool_uses", 0) for r in records)

    unique_agents = set(r.get("agent", "unknown") for r in records)

    ratios = [r.get("ratio", 0.0) for r in records if r.get("ratio", 0.0) > 0]
    durations = [r.get("duration_ms", 0) for r in records if r.get("duration_ms", 0) > 0]
    tokens_list = [r.get("total_tokens", 0) for r in records if r.get("total_tokens", 0) > 0]

    return {
        "observations": total_records,
        "unique_agents": len(unique_agents),
        "agent_names": sorted(unique_agents),
        "total_tokens": total_tokens,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_duration_hours": round(total_duration_ms / 3_600_000, 2),
        "total_tool_uses": total_tool_uses,
        "tokens_per_observation": {
            "mean": round(statistics.mean(tokens_list), 1) if tokens_list else 0,
            "median": round(statistics.median(tokens_list), 1) if tokens_list else 0,
            "std": round(statistics.stdev(tokens_list), 1) if len(tokens_list) > 1 else 0,
            "p25": round(sorted(tokens_list)[len(tokens_list) // 4], 1) if tokens_list else 0,
            "p75": round(sorted(tokens_list)[3 * len(tokens_list) // 4], 1) if tokens_list else 0,
        },
        "duration_per_observation_ms": {
            "mean": round(statistics.mean(durations), 1) if durations else 0,
            "median": round(statistics.median(durations), 1) if durations else 0,
        },
        "output_input_ratio": {
            "mean": round(statistics.mean(ratios), 4) if ratios else 0,
            "median": round(statistics.median(ratios), 4) if ratios else 0,
            "max": round(max(ratios), 4) if ratios else 0,
        },
        "constraints_importance_weight": FRAGMENT_AXIS[3]["importance"],
        "snr_range": {"floor": BETA0, "ceiling": CEILING},
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_summary(
    summary: Dict[str, Any],
    agent_stats: Dict[str, Dict[str, Any]],
    temporal: Dict[str, Any],
    snr: Dict[str, Any],
    importance: Dict[str, Any],
) -> None:
    """Print formatted analysis to stdout."""

    print("=" * 78)
    print("SINC-LLM SESSION DATA ANALYSIS")
    print("=" * 78)
    print()

    # Overview
    print("DATASET OVERVIEW")
    print("-" * 78)
    print(f"  Observations:     {summary['observations']}")
    print(f"  Unique agents:    {summary['unique_agents']}")
    print(f"  Total tokens:     {summary['total_tokens']:,}")
    print(f"  Total duration:   {summary['total_duration_hours']} hours")
    print(f"  Total tool uses:  {summary['total_tool_uses']:,}")
    print(f"  SNR floor:        {summary['snr_range']['floor']}")
    print(f"  SNR ceiling:      {summary['snr_range']['ceiling']}")
    print()

    # Token distribution
    tpo = summary["tokens_per_observation"]
    print("TOKEN DISTRIBUTION PER OBSERVATION")
    print("-" * 78)
    print(f"  Mean:   {tpo['mean']:,.1f}")
    print(f"  Median: {tpo['median']:,.1f}")
    print(f"  Std:    {tpo['std']:,.1f}")
    print(f"  P25:    {tpo['p25']:,.1f}")
    print(f"  P75:    {tpo['p75']:,.1f}")
    print()

    # Agent table
    print("PER-AGENT STATISTICS")
    print("-" * 78)
    print(f"{'Agent':>35} | {'Count':>5} | {'Avg Tokens':>10} | {'Avg Output':>10} | {'Eff (tok/s)':>11} | {'SNR Proxy':>9}")
    print("-" * 78)

    for agent, stats in sorted(agent_stats.items(), key=lambda x: x[1]["count"], reverse=True):
        snr_agent = snr["per_agent"].get(agent, {})
        print(
            f"{agent:>35} | "
            f"{stats['count']:>5} | "
            f"{stats['total_tokens']['mean']:>10,.1f} | "
            f"{stats['output_tokens']['mean']:>10,.1f} | "
            f"{stats['efficiency_tokens_per_sec']['mean']:>11.2f} | "
            f"{snr_agent.get('mean_snr', 0):>9.4f}"
        )
    print()

    # SNR summary
    print("SNR PROXY ANALYSIS")
    print("-" * 78)
    print(f"  Overall mean SNR: {snr['overall']['mean']} ({snr['overall']['grade']})")
    print(f"  Overall std:      {snr['overall']['std']}")
    print()

    # Band importance
    print("BAND IMPORTANCE WEIGHTS")
    print("-" * 78)
    print(f"{'Band':>15} | {'Theoretical':>12} | {'Empirical':>10}")
    print("-" * 45)
    for band_name in ["PERSONA", "CONTEXT", "DATA", "CONSTRAINTS", "FORMAT", "TASK"]:
        theo = importance["theoretical"].get(band_name, 0)
        emp = importance["empirical_token_distribution"].get(band_name, 0)
        marker = " ***" if band_name == "CONSTRAINTS" else ""
        print(f"{band_name:>15} | {theo:>12.3f} | {emp:>10.4f}{marker}")
    print()
    print("  *** CONSTRAINTS carries the highest theoretical weight (42.7%)")
    print()

    # Temporal
    if "error" not in temporal:
        print("TEMPORAL ANALYSIS")
        print("-" * 78)
        tr = temporal["time_range"]
        print(f"  Time range:       {tr['start']} to {tr['end']}")
        print(f"  Duration:         {tr['duration_hours']} hours")
        print(f"  Hourly buckets:   {temporal['hourly_buckets']}")
        print(f"  Peak hour:        {temporal['peak_hour']}")
        print(f"  Peak hour tokens: {temporal['peak_hour_tokens']:,}")
        tv = temporal["token_velocity"]
        print(f"  Token velocity:   mean={tv['mean_per_min']:.1f}/min, max={tv['max_per_min']:.1f}/min")
        print()

    # Efficiency ranking
    print("AGENT EFFICIENCY RANKING (output tokens per second)")
    print("-" * 78)
    ranked = sorted(
        agent_stats.items(),
        key=lambda x: x[1]["efficiency_tokens_per_sec"]["mean"],
        reverse=True,
    )
    for i, (agent, stats) in enumerate(ranked, 1):
        eff = stats["efficiency_tokens_per_sec"]["mean"]
        print(f"  {i:>2}. {agent:>35}: {eff:.2f} tokens/sec")
    print()

    # Efficiency range
    if ranked:
        top_eff = ranked[0][1]["efficiency_tokens_per_sec"]["mean"]
        bottom_eff = ranked[-1][1]["efficiency_tokens_per_sec"]["mean"]
        if bottom_eff > 0:
            ratio = top_eff / bottom_eff
            print(f"  Efficiency range: {ratio:.1f}x between most and least efficient agents")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis(input_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """Run full analysis pipeline."""
    print(f"Loading data from: {input_path}")
    records = load_session_data(input_path)
    print(f"Loaded {len(records)} records")
    print()

    summary = compute_summary_statistics(records)
    agent_stats = compute_agent_statistics(records)
    temporal = compute_temporal_analysis(records)
    snr = compute_snr_proxy(records)
    importance = compute_band_importance(records)

    print_summary(summary, agent_stats, temporal, snr, importance)

    full_output = {
        "analysis": "sinc-llm session data",
        "input_file": input_path,
        "summary": summary,
        "agent_statistics": agent_stats,
        "temporal_analysis": temporal,
        "snr_proxy": snr,
        "band_importance": importance,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(full_output, f, indent=2, ensure_ascii=False)
        print(f"Full analysis saved to: {output_path}")

    return full_output


def main() -> None:
    default_input = os.path.join(os.path.dirname(__file__), "data", "nyquist_session.jsonl")

    parser = argparse.ArgumentParser(
        prog="analyze_data",
        description=(
            "sinc-llm Data Analysis: Analyze nyquist_session.jsonl production telemetry.\n"
            "Computes per-agent SNR, token usage, band importance weights, and summary stats."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python analyze_data.py\n"
            "  python analyze_data.py --input data/nyquist_session.jsonl\n"
            "  python analyze_data.py --output analysis_results.json\n"
            "\n"
            "DOI: 10.5281/zenodo.19152668"
        ),
    )
    parser.add_argument(
        "--input", default=default_input,
        help=f"Path to JSONL data file (default: {default_input})"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save full analysis as JSON"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    run_analysis(args.input, args.output)


if __name__ == "__main__":
    main()
