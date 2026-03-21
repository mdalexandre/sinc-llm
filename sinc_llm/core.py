"""
sinc-llm core: Zero-dependency SNR computation and fragment detection.

Implements the Nyquist-Shannon sampling theorem applied to LLM prompts.
All functions here use only Python stdlib -- no external packages required.

Formula: x(t) = Sigma x(nT) * sinc((t - nT) / T)

Zone functions (MATLAB-fitted from dropout simulation):
    G(Z1) = gate function for PERSONA       (binary: present/absent)
    H(Z2) = Hill function for CONTEXT+DATA   (peaks at 94 tokens, decays past)
    R(Z3) = ramp function for CONSTRAINTS    (linear above 37, saturates at 1)
    G(Z4) = gate function for FORMAT         (binary: present/absent)

    SNR = 0.588 + 0.267 * G(Z1) * H(Z2) * R(Z3) * G(Z4)

Author: Mario Alexandre
License: MIT
DOI: 10.5281/zenodo.19152668
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BETA0: float = 0.588
AMPLITUDE: float = 0.267
CEILING: float = BETA0 + AMPLITUDE  # 0.855

FORMULA: str = "x(t) = Sigma x(nT) * sinc((t - nT) / T)"

FRAGMENT_AXIS: Dict[int, Dict[str, Any]] = {
    0: {"t": "PERSONA",     "zone": "Z1", "importance": 0.070, "role": "system"},
    1: {"t": "CONTEXT",     "zone": "Z2", "importance": 0.063, "role": "system"},
    2: {"t": "DATA",        "zone": "Z2", "importance": 0.038, "role": "system"},
    3: {"t": "CONSTRAINTS", "zone": "Z3", "importance": 0.427, "role": "system"},
    4: {"t": "FORMAT",      "zone": "Z4", "importance": 0.263, "role": "system"},
    5: {"t": "TASK",        "zone": "Z3", "importance": 0.028, "role": "user"},
}

BAND_NAMES: List[str] = [
    "PERSONA", "CONTEXT", "DATA", "CONSTRAINTS", "FORMAT", "TASK",
]

# Classification markers for fragment detection
PERSONA_MARKERS: List[str] = [
    "you are", "role:", "act as", "expert", "specialist", "senior", "analyst",
]
CONSTRAINT_MARKERS: List[str] = [
    "must", "never", "always", "constraint:", "do not", 'no "',
    "required", "forbidden", "ensure", "state facts", "list every",
]
FORMAT_MARKERS: List[str] = [
    "format:", "output:", "table", "lead with", "structured",
    "headers", "numbered", "bullet", "no disclaimers", "comparison table",
]
TASK_MARKERS: List[str] = [
    "task:", "assess whether", "design the", "build a", "create a",
    "find the", "determine", "evaluate", "analyze the", "deliver",
]
DATA_MARKERS: List[str] = [
    "data:", "cutoff", "ceiling", "statistics:", "metrics:", "figures:",
]

# Fragment templates injected when missing
FRAGMENT_TEMPLATES: Dict[str, str] = {
    "PERSONA": "You are an expert analyst with domain authority.",
    "CONSTRAINTS": (
        'State facts directly. Never hedge -- no "I think", "probably", '
        '"perhaps", "might", "may", "it seems". '
        'Never qualify -- no "however", "although", "but", "while", "despite". '
        'Never use vague quantities -- no "several", "many", "some", "a few". '
        "Use exact numbers, dates, percentages, counts for every claim."
    ),
    "FORMAT": (
        "Lead with the definitive answer. Structured headers. "
        "Tables for comparisons. Numbered lists for sequences. "
        "No parenthetical asides. No trailing summaries."
    ),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Fragment:
    """A single sinc frequency band on the specification axis.

    Attributes:
        n: Band index (0=PERSONA .. 5=TASK, 6=TASK_ARCHIVED optional).
        t: Band name (PERSONA, CONTEXT, DATA, CONSTRAINTS, FORMAT, TASK).
        x: Content string for this band.
    """
    n: int
    t: str
    x: str

    def to_dict(self) -> Dict[str, Any]:
        return {"n": self.n, "t": self.t, "x": self.x}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Fragment":
        return cls(n=d.get("n", 0), t=d.get("t", ""), x=d.get("x", ""))

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.x)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate token count. Approximately 4 characters per token for English.

    This is a fast heuristic -- exact counts require a tokenizer.
    Returns at least 1 for non-empty text, 0 for empty.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Zone functions (MATLAB-fitted)
# ---------------------------------------------------------------------------

def G(x: float) -> float:
    """Gate function for Z1 (PERSONA) and Z4 (FORMAT).

    Binary presence gate: smoothly transitions from 0 to 1
    between 2 and 5 tokens. Below 2 tokens = absent.

    G(x) = clamp((x - 2) / 3, 0, 1)
    """
    return min(1.0, max(0.0, (x - 2.0) / 3.0))


def H(z: float) -> float:
    """Hill function for Z2 (CONTEXT + DATA).

    Peaks at 94 tokens with sigmoidal rise and Gaussian decay.
    Models the information-theoretic sweet spot: too little context
    starves the model, too much causes spectral leakage.

    H(z) = (z/94)^4.56 / (1 + (z/94)^4.56) * exp(-1.36e-5 * (z-94)^2)
    """
    if z <= 0:
        return 0.0
    ratio = z / 94.0
    sigmoid = (ratio ** 4.56) / (1.0 + ratio ** 4.56)
    decay = math.exp(-1.36e-5 * (z - 94.0) ** 2)
    return sigmoid * decay


def R(z: float, tau: float = 37.0, k: float = 0.03) -> float:
    """Ramp function for Z3 (CONSTRAINTS).

    Linear above threshold tau, saturating at 1.0.
    Constraints below 37 tokens provide zero contribution
    (insufficient specificity). Each additional token adds 3% up to saturation.

    R(z) = clamp(k * max(0, z - tau), 0, 1)
    """
    return min(1.0, k * max(0.0, z - tau))


# ---------------------------------------------------------------------------
# SNR computation
# ---------------------------------------------------------------------------

def compute_snr_from_tokens(z1: int, z2: int, z3: int, z4: int) -> float:
    """Compute Signal-to-Noise Ratio from zone token counts.

    SNR = beta0 + amplitude * G(Z1) * H(Z2) * R(Z3) * G(Z4)
        = 0.588 + 0.267 * G(Z1) * H(Z2) * R(Z3) * G(Z4)

    Args:
        z1: Token count for zone 1 (PERSONA).
        z2: Token count for zone 2 (CONTEXT + DATA combined).
        z3: Token count for zone 3 (CONSTRAINTS).
        z4: Token count for zone 4 (FORMAT).

    Returns:
        SNR value in [0.588, 0.855]. Floor of 0.588 is the baseline
        (raw prompt with no structure). Ceiling of 0.855 is theoretical max.
    """
    return BETA0 + AMPLITUDE * G(z1) * H(z2) * R(z3) * G(z4)


def grade_snr(snr: float) -> str:
    """Grade an SNR value.

    Returns:
        EXCELLENT (>= 0.80), GOOD (>= 0.70), ADEQUATE (>= 0.65),
        ALIASED (>= 0.60), or CRITICAL (< 0.60).
    """
    if snr >= 0.80:
        return "EXCELLENT"
    if snr >= 0.70:
        return "GOOD"
    if snr >= 0.65:
        return "ADEQUATE"
    if snr >= 0.60:
        return "ALIASED"
    return "CRITICAL"


def compute_snr(sinc_json: Dict[str, Any]) -> Dict[str, Any]:
    """Compute SNR from a sinc JSON structure.

    Parses fragments, computes zone token counts, evaluates zone functions,
    and returns a complete quality report.

    Args:
        sinc_json: Dict with "fragments" array of {n, t, x} objects.

    Returns:
        Dict with keys: snr, grade, fragments (count string),
        zones (G/H/R/G values), tokens (Z1-Z4 counts).
    """
    fragments = {f.get("n"): f for f in sinc_json.get("fragments", [])}

    z1 = estimate_tokens(fragments.get(0, {}).get("x", ""))
    z2 = (estimate_tokens(fragments.get(1, {}).get("x", ""))
          + estimate_tokens(fragments.get(2, {}).get("x", "")))
    z3 = estimate_tokens(fragments.get(3, {}).get("x", ""))
    z4 = estimate_tokens(fragments.get(4, {}).get("x", ""))

    snr = compute_snr_from_tokens(z1, z2, z3, z4)
    n_present = sum(1 for n in range(6) if n in fragments and fragments[n].get("x"))

    return {
        "snr": round(snr, 4),
        "grade": grade_snr(snr),
        "fragments": f"{n_present}/6",
        "zones": {
            "G(Z1)": round(G(z1), 4),
            "H(Z2)": round(H(z2), 4),
            "R(Z3)": round(R(z3), 4),
            "G(Z4)": round(G(z4), 4),
        },
        "tokens": {"Z1": z1, "Z2": z2, "Z3": z3, "Z4": z4},
    }


# ---------------------------------------------------------------------------
# Fragment detection
# ---------------------------------------------------------------------------

def detect_fragments(prompt: str) -> Dict[str, bool]:
    """Detect which sinc frequency bands are present in a raw prompt.

    Scans the prompt text for classification markers associated with
    each band. Returns a dict mapping band name to boolean presence.

    Args:
        prompt: Raw prompt text to analyze.

    Returns:
        Dict with keys PERSONA, CONTEXT, DATA, CONSTRAINTS, FORMAT, TASK.
        Each value is True if that band's markers were detected.
    """
    lower = prompt.lower()
    return {
        "PERSONA": any(m in lower for m in PERSONA_MARKERS),
        "CONTEXT": (
            len(prompt.split()) > 20
            and not all(
                any(m in lower for m in CONSTRAINT_MARKERS + FORMAT_MARKERS + TASK_MARKERS)
                for _ in [None]
            )
        ),
        "DATA": any(m in lower for m in DATA_MARKERS) or bool(re.search(r"\d{4,}", prompt)),
        "CONSTRAINTS": any(m in lower for m in CONSTRAINT_MARKERS),
        "FORMAT": any(m in lower for m in FORMAT_MARKERS),
        "TASK": any(m in lower for m in TASK_MARKERS),
    }


# ---------------------------------------------------------------------------
# Sinc JSON construction
# ---------------------------------------------------------------------------

def build_sinc_json(
    persona: str = "",
    context: str = "",
    data: str = "",
    constraints: str = "",
    fmt: str = "",
    task: str = "",
    task_archived: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a complete sinc JSON structure from band contents.

    Convenience function to construct the standard sinc prompt format.
    Empty bands are included with empty strings (they will contribute
    zero tokens to their zone function, correctly zeroing that gate).

    Args:
        persona: Who should answer (n=0, Z1).
        context: Situation, facts, dates (n=1, Z2).
        data: Specific inputs, metrics (n=2, Z2).
        constraints: Rules, MUST/NEVER directives (n=3, Z3).
        fmt: Output structure specification (n=4, Z4).
        task: The objective (n=5, Z3).
        task_archived: Optional original raw prompt (n=6).

    Returns:
        Complete sinc JSON dict ready for serialization or execution.
    """
    fragments = [
        {"n": 0, "t": "PERSONA", "x": persona},
        {"n": 1, "t": "CONTEXT", "x": context},
        {"n": 2, "t": "DATA", "x": data},
        {"n": 3, "t": "CONSTRAINTS", "x": constraints},
        {"n": 4, "t": "FORMAT", "x": fmt},
        {"n": 5, "t": "TASK", "x": task},
    ]
    if task_archived is not None:
        fragments.append({"n": 6, "t": "TASK_ARCHIVED", "x": task_archived})

    return {
        "formula": "x(t) = Sigma x(nT) * sinc((t - nT) / T)",
        "T": "specification-axis",
        "fragments": fragments,
    }


# ---------------------------------------------------------------------------
# Sinc JSON parsing and validation
# ---------------------------------------------------------------------------

def parse_sinc_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and validate a sinc-formatted JSON prompt.

    Validates structure, computes zone token counts, SNR, and builds
    system/user prompts suitable for LLM execution.

    Args:
        data: Dict with "formula", "T", and "fragments" fields.

    Returns:
        Dict with keys:
            valid (bool), errors (list), warnings (list),
            fragments (dict by n), z_tokens (Z1-Z4), snr (float),
            system_prompt (str), user_prompt (str), metadata (dict).
    """
    result: Dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "fragments": {},
        "z_tokens": {"Z1": 0, "Z2": 0, "Z3": 0, "Z4": 0},
        "snr": 0.0,
        "system_prompt": "",
        "user_prompt": "",
        "metadata": {},
    }

    # Validate top-level structure
    if "formula" not in data:
        result["errors"].append("Missing 'formula' field")
        result["valid"] = False
    elif "sinc" not in data.get("formula", "").lower() and "sigma" not in data.get("formula", "").lower():
        result["warnings"].append(f"Unexpected formula: {data['formula']}")

    if "fragments" not in data or not isinstance(data["fragments"], list):
        result["errors"].append("Missing or invalid 'fragments' array")
        result["valid"] = False
        return result

    # Parse fragments
    for frag in data["fragments"]:
        n = frag.get("n")
        t = frag.get("t", frag.get("zone", ""))
        x = frag.get("x", "")

        if n is None:
            result["warnings"].append(f"Fragment missing 'n' field: {frag}")
            continue

        tokens = estimate_tokens(x)
        result["fragments"][n] = {"t": t, "x": x, "tokens": tokens}

        # Map to zone token counts
        if n in FRAGMENT_AXIS:
            zone = FRAGMENT_AXIS[n]["zone"]
            result["z_tokens"][zone] += tokens

    # Check for missing critical fragments
    for n, spec in FRAGMENT_AXIS.items():
        if n not in result["fragments"]:
            result["warnings"].append(f"Missing fragment n={n} ({spec['t']})")
            if spec["importance"] > 0.05:
                result["errors"].append(
                    f"CRITICAL: n={n} ({spec['t']}) missing -- "
                    f"{spec['importance'] * 100:.1f}% of reconstruction quality lost"
                )

    # Compute SNR
    z = result["z_tokens"]
    result["snr"] = compute_snr_from_tokens(z["Z1"], z["Z2"], z["Z3"], z["Z4"])

    # Aliasing checks
    if z["Z3"] == 0:
        result["errors"].append("Z3 ALIASING: zero constraint tokens -- reconstruction unreliable")
    if z["Z1"] == 0:
        result["errors"].append("Z1 GATE COLLAPSE: no persona -- amplitude term zeroed")
    if z["Z4"] == 0:
        result["errors"].append("Z4 GATE COLLAPSE: no format -- amplitude term zeroed")
    if z["Z2"] > 165 * 4:
        result["warnings"].append(f"Z2 spectral leakage risk: {z['Z2']} tokens (ceiling: 165)")

    # Build system prompt (n=0 through n=4, ordered broad -> narrow)
    optional_fragments = {
        6: {"t": "TASK_ARCHIVED", "role": "meta"},
        7: {"t": "INTROSPECTION", "role": "meta"},
    }
    system_parts = []
    for n in sorted(result["fragments"].keys()):
        frag = result["fragments"][n]
        spec = FRAGMENT_AXIS.get(n, optional_fragments.get(n, {}))
        if spec.get("role") == "system" and frag["x"]:
            system_parts.append(f"[{frag['t']}]\n{frag['x']}")
        elif spec.get("role") == "meta" and frag["x"]:
            system_parts.append(f"[{frag['t']}]\n{frag['x']}")

    result["system_prompt"] = "\n\n".join(system_parts)

    # Build user prompt (n=5 TASK)
    task_frag = result["fragments"].get(5, {})
    result["user_prompt"] = task_frag.get("x", "")

    # Metadata
    result["metadata"] = {
        "T": data.get("T", "specification-axis"),
        "fragment_count": len(result["fragments"]),
        "total_tokens": sum(f["tokens"] for f in result["fragments"].values()),
        "snr_grade": grade_snr(result["snr"]),
    }

    if result["errors"]:
        result["valid"] = False

    return result
