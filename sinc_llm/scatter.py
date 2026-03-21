"""
sinc-llm scatter: Auto-scatter via Anthropic API.

Converts any raw prompt into 6 Nyquist-compliant frequency bands using
the Anthropic Messages API (not CLI). Requires the `anthropic` SDK.

Formula: x(t) = Sigma x(nT) * sinc((t - nT) / T)

Install with: pip install sinc-llm[execute]

Author: Mario Alexandre
License: MIT
DOI: 10.5281/zenodo.19152668
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from sinc_llm.core import compute_snr

SCATTER_SYSTEM_PROMPT: str = """You are the Auto-Scatter engine. You decompose raw prompts into 6 Nyquist-compliant frequency bands on the specification axis.

Formula: x(t) = Sigma x(nT) * sinc((t - nT) / T)

Each fragment is a frequency band:
  n=0 PERSONA      = f0 (lowest freq -- WHO should answer this)
  n=1 CONTEXT      = f1 (situation, background, dates, numbers)
  n=2 DATA         = f2 (specific inputs the model needs)
  n=3 CONSTRAINTS  = f3 (rules -- HIGHEST ENERGY band, 42.7% of quality)
  n=4 FORMAT       = f4 (exact output structure)
  n=5 TASK         = f5 (highest freq -- the specific objective)

RULES:
1. Output ONLY valid JSON. No markdown. No explanation. No preamble.
2. CONSTRAINTS (n=3) must be the LONGEST fragment -- it carries 42.7% of reconstruction quality.
3. PERSONA (n=0) must be specific -- not "helpful assistant" but the exact expert type needed.
4. FORMAT (n=4) must specify the exact output structure -- sections, tables, lists.
5. CONTEXT (n=1) must include any facts, dates, numbers, or situation details from the raw prompt.
6. DATA (n=2) must extract specific inputs, metrics, or data points. If none exist in the prompt, state what data WOULD be needed and instruct the model to search for it.
7. TASK (n=5) must be a clear imperative -- not a restatement of the prompt.
8. Include n=6 TASK_ARCHIVED with the original raw prompt preserved verbatim.
9. CONSTRAINTS must include at least 5 NEVER/ALWAYS/MUST rules specific to the task.
10. Every fragment "x" value must be a substantial paragraph, not a single sentence.

OUTPUT FORMAT (exact -- no deviation):
{
  "formula": "x(t) = Sigma x(nT) * sinc((t - nT) / T)",
  "T": "specification-axis",
  "fragments": [
    {"n": 0, "t": "PERSONA", "x": "..."},
    {"n": 1, "t": "CONTEXT", "x": "..."},
    {"n": 2, "t": "DATA", "x": "..."},
    {"n": 3, "t": "CONSTRAINTS", "x": "..."},
    {"n": 4, "t": "FORMAT", "x": "..."},
    {"n": 5, "t": "TASK", "x": "..."},
    {"n": 6, "t": "TASK_ARCHIVED", "x": "..."}
  ]
}"""


def _extract_json(text: str) -> str:
    """Extract JSON object from LLM response text.

    Handles markdown code blocks, bare JSON, and surrounding text.
    """
    # Strip markdown code fences
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()

    # Find the outermost JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return text[start:end]
    return text


def scatter(
    raw_prompt: str,
    *,
    model: str = "claude-haiku-4-20250514",
    api_key: Optional[str] = None,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """Convert a raw prompt into sinc-formatted JSON via Anthropic API.

    This is the core auto-scatter operation: a single raw string goes in,
    a 6-band sinc JSON comes out.

    Args:
        raw_prompt: The unstructured prompt to decompose.
        model: Anthropic model to use for decomposition. Defaults to Haiku
               (fast, cheap, sufficient for structural decomposition).
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        max_tokens: Maximum response tokens.

    Returns:
        Sinc JSON dict with formula, T, and fragments array.
        On error, returns dict with "error" key.

    Raises:
        ImportError: If the anthropic package is not installed.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is required for scatter. "
            "Install with: pip install sinc-llm[execute]"
        )

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return {"error": "No API key. Set ANTHROPIC_API_KEY or pass api_key=."}

    client = anthropic.Anthropic(api_key=key)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SCATTER_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Decompose this raw prompt into sinc format:\n\n{raw_prompt}",
                }
            ],
        )
    except Exception as e:
        return {"error": f"Anthropic API error: {e}"}

    # Extract text from response
    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text

    # Parse JSON from response
    json_str = _extract_json(text)
    try:
        sinc_json = json.loads(json_str)
    except json.JSONDecodeError:
        return {"error": "Failed to parse scatter output as JSON", "raw": text[:500]}

    # Validate structure
    if "fragments" not in sinc_json:
        return {"error": "No fragments array in output", "raw": text[:500]}

    # Ensure formula and T are present
    sinc_json.setdefault("formula", "x(t) = Sigma x(nT) * sinc((t - nT) / T)")
    sinc_json.setdefault("T", "specification-axis")

    return sinc_json


def scatter_and_execute(
    raw_prompt: str,
    *,
    scatter_model: str = "claude-haiku-4-20250514",
    execute_model: str = "claude-sonnet-4-20250514",
    api_key: Optional[str] = None,
    max_tokens: int = 8192,
) -> Dict[str, Any]:
    """Scatter a raw prompt and execute it in one call.

    Two-phase pipeline:
    1. Decompose raw prompt into sinc JSON (via Haiku -- fast, cheap).
    2. Execute the structured prompt (via Sonnet -- powerful, thorough).

    Args:
        raw_prompt: The unstructured prompt to process.
        scatter_model: Model for decomposition phase.
        execute_model: Model for execution phase.
        api_key: Anthropic API key.
        max_tokens: Maximum response tokens for execution.

    Returns:
        Dict with response, sinc_format, snr, and model fields.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is required. "
            "Install with: pip install sinc-llm[execute]"
        )

    # Phase 1: Scatter
    sinc_json = scatter(raw_prompt, model=scatter_model, api_key=api_key)
    if "error" in sinc_json:
        return sinc_json

    snr_data = compute_snr(sinc_json)

    # Phase 2: Build prompts from fragments
    system_parts = []
    user_prompt = ""
    for frag in sorted(sinc_json.get("fragments", []), key=lambda f: f.get("n", 0)):
        n = frag.get("n", -1)
        t = frag.get("t", "")
        x = frag.get("x", "")
        if not x:
            continue
        if n <= 4:
            system_parts.append(f"[{t}]\n{x}")
        elif n == 5:
            user_prompt = x

    system_prompt = "\n\n".join(system_parts)

    # Execute
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=key)

    try:
        response = client.messages.create(
            model=execute_model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as e:
        return {
            "error": f"Execution API error: {e}",
            "sinc_format": sinc_json,
            "snr": snr_data,
        }

    response_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            response_text += block.text

    return {
        "response": response_text,
        "sinc_format": sinc_json,
        "snr": snr_data,
        "model": execute_model,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }
