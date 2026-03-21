<p align="center">
  <img src="assets/logo.png" alt="sinc-llm logo" width="180">
</p>

<h1 align="center">sinc-llm</h1>

<p align="center"><strong>Nyquist-Shannon Sampling for LLM Prompts</strong></p>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19152668.svg)](https://doi.org/10.5281/zenodo.19152668)
[![PyPI](https://img.shields.io/pypi/v/sinc-llm)](https://pypi.org/project/sinc-llm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> A raw prompt is 1 sample of a 6-band signal. Nyquist requires 6 samples.
> Sending a raw prompt = 6:1 undersampling = guaranteed aliasing (hallucination).

sinc-llm decomposes any raw prompt into 6 frequency bands on the specification axis,
computes a Signal-to-Noise Ratio, and reconstructs with formal quality guarantees.

```
x(t) = Sigma x(nT) * sinc((t - nT) / T)
```

## The 6 Frequency Bands

| Band | n | Zone | Importance | Role |
|------|---|------|-----------|------|
| **PERSONA** | 0 | Z1 | 7.0% | WHO should answer |
| **CONTEXT** | 1 | Z2 | 6.3% | Situation, facts, dates |
| **DATA** | 2 | Z2 | 3.8% | Specific inputs, metrics |
| **CONSTRAINTS** | 3 | Z3 | **42.7%** | Rules (highest energy band) |
| **FORMAT** | 4 | Z4 | 26.3% | Output structure |
| **TASK** | 5 | Z3 | 2.8% | The objective |

## SNR Formula

```
SNR = 0.588 + 0.267 * G(Z1) * H(Z2) * R(Z3) * G(Z4)
```

Where:
- **G(x)** = gate function: `clamp((x - 2) / 3, 0, 1)` -- binary presence for PERSONA and FORMAT
- **H(z)** = Hill function: peaks at 94 tokens, sigmoidal rise + Gaussian decay -- sweet spot for CONTEXT+DATA
- **R(z)** = ramp function: `clamp(0.03 * max(0, z - 37), 0, 1)` -- linear above 37 tokens for CONSTRAINTS
- Floor: 0.588 (raw prompt baseline)
- Ceiling: 0.855 (theoretical maximum)

## Installation

```bash
# Core only (SNR computation, fragment detection, building -- zero dependencies)
pip install sinc-llm

# With Anthropic API support (scatter, execute)
pip install sinc-llm[execute]
```

## Quick Start

### Python Library

```python
from sinc_llm import build_sinc_json, compute_snr, detect_fragments

# Build a structured prompt
prompt = build_sinc_json(
    persona="You are a senior Python architect.",
    context="Migrating a Django 4.2 monolith to microservices.",
    data="Current codebase: 150k LOC, 47 models, 12 apps.",
    constraints=(
        "MUST preserve all existing API contracts. "
        "NEVER break backward compatibility. "
        "ALWAYS use strangler fig pattern for incremental migration. "
        "Each service MUST own its own database. "
        "MUST include rollback procedure for each step."
    ),
    fmt="Numbered migration plan. Each step: action, risk, rollback. Table of service boundaries.",
    task="Design the migration plan with service boundaries and execution order.",
)

# Compute quality
snr = compute_snr(prompt)
print(f"SNR: {snr['snr']} ({snr['grade']})")
# SNR: 0.8229 (EXCELLENT)

# Detect missing bands in a raw prompt
raw = "How do I optimize my database?"
fragments = detect_fragments(raw)
print(fragments)
# {'PERSONA': False, 'CONTEXT': False, 'DATA': False,
#  'CONSTRAINTS': False, 'FORMAT': False, 'TASK': False}
```

### Auto-Scatter (requires `anthropic` SDK)

```python
from sinc_llm.scatter import scatter, scatter_and_execute

# Decompose a raw prompt into 6 bands
sinc_json = scatter("How do I optimize my database?")

# Full pipeline: scatter + execute
result = scatter_and_execute("How do I optimize my database?")
print(result["response"])
print(f"SNR: {result['snr']['snr']}")
```

### CLI

```bash
# Scatter a raw prompt into sinc JSON
sinc-scatter "Build me a REST API for user management"

# Scatter and execute
sinc-scatter "Build me a REST API" --execute

# Validate a sinc JSON file
sinc-engine --dry-run prompt.json

# Execute a sinc JSON file
sinc-engine prompt.json

# Compute SNR
sinc-llm snr prompt.json

# Build sinc JSON from arguments
sinc-llm build \
  --persona "Senior backend engineer" \
  --constraints "Must use PostgreSQL. Must include auth." \
  --task "Design the API schema"

# Start HTTP server
sinc-server --port 8461
```

### HTTP Server

```bash
sinc-server --port 8461
```

Endpoints:
- `POST /scatter` -- Raw text body -> sinc JSON
- `POST /execute` -- Raw text -> sinc JSON -> LLM -> response
- `POST /snr` -- Sinc JSON -> SNR quality report
- `POST /reconstruct` -- Sinc JSON -> LLM -> response
- `GET /health` -- Health check + capabilities

```bash
# Scatter
curl -X POST http://localhost:8461/scatter -d "How do I optimize my database?"

# SNR check
curl -X POST http://localhost:8461/snr -H "Content-Type: application/json" \
  -d '{"formula":"x(t) = Sigma x(nT) * sinc((t - nT) / T)","T":"specification-axis","fragments":[{"n":0,"t":"PERSONA","x":"Expert DBA"},{"n":3,"t":"CONSTRAINTS","x":"Must use indexes. Never full table scan."},{"n":4,"t":"FORMAT","x":"Numbered list"},{"n":5,"t":"TASK","x":"Optimize query performance"}]}'
```

### MCP Server (Claude Code Integration)

Add to your Claude Code `settings.json`:

```json
{
    "mcpServers": {
        "sinc-llm": {
            "command": "sinc-mcp",
            "args": []
        }
    }
}
```

Available MCP tools:
- `sinc_scatter` -- Decompose raw prompt into 6 bands
- `sinc_execute` -- Execute sinc JSON via Anthropic API
- `sinc_snr` -- Compute SNR quality score
- `sinc_build` -- Build sinc JSON from band arguments
- `sinc_detect` -- Detect which bands are present in raw text

## SNR Grading Scale

| Grade | SNR Range | Meaning |
|-------|-----------|---------|
| EXCELLENT | >= 0.80 | All bands populated, high-quality reconstruction |
| GOOD | >= 0.70 | Most bands present, reliable output |
| ADEQUATE | >= 0.65 | Functional but some aliasing risk |
| ALIASED | >= 0.60 | Missing critical bands, hallucination likely |
| CRITICAL | < 0.60 | Severe undersampling, unreliable output |

## The Theory

A prompt to an LLM is a signal on the *specification axis*. Like any signal, it has
frequency components:

- **Low frequencies** (PERSONA, CONTEXT): slowly-varying context that frames everything
- **Mid frequencies** (DATA, CONSTRAINTS): the specific rules and inputs
- **High frequencies** (FORMAT, TASK): the precise objective and output structure

The Nyquist-Shannon theorem states that to reconstruct a signal perfectly, you must
sample at >= 2x the highest frequency. A raw prompt like "optimize my database" is
a single sample of a 6-band signal -- 6:1 undersampling. The LLM must *hallucinate*
the missing bands (who are you? what constraints? what format?).

sinc-llm ensures all 6 bands are explicitly sampled, eliminating aliasing at the source.

## Citation

```bibtex
@software{sinc_llm,
    author = {Alexandre, Mario},
    title = {sinc-llm: Nyquist-Shannon Sampling for LLM Prompts},
    year = {2025},
    doi = {10.5281/zenodo.19152668},
    url = {https://github.com/marioalexandre/sinc-llm}
}
```

## Author

**Mario Alexandre** -- [DLux.digital](https://dlux.digital)

## Support Development

If sinc-llm saves you tokens, time, or hallucination headaches:

**[Donate at tokencalc.pro](https://tokencalc.pro/)**

## License

MIT License. See [LICENSE](LICENSE).
