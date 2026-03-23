# sinc-llm Experiments

Reproducible experiments for the sinc-LLM paper: *Nyquist-Shannon Sampling for LLM Prompts*.

DOI: [10.5281/zenodo.19152668](https://doi.org/10.5281/zenodo.19152668)

## Prerequisites

```bash
# Python 3.8+ required
python --version  # must be >= 3.8

# Install sinc-llm from the parent directory
pip install -e ..

# The only external dependency beyond sinc-llm is requests (stdlib-compatible HTTP)
pip install requests
```

### Provider Setup

All experiments support three LLM providers. You only need ONE:

| Provider | Setup | Cost |
|----------|-------|------|
| **Ollama** (recommended for reproduction) | `ollama pull llama3` | Free (local) |
| **Anthropic** | `export ANTHROPIC_API_KEY=sk-ant-...` | ~$0.50/run |
| **OpenAI** | `export OPENAI_API_KEY=sk-...` | ~$0.40/run |

Ollama requires zero API keys and runs entirely on your machine. Install from https://ollama.com.

## Experiments

### 1. Ablation Study (`run_ablation.py`)

Tests prompt quality degradation as frequency bands are removed. Measures SNR, response quality, hedging density, and specificity across 1-band through 6-band prompts on 10 diverse tasks.

```bash
# Run with Ollama (free, no API key needed)
python run_ablation.py --provider ollama --model llama3

# Run with Anthropic
python run_ablation.py --provider anthropic --model claude-haiku-4-20250514

# Run with OpenAI
python run_ablation.py --provider openai --model gpt-4o-mini

# Custom output file
python run_ablation.py --provider ollama --model llama3 --output results_ablation.json
```

**What it measures:**
- Response quality score (0.0-1.0) based on specificity, structure, and directness
- Word count (response length)
- Hedging density (fraction of hedging phrases per 100 words)
- Specificity score (presence of numbers, technical terms, concrete nouns)

**Expected results:**
- 6-band prompts score 0.78-0.85 quality (EXCELLENT SNR range)
- Removing CONSTRAINTS (n=3) causes the largest single-band quality drop (~40%)
- 1-band prompts score 0.45-0.55 quality (CRITICAL SNR range)
- Quality scales monotonically with band count in 95%+ of trials

### 2. Head-to-Head Battles (`run_battles.py`)

Compares sinc-prompt format against 5 alternative prompting techniques on 5 tasks.

```bash
# Run with Ollama
python run_battles.py --provider ollama --model llama3

# Run with Anthropic
python run_battles.py --provider anthropic --model claude-sonnet-4-20250514

# Run with OpenAI
python run_battles.py --provider openai --model gpt-4o

# Custom output
python run_battles.py --provider ollama --model llama3 --output results_battles.json
```

**Techniques compared:**
1. **Raw prompt** -- single unstructured sentence
2. **Chain-of-thought** -- "Let's think step by step" prefix
3. **Few-shot** -- 2 example input/output pairs before the task
4. **RISEN** -- Role, Instructions, Steps, End goal, Narrowing
5. **CO-STAR** -- Context, Objective, Style, Tone, Audience, Response
6. **sinc-prompt** -- 6-band Nyquist-Shannon decomposition

**Expected results:**
- sinc-prompt wins 70-85% of head-to-head comparisons
- Largest advantage over raw prompts (30-40% quality improvement)
- Closest competitor: CO-STAR (sinc wins ~60% of matchups)
- sinc-prompt produces 15-25% fewer hedging phrases than alternatives

### 3. Data Analysis (`analyze_data.py`)

Analyzes the raw Nyquist session data from 275 production observations.

```bash
# Basic analysis (prints summary tables)
python analyze_data.py

# Custom data file
python analyze_data.py --input data/nyquist_session.jsonl

# Save full analysis as JSON
python analyze_data.py --output analysis_results.json
```

**What it computes:**
- SNR per agent (mean, std, min, max)
- Token usage distribution over time
- Band importance weights (empirical vs. theoretical)
- Session duration statistics
- Agent efficiency rankings (output tokens per ms)

**Expected outputs matching the paper:**
- 275+ observations across 10+ agent types
- Mean SNR: 0.72 (GOOD range)
- CONSTRAINTS band carries 42.7% of quality weight
- Token efficiency varies 100x between agent types

## Data Files

| File | Description |
|------|-------------|
| `data/nyquist_session.jsonl` | Raw production telemetry: 871 observations with timestamp, agent, token counts, tool uses, duration, and output/input ratio |

## Reproducing Paper Results

To fully reproduce the paper's findings:

```bash
# Step 1: Run ablation (generates ablation curve data)
python run_ablation.py --provider ollama --model llama3 --output data/ablation_results.json

# Step 2: Run battles (generates comparison data)
python run_battles.py --provider ollama --model llama3 --output data/battle_results.json

# Step 3: Analyze production data
python analyze_data.py --output data/analysis_results.json

# All three produce JSON files suitable for plotting with matplotlib/seaborn
```

## Notes

- All scripts use `argparse` with `--help` for full usage details
- JSON output files are machine-readable for downstream analysis
- Console output includes formatted tables for quick inspection
- No paid API keys required when using `--provider ollama`
- Scripts depend only on `requests` and `sinc_prompt` (the parent package)
