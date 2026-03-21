"""
sinc-llm: Nyquist-Shannon Sampling for LLM Prompts.

Decomposes raw prompts into 6 frequency bands on the specification axis,
computes Signal-to-Noise Ratio, and reconstructs with formal quality guarantees.

Formula: x(t) = Sigma x(nT) * sinc((t - nT) / T)

    n=0  PERSONA      (f0 -- lowest frequency, WHO answers)
    n=1  CONTEXT      (f1 -- situation, facts, dates)
    n=2  DATA         (f2 -- specific inputs)
    n=3  CONSTRAINTS  (f3 -- rules, 42.7% of quality)
    n=4  FORMAT       (f4 -- output structure)
    n=5  TASK         (f5 -- highest frequency, the objective)

Author: Mario Alexandre
License: MIT
DOI: 10.5281/zenodo.19152668
Donate: https://tokencalc.pro/
"""

__version__ = "0.1.0"
__author__ = "Mario Alexandre"
__license__ = "MIT"
__doi__ = "10.5281/zenodo.19152668"

FORMULA = "x(t) = Sigma x(nT) * sinc((t - nT) / T)"

from sinc_llm.core import (
    Fragment,
    FRAGMENT_AXIS,
    BETA0,
    AMPLITUDE,
    CEILING,
    detect_fragments,
    compute_snr,
    compute_snr_from_tokens,
    build_sinc_json,
    parse_sinc_json,
    estimate_tokens,
    grade_snr,
    G,
    H,
    R,
)

__all__ = [
    "Fragment",
    "FRAGMENT_AXIS",
    "BETA0",
    "AMPLITUDE",
    "CEILING",
    "FORMULA",
    "detect_fragments",
    "compute_snr",
    "compute_snr_from_tokens",
    "build_sinc_json",
    "parse_sinc_json",
    "estimate_tokens",
    "grade_snr",
    "G",
    "H",
    "R",
    "__version__",
    "__author__",
    "__doi__",
]
