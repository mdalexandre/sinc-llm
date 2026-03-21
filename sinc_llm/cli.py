"""
sinc-llm CLI: Command-line interface for sinc operations.

Provides three entry points:
    sinc-scatter    Decompose a raw prompt into 6 sinc frequency bands
    sinc-engine     Execute a sinc-formatted JSON prompt
    sinc-server     Start the HTTP server

Formula: x(t) = Sigma x(nT) * sinc((t - nT) / T)

Author: Mario Alexandre
License: MIT
DOI: 10.5281/zenodo.19152668
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List, Optional

from sinc_llm import __version__, __doi__
from sinc_llm.core import compute_snr, parse_sinc_json, build_sinc_json, FORMULA


def _setup_encoding() -> None:
    """Configure stdout/stderr for UTF-8 on Windows."""
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# sinc-scatter
# ---------------------------------------------------------------------------

def scatter_main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point: sinc-scatter.

    Decomposes a raw prompt into sinc JSON via Anthropic API.
    Optionally executes the structured prompt in a second pass.

    Usage:
        sinc-scatter "Find me clients for my company"
        sinc-scatter "Build a REST API" --execute
        sinc-scatter --execute --model claude-sonnet-4-20250514 < prompt.txt
        echo "raw prompt" | sinc-scatter
    """
    _setup_encoding()

    parser = argparse.ArgumentParser(
        prog="sinc-scatter",
        description="Decompose raw prompts into sinc-formatted JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Formula: {FORMULA}\n"
            f"DOI: {__doi__}\n"
            "Donate: https://tokencalc.pro/"
        ),
    )
    parser.add_argument("prompt", nargs="*", help="Raw prompt text (or pipe via stdin)")
    parser.add_argument("--execute", action="store_true", help="Also execute the scattered prompt")
    parser.add_argument("--model", default="claude-haiku-4-20250514",
                        help="Model for scatter (default: haiku)")
    parser.add_argument("--execute-model", default="claude-sonnet-4-20250514",
                        help="Model for execution (default: sonnet)")
    parser.add_argument("--api-key", default=None, help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument("--version", action="version", version=f"sinc-scatter {__version__}")

    args = parser.parse_args(argv)

    # Read prompt
    if args.prompt:
        raw_prompt = " ".join(args.prompt)
    elif not sys.stdin.isatty():
        raw_prompt = sys.stdin.read().strip()
    else:
        parser.print_help()
        sys.exit(1)

    if not raw_prompt:
        print("Error: empty prompt", file=sys.stderr)
        sys.exit(1)

    # Import scatter (requires anthropic)
    try:
        from sinc_llm.scatter import scatter, scatter_and_execute
    except ImportError:
        print(
            "Error: scatter requires the anthropic SDK.\n"
            "Install with: pip install sinc-llm[execute]",
            file=sys.stderr,
        )
        sys.exit(1)

    t0 = time.time()

    if args.execute:
        # Full pipeline: scatter + execute
        print("[SCATTER] Decomposing into 6 frequency bands...", file=sys.stderr)
        result = scatter_and_execute(
            raw_prompt,
            scatter_model=args.model,
            execute_model=args.execute_model,
            api_key=args.api_key,
        )

        if "error" in result:
            print(json.dumps(result, indent=2, ensure_ascii=False))
            sys.exit(1)

        total_time = time.time() - t0
        snr_data = result.get("snr", {})

        # Print response
        print(result.get("response", ""))
        print()
        print("--- SINC RECONSTRUCTION ---")
        print(f"SNR: {snr_data.get('snr', 'N/A')} ({snr_data.get('grade', 'N/A')})")
        print(f"Fragments: {snr_data.get('fragments', 'N/A')}")
        zones = snr_data.get("zones", {})
        print(
            f"Zones: G(Z1)={zones.get('G(Z1)', 'N/A')} "
            f"H(Z2)={zones.get('H(Z2)', 'N/A')} "
            f"R(Z3)={zones.get('R(Z3)', 'N/A')} "
            f"G(Z4)={zones.get('G(Z4)', 'N/A')}"
        )
        print(f"Model: {result.get('model', 'N/A')}")
        usage = result.get("usage", {})
        print(f"Tokens: {usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out")
        print(f"Elapsed: {total_time:.1f}s")
    else:
        # Scatter only
        print("[SCATTER] Decomposing into 6 frequency bands...", file=sys.stderr)
        sinc_json = scatter(raw_prompt, model=args.model, api_key=args.api_key)

        if "error" in sinc_json:
            print(json.dumps(sinc_json, indent=2, ensure_ascii=False))
            sys.exit(1)

        snr_data = compute_snr(sinc_json)
        scatter_time = time.time() - t0

        # Print sinc JSON to stdout
        print(json.dumps(sinc_json, indent=2, ensure_ascii=False))

        # Print metrics to stderr
        print(f"\n--- SCATTER COMPLETE ---", file=sys.stderr)
        print(f"SNR: {snr_data['snr']} ({snr_data['grade']})", file=sys.stderr)
        print(f"Fragments: {snr_data['fragments']}", file=sys.stderr)
        zones = snr_data["zones"]
        print(
            f"Zones: G(Z1)={zones['G(Z1)']} H(Z2)={zones['H(Z2)']} "
            f"R(Z3)={zones['R(Z3)']} G(Z4)={zones['G(Z4)']}",
            file=sys.stderr,
        )
        print(f"Elapsed: {scatter_time:.1f}s", file=sys.stderr)


# ---------------------------------------------------------------------------
# sinc-engine
# ---------------------------------------------------------------------------

def engine_main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point: sinc-engine.

    Validates and optionally executes a sinc-formatted JSON prompt.

    Usage:
        sinc-engine prompt.json
        sinc-engine --dry-run < prompt.json
        cat prompt.json | sinc-engine --model claude-sonnet-4-20250514
    """
    _setup_encoding()

    parser = argparse.ArgumentParser(
        prog="sinc-engine",
        description="Execute sinc-formatted JSON prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Formula: {FORMULA}\n"
            f"DOI: {__doi__}\n"
            "Donate: https://tokencalc.pro/"
        ),
    )
    parser.add_argument("file", nargs="?", help="Sinc JSON file (or pipe via stdin)")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, do not execute")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Model for execution")
    parser.add_argument("--api-key", default=None, help="Anthropic API key")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max response tokens")
    parser.add_argument("--version", action="version", version=f"sinc-engine {__version__}")

    args = parser.parse_args(argv)

    # Read input
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            print(f"Error reading {args.file}: {e}", file=sys.stderr)
            sys.exit(1)
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.print_help()
        sys.exit(1)

    try:
        sinc_json = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse and validate
    parsed = parse_sinc_json(sinc_json)
    snr_data = compute_snr(sinc_json)

    if args.dry_run or not parsed["valid"]:
        # Validation report
        report = {
            "valid": parsed["valid"],
            "snr": round(parsed["snr"], 4),
            "snr_grade": parsed["metadata"]["snr_grade"],
            "fragment_coverage": f"{parsed['metadata']['fragment_count']}/6",
            "total_tokens": parsed["metadata"]["total_tokens"],
            "zone_tokens": parsed["z_tokens"],
            "zone_scores": {
                "G(Z1)": snr_data["zones"]["G(Z1)"],
                "H(Z2)": snr_data["zones"]["H(Z2)"],
                "R(Z3)": snr_data["zones"]["R(Z3)"],
                "G(Z4)": snr_data["zones"]["G(Z4)"],
            },
            "warnings": parsed["warnings"],
            "errors": parsed["errors"],
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if not parsed["valid"]:
            sys.exit(1)
        return

    # Execute via Anthropic API
    try:
        import anthropic
    except ImportError:
        print(
            "Error: execution requires the anthropic SDK.\n"
            "Install with: pip install sinc-llm[execute]",
            file=sys.stderr,
        )
        sys.exit(1)

    import os
    key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("Error: no API key. Set ANTHROPIC_API_KEY or pass --api-key.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=key)

    t0 = time.time()
    try:
        response = client.messages.create(
            model=args.model,
            max_tokens=args.max_tokens,
            system=parsed["system_prompt"],
            messages=[{"role": "user", "content": parsed["user_prompt"]}],
        )
    except Exception as e:
        print(f"Error: Anthropic API: {e}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - t0

    response_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            response_text += block.text

    # Output
    print(response_text)
    print()
    print("--- SINC RECONSTRUCTION ---")
    print(f"SNR: {snr_data['snr']} ({snr_data['grade']})")
    print(f"Fragments: {snr_data['fragments']}")
    zones = snr_data["zones"]
    print(
        f"Zones: G(Z1)={zones['G(Z1)']} H(Z2)={zones['H(Z2)']} "
        f"R(Z3)={zones['R(Z3)']} G(Z4)={zones['G(Z4)']}"
    )
    print(f"Model: {args.model}")
    print(f"Tokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out")
    print(f"Elapsed: {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# sinc-server (delegates)
# ---------------------------------------------------------------------------

def server_main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point: sinc-server. Delegates to server.main()."""
    from sinc_llm.server import main as _server_main
    _server_main()


# ---------------------------------------------------------------------------
# Unified CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """Unified sinc-llm CLI entry point.

    Usage:
        sinc-llm scatter "your prompt"
        sinc-llm engine prompt.json
        sinc-llm server --port 8461
        sinc-llm snr prompt.json
        sinc-llm build --task "objective" --constraints "rules" ...
    """
    _setup_encoding()

    parser = argparse.ArgumentParser(
        prog="sinc-llm",
        description="Nyquist-Shannon sampling for LLM prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Formula: {FORMULA}\n"
            f"DOI: {__doi__}\n"
            "Donate: https://tokencalc.pro/"
        ),
    )
    parser.add_argument("--version", action="version", version=f"sinc-llm {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # scatter
    sp_scatter = subparsers.add_parser("scatter", help="Decompose raw prompt into sinc JSON")
    sp_scatter.add_argument("prompt", nargs="*")
    sp_scatter.add_argument("--execute", action="store_true")
    sp_scatter.add_argument("--model", default="claude-haiku-4-20250514")
    sp_scatter.add_argument("--execute-model", default="claude-sonnet-4-20250514")
    sp_scatter.add_argument("--api-key", default=None)

    # engine
    sp_engine = subparsers.add_parser("engine", help="Execute sinc JSON prompt")
    sp_engine.add_argument("file", nargs="?")
    sp_engine.add_argument("--dry-run", action="store_true")
    sp_engine.add_argument("--model", default="claude-sonnet-4-20250514")
    sp_engine.add_argument("--api-key", default=None)
    sp_engine.add_argument("--max-tokens", type=int, default=8192)

    # server
    sp_server = subparsers.add_parser("server", help="Start HTTP server")
    sp_server.add_argument("--host", default="127.0.0.1")
    sp_server.add_argument("--port", type=int, default=8461)

    # snr
    sp_snr = subparsers.add_parser("snr", help="Compute SNR for sinc JSON")
    sp_snr.add_argument("file", nargs="?", help="Sinc JSON file (or stdin)")

    # build
    sp_build = subparsers.add_parser("build", help="Build sinc JSON from band arguments")
    sp_build.add_argument("--persona", default="", help="n=0 PERSONA band")
    sp_build.add_argument("--context", default="", help="n=1 CONTEXT band")
    sp_build.add_argument("--data", default="", help="n=2 DATA band")
    sp_build.add_argument("--constraints", default="", help="n=3 CONSTRAINTS band")
    sp_build.add_argument("--format", default="", dest="fmt", help="n=4 FORMAT band")
    sp_build.add_argument("--task", default="", help="n=5 TASK band")

    args = parser.parse_args(argv)

    if args.command == "scatter":
        # Rebuild argv for scatter_main
        scatter_argv = list(args.prompt or [])
        if args.execute:
            scatter_argv.append("--execute")
        if args.model != "claude-haiku-4-20250514":
            scatter_argv.extend(["--model", args.model])
        if args.execute_model != "claude-sonnet-4-20250514":
            scatter_argv.extend(["--execute-model", args.execute_model])
        if args.api_key:
            scatter_argv.extend(["--api-key", args.api_key])
        scatter_main(scatter_argv)

    elif args.command == "engine":
        engine_argv = []
        if args.file:
            engine_argv.append(args.file)
        if args.dry_run:
            engine_argv.append("--dry-run")
        if args.model != "claude-sonnet-4-20250514":
            engine_argv.extend(["--model", args.model])
        if args.api_key:
            engine_argv.extend(["--api-key", args.api_key])
        if args.max_tokens != 8192:
            engine_argv.extend(["--max-tokens", str(args.max_tokens)])
        engine_main(engine_argv)

    elif args.command == "server":
        from sinc_llm.server import serve
        serve(host=args.host, port=args.port)

    elif args.command == "snr":
        # Quick SNR computation
        if args.file:
            try:
                with open(args.file, "r", encoding="utf-8") as f:
                    text = f.read()
            except OSError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
        elif not sys.stdin.isatty():
            text = sys.stdin.read()
        else:
            sp_snr.print_help()
            sys.exit(1)

        try:
            sinc_json = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"Error: invalid JSON: {e}", file=sys.stderr)
            sys.exit(1)

        snr_data = compute_snr(sinc_json)
        print(json.dumps(snr_data, indent=2, ensure_ascii=False))

    elif args.command == "build":
        sinc_json = build_sinc_json(
            persona=args.persona,
            context=args.context,
            data=args.data,
            constraints=args.constraints,
            fmt=args.fmt,
            task=args.task,
        )
        snr_data = compute_snr(sinc_json)
        sinc_json["_snr"] = snr_data
        print(json.dumps(sinc_json, indent=2, ensure_ascii=False))

    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
