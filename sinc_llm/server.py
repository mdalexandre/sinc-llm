"""
sinc-llm server: HTTP API for sinc-LLM operations.

Provides REST endpoints for scatter, execute, SNR computation, and health.
Uses only Python stdlib (http.server) -- no FastAPI or Flask required.

Formula: x(t) = Sigma x(nT) * sinc((t - nT) / T)

Endpoints:
    POST /scatter    Raw text body -> sinc JSON
    POST /execute    Raw text body -> sinc JSON -> LLM -> response
    POST /snr        Sinc JSON body -> SNR report
    GET  /health     Health check + engine info

Usage:
    sinc-server --port 8461
    python -m sinc_llm.server --port 8461

Author: Mario Alexandre
License: MIT
DOI: 10.5281/zenodo.19152668
Donate: https://tokencalc.pro/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict

from sinc_llm import __version__, __doi__
from sinc_llm.core import compute_snr, parse_sinc_json, FORMULA

# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class SincHandler(BaseHTTPRequestHandler):
    """HTTP request handler for sinc-LLM operations."""

    server_version = f"sinc-llm/{__version__}"

    def log_message(self, fmt: str, *args: Any) -> None:
        """Log to stderr with timestamp."""
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        sys.stderr.write(f"[{ts}] {fmt % args}\n")

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def _respond_json(self, data: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _respond_error(self, message: str, status: int = 400) -> None:
        self._respond_json({"error": message}, status)

    # -- POST endpoints --

    def do_POST(self) -> None:
        raw = self._read_body()
        try:
            body_text = raw.decode("utf-8")
        except UnicodeDecodeError:
            self._respond_error("Request body must be UTF-8", 400)
            return

        if self.path == "/scatter":
            self._handle_scatter(body_text)
        elif self.path == "/execute":
            self._handle_execute(body_text)
        elif self.path == "/snr":
            self._handle_snr(body_text)
        elif self.path == "/reconstruct":
            self._handle_reconstruct(body_text)
        else:
            self._respond_error(f"Unknown endpoint: {self.path}", 404)

    def _handle_scatter(self, body: str) -> None:
        """POST /scatter -- Raw prompt text -> sinc JSON."""
        if not body.strip():
            self._respond_error("Empty request body")
            return

        try:
            from sinc_llm.scatter import scatter
        except ImportError:
            self._respond_error(
                "Scatter requires the anthropic SDK. "
                "Install with: pip install sinc-llm[execute]",
                501,
            )
            return

        model = self.headers.get("X-Model", "claude-haiku-4-20250514")
        api_key = self.headers.get("X-Api-Key")

        t0 = time.time()
        sinc_json = scatter(body, model=model, api_key=api_key)
        elapsed = time.time() - t0

        if "error" not in sinc_json:
            sinc_json["_snr"] = compute_snr(sinc_json)
            sinc_json["_elapsed_ms"] = round(elapsed * 1000, 1)

        self._respond_json(sinc_json)

    def _handle_execute(self, body: str) -> None:
        """POST /execute -- Raw prompt text -> scatter -> execute -> response."""
        if not body.strip():
            self._respond_error("Empty request body")
            return

        try:
            from sinc_llm.scatter import scatter_and_execute
        except ImportError:
            self._respond_error(
                "Execute requires the anthropic SDK. "
                "Install with: pip install sinc-llm[execute]",
                501,
            )
            return

        scatter_model = self.headers.get("X-Scatter-Model", "claude-haiku-4-20250514")
        execute_model = self.headers.get("X-Execute-Model", "claude-sonnet-4-20250514")
        api_key = self.headers.get("X-Api-Key")

        t0 = time.time()
        result = scatter_and_execute(
            body,
            scatter_model=scatter_model,
            execute_model=execute_model,
            api_key=api_key,
        )
        result["_elapsed_ms"] = round((time.time() - t0) * 1000, 1)

        status = 200 if "error" not in result else 502
        self._respond_json(result, status)

    def _handle_snr(self, body: str) -> None:
        """POST /snr -- Sinc JSON -> SNR quality report."""
        try:
            sinc_json = json.loads(body)
        except json.JSONDecodeError as e:
            self._respond_error(f"Invalid JSON: {e}")
            return

        snr_report = compute_snr(sinc_json)
        parsed = parse_sinc_json(sinc_json)
        snr_report["valid"] = parsed["valid"]
        snr_report["errors"] = parsed["errors"]
        snr_report["warnings"] = parsed["warnings"]
        self._respond_json(snr_report)

    def _handle_reconstruct(self, body: str) -> None:
        """POST /reconstruct -- Sinc JSON -> execute (no scatter)."""
        try:
            sinc_json = json.loads(body)
        except json.JSONDecodeError as e:
            self._respond_error(f"Invalid JSON: {e}")
            return

        try:
            from sinc_llm.scatter import scatter_and_execute
            import anthropic
        except ImportError:
            self._respond_error(
                "Reconstruct requires the anthropic SDK. "
                "Install with: pip install sinc-llm[execute]",
                501,
            )
            return

        parsed = parse_sinc_json(sinc_json)
        if not parsed["valid"]:
            self._respond_error(
                f"Invalid sinc JSON: {'; '.join(parsed['errors'])}", 400
            )
            return

        model = self.headers.get("X-Model", "claude-sonnet-4-20250514")
        api_key = self.headers.get("X-Api-Key")
        import os
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

        if not key:
            self._respond_error("No API key. Set ANTHROPIC_API_KEY or send X-Api-Key header.")
            return

        client = anthropic.Anthropic(api_key=key)
        max_tokens = int(self.headers.get("X-Max-Tokens", "8192"))

        t0 = time.time()
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=parsed["system_prompt"],
                messages=[{"role": "user", "content": parsed["user_prompt"]}],
            )
        except Exception as e:
            self._respond_error(f"Anthropic API error: {e}", 502)
            return

        response_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                response_text += block.text

        snr_data = compute_snr(sinc_json)
        self._respond_json({
            "response": response_text,
            "snr": snr_data,
            "model": model,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "_elapsed_ms": round((time.time() - t0) * 1000, 1),
        })

    # -- GET endpoints --

    def do_GET(self) -> None:
        if self.path == "/health":
            self._handle_health()
        else:
            self._respond_json({
                "engine": f"sinc-llm v{__version__}",
                "formula": FORMULA,
                "doi": __doi__,
                "donate": "https://tokencalc.pro/",
                "endpoints": {
                    "POST /scatter": "Raw text -> sinc JSON (decomposition only)",
                    "POST /execute": "Raw text -> sinc JSON -> LLM -> response (full pipeline)",
                    "POST /snr": "Sinc JSON -> SNR quality report",
                    "POST /reconstruct": "Sinc JSON -> LLM -> response (no scatter)",
                    "GET /health": "Health check and engine info",
                },
            })

    def _handle_health(self) -> None:
        """GET /health -- Health check with engine info."""
        import os

        has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY", ""))

        try:
            import anthropic
            has_sdk = True
        except ImportError:
            has_sdk = False

        self._respond_json({
            "status": "healthy",
            "engine": f"sinc-llm v{__version__}",
            "formula": FORMULA,
            "doi": __doi__,
            "donate": "https://tokencalc.pro/",
            "capabilities": {
                "scatter": has_sdk,
                "execute": has_sdk and has_api_key,
                "snr": True,
                "reconstruct": has_sdk and has_api_key,
            },
            "api_key_set": has_api_key,
            "anthropic_sdk": has_sdk,
        })

    # -- OPTIONS (CORS preflight) --

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, X-Model, X-Api-Key, X-Scatter-Model, X-Execute-Model, X-Max-Tokens",
        )
        self.end_headers()


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------

def serve(host: str = "127.0.0.1", port: int = 8461) -> None:
    """Start the sinc-LLM HTTP server.

    Args:
        host: Bind address (default localhost).
        port: Bind port (default 8461).
    """
    server = HTTPServer((host, port), SincHandler)

    print(f"sinc-llm v{__version__} HTTP Server")
    print(f"  Formula: {FORMULA}")
    print(f"  DOI: {__doi__}")
    print()
    print(f"  POST /scatter      Raw prompt -> sinc JSON")
    print(f"  POST /execute      Raw prompt -> sinc JSON -> LLM -> response")
    print(f"  POST /snr          Sinc JSON -> SNR quality report")
    print(f"  POST /reconstruct  Sinc JSON -> LLM -> response")
    print(f"  GET  /health       Health check")
    print()
    print(f"  Listening on http://{host}:{port}")
    print(f"  Donate: https://tokencalc.pro/")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


def main() -> None:
    """CLI entry point for sinc-server."""
    parser = argparse.ArgumentParser(
        description="sinc-llm HTTP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Formula: x(t) = Sigma x(nT) * sinc((t - nT) / T)\n"
            "DOI: 10.5281/zenodo.19152668\n"
            "Donate: https://tokencalc.pro/"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8461, help="Bind port (default: 8461)")

    args = parser.parse_args()
    serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
