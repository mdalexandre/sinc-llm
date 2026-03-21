"""
sinc-llm MCP server: Model Context Protocol server for Claude Code integration.

Implements the MCP JSON-RPC protocol over stdin/stdout, providing three tools:
    sinc_scatter  -- Decompose raw prompt into 6 sinc frequency bands
    sinc_execute  -- Execute a sinc-formatted JSON prompt
    sinc_snr      -- Compute SNR quality score for sinc JSON

Formula: x(t) = Sigma x(nT) * sinc((t - nT) / T)

Usage in Claude Code settings.json:
    {
        "mcpServers": {
            "sinc-llm": {
                "command": "sinc-mcp",
                "args": []
            }
        }
    }

Or run directly:
    python -m sinc_llm.mcp_server

Author: Mario Alexandre
License: MIT
DOI: 10.5281/zenodo.19152668
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

from sinc_llm import __version__, __doi__
from sinc_llm.core import (
    compute_snr,
    parse_sinc_json,
    build_sinc_json,
    detect_fragments,
    FORMULA,
)

# ---------------------------------------------------------------------------
# MCP Protocol Constants
# ---------------------------------------------------------------------------

JSONRPC_VERSION = "2.0"
MCP_PROTOCOL_VERSION = "2024-11-05"

SERVER_INFO = {
    "name": "sinc-llm",
    "version": __version__,
}

SERVER_CAPABILITIES = {
    "tools": {},
}

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "sinc_scatter",
        "description": (
            "Decompose a raw prompt into 6 Nyquist-compliant frequency bands "
            "(PERSONA, CONTEXT, DATA, CONSTRAINTS, FORMAT, TASK). "
            "Returns sinc JSON with SNR quality score. "
            "Requires ANTHROPIC_API_KEY environment variable."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The raw prompt text to decompose into sinc format.",
                },
                "model": {
                    "type": "string",
                    "description": "Anthropic model for decomposition (default: claude-haiku-4-20250514).",
                    "default": "claude-haiku-4-20250514",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "sinc_execute",
        "description": (
            "Execute a sinc-formatted JSON prompt via Anthropic API. "
            "Takes a complete sinc JSON structure (with formula, T, fragments) "
            "and sends it to Claude for execution. "
            "Requires ANTHROPIC_API_KEY environment variable."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sinc_json": {
                    "type": "object",
                    "description": "Complete sinc JSON with formula, T, and fragments array.",
                },
                "model": {
                    "type": "string",
                    "description": "Anthropic model for execution (default: claude-sonnet-4-20250514).",
                    "default": "claude-sonnet-4-20250514",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum response tokens (default: 8192).",
                    "default": 8192,
                },
            },
            "required": ["sinc_json"],
        },
    },
    {
        "name": "sinc_snr",
        "description": (
            "Compute Signal-to-Noise Ratio for a sinc-formatted JSON prompt. "
            "Returns SNR value, grade (EXCELLENT/GOOD/ADEQUATE/ALIASED/CRITICAL), "
            "zone function scores G(Z1), H(Z2), R(Z3), G(Z4), and token counts. "
            "Zero dependencies -- works offline."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sinc_json": {
                    "type": "object",
                    "description": "Sinc JSON with fragments array to analyze.",
                },
            },
            "required": ["sinc_json"],
        },
    },
    {
        "name": "sinc_build",
        "description": (
            "Build a sinc JSON structure from individual band contents. "
            "Convenience tool to construct the standard sinc prompt format. "
            "Returns the sinc JSON with computed SNR."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "persona": {"type": "string", "description": "n=0 PERSONA: who answers", "default": ""},
                "context": {"type": "string", "description": "n=1 CONTEXT: situation, facts", "default": ""},
                "data": {"type": "string", "description": "n=2 DATA: specific inputs", "default": ""},
                "constraints": {"type": "string", "description": "n=3 CONSTRAINTS: rules (42.7% of quality)", "default": ""},
                "format": {"type": "string", "description": "n=4 FORMAT: output structure", "default": ""},
                "task": {"type": "string", "description": "n=5 TASK: the objective", "default": ""},
            },
            "required": ["task"],
        },
    },
    {
        "name": "sinc_detect",
        "description": (
            "Detect which sinc frequency bands are present in a raw prompt. "
            "Returns boolean presence for each of the 6 bands. "
            "Zero dependencies -- works offline."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Raw prompt text to analyze for fragment presence.",
                },
            },
            "required": ["prompt"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def handle_sinc_scatter(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle sinc_scatter tool call."""
    prompt = arguments.get("prompt", "")
    model = arguments.get("model", "claude-haiku-4-20250514")

    if not prompt:
        return _error_content("Empty prompt")

    try:
        from sinc_llm.scatter import scatter
    except ImportError:
        return _error_content(
            "Scatter requires the anthropic SDK. Install with: pip install sinc-llm[execute]"
        )

    sinc_json = scatter(prompt, model=model)

    if "error" in sinc_json:
        return _error_content(sinc_json["error"])

    snr_data = compute_snr(sinc_json)
    sinc_json["_snr"] = snr_data

    return _text_content(json.dumps(sinc_json, indent=2, ensure_ascii=False))


def handle_sinc_execute(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle sinc_execute tool call."""
    sinc_json = arguments.get("sinc_json", {})
    model = arguments.get("model", "claude-sonnet-4-20250514")
    max_tokens = arguments.get("max_tokens", 8192)

    if not sinc_json or "fragments" not in sinc_json:
        return _error_content("Invalid sinc JSON: missing fragments array")

    parsed = parse_sinc_json(sinc_json)

    if not parsed["valid"]:
        return _error_content(f"Invalid sinc JSON: {'; '.join(parsed['errors'])}")

    try:
        import os
        import anthropic
    except ImportError:
        return _error_content(
            "Execute requires the anthropic SDK. Install with: pip install sinc-llm[execute]"
        )

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return _error_content("No API key. Set ANTHROPIC_API_KEY environment variable.")

    client = anthropic.Anthropic(api_key=key)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=parsed["system_prompt"],
            messages=[{"role": "user", "content": parsed["user_prompt"]}],
        )
    except Exception as e:
        return _error_content(f"Anthropic API error: {e}")

    response_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            response_text += block.text

    snr_data = compute_snr(sinc_json)

    result = {
        "response": response_text,
        "snr": snr_data,
        "model": model,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }

    return _text_content(json.dumps(result, indent=2, ensure_ascii=False))


def handle_sinc_snr(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle sinc_snr tool call."""
    sinc_json = arguments.get("sinc_json", {})

    if not sinc_json:
        return _error_content("Empty sinc_json")

    snr_data = compute_snr(sinc_json)
    parsed = parse_sinc_json(sinc_json)

    result = {
        **snr_data,
        "valid": parsed["valid"],
        "errors": parsed["errors"],
        "warnings": parsed["warnings"],
        "metadata": parsed["metadata"],
    }

    return _text_content(json.dumps(result, indent=2, ensure_ascii=False))


def handle_sinc_build(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle sinc_build tool call."""
    sinc_json = build_sinc_json(
        persona=arguments.get("persona", ""),
        context=arguments.get("context", ""),
        data=arguments.get("data", ""),
        constraints=arguments.get("constraints", ""),
        fmt=arguments.get("format", ""),
        task=arguments.get("task", ""),
    )

    snr_data = compute_snr(sinc_json)
    sinc_json["_snr"] = snr_data

    return _text_content(json.dumps(sinc_json, indent=2, ensure_ascii=False))


def handle_sinc_detect(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle sinc_detect tool call."""
    prompt = arguments.get("prompt", "")

    if not prompt:
        return _error_content("Empty prompt")

    fragments = detect_fragments(prompt)
    present_count = sum(1 for v in fragments.values() if v)

    result = {
        "fragments": fragments,
        "present": present_count,
        "total": 6,
        "coverage": f"{present_count}/6",
        "aliased": present_count < 4,
    }

    return _text_content(json.dumps(result, indent=2, ensure_ascii=False))


# Tool dispatch table
TOOL_HANDLERS = {
    "sinc_scatter": handle_sinc_scatter,
    "sinc_execute": handle_sinc_execute,
    "sinc_snr": handle_sinc_snr,
    "sinc_build": handle_sinc_build,
    "sinc_detect": handle_sinc_detect,
}


# ---------------------------------------------------------------------------
# MCP response helpers
# ---------------------------------------------------------------------------

def _text_content(text: str) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _error_content(message: str) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": f"Error: {message}"}], "isError": True}


def _jsonrpc_response(id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": id, "result": result}


def _jsonrpc_error(id: Any, code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": id,
        "error": {"code": code, "message": message},
    }


# ---------------------------------------------------------------------------
# MCP message handling
# ---------------------------------------------------------------------------

def handle_message(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handle a single MCP JSON-RPC message.

    Returns a response dict, or None for notifications (no id).
    """
    method = msg.get("method", "")
    msg_id = msg.get("id")
    params = msg.get("params", {})

    # Notifications (no id) -- no response required
    if msg_id is None:
        return None

    if method == "initialize":
        return _jsonrpc_response(msg_id, {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": SERVER_CAPABILITIES,
            "serverInfo": SERVER_INFO,
        })

    elif method == "tools/list":
        return _jsonrpc_response(msg_id, {"tools": TOOLS})

    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        handler = TOOL_HANDLERS.get(tool_name)
        if handler is None:
            return _jsonrpc_error(msg_id, -32601, f"Unknown tool: {tool_name}")

        try:
            result = handler(arguments)
        except Exception as e:
            result = _error_content(f"Tool execution error: {e}")

        return _jsonrpc_response(msg_id, result)

    elif method == "ping":
        return _jsonrpc_response(msg_id, {})

    else:
        return _jsonrpc_error(msg_id, -32601, f"Method not found: {method}")


# ---------------------------------------------------------------------------
# Stdio transport
# ---------------------------------------------------------------------------

def run_stdio() -> None:
    """Run the MCP server over stdin/stdout JSON-RPC.

    Reads one JSON-RPC message per line from stdin, writes responses
    to stdout. Designed for Claude Code integration.
    """
    if sys.platform == "win32":
        # Ensure binary-safe I/O on Windows
        import msvcrt
        import os
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        try:
            # Read Content-Length header or raw line
            line = stdin.readline()
            if not line:
                break  # EOF

            line_str = line.decode("utf-8", errors="replace").strip()

            # Handle Content-Length framing (LSP-style)
            if line_str.startswith("Content-Length:"):
                content_length = int(line_str.split(":")[1].strip())
                # Read blank separator line
                stdin.readline()
                # Read exact content
                content = stdin.read(content_length)
                line_str = content.decode("utf-8", errors="replace").strip()
            elif not line_str:
                continue

            # Parse JSON-RPC message
            try:
                msg = json.loads(line_str)
            except json.JSONDecodeError:
                continue

            # Handle message
            response = handle_message(msg)

            if response is not None:
                response_bytes = json.dumps(response, ensure_ascii=False).encode("utf-8")
                header = f"Content-Length: {len(response_bytes)}\r\n\r\n".encode("utf-8")
                stdout.write(header)
                stdout.write(response_bytes)
                stdout.flush()

        except (BrokenPipeError, ConnectionResetError):
            break
        except Exception:
            # Never crash the MCP server on unexpected errors
            continue


def main() -> None:
    """Entry point for sinc-mcp command."""
    run_stdio()


if __name__ == "__main__":
    main()
