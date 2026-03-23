"""
sinc-llm providers: Multi-provider LLM abstraction layer.

Supports four providers with zero external SDK dependencies (uses only
stdlib urllib for HTTP calls):

    AnthropicProvider  -- Anthropic Messages API (Claude models)
    OpenAIProvider     -- OpenAI Chat Completions API (GPT models)
    OllamaProvider     -- Ollama local API at localhost:11434 (zero cost)
    VLLMProvider       -- vLLM OpenAI-compatible endpoint at localhost:8000

All providers implement: generate(prompt: str, model: str) -> str

Formula: x(t) = Sigma x(nT) * sinc((t - nT) / T)

Author: Mario Alexandre
License: MIT
DOI: 10.5281/zenodo.19152668
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseProvider:
    """Abstract base for LLM providers."""

    name: str = "base"

    def generate(self, prompt: str, model: str) -> str:
        """Generate a response from a prompt.

        Args:
            prompt: The input text prompt.
            model: Model identifier (provider-specific).

        Returns:
            Response text. On error, returns a string starting with
            "[{PROVIDER}_ERROR]".
        """
        raise NotImplementedError

    def _post_json(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        timeout: int = 120,
    ) -> Dict[str, Any]:
        """Send a POST request with JSON body and return parsed response.

        Uses only stdlib urllib -- no requests or httpx dependency.

        Raises:
            RuntimeError: On HTTP or parse errors.
        """
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")[:500]
            except Exception:
                pass
            raise RuntimeError(f"HTTP {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Connection error: {e.reason}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response: {e}") from e


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicProvider(BaseProvider):
    """Anthropic Messages API provider.

    Uses ANTHROPIC_API_KEY from environment or constructor parameter.
    Endpoint: https://api.anthropic.com/v1/messages

    Default model: claude-haiku-4-20250514
    """

    name = "anthropic"

    def __init__(self, api_key: Optional[str] = None, max_tokens: int = 4096):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.max_tokens = max_tokens
        self.api_url = "https://api.anthropic.com/v1/messages"

    def generate(self, prompt: str, model: str = "claude-haiku-4-20250514") -> str:
        """Generate via Anthropic Messages API."""
        if not self.api_key:
            return "[ANTHROPIC_ERROR] No API key. Set ANTHROPIC_API_KEY or pass api_key=."

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            data = self._post_json(self.api_url, payload, headers)
            blocks = data.get("content", [])
            return "".join(
                b.get("text", "") for b in blocks if b.get("type") == "text"
            )
        except RuntimeError as e:
            return f"[ANTHROPIC_ERROR] {e}"


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIProvider(BaseProvider):
    """OpenAI Chat Completions API provider.

    Uses OPENAI_API_KEY from environment or constructor parameter.
    Endpoint: https://api.openai.com/v1/chat/completions

    Default model: gpt-4o-mini
    """

    name = "openai"

    def __init__(self, api_key: Optional[str] = None, max_tokens: int = 4096):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.max_tokens = max_tokens
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def generate(self, prompt: str, model: str = "gpt-4o-mini") -> str:
        """Generate via OpenAI Chat Completions API."""
        if not self.api_key:
            return "[OPENAI_ERROR] No API key. Set OPENAI_API_KEY or pass api_key=."

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0.3,
        }

        try:
            data = self._post_json(self.api_url, payload, headers)
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""
        except RuntimeError as e:
            return f"[OPENAI_ERROR] {e}"


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

class OllamaProvider(BaseProvider):
    """Ollama local API provider.

    Calls localhost:11434/api/generate. Zero API keys, zero cost.
    Requires Ollama running locally: https://ollama.com

    Default model: llama3
    """

    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, model: str = "llama3") -> str:
        """Generate via Ollama HTTP API."""
        url = f"{self.base_url}/api/generate"

        headers = {"Content-Type": "application/json"}

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 4096,
            },
        }

        try:
            data = self._post_json(url, payload, headers, timeout=self.timeout)
            return data.get("response", "")
        except RuntimeError as e:
            return f"[OLLAMA_ERROR] {e}"

    def list_models(self) -> list:
        """List available models in the local Ollama instance."""
        url = f"{self.base_url}/api/tags"
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------

class VLLMProvider(BaseProvider):
    """vLLM OpenAI-compatible endpoint provider.

    Calls an OpenAI-compatible API at localhost:8000 (default vLLM port).
    vLLM serves models with the OpenAI Chat Completions format.

    Default model: whatever is loaded in the vLLM instance.
    """

    name = "vllm"

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, model: str = "default") -> str:
        """Generate via vLLM OpenAI-compatible endpoint."""
        url = f"{self.base_url}/v1/chat/completions"

        headers = {"Content-Type": "application/json"}

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "temperature": 0.3,
        }

        try:
            data = self._post_json(url, payload, headers, timeout=self.timeout)
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""
        except RuntimeError as e:
            return f"[VLLM_ERROR] {e}"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

PROVIDERS = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
    "vllm": VLLMProvider,
}


def get_provider(name: str, **kwargs: Any) -> BaseProvider:
    """Get a provider instance by name.

    Args:
        name: Provider name (anthropic, openai, ollama, vllm).
        **kwargs: Provider-specific constructor arguments.

    Returns:
        Provider instance.

    Raises:
        ValueError: If provider name is not recognized.
    """
    cls = PROVIDERS.get(name.lower())
    if cls is None:
        valid = ", ".join(sorted(PROVIDERS.keys()))
        raise ValueError(f"Unknown provider: {name!r}. Valid providers: {valid}")
    return cls(**kwargs)
