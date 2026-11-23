"""LLM provider abstraction.

Business code depends only on the LLMProvider protocol (one method:
generate). Which backend actually runs is decided by the LLM_PROVIDER
setting. This avoids vendor lock-in and makes the model swappable in tests.
"""

from typing import Protocol

import httpx

from bankdoc.core.config import get_settings


class LLMProvider(Protocol):
    """Anything with a generate(system, user) -> str method is an LLM."""

    def generate(self, system: str, user: str) -> str: ...


class OllamaProvider:
    """Talks to a local Ollama server over its HTTP API."""

    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, system: str, user: str) -> str:
        """Send one chat request and return the model's text reply."""
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                # temperature 0 = (near) deterministic, which makes evaluation reproducible.
                # num_ctx is the context window in tokens; Ollama's default is too small for RAG.
                "options": {"temperature": 0, "num_ctx": 8192},
            },
            timeout=self.timeout,  
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


class OpenAIProvider:
    """Uses the OpenAI Chat Completions API."""

    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI  # imported lazily: not needed for other providers

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


class AnthropicProvider:
    """Uses the Anthropic Messages API."""

    def __init__(self, api_key: str, model: str) -> None:
        from anthropic import Anthropic  # imported lazily

        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # The reply is a list of content blocks; keep only the text ones.
        return "".join(block.text for block in response.content if block.type == "text")


def get_llm() -> LLMProvider:
    """Build the provider selected by the LLM_PROVIDER setting.

    Raises:
        ValueError: if the provider name is unknown or its API key is missing.
    """
    s = get_settings()
    provider = s.llm_provider.lower()
    if provider == "ollama":        
        return OllamaProvider(s.ollama_base_url, s.llm_model, s.llm_timeout_seconds)    
    if provider == "openai":
        if not s.openai_api_key:
            raise ValueError("OPENAI_API_KEY is empty in .env")
        return OpenAIProvider(s.openai_api_key, s.llm_model)
    if provider == "anthropic":
        if not s.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is empty in .env")
        return AnthropicProvider(s.anthropic_api_key, s.llm_model)
    raise ValueError(f"Unknown LLM_PROVIDER: {s.llm_provider!r}")
