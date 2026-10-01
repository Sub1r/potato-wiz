"""
services/ai/ollama_provider.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AIProvider implementation for a locally running Ollama instance.

Ollama does NOT support server-side web search, so any research must be
done externally before building the prompt.  The provider sends a single
chat-completion request and returns the model's text response.

Gracefully handles:
- Ollama not running (ConnectionError / connection refused)
- Model missing (404 from Ollama)
- Timeout
- Malformed JSON response body
- Any unexpected error

Never crashes Flask.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from config import Config
from services.ai.base import AIProvider, AIProviderError

logger = logging.getLogger(__name__)

# Ollama chat-completion endpoint (OpenAI-compatible)
_CHAT_PATH = "/api/chat"


class OllamaProvider(AIProvider):
    """
    Sends chat-completion requests to a local Ollama instance.

    Uses only the Python standard library (urllib) to avoid adding a
    heavyweight dependency.  If ``requests`` is available it is NOT used here
    intentionally — keeping the provider self-contained.
    """

    provider_name = "ollama"

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.base_url = (base_url or Config.OLLAMA_BASE_URL).rstrip("/")
        self.model_name = model or Config.OLLAMA_MODEL
        self.timeout = timeout or Config.OLLAMA_TIMEOUT

    # ── Public interface ──────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """
        Return True when a base URL and model name are configured.

        Does NOT make a network call — availability is verified lazily when
        ``complete()`` is first called.
        """
        return bool(self.base_url and self.model_name)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Send a chat-completion request to Ollama and return the response text.

        ``tools`` is accepted for API compatibility but Ollama does not
        support server-side web search; the parameter is ignored.

        Raises
        ------
        AIProviderError
            On any network, timeout, or model error.
        """
        if tools:
            logger.debug(
                "OllamaProvider: %d tool(s) passed but Ollama does not support "
                "server-side tools — proceeding without web search.",
                len(tools),
            )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",  # request JSON output when supported
            "options": {
                "temperature": 0.1,  # low temperature for deterministic structured output
            },
        }

        url = f"{self.base_url}{_CHAT_PATH}"
        logger.debug("OllamaProvider: POST %s model=%s", url, self.model_name)

        try:
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise AIProviderError(
                f"Ollama connection failed ({self.base_url}): {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise AIProviderError(
                f"Ollama request timed out after {self.timeout}s"
            ) from exc
        except Exception as exc:  # pylint: disable=broad-except
            raise AIProviderError(f"Ollama request error: {exc}") from exc

        # Parse the Ollama response envelope
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AIProviderError(
                f"Ollama returned non-JSON response: {raw[:200]!r}"
            ) from exc

        # Extract the assistant message content
        message = (
            envelope.get("message", {})
            or {}
        )
        content = message.get("content", "")
        if not content:
            # Some Ollama versions use a different shape
            content = envelope.get("response", "")

        if not content:
            raise AIProviderError(
                f"Ollama response had no content. Envelope keys: {list(envelope.keys())}"
            )

        logger.debug(
            "OllamaProvider: received %d chars from model %s",
            len(content),
            self.model_name,
        )
        return content
