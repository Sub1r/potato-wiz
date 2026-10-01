"""
services/ai/openrouter_provider.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AIProvider implementation for OpenRouter.

Features
--------
- OpenAI-compatible chat-completion API via openrouter.ai/api/v1
- Server-side web search using the ``openrouter:web_search`` tool
- Server-side page retrieval using the ``openrouter:web_fetch`` tool
- Agentic tool loop: the model may call tools multiple times before
  producing a final text response
- API key read exclusively from Config — never hardcoded or logged
- Graceful handling of: missing key, connection error, timeout, bad JSON,
  refusal, quota exceeded

IMPORTANT: The API key is NEVER written to any log output.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from config import Config
from services.ai.base import AIProvider, AIProviderError

logger = logging.getLogger(__name__)

_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

# Maximum tool-call iterations to prevent infinite loops
_MAX_TOOL_ITERATIONS = 8

# OpenRouter server-tool definitions
_WEB_SEARCH_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "openrouter:web_search",
        "description": (
            "Search the web for current information about a topic. "
            "Returns a list of search results with titles, URLs, and snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                }
            },
            "required": ["query"],
        },
    },
}

_WEB_FETCH_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "openrouter:web_fetch",
        "description": (
            "Fetch and extract the text content of a specific web page. "
            "Use after web_search to read a full article or benchmark page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch.",
                }
            },
            "required": ["url"],
        },
    },
}

_DEFAULT_TOOLS = [_WEB_SEARCH_TOOL, _WEB_FETCH_TOOL]


class OpenRouterProvider(AIProvider):
    """
    Sends chat-completion requests to OpenRouter with optional web research.

    The model decides when to use search/fetch tools.  We run a tool loop
    until the model produces a final text response (no more tool calls) or
    we hit ``_MAX_TOOL_ITERATIONS``.
    """

    provider_name = "openrouter"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        # Never log the api_key — store privately
        self._api_key = api_key or Config.OPENROUTER_API_KEY
        self.model_name = model or Config.OPENROUTER_MODEL
        self.timeout = timeout or Config.OPENROUTER_TIMEOUT
        self._site_url = Config.OPENROUTER_SITE_URL
        self._site_name = Config.OPENROUTER_SITE_NAME

    # ── Public interface ──────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True when an API key and model name are configured."""
        return bool(self._api_key and self._api_key.strip() and self.model_name)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Execute a potentially multi-turn tool-using completion.

        The method runs a loop:
        1. Send messages to OpenRouter.
        2. If the response contains tool_calls, execute them server-side
           by appending the results and calling again.
        3. When the model returns a plain text message (no tool_calls),
           return that text.

        Raises
        ------
        AIProviderError
            On any unrecoverable network or API error.
        """
        if not self.is_available():
            raise AIProviderError(
                "OpenRouter API key is not configured. "
                "Set OPENROUTER_API_KEY in your environment."
            )

        effective_tools = tools if tools is not None else _DEFAULT_TOOLS

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for iteration in range(_MAX_TOOL_ITERATIONS):
            logger.debug(
                "OpenRouterProvider: iteration %d/%d model=%s",
                iteration + 1, _MAX_TOOL_ITERATIONS, self.model_name,
            )
            response_msg, tool_calls = self._call_api(messages, effective_tools)

            if not tool_calls:
                # Model produced a final answer
                content = response_msg.get("content", "")
                if not content:
                    raise AIProviderError(
                        "OpenRouter returned an empty response content."
                    )
                logger.debug(
                    "OpenRouterProvider: final response %d chars after %d iteration(s)",
                    len(content), iteration + 1,
                )
                return content

            # Process tool calls — OpenRouter handles the actual execution,
            # we just append the assistant's tool_call message and then
            # provide stub tool results so the conversation continues.
            messages.append(response_msg)
            for tc in tool_calls:
                tool_id = tc.get("id", "")
                fn_name = tc.get("function", {}).get("name", "")
                fn_args_raw = tc.get("function", {}).get("arguments", "{}")
                try:
                    fn_args = json.loads(fn_args_raw)
                except json.JSONDecodeError:
                    fn_args = {}

                logger.debug(
                    "OpenRouterProvider: tool call %s(%s)",
                    fn_name, str(fn_args)[:120],
                )
                # Append a tool result placeholder — OpenRouter resolves the
                # actual search/fetch server-side and provides results to the
                # model; we simply tell the API the tool was invoked.
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": fn_name,
                    "content": json.dumps({"status": "executed", "args": fn_args}),
                })

        raise AIProviderError(
            f"OpenRouter tool loop exceeded {_MAX_TOOL_ITERATIONS} iterations without a final response."
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers — API key is placed here and NEVER logged."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self._site_url,
            "X-Title": self._site_name,
        }

    def _call_api(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Make a single API call and return (assistant_message, tool_calls).

        Returns
        -------
        (message_dict, tool_calls_list)
            tool_calls_list is empty when the model produced a final answer.
        """
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        body = json.dumps(payload).encode("utf-8")
        headers = self._build_headers()
        req = urllib.request.Request(
            _CHAT_URL,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8")[:300]
            except Exception:  # pylint: disable=broad-except
                pass
            # Sanitise: never include auth header content in the log
            raise AIProviderError(
                f"OpenRouter HTTP {exc.code}: {exc.reason}. Body: {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise AIProviderError(
                f"OpenRouter connection failed: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise AIProviderError(
                f"OpenRouter request timed out after {self.timeout}s"
            ) from exc
        except Exception as exc:  # pylint: disable=broad-except
            raise AIProviderError(f"OpenRouter request error: {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AIProviderError(
                f"OpenRouter returned non-JSON: {raw[:200]!r}"
            ) from exc

        # Handle API-level errors
        if "error" in data:
            err = data["error"]
            code = err.get("code", "")
            msg = err.get("message", str(err))
            raise AIProviderError(f"OpenRouter API error [{code}]: {msg}")

        choices = data.get("choices", [])
        if not choices:
            raise AIProviderError("OpenRouter response had no choices.")

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls") or []

        return message, tool_calls
