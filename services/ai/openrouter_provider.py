"""
services/ai/openrouter_provider.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AIProvider implementation for OpenRouter.

Features
--------
- OpenAI-compatible chat-completion API via openrouter.ai/api/v1
- Server-side web search using the ``openrouter:web_search`` SERVER TOOL
- Server-side page retrieval using the ``openrouter:web_fetch`` SERVER TOOL
- ONE request per completion: OpenRouter executes the server tools itself and
  returns the tool results plus the final model response.  There is no
  client-side tool loop — these tools are not client-executed functions.
- API key read exclusively from Config — never hardcoded or logged
- Graceful handling of: missing key, connection error, timeout, bad JSON,
  refusal, quota exceeded

Tool format
-----------
OpenRouter's server tools are declared by their TYPE, not by a function name::

    "tools": [
        {"type": "openrouter:web_search", "max_results": 3},
        {"type": "openrouter:web_fetch"}
    ]

Declaring them as ``{"type": "function", "function": {"name":
"openrouter:web_search"}}`` is rejected by the API with::

    Invalid tools.0.name: string does not match the pattern ^[a-zA-Z0-9_.-]+$

because the colon is not allowed in a function/tool name.  That is exactly
what this provider used to send.

Cost controls
-------------
OpenRouter returns HTTP 402 when ``max_tokens`` exceeds the credit balance of
the account ("You requested up to 65536 tokens, but can afford 1793").
Leaving ``max_tokens`` unset makes the request inherit the model's full
65536-token output window, which is unaffordable for a small account.  This
provider therefore always sends an explicit, bounded token budget
(``OPENROUTER_MAX_TOKENS``, default 2048, hard-capped to
``[OPENROUTER_MAX_TOKENS_FLOOR, OPENROUTER_MAX_TOKENS_CEILING]``) plus a bounded
web-research budget expressed through OpenRouter's own supported knobs
(per-tool ``max_results`` and the top-level ``max_tool_calls``).

IMPORTANT: The API key is NEVER written to any log output.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from config import Config
from services.ai.base import AIProvider, AIProviderError

logger = logging.getLogger(__name__)


# ── Structured provider errors ───────────────────────────────────────────────

#: Structured reasons for an unusable provider response.  They let the logs and
#: the tests say exactly what happened instead of one generic message.
REASON_NO_CHOICES = "no_choices"
REASON_EMPTY_MESSAGE = "empty_message"
REASON_EMPTY_CONTENT = "empty_content"
REASON_TRUNCATED = "truncated_no_content"
REASON_TOOL_CALLS_NO_CONTENT = "tool_calls_without_content"
REASON_REFUSAL = "refusal"
REASON_API_ERROR = "api_error"
REASON_MALFORMED = "malformed_response"
REASON_NO_TOOL_SUPPORT = "model_lacks_tool_support"


class OpenRouterResponseError(AIProviderError):
    """
    An OpenRouter response that cannot be turned into a recommendation.

    Carries a machine-readable ``reason`` and a sanitized ``diagnosis`` dict
    (top-level keys, choices count, finish_reason, message keys, whether
    content/tool_calls/refusal are present, usage, provider metadata).  It
    never contains prompts, credentials or the raw response text.
    """

    def __init__(
        self,
        message: str,
        reason: str = REASON_EMPTY_CONTENT,
        diagnosis: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.diagnosis = diagnosis or {}

    def __str__(self) -> str:
        return f"{super().__str__()} [reason={self.reason}]"


def summarize_response(data: Any) -> Dict[str, Any]:
    """
    Build a sanitized description of a raw OpenRouter response.

    Only structural metadata is collected — never prompt text, credentials or
    the response body itself.
    """
    summary: Dict[str, Any] = {
        "top_level_keys": [],
        "choices": 0,
        "finish_reason": None,
        "message_keys": [],
        "content_type": None,
        "content_empty": True,
        "has_tool_calls": False,
        "tool_calls": 0,
        "tool_call_types": [],
        "has_refusal": False,
        "has_reasoning": False,
        "usage": None,
        "provider": None,
        "model": None,
        "error": None,
    }
    if not isinstance(data, dict):
        summary["top_level_keys"] = [type(data).__name__]
        return summary

    summary["top_level_keys"] = sorted(str(k) for k in data.keys())
    summary["provider"] = data.get("provider")
    summary["model"] = data.get("model")

    usage = data.get("usage")
    if isinstance(usage, dict):
        summary["usage"] = {
            key: usage.get(key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                        "cost", "is_byok")
            if key in usage
        }

    error = data.get("error")
    if isinstance(error, dict):
        summary["error"] = {
            "code": error.get("code"),
            "message": str(error.get("message", ""))[:200],
        }

    choices = data.get("choices")
    if not isinstance(choices, list):
        return summary
    summary["choices"] = len(choices)
    if not choices:
        return summary

    first = choices[0] if isinstance(choices[0], dict) else {}
    summary["finish_reason"] = first.get("finish_reason")

    message = first.get("message")
    if not isinstance(message, dict) or not message:
        summary["message_keys"] = []
        return summary

    summary["message_keys"] = sorted(str(k) for k in message.keys())

    content = message.get("content")
    summary["content_type"] = type(content).__name__
    summary["content_empty"] = extract_text(content) == ""

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        summary["has_tool_calls"] = True
        summary["tool_calls"] = len(tool_calls)
        summary["tool_call_types"] = [
            str((tc or {}).get("type", "")) for tc in tool_calls if isinstance(tc, dict)
        ][:5]

    refusal = message.get("refusal") or first.get("refusal")
    summary["has_refusal"] = bool(refusal)
    summary["has_reasoning"] = bool(message.get("reasoning"))
    return summary


def extract_text(content: Any) -> str:
    """
    Return the text of a message content field.

    Handles the plain-string form and the multi-part list form
    (``[{"type": "text", "text": "..."}]``) that some providers return.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return ""

_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Cost policy ──────────────────────────────────────────────────────────────
# Safe token budget defaults.  Config may override these through the
# environment, but the code bounds below always win.
DEFAULT_MAX_TOKENS = 2048
MIN_MAX_TOKENS = 512
MAX_MAX_TOKENS = 4096

DEFAULT_MAX_SEARCH_RESULTS = 3
DEFAULT_MAX_TOTAL_RESULTS = 6
DEFAULT_MAX_SEARCH_CALLS = 2
DEFAULT_MAX_FETCH_PAGES = 1

# Top-level tool-call budget sent to OpenRouter.  OpenRouter stops executing
# server tools after this many calls, which replaces the old client-side loop.
DEFAULT_MAX_TOOL_CALLS = 4
MAX_MAX_TOOL_CALLS = 8

#: Characters/tokens of page content a web_fetch may return.
DEFAULT_MAX_FETCH_CONTENT_TOKENS = 2000

#: ``openrouter:web_fetch`` is disabled while the server-tool configuration is
#: being validated; set OPENROUTER_WEB_FETCH_ENABLED=true to send it again.
DEFAULT_INCLUDE_WEB_FETCH = False

#: OpenRouter model catalog, used for the tool-calling capability check.
#: Only the list endpoint is reliable: /models/{author}/{slug} answers 404.
_MODELS_URL = "https://openrouter.ai/api/v1/models"

#: Models known to reject or ignore server tools.  The live capability check
#: is authoritative; this only covers the catalog check being unavailable.
_KNOWN_NON_TOOL_MODEL_HINTS = ("embedding", "moderation", "whisper", "tts", "rerank")

#: Tool-support verdicts, cached per model for the process.
_TOOL_SUPPORT_CACHE: Dict[str, Dict[str, Any]] = {}

# The colon belongs in the server-tool TYPE, never in a function name.
WEB_SEARCH_TYPE = "openrouter:web_search"
WEB_FETCH_TYPE = "openrouter:web_fetch"

#: Pattern every OpenRouter tool/function name must satisfy.  Server-tool
#: types contain a colon and therefore may never be used as a name.
_TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")


def clamp_max_tokens(value: Any, default: int = DEFAULT_MAX_TOKENS) -> int:
    """
    Clamp a requested max_tokens value into the safe range.

    Anything unparseable falls back to ``default``; anything outside
    ``[MIN_MAX_TOKENS, MAX_MAX_TOKENS]`` is clamped.  This is what stops a
    large ``OPENROUTER_MAX_TOKENS`` environment value from requesting the
    model's full 65536-token output window.
    """
    try:
        tokens = int(value)
    except (TypeError, ValueError):
        tokens = int(default)
    if tokens < MIN_MAX_TOKENS:
        return MIN_MAX_TOKENS
    if tokens > MAX_MAX_TOKENS:
        return MAX_MAX_TOKENS
    return tokens


def _positive_int(value: Any, default: int, minimum: int = 0) -> int:
    """Coerce a value to a non-negative int, falling back to ``default``."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return int(default)
    if number < minimum:
        return minimum
    return number


class ResearchLimits:
    """
    The web-research budget for a single ``complete()`` call.

    These values are translated into OpenRouter's own supported controls
    rather than enforced client-side, because OpenRouter executes the
    ``openrouter:web_search`` / ``openrouter:web_fetch`` server tools itself:

    - ``max_search_results`` → ``max_results`` on the web_search tool,
    - ``max_total_results`` → total results the request may consume,
      expressed as the per-tool cap repeated across ``max_search_calls`` and
      kept consistent with the top-level ``max_tool_calls`` budget,
    - ``max_search_calls`` → how many searches the model may make,
    - ``max_fetch_pages`` → how many pages may be fetched,
    - ``max_tool_calls`` → OpenRouter's top-level server-tool call budget.
    """

    def __init__(
        self,
        max_search_results: int = DEFAULT_MAX_SEARCH_RESULTS,
        max_total_results: int = DEFAULT_MAX_TOTAL_RESULTS,
        max_search_calls: int = DEFAULT_MAX_SEARCH_CALLS,
        max_fetch_pages: int = DEFAULT_MAX_FETCH_PAGES,
        max_tool_calls: Optional[int] = None,
    ) -> None:
        self.max_search_results = max_search_results
        self.max_total_results = max_total_results
        self.max_search_calls = max_search_calls
        self.max_fetch_pages = max_fetch_pages
        self.max_tool_calls = self._resolve_tool_calls(max_tool_calls)

    def _resolve_tool_calls(self, max_tool_calls: Optional[int]) -> int:
        """
        Derive the top-level server-tool call budget.

        Enough calls for the allowed searches and page fetches (plus one
        search call for the model's first look), but never more than the
        provider's hard ceiling.
        """
        if max_tool_calls is None:
            max_tool_calls = self.max_search_calls + self.max_fetch_pages + 1
        max_tool_calls = _positive_int(max_tool_calls, DEFAULT_MAX_TOOL_CALLS, minimum=1)
        return min(max_tool_calls, MAX_MAX_TOOL_CALLS)

    def results_per_search(self) -> int:
        """
        Per-search result cap, additionally limited by what is left of the
        cumulative budget.
        """
        if self.max_search_calls <= 0:
            return 1
        per_call = self.max_total_results // self.max_search_calls
        return max(1, min(self.max_search_results, per_call))

    def as_dict(self) -> Dict[str, Any]:
        """Snapshot of the configured limits (used for logging and tests)."""
        return {
            "max_search_results": self.max_search_results,
            "max_total_results": self.max_total_results,
            "max_search_calls": self.max_search_calls,
            "max_fetch_pages": self.max_fetch_pages,
            "max_tool_calls": self.max_tool_calls,
        }


# ── OpenRouter server tools ───────────────────────────────────────────────────
# Declared by TYPE with their limits nested under "parameters".
# ``{"type": "function", "function": {"name": "openrouter:web_search"}}`` is
# invalid: the colon is not allowed in a name.


def build_tools(
    max_search_results: int = DEFAULT_MAX_SEARCH_RESULTS,
    max_total_results: int = DEFAULT_MAX_TOTAL_RESULTS,
    max_search_calls: int = DEFAULT_MAX_SEARCH_CALLS,
    max_fetch_pages: int = DEFAULT_MAX_FETCH_PAGES,
    include_web_fetch: bool = DEFAULT_INCLUDE_WEB_FETCH,
) -> List[Dict[str, Any]]:
    """
    Build the OpenRouter server-tool list with the cost budget nested inside
    each tool's ``parameters``.

    Web search::

        {"type": "openrouter:web_search",
         "parameters": {"max_results": 3, "max_total_results": 6, "max_uses": 2}}

    Web fetch (only when enabled)::

        {"type": "openrouter:web_fetch",
         "parameters": {"max_uses": 1, "max_content_tokens": 2000}}

    ``web_fetch`` has no ``max_results``: its limits are ``max_uses`` and
    ``max_content_tokens``.
    """
    limits = ResearchLimits(
        max_search_results=max_search_results,
        max_total_results=max_total_results,
        max_search_calls=max_search_calls,
        max_fetch_pages=max_fetch_pages,
    )
    tools: List[Dict[str, Any]] = [
        {
            "type": WEB_SEARCH_TYPE,
            "parameters": {
                "max_results": limits.results_per_search(),
                "max_total_results": limits.max_total_results,
                "max_uses": limits.max_search_calls,
            },
        },
    ]
    if include_web_fetch and max_fetch_pages > 0:
        tools.append({
            "type": WEB_FETCH_TYPE,
            "parameters": {
                "max_uses": max_fetch_pages,
                "max_content_tokens": DEFAULT_MAX_FETCH_CONTENT_TOKENS,
            },
        })
    return tools


#: Server-tool types this provider knows how to send.
KNOWN_TOOL_TYPES = (WEB_SEARCH_TYPE, WEB_FETCH_TYPE)


def validate_tools(tools: List[Dict[str, Any]]) -> None:
    """
    Guard against re-introducing invalid server-tool declarations.

    Raises
    ------
    AIProviderError
        If a tool is declared as a function whose name contains a colon, if a
        tool type is unknown, or if a known server tool carries its limits
        outside ``parameters`` (the shape that made OpenRouter ignore the
        tool configuration).
    """
    for index, tool in enumerate(tools or []):
        if not isinstance(tool, dict):
            raise AIProviderError(f"OpenRouter tool {index} is not an object.")
        if tool.get("type") == "function":
            name = (tool.get("function") or {}).get("name", "")
            if not _TOOL_NAME_PATTERN.match(str(name)):
                raise AIProviderError(
                    f"Invalid tools.{index}.name: string does not match the "
                    f"pattern ^[a-zA-Z0-9_.-]+$. Server tools such as "
                    f"'{name}' must be declared by type, not by name."
                )
            continue
        tool_type = str(tool.get("type", ""))
        if tool_type not in KNOWN_TOOL_TYPES:
            raise AIProviderError(f"Unknown OpenRouter tool type: {tool_type!r}.")
        if not isinstance(tool.get("parameters"), dict):
            raise AIProviderError(
                f"OpenRouter tool {index} ({tool_type}) must carry its limits "
                f"inside a 'parameters' object."
            )
        for misplaced in ("max_results", "max_uses", "max_content_tokens",
                          "max_total_results"):
            if misplaced in tool:
                raise AIProviderError(
                    f"OpenRouter tool {index} ({tool_type}) must not set "
                    f"'{misplaced}' at the tool level; it belongs inside "
                    f"'parameters'."
                )
        if tool_type == WEB_FETCH_TYPE and "max_results" in tool.get("parameters", {}):
            raise AIProviderError(
                "openrouter:web_fetch does not support 'max_results'; use "
                "'max_uses' and 'max_content_tokens'."
            )


_DEFAULT_TOOLS = build_tools()


class OpenRouterProvider(AIProvider):
    """
    Sends ONE chat-completion request to OpenRouter with optional research.

    Web search and page fetch are OpenRouter *server* tools declared by type
    (``{"type": "openrouter:web_search"}``).  OpenRouter executes them and
    returns the tool results plus the final model response, so this class
    contains no client-side tool loop.
    """

    provider_name = "openrouter"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_search_results: Optional[int] = None,
        max_total_results: Optional[int] = None,
        max_search_calls: Optional[int] = None,
        max_fetch_pages: Optional[int] = None,
        max_tool_calls: Optional[int] = None,
        include_web_fetch: Optional[bool] = None,
    ) -> None:
        # Never log the api_key — store privately.
        #
        # Sentinel logic:
        #   api_key=None  → caller did not supply a key; load from environment.
        #   api_key=""    → caller explicitly supplied an empty string;
        #                   do NOT fall back to the environment.
        #   api_key="x"   → use exactly what was supplied.
        self._api_key = Config.OPENROUTER_API_KEY if api_key is None else api_key
        self.model_name = model or Config.OPENROUTER_MODEL
        self.timeout = timeout or Config.OPENROUTER_TIMEOUT
        self._site_url = Config.OPENROUTER_SITE_URL
        self._site_name = Config.OPENROUTER_SITE_NAME

        # Cost budget.  Config values come from the environment but are always
        # clamped, so an oversized OPENROUTER_MAX_TOKENS cannot trigger the
        # HTTP 402 "requires more credits" rejection.
        self.max_tokens = clamp_max_tokens(
            Config.OPENROUTER_MAX_TOKENS if max_tokens is None else max_tokens
        )
        self.max_search_results = _positive_int(
            Config.OPENROUTER_MAX_SEARCH_RESULTS if max_search_results is None else max_search_results,
            DEFAULT_MAX_SEARCH_RESULTS,
            minimum=1,
        )
        self.max_total_results = _positive_int(
            Config.OPENROUTER_MAX_TOTAL_RESULTS if max_total_results is None else max_total_results,
            DEFAULT_MAX_TOTAL_RESULTS,
            minimum=1,
        )
        self.max_search_calls = _positive_int(
            Config.OPENROUTER_MAX_SEARCH_CALLS if max_search_calls is None else max_search_calls,
            DEFAULT_MAX_SEARCH_CALLS,
            minimum=1,
        )
        self.max_fetch_pages = _positive_int(
            Config.OPENROUTER_MAX_FETCH_PAGES if max_fetch_pages is None else max_fetch_pages,
            DEFAULT_MAX_FETCH_PAGES,
            minimum=0,
        )
        self.max_tool_calls = ResearchLimits(
            max_search_results=self.max_search_results,
            max_total_results=self.max_total_results,
            max_search_calls=self.max_search_calls,
            max_fetch_pages=self.max_fetch_pages,
            max_tool_calls=Config.OPENROUTER_MAX_TOOL_CALLS if max_tool_calls is None else max_tool_calls,
        ).max_tool_calls
        self.include_web_fetch = (
            Config.OPENROUTER_WEB_FETCH_ENABLED
            if include_web_fetch is None
            else bool(include_web_fetch)
        )
        logger.debug(
            "OpenRouterProvider: model=%s max_tokens=%d "
            "search_results=%d total_results=%d search_calls=%d fetch_pages=%d "
            "tool_calls=%d web_fetch=%s",
            self.model_name,
            self.max_tokens,
            self.max_search_results,
            self.max_total_results,
            self.max_search_calls,
            self.max_fetch_pages,
            self.max_tool_calls,
            self.include_web_fetch,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def research_limits(self) -> ResearchLimits:
        """Return the configured web-research limits for a single request."""
        return ResearchLimits(
            max_search_results=self.max_search_results,
            max_total_results=self.max_total_results,
            max_search_calls=self.max_search_calls,
            max_fetch_pages=self.max_fetch_pages,
            max_tool_calls=self.max_tool_calls,
        )

    # Alias kept for callers that used the previous accessor name.
    budget = research_limits

    def build_tools(self) -> List[Dict[str, Any]]:
        """Return the OpenRouter server-tool list with limits applied."""
        return build_tools(
            max_search_results=self.max_search_results,
            max_total_results=self.max_total_results,
            max_search_calls=self.max_search_calls,
            max_fetch_pages=self.max_fetch_pages,
            include_web_fetch=self.include_web_fetch,
        )

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
        Execute ONE completion request.

        ``openrouter:web_search`` and ``openrouter:web_fetch`` are OpenRouter
        server tools: OpenRouter executes them itself and returns the tool
        results together with the model's final response.  There is no
        client-side tool loop, so exactly one HTTP request is made.

        Parameters
        ----------
        system_prompt, user_prompt:
            The conversation sent to the model.
        tools:
            Optional override of the tool list.  Defaults to the configured
            server tools.  Pass ``[]`` to disable web research.

        Raises
        ------
        AIProviderError
            On any unrecoverable network or API error, or when the response
            carries no usable content.  ``ai_optimizer`` catches this and
            falls back to the deterministic optimizer.
        """
        if not self.is_available():
            raise AIProviderError(
                "OpenRouter API key is not configured. "
                "Set OPENROUTER_API_KEY in your environment."
            )

        effective_tools = tools if tools is not None else self.build_tools()
        validate_tools(effective_tools)

        if effective_tools:
            # Never silently send tools to a model that cannot execute them.
            self.verify_tool_support()

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.debug(
            "OpenRouterProvider: single request model=%s max_tokens=%d "
            "tools=%s research=%s",
            self.model_name,
            self.max_tokens,
            [t.get("type") for t in effective_tools],
            self.research_limits().as_dict(),
        )

        message, diagnosis = self._call_api(messages, effective_tools)

        content = extract_text(message.get("content"))
        if content:
            if diagnosis.get("finish_reason") == "length":
                # Reasoning models can spend the whole output budget on
                # reasoning and truncate the JSON answer.  Surface it: the
                # answer is usually unparseable and the caller falls back.
                logger.warning(
                    "OpenRouter response truncated (finish_reason=length) with "
                    "max_tokens=%d: content=%d chars, usage=%s. Raise "
                    "OPENROUTER_MAX_TOKENS or lower OPENROUTER_REASONING_EFFORT.",
                    self.max_tokens, len(content), diagnosis.get("usage"),
                )
            logger.debug(
                "OpenRouterProvider: final response %d chars from a single request",
                len(content),
            )
            return content

        # No usable answer.  Diagnose exactly what came back (sanitized
        # metadata only) so the logs and tests distinguish empty content, a
        # client-side tool call, a refusal and truncated output, instead of a
        # single opaque message.
        raise self._diagnosis_error(message, diagnosis)

    def _diagnosis_error(
        self,
        message: Dict[str, Any],
        diagnosis: Optional[Dict[str, Any]] = None,
    ) -> OpenRouterResponseError:
        """
        Turn an unusable assistant message into a structured error.

        Handles the distinct situations explicitly:

        A. refusal present          → REASON_REFUSAL
        B. tool_calls, no content   → REASON_TOOL_CALLS_NO_CONTENT
        C. finish_reason=length     → REASON_TRUNCATED
        D. nothing but empty text   → REASON_EMPTY_CONTENT
        E. empty message object     → REASON_EMPTY_MESSAGE
        """
        diagnosis = diagnosis or summarize_response(
            {"choices": [{"message": message or {}}]}
        )

        if diagnosis["has_refusal"]:
            reason = REASON_REFUSAL
            message_text = "OpenRouter model refused the request."
        elif diagnosis["has_tool_calls"]:
            reason = REASON_TOOL_CALLS_NO_CONTENT
            message_text = (
                "OpenRouter returned client-side tool calls without final "
                "content. openrouter:web_search is a server tool and must be "
                "executed by OpenRouter; check the model supports tools and "
                "that the tool 'parameters' block is well formed."
            )
        elif not message:
            reason = REASON_EMPTY_MESSAGE
            message_text = "OpenRouter returned an empty message object."
        elif diagnosis["finish_reason"] == "length":
            reason = REASON_TRUNCATED
            message_text = (
                f"OpenRouter returned no content (finish_reason=length). "
                f"max_tokens={self.max_tokens} was likely exhausted by "
                f"reasoning before any answer was written."
            )
        else:
            reason = REASON_EMPTY_CONTENT
            message_text = "OpenRouter returned an empty response content."

        logger.warning(
            "OpenRouter unusable response: reason=%s diagnosis=%s",
            reason, diagnosis,
        )
        return OpenRouterResponseError(message_text, reason=reason,
                                       diagnosis=diagnosis)

    # ── Internal ──────────────────────────────────────────────────────────────

    def verify_tool_support(self) -> Dict[str, Any]:
        """
        Check that the configured model supports tool calling / server tools.

        Uses the OpenRouter model catalog (``supported_parameters``), cached per
        model for the process.  Returns a sanitized verdict:

            {"model": ..., "supported": True|False|None, "source": ..., "reason": ...}

        ``supported`` is ``None`` when the catalog could not be consulted; in
        that case the request is still attempted rather than silently changing
        the tool configuration.

        Raises
        ------
        OpenRouterResponseError
            Only when the catalog positively reports that the model does not
            support tools — never a silent downgrade of the tool config.
        """
        model = self.model_name
        cached = _TOOL_SUPPORT_CACHE.get(model)
        if cached is not None:
            return cached

        verdict: Dict[str, Any] = {
            "model": model,
            "supported": None,
            "source": "not_checked",
            "reason": "",
        }

        lowered = model.lower()
        hint = next((h for h in _KNOWN_NON_TOOL_MODEL_HINTS if h in lowered), None)
        if hint:
            verdict.update(
                supported=False,
                source="model_id_hint",
                reason=f"model id suggests a non-tool model ('{hint}')",
            )
            _TOOL_SUPPORT_CACHE[model] = verdict
            raise OpenRouterResponseError(
                f"OpenRouter model '{model}' does not support tool calling "
                f"(reason: {verdict['reason']}). Choose a tool-capable model "
                f"such as 'openai/gpt-5-mini' or 'anthropic/claude-3.5-haiku' "
                f"and set OPENROUTER_MODEL.",
                reason=REASON_NO_TOOL_SUPPORT,
                diagnosis=verdict,
            )

        entry = self._lookup_model()
        if entry is None:
            verdict.update(
                supported=None,
                source="catalog_unavailable",
                reason="model catalog unavailable or model not listed",
            )
            logger.warning(
                "OpenRouter tool-support check inconclusive for model=%s: %s",
                model, verdict["reason"],
            )
        else:
            params = entry.get("supported_parameters") or []
            supports = "tools" in [str(p) for p in params]
            verdict.update(
                supported=bool(supports),
                source="catalog",
                reason="" if supports else "catalog lists no 'tools' parameter",
            )
            _TOOL_SUPPORT_CACHE[model] = verdict
            if not supports:
                logger.warning(
                    "OpenRouter model %s does not support tool calling per the "
                    "model catalog.", model,
                )
                raise OpenRouterResponseError(
                    f"OpenRouter model '{model}' does not support tool calling "
                    f"(reason: {verdict['reason']}). Pick a tool-capable model "
                    f"such as 'openai/gpt-5-mini' or 'anthropic/claude-3.5-haiku' "
                    f"and set OPENROUTER_MODEL.",
                    reason=REASON_NO_TOOL_SUPPORT,
                    diagnosis=verdict,
                )
            return verdict

        _TOOL_SUPPORT_CACHE[model] = verdict
        return verdict

    def _lookup_model(self) -> Optional[Dict[str, Any]]:
        """
        Fetch the model catalog and return this model's entry, or None.

        Only the list endpoint (``/api/v1/models``) is used; the per-model path
        answers 404.
        """
        if not Config.OPENROUTER_VERIFY_TOOL_SUPPORT:
            return None
        req = urllib.request.Request(
            _MODELS_URL,
            headers=self._build_headers(),
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("OpenRouter model catalog lookup failed: %s", exc)
            return None
        if not isinstance(payload, dict):
            return None
        data = payload.get("data")
        if isinstance(data, dict):
            return data if data.get("id") == self.model_name else None
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict) and entry.get("id") == self.model_name:
                    return entry
        return None

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
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Make the single API call and return the assistant message.

        OpenRouter executes the server tools itself and returns the final
        model response, so no client-side tool handling is needed here.
        """
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            # Always explicit: without this the request inherits the model's
            # full 65536-token output window and OpenRouter rejects it with
            # HTTP 402 on a small credit balance.
            "max_tokens": self.max_tokens,
        }
        # Reasoning models can burn the whole output budget on reasoning and
        # return no content at all.  Off unless configured.
        reasoning_effort = Config.OPENROUTER_REASONING_EFFORT
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        if tools:
            # Server tools are declared by type with their limits nested under
            # "parameters"; OpenRouter executes them and enforces the
            # tool-call budget server-side.  No tool_choice is sent: the
            # server tools are optional by nature.
            payload["tools"] = tools
            payload["max_tool_calls"] = self.research_limits().max_tool_calls

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
            if exc.code == 402:
                raise AIProviderError(
                    f"OpenRouter HTTP 402: not enough credits for this request "
                    f"(max_tokens={self.max_tokens}). Body: {error_body}"
                ) from exc
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
            logger.warning(
                "OpenRouter non-JSON response: %d bytes, starts with %r",
                len(raw), raw[:40],
            )
            raise OpenRouterResponseError(
                "OpenRouter returned a non-JSON response.",
                reason=REASON_MALFORMED,
                diagnosis={"content_type": type(raw).__name__, "bytes": len(raw)},
            ) from exc

        diagnosis = summarize_response(data)
        logger.debug("OpenRouter response metadata: %s", diagnosis)

        # Handle API-level errors
        if isinstance(data, dict) and data.get("error"):
            err = data.get("error") or {}
            code = err.get("code", "") if isinstance(err, dict) else ""
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise OpenRouterResponseError(
                f"OpenRouter API error [{code}]: {msg}",
                reason=REASON_API_ERROR,
                diagnosis=diagnosis,
            )

        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise OpenRouterResponseError(
                "OpenRouter response had no choices.",
                reason=REASON_NO_CHOICES,
                diagnosis=diagnosis,
            )

        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        return (message if isinstance(message, dict) else {}), diagnosis
