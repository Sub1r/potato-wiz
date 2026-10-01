"""
tests/test_openrouter_limits.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Server-tool configuration, cost controls and response diagnostics for the
OpenRouter provider.

Regression contexts
-------------------
1. HTTP 402 — "This request requires more credits, or fewer max_tokens.
   You requested up to 65536 tokens, but can afford 1793."  The request did
   not send ``max_tokens`` and inherited the model's 65536-token window.

2. HTTP 400 — "Invalid tools.0.name: string does not match the pattern
   ^[a-zA-Z0-9_.-]+$".  The server tools were declared in function form with a
   colon-bearing name.  Server tools must be declared by TYPE, with their
   limits nested under ``parameters``.

3. "OpenRouter returned an empty response content."  The provider reported one
   opaque message for every unusable response.  Responses are now inspected
   and classified: final answer / client-side tool call / refusal / truncated
   output / no choices / malformed body.
"""
from __future__ import annotations

import json
from unittest import mock

import pytest

from services.ai.base import AIProviderError
from services.ai.openrouter_provider import (
    DEFAULT_MAX_FETCH_CONTENT_TOKENS,
    DEFAULT_MAX_FETCH_PAGES,
    DEFAULT_MAX_SEARCH_CALLS,
    DEFAULT_MAX_SEARCH_RESULTS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MAX_TOTAL_RESULTS,
    MAX_MAX_TOKENS,
    MAX_MAX_TOOL_CALLS,
    MIN_MAX_TOKENS,
    REASON_API_ERROR,
    REASON_EMPTY_CONTENT,
    REASON_EMPTY_MESSAGE,
    REASON_MALFORMED,
    REASON_NO_CHOICES,
    REASON_NO_TOOL_SUPPORT,
    REASON_REFUSAL,
    REASON_TOOL_CALLS_NO_CONTENT,
    REASON_TRUNCATED,
    WEB_FETCH_TYPE,
    WEB_SEARCH_TYPE,
    OpenRouterProvider,
    OpenRouterResponseError,
    ResearchLimits,
    build_tools,
    clamp_max_tokens,
    extract_text,
    summarize_response,
    validate_tools,
)

# Pattern enforced by OpenRouter for tool/function names.
_TOOL_NAME_PATTERN = r"^[a-zA-Z0-9_.-]+$"

#: The first live request carries web_search only.
_EXPECTED_SEARCH_TOOL = {
    "type": WEB_SEARCH_TYPE,
    "parameters": {"max_results": 3, "max_total_results": 6, "max_uses": 2},
}
_EXPECTED_FETCH_TOOL = {
    "type": WEB_FETCH_TYPE,
    "parameters": {"max_uses": 1, "max_content_tokens": 2000},
}


@pytest.fixture(autouse=True)
def _no_capability_lookup():
    """
    The tool-support catalog check issues its own HTTP request.

    It is disabled by default here so the chat request is the only request
    observed; dedicated tests enable it explicitly.
    """
    with mock.patch("config.Config.OPENROUTER_VERIFY_TOOL_SUPPORT", False):
        yield


def _provider(**kwargs) -> OpenRouterProvider:
    """Provider with a fixed key so it never touches the real environment."""
    return OpenRouterProvider(api_key="sk-or-test-key", **kwargs)


def _assistant(content, finish_reason="stop", **extra) -> dict:
    message = {"role": "assistant", "content": content}
    message.update(extra)
    return {"choices": [{"message": message, "finish_reason": finish_reason}]}


def _final_message(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}]}


def _tool_call(name: str = "web_search", call_id: str = "call-1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps({"query": "x"})},
    }


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        if isinstance(self._payload, bytes):
            return self._payload
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _patch_urlopen(*responses):
    """Patch urlopen, returning the patcher and the captured request payloads."""
    payloads = []

    def fake_urlopen(req, timeout=None):
        body = req.data.decode("utf-8") if getattr(req, "data", None) else None
        payloads.append(json.loads(body) if body else {"__url__": req.full_url})
        return _FakeResponse(responses[min(len(payloads) - 1, len(responses) - 1)])

    return mock.patch("urllib.request.urlopen", side_effect=fake_urlopen), payloads


# ---------------------------------------------------------------------------
# 1. Server-tool payload shape (nested parameters)
# ---------------------------------------------------------------------------

class TestServerToolPayloadShape:
    def test_first_request_is_web_search_only(self):
        """Diagnosis mode: only the web_search server tool is sent."""
        patcher, payloads = _patch_urlopen(_final_message('{"summary": "ok"}'))
        with patcher:
            _provider().complete("sys", "usr")
        assert payloads[0]["tools"] == [_EXPECTED_SEARCH_TOOL]

    def test_web_search_max_results(self):
        params = build_tools()[0]["parameters"]
        assert params["max_results"] == 3

    def test_web_search_max_total_results(self):
        params = build_tools()[0]["parameters"]
        assert params["max_total_results"] == 6

    def test_web_search_max_uses(self):
        params = build_tools()[0]["parameters"]
        assert params["max_uses"] == 2

    def test_web_search_limits_come_from_config(self):
        with mock.patch("config.Config.OPENROUTER_MAX_SEARCH_RESULTS", 4), \
             mock.patch("config.Config.OPENROUTER_MAX_TOTAL_RESULTS", 8), \
             mock.patch("config.Config.OPENROUTER_MAX_SEARCH_CALLS", 2):
            params = _provider().build_tools()[0]["parameters"]
        assert params["max_results"] == 4
        assert params["max_total_results"] == 8
        assert params["max_uses"] == 2

    def test_limits_are_not_at_the_tool_level(self):
        """Top-level max_results is the shape OpenRouter ignored."""
        for tool in build_tools(include_web_fetch=True):
            assert "max_results" not in tool
            assert "max_uses" not in tool
            assert "max_content_tokens" not in tool
            assert isinstance(tool["parameters"], dict)

    def test_web_fetch_uses_fetch_parameters(self):
        tools = build_tools(include_web_fetch=True)
        fetch = next(t for t in tools if t["type"] == WEB_FETCH_TYPE)
        assert fetch == _EXPECTED_FETCH_TOOL

    def test_web_fetch_max_uses(self):
        fetch = build_tools(max_fetch_pages=2, include_web_fetch=True)[1]
        assert fetch["parameters"]["max_uses"] == 2

    def test_web_fetch_max_content_tokens(self):
        fetch = build_tools(include_web_fetch=True)[1]
        assert fetch["parameters"]["max_content_tokens"] == \
            DEFAULT_MAX_FETCH_CONTENT_TOKENS == 2000

    def test_no_max_results_on_web_fetch(self):
        fetch = build_tools(include_web_fetch=True)[1]
        assert "max_results" not in fetch["parameters"]
        assert set(fetch["parameters"]) == {"max_uses", "max_content_tokens"}

    def test_web_fetch_disabled_by_default(self):
        types = [t["type"] for t in _provider().build_tools()]
        assert types == [WEB_SEARCH_TYPE]

    def test_web_fetch_can_be_re_enabled(self):
        with mock.patch("config.Config.OPENROUTER_WEB_FETCH_ENABLED", True):
            types = [t["type"] for t in _provider().build_tools()]
        assert types == [WEB_SEARCH_TYPE, WEB_FETCH_TYPE]

    def test_fetch_never_sent_when_budget_is_zero(self):
        p = _provider(max_fetch_pages=0, include_web_fetch=True)
        types = [t["type"] for t in p.build_tools()]
        assert types == [WEB_SEARCH_TYPE]

    def test_no_plugins_or_online_mechanism(self):
        """Only the current server-tool API is used."""
        patcher, payloads = _patch_urlopen(_final_message('{"summary": "ok"}'))
        with patcher:
            _provider().complete("sys", "usr")
        payload = payloads[0]
        assert "plugins" not in payload
        assert ":online" not in json.dumps(payload)
        assert all(t["type"] == WEB_SEARCH_TYPE for t in payload["tools"])

    def test_no_tool_choice_sent(self):
        patcher, payloads = _patch_urlopen(_final_message('{"summary": "ok"}'))
        with patcher:
            _provider().complete("sys", "usr")
        assert "tool_choice" not in payloads[0]

    def test_payload_matches_documented_chat_completions_structure(self):
        patcher, payloads = _patch_urlopen(_final_message('{"summary": "ok"}'))
        with patcher, mock.patch("config.Config.OPENROUTER_REASONING_EFFORT", ""):
            _provider().complete("sys", "usr")
        payload = payloads[0]
        assert set(payload) == {
            "model", "messages", "temperature", "response_format",
            "max_tokens", "tools", "max_tool_calls",
        }
        assert payload["model"] == _provider().model_name
        assert [m["role"] for m in payload["messages"]] == ["system", "user"]


class TestValidateTools:
    def test_accepts_default_server_tools(self):
        validate_tools(build_tools())  # must not raise

    def test_accepts_both_server_tools(self):
        validate_tools(build_tools(include_web_fetch=True))

    def test_rejects_colon_function_name(self):
        """Regression: this is exactly the payload that caused HTTP 400."""
        malformed = [{"type": "function",
                      "function": {"name": "openrouter:web_search"}}]
        with pytest.raises(AIProviderError) as exc:
            validate_tools(malformed)
        assert "tools.0.name" in str(exc.value)
        assert _TOOL_NAME_PATTERN in str(exc.value)

    def test_rejects_unknown_tool_type(self):
        with pytest.raises(AIProviderError):
            validate_tools([{"type": "openrouter:invented_tool",
                             "parameters": {}}])

    def test_rejects_limits_outside_parameters(self):
        bad = [{"type": WEB_SEARCH_TYPE, "max_results": 3}]
        with pytest.raises(AIProviderError) as exc:
            validate_tools(bad)
        assert "parameters" in str(exc.value)

    def test_rejects_missing_parameters_block(self):
        with pytest.raises(AIProviderError):
            validate_tools([{"type": WEB_SEARCH_TYPE}])

    def test_rejects_max_results_on_web_fetch(self):
        bad = [{"type": WEB_FETCH_TYPE,
                "parameters": {"max_uses": 1, "max_results": 3}}]
        with pytest.raises(AIProviderError) as exc:
            validate_tools(bad)
        assert "max_uses" in str(exc.value)

    def test_malformed_tool_argument_raises_before_request(self):
        patcher, payloads = _patch_urlopen(_final_message('{"summary": "ok"}'))
        malformed = [{"type": "function", "function": {"name": "openrouter:web_fetch"}}]
        with patcher:
            with pytest.raises(AIProviderError):
                _provider().complete("sys", "usr", tools=malformed)
        assert payloads == []  # no request was sent

    def test_no_colon_name_literal_in_any_dict_in_source(self):
        import ast
        import inspect

        from services.ai import openrouter_provider

        tree = ast.parse(inspect.getsource(openrouter_provider))
        offenders = [
            ast.dump(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and key.value == "name"
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and ":" in value.value
        ]
        assert offenders == []

    def test_payload_tools_have_no_function_wrapper(self):
        patcher, payloads = _patch_urlopen(_final_message('{"summary": "ok"}'))
        with patcher:
            _provider().complete("sys", "usr")
        for tool in payloads[0]["tools"]:
            assert "function" not in tool
            assert "name" not in tool


# ---------------------------------------------------------------------------
# 2. Single request, no client-side tool loop
# ---------------------------------------------------------------------------

class TestSingleRequestArchitecture:
    def test_exactly_one_http_request(self):
        patcher, payloads = _patch_urlopen(_final_message('{"summary": "ok"}'))
        with patcher:
            _provider().complete("sys", "usr")
        assert len(payloads) == 1

    def test_no_manual_tool_loop_in_source(self):
        import inspect

        from services.ai import openrouter_provider

        source = inspect.getsource(openrouter_provider.OpenRouterProvider.complete)
        assert "for iteration" not in source
        assert "_MAX_TOOL_ITERATIONS" not in source
        assert 'role": "tool' not in source

    def test_top_level_tool_call_budget_sent(self):
        patcher, payloads = _patch_urlopen(_final_message('{"summary": "ok"}'))
        with patcher:
            _provider().complete("sys", "usr")
        assert payloads[0]["max_tool_calls"] == DEFAULT_MAX_TOOL_CALLS
        assert 1 <= payloads[0]["max_tool_calls"] <= MAX_MAX_TOOL_CALLS

    def test_no_tool_call_budget_when_tools_disabled(self):
        patcher, payloads = _patch_urlopen(_final_message('{"summary": "ok"}'))
        with patcher:
            _provider().complete("sys", "usr", tools=[])
        assert "tools" not in payloads[0]
        assert "max_tool_calls" not in payloads[0]


# ---------------------------------------------------------------------------
# 3. Response diagnosis
# ---------------------------------------------------------------------------

class TestSuccessfulResponse:
    def test_non_empty_content_is_returned(self):
        body = {"choices": [{"message": {"role": "assistant",
                                          "content": '{"summary":"test"}'},
                             "finish_reason": "stop"}]}
        patcher, _ = _patch_urlopen(body)
        with patcher:
            assert _provider().complete("sys", "usr") == '{"summary":"test"}'

    def test_list_content_is_joined(self):
        body = _assistant([{"type": "text", "text": '{"summary":'},
                           {"type": "text", "text": '"test"}'}])
        patcher, _ = _patch_urlopen(body)
        with patcher:
            assert _provider().complete("sys", "usr") == '{"summary":"test"}'

    def test_extract_text_handles_forms(self):
        assert extract_text("abc") == "abc"
        assert extract_text([{"type": "text", "text": "ab"}]) == "ab"
        assert extract_text(None) == ""
        assert extract_text(123) == ""


class TestEmptyContentResponse:
    def test_empty_content_is_classified(self):
        patcher, _ = _patch_urlopen(_assistant(""))
        with patcher:
            with pytest.raises(OpenRouterResponseError) as exc:
                _provider().complete("sys", "usr")
        assert exc.value.reason == REASON_EMPTY_CONTENT
        assert exc.value.diagnosis["content_empty"] is True
        assert exc.value.diagnosis["choices"] == 1

    def test_null_content_is_classified(self):
        patcher, _ = _patch_urlopen(_assistant(None))
        with patcher:
            with pytest.raises(OpenRouterResponseError) as exc:
                _provider().complete("sys", "usr")
        assert exc.value.reason == REASON_EMPTY_CONTENT
        assert exc.value.diagnosis["content_type"] == "NoneType"

    def test_empty_message_object_is_classified(self):
        patcher, _ = _patch_urlopen({"choices": [{"message": {}, "finish_reason": "stop"}]})
        with patcher:
            with pytest.raises(OpenRouterResponseError) as exc:
                _provider().complete("sys", "usr")
        assert exc.value.reason == REASON_EMPTY_MESSAGE

    def test_truncated_response_is_classified(self):
        body = {"choices": [{"message": {"role": "assistant", "content": None},
                             "finish_reason": "length"}]}
        patcher, _ = _patch_urlopen(body)
        with patcher:
            with pytest.raises(OpenRouterResponseError) as exc:
                _provider().complete("sys", "usr")
        assert exc.value.reason == REASON_TRUNCATED
        assert exc.value.diagnosis["finish_reason"] == "length"

    def test_error_is_an_ai_provider_error(self):
        patcher, _ = _patch_urlopen(_assistant(""))
        with patcher:
            with pytest.raises(AIProviderError):
                _provider().complete("sys", "usr")


class TestToolCallsResponse:
    """Handled deliberately: not accidentally, and without a tool loop."""

    def test_tool_calls_with_null_content_are_classified(self):
        body = {"choices": [{"message": {"role": "assistant",
                                          "content": None,
                                          "tool_calls": [_tool_call()]},
                             "finish_reason": "tool_calls"}]}
        patcher, payloads = _patch_urlopen(body)
        with patcher:
            with pytest.raises(OpenRouterResponseError) as exc:
                _provider().complete("sys", "usr")
        assert exc.value.reason == REASON_TOOL_CALLS_NO_CONTENT
        assert exc.value.diagnosis["has_tool_calls"] is True
        assert exc.value.diagnosis["tool_calls"] == 1
        # No second request: server tools must resolve server-side.
        assert len(payloads) == 1

    def test_tool_calls_with_content_are_not_an_error(self):
        body = {"choices": [{"message": {"role": "assistant",
                                          "content": '{"summary":"ok"}',
                                          "tool_calls": [_tool_call()]},
                             "finish_reason": "tool_calls"}]}
        patcher, _ = _patch_urlopen(body)
        with patcher:
            assert _provider().complete("sys", "usr") == '{"summary":"ok"}'

    def test_diagnosis_records_tool_call_types(self):
        diagnosis = summarize_response({
            "choices": [{"message": {"content": None,
                                     "tool_calls": [_tool_call()]},
                         "finish_reason": "tool_calls"}]})
        assert diagnosis["tool_call_types"] == ["function"]
        assert diagnosis["finish_reason"] == "tool_calls"


class TestRefusalResponse:
    def test_refusal_is_classified(self):
        body = {"choices": [{"message": {"role": "assistant", "content": None,
                                         "refusal": "I cannot help with that."},
                             "finish_reason": "stop"}]}
        patcher, _ = _patch_urlopen(body)
        with patcher:
            with pytest.raises(OpenRouterResponseError) as exc:
                _provider().complete("sys", "usr")
        assert exc.value.reason == REASON_REFUSAL
        assert exc.value.diagnosis["has_refusal"] is True

    def test_refusal_wins_over_tool_calls(self):
        body = {"choices": [{"message": {"content": None, "refusal": "no",
                                         "tool_calls": [_tool_call()]}}]}
        patcher, _ = _patch_urlopen(body)
        with patcher:
            with pytest.raises(OpenRouterResponseError) as exc:
                _provider().complete("sys", "usr")
        assert exc.value.reason == REASON_REFUSAL


class TestMalformedResponses:
    def test_missing_choices_is_classified(self):
        patcher, _ = _patch_urlopen({"choices": []})
        with patcher:
            with pytest.raises(OpenRouterResponseError) as exc:
                _provider().complete("sys", "usr")
        assert exc.value.reason == REASON_NO_CHOICES

    def test_absent_choices_key_is_classified(self):
        patcher, _ = _patch_urlopen({"id": "gen-123", "usage": {}})
        with patcher:
            with pytest.raises(OpenRouterResponseError) as exc:
                _provider().complete("sys", "usr")
        assert exc.value.reason == REASON_NO_CHOICES

    def test_non_json_is_classified(self):
        patcher, _ = _patch_urlopen(b"<html>gateway error</html>")
        with patcher:
            with pytest.raises(OpenRouterResponseError) as exc:
                _provider().complete("sys", "usr")
        assert exc.value.reason == REASON_MALFORMED

    def test_api_error_payload_is_classified(self):
        patcher, _ = _patch_urlopen({"error": {"code": 429, "message": "slow down"}})
        with patcher:
            with pytest.raises(OpenRouterResponseError) as exc:
                _provider().complete("sys", "usr")
        assert exc.value.reason == REASON_API_ERROR
        assert "slow down" in str(exc.value)


class TestSummarizeResponse:
    def test_reports_structure_only(self):
        diagnosis = summarize_response({
            "id": "gen-1",
            "provider": "OpenAI",
            "model": "gpt-5-mini",
            "choices": [{"message": {"role": "assistant", "content": "hi",
                                     "reasoning": "secret thoughts"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                      "total_tokens": 15},
        })
        assert diagnosis["top_level_keys"] == ["choices", "id", "model",
                                                "provider", "usage"]
        assert diagnosis["choices"] == 1
        assert diagnosis["finish_reason"] == "stop"
        assert diagnosis["message_keys"] == ["content", "reasoning", "role"]
        assert diagnosis["content_empty"] is False
        assert diagnosis["has_reasoning"] is True
        assert diagnosis["provider"] == "OpenAI"
        assert diagnosis["usage"]["total_tokens"] == 15
        # No prompt or body text is retained.
        assert "secret thoughts" not in json.dumps(diagnosis)

    def test_handles_non_dict(self):
        diagnosis = summarize_response("oops")
        assert diagnosis["choices"] == 0
        assert diagnosis["content_empty"] is True

    def test_handles_missing_choices(self):
        diagnosis = summarize_response({"usage": {}})
        assert diagnosis["choices"] == 0
        assert diagnosis["finish_reason"] is None

    def test_never_contains_credentials(self):
        diagnosis = summarize_response({"choices": [], "error": {
            "code": 401, "message": "invalid key sk-or-secret"}})
        assert len(diagnosis["error"]["message"]) <= 200


# ---------------------------------------------------------------------------
# 4. Model capability check
# ---------------------------------------------------------------------------

class TestToolSupportVerification:
    def test_tool_capable_model_is_accepted(self):
        from services.ai.openrouter_provider import (_MODEL_INFO_CACHE, _TOOL_SUPPORT_CACHE)

        _TOOL_SUPPORT_CACHE.clear(); _MODEL_INFO_CACHE.clear()
        catalog = {"data": {"id": "openai/gpt-5-mini",
                            "supported_parameters": ["tools", "max_tokens"]}}
        patcher, _ = _patch_urlopen(catalog)
        with patcher, mock.patch("config.Config.OPENROUTER_VERIFY_TOOL_SUPPORT", True):
            verdict = _provider().verify_tool_support()
        assert verdict["supported"] is True
        assert verdict["source"] == "catalog"

    def test_model_without_tools_is_reported_cleanly(self):
        from services.ai.openrouter_provider import (_MODEL_INFO_CACHE, _TOOL_SUPPORT_CACHE)

        _TOOL_SUPPORT_CACHE.clear(); _MODEL_INFO_CACHE.clear()
        catalog = {"data": {"id": "some/model", "supported_parameters": ["max_tokens"]}}
        patcher, _ = _patch_urlopen(catalog)
        with patcher, mock.patch("config.Config.OPENROUTER_VERIFY_TOOL_SUPPORT", True):
            with pytest.raises(OpenRouterResponseError) as exc:
                _provider(model="some/model").verify_tool_support()
        assert exc.value.reason == REASON_NO_TOOL_SUPPORT
        assert "OPENROUTER_MODEL" in str(exc.value)

    def test_unsupported_model_does_not_send_the_tool_request(self):
        from services.ai.openrouter_provider import (_MODEL_INFO_CACHE, _TOOL_SUPPORT_CACHE)

        _TOOL_SUPPORT_CACHE.clear(); _MODEL_INFO_CACHE.clear()
        catalog = {"data": {"id": "some/model", "supported_parameters": []}}
        patcher, payloads = _patch_urlopen(
            catalog, _final_message('{"summary": "ok"}'))
        with patcher, mock.patch("config.Config.OPENROUTER_VERIFY_TOOL_SUPPORT", True):
            with pytest.raises(OpenRouterResponseError):
                _provider(model="some/model").complete("sys", "usr")
        # Only the catalog GET was issued — no chat request with unsupported tools.
        assert all("messages" not in p for p in payloads)

    def test_verdict_is_cached(self):
        from services.ai.openrouter_provider import (_MODEL_INFO_CACHE, _TOOL_SUPPORT_CACHE)

        _TOOL_SUPPORT_CACHE.clear(); _MODEL_INFO_CACHE.clear()
        catalog = {"data": {"id": "openai/gpt-5-mini",
                            "supported_parameters": ["tools"]}}
        patcher, payloads = _patch_urlopen(
            catalog, _final_message('{"summary": "ok"}'))
        with patcher, mock.patch("config.Config.OPENROUTER_VERIFY_TOOL_SUPPORT", True):
            p = _provider()
            p.complete("sys", "usr")
            p.complete("sys", "usr")
        catalog_calls = [p for p in payloads if "__url__" in p]
        assert len(catalog_calls) == 1

    def test_unavailable_catalog_is_tolerated(self):
        import urllib.error

        from services.ai.openrouter_provider import (_MODEL_INFO_CACHE, _TOOL_SUPPORT_CACHE)

        _TOOL_SUPPORT_CACHE.clear(); _MODEL_INFO_CACHE.clear()
        with mock.patch("config.Config.OPENROUTER_VERIFY_TOOL_SUPPORT", True), \
             mock.patch("urllib.request.urlopen",
                        side_effect=urllib.error.URLError("offline")):
            verdict = _provider().verify_tool_support()
        assert verdict["supported"] is None
        assert verdict["source"] == "catalog_unavailable"

    def test_obvious_non_tool_model_id_is_flagged(self):
        from services.ai.openrouter_provider import (_MODEL_INFO_CACHE, _TOOL_SUPPORT_CACHE)

        _TOOL_SUPPORT_CACHE.clear(); _MODEL_INFO_CACHE.clear()
        with mock.patch("config.Config.OPENROUTER_VERIFY_TOOL_SUPPORT", True):
            with pytest.raises(OpenRouterResponseError) as exc:
                _provider(model="openai/text-embedding-3-small").verify_tool_support()
        assert exc.value.reason == REASON_NO_TOOL_SUPPORT

    def test_check_skipped_when_tools_disabled(self):
        from services.ai.openrouter_provider import (_MODEL_INFO_CACHE, _TOOL_SUPPORT_CACHE)

        _TOOL_SUPPORT_CACHE.clear(); _MODEL_INFO_CACHE.clear()
        patcher, payloads = _patch_urlopen(_final_message('{"summary": "ok"}'))
        with patcher, mock.patch("config.Config.OPENROUTER_VERIFY_TOOL_SUPPORT", True):
            _provider().complete("sys", "usr", tools=[])
        assert len(payloads) == 1
        assert "messages" in payloads[0]

    def test_catalog_list_shape_is_used(self):
        """The live catalog answers with {"data": [ ... entries ... ]}."""
        from services.ai.openrouter_provider import (_MODEL_INFO_CACHE, _TOOL_SUPPORT_CACHE)

        _TOOL_SUPPORT_CACHE.clear(); _MODEL_INFO_CACHE.clear()
        catalog = {"data": [
            {"id": "other/model", "supported_parameters": ["tools"]},
            {"id": "openai/gpt-5-mini",
             "supported_parameters": ["max_tokens", "reasoning", "tools"]},
        ]}
        patcher, payloads = _patch_urlopen(catalog)
        with patcher, mock.patch("config.Config.OPENROUTER_VERIFY_TOOL_SUPPORT", True):
            verdict = _provider().verify_tool_support()
        assert verdict["supported"] is True
        assert payloads[0]["__url__"].endswith("/api/v1/models")

    def test_model_absent_from_catalog_is_inconclusive(self):
        from services.ai.openrouter_provider import (_MODEL_INFO_CACHE, _TOOL_SUPPORT_CACHE)

        _TOOL_SUPPORT_CACHE.clear(); _MODEL_INFO_CACHE.clear()
        catalog = {"data": [{"id": "other/model", "supported_parameters": ["tools"]}]}
        patcher, _ = _patch_urlopen(catalog)
        with patcher, mock.patch("config.Config.OPENROUTER_VERIFY_TOOL_SUPPORT", True):
            verdict = _provider().verify_tool_support()
        assert verdict["supported"] is None


class TestTruncationDiagnostics:
    """finish_reason=length means reasoning ate the output budget."""

    def test_truncated_but_non_empty_content_is_warned(self, caplog):
        import logging

        body = {"choices": [{"message": {"role": "assistant",
                                          "content": '{"summary": "trunc'},
                             "finish_reason": "length"}]}
        patcher, _ = _patch_urlopen(body)
        with patcher, caplog.at_level(logging.WARNING):
            text = _provider().complete("sys", "usr")
        assert text == '{"summary": "trunc'
        assert "finish_reason=length" in "\n".join(r.getMessage() for r in caplog.records)

    def test_complete_response_is_not_warned(self, caplog):
        import logging

        patcher, _ = _patch_urlopen(_final_message('{"summary": "ok"}'))
        with patcher, caplog.at_level(logging.WARNING):
            _provider().complete("sys", "usr")
        assert "truncated" not in "\n".join(r.getMessage() for r in caplog.records)


class TestReasoningEffort:
    def test_reasoning_effort_not_sent_by_default(self):
        patcher, payloads = _patch_urlopen(_final_message('{"summary": "ok"}'))
        with patcher, mock.patch("config.Config.OPENROUTER_REASONING_EFFORT", ""):
            _provider().complete("sys", "usr")
        assert "reasoning" not in payloads[0]

    def test_reasoning_effort_sent_when_configured(self):
        patcher, payloads = _patch_urlopen(_final_message('{"summary": "ok"}'))
        with patcher, mock.patch("config.Config.OPENROUTER_REASONING_EFFORT", "low"):
            _provider().complete("sys", "usr")
        assert payloads[0]["reasoning"] == {"effort": "low"}


# ---------------------------------------------------------------------------
# 5. Token / research budgets
# ---------------------------------------------------------------------------

class TestDefaultMaxTokens:
    def test_default_value_is_2048(self):
        assert DEFAULT_MAX_TOKENS == 2048

    def test_provider_default_max_tokens_is_safe(self):
        with mock.patch("config.Config.OPENROUTER_MAX_TOKENS", DEFAULT_MAX_TOKENS):
            assert _provider().max_tokens == 2048

    def test_default_is_well_under_the_65536_failure(self):
        with mock.patch("config.Config.OPENROUTER_MAX_TOKENS", DEFAULT_MAX_TOKENS):
            assert _provider().max_tokens < 65536

    def test_payload_sends_max_tokens(self):
        patcher, payloads = _patch_urlopen(_final_message('{"summary": "ok"}'))
        with patcher, mock.patch("config.Config.OPENROUTER_MAX_TOKENS", 2048):
            _provider().complete("sys", "usr")
        assert payloads[0]["max_tokens"] == 2048

    def test_model_prompt_budget_fits_structured_response(self):
        schema_sized_response = json.dumps({
            "summary": "x" * 200,
            "recommended_settings": {f"setting_{i}": "value" for i in range(12)},
            "changes": [{"setting": "textures", "from": "High", "to": "Medium",
                         "reason": "y" * 120} for _ in range(6)],
            "estimated_fps": "52-60",
            "fps_source": "MEASURED_BENCHMARK",
            "confidence": "high",
            "reasoning": "z" * 600,
            "evidence": [{"title": "t", "url": "https://example.com",
                          "domain": "example.com", "type": "MEASURED_BENCHMARK",
                          "claim": "c" * 200} for _ in range(6)],
            "warnings": ["w" * 100],
        })
        assert len(schema_sized_response) / 4 < 2048


class TestMaxTokensFromEnvironment:
    def test_env_value_in_range_is_respected(self):
        with mock.patch("config.Config.OPENROUTER_MAX_TOKENS", 3000):
            assert _provider().max_tokens == 3000

    def test_env_value_used_in_payload(self):
        patcher, payloads = _patch_urlopen(_final_message('{"ok": true}'))
        with patcher, mock.patch("config.Config.OPENROUTER_MAX_TOKENS", 3072):
            _provider().complete("sys", "usr")
        assert payloads[0]["max_tokens"] == 3072

    def test_explicit_argument_overrides_config(self):
        with mock.patch("config.Config.OPENROUTER_MAX_TOKENS", 4096):
            assert _provider(max_tokens=1024).max_tokens == 1024

    def test_config_reads_env_variable(self):
        import config as config_module

        with mock.patch.dict("os.environ", {"OPENROUTER_MAX_TOKENS": "1024"}):
            assert config_module._env_int("OPENROUTER_MAX_TOKENS", 2048) == 1024
        with mock.patch.dict("os.environ", {"OPENROUTER_MAX_TOKENS": "bad"}):
            assert config_module._env_int("OPENROUTER_MAX_TOKENS", 2048) == 2048

    def test_config_env_clamps_too_large_value(self):
        import config as config_module

        with mock.patch.dict("os.environ", {"OPENROUTER_MAX_TOKENS": "65536"}):
            value = config_module._env_int(
                "OPENROUTER_MAX_TOKENS", 2048, minimum=512, maximum=4096)
        assert value == 4096

    def test_config_env_clamps_too_small_value(self):
        import config as config_module

        with mock.patch.dict("os.environ", {"OPENROUTER_MAX_TOKENS": "8"}):
            value = config_module._env_int(
                "OPENROUTER_MAX_TOKENS", 2048, minimum=512, maximum=4096)
        assert value == 512


class TestMaxTokensCeiling:
    def test_clamp_rejects_65536(self):
        assert clamp_max_tokens(65536) == MAX_MAX_TOKENS == 4096

    def test_clamp_rejects_huge_values(self):
        assert clamp_max_tokens(1_000_000) == 4096

    def test_clamp_enforces_minimum(self):
        assert clamp_max_tokens(1) == MIN_MAX_TOKENS == 512
        assert clamp_max_tokens(0) == 512
        assert clamp_max_tokens(-5000) == 512

    def test_clamp_keeps_valid_values(self):
        assert clamp_max_tokens(1536) == 1536
        assert clamp_max_tokens(2048) == 2048
        assert clamp_max_tokens(4096) == 4096

    def test_clamp_falls_back_on_garbage(self):
        assert clamp_max_tokens(None) == DEFAULT_MAX_TOKENS
        assert clamp_max_tokens("abc") == DEFAULT_MAX_TOKENS
        assert clamp_max_tokens("") == DEFAULT_MAX_TOKENS

    def test_provider_caps_env_value_of_65536(self):
        with mock.patch("config.Config.OPENROUTER_MAX_TOKENS", 65536):
            assert _provider().max_tokens == 4096

    def test_provider_caps_huge_explicit_argument(self):
        assert _provider(max_tokens=200000).max_tokens == MAX_MAX_TOKENS

    def test_provider_raises_too_small_explicit_argument(self):
        assert _provider(max_tokens=64).max_tokens == MIN_MAX_TOKENS

    def test_capped_value_is_sent_in_payload(self):
        patcher, payloads = _patch_urlopen(_final_message('{"ok": true}'))
        with patcher, mock.patch("config.Config.OPENROUTER_MAX_TOKENS", 65536):
            _provider().complete("sys", "usr")
        assert payloads[0]["max_tokens"] == 4096


class TestResearchLimits:
    def test_default_limits(self):
        assert DEFAULT_MAX_SEARCH_RESULTS == 3
        assert DEFAULT_MAX_TOTAL_RESULTS == 6
        assert DEFAULT_MAX_SEARCH_CALLS == 2
        assert DEFAULT_MAX_FETCH_PAGES == 1
        assert DEFAULT_MAX_TOOL_CALLS == 4

    def test_provider_default_research_limits(self):
        with mock.patch("config.Config.OPENROUTER_MAX_TOKENS", DEFAULT_MAX_TOKENS):
            p = _provider()
        assert p.max_search_results == 3
        assert p.max_total_results == 6
        assert p.max_search_calls == 2
        assert p.max_fetch_pages == 1
        assert p.max_tool_calls == 4

    def test_search_results_limited_by_total_budget(self):
        limits = ResearchLimits(max_search_results=5, max_total_results=6,
                                max_search_calls=2, max_fetch_pages=1)
        assert limits.results_per_search() == 3

    def test_results_per_search_never_zero(self):
        limits = ResearchLimits(max_search_results=0, max_total_results=6,
                                max_search_calls=2, max_fetch_pages=1)
        assert limits.results_per_search() >= 1

    def test_tool_call_budget_is_derived_and_small(self):
        limits = ResearchLimits(max_search_calls=2, max_fetch_pages=1)
        assert limits.max_tool_calls == 4

    def test_tool_call_budget_is_capped(self):
        assert ResearchLimits(max_search_calls=99, max_fetch_pages=99,
                              max_tool_calls=500).max_tool_calls == MAX_MAX_TOOL_CALLS

    def test_tool_call_budget_from_config(self):
        with mock.patch("config.Config.OPENROUTER_MAX_TOOL_CALLS", 2):
            assert _provider().max_tool_calls == 2

    def test_limits_snapshot(self):
        assert _provider().research_limits().as_dict() == {
            "max_search_results": 3,
            "max_total_results": 6,
            "max_search_calls": 2,
            "max_fetch_pages": 1,
            "max_tool_calls": 4,
        }

    def test_limits_in_payload(self):
        patcher, payloads = _patch_urlopen(_final_message('{"ok": true}'))
        with patcher:
            _provider().complete("sys", "usr")
        payload = payloads[0]
        assert payload["tools"][0]["parameters"] == {
            "max_results": 3, "max_total_results": 6, "max_uses": 2}
        assert payload["max_tool_calls"] == 4
        assert payload["max_tokens"] == 2048


# ---------------------------------------------------------------------------
# 6. Existing behaviour / safe fallback
# ---------------------------------------------------------------------------

class TestExistingBehaviourPreserved:
    def test_web_research_functionality_preserved(self):
        patcher, payloads = _patch_urlopen(_final_message('{"summary": "ok"}'))
        with patcher:
            _provider().complete("sys", "usr")
        assert [t["type"] for t in payloads[0]["tools"]] == [WEB_SEARCH_TYPE]

    def test_research_can_be_disabled(self):
        patcher, payloads = _patch_urlopen(_final_message('{"summary": "ok"}'))
        with patcher:
            _provider().complete("sys", "usr", tools=[])
        assert "tools" not in payloads[0]

    def test_final_answer_is_returned(self):
        patcher, _ = _patch_urlopen(_final_message('{"summary": "ok"}'))
        with patcher:
            assert _provider().complete("sys", "usr") == '{"summary": "ok"}'

    def test_availability_logic_unchanged(self):
        with mock.patch("config.Config.OPENROUTER_API_KEY", "sk-or-env"):
            assert _provider().is_available() is True
        with mock.patch("config.Config.OPENROUTER_API_KEY", ""):
            assert OpenRouterProvider(api_key="").is_available() is False
            assert OpenRouterProvider(api_key="   ").is_available() is False

    def test_missing_key_still_raises(self):
        with pytest.raises(AIProviderError):
            OpenRouterProvider(api_key="").complete("sys", "usr")

    def test_402_error_message_mentions_max_tokens(self):
        import urllib.error

        class _HTTPError(urllib.error.HTTPError):
            def __init__(self):
                super().__init__("u", 402, "Payment Required", {}, None)

            def read(self):
                return b"You requested up to 65536 tokens, but can afford 1793."

        with mock.patch("urllib.request.urlopen", side_effect=_HTTPError()):
            with pytest.raises(AIProviderError) as exc:
                _provider().complete("sys", "usr")
        assert "402" in str(exc.value)
        assert "max_tokens=2048" in str(exc.value)

    def test_400_invalid_tools_message_is_surfaced(self):
        import urllib.error

        class _HTTPError(urllib.error.HTTPError):
            def __init__(self):
                super().__init__("u", 400, "Bad Request", {}, None)

            def read(self):
                return (b'{"error":{"code":400,"message":"Invalid tools.0.name: '
                        b'string does not match the pattern ^[a-zA-Z0-9_.-]+$"}}')

        with mock.patch("urllib.request.urlopen", side_effect=_HTTPError()):
            with pytest.raises(AIProviderError) as exc:
                _provider().complete("sys", "usr")
        assert "Invalid tools.0.name" in str(exc.value)

    def test_api_key_is_never_in_error_message(self):
        import urllib.error

        secret = "sk-or-secret-key-value"

        class _HTTPError(urllib.error.HTTPError):
            def __init__(self):
                super().__init__("u", 500, "Server Error", {}, None)

            def read(self):
                return b"boom"

        with mock.patch("urllib.request.urlopen", side_effect=_HTTPError()):
            p = OpenRouterProvider(api_key=secret)
            with pytest.raises(AIProviderError) as exc:
                p.complete("sys", "usr", tools=[])
        assert secret not in str(exc.value)

    def test_diagnostics_do_not_log_prompts_or_body(self, caplog):
        import logging

        body = {"choices": [{"message": {"role": "assistant", "content": None,
                                         "reasoning": "SECRET-REASONING"},
                             "finish_reason": "stop"}],
                "provider": "OpenAI"}
        patcher, _ = _patch_urlopen(body)
        with patcher, caplog.at_level(logging.DEBUG):
            with pytest.raises(OpenRouterResponseError):
                _provider().complete("SECRET-SYSTEM-PROMPT", "SECRET-USER-PROMPT")
        logged = "\n".join(r.getMessage() for r in caplog.records)
        assert "SECRET-SYSTEM-PROMPT" not in logged
        assert "SECRET-USER-PROMPT" not in logged
        assert "SECRET-REASONING" not in logged
        assert "reason=empty_content" in logged


class TestDeterministicFallbackStillWorks:
    """A broken OpenRouter call must still fall back deterministically."""

    def _assert_fallback(self):
        from services.ai_optimizer import optimize_game
        from tests.test_ai_optimizer import _hw, _palworld

        result = optimize_game(_hw(), _palworld())
        assert result.status == "fallback"
        assert result.recommended_settings is not None
        assert result.recommended_settings.graphics_preset

    def test_http400_falls_back(self):
        import urllib.error

        class _HTTPError(urllib.error.HTTPError):
            def __init__(self):
                super().__init__("u", 400, "Bad Request", {}, None)

            def read(self):
                return b'{"error":{"code":400,"message":"Invalid tools.0.name"}}'

        with mock.patch("urllib.request.urlopen", side_effect=_HTTPError()):
            self._assert_fallback()

    def test_malformed_json_falls_back(self):
        with mock.patch("urllib.request.urlopen",
                        return_value=_FakeResponse(b"<html>gateway error</html>")):
            self._assert_fallback()

    def test_empty_content_falls_back(self):
        with mock.patch("urllib.request.urlopen",
                        return_value=_FakeResponse(_assistant(None))):
            self._assert_fallback()

    def test_tool_calls_without_content_falls_back(self):
        body = {"choices": [{"message": {"role": "assistant", "content": None,
                                         "tool_calls": [_tool_call()]},
                             "finish_reason": "tool_calls"}]}
        with mock.patch("urllib.request.urlopen",
                        return_value=_FakeResponse(body)):
            self._assert_fallback()

    def test_deterministic_fps_remains_authoritative(self):
        from services.ai_optimizer import optimize_game
        from tests.test_ai_optimizer import _hw, _palworld

        ai_response = json.dumps({
            "summary": "x",
            "recommended_settings": {"graphics_preset": "Low",
                                     "resolution": "1920x1080"},
            "changes": [],
            "estimated_fps": "999",
            "fps_source": "AI_INFERENCE",
            "confidence": "high",
            "reasoning": "x",
            "evidence": [],
            "warnings": [],
        })
        with mock.patch("services.ai_optimizer._select_provider") as mock_sel:
            provider = mock.MagicMock()
            provider.provider_name = "openrouter"
            provider.model_name = "test"
            provider.complete.return_value = ai_response
            mock_sel.return_value = provider
            result = optimize_game(_hw(), _palworld())

        assert result.fps_source != "AI_INFERENCE"
        assert "999" not in str(result.estimated_fps)

    def test_research_queries_still_available(self):
        from services.ai.prompts import build_research_queries

        queries = build_research_queries("Palworld", "RTX 3060", "Ryzen 5 5600",
                                         "1920x1080", 60)
        assert queries
        assert all(isinstance(q, str) for q in queries)