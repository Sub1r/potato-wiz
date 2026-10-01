"""
test_ai_optimizer.py
Tests for the AI optimizer pipeline.

Test cases covered:
 1. AI response schema validation
 2. Malformed AI JSON
 3. Invalid setting value
 4. Unsupported resolution
 5. Missing evidence
 6. Missing source URL in evidence
 7. Benchmark data included in AI context
 8. AI never overrides deterministic benchmark FPS
 9. OpenRouter unavailable
10. Ollama unavailable
11. Both providers unavailable → fallback optimizer works
12. Fallback optimizer works
13. No API key in logs
14. Prompt contains hardware information
15. Prompt contains game information
16. Source evidence survives parsing
17. Fixture benchmark records are excluded
18. Game-specific supported settings are respected
"""
import json
import logging
import os
import sys
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from models.models import (
    AIOptimizationResult,
    AIRecommendedSettings,
    EvidenceItem,
    HardwareSpecs,
    AI_STATUS_OK,
    AI_STATUS_FALLBACK,
    AI_PROVIDER_NONE,
    AI_PROVIDER_OLLAMA,
    AI_PROVIDER_OPENROUTER,
    EVIDENCE_FALLBACK_ESTIMATE,
    EVIDENCE_MEASURED_BENCHMARK,
    EVIDENCE_AI_INFERENCE,
    EVIDENCE_COMMUNITY_REPORT,
)
from services.ai.base import AIProviderError
from services.ai.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    build_benchmark_context,
    build_research_queries,
)
from services.ai_optimizer import (
    optimize_game,
    _parse_ai_response,
    _validate_settings,
    _safe_fps,
    _parse_evidence,
    _wrap_deterministic,
    _select_provider,
)
from services.benchmark_service import clear_cache
from services.game_service import get_game_by_slug


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hw(
    gpu="NVIDIA GeForce GTX 1650",
    cpu="AMD Ryzen 5 3600",
    ram=16,
    vram=4,
):
    return HardwareSpecs(gpu=gpu, cpu=cpu, ram_gb=ram, vram_gb=vram)


def _good_ai_json(**overrides):
    """Return a complete, valid AI response dict."""
    base = {
        "summary": "Reduce shadows and enable FSR for better FPS.",
        "recommended_settings": {
            "graphics_preset": "Medium",
            "resolution": "1920x1080",
            "upscaling": "FSR Balanced",
            "view_distance": "High",
            "shadows": "Low",
            "effects": "Medium",
            "textures": "High",
            "anti_aliasing": "TAA",
            "motion_blur": "Off",
            "vsync": "Off",
            "fps_limit": 60,
        },
        "changes": [
            {"setting": "Shadows", "from": "High", "to": "Low", "reason": "Large FPS gain."},
        ],
        "estimated_fps": "",
        "fps_source": "FALLBACK_ESTIMATE",
        "confidence": "medium",
        "reasoning": "Based on GPU tier and game engine demands.",
        "evidence": [
            {
                "title": "TechPowerUp GTX 1650 Review",
                "url": "https://www.techpowerup.com/review/gtx-1650",
                "domain": "techpowerup.com",
                "type": "PUBLISHED_BENCHMARK",
                "claim": "GTX 1650 achieves ~45 FPS in demanding open-world games at 1080p High.",
            }
        ],
        "warnings": [],
    }
    base.update(overrides)
    return base


def _palworld():
    return get_game_by_slug("palworld")


# ---------------------------------------------------------------------------
# 1. AI response schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    """Test 1: _validate_settings correctly validates AI response schema."""

    def test_valid_json_accepted(self):
        """A well-formed AI JSON dict passes validation without warnings."""
        good = _good_ai_json()
        warnings = []
        settings = _validate_settings(
            raw=good["recommended_settings"],
            fallback=None,
            resolution="1920x1080",
            game=_palworld() or {},
            warnings=warnings,
        )
        assert isinstance(settings, AIRecommendedSettings)
        assert settings.graphics_preset == "Medium"

    def test_all_required_fields_present(self):
        """Validated settings have all required fields populated."""
        warnings = []
        settings = _validate_settings(
            raw=_good_ai_json()["recommended_settings"],
            fallback=None,
            resolution="1920x1080",
            game={},
            warnings=warnings,
        )
        assert settings.resolution
        assert settings.upscaling is not None
        assert settings.shadows
        assert settings.textures

    def test_result_to_dict_schema(self):
        """AIOptimizationResult.to_dict() contains all expected keys."""
        r = AIOptimizationResult()
        d = r.to_dict()
        for key in ("status", "provider", "model", "summary", "recommended_settings",
                    "changes", "estimated_fps", "fps_source", "confidence",
                    "reasoning", "evidence", "warnings", "fallback"):
            assert key in d, f"Missing key in to_dict(): {key}"


# ---------------------------------------------------------------------------
# 2. Malformed AI JSON
# ---------------------------------------------------------------------------

class TestMalformedAiJson:
    """Test 2: _parse_ai_response handles malformed / non-JSON output."""

    def test_clean_json_parsed(self):
        data = {"summary": "ok"}
        parsed, warning = _parse_ai_response(json.dumps(data))
        assert parsed == data
        assert warning is None

    def test_json_in_markdown_fence(self):
        raw = '```json\n{"summary": "from fence"}\n```'
        parsed, warning = _parse_ai_response(raw)
        assert parsed is not None
        assert parsed["summary"] == "from fence"

    def test_json_embedded_in_prose(self):
        raw = 'Here is my response: {"summary": "embedded"} and some trailing text.'
        parsed, warning = _parse_ai_response(raw)
        assert parsed is not None
        assert parsed["summary"] == "embedded"
        assert warning  # warns that it wasn't clean JSON

    def test_completely_invalid_returns_none(self):
        parsed, warning = _parse_ai_response("This is just plain text with no JSON.")
        assert parsed is None
        assert warning

    def test_empty_string_returns_none(self):
        parsed, warning = _parse_ai_response("")
        assert parsed is None
        assert warning

    def test_optimize_game_survives_malformed_response(self):
        """optimize_game() falls back gracefully when AI returns garbage."""
        game = _palworld()
        hw = _hw()
        with mock.patch("services.ai_optimizer._select_provider") as mock_sel:
            prov = mock.MagicMock()
            prov.provider_name = "test"
            prov.model_name = "test-model"
            prov.complete.return_value = "This is NOT valid JSON!!!!"
            mock_sel.return_value = prov
            result = optimize_game(hw, game)
        assert isinstance(result, AIOptimizationResult)
        # Should fall back, not crash
        assert result.recommended_settings is not None


# ---------------------------------------------------------------------------
# 3. Invalid setting value
# ---------------------------------------------------------------------------

class TestInvalidSettingValue:
    """Test 3: Invalid setting values are rejected and replaced with fallback."""

    def test_invalid_preset_replaced(self):
        warnings = []
        raw = {**_good_ai_json()["recommended_settings"], "graphics_preset": "ULTRA_MAX_9999"}
        settings = _validate_settings(raw, None, "1920x1080", {}, warnings)
        # Should have fallen back to a valid value
        from services.ai_optimizer import _VALID_PRESETS
        assert settings.graphics_preset in _VALID_PRESETS
        assert any("graphics_preset" in w for w in warnings)

    def test_invalid_shadows_replaced(self):
        warnings = []
        raw = {**_good_ai_json()["recommended_settings"], "shadows": "FakeShadows"}
        settings = _validate_settings(raw, None, "1920x1080", {}, warnings)
        from services.ai_optimizer import _VALID_SHADOWS
        assert settings.shadows in _VALID_SHADOWS
        assert any("shadows" in w for w in warnings)

    def test_invalid_aa_replaced(self):
        warnings = []
        raw = {**_good_ai_json()["recommended_settings"], "anti_aliasing": "DLSS_QUANTUM"}
        settings = _validate_settings(raw, None, "1920x1080", {}, warnings)
        from services.ai_optimizer import _VALID_AA
        assert settings.anti_aliasing in _VALID_AA


# ---------------------------------------------------------------------------
# 4. Unsupported resolution
# ---------------------------------------------------------------------------

class TestUnsupportedResolution:
    """Test 4: AI-suggested unsupported resolutions are replaced."""

    def test_fake_resolution_replaced_with_user_resolution(self):
        warnings = []
        raw = {**_good_ai_json()["recommended_settings"], "resolution": "1337x420"}
        settings = _validate_settings(raw, None, "1920x1080", {}, warnings)
        assert settings.resolution == "1920x1080"
        assert any("resolution" in w.lower() for w in warnings)

    def test_valid_resolution_kept(self):
        warnings = []
        raw = {**_good_ai_json()["recommended_settings"], "resolution": "2560x1440"}
        settings = _validate_settings(raw, None, "1920x1080", {}, warnings)
        assert settings.resolution == "2560x1440"


# ---------------------------------------------------------------------------
# 5. Missing evidence
# ---------------------------------------------------------------------------

class TestMissingEvidence:
    """Test 5: Missing or empty evidence is handled gracefully."""

    def test_empty_evidence_list(self):
        items = _parse_evidence([])
        assert items == []

    def test_none_evidence_handled(self):
        items = _parse_evidence(None)
        assert items == []

    def test_result_with_no_evidence_still_valid(self):
        """optimize_game() with fallback (no AI) returns valid result even with no evidence."""
        game = _palworld()
        hw = _hw()
        with mock.patch("services.ai_optimizer._select_provider", return_value=None):
            result = optimize_game(hw, game)
        assert isinstance(result, AIOptimizationResult)
        assert result.recommended_settings is not None


# ---------------------------------------------------------------------------
# 6. Missing source URL in evidence
# ---------------------------------------------------------------------------

class TestMissingSourceUrl:
    """Test 6: Evidence items without a URL are stripped from results."""

    def test_evidence_without_url_stripped(self):
        raw = [
            {"title": "No URL entry", "url": "", "type": "GUIDE", "claim": "something"},
            {"title": "With URL",     "url": "https://example.com", "type": "GUIDE", "claim": "x"},
        ]
        items = _parse_evidence(raw)
        assert len(items) == 1
        assert items[0].url == "https://example.com"

    def test_evidence_with_none_url_stripped(self):
        raw = [{"title": "None URL", "url": None, "type": "AI_INFERENCE", "claim": "y"}]
        items = _parse_evidence(raw)
        assert items == []

    def test_evidence_url_preserved(self):
        raw = [{"title": "T", "url": "https://valid.com/page", "type": "GUIDE", "claim": "c"}]
        items = _parse_evidence(raw)
        assert items[0].url == "https://valid.com/page"


# ---------------------------------------------------------------------------
# 7. Benchmark data included in AI context
# ---------------------------------------------------------------------------

class TestBenchmarkInContext:
    """Test 7: Benchmark data from benchmark_service is included in the AI prompt."""

    def setup_method(self):
        clear_cache()

    def test_benchmark_context_included_in_prompt(self):
        """When a benchmark match exists, it appears in the user prompt."""
        import tempfile
        from services import benchmark_service

        prod_record = {
            "gpu": "NVIDIA GeForce GTX 1650",
            "cpu": "AMD Ryzen 5 3600",
            "resolution": "1920x1080",
            "preset": "Low",
            "avg_fps": 48,
            "one_percent_low": 36,
            "source": "TechPowerUp",
            "source_url": "https://www.techpowerup.com/",
            "confidence": "high",
        }
        data = {"game": "palworld", "benchmarks": [prod_record]}
        original_dir = benchmark_service._BENCHMARKS_DIR
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "palworld.json")
        with open(path, "w") as f:
            json.dump(data, f)
        benchmark_service._BENCHMARKS_DIR = tmpdir

        try:
            from services.ai.research import build_research_context
            hw = _hw()
            game = _palworld()
            ctx = build_research_context(hw, game, preset="Low",
                                         resolution="1920x1080", upscaling_mode="Off")
            assert ctx.benchmark_context_text.strip() != "", (
                "Benchmark context text should not be empty when a match is found"
            )
            assert "48" in ctx.benchmark_context_text  # avg_fps present
        finally:
            benchmark_service._BENCHMARKS_DIR = original_dir
            clear_cache()

    def test_no_benchmark_context_when_no_match(self):
        """When no benchmark exists, context text is empty (no invented data)."""
        from services.ai.research import build_research_context
        hw = _hw(gpu="NVIDIA GeForce RTX 9999 Ti")  # no match
        game = _palworld()
        ctx = build_research_context(hw, game, preset="High",
                                     resolution="1920x1080", upscaling_mode="Off")
        assert ctx.benchmark_context_text == ""


# ---------------------------------------------------------------------------
# 8. AI never overrides deterministic benchmark FPS
# ---------------------------------------------------------------------------

class TestAiNeverOverridesFps:
    """Test 8: FPS values come from deterministic sources, never from AI invention."""

    def test_safe_fps_uses_deterministic_meta(self):
        """_safe_fps() returns the deterministic estimate when available."""
        ai_json = {"estimated_fps": "999", "fps_source": "AI_INFERENCE"}

        class FakeCtx:
            deterministic_fps_meta = {
                "fps_low": 42, "fps_high": 55, "source_type": "estimated"
            }

        fps_str, source = _safe_fps(ai_json, FakeCtx())
        assert "42" in fps_str or "55" in fps_str
        assert "999" not in fps_str  # AI-invented number not used
        assert source == EVIDENCE_FALLBACK_ESTIMATE

    def test_safe_fps_uses_benchmark_when_available(self):
        """When the deterministic path found a benchmark, source is MEASURED_BENCHMARK."""
        ai_json = {"estimated_fps": "", "fps_source": ""}

        class FakeCtx:
            deterministic_fps_meta = {
                "fps_low": 44, "fps_high": 52, "source_type": "benchmark"
            }

        fps_str, source = _safe_fps(ai_json, FakeCtx())
        assert source == EVIDENCE_MEASURED_BENCHMARK

    def test_safe_fps_empty_when_no_context(self):
        """No context → empty fps string."""
        fps_str, source = _safe_fps({}, None)
        assert fps_str == ""
        assert source == EVIDENCE_FALLBACK_ESTIMATE

    def test_optimize_game_fps_not_invented(self):
        """Full optimize_game() — fps_source is never AI_INFERENCE."""
        game = _palworld()
        hw = _hw()
        with mock.patch("services.ai_optimizer._select_provider") as mock_sel:
            prov = mock.MagicMock()
            prov.provider_name = "openrouter"
            prov.model_name = "test"
            ai_resp = _good_ai_json()
            ai_resp["estimated_fps"] = "999"
            ai_resp["fps_source"] = "AI_INFERENCE"
            prov.complete.return_value = json.dumps(ai_resp)
            mock_sel.return_value = prov
            result = optimize_game(hw, game)
        # fps_source must NOT be AI_INFERENCE
        assert result.fps_source != EVIDENCE_AI_INFERENCE
        # The invented 999 must NOT appear in the result
        assert "999" not in str(result.estimated_fps)


# ---------------------------------------------------------------------------
# 9. OpenRouter unavailable
# ---------------------------------------------------------------------------

class TestOpenRouterUnavailable:
    """Test 9: OpenRouter unavailable falls back gracefully.

    Also covers the sentinel distinction between api_key=None (not provided,
    load from env) and api_key="" / api_key="   " (explicitly empty, do NOT
    fall back to env).
    """

    # ── Availability logic ────────────────────────────────────────────────────

    def test_explicit_empty_string_is_not_available(self):
        """api_key="" must NOT fall back to env and must return is_available=False."""
        from services.ai.openrouter_provider import OpenRouterProvider
        # Patch env to a real-looking key so we can prove it is NOT used
        with mock.patch("config.Config.OPENROUTER_API_KEY", "sk-or-env-key-99"):
            p = OpenRouterProvider(api_key="")
        assert p.is_available() is False

    # Alias kept so the original test name still exists
    test_no_api_key_is_not_available = test_explicit_empty_string_is_not_available

    def test_whitespace_only_key_is_not_available(self):
        """api_key='   ' (whitespace only) must be treated as absent."""
        from services.ai.openrouter_provider import OpenRouterProvider
        with mock.patch("config.Config.OPENROUTER_API_KEY", "sk-or-env-key-99"):
            p = OpenRouterProvider(api_key="   ")
        assert p.is_available() is False

    def test_no_argument_loads_from_env(self):
        """api_key=None (default) must load from Config.OPENROUTER_API_KEY."""
        from services.ai.openrouter_provider import OpenRouterProvider
        with mock.patch("config.Config.OPENROUTER_API_KEY", "sk-or-real-env-key"):
            p = OpenRouterProvider()   # no api_key argument
        assert p.is_available() is True

    def test_no_argument_no_env_is_not_available(self):
        """api_key=None with empty env key → not available."""
        from services.ai.openrouter_provider import OpenRouterProvider
        with mock.patch("config.Config.OPENROUTER_API_KEY", ""):
            p = OpenRouterProvider()
        assert p.is_available() is False

    def test_explicit_valid_key_is_available(self):
        """api_key='real-key' must make the provider available regardless of env."""
        from services.ai.openrouter_provider import OpenRouterProvider
        with mock.patch("config.Config.OPENROUTER_API_KEY", ""):
            p = OpenRouterProvider(api_key="sk-or-explicit-real-key")
        assert p.is_available() is True

    # ── Error raising ─────────────────────────────────────────────────────────

    def test_openrouter_raises_provider_error_on_empty_key(self):
        """complete() raises AIProviderError when key is explicitly empty."""
        from services.ai.openrouter_provider import OpenRouterProvider
        p = OpenRouterProvider(api_key="")
        with pytest.raises(AIProviderError):
            p.complete("sys", "usr")

    # ── Full pipeline fallback ────────────────────────────────────────────────

    def test_optimize_game_falls_back_when_openrouter_fails(self):
        game = _palworld()
        hw = _hw()
        with mock.patch("services.ai_optimizer._select_provider") as mock_sel:
            prov = mock.MagicMock()
            prov.provider_name = "openrouter"
            prov.model_name = "test"
            prov.complete.side_effect = AIProviderError("Connection refused")
            mock_sel.return_value = prov
            result = optimize_game(hw, game)
        assert isinstance(result, AIOptimizationResult)
        assert result.status == AI_STATUS_FALLBACK
        assert result.recommended_settings is not None


# ---------------------------------------------------------------------------
# 10. Ollama unavailable
# ---------------------------------------------------------------------------

class TestOllamaUnavailable:
    """Test 10: Ollama unavailable (not running) falls back gracefully."""

    def test_ollama_raises_provider_error_on_connection_refused(self):
        from services.ai.ollama_provider import OllamaProvider
        p = OllamaProvider(base_url="http://127.0.0.1:19999", model="llama3.2")
        with pytest.raises(AIProviderError):
            p.complete("sys", "usr")

    def test_optimize_game_falls_back_when_ollama_fails(self):
        game = _palworld()
        hw = _hw()
        with mock.patch("services.ai_optimizer._select_provider") as mock_sel:
            prov = mock.MagicMock()
            prov.provider_name = "ollama"
            prov.model_name = "llama3.2"
            prov.complete.side_effect = AIProviderError("Connection refused")
            mock_sel.return_value = prov
            result = optimize_game(hw, game)
        assert result.status == AI_STATUS_FALLBACK
        assert result.provider == AI_PROVIDER_NONE


# ---------------------------------------------------------------------------
# 11. Both providers unavailable → fallback optimizer
# ---------------------------------------------------------------------------

class TestBothProvidersUnavailable:
    """Test 11: When no provider is available the deterministic optimizer is used."""

    def test_no_provider_returns_fallback_result(self):
        game = _palworld()
        hw = _hw()
        with mock.patch("services.ai_optimizer._select_provider", return_value=None):
            result = optimize_game(hw, game)
        assert isinstance(result, AIOptimizationResult)
        assert result.status == AI_STATUS_FALLBACK
        assert result.provider == AI_PROVIDER_NONE

    def test_fallback_result_has_valid_settings(self):
        game = _palworld()
        hw = _hw()
        with mock.patch("services.ai_optimizer._select_provider", return_value=None):
            result = optimize_game(hw, game)
        assert result.recommended_settings is not None
        assert result.recommended_settings.graphics_preset

    def test_fallback_result_is_not_empty(self):
        game = _palworld()
        hw = _hw()
        with mock.patch("services.ai_optimizer._select_provider", return_value=None):
            result = optimize_game(hw, game)
        d = result.to_dict()
        assert d["status"] == AI_STATUS_FALLBACK
        assert d["recommended_settings"]


# ---------------------------------------------------------------------------
# 12. Fallback optimizer works
# ---------------------------------------------------------------------------

class TestFallbackOptimizer:
    """Test 12: _wrap_deterministic produces a valid AIOptimizationResult."""

    def test_wrap_deterministic_with_valid_fallback(self):
        from services.optimizer import generate_recommendations
        game = _palworld()
        hw = _hw()
        det_result = generate_recommendations(hw, game)
        wrapped = _wrap_deterministic(det_result, None)
        assert isinstance(wrapped, AIOptimizationResult)
        assert wrapped.status == AI_STATUS_FALLBACK
        assert wrapped.recommended_settings is not None
        assert wrapped.recommended_settings.graphics_preset == det_result.preset

    def test_wrap_deterministic_with_none_fallback(self):
        wrapped = _wrap_deterministic(None, None)
        assert isinstance(wrapped, AIOptimizationResult)
        assert wrapped.status == AI_STATUS_FALLBACK

    def test_api_ai_optimize_route_returns_200(self):
        """Route /api/ai-optimize returns HTTP 200 even with no AI provider."""
        from app import create_app
        flask_app = create_app()
        flask_app.config["TESTING"] = True
        with flask_app.test_client() as client:
            with mock.patch("services.ai_optimizer._select_provider", return_value=None):
                resp = client.post(
                    "/api/ai-optimize",
                    json={
                        "game": "palworld",
                        "priority": "balanced",
                        "resolution": "1920x1080",
                        "target_fps": 60,
                    },
                )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "result" in data


# ---------------------------------------------------------------------------
# 13. No API key exposed in logs
# ---------------------------------------------------------------------------

class TestNoApiKeyInLogs:
    """Test 13: API keys must never appear in log output."""

    def test_openrouter_key_not_in_repr(self):
        from services.ai.openrouter_provider import OpenRouterProvider
        p = OpenRouterProvider(api_key="sk-or-secret-key-12345")
        rep = repr(p)
        assert "sk-or-secret-key-12345" not in rep

    def test_openrouter_key_not_logged_on_error(self, caplog):
        from services.ai.openrouter_provider import OpenRouterProvider
        p = OpenRouterProvider(api_key="sk-or-very-secret-key-99999")
        with caplog.at_level(logging.DEBUG):
            try:
                # Will fail with no real server, but must not log the key
                p.complete("sys", "usr")
            except AIProviderError:
                pass
        assert "sk-or-very-secret-key-99999" not in caplog.text

    def test_openrouter_key_not_in_error_message(self):
        from services.ai.openrouter_provider import OpenRouterProvider
        p = OpenRouterProvider(api_key="sk-or-exposed-key-77777")
        try:
            p.complete("sys", "usr")
        except AIProviderError as exc:
            assert "sk-or-exposed-key-77777" not in str(exc)


# ---------------------------------------------------------------------------
# 14. Prompt contains hardware information
# ---------------------------------------------------------------------------

class TestPromptContainsHardware:
    """Test 14: The user prompt includes all hardware data."""

    def _make_prompt(self, gpu="NVIDIA GeForce GTX 1650", cpu="AMD Ryzen 5 3600",
                     ram=16, vram=4):
        hw = HardwareSpecs(gpu=gpu, cpu=cpu, ram_gb=ram, vram_gb=vram)
        game = _palworld() or {}
        return build_user_prompt(
            hardware=hw, game=game, target_fps=60, priority="balanced",
            resolution="1920x1080", benchmark_context="", current_settings_text="",
        )

    def test_prompt_contains_gpu(self):
        prompt = self._make_prompt(gpu="NVIDIA GeForce RTX 3070")
        assert "RTX 3070" in prompt

    def test_prompt_contains_cpu(self):
        prompt = self._make_prompt(cpu="Intel Core i9-12900K")
        assert "i9-12900K" in prompt

    def test_prompt_contains_ram(self):
        prompt = self._make_prompt(ram=32)
        assert "32" in prompt

    def test_prompt_contains_vram(self):
        prompt = self._make_prompt(vram=8)
        assert "8" in prompt


# ---------------------------------------------------------------------------
# 15. Prompt contains game information
# ---------------------------------------------------------------------------

class TestPromptContainsGameInfo:
    """Test 15: The user prompt includes game name, engine, and target."""

    def test_prompt_contains_game_name(self):
        hw = _hw()
        game = _palworld() or {}
        prompt = build_user_prompt(
            hardware=hw, game=game, target_fps=60, priority="balanced",
            resolution="1920x1080", benchmark_context="", current_settings_text="",
        )
        assert "Palworld" in prompt

    def test_prompt_contains_target_fps(self):
        hw = _hw()
        game = _palworld() or {}
        prompt = build_user_prompt(
            hardware=hw, game=game, target_fps=90, priority="fps",
            resolution="1920x1080", benchmark_context="", current_settings_text="",
        )
        assert "90" in prompt

    def test_prompt_contains_resolution(self):
        hw = _hw()
        game = _palworld() or {}
        prompt = build_user_prompt(
            hardware=hw, game=game, target_fps=60, priority="balanced",
            resolution="2560x1440", benchmark_context="", current_settings_text="",
        )
        assert "2560x1440" in prompt

    def test_system_prompt_contains_key_rules(self):
        assert "NEVER invent benchmark" in SYSTEM_PROMPT
        assert "NEVER fabricate URLs" in SYSTEM_PROMPT
        assert "fps_source" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# 16. Source evidence survives parsing
# ---------------------------------------------------------------------------

class TestSourceEvidenceSurvives:
    """Test 16: Evidence items with URLs are preserved through parsing."""

    def test_evidence_parsed_correctly(self):
        raw = [
            {
                "title": "TechPowerUp Review",
                "url": "https://www.techpowerup.com/review",
                "domain": "techpowerup.com",
                "type": "PUBLISHED_BENCHMARK",
                "claim": "GTX 1650 achieves 45 FPS",
            }
        ]
        items = _parse_evidence(raw)
        assert len(items) == 1
        assert items[0].url == "https://www.techpowerup.com/review"
        assert items[0].evidence_type == "PUBLISHED_BENCHMARK"
        assert items[0].claim == "GTX 1650 achieves 45 FPS"

    def test_evidence_domain_inferred_when_missing(self):
        raw = [{"title": "T", "url": "https://some-site.com/page/1", "type": "GUIDE", "claim": "c"}]
        items = _parse_evidence(raw)
        assert "some-site.com" in items[0].domain

    def test_unknown_evidence_type_becomes_ai_inference(self):
        raw = [{"title": "T", "url": "https://x.com", "type": "MADE_UP_TYPE", "claim": "c"}]
        items = _parse_evidence(raw)
        assert items[0].evidence_type == EVIDENCE_AI_INFERENCE

    def test_evidence_in_full_result_to_dict(self):
        result = AIOptimizationResult(
            evidence=[EvidenceItem(
                title="Test", url="https://example.com",
                domain="example.com", evidence_type=EVIDENCE_COMMUNITY_REPORT,
                claim="Users report better FPS with shadows Low."
            )]
        )
        d = result.to_dict()
        assert len(d["evidence"]) == 1
        assert d["evidence"][0]["url"] == "https://example.com"
        assert d["evidence"][0]["type"] == EVIDENCE_COMMUNITY_REPORT


# ---------------------------------------------------------------------------
# 17. Fixture benchmark records are excluded
# ---------------------------------------------------------------------------

class TestFixtureExclusionInAiPipeline:
    """Test 17: Fixture records never reach the AI context as production data."""

    def setup_method(self):
        clear_cache()

    def test_palworld_fixture_not_in_benchmark_context(self):
        """palworld.json has only fixture records → benchmark context must be empty."""
        from services.ai.research import build_research_context
        hw = _hw()
        game = _palworld()
        ctx = build_research_context(hw, game, preset="Low",
                                     resolution="1920x1080", upscaling_mode="Off")
        # Fixture records excluded → no benchmark match → empty context text
        assert ctx.benchmark_context_text == "", (
            "Fixture records must not appear in the AI benchmark context"
        )

    def test_fixture_not_in_initial_evidence(self):
        """Fixture records must not seed the initial_evidence list."""
        from services.ai.research import build_research_context
        hw = _hw()
        game = _palworld()
        ctx = build_research_context(hw, game, preset="Low",
                                     resolution="1920x1080", upscaling_mode="Off")
        for item in ctx.initial_evidence:
            assert "TEST_FIXTURE" not in item.source, (
                "TEST_FIXTURE source must not appear in AI evidence"
            )


# ---------------------------------------------------------------------------
# 18. Game-specific supported settings are respected
# ---------------------------------------------------------------------------

class TestGameSpecificSettings:
    """Test 18: Validator uses game.resolutions and game.upscaling_support."""

    def test_game_upscaling_accepted(self):
        """Upscaling mode listed in game data is accepted."""
        game = {"resolutions": ["1920x1080"], "upscaling_support": ["FSR 2.0"]}
        warnings = []
        raw = {**_good_ai_json()["recommended_settings"], "upscaling": "FSR 2.0"}
        settings = _validate_settings(raw, None, "1920x1080", game, warnings)
        assert settings.upscaling == "FSR 2.0"
        assert not any("upscaling" in w for w in warnings)

    def test_unsupported_upscaling_replaced(self):
        """Upscaling not in game data and not in global allowed set is replaced."""
        game = {"resolutions": ["1920x1080"], "upscaling_support": ["FSR 2.0"]}
        warnings = []
        raw = {**_good_ai_json()["recommended_settings"], "upscaling": "MAGIC_UPSCALE_3000"}
        settings = _validate_settings(raw, None, "1920x1080", game, warnings)
        assert settings.upscaling != "MAGIC_UPSCALE_3000"
        assert any("upscaling" in w for w in warnings)

    def test_game_resolution_accepted(self):
        """Resolution listed in game data (but not global set) is accepted."""
        game = {"resolutions": ["1680x1050"], "upscaling_support": []}
        warnings = []
        raw = {**_good_ai_json()["recommended_settings"], "resolution": "1680x1050"}
        settings = _validate_settings(raw, None, "1920x1080", game, warnings)
        assert settings.resolution == "1680x1050"

    def test_optimize_game_respects_game_resolutions(self):
        """Full pipeline: AI is given game resolution/upscaling in the prompt."""
        hw = _hw()
        game = _palworld()
        if not game:
            pytest.skip("Palworld game data not available")
        prompt = build_user_prompt(
            hardware=hw, game=game, target_fps=60, priority="balanced",
            resolution="1920x1080", benchmark_context="", current_settings_text="",
        )
        # Check that game supported resolutions/upscaling from games.json appear
        supported_res = game.get("resolutions", [])
        if supported_res:
            assert supported_res[0] in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
