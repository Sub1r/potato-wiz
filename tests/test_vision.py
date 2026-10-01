"""
tests/test_vision.py
~~~~~~~~~~~~~~~~~~~~
Phase 2: screenshot-based game settings analysis.

Covers upload validation, preprocessing, the vision provider abstraction
(OpenRouter + Ollama), structured extraction, game-aware deterministic
validation, the confirmation step, and Phase 1 integration guarantees
(manual settings still work, deterministic FPS stays authoritative).
"""
from __future__ import annotations

import io
import json
import logging
from unittest import mock

import pytest
from PIL import Image

from config import Config
from services.ai.screenshot_flow import (analyze_screenshot_upload,
                                         build_screenshot_context_text)
from services.ai.vision import (ERR_MODEL_NOT_VISION, ERR_NO_VISION_PROVIDER,
                                ERR_OLLAMA_UNAVAILABLE, OllamaVisionProvider,
                                OpenRouterVisionProvider, VisionError,
                                VisionProvider, select_vision_provider)
from services.ai.vision_prompts import (SOURCE_SCREENSHOT, SYSTEM_PROMPT,
                                        build_game_context, build_vision_prompt,
                                        canonical_key, normalize_resolution,
                                        parse_vision_response, validate_extraction)
from services.screenshot import (ScreenshotError, build_preview_data_url,
                                 preprocess_image, read_upload, validate_image)


# ── Fixtures / helpers ────────────────────────────────────────────────────────

GAME = {
    "name": "Palworld",
    "slug": "palworld",
    "settings_detail": [
        {"name": "Graphics Preset"}, {"name": "Resolution"},
        {"name": "Shadows"}, {"name": "Textures"}, {"name": "Effects"},
        {"name": "FSR / Upscaling"},
    ],
    "resolutions": ["1280x720", "1920x1080", "2560x1440"],
    "upscaling_support": ["FSR 2.0", "DLSS 3.0"],
    "recommended_settings": {"preset": "High"},
}

VISION_JSON = {
    "settings": [
        {"name": "Graphics Preset", "value": "High", "confidence": "high"},
        {"name": "Resolution", "value": "1920 x 1080", "confidence": "high"},
        {"name": "Shadows", "value": "High", "confidence": "high"},
        {"name": "Textures", "value": "High", "confidence": "high"},
        {"name": "FSR / Upscaling", "value": "Off", "confidence": "high"},
    ],
    "unreadable": [],
    "notes": [],
}


class Upload:
    """Minimal stand-in for a Werkzeug FileStorage (in memory only)."""

    def __init__(self, data: bytes, filename: str = "shot.png",
                 mimetype: str = "image/png"):
        self._data = data
        self.filename = filename
        self.mimetype = mimetype

    def read(self, size: int = -1) -> bytes:
        return self._data if size is None or size < 0 else self._data[:size]


def make_image(fmt: str = "PNG", size=(400, 300), color=(20, 30, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def make_jpeg(size=(400, 300)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 90, 60)).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def make_webp(size=(400, 300)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 200, 90)).save(buf, format="WEBP")
    return buf.getvalue()


class FakeVisionProvider(VisionProvider):
    """Vision provider stub returning a canned JSON response."""

    provider_name = "fake_vision"

    def __init__(self, response=None, error=None, model_name="fake-vision-1"):
        self.model_name = model_name
        self._response = response if response is not None else json.dumps(VISION_JSON)
        self._error = error
        self.calls = []

    def is_available(self) -> bool:
        return self._error != "unavailable"

    def analyze_image(self, image_data_url, prompt, game_context=None,
                      system_prompt=""):
        self.calls.append({
            "image_data_url": image_data_url,
            "prompt": prompt,
            "game_context": game_context,
            "system_prompt": system_prompt,
        })
        if self._error:
            raise VisionError(self._error)
        return self._response


# ── 1-3. PNG / JPEG / WebP uploads ────────────────────────────────────────────

class TestValidImageFormats:
    @pytest.mark.parametrize("fmt,ext,mime", [
        ("PNG", "png", "image/png"),
        ("JPEG", "jpg", "image/jpeg"),
        ("JPEG", "jpeg", "image/jpeg"),
        ("WEBP", "webp", "image/webp"),
    ])
    def test_format_accepted(self, fmt, ext, mime):
        data = make_image(fmt)
        info = validate_image(data)
        assert info["format"] == fmt
        upload = Upload(data, f"settings.{ext}", mime)
        assert read_upload(upload) == data

    def test_png_end_to_end(self):
        report = analyze_screenshot_upload(
            Upload(make_image("PNG"), "shot.png", "image/png"),
            game=GAME, provider=FakeVisionProvider())
        assert report["source"] == SOURCE_SCREENSHOT
        assert report["recognized_count"] == 5

    def test_jpeg_end_to_end(self):
        report = analyze_screenshot_upload(
            Upload(make_jpeg(), "shot.jpg", "image/jpeg"),
            game=GAME, provider=FakeVisionProvider())
        assert report["current_settings_text"]

    def test_webp_end_to_end(self):
        report = analyze_screenshot_upload(
            Upload(make_webp(), "shot.webp", "image/webp"),
            game=GAME, provider=FakeVisionProvider())
        assert report["image"]["format"] == "WEBP"


# ── 4-7. Rejected uploads ─────────────────────────────────────────────────────

class TestRejectedUploads:
    def test_invalid_extension(self):
        upload = Upload(make_image("PNG"), "settings.txt", "image/png")
        with pytest.raises(ScreenshotError) as exc:
            read_upload(upload)
        assert exc.value.code == "invalid_extension"
        assert Config.SCREENSHOT_INVALID_MESSAGE in str(exc.value)

    def test_invalid_mime_type(self):
        upload = Upload(make_image("PNG"), "shot.png", "application/x-msdownload")
        with pytest.raises(ScreenshotError) as exc:
            read_upload(upload)
        assert exc.value.code == "invalid_mime_type"

    def test_executable_content_is_rejected(self):
        upload = Upload(b"MZ\x90\x00" + b"\x00" * 500, "evil.png", "image/png")
        with pytest.raises(ScreenshotError) as exc:
            validate_image(read_upload(upload))
        assert exc.value.code == "corrupt_image"
        assert Config.SCREENSHOT_INVALID_MESSAGE in str(exc.value)

    def test_pdf_content_is_rejected(self):
        pdf = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n" + b"0" * 200
        with pytest.raises(ScreenshotError) as exc:
            validate_image(pdf)
        assert exc.value.code == "corrupt_image"

    def test_oversized_image(self):
        with mock.patch("config.Config.SCREENSHOT_MAX_UPLOAD_BYTES", 1024):
            upload = Upload(make_image("PNG", size=(1200, 1200)), "shot.png", "image/png")
            with pytest.raises(ScreenshotError) as exc:
                read_upload(upload)
        assert exc.value.code == "too_large"
        assert "10 MB" in exc.value.message

    def test_corrupted_image(self):
        truncated = make_image("PNG")[:120]   # valid header, broken body
        with pytest.raises(ScreenshotError) as exc:
            validate_image(truncated)
        assert exc.value.code in ("corrupt_image", "unsupported_format")

    def test_empty_file(self):
        with pytest.raises(ScreenshotError):
            read_upload(Upload(b"", "shot.png", "image/png"))

    def test_missing_file(self):
        with pytest.raises(ScreenshotError):
            read_upload(None)

    def test_client_filename_is_never_used_as_a_path(self):
        """Path traversal attempts are irrelevant: nothing touches the fs."""
        upload = Upload(make_image("PNG"), "../../etc/passwd.png", "image/png")
        assert read_upload(upload)  # extension check only, no filesystem access


# ── 8. Preprocessing ──────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_small_image_is_not_resized(self):
        url, mime, meta = preprocess_image(make_image("PNG", size=(800, 600)))
        assert meta["downscaled"] is False
        assert meta["width"] == 800 and meta["height"] == 600
        assert url.startswith("data:image/png;base64,")

    def test_small_image_is_not_upscaled(self):
        _, _, meta = preprocess_image(make_image("PNG", size=(200, 150)))
        assert meta["height"] == 150

    def test_large_image_is_downscaled(self):
        data = make_image("PNG", size=(5000, 2500))
        url, mime, meta = preprocess_image(data)
        assert meta["downscaled"] is True
        assert meta["output_width"] <= Config.SCREENSHOT_MAX_DIMENSION
        assert mime == "image/jpeg"
        assert url.startswith("data:image/jpeg;base64,")

    def test_downscaled_payload_is_smaller(self):
        data = make_image("PNG", size=(4000, 2000))
        _, _, meta = preprocess_image(data)
        assert meta["bytes"] < len(data)

    def test_text_readability_floor_is_respected(self):
        """Never shrink a settings menu below the readability floor."""
        _, _, meta = preprocess_image(make_image("PNG", size=(6000, 1500)))
        assert meta["output_height"] >= 1024
        assert meta["output_width"] >= 1024

    def test_corrupt_image_preprocessing_fails_cleanly(self):
        with pytest.raises(ScreenshotError):
            preprocess_image(b"not an image at all")

    def test_preview_is_small_data_url(self):
        url = build_preview_data_url(make_image("PNG", size=(2000, 1500)))
        assert url.startswith("data:image/jpeg;base64,")

    def test_preview_failure_returns_none(self):
        assert build_preview_data_url(b"garbage") is None


# ── 9-11. Provider availability ───────────────────────────────────────────────

class TestProviderSelection:
    def test_no_provider_configured(self):
        with mock.patch("config.Config.OPENROUTER_VISION_MODEL", ""), \
             mock.patch("config.Config.OLLAMA_VISION_MODEL", ""):
            assert select_vision_provider() is None

    def test_openrouter_vision_selected(self):
        with mock.patch("config.Config.OPENROUTER_VISION_MODEL", "some/vision-model"), \
             mock.patch("config.Config.OPENROUTER_API_KEY", "sk-or-test"):
            provider = select_vision_provider()
        assert isinstance(provider, OpenRouterVisionProvider)
        assert provider.model_name == "some/vision-model"

    def test_ollama_vision_selected(self):
        with mock.patch("config.Config.OPENROUTER_VISION_MODEL", ""), \
             mock.patch("config.Config.OLLAMA_VISION_MODEL", "llava:13b"):
            provider = select_vision_provider()
        assert isinstance(provider, OllamaVisionProvider)
        assert provider.model_name == "llava:13b"

    def test_flow_reports_missing_provider(self):
        with mock.patch("services.ai.screenshot_flow.select_vision_provider",
                        return_value=None):
            with pytest.raises(VisionError) as exc:
                analyze_screenshot_upload(
                    Upload(make_image("PNG"), "shot.png", "image/png"), game=GAME)
        assert ERR_NO_VISION_PROVIDER in str(exc.value)

    def test_openrouter_requires_vision_model_config(self):
        with mock.patch("config.Config.OPENROUTER_VISION_MODEL", ""):
            assert OpenRouterVisionProvider(api_key="k").is_available() is False

    def test_openrouter_vision_model_not_found(self):
        import urllib.error

        provider = OpenRouterVisionProvider(api_key="k", model="nope/vision")
        with mock.patch.object(provider, "verify_vision_support",
                               return_value={"vision": True}), \
             mock.patch("urllib.request.urlopen",
                        side_effect=urllib.error.HTTPError(
                            "u", 404, "Not Found", {}, None)):
            with pytest.raises(VisionError) as exc:
                provider.analyze_image("data:image/png;base64,AAA", "read this")
        assert "OPENROUTER_VISION_MODEL" in str(exc.value)

    def test_text_only_model_is_refused_before_request(self):
        """The image must never be sent to a model that cannot read images."""
        provider = OpenRouterVisionProvider(api_key="k", model="openai/gpt-5-mini")
        entry = {"id": "openai/gpt-5-mini",
                 "architecture": {"input_modalities": ["text"]}}
        with mock.patch("services.ai.openrouter_provider.fetch_model_info",
                        return_value=entry), \
             mock.patch("urllib.request.urlopen",
                        side_effect=AssertionError("no request must be sent")):
            with pytest.raises(VisionError) as exc:
                provider.analyze_image("data:image/png;base64,AAA", "read this")
        assert ERR_MODEL_NOT_VISION in str(exc.value)

    def test_vision_capable_model_is_accepted(self):
        provider = OpenRouterVisionProvider(api_key="k", model="some/vision")
        entry = {"id": "some/vision",
                 "architecture": {"input_modalities": ["text", "image"]}}
        with mock.patch("services.ai.openrouter_provider.fetch_model_info",
                        return_value=entry):
            verdict = provider.verify_vision_support()
        assert verdict["vision"] is True
        assert "image" in verdict["input_modalities"]

    def test_unknown_capability_is_tolerated(self):
        provider = OpenRouterVisionProvider(api_key="k", model="mystery/model")
        with mock.patch("services.ai.openrouter_provider.fetch_model_info",
                        return_value=None):
            verdict = provider.verify_vision_support()
        assert verdict["vision"] is None

    def test_ollama_unavailable(self):
        import urllib.error

        provider = OllamaVisionProvider(base_url="http://127.0.0.1:11434",
                                        model="llava:13b")
        with mock.patch("urllib.request.urlopen",
                        side_effect=urllib.error.URLError("refused")):
            with pytest.raises(VisionError) as exc:
                provider.analyze_image("data:image/png;base64,AAA", "read this")
        assert ERR_OLLAMA_UNAVAILABLE in str(exc.value)

    def test_ollama_vision_model_missing(self):
        import urllib.error

        provider = OllamaVisionProvider(base_url="http://127.0.0.1:11434",
                                        model="not-installed")
        with mock.patch("urllib.request.urlopen",
                        side_effect=urllib.error.HTTPError(
                            "u", 404, "Not Found", {}, None)):
            with pytest.raises(VisionError) as exc:
                provider.analyze_image("data:image/png;base64,AAA", "read this")
        assert "OLLAMA_VISION_MODEL" in str(exc.value)

    def test_ollama_timeout_is_handled(self):
        provider = OllamaVisionProvider(base_url="http://127.0.0.1:11434",
                                        model="llava:13b")
        with mock.patch("urllib.request.urlopen", side_effect=TimeoutError()):
            with pytest.raises(VisionError) as exc:
                provider.analyze_image("data:image/png;base64,AAA", "read this")
        assert "timed out" in str(exc.value)

    def test_ollama_malformed_response(self):
        provider = OllamaVisionProvider(base_url="http://127.0.0.1:11434",
                                        model="llava:13b")

        class _Resp:
            def read(self):
                return b"<<not json>>"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        with mock.patch("urllib.request.urlopen", return_value=_Resp()):
            with pytest.raises(VisionError):
                provider.analyze_image("data:image/png;base64,AAA", "read this")


# ── OpenRouter / Ollama request shapes ────────────────────────────────────────

class _Resp:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestProviderRequestShapes:
    def test_openrouter_uses_multimodal_content(self):
        provider = OpenRouterVisionProvider(api_key="k", model="some/vision")
        captured = {}

        def fake(req, timeout=None):
            captured["payload"] = json.loads(req.data.decode())
            return _Resp({"choices": [{"message": {"content": json.dumps(VISION_JSON)}}]})

        with mock.patch.object(provider, "verify_vision_support",
                               return_value={"vision": True}), \
             mock.patch("urllib.request.urlopen", side_effect=fake):
            provider.analyze_image("data:image/png;base64,AAA", "read this")

        payload = captured["payload"]
        assert payload["model"] == "some/vision"
        user_message = payload["messages"][-1]
        assert user_message["role"] == "user"
        assert user_message["content"][0]["type"] == "text"
        assert user_message["content"][1]["type"] == "image_url"
        assert user_message["content"][1]["image_url"]["url"].startswith("data:image/")
        assert payload["max_tokens"] == Config.OPENROUTER_VISION_MAX_TOKENS

    def test_openrouter_empty_response_is_reported(self):
        provider = OpenRouterVisionProvider(api_key="k", model="some/vision")
        with mock.patch.object(provider, "verify_vision_support",
                               return_value={"vision": True}), \
             mock.patch("urllib.request.urlopen",
                        return_value=_Resp({"choices": [{"message": {"content": ""}}]})):
            with pytest.raises(VisionError) as exc:
                provider.analyze_image("data:image/png;base64,AAA", "read this")
        assert "empty" in str(exc.value).lower()

    def test_ollama_uses_images_field(self):
        provider = OllamaVisionProvider(base_url="http://127.0.0.1:11434",
                                        model="llava:13b")
        captured = {}

        def fake(req, timeout=None):
            captured["payload"] = json.loads(req.data.decode())
            return _Resp({"message": {"content": json.dumps(VISION_JSON)}})

        with mock.patch("urllib.request.urlopen", side_effect=fake):
            provider.analyze_image("data:image/png;base64,QUJD", "read this")

        payload = captured["payload"]
        assert payload["model"] == "llava:13b"
        assert payload["messages"][-1]["images"] == ["QUJD"]


# ── 12-16. Extraction format ──────────────────────────────────────────────────

class TestVisionExtraction:
    def test_valid_vision_json(self):
        report = analyze_screenshot_upload(
            Upload(make_image("PNG"), "shot.png", "image/png"),
            game=GAME, provider=FakeVisionProvider())
        names = {s["canonical_name"] for s in report["settings"]}
        assert "graphics_preset" in names
        assert "shadows" in names
        assert report["settings"][0]["source"] == SOURCE_SCREENSHOT
        assert report["requires_confirmation"] is True

    def test_malformed_vision_json(self):
        provider = FakeVisionProvider(response="this is not json")
        with pytest.raises(Exception) as exc:
            analyze_screenshot_upload(
                Upload(make_image("PNG"), "shot.png", "image/png"),
                game=GAME, provider=provider)
        assert "JSON" in str(exc.value)

    def test_fenced_json_is_tolerated(self):
        fenced = "```json\n" + json.dumps(VISION_JSON) + "\n```"
        report = analyze_screenshot_upload(
            Upload(make_image("PNG"), "shot.png", "image/png"),
            game=GAME, provider=FakeVisionProvider(response=fenced))
        assert report["recognized_count"] == 5

    def test_missing_setting_is_not_invented(self):
        """Textures is absent from the screenshot → no value appears."""
        partial = {
            "settings": [
                {"name": "Graphics Preset", "value": "High", "confidence": "high"},
                {"name": "Shadows", "value": "Medium", "confidence": "high"},
            ],
            "unreadable": [],
            "notes": [],
        }
        report = analyze_screenshot_upload(
            Upload(make_image("PNG"), "shot.png", "image/png"), game=GAME,
            provider=FakeVisionProvider(response=json.dumps(partial)))
        names = {s["canonical_name"] for s in report["settings"]}
        assert "textures" not in names
        assert "textures" not in report["current_settings_text"]

    def test_unreadable_setting_returns_null(self):
        partial = {
            "settings": [
                {"name": "Graphics Preset", "value": "High", "confidence": "high"},
                {"name": "Shadows", "value": None, "confidence": "high"},
            ],
            "unreadable": ["Shadows"],
            "notes": [],
        }
        report = analyze_screenshot_upload(
            Upload(make_image("PNG"), "shot.png", "image/png"), game=GAME,
            provider=FakeVisionProvider(response=json.dumps(partial)))
        shadows = next(s for s in report["settings"] if s["canonical_name"] == "shadows")
        assert shadows["value"] is None
        assert shadows["confidence"] == "low"
        assert shadows["status"] == "unreadable"
        assert "shadows" not in report["current_settings_text"]

    def test_unknown_setting_is_marked_unrecognized(self):
        partial = {
            "settings": [
                {"name": "Quantum Flux", "value": "7", "confidence": "low"},
            ],
            "unreadable": [],
            "notes": [],
        }
        report = analyze_screenshot_upload(
            Upload(make_image("PNG"), "shot.png", "image/png"), game=GAME,
            provider=FakeVisionProvider(response=json.dumps(partial)))
        entry = report["settings"][0]
        assert entry["status"] == "unrecognized"
        assert entry["value"] == "7"          # still shown to the user
        assert report["unrecognized"]

    def test_unknown_settings_do_not_crash(self):
        with mock.patch("services.ai.vision_prompts.SUPPORTED_VALUES", {}):
            report = analyze_screenshot_upload(
                Upload(make_image("PNG"), "shot.png", "image/png"), game=GAME,
                provider=FakeVisionProvider())
        assert report["settings"]


# ── 17. Game-aware validation ─────────────────────────────────────────────────

class TestGameAwareValidation:
    def test_game_vocabulary_is_passed_to_the_model(self):
        provider = FakeVisionProvider()
        analyze_screenshot_upload(Upload(make_image("PNG"), "shot.png", "image/png"),
                                 game=GAME, provider=provider)
        prompt = provider.calls[0]["prompt"]
        assert "Palworld" in prompt
        assert "vocabulary only" in prompt

    def test_game_vocabulary_does_not_cause_invention(self):
        """Textures exists in games.json but was not in the screenshot."""
        partial = {"settings": [
            {"name": "Shadows", "value": "High", "confidence": "high"}],
            "unreadable": [], "notes": []}
        provider = FakeVisionProvider(response=json.dumps(partial))
        report = analyze_screenshot_upload(
            Upload(make_image("PNG"), "shot.png", "image/png"), game=GAME,
            provider=provider)
        assert "textures" not in report["current_settings_text"].lower()

    def test_resolution_is_normalised(self):
        report = validate_extraction(parse_vision_response(json.dumps({
            "settings": [{"name": "Resolution", "value": "1920 × 1080",
                          "confidence": "high"}]})))
        assert report["settings"][0]["value"] == "1920x1080"

    def test_unreadable_resolution_is_not_invented(self):
        report = validate_extraction(parse_vision_response(json.dumps({
            "settings": [{"name": "Resolution", "value": "unknown",
                          "confidence": "low"}]})))
        assert report["settings"][0]["value"] is None
        assert report["settings"][0]["status"] == "unreadable"

    def test_resolution_outside_game_list_is_flagged(self):
        report = validate_extraction(parse_vision_response(json.dumps({
            "settings": [{"name": "Resolution", "value": "800x600",
                          "confidence": "high"}]})), GAME)
        assert "not listed" in report["settings"][0]["note"]

    def test_supported_values_are_normalised(self):
        report = validate_extraction(parse_vision_response(json.dumps({
            "settings": [{"name": "Shadows", "value": "high",
                          "confidence": "high"}]})), GAME)
        assert report["settings"][0]["value"] == "High"

    def test_non_standard_value_is_kept_and_flagged(self):
        report = validate_extraction(parse_vision_response(json.dumps({
            "settings": [{"name": "Shadows", "value": "Cinematic",
                          "confidence": "medium"}]})), GAME)
        entry = report["settings"][0]
        assert entry["value"] == "Cinematic"
        assert "not a standard" in entry["note"]

    def test_upscaler_aliases_map_to_upscaling(self):
        for label in ("FSR", "AMD FSR", "Upscaling", "Upscaler", "DLSS", "XeSS"):
            assert canonical_key(label) == "upscaling"

    @pytest.mark.parametrize("label,key", [
        ("V-Sync", "vsync"),
        ("Vsync", "vsync"),
        ("Anti-Aliasing", "anti_aliasing"),
        ("Texture Quality", "textures"),
        ("Frame Rate Limit", "fps_limit"),
        ("Motion Blur", "motion_blur"),
        ("Screen Space Reflections", "reflections"),
    ])
    def test_label_variants_map_to_canonical_keys(self, label, key):
        assert canonical_key(label) == key

    def test_game_known_key_covers_aliases(self):
        """A game's 'FSR / Upscaling' row covers an 'AMD FSR' screenshot row."""
        report = validate_extraction(parse_vision_response(json.dumps({
            "settings": [{"name": "AMD FSR", "value": "Off",
                          "confidence": "high"}]})), GAME)
        entry = report["settings"][0]
        assert entry["status"] == "recognized"
        assert "not listed" not in entry.get("note", "").lower()

    def test_unknown_upscaler_is_flagged(self):
        game_without_dlss = dict(GAME, upscaling_support=["FSR 2.0"])
        report = validate_extraction(parse_vision_response(json.dumps({
            "settings": [{"name": "DLSS", "value": "Quality",
                          "confidence": "high"}]})), game_without_dlss)
        assert "upscaler" in report["settings"][0].get("note", "").lower()

    def test_supported_upscaler_has_no_note(self):
        report = validate_extraction(parse_vision_response(json.dumps({
            "settings": [{"name": "FSR", "value": "Balanced",
                          "confidence": "high"}]})), GAME)
        assert "upscaler" not in report["settings"][0].get("note", "").lower()

    def test_resolution_normalisation_helper(self):
        assert normalize_resolution("2560x1440") == "2560x1440"
        assert normalize_resolution("1920 X 1080") == "1920x1080"
        assert normalize_resolution("Native") is None
        assert normalize_resolution("") is None


# ── 18-20. Phase 1 integration ────────────────────────────────────────────────

class TestPhase1Integration:
    def test_manual_settings_still_work(self):
        from services.ai_optimizer import optimize_game
        from tests.test_ai_optimizer import _hw, _palworld

        with mock.patch("services.ai_optimizer._select_provider") as mock_sel:
            provider = mock.MagicMock()
            provider.provider_name = "openrouter"
            provider.model_name = "test"
            provider.complete.return_value = json.dumps({
                "summary": "ok",
                "recommended_settings": {"graphics_preset": "Medium",
                                         "resolution": "1920x1080"},
                "changes": [], "estimated_fps": "", "fps_source": "",
                "confidence": "medium", "reasoning": "x", "evidence": [],
                "warnings": [],
            })
            mock_sel.return_value = provider
            result = optimize_game(_hw(), _palworld(),
                                   current_settings="Preset: High\nShadows: High")
        sent_prompt = provider.complete.call_args.kwargs["user_prompt"]
        assert "Preset: High" in sent_prompt
        assert "Shadows: High" in sent_prompt
        assert result.recommended_settings is not None

    def test_screenshot_settings_are_used_as_current_settings(self):
        provider = FakeVisionProvider()
        analyze_screenshot_upload(Upload(make_image("PNG"), "shot.png", "image/png"),
                                 game=GAME, provider=provider)
        text = settings_text(provider)
        assert "shadows: High" in text
        assert "textures: High" in text

    def test_manual_text_used_as_supplementary_context(self):
        provider = FakeVisionProvider()
        report = analyze_screenshot_upload(
            Upload(make_image("PNG"), "shot.png", "image/png"),
            game=GAME, manual_context="I use FSR Balanced", provider=provider)
        assert "I use FSR Balanced" in provider.calls[0]["prompt"]
        assert report["manual_context_included"] is True

    def test_screenshot_takes_priority_over_manual_text(self):
        from routes.optimizer import _merge_current_settings

        merged = _merge_current_settings("shadows: High", "shadows: Ultra")
        screenshot_part, manual_part = merged.split("\n\n")
        assert "screenshot" in screenshot_part.lower()
        assert "shadows: High" in screenshot_part
        assert "supplementary" in manual_part.lower()
        assert "shadows: Ultra" in manual_part

    def test_manual_text_only_merge(self):
        from routes.optimizer import _merge_current_settings

        assert "Preset: High" in _merge_current_settings("", "Preset: High")

    def test_screenshot_only_merge(self):
        from routes.optimizer import _merge_current_settings

        merged = _merge_current_settings("shadows: High", "")
        assert "shadows: High" in merged

    def test_confirmed_settings_text_for_phase1(self):
        report = validate_extraction(parse_vision_response(json.dumps({
            "settings": [
                {"name": "Shadows", "value": "High", "confidence": "high"},
                {"name": "Textures", "value": None, "confidence": "low"},
            ]})))
        text = build_screenshot_context_text(report)
        assert "shadows: High" in text
        assert "textures" not in text

    def test_screenshot_settings_reach_the_optimizer(self):
        from services.ai_optimizer import optimize_game
        from tests.test_ai_optimizer import _hw, _palworld

        report = analyze_screenshot_upload(
            Upload(make_image("PNG"), "shot.png", "image/png"),
            game=GAME, provider=FakeVisionProvider())
        with mock.patch("services.ai_optimizer._select_provider") as mock_sel:
            provider = mock.MagicMock()
            provider.provider_name = "openrouter"
            provider.model_name = "test"
            provider.complete.return_value = json.dumps({
                "summary": "ok",
                "recommended_settings": {"graphics_preset": "Medium",
                                         "resolution": "1920x1080"},
                "changes": [], "estimated_fps": "", "fps_source": "",
                "confidence": "medium", "reasoning": "x", "evidence": [],
                "warnings": [],
            })
            mock_sel.return_value = provider
            optimize_game(_hw(), _palworld(),
                          current_settings=report["current_settings_text"])
        sent = provider.complete.call_args.kwargs["user_prompt"]
        assert "shadows: High" in sent


def settings_text(provider) -> str:
    return validate_extraction(
        parse_vision_response(provider._response), GAME
    )["current_settings_text"]


# ── 21-23. Privacy, storage, determinism ──────────────────────────────────────

class TestPrivacyAndStorage:
    def test_screenshot_is_not_stored_on_disk(self):
        """Nothing is written: the flow only ever works in memory."""
        report = analyze_screenshot_upload(
            Upload(make_image("PNG"), "shot.png", "image/png"),
            game=GAME, provider=FakeVisionProvider())
        assert "path" not in report["image"]
        assert set(report["image"]) == {
            "format", "width", "height", "prepared_mime", "downscaled"}

    def test_no_base64_in_report(self):
        report = analyze_screenshot_upload(
            Upload(make_image("PNG"), "shot.png", "image/png"),
            game=GAME, provider=FakeVisionProvider())
        assert "base64" not in json.dumps(report)

    def test_no_image_data_logged(self, caplog):
        provider = FakeVisionProvider()
        with caplog.at_level(logging.DEBUG):
            analyze_screenshot_upload(
                Upload(make_image("PNG"), "shot.png", "image/png"),
                game=GAME, provider=provider)
        logged = "\n".join(r.getMessage() for r in caplog.records)
        assert "base64" not in logged
        assert "data:image" not in logged

    def test_vision_error_never_contains_image_data(self):
        import urllib.error

        provider = OpenRouterVisionProvider(api_key="k", model="some/vision")
        secret = "data:image/png;base64,VERY_SECRET_PAYLOAD"
        with mock.patch.object(provider, "verify_vision_support",
                               return_value={"vision": True}), \
             mock.patch("urllib.request.urlopen",
                        side_effect=urllib.error.HTTPError(
                            "u", 500, "Server Error", {}, None)):
            with pytest.raises(VisionError) as exc:
                provider.analyze_image(secret, "read this")
        assert "VERY_SECRET_PAYLOAD" not in str(exc.value)

    def test_api_key_not_in_repr(self):
        provider = OpenRouterVisionProvider(api_key="sk-or-secret-key")
        assert "sk-or-secret-key" not in repr(provider)


class TestDeterministicFpsAuthority:
    def test_vision_output_never_becomes_fps(self):
        """Vision JSON may contain a stray number; it is not a performance estimate."""
        partial = {"settings": [
            {"name": "FPS Limit", "value": "60", "confidence": "high"}],
            "unreadable": [], "notes": []}
        report = analyze_screenshot_upload(
            Upload(make_image("PNG"), "shot.png", "image/png"), game=GAME,
            provider=FakeVisionProvider(response=json.dumps(partial)))
        assert report["source"] == SOURCE_SCREENSHOT
        assert "fps" not in report

    def test_deterministic_fps_remains_authoritative(self):
        from services.ai_optimizer import optimize_game
        from tests.test_ai_optimizer import _hw, _palworld

        report = analyze_screenshot_upload(
            Upload(make_image("PNG"), "shot.png", "image/png"),
            game=GAME, provider=FakeVisionProvider())
        ai_response = json.dumps({
            "summary": "x",
            "recommended_settings": {"graphics_preset": "Low",
                                     "resolution": "1920x1080"},
            "changes": [], "estimated_fps": "999", "fps_source": "AI_INFERENCE",
            "confidence": "high", "reasoning": "x", "evidence": [], "warnings": [],
        })
        with mock.patch("services.ai_optimizer._select_provider") as mock_sel:
            provider = mock.MagicMock()
            provider.provider_name = "openrouter"
            provider.model_name = "test"
            provider.complete.return_value = ai_response
            mock_sel.return_value = provider
            result = optimize_game(
                _hw(), _palworld(),
                current_settings=report["current_settings_text"])

        assert result.fps_source != "AI_INFERENCE"
        assert "999" not in str(result.estimated_fps)

    def test_screenshot_falls_back_when_vision_fails(self):
        from services.ai.base import AIProviderError
        from services.ai_optimizer import optimize_game
        from tests.test_ai_optimizer import _hw, _palworld

        with mock.patch("services.ai_optimizer._select_provider") as mock_sel:
            provider = mock.MagicMock()
            provider.provider_name = "openrouter"
            provider.model_name = "test"
            provider.complete.side_effect = AIProviderError("vision exploded")
            mock_sel.return_value = provider
            result = optimize_game(_hw(), _palworld())

        assert result.status == "fallback"
        assert result.recommended_settings.graphics_preset


# ── Prompt content ────────────────────────────────────────────────────────────

class TestVisionPrompt:
    def test_system_prompt_forbids_invention(self):
        assert "NEVER guess" in SYSTEM_PROMPT
        assert "you can actually SEE it in the screenshot" in SYSTEM_PROMPT
        assert "null" in SYSTEM_PROMPT

    def test_system_prompt_requires_json(self):
        assert "JSON" in SYSTEM_PROMPT
        assert '"confidence"' in SYSTEM_PROMPT

    def test_game_context_lists_vocabulary(self):
        context = build_game_context(GAME)
        assert "Palworld" in context
        assert "Shadows" in context
        assert "1920x1080" in context

    def test_game_context_without_game(self):
        assert build_game_context(None) == "No game is selected."

    def test_prompt_includes_manual_context_last(self):
        prompt = build_vision_prompt(build_game_context(GAME), "Preset: Ultra")
        assert "Preset: Ultra" in prompt

    def test_prompt_without_manual_context(self):
        prompt = build_vision_prompt(build_game_context(GAME))
        assert "supplementary" not in prompt