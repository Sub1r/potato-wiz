"""
tests/test_vision_routes.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
HTTP-level tests for the Phase 2 screenshot endpoints.

Guarantees:
- the Phase 1 JSON API of ``/api/ai-optimize`` is unchanged,
- multipart submissions are accepted and the screenshot takes priority,
- the JSON API stays free of image data,
- the screenshot analysis endpoint never persists the upload.
"""
from __future__ import annotations

import io
import json
from unittest import mock

import pytest
from PIL import Image

from services.ai.vision import VisionError
from services.screenshot import ScreenshotError


def _png(size=(320, 240)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


VISION_JSON = json.dumps({
    "settings": [
        {"name": "Graphics Preset", "value": "High", "confidence": "high"},
        {"name": "Shadows", "value": "High", "confidence": "high"},
    ],
    "unreadable": [],
    "notes": [],
})


@pytest.fixture(autouse=True)
def _fast_hardware():
    """Hardware detection is unrelated here and slow (psutil)."""
    from tests.test_ai_optimizer import _hw

    hardware = _hw()
    with mock.patch("routes.optimizer.detect_hardware", return_value=hardware):
        yield


@pytest.fixture
def client():
    from app import create_app

    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture
def fake_vision():
    provider = mock.MagicMock()
    provider.provider_name = "fake_vision"
    provider.model_name = "fake-vision-1"
    provider.is_available.return_value = True
    provider.analyze_image.return_value = VISION_JSON
    return provider


def _patch_flow(provider):
    return mock.patch("services.ai.screenshot_flow.select_vision_provider",
                      return_value=provider)


class TestJsonApiUnchanged:
    def test_json_request_still_works(self, client, fake_vision):
        with _patch_flow(fake_vision):
            res = client.post("/api/ai-optimize", json={
                "game": "palworld",
                "priority": "balanced",
                "resolution": "1920x1080",
                "target_fps": 60,
                "current_settings": "Preset: High\nShadows: High",
            })
        assert res.status_code == 200
        body = res.get_json()
        assert body["success"] is True
        assert body["result"]["recommended_settings"]
        assert "screenshot" not in body
        # No vision call happened on the plain JSON path.
        assert fake_vision.analyze_image.call_count == 0

    def test_unknown_game_still_404s(self, client):
        res = client.post("/api/ai-optimize", json={"game": "does-not-exist"})
        assert res.status_code == 404

    def test_manual_settings_are_forwarded(self, client):
        with mock.patch("services.ai_optimizer.optimize_game") as mock_opt:
            mock_opt.return_value.to_dict.return_value = {"status": "ok"}
            client.post("/api/ai-optimize", json={
                "game": "palworld",
                "current_settings": "Preset: Ultra",
            })
        sent = mock_opt.call_args.kwargs["current_settings"]
        assert "Preset: Ultra" in sent


class TestMultipartOptimize:
    def test_multipart_with_screenshot(self, client, fake_vision):
        data = {
            "game": "palworld",
            "priority": "balanced",
            "resolution": "1920x1080",
            "target_fps": "60",
            "current_settings": "Preset: Ultra",
            "settings_screenshot": (io.BytesIO(_png()), "shot.png"),
        }
        with _patch_flow(fake_vision):
            res = client.post("/api/ai-optimize", data=data,
                              content_type="multipart/form-data")
        assert res.status_code == 200
        body = res.get_json()
        assert body["success"] is True
        assert body["screenshot"]["source"] == "USER_SCREENSHOT"
        assert fake_vision.analyze_image.call_count == 1

    def test_screenshot_settings_take_priority(self, client, fake_vision):
        data = {
            "game": "palworld",
            "current_settings": "Shadows: Ultra",
            "settings_screenshot": (io.BytesIO(_png()), "shot.png"),
        }
        with _patch_flow(fake_vision), \
             mock.patch("services.ai_optimizer.optimize_game") as mock_opt:
            mock_opt.return_value.to_dict.return_value = {"status": "ok"}
            client.post("/api/ai-optimize", data=data,
                        content_type="multipart/form-data")
        sent = mock_opt.call_args.kwargs["current_settings"]
        screenshot_part = sent.split("\n\n")[0]
        manual_part = sent.split("\n\n")[1]
        assert "screenshot" in screenshot_part.lower()
        assert "shadows: High" in screenshot_part
        assert "supplementary" in manual_part.lower()
        assert "Shadows: Ultra" in manual_part

    def test_multipart_without_file_still_works(self, client):
        res = client.post("/api/ai-optimize", data={
            "game": "palworld",
            "current_settings": "Preset: High",
        }, content_type="multipart/form-data")
        assert res.status_code == 200
        assert res.get_json()["success"] is True

    def test_invalid_screenshot_returns_friendly_error(self, client, fake_vision):
        data = {
            "game": "palworld",
            "settings_screenshot": (io.BytesIO(b"not an image"), "shot.png"),
        }
        with _patch_flow(fake_vision):
            res = client.post("/api/ai-optimize", data=data,
                              content_type="multipart/form-data")
        assert res.status_code == 400
        assert res.get_json()["error"] == \
            "Please upload a valid PNG, JPG, JPEG, or WebP image."

    def test_bad_extension_returns_friendly_error(self, client, fake_vision):
        data = {
            "game": "palworld",
            "settings_screenshot": (io.BytesIO(_png()), "shot.exe"),
        }
        with _patch_flow(fake_vision):
            res = client.post("/api/ai-optimize", data=data,
                              content_type="multipart/form-data")
        assert res.status_code == 400
        assert res.get_json()["code"] == "invalid_extension"


class TestScreenshotEndpoint:
    def test_returns_structured_detection(self, client, fake_vision):
        data = {
            "game": "palworld",
            "settings_screenshot": (io.BytesIO(_png()), "shot.png"),
        }
        with _patch_flow(fake_vision):
            res = client.post("/api/ai/screenshot-settings", data=data,
                              content_type="multipart/form-data")
        assert res.status_code == 200
        body = res.get_json()
        assert body["success"] is True
        report = body["screenshot"]
        assert report["source"] == "USER_SCREENSHOT"
        assert report["requires_confirmation"] is True
        assert report["current_settings_text"]
        assert report["preview"].startswith("data:image/jpeg;base64,")

    def test_response_contains_no_base64_image_payload(self, client, fake_vision):
        data = {
            "game": "palworld",
            "settings_screenshot": (io.BytesIO(_png()), "shot.png"),
        }
        with _patch_flow(fake_vision):
            res = client.post("/api/ai/screenshot-settings", data=data,
                              content_type="multipart/form-data")
        raw = res.get_data(as_text=True)
        # Only the small preview may be base64; the raw upload is not echoed.
        assert raw.count("data:image") == 1
        assert len(raw) < 200_000

    def test_preview_can_be_disabled(self, client, fake_vision):
        data = {
            "game": "palworld",
            "include_preview": "false",
            "settings_screenshot": (io.BytesIO(_png()), "shot.png"),
        }
        with _patch_flow(fake_vision):
            res = client.post("/api/ai/screenshot-settings", data=data,
                              content_type="multipart/form-data")
        assert "preview" not in res.get_json()["screenshot"]

    def test_missing_file_is_rejected(self, client):
        res = client.post("/api/ai/screenshot-settings", data={"game": "palworld"},
                          content_type="multipart/form-data")
        assert res.status_code == 400
        assert res.get_json()["error"]

    def test_json_body_is_rejected(self, client):
        res = client.post("/api/ai/screenshot-settings", json={"game": "palworld"})
        assert res.status_code == 400

    def test_unknown_game_returns_404(self, client, fake_vision):
        data = {"game": "nope", "settings_screenshot": (io.BytesIO(_png()), "s.png")}
        with _patch_flow(fake_vision):
            res = client.post("/api/ai/screenshot-settings", data=data,
                              content_type="multipart/form-data")
        assert res.status_code == 404

    def test_vision_error_is_user_facing(self, client):
        provider = mock.MagicMock()
        provider.is_available.return_value = True
        provider.analyze_image.side_effect = VisionError(
            "Your configured AI model does not support image analysis. "
            "Configure a vision-capable model.")
        data = {"game": "palworld",
                "settings_screenshot": (io.BytesIO(_png()), "shot.png")}
        with _patch_flow(provider):
            res = client.post("/api/ai/screenshot-settings", data=data,
                              content_type="multipart/form-data")
        assert res.status_code == 400
        assert "does not support image analysis" in res.get_json()["error"]

    def test_screenshot_error_is_user_facing(self, client, fake_vision):
        with mock.patch("services.ai.screenshot_flow.analyze_screenshot_upload",
                        side_effect=ScreenshotError("nope", code="corrupt_image")):
            data = {"game": "palworld",
                    "settings_screenshot": (io.BytesIO(_png()), "shot.png")}
            res = client.post("/api/ai/screenshot-settings", data=data,
                              content_type="multipart/form-data")
        assert res.status_code == 400
        assert res.get_json()["code"] == "corrupt_image"

    def test_oversized_upload_is_rejected(self, client, fake_vision):
        big = io.BytesIO()
        Image.new("RGB", (3000, 3000), (1, 2, 3)).save(big, format="PNG")
        payload = big.getvalue()
        with mock.patch("config.Config.SCREENSHOT_MAX_UPLOAD_BYTES", 1024):
            data = {"game": "palworld",
                    "settings_screenshot": (io.BytesIO(payload), "shot.png")}
            with _patch_flow(fake_vision):
                res = client.post("/api/ai/screenshot-settings", data=data,
                                  content_type="multipart/form-data")
        assert res.status_code == 400
        assert res.get_json()["code"] == "too_large"