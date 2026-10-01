"""
services/ai/screenshot_flow.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Orchestration for screenshot-based settings extraction.

Pipeline position (Phase 2)::

    PC detection → game selection → screenshot → vision extraction
      → structured current settings (user confirms) → existing Phase 1
      web research → recommendation → deterministic validation/FPS

This module owns no recommendation logic.  It validates the upload, prepares
the image, asks a vision provider to read it, validates the answer
deterministically, and returns a report the UI can confirm.  Everything happens
in memory: no screenshot is ever written to disk.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from services.ai.vision import (ERR_NO_VISION_PROVIDER, VisionError,
                                VisionProvider, select_vision_provider)
from services.ai.vision_prompts import (SOURCE_SCREENSHOT, SYSTEM_PROMPT,
                                        build_game_context,
                                        build_vision_prompt, parse_vision_response,
                                        validate_extraction)
from services.screenshot import (ScreenshotError, preprocess_image, read_upload,
                                 validate_image)

logger = logging.getLogger(__name__)


def analyze_screenshot_upload(
    upload,
    game: Optional[Dict[str, Any]] = None,
    manual_context: str = "",
    provider: Optional[VisionProvider] = None,
) -> Dict[str, Any]:
    """
    Read one uploaded screenshot and return the structured detection report.

    Parameters
    ----------
    upload:
        A Werkzeug ``FileStorage`` (or anything with ``read``/``filename``/
        ``mimetype``).
    game:
        The selected game dict, used only as vocabulary for the model and for
        deterministic validation.
    manual_context:
        Optional manual notes; passed to the model as supporting context.
    provider:
        Optional vision provider override (tests); otherwise selected from
        configuration.

    Returns
    -------
    dict
        Report with ``source="USER_SCREENSHOT"``, per-setting ``status``,
        ``current_settings_text`` and ``requires_confirmation=True``.

    Raises
    ------
    ScreenshotError
        Rejected upload (extension, MIME, size, corrupt, unsupported format).
    VisionError
        No provider configured, model cannot read images, or the request failed.
    """
    raw = read_upload(upload)
    image_info = validate_image(raw)

    vision = provider or select_vision_provider()
    if vision is None or not vision.is_available():
        raise VisionError(ERR_NO_VISION_PROVIDER)

    data_url, mime_type, prepared = preprocess_image(raw)
    logger.info(
        "Screenshot analysis: provider=%s model=%s format=%s %dx%d prepared=%s "
        "downscaled=%s (image data omitted from log)",
        vision.provider_name, vision.model_name, image_info["format"],
        image_info["width"], image_info["height"], mime_type,
        prepared.get("downscaled"),
    )

    game_context = build_game_context(game)
    prompt = build_vision_prompt(game_context, manual_context)

    response_text = vision.analyze_image(
        image_data_url=data_url,
        prompt=prompt,
        game_context=game,
        system_prompt=SYSTEM_PROMPT,
    )

    parsed = parse_vision_response(response_text)
    result = validate_extraction(parsed, game)

    result.update({
        "provider": vision.provider_name,
        "model": vision.model_name,
        "source": SOURCE_SCREENSHOT,
        "image": {
            "format": image_info["format"],
            "width": image_info["width"],
            "height": image_info["height"],
            "prepared_mime": mime_type,
            "downscaled": bool(prepared.get("downscaled")),
        },
        "raw_settings_text": response_text,
        "manual_context_included": bool(manual_context and manual_context.strip()),
    })
    return result


def build_screenshot_context_text(report: Dict[str, Any]) -> str:
    """
    Render a confirmed detection report as Phase 1 ``current_settings`` text.

    Used when the client sends the confirmed settings back instead of the
    original screenshot, so the pipeline works without re-uploading.
    """
    lines = []
    for entry in report.get("settings", []):
        if entry.get("status") == "unreadable":
            continue
        value = entry.get("value")
        if value in (None, ""):
            continue
        label = entry.get("canonical_name") or entry.get("name") or "Setting"
        lines.append(f"{label}: {value}")
    if not lines:
        return ""
    header = "Settings the user confirmed from their screenshot:"
    return f"{header}\n" + "\n".join(lines)