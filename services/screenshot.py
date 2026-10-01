"""
services/screenshot.py
~~~~~~~~~~~~~~~~~~~~~~
Validation and preprocessing for user-uploaded settings screenshots.

Privacy
-------
Screenshots are processed **in memory only**.  Nothing is written to disk, no
temporary files are created, and no image data or base64 payload is ever
logged.

Security
--------
Every upload is checked on four independent levels:
1. file extension (client supplied, untrusted),
2. MIME type (client supplied, untrusted),
3. file size,
4. actual decoding with Pillow (the only trustworthy check).

The client-supplied filename is never used for filesystem access — this module
never touches the filesystem at all.
"""
from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any, Dict, Optional, Tuple

from config import Config

logger = logging.getLogger(__name__)

#: Formats a vision model can be given directly without re-encoding.
_PASSTHROUGH_FORMATS = ("PNG", "JPEG", "WEBP")


class ScreenshotError(Exception):
    """
    A rejected upload.  ``message`` is user-facing and never contains image
    data, file paths or base64 content.
    """

    def __init__(self, message: str, code: str = "invalid_image") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def _allowed_extensions() -> Tuple[str, ...]:
    return tuple(Config.SCREENSHOT_ALLOWED_EXTENSIONS)


def _allowed_mime_types() -> Tuple[str, ...]:
    return tuple(Config.SCREENSHOT_ALLOWED_MIME_TYPES)


def read_upload(file_storage) -> bytes:
    """
    Read an uploaded file into memory, enforcing the size cap.

    The client-supplied filename is used only to read the extension; it is
    never used to build a path.
    """
    if file_storage is None:
        raise ScreenshotError(Config.SCREENSHOT_INVALID_MESSAGE)

    # Never trust the client filename for anything but an extension check.
    filename = os.path.basename(str(getattr(file_storage, "filename", "") or ""))
    extension = os.path.splitext(filename)[1].lower()
    if extension not in _allowed_extensions():
        raise ScreenshotError(
            Config.SCREENSHOT_INVALID_MESSAGE, code="invalid_extension")

    mime_type = str(getattr(file_storage, "mimetype", "") or "").lower()
    if mime_type and mime_type not in _allowed_mime_types():
        raise ScreenshotError(
            Config.SCREENSHOT_INVALID_MESSAGE, code="invalid_mime_type")

    max_bytes = int(Config.SCREENSHOT_MAX_UPLOAD_BYTES)
    stream = getattr(file_storage, "stream", None)
    if stream is not None and hasattr(stream, "seek"):
        try:
            stream.seek(0)
        except (OSError, ValueError):
            pass
    try:
        data = file_storage.read(max_bytes + 1)
    except OSError as exc:
        raise ScreenshotError(
            Config.SCREENSHOT_INVALID_MESSAGE, code="unreadable") from exc
    if not data:
        raise ScreenshotError(Config.SCREENSHOT_INVALID_MESSAGE, code="empty_file")
    if len(data) > max_bytes:
        raise ScreenshotError(
            Config.SCREENSHOT_TOO_LARGE_MESSAGE, code="too_large")
    return data


def validate_image(data: bytes) -> Dict[str, Any]:
    """
    Decode the image and return its metadata.

    Raises
    ------
    ScreenshotError
        If the bytes are not a decodable PNG/JPEG/WebP image (this is what
        rejects executables, PDFs, renamed files and corrupted images).
    """
    try:
        from PIL import Image  # imported lazily: keeps startup cheap
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ScreenshotError(
            "Image support is unavailable on the server.", code="no_image_lib") from exc

    if not data:
        raise ScreenshotError(Config.SCREENSHOT_INVALID_MESSAGE, code="empty_file")

    try:
        with Image.open(io.BytesIO(data)) as image:
            # force_decode() touches the pixel data, so truncated or fake
            # files are rejected here rather than by the AI provider.
            image.load()
            fmt = (image.format or "").upper()
            width, height = image.size
            mode = image.mode
    except ScreenshotError:
        raise
    except Exception as exc:  # pylint: disable=broad-except
        logger.info("Rejected screenshot upload: not a decodable image (%s)",
                    type(exc).__name__)
        raise ScreenshotError(
            Config.SCREENSHOT_INVALID_MESSAGE, code="corrupt_image") from exc

    if fmt not in _PASSTHROUGH_FORMATS:
        raise ScreenshotError(
            Config.SCREENSHOT_INVALID_MESSAGE, code="unsupported_format")

    if width <= 0 or height <= 0:
        raise ScreenshotError(
            Config.SCREENSHOT_INVALID_MESSAGE, code="invalid_dimensions")

    return {"format": fmt, "width": width, "height": height, "mode": mode,
            "bytes": len(data)}


def preprocess_image(data: bytes) -> Tuple[bytes, str, Dict[str, Any]]:
    """
    Prepare an uploaded image for a vision model.

    - Downscales only images larger than ``SCREENSHOT_MAX_DIMENSION`` (small
      screenshots are never upscaled).
    - Keeps PNG/WebP when they are already small enough, so UI text stays
      crisp; converts oversized non-JPEG images to high-quality JPEG instead
      of shipping a huge base64 payload.
    - Never downscales below a floor where the settings text would become
      unreadable.

    Returns
    -------
    (data_url, mime_type, metadata)
        ``metadata`` contains only numbers and format names — never image data.

    Raises
    ------
    ScreenshotError
        When the image cannot be decoded or re-encoded.
    """
    from PIL import Image

    max_dimension = int(Config.SCREENSHOT_MAX_DIMENSION)
    quality = int(Config.SCREENSHOT_JPEG_QUALITY)
    # Text-heavy UI screenshots: do not shrink below this or labels blur.
    min_dimension = 1024

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            fmt = (image.format or "").upper()
            source_w, source_h = image.size
    except Exception as exc:  # pylint: disable=broad-except
        raise ScreenshotError(
            Config.SCREENSHOT_INVALID_MESSAGE, code="corrupt_image") from exc

    longest = max(source_w, source_h)
    needs_downscale = longest > max_dimension

    if not needs_downscale:
        payload, mime = data, _mime_for(fmt)
        return _to_data_url(payload, mime), mime, {
            "source_format": fmt,
            "width": source_w,
            "height": source_h,
            "downscaled": False,
            "bytes": len(payload),
        }

    scale = max_dimension / float(longest)
    target_w = max(min_dimension, int(source_w * scale))
    target_h = max(min_dimension, int(source_h * scale))
    # Keep the aspect ratio; only clamp against the readability floor.
    if target_w > max_dimension:
        target_w = max_dimension
    if target_h > max_dimension:
        target_h = max_dimension

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            working = image.convert("RGB")
            resized = working.resize((target_w, target_h), Image.LANCZOS)
            buffer = io.BytesIO()
            resized.save(buffer, format="JPEG", quality=quality, optimize=True)
            payload = buffer.getvalue()
    except ScreenshotError:
        raise
    except Exception as exc:  # pylint: disable=broad-except
        raise ScreenshotError(
            "That image could not be prepared for analysis.",
            code="preprocess_failed") from exc

    return _to_data_url(payload, "image/jpeg"), "image/jpeg", {
        "source_format": fmt,
        "width": source_w,
        "height": source_h,
        "output_width": target_w,
        "output_height": target_h,
        "downscaled": True,
        "bytes": len(payload),
    }


def _mime_for(image_format: str) -> str:
    if image_format == "PNG":
        return "image/png"
    if image_format == "WEBP":
        return "image/webp"
    return "image/jpeg"


def _to_data_url(payload: bytes, mime_type: str) -> str:
    """Build the ``data:`` URL expected by multimodal chat APIs."""
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_preview_data_url(data: bytes, max_dimension: int = 480) -> Optional[str]:
    """
    Build a small data URL for the browser preview only.

    Kept small on purpose: the preview never leaves the browser.
    """
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            working = image.convert("RGB")
            working.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
            buffer = io.BytesIO()
            working.save(buffer, format="JPEG", quality=80, optimize=True)
            return _to_data_url(buffer.getvalue(), "image/jpeg")
    except Exception as exc:  # pylint: disable=broad-except
        logger.info("Screenshot preview generation failed: %s", type(exc).__name__)
        return None