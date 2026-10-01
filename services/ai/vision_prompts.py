"""
services/ai/vision_prompts.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Prompt construction and deterministic post-processing for screenshot-based
settings extraction.

Design rules enforced here
--------------------------
- The vision model may only report what is *visible* in the screenshot.  It is
  explicitly told not to invent values, and unknown settings are never dropped
  silently — they are marked ``unrecognized`` so the user can decide.
- The game's known settings (from ``data/games.json``) are supplied as a
  vocabulary hint only.  A value is accepted only when the model actually read
  it from the image.
- Vision output is the source of **current settings** (user screenshot).
  It is never treated as benchmark evidence or as a performance estimate —
  deterministic FPS stays authoritative.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Canonical setting vocabulary ───────────────────────────────────────────────
# Maps a screenshot label to the internal Phase 1 setting it feeds.  Keys are
# normalised lowercase labels.
CANONICAL_SETTINGS: Dict[str, str] = {
    "graphics preset": "graphics_preset",
    "preset": "graphics_preset",
    "quality preset": "graphics_preset",
    "display mode": "display_mode",
    "screen mode": "display_mode",
    "window mode": "display_mode",
    "fullscreen": "display_mode",
    "resolution": "resolution",
    "screen resolution": "resolution",
    "output resolution": "resolution",
    "view distance": "view_distance",
    "draw distance": "view_distance",
    "render distance": "view_distance",
    "level of detail": "view_distance",
    "shadows": "shadows",
    "shadow quality": "shadows",
    "shadow": "shadows",
    "sun shadows": "shadows",
    "contact shadows": "shadows",
    "ambient occlusion": "ambient_occlusion",
    "ao": "ambient_occlusion",
    "effects": "effects",
    "effect quality": "effects",
    "post processing": "effects",
    "post-processing": "effects",
    "textures": "textures",
    "texture quality": "textures",
    "texture": "textures",
    "texture filtering": "textures",
    "anisotropic filtering": "textures",
    "anti aliasing": "anti_aliasing",
    "anti-aliasing": "anti_aliasing",
    "antialiasing": "anti_aliasing",
    "aa": "anti_aliasing",
    "msaa": "anti_aliasing",
    "motion blur": "motion_blur",
    "camera blur": "motion_blur",
    "vsync": "vsync",
    "v-sync": "vsync",
    "vertical sync": "vsync",
    "fps limit": "fps_limit",
    "frame rate limit": "fps_limit",
    "framerate limit": "fps_limit",
    "refresh rate": "refresh_rate",
    "upscaling": "upscaling",
    "upscaler": "upscaling",
    "fsr": "upscaling",
    "amd fsr": "upscaling",
    "fsr 2": "upscaling",
    "fsr 3": "upscaling",
    "dlss": "upscaling",
    "nvidia dlss": "upscaling",
    "xess": "upscaling",
    "intel xess": "upscaling",
    "tsr": "upscaling",
    "dynamic resolution": "upscaling",
    "render scale": "render_scale",
    "render resolution": "render_scale",
    "scalability": "render_scale",
    "graphics api": "graphics_api",
    "directx version": "graphics_api",
    "graphics engine": "graphics_engine",
    "renderer": "graphics_engine",
    "ssao": "ssao_quality",
    "ssao quality": "ssao_quality",
    "gpu particles": "gpu_particles",
    "gpu particles quality": "gpu_particles",
    "enhanced decals": "enhanced_decals",
    "decals": "enhanced_decals",
    "fov": "field_of_view",
    "field of view": "field_of_view",
    "foliage": "foliage",
    "grass": "foliage",
    "foliage density": "foliage",
    "weather": "weather",
    "volumetric clouds": "weather",
    "clouds": "weather",
    "screen space reflections": "reflections",
    "ssr": "reflections",
    "ray tracing": "ray_tracing",
    "raytracing": "ray_tracing",
    "global illumination": "global_illumination",
    "gi": "global_illumination",
    "gamma": "gamma",
    "brightness": "gamma",
    "anisotropy": "textures",
    "crowd density": "view_distance",
    "character quality": "character_quality",
    "water quality": "water_quality",
    "reflection quality": "reflections",
    "shadow quality preset": "shadows",
    "texture streaming": "textures",
    "mesh quality": "mesh_quality",
    "mesh detail": "mesh_quality",
}

#: Accepted values per canonical setting (used for deterministic validation).
SUPPORTED_VALUES: Dict[str, Tuple[str, ...]] = {
    "graphics_preset": ("Low", "Normal", "Medium", "High", "Very High", "Ultra",
                        "Maximum", "Extreme", "Custom", "Manual"),
    "view_distance": ("Low", "Medium", "High", "Ultra", "Maximum"),
    "shadows": ("Off", "Low", "Medium", "High", "Ultra", "Extreme"),
    "effects": ("Off", "Low", "Medium", "High", "Ultra", "Extreme"),
    "textures": ("Off", "Low", "Medium", "High", "Ultra", "Extreme"),
    "ambient_occlusion": ("Off", "Low", "Medium", "High", "Ultra"),
    "motion_blur": ("Off", "Low", "Medium", "High"),
    "vsync": ("Off", "On", "Double Buffer", "Triple Buffer"),
    "anti_aliasing": ("Off", "TAA", "FXAA", "MSAA", "TAA 2x", "SMAA", "SMAA 2x",
                      "DLSS", "None"),
    "upscaling": ("Off", "Quality", "Balanced", "Performance", "Ultra Performance",
                  "Ultra Perf", "Native", "Auto"),
    "display_mode": ("Fullscreen", "Borderless Windowed", "Windowed"),
    "ray_tracing": ("Off", "Low", "Medium", "High", "Ultra", "On"),
    "global_illumination": ("Off", "Low", "Medium", "High", "Ultra"),
    "reflections": ("Off", "Low", "Medium", "High", "Ultra"),
    "foliage": ("Off", "Low", "Medium", "High", "Ultra"),
    "weather": ("Off", "Low", "Medium", "High", "Ultra"),
    "water_quality": ("Off", "Low", "Medium", "High", "Ultra"),
    "mesh_quality": ("Low", "Medium", "High", "Ultra"),
    "character_quality": ("Low", "Medium", "High", "Ultra"),
    "render_scale": ("50", "60", "70", "80", "90", "100"),
    "gamma": ("0", "1", "2", "3"),
}

#: Settings whose value must be numeric.
NUMERIC_SETTINGS = ("fps_limit", "refresh_rate", "gamma")

CONFIDENCE_VALUES = ("high", "medium", "low")

STATUS_RECOGNIZED = "recognized"
STATUS_UNRECOGNIZED = "unrecognized"
STATUS_UNREADABLE = "unreadable"

SOURCE_SCREENSHOT = "USER_SCREENSHOT"

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are extracting graphics settings from a game settings screenshot.

ABSOLUTE RULES:
1. Only report a setting if you can actually SEE it in the screenshot.
2. NEVER guess, complete or infer a value. If a setting is not visible, omit it.
3. If a setting label is visible but its value is unreadable or covered, return
   the setting with "value": null and "confidence": "low", and list its name in
   "unreadable".
4. NEVER invent a setting merely because the game is known to have it. The
   reference list of known settings is vocabulary help only — visibility in the
   screenshot is what matters.
5. Copy values exactly as displayed (e.g. "High", "Ultra", "1920 x 1080").
6. Treat "FSR", "AMD FSR", "Upscaling", "Upscaler", "DLSS" and "XeSS" as
   upscaling settings and report them under the label shown on screen.
7. Your entire response MUST be a single valid JSON object and nothing else.
8. No markdown fences, no commentary, no trailing text.

Required JSON schema:
{
  "settings": [
    {
      "name": "<exact label as shown on screen>",
      "value": "<exact value, or null when unreadable>",
      "confidence": "<high|medium|low>"
    }
  ],
  "unreadable": ["<label of a setting whose value could not be read>"],
  "notes": ["<short observation, e.g. 'menu partially covered by overlay'>"]
}

Report every setting row you can read, in the order it appears. Empty arrays are
fine when the screenshot shows no readable settings.
"""


def build_game_context(game: Optional[Dict[str, Any]]) -> str:
    """
    Render the selected game's known settings as a vocabulary hint.

    This is context only: the model must still see each setting in the image.
    """
    if not game:
        return "No game is selected."

    lines: List[str] = []
    name = game.get("name") or game.get("slug") or "Unknown Game"
    lines.append(f"Selected game: {name}")

    settings_detail = game.get("settings_detail") or []
    known: List[str] = []
    for entry in settings_detail:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("name", "")).strip()
        if label:
            known.append(label)

    upscaling = game.get("upscaling_support") or []
    resolutions = game.get("resolutions") or []
    recommended = game.get("recommended_settings") or {}

    if known:
        lines.append("Known settings labels for this game (vocabulary only): "
                     + ", ".join(known))
    if upscaling:
        lines.append("Upscalers supported by this game: " + ", ".join(
            str(u) for u in upscaling))
    if resolutions:
        lines.append("Supported resolutions: " + ", ".join(
            str(r) for r in resolutions))
    if isinstance(recommended, dict) and recommended.get("preset"):
        lines.append("Typical preset for this game: "
                     f"{recommended.get('preset')} — do not report it unless "
                     f"it is visible in the screenshot.")

    lines.append(
        "Only report values that are visible in the screenshot. If a setting "
        "from this list is not shown in the image, do not include it."
    )
    return "\n".join(lines)


def build_vision_prompt(game_context: str, manual_context: str = "") -> str:
    """Build the user-turn instruction for the screenshot analysis."""
    parts = [
        "Read the attached screenshot of the game's graphics settings menu.",
        "",
        game_context or "",
    ]
    if manual_context and manual_context.strip():
        parts += [
            "",
            "The user also typed these notes about their settings "
            "(supporting context only — prefer what you can see):",
            manual_context.strip(),
        ]
    parts += [
        "",
        'Return only the JSON object described in your instructions.',
    ]
    return "\n".join(parts)


# ── Parsing and deterministic validation ──────────────────────────────────────

def parse_vision_response(raw: str) -> Dict[str, Any]:
    """
    Parse a model response into the structured extraction shape.

    Raises
    ------
    VisionParseError
        When the response is not usable JSON.  Callers surface a friendly
        message and let the user fall back to manual entry.
    """
    if not raw or not raw.strip():
        raise VisionParseError("The vision model returned an empty response.")

    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    data: Any = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(text)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None

    if not isinstance(data, dict):
        raise VisionParseError(
            "The vision model did not return a JSON object.")

    settings_raw = data.get("settings")
    if settings_raw is None:
        settings_raw = []
    if not isinstance(settings_raw, list):
        settings_raw = []

    settings: List[Dict[str, Any]] = []
    for item in settings_raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        if not name:
            continue
        value = item.get("value")
        if isinstance(value, (int, float)):
            value = str(value)
        if value is not None:
            value = str(value).strip()
            value = value or None
        confidence = str(item.get("confidence", "") or "").strip().lower()
        if confidence not in CONFIDENCE_VALUES:
            confidence = "low" if value is None else "medium"
        settings.append({
            "name": name,
            "value": value,
            "confidence": confidence,
        })

    unreadable: List[str] = []
    for item in data.get("unreadable") or []:
        if isinstance(item, str) and item.strip():
            unreadable.append(item.strip())

    notes: List[str] = []
    for item in data.get("notes") or []:
        if isinstance(item, str) and item.strip():
            notes.append(item.strip())

    return {"settings": settings, "unreadable": unreadable, "notes": notes}


class VisionParseError(Exception):
    """Raised when a vision response cannot be parsed into JSON."""


def normalize_label(label: str) -> str:
    """Normalise a screenshot label for vocabulary lookup."""
    text = str(label or "").lower()
    text = re.sub(r"[\s_\-/]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _build_normalized_vocabulary() -> Dict[str, str]:
    """Normalise every vocabulary key once ("V-Sync" and "vsync" both match)."""
    vocabulary: Dict[str, str] = {}
    for label, key in CANONICAL_SETTINGS.items():
        vocabulary.setdefault(normalize_label(label), key)
    return vocabulary


_NORMALIZED_VOCABULARY = _build_normalized_vocabulary()


def canonical_key(label: str) -> Optional[str]:
    """Return the internal setting key for a screenshot label, if known."""
    normalized = normalize_label(label)
    if normalized in _NORMALIZED_VOCABULARY:
        return _NORMALIZED_VOCABULARY[normalized]
    # Tolerate suffixes/prefixes such as "Shadow Quality High" or
    # "texture quality: ultra".
    for candidate in sorted(_NORMALIZED_VOCABULARY, key=len, reverse=True):
        if candidate in normalized:
            return _NORMALIZED_VOCABULARY[candidate]
    return None


def normalize_value(value: Any) -> str:
    """Normalise a value for comparison (case/space insensitive)."""
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def validate_value(key: Optional[str], value: Optional[str]) -> Tuple[bool, str]:
    """
    Deterministically validate a value against the supported value list.

    Returns ``(ok, normalized_value)``.  Unknown values are kept verbatim but
    reported as not matching, so the user sees them in the confirmation step
    instead of them being silently discarded.
    """
    if value is None:
        return True, ""
    allowed = SUPPORTED_VALUES.get(key or "")
    if not allowed:
        return True, str(value)
    text = str(value).strip()
    normalized = normalize_value(text)
    for candidate in allowed:
        if normalize_value(candidate) == normalized:
            return True, candidate
    # Frequencies/percent-style equivalents: "60 fps" → 60, "144 Hz" → 144.
    numbers = re.findall(r"\d+", normalized)
    if numbers and key in NUMERIC_SETTINGS:
        for candidate in allowed:
            if candidate in numbers:
                return True, candidate
    # e.g. "High (Ultra)" → High
    for candidate in allowed:
        if normalized.startswith(normalize_value(candidate)):
            return True, candidate
    return False, text


def normalize_resolution(value: Optional[str]) -> Optional[str]:
    """
    Normalise a resolution string to ``WIDTHxHEIGHT``.

    Returns ``None`` for anything that is not a plausible resolution so an
    unreadable/garbled value never silently becomes a real setting.
    """
    if not value:
        return None
    text = str(value).lower().replace("×", "x").replace("*", "x")
    numbers = re.findall(r"\d{3,5}", text)
    if len(numbers) < 2:
        return None
    width, height = int(numbers[0]), int(numbers[1])
    if not (640 <= width <= 16384 and 480 <= height <= 16384):
        return None
    return f"{width}x{height}"


def build_settings_text(settings: List[Dict[str, Any]]) -> str:
    """
    Render detected settings as the Phase 1 ``current_settings`` free text.

    Only settings that were actually read with a usable value are included —
    unreadable and unrecognized rows are never presented as current state.
    """
    lines: List[str] = []
    for item in settings:
        if item.get("status") == STATUS_UNREADABLE:
            continue
        value = item.get("value")
        if value in (None, ""):
            continue
        label = item.get("canonical_name") or item.get("name") or "Setting"
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


#: Upscaler families we can name from a screenshot label (never from a value).
UPSCALER_FAMILIES = ("dlss", "fsr", "xess", "tsr", "nis")


def _detect_upscaler_name(label: str) -> Optional[str]:
    """Return the upscaler family named in a label, e.g. "AMD FSR" → ``fsr``."""
    normalized = normalize_label(label)
    for family in UPSCALER_FAMILIES:
        if family in normalized.split() or family in normalized:
            return family
    return None


def validate_extraction(
    parsed: Dict[str, Any],
    game: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Deterministically validate an extraction against the selected game.

    Produces the structured result consumed by the confirmation UI:

        {
          "source": "USER_SCREENSHOT",
          "settings": [
            {"name", "value", "confidence", "canonical_name", "status",
             "supported_values", "note"}
          ],
          "recognized", "unrecognized", "unreadable", "notes",
          "current_settings_text", "requires_confirmation": True
        }

    Nothing here raises: unknown settings are marked, never dropped.
    """
    game = game or {}
    game_known_labels = {
        normalize_label(entry.get("name", ""))
        for entry in (game.get("settings_detail") or [])
        if isinstance(entry, dict)
    }
    game_known_keys = {
        canonical_key(entry.get("name", ""))
        for entry in (game.get("settings_detail") or [])
        if isinstance(entry, dict)
    }
    game_known_keys.discard(None)
    game_upscaling = {
        normalize_value(u) for u in (game.get("upscaling_support") or [])
    }
    game_resolutions = {
        normalize_value(r) for r in (game.get("resolutions") or [])
    }

    declared_unreadable = {
        normalize_label(label) for label in parsed.get("unreadable", [])
    }

    validated: List[Dict[str, Any]] = []
    for item in parsed.get("settings", []):
        label = item["name"]
        value = item.get("value")
        confidence = item.get("confidence", "low")
        key = canonical_key(label)

        note = ""
        status = STATUS_RECOGNIZED if key else STATUS_UNRECOGNIZED

        if value in (None, ""):
            status = STATUS_UNREADABLE
            confidence = "low"
            note = "Value not readable in the screenshot."
        else:
            if key == "resolution":
                normalized_resolution = normalize_resolution(value)
                if normalized_resolution is None:
                    status = STATUS_UNREADABLE
                    confidence = "low"
                    note = f"Could not read a resolution from '{value}'."
                    value = None
                else:
                    value = normalized_resolution
                    if game_resolutions and normalize_value(value) not in game_resolutions:
                        note = (f"Resolution {value} is not listed for this "
                                f"game in Potato Wiz.")
            else:
                ok, normalized = validate_value(key, value)
                value = normalized
                if not ok:
                    note = (f"'{value}' is not a standard Potato Wiz value for "
                            f"{label}; kept as read.")

            if key == "upscaling" and game_upscaling:
                upscaler = _detect_upscaler_name(label)
                if upscaler and not any(
                    upscaler in u or u in upscaler for u in game_upscaling
                ):
                    note = (f"The upscaler '{upscaler}' read from the screenshot "
                            f"is not in this game's listed upscaler support.")

        if (status == STATUS_RECOGNIZED
                and normalize_label(label) not in game_known_labels
                and (key or "") not in game_known_keys):
            # Recognised by vocabulary but not in this game's stored list —
            # the user still sees it, so this is informational only.
            note = note or "Not listed in this game's stored settings; read from your screenshot."

        if normalize_label(label) in declared_unreadable and status != STATUS_UNREADABLE:
            note = note or "The vision model flagged this row as hard to read."

        entry: Dict[str, Any] = {
            "name": label,
            "value": value,
            "confidence": confidence,
            "canonical_name": key or label,
            "status": status,
            "source": SOURCE_SCREENSHOT,
            "supported_values": list(SUPPORTED_VALUES.get(key or "", [])),
        }
        if note:
            entry["note"] = note
        validated.append(entry)

    result: Dict[str, Any] = {
        "source": SOURCE_SCREENSHOT,
        "settings": validated,
        "recognized": [e for e in validated if e["status"] == STATUS_RECOGNIZED],
        "unrecognized": [e for e in validated if e["status"] == STATUS_UNRECOGNIZED],
        "unreadable": [e for e in validated if e["status"] == STATUS_UNREADABLE],
        "notes": list(parsed.get("notes", [])),
        "requires_confirmation": True,
    }
    result["current_settings_text"] = build_settings_text(validated)
    result["recognized_count"] = len(result["recognized"])
    return result