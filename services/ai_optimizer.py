"""
services/ai_optimizer.py
~~~~~~~~~~~~~~~~~~~~~~~~
Main facade for the Potato Wiz AI optimization pipeline.

Public API
----------
optimize_game(hardware, game, target_fps, priority, resolution, current_settings)
    → AIOptimizationResult

Architecture
------------
Hardware + Game + Benchmark data
        ↓
build_research_context()          ← assembles authoritative local evidence
        ↓
Provider selection (auto/ollama/openrouter)
        ↓
AI call (with web research tools if OpenRouter)
        ↓
_parse_ai_response()              ← extract JSON from model output
        ↓
_validate_and_merge()             ← reject/fix invalid settings; merge deterministic FPS
        ↓
AIOptimizationResult              ← never contains invented FPS numbers

Fallback
--------
If the AI call fails for any reason the function falls back to the
deterministic optimizer (services/optimizer.py) and returns the result
wrapped in an AIOptimizationResult with status=AI_STATUS_FALLBACK.
The website always produces a usable recommendation.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from config import Config
from models.models import (
    AIOptimizationResult,
    AIRecommendedSettings,
    EvidenceItem,
    HardwareSpecs,
    AI_PROVIDER_NONE,
    AI_PROVIDER_OLLAMA,
    AI_PROVIDER_OPENROUTER,
    AI_STATUS_OK,
    AI_STATUS_FALLBACK,
    AI_STATUS_ERROR,
    EVIDENCE_FALLBACK_ESTIMATE,
    EVIDENCE_AI_INFERENCE,
    EVIDENCE_MEASURED_BENCHMARK,
    EVIDENCE_PUBLISHED_BENCHMARK,
    EVIDENCE_COMMUNITY_REPORT,
    EVIDENCE_OFFICIAL_INFO,
    EVIDENCE_GUIDE,
)
from services.ai.base import AIProviderError
from services.ai.prompts import SYSTEM_PROMPT, build_user_prompt
from services.ai.research import build_research_context, deterministic_fps_to_display

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid setting values for deterministic validation
# ---------------------------------------------------------------------------

_VALID_PRESETS = {
    "Low", "Normal", "Medium", "High", "Very High", "Ultra", "Maximum", "Extreme"
}
_VALID_VIEW_DISTANCE = {"Low", "Medium", "High", "Ultra"}
_VALID_SHADOWS = {"Off", "Low", "Medium", "High", "Ultra"}
_VALID_EFFECTS = {"Low", "Medium", "High", "Ultra"}
_VALID_TEXTURES = {"Low", "Medium", "High", "Ultra"}
_VALID_AA = {"Off", "TAA", "FXAA", "MSAA", "SMAA"}
_VALID_MOTION_BLUR = {"Off", "Low", "Medium", "High"}
_VALID_VSYNC = {"Off", "On"}
_VALID_UPSCALING = {
    "Off", "Quality", "Balanced", "Performance", "Ultra Performance",
    "DLSS Quality", "DLSS Balanced", "DLSS Performance",
    "FSR Quality", "FSR Balanced", "FSR Performance", "FSR Ultra Perf",
    "XeSS Quality", "XeSS Balanced",
}
_VALID_RESOLUTIONS = {
    "1280x720", "1600x900", "1920x1080", "2560x1440", "3840x2160"
}
_VALID_EVIDENCE_TYPES = {
    EVIDENCE_MEASURED_BENCHMARK,
    EVIDENCE_PUBLISHED_BENCHMARK,
    EVIDENCE_COMMUNITY_REPORT,
    EVIDENCE_OFFICIAL_INFO,
    EVIDENCE_GUIDE,
    EVIDENCE_AI_INFERENCE,
    EVIDENCE_FALLBACK_ESTIMATE,
}

# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

def _select_provider():
    """
    Return an instantiated provider based on Config.AI_PROVIDER.

    Priority when AI_PROVIDER=="auto":
        1. OpenRouter (if key configured)
        2. Ollama (if base_url configured)
        3. None  → deterministic fallback

    Returns None when no provider is available.
    """
    from services.ai.ollama_provider import OllamaProvider
    from services.ai.openrouter_provider import OpenRouterProvider

    pref = (Config.AI_PROVIDER or "auto").lower().strip()

    if pref == "openrouter":
        p = OpenRouterProvider()
        if p.is_available():
            return p
        logger.warning("AI_PROVIDER=openrouter but no key configured.")
        return None

    if pref == "ollama":
        p = OllamaProvider()
        if p.is_available():
            return p
        logger.warning("AI_PROVIDER=ollama but Ollama is not configured.")
        return None

    # auto
    or_provider = OpenRouterProvider()
    if or_provider.is_available():
        logger.debug("AI provider selected: OpenRouter (%s)", or_provider.model_name)
        return or_provider

    ol_provider = OllamaProvider()
    if ol_provider.is_available():
        logger.debug("AI provider selected: Ollama (%s)", ol_provider.model_name)
        return ol_provider

    logger.info("No AI provider available — will use deterministic optimizer.")
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def optimize_game(
    hardware: HardwareSpecs,
    game: Dict[str, Any],
    target_fps: int = 60,
    priority: str = "balanced",
    resolution: str = "1920x1080",
    current_settings: str = "",
) -> AIOptimizationResult:
    """
    Run the full AI optimization pipeline for a given game and hardware.

    Always returns a valid AIOptimizationResult — never raises.
    Falls back to the deterministic optimizer on any failure.

    Parameters
    ----------
    hardware:          Detected PC hardware specs.
    game:              Game dict from games.json.
    target_fps:        User's desired FPS target.
    priority:          "fps" | "balanced" | "quality".
    resolution:        Target resolution string.
    current_settings:  Optional free-text of current in-game settings.

    Returns
    -------
    AIOptimizationResult
    """
    game_name = game.get("name", "Unknown Game")
    logger.info(
        "AI optimize: game=%s gpu=%s res=%s fps=%d priority=%s",
        game_name, hardware.gpu, resolution, target_fps, priority,
    )

    # Always compute deterministic fallback first — it's the safety net
    fallback = _get_deterministic_result(hardware, game, target_fps, priority, resolution)

    # Get deterministic preset for benchmark lookup
    det_preset = fallback.preset if fallback else "High"

    # Assemble research context (benchmark lookup + FPS estimate)
    try:
        research_ctx = build_research_context(
            hardware=hardware,
            game=game,
            preset=det_preset,
            resolution=resolution,
            upscaling_mode=fallback.upscaling_mode if fallback else "Off",
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Research context failed: %s", exc)
        research_ctx = None

    # Select AI provider
    provider = _select_provider()
    if provider is None:
        logger.info("No AI provider — returning deterministic result.")
        return _wrap_deterministic(fallback, research_ctx)

    # Build prompts
    try:
        benchmark_ctx_text = (
            research_ctx.benchmark_context_text if research_ctx else ""
        )
        user_prompt = build_user_prompt(
            hardware=hardware,
            game=game,
            target_fps=target_fps,
            priority=priority,
            resolution=resolution,
            benchmark_context=benchmark_ctx_text,
            current_settings_text=current_settings,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Prompt construction failed: %s", exc)
        return _wrap_deterministic(fallback, research_ctx)

    # Call the provider
    try:
        logger.debug("Calling provider %s…", provider.provider_name)
        raw_response = provider.complete(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
    except AIProviderError as exc:
        logger.warning("AI provider error (%s): %s", provider.provider_name, exc)
        return _wrap_deterministic(
            fallback, research_ctx,
            warning=f"AI provider unavailable ({provider.provider_name}): {exc}",
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Unexpected AI error: %s", exc)
        return _wrap_deterministic(
            fallback, research_ctx,
            warning=f"Unexpected AI error: {exc}",
        )

    # Parse the AI response
    ai_json, parse_warning = _parse_ai_response(raw_response)
    if ai_json is None:
        return _wrap_deterministic(
            fallback, research_ctx,
            warning=parse_warning or "AI returned unparseable response.",
        )

    # Validate and build the structured result
    try:
        result = _build_result(
            ai_json=ai_json,
            provider=provider,
            research_ctx=research_ctx,
            fallback=fallback,
            resolution=resolution,
            game=game,
        )
        if parse_warning:
            result.warnings.append(parse_warning)
        logger.info(
            "AI optimize complete: provider=%s confidence=%s fps=%s",
            result.provider, result.confidence, result.estimated_fps,
        )
        return result
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Result construction failed: %s", exc)
        return _wrap_deterministic(
            fallback, research_ctx,
            warning=f"AI result processing error: {exc}",
        )


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_ai_response(raw: str) -> tuple[Optional[Dict], Optional[str]]:
    """
    Extract JSON from the model's raw response text.

    Returns (parsed_dict, warning_string).
    If parsing fails, parsed_dict is None.
    """
    if not raw or not raw.strip():
        return None, "AI response was empty."

    # Try direct parse first
    try:
        return json.loads(raw.strip()), None
    except json.JSONDecodeError:
        pass

    # Try to extract a JSON block from markdown fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1)), None
        except json.JSONDecodeError:
            pass

    # Try to find the first {...} block in the text
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0)), (
                "AI response was not clean JSON — extracted from prose."
            )
        except json.JSONDecodeError:
            pass

    return None, f"Could not parse AI response as JSON. Preview: {raw[:200]!r}"


# ---------------------------------------------------------------------------
# Result construction and validation
# ---------------------------------------------------------------------------

def _build_result(
    ai_json: Dict,
    provider,
    research_ctx,
    fallback,
    resolution: str,
    game: Dict,
) -> AIOptimizationResult:
    """
    Build a validated AIOptimizationResult from the AI's parsed JSON.

    Validation rules:
    - Settings values must be in the allowed sets defined at module level.
    - Resolution must be known; falls back to user-requested resolution.
    - Upscaling must be a known value; falls back to "Off".
    - FPS numbers from the AI are DISCARDED unless the source is credible.
    - FPS values are always taken from the deterministic estimator or the
      benchmark service — never invented by the AI.
    - Evidence items with missing/fabricated URLs are stripped.
    """
    warnings: List[str] = []

    # ── Recommended settings ─────────────────────────────────────────────────
    raw_settings = ai_json.get("recommended_settings") or {}
    settings = _validate_settings(raw_settings, fallback, resolution, game, warnings)

    # ── Changes list ─────────────────────────────────────────────────────────
    changes = _clean_changes(ai_json.get("changes") or [])

    # ── FPS — NEVER use AI-invented numbers ─────────────────────────────────
    estimated_fps, fps_source = _safe_fps(ai_json, research_ctx)

    # ── Confidence ───────────────────────────────────────────────────────────
    confidence = _clean_confidence(ai_json.get("confidence", "low"))

    # ── Evidence ─────────────────────────────────────────────────────────────
    evidence = _parse_evidence(ai_json.get("evidence") or [])
    # Merge in pre-seeded benchmark evidence
    if research_ctx and research_ctx.initial_evidence:
        # Avoid duplicates by URL
        existing_urls = {e.url for e in evidence if e.url}
        for item in research_ctx.initial_evidence:
            if not item.url or item.url not in existing_urls:
                evidence.insert(0, item)

    # ── Warnings from AI ────────────────────────────────────────────────────
    ai_warnings = ai_json.get("warnings") or []
    if isinstance(ai_warnings, list):
        for w in ai_warnings:
            if isinstance(w, str) and w.strip():
                warnings.append(w.strip())

    return AIOptimizationResult(
        status=AI_STATUS_OK,
        provider=provider.provider_name,
        model=provider.model_name,
        summary=str(ai_json.get("summary", "")).strip(),
        recommended_settings=settings,
        changes=changes,
        estimated_fps=estimated_fps,
        fps_source=fps_source,
        confidence=confidence,
        reasoning=str(ai_json.get("reasoning", "")).strip(),
        evidence=evidence,
        warnings=warnings,
        fallback_result=fallback,
    )


def _validate_settings(
    raw: Dict,
    fallback,
    resolution: str,
    game: Dict,
    warnings: List[str],
) -> AIRecommendedSettings:
    """
    Validate AI-suggested settings against allowed values.

    Invalid values are replaced with the deterministic fallback value.
    Warnings are added for each replacement.
    """
    def pick(ai_val, allowed_set, fallback_val, field_name):
        if ai_val and str(ai_val).strip() in allowed_set:
            return str(ai_val).strip()
        if ai_val:
            warnings.append(
                f"AI suggested invalid {field_name} '{ai_val}' — using '{fallback_val}'."
            )
        return fallback_val

    # Resolution: prefer AI suggestion if valid, else user-requested resolution
    suggested_res = str(raw.get("resolution", "")).strip()
    if not suggested_res or suggested_res not in _VALID_RESOLUTIONS:
        # Also check against game-supported resolutions
        game_resolutions = set(game.get("resolutions", []))
        all_valid_res = _VALID_RESOLUTIONS | game_resolutions
        if suggested_res not in all_valid_res:
            if suggested_res:
                warnings.append(
                    f"AI suggested unsupported resolution '{suggested_res}' — using '{resolution}'."
                )
            suggested_res = resolution

    # Upscaling: check against game's supported upscaling
    game_upscaling = set(game.get("upscaling_support", []))
    all_valid_upscaling = _VALID_UPSCALING | game_upscaling
    suggested_upscaling = str(raw.get("upscaling", "Off")).strip()
    if suggested_upscaling not in all_valid_upscaling:
        if suggested_upscaling:
            warnings.append(
                f"AI suggested unsupported upscaling '{suggested_upscaling}' — using 'Off'."
            )
        suggested_upscaling = "Off"

    fb_preset = fallback.preset if fallback else "Medium"
    fb_vd = fallback.view_distance if fallback else "Medium"
    fb_shd = fallback.shadows if fallback else "Medium"
    fb_eff = fallback.effects if fallback else "Medium"
    fb_tex = fallback.textures if fallback else "Medium"
    fb_aa = fallback.anti_aliasing if fallback else "TAA"

    return AIRecommendedSettings(
        graphics_preset=pick(raw.get("graphics_preset"), _VALID_PRESETS, fb_preset, "graphics_preset"),
        resolution=suggested_res,
        upscaling=suggested_upscaling,
        view_distance=pick(raw.get("view_distance"), _VALID_VIEW_DISTANCE, fb_vd, "view_distance"),
        shadows=pick(raw.get("shadows"), _VALID_SHADOWS, fb_shd, "shadows"),
        effects=pick(raw.get("effects"), _VALID_EFFECTS, fb_eff, "effects"),
        textures=pick(raw.get("textures"), _VALID_TEXTURES, fb_tex, "textures"),
        anti_aliasing=pick(raw.get("anti_aliasing"), _VALID_AA, fb_aa, "anti_aliasing"),
        motion_blur=pick(raw.get("motion_blur"), _VALID_MOTION_BLUR, "Off", "motion_blur"),
        vsync=pick(raw.get("vsync"), _VALID_VSYNC, "Off", "vsync"),
        fps_limit=_safe_int(raw.get("fps_limit"), 0),
    )


def _safe_fps(ai_json: Dict, research_ctx) -> tuple[str, str]:
    """
    Return (estimated_fps_string, fps_source) — NEVER using invented numbers.

    Priority:
    1. Benchmark service result (authoritative).
    2. Deterministic FPS estimator.
    3. Empty string (AI cannot claim a number it didn't source).

    The AI's own ``estimated_fps`` field is checked only if the AI also
    provided a matching source that isn't FALLBACK_ESTIMATE or AI_INFERENCE.
    Even then, we only use it as a display hint, not as a numeric claim.
    """
    # Best source: deterministic (benchmark or math estimate)
    if research_ctx and research_ctx.deterministic_fps_meta:
        meta = research_ctx.deterministic_fps_meta
        display = deterministic_fps_to_display(meta)
        source = meta.get("source_type", EVIDENCE_FALLBACK_ESTIMATE)
        # Normalise source type to our constants
        if source == "benchmark":
            return display, EVIDENCE_MEASURED_BENCHMARK
        return display, EVIDENCE_FALLBACK_ESTIMATE

    # Second best: AI provided a sourced FPS claim we can use as display
    ai_fps = str(ai_json.get("estimated_fps", "")).strip()
    ai_fps_source = str(ai_json.get("fps_source", "")).strip()
    if ai_fps and ai_fps_source and ai_fps_source not in (
        EVIDENCE_FALLBACK_ESTIMATE, EVIDENCE_AI_INFERENCE, ""
    ):
        return ai_fps, ai_fps_source

    return "", EVIDENCE_FALLBACK_ESTIMATE


def _parse_evidence(raw_list: List) -> List[EvidenceItem]:
    """
    Parse the AI's evidence array into EvidenceItem objects.

    Filters out:
    - None input (treated as empty list)
    - entries that are not dicts
    - entries with no URL (we cannot verify them)
    - entries whose URL is None or the literal string "None"
    - entries with an evidence_type not in our allowed set
    """
    if not raw_list:
        return []
    items = []
    for entry in raw_list:
        if not isinstance(entry, dict):
            continue
        raw_url = entry.get("url")
        # Reject None, empty string, and the string "None"
        if not raw_url:
            continue
        url = str(raw_url).strip()
        if not url or url.lower() == "none":
            continue
        # Normalise evidence type
        raw_type = str(entry.get("type", EVIDENCE_AI_INFERENCE)).strip()
        evidence_type = raw_type if raw_type in _VALID_EVIDENCE_TYPES else EVIDENCE_AI_INFERENCE
        # Minimal domain extraction
        domain = str(entry.get("domain", "")).strip()
        if not domain:
            domain = url.split("//")[-1].split("/")[0].replace("www.", "")
        items.append(EvidenceItem(
            title=str(entry.get("title", "")).strip(),
            url=url,
            domain=domain,
            evidence_type=evidence_type,
            claim=str(entry.get("claim", "")).strip(),
        ))
    return items


def _clean_changes(raw: List) -> List[Dict[str, str]]:
    """Sanitise the changes list from AI output."""
    cleaned = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        setting = str(item.get("setting", "")).strip()
        if not setting:
            continue
        cleaned.append({
            "setting": setting,
            "from": str(item.get("from", "Unknown")).strip(),
            "to": str(item.get("to", "")).strip(),
            "reason": str(item.get("reason", "")).strip(),
        })
    return cleaned


def _clean_confidence(raw: Any) -> str:
    val = str(raw).lower().strip()
    if val in ("high", "medium", "low"):
        return val
    return "low"


def _safe_int(val: Any, default: int) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Deterministic fallback helpers
# ---------------------------------------------------------------------------

def _get_deterministic_result(hardware, game, target_fps, priority, resolution):
    """Run the deterministic optimizer and return the result (or None on error)."""
    try:
        from services.optimizer import generate_recommendations
        return generate_recommendations(
            hardware=hardware,
            game=game,
            target_fps=target_fps,
            priority=priority,
            resolution=resolution,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Deterministic optimizer failed: %s", exc)
        return None


def _wrap_deterministic(
    fallback,
    research_ctx,
    warning: str = "",
) -> AIOptimizationResult:
    """
    Wrap the deterministic result as an AIOptimizationResult with
    status=AI_STATUS_FALLBACK.
    """
    estimated_fps = ""
    fps_source = EVIDENCE_FALLBACK_ESTIMATE
    if research_ctx and research_ctx.deterministic_fps_meta:
        meta = research_ctx.deterministic_fps_meta
        estimated_fps = deterministic_fps_to_display(meta)
        src = meta.get("source_type", "")
        fps_source = (
            EVIDENCE_MEASURED_BENCHMARK if src == "benchmark"
            else EVIDENCE_FALLBACK_ESTIMATE
        )
    elif fallback:
        estimated_fps = fallback.estimated_fps
        fps_source = EVIDENCE_FALLBACK_ESTIMATE

    # Convert deterministic settings to AIRecommendedSettings
    rec_settings = None
    if fallback:
        rec_settings = AIRecommendedSettings(
            graphics_preset=fallback.preset,
            resolution=fallback.resolution,
            upscaling=fallback.upscaling_mode,
            view_distance=fallback.view_distance,
            shadows=fallback.shadows,
            effects=fallback.effects,
            textures=fallback.textures,
            anti_aliasing=fallback.anti_aliasing,
            motion_blur=fallback.motion_blur,
            vsync=fallback.vsync,
            fps_limit=fallback.fps_limit,
        )

    initial_evidence = (research_ctx.initial_evidence if research_ctx else [])
    warnings = [warning] if warning else []

    return AIOptimizationResult(
        status=AI_STATUS_FALLBACK,
        provider=AI_PROVIDER_NONE,
        model="",
        summary=(
            "Settings generated by the deterministic optimizer. "
            "AI assistance was unavailable."
        ),
        recommended_settings=rec_settings,
        changes=[],
        estimated_fps=estimated_fps,
        fps_source=fps_source,
        confidence="medium" if fps_source == EVIDENCE_MEASURED_BENCHMARK else "low",
        reasoning=(
            "The deterministic optimizer selected these settings based on your "
            "GPU tier and performance profile."
        ),
        evidence=initial_evidence,
        warnings=warnings,
        fallback_result=fallback,
    )
