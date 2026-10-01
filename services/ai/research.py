"""
services/ai/research.py
~~~~~~~~~~~~~~~~~~~~~~~
Evidence collection helpers for the AI optimizer.

``ResearchContext`` collects benchmark evidence already stored in Potato Wiz
and formats it for inclusion in the AI prompt so the model can treat it as
authoritative.

Live web research (search + fetch) is performed by the AI provider itself
(specifically OpenRouter's server-side tools).  This module handles only the
pre-research phase of assembling locally available evidence.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from models.models import (
    EvidenceItem,
    EVIDENCE_MEASURED_BENCHMARK,
    EVIDENCE_FALLBACK_ESTIMATE,
    BenchmarkMatch,
    MATCH_TYPE_EXACT,
    MATCH_TYPE_GPU,
    MATCH_TYPE_APPROXIMATE,
    MATCH_TYPE_NONE,
)

logger = logging.getLogger(__name__)


@dataclass
class ResearchContext:
    """
    All locally-available evidence assembled before the AI call.

    Fields
    ------
    benchmark_match:
        The best benchmark record found by benchmark_service for this
        game/hardware combination.  None if no file or no match.
    deterministic_fps_meta:
        The dict returned by estimate_fps_with_metadata() —
        contains fps_low, fps_high, source_type, confidence, etc.
    initial_evidence:
        EvidenceItem list seeded from stored benchmark data.
        The AI may add more items from web research.
    benchmark_context_text:
        Pre-formatted text block for inclusion in the user prompt.
    """
    benchmark_match: Optional[BenchmarkMatch] = None
    deterministic_fps_meta: Dict[str, Any] = field(default_factory=dict)
    initial_evidence: List[EvidenceItem] = field(default_factory=list)
    benchmark_context_text: str = ""


def build_research_context(
    hardware: Any,
    game: Dict[str, Any],
    preset: str,
    resolution: str,
    upscaling_mode: str,
) -> ResearchContext:
    """
    Assemble all locally available evidence for the given hardware+game combo.

    This runs BEFORE the AI call so the model receives authoritative data
    upfront rather than having to infer it from web results alone.

    Parameters
    ----------
    hardware:   HardwareSpecs instance.
    game:       Game dict from games.json.
    preset:     Recommended preset from the deterministic optimizer.
    resolution: Target resolution string.
    upscaling_mode: Upscaling setting string.

    Returns
    -------
    ResearchContext
    """
    game_slug = game.get("slug", "")
    ctx = ResearchContext()

    # ── Step 1: Benchmark lookup ─────────────────────────────────────────────
    try:
        from services.benchmark_service import find_benchmark
        ctx.benchmark_match = find_benchmark(
            game_slug=game_slug,
            gpu=hardware.gpu,
            resolution=resolution,
            preset=preset,
            cpu=hardware.cpu,
            upscaling=upscaling_mode,
        )
        logger.debug(
            "Research context: %s benchmark for %s — match=%s",
            game_slug,
            hardware.gpu,
            ctx.benchmark_match.match_type if ctx.benchmark_match else "none",
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Benchmark lookup failed during research: %s", exc)

    # ── Step 2: Deterministic FPS estimate ───────────────────────────────────
    try:
        from services.fps_estimator import estimate_fps_with_metadata
        ctx.deterministic_fps_meta = estimate_fps_with_metadata(
            hardware=hardware,
            game_slug=game_slug,
            preset=preset,
            resolution=resolution,
            upscaling_mode=upscaling_mode,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("FPS estimator failed during research: %s", exc)
        ctx.deterministic_fps_meta = {}

    # ── Step 3: Seed evidence list from benchmark data ───────────────────────
    if ctx.benchmark_match and ctx.benchmark_match.found:
        match = ctx.benchmark_match
        evidence_type = (
            EVIDENCE_MEASURED_BENCHMARK
            if match.match_type in (MATCH_TYPE_EXACT, MATCH_TYPE_GPU)
            else EVIDENCE_MEASURED_BENCHMARK  # approximate still a real record
        )
        item = EvidenceItem(
            title=f"Potato Wiz Benchmark: {game.get('name', game_slug)}",
            url=match.source_url or "",
            domain=_extract_domain(match.source_url or ""),
            evidence_type=evidence_type,
            claim=(
                f"{hardware.gpu} achieved avg {match.avg_fps} FPS "
                f"at {resolution} / {preset} preset"
                + (
                    f" (1% low: {match.one_percent_low} FPS)"
                    if match.one_percent_low else ""
                )
                + f". Match type: {match.match_type}."
            ),
        )
        ctx.initial_evidence.append(item)
        logger.debug("Seeded benchmark evidence: %s", item.claim)

    # ── Step 4: Build benchmark context text for the prompt ──────────────────
    from services.ai.prompts import build_benchmark_context
    ctx.benchmark_context_text = build_benchmark_context(ctx.benchmark_match)

    return ctx


def _extract_domain(url: str) -> str:
    """Return the domain from a URL string, or '' if unparseable."""
    if not url:
        return ""
    try:
        # Minimal domain extraction without importing urllib for speed
        stripped = url.split("//")[-1]
        return stripped.split("/")[0].replace("www.", "")
    except Exception:  # pylint: disable=broad-except
        return ""


def deterministic_fps_to_display(fps_meta: Dict[str, Any]) -> str:
    """
    Convert the fps_meta dict from estimate_fps_with_metadata() into a
    display string such as "42–52".
    """
    low = fps_meta.get("fps_low")
    high = fps_meta.get("fps_high")
    if low is not None and high is not None:
        return f"{low}–{high}"
    return ""
