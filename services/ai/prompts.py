"""
services/ai/prompts.py
~~~~~~~~~~~~~~~~~~~~~~
Prompt construction for the Potato Wiz AI optimizer.

All prompt text lives here so it can be reviewed, tested, and updated
independently from the provider or optimizer logic.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from models.models import (
    EVIDENCE_MEASURED_BENCHMARK,
    EVIDENCE_PUBLISHED_BENCHMARK,
    EVIDENCE_COMMUNITY_REPORT,
    EVIDENCE_OFFICIAL_INFO,
    EVIDENCE_GUIDE,
    EVIDENCE_AI_INFERENCE,
    EVIDENCE_FALLBACK_ESTIMATE,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are Potato Wiz's optimization research assistant.
Your job is to produce a single, structured JSON recommendation for the user's PC game graphics settings.

## ABSOLUTE RULES — follow these without exception:

1. NEVER invent benchmark FPS numbers.
   - Use only numbers sourced from search results, web pages, or the benchmark data provided.
   - If you have no measured source for a specific FPS number, do NOT include a numeric FPS in `estimated_fps`.
   - Instead set `fps_source` to "FALLBACK_ESTIMATE" and leave `estimated_fps` blank; the application will fill it from the deterministic estimator.

2. NEVER fabricate URLs or source citations.
   - Only include URLs that you actually retrieved from a search tool or web fetch.
   - If you cannot verify a URL exists, omit the evidence entry entirely.

3. NEVER claim a community report or your own inference is a measured benchmark.
   - Use the evidence type field exactly:
     * "MEASURED_BENCHMARK"   — an actual hardware test with logged FPS numbers from a publication.
     * "PUBLISHED_BENCHMARK"  — a review or article with stated FPS results.
     * "COMMUNITY_REPORT"     — forum/Reddit/Discord posts from users.
     * "OFFICIAL_INFORMATION" — developer patch notes, official system requirements.
     * "GUIDE"                — optimization guide or tutorial.
     * "AI_INFERENCE"         — your own reasoning from the available evidence.
     * "FALLBACK_ESTIMATE"    — mathematical estimate only (no real source).

4. Hardware data provided in the user message is AUTHORITATIVE.
   Do not contradict or re-derive it.

5. Game data and available settings provided in the user message are AUTHORITATIVE.
   Only recommend settings that exist in the game.

6. Benchmark data provided in the user message is AUTHORITATIVE.
   It supersedes any conflicting web search result.

7. If evidence conflicts across sources, acknowledge it explicitly in `reasoning`.

8. If evidence is insufficient, say so — do not guess silently.

9. Do not blindly reduce every setting.
   Preserve settings that have low performance cost.
   Explain every change you recommend.

10. Your entire response MUST be valid JSON matching the schema below.
    Do not include any text outside the JSON object.

## Required JSON schema:

{
  "summary": "<one or two sentences explaining what you recommend and why>",
  "recommended_settings": {
    "graphics_preset": "<Low|Normal|Medium|High|Very High|Ultra|Maximum|Extreme>",
    "resolution": "<e.g. 1920x1080>",
    "upscaling": "<Off|FSR Quality|FSR Balanced|FSR Performance|FSR Ultra Perf|DLSS Quality|DLSS Balanced|Quality|Balanced|Performance|Ultra Performance>",
    "view_distance": "<Low|Medium|High|Ultra>",
    "shadows": "<Off|Low|Medium|High|Ultra>",
    "effects": "<Low|Medium|High|Ultra>",
    "textures": "<Low|Medium|High|Ultra>",
    "anti_aliasing": "<Off|TAA|FXAA|MSAA>",
    "motion_blur": "<Off|Low|Medium|High>",
    "vsync": "<Off|On>",
    "fps_limit": <integer, 0 = unlimited>
  },
  "changes": [
    {
      "setting": "<setting name>",
      "from": "<current value or Unknown>",
      "to": "<recommended value>",
      "reason": "<brief explanation>"
    }
  ],
  "estimated_fps": "<leave blank if unsourced, e.g. '52-60' or ''>",
  "fps_source": "<one of the evidence type strings>",
  "confidence": "<high|medium|low>",
  "reasoning": "<full explanation of your analysis and reasoning>",
  "evidence": [
    {
      "title": "<page title>",
      "url": "<exact URL from search results>",
      "domain": "<domain.com>",
      "type": "<evidence type string>",
      "claim": "<specific claim from this source>"
    }
  ],
  "warnings": ["<any important warnings for the user>"]
}
"""

# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(
    hardware: Any,
    game: Dict[str, Any],
    target_fps: int,
    priority: str,
    resolution: str,
    benchmark_context: str,
    current_settings_text: str = "",
) -> str:
    """
    Build the user-turn prompt with all authoritative context.

    Parameters
    ----------
    hardware:
        HardwareSpecs instance.
    game:
        Game dict from games.json.
    target_fps:
        User's target FPS.
    priority:
        "fps" | "balanced" | "quality".
    resolution:
        Target resolution string (e.g. "1920x1080").
    benchmark_context:
        Pre-formatted string from ``build_benchmark_context()``.
    current_settings_text:
        Optional free-text describing current in-game settings.
    """
    game_name = game.get("name", "Unknown Game")
    game_slug = game.get("slug", "")
    engine = game.get("engine", "Unknown Engine")
    release_year = game.get("release_year", "")

    # Supported resolutions and upscaling from game data
    supported_resolutions = game.get("resolutions", [])
    supported_upscaling = game.get("upscaling_support", [])

    parts: List[str] = []

    parts.append("## OPTIMIZATION REQUEST\n")

    # ── Hardware ─────────────────────────────────────────────────────────────
    parts.append("### Hardware (authoritative — do not contradict)")
    parts.append(f"- GPU: {hardware.gpu}")
    parts.append(f"- CPU: {hardware.cpu}")
    parts.append(f"- RAM: {hardware.ram_gb} GB")
    parts.append(f"- VRAM: {hardware.vram_gb} GB")
    parts.append(f"- OS: {hardware.os_name}")
    parts.append(f"- CPU Cores: {hardware.cpu_cores}")
    parts.append(f"- CPU Frequency: {hardware.cpu_freq_ghz} GHz")
    parts.append("")

    # ── Game ─────────────────────────────────────────────────────────────────
    parts.append("### Game (authoritative)")
    parts.append(f"- Name: {game_name}")
    parts.append(f"- Engine: {engine}")
    if release_year:
        parts.append(f"- Release Year: {release_year}")
    if supported_resolutions:
        parts.append(f"- Supported Resolutions: {', '.join(supported_resolutions)}")
    if supported_upscaling:
        parts.append(f"- Supported Upscaling: {', '.join(supported_upscaling)}")
    parts.append("")

    # ── Target ───────────────────────────────────────────────────────────────
    parts.append("### Optimization Target")
    parts.append(f"- Target FPS: {target_fps}")
    parts.append(f"- Target Resolution: {resolution}")
    parts.append(f"- Priority: {priority.upper()} (fps = max framerate, balanced = fps+quality, quality = best visuals)")
    parts.append("")

    # ── Benchmark context ────────────────────────────────────────────────────
    if benchmark_context.strip():
        parts.append("### Potato Wiz Benchmark Data (authoritative — supersedes web search)")
        parts.append(benchmark_context)
        parts.append("")

    # ── Current settings ─────────────────────────────────────────────────────
    if current_settings_text.strip():
        parts.append("### User's Current In-Game Settings")
        parts.append(current_settings_text.strip())
        parts.append("")

    # ── Instruction ──────────────────────────────────────────────────────────
    parts.append("### Your Task")
    parts.append(
        f"Research {game_name} performance on {hardware.gpu} at {resolution}. "
        f"Use search tools to find benchmark results, optimization guides, and community reports. "
        f"Then produce the JSON recommendation following the schema in your system instructions."
    )
    parts.append(
        "Remember: only include FPS numbers you can attribute to a real source. "
        "Set fps_source to FALLBACK_ESTIMATE and leave estimated_fps blank if no sourced number is available."
    )

    return "\n".join(parts)


def build_benchmark_context(benchmark_match: Any) -> str:
    """
    Convert a BenchmarkMatch into a text block for inclusion in the prompt.

    Clearly labels the data as authoritative Potato Wiz benchmark evidence.
    """
    if not benchmark_match or not benchmark_match.found:
        return ""

    lines = [
        f"- Match type: {benchmark_match.match_type}",
        f"- Average FPS: {benchmark_match.avg_fps}",
    ]
    if benchmark_match.one_percent_low is not None:
        lines.append(f"- 1% Low FPS: {benchmark_match.one_percent_low}")
    if benchmark_match.source:
        lines.append(f"- Source: {benchmark_match.source}")
    if benchmark_match.source_url:
        lines.append(f"- Source URL: {benchmark_match.source_url}")
    if benchmark_match.confidence:
        lines.append(f"- Confidence: {benchmark_match.confidence}")

    return "\n".join(lines)


def build_research_queries(
    game_name: str,
    gpu: str,
    cpu: str,
    resolution: str,
    target_fps: int,
) -> List[str]:
    """
    Return a prioritised list of research queries for web search.

    Multiple targeted queries are better than one broad query.
    Queries are ordered from most specific to most general.
    """
    queries = [
        f"{game_name} {gpu} {resolution} benchmark FPS",
        f"{game_name} {gpu} graphics settings optimization",
        f"{game_name} performance optimization guide",
        f"{game_name} low FPS fix settings",
        f"{game_name} {gpu} reddit performance",
    ]
    # Add CPU query only when GPU is not dominant factor (lower-end setups)
    if cpu and cpu.lower() not in ("unknown cpu", ""):
        queries.insert(2, f"{game_name} {cpu} {gpu} performance benchmark")
    return queries
