"""
benchmark_service.py

Loads and queries benchmark data from data/benchmarks/<game_slug>.json.

Design principles:
- Never crashes the caller; all errors are caught and logged internally.
- Returns BenchmarkMatch(found=False) whenever no suitable benchmark exists.
- Matching is deterministic: exact > gpu_match > approximate > none.
- GPU name normalisation strips common vendor prefixes so that
  "NVIDIA GeForce GTX 1650" and "GTX 1650" are treated as the same card,
  but different GPU models are NEVER confused for one another.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Any

from models.models import BenchmarkMatch, MATCH_TYPE_EXACT, MATCH_TYPE_GPU, MATCH_TYPE_APPROXIMATE, MATCH_TYPE_NONE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_BENCHMARKS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "benchmarks",
)

# Simple in-process cache: {game_slug: list-of-records | None}
_CACHE: Dict[str, Optional[List[Dict[str, Any]]]] = {}

# ---------------------------------------------------------------------------
# GPU name normalisation
# ---------------------------------------------------------------------------

# Vendor prefix tokens that add no information for matching purposes.
_VENDOR_PREFIXES = re.compile(
    r"^(nvidia\s+)?(geforce\s+)?|^(amd\s+)?(radeon\s+)?",
    re.IGNORECASE,
)

# Common contractions we want to expand for consistent matching
_EXPANSIONS = {
    r"\bgtx\b": "gtx",
    r"\brtx\b": "rtx",
    r"\brx\b": "rx",
}


def _normalise_gpu(name: str) -> str:
    """
    Return a compact, lower-cased GPU identifier stripped of vendor strings.

    Examples
    --------
    "NVIDIA GeForce GTX 1650"  →  "gtx 1650"
    "GTX 1650"                 →  "gtx 1650"
    "AMD Radeon RX 6700 XT"    →  "rx 6700 xt"
    "RX 6700 XT"               →  "rx 6700 xt"
    """
    s = name.strip().lower()
    # Strip leading vendor prefixes
    s = re.sub(r"^(nvidia\s+)?(geforce\s+)?", "", s)
    s = re.sub(r"^(amd\s+)?(radeon\s+)?", "", s)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalise_cpu(name: str) -> str:
    """
    Return a compact, lower-cased CPU identifier.

    Examples
    --------
    "AMD Ryzen 5 3600"   →  "ryzen 5 3600"
    "Intel Core i5-9600K"→  "i5-9600k"
    """
    s = name.strip().lower()
    s = re.sub(r"^(amd\s+)?", "", s)
    s = re.sub(r"^(intel\s+)?(core\s+)?", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _gpus_match(name_a: str, name_b: str) -> bool:
    """
    Return True only when both names refer to the same GPU model.

    We normalise both sides and require an exact string match after
    normalisation.  This prevents "GTX 1650" matching "GTX 1660".
    """
    return _normalise_gpu(name_a) == _normalise_gpu(name_b)


def _cpus_match(name_a: str, name_b: str) -> bool:
    """Return True when both CPU names normalise to the same identifier."""
    return _normalise_cpu(name_a) == _normalise_cpu(name_b)


def _resolutions_match(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()


def _presets_match(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()


def _upscaling_match(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()


# ---------------------------------------------------------------------------
# Record validation
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = {"gpu", "resolution", "preset", "avg_fps"}


def _is_valid_record(record: Any) -> bool:
    """
    Return True when a benchmark record has the minimum required fields,
    sensible values, and is NOT a test fixture.

    Fixture records (``"_fixture": true``) are explicitly excluded so they
    can never appear as production benchmark matches regardless of how well
    their GPU/CPU/resolution/preset values match the caller's hardware.
    """
    if not isinstance(record, dict):
        return False
    # Exclude synthetic test fixtures — they must never reach find_benchmark().
    if record.get("_fixture") is True:
        logger.debug("Skipping fixture benchmark record: %s", record.get("gpu", "<unknown>"))
        return False
    for field in _REQUIRED_FIELDS:
        if field not in record:
            logger.debug("Benchmark record missing required field '%s': %s", field, record)
            return False
    try:
        fps = int(record["avg_fps"])
        if fps <= 0:
            logger.debug("Benchmark record has non-positive avg_fps: %s", record)
            return False
    except (TypeError, ValueError):
        logger.debug("Benchmark record has non-numeric avg_fps: %s", record)
        return False
    return True


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_benchmarks(game_slug: str) -> List[Dict[str, Any]]:
    """
    Load and return all valid benchmark records for *game_slug*.

    Returns an empty list when:
    - the file does not exist
    - the file is malformed JSON
    - no valid records are found

    The result is cached for the lifetime of the process.
    """
    if game_slug in _CACHE:
        cached = _CACHE[game_slug]
        return cached if cached is not None else []

    path = os.path.join(_BENCHMARKS_DIR, f"{game_slug}.json")

    if not os.path.isfile(path):
        logger.debug("No benchmark file for game '%s' at %s", game_slug, path)
        _CACHE[game_slug] = []
        return []

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        logger.warning("Malformed benchmark JSON for '%s': %s", game_slug, exc)
        _CACHE[game_slug] = []
        return []
    except OSError as exc:
        logger.warning("Cannot read benchmark file for '%s': %s", game_slug, exc)
        _CACHE[game_slug] = []
        return []

    if not isinstance(data, dict):
        logger.warning("Benchmark file for '%s' is not a JSON object", game_slug)
        _CACHE[game_slug] = []
        return []

    raw_records: List[Any] = data.get("benchmarks", [])
    if not isinstance(raw_records, list):
        logger.warning("'benchmarks' key for '%s' is not a list", game_slug)
        _CACHE[game_slug] = []
        return []

    valid = [r for r in raw_records if _is_valid_record(r)]
    logger.debug(
        "Loaded %d/%d valid benchmark records for '%s'",
        len(valid), len(raw_records), game_slug,
    )
    _CACHE[game_slug] = valid
    return valid


def clear_cache() -> None:
    """Clear the in-process benchmark cache (useful in tests)."""
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def find_benchmark(
    game_slug: str,
    gpu: str,
    resolution: str,
    preset: str,
    cpu: str = "",
    upscaling: str = "Off",
) -> BenchmarkMatch:
    """
    Search benchmark records for *game_slug* and return the best match.

    Match priority (highest → lowest):
    1. Exact: GPU + CPU + resolution + preset all match.
    2. gpu_match: GPU + resolution + preset match (CPU absent or different).
    3. approximate: GPU matches, resolution or preset differ.
    4. none: No record found.

    Parameters
    ----------
    game_slug:  Game identifier (e.g. "palworld").
    gpu:        Hardware GPU name (e.g. "NVIDIA GeForce GTX 1650").
    resolution: Target resolution string (e.g. "1920x1080").
    preset:     Graphics preset string (e.g. "High").
    cpu:        Hardware CPU name (optional; used to upgrade match to exact).
    upscaling:  Upscaling mode (informational; not used in match ranking yet).

    Returns
    -------
    BenchmarkMatch — always returns an instance, never raises.
    """
    try:
        return _find_benchmark_internal(game_slug, gpu, resolution, preset, cpu, upscaling)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Unexpected error in find_benchmark for '%s': %s", game_slug, exc)
        return BenchmarkMatch(
            found=False,
            match_type=MATCH_TYPE_NONE,
            reason=f"Internal error during benchmark lookup: {exc}",
        )


def _find_benchmark_internal(
    game_slug: str,
    gpu: str,
    resolution: str,
    preset: str,
    cpu: str,
    upscaling: str,
) -> BenchmarkMatch:
    records = load_benchmarks(game_slug)

    if not records:
        logger.debug(
            "Benchmark lookup: %s | GPU=%s | no records available → fallback",
            game_slug, gpu,
        )
        return BenchmarkMatch(
            found=False,
            match_type=MATCH_TYPE_NONE,
            reason="No benchmark records available for this game.",
        )

    # Collect candidates by match tier
    exact_candidates: List[Dict] = []
    gpu_candidates: List[Dict] = []
    approx_candidates: List[Dict] = []

    for rec in records:
        rec_gpu = rec.get("gpu", "")
        rec_cpu = rec.get("cpu", "")
        rec_res = rec.get("resolution", "")
        rec_preset = rec.get("preset", "")

        gpu_ok = _gpus_match(gpu, rec_gpu)
        if not gpu_ok:
            continue  # GPU mismatch — not useful at all

        res_ok = _resolutions_match(resolution, rec_res)
        preset_ok = _presets_match(preset, rec_preset)
        cpu_ok = bool(cpu) and bool(rec_cpu) and _cpus_match(cpu, rec_cpu)

        if gpu_ok and res_ok and preset_ok and cpu_ok:
            exact_candidates.append(rec)
        elif gpu_ok and res_ok and preset_ok:
            gpu_candidates.append(rec)
        else:
            # GPU matches but other criteria differ
            approx_candidates.append(rec)

    def _make_match(record: Dict, match_type: str, reason: str) -> BenchmarkMatch:
        fps_low_raw = record.get("one_percent_low")
        return BenchmarkMatch(
            found=True,
            match_type=match_type,
            avg_fps=int(record["avg_fps"]),
            one_percent_low=int(fps_low_raw) if fps_low_raw is not None else None,
            source=record.get("source", ""),
            source_url=record.get("source_url", ""),
            confidence=record.get("confidence", ""),
            record=record,
            reason=reason,
        )

    if exact_candidates:
        rec = exact_candidates[0]
        logger.debug(
            "Benchmark lookup: %s | GPU=%s CPU=%s res=%s preset=%s → EXACT match | source=%s",
            game_slug, gpu, cpu, resolution, preset, rec.get("source", ""),
        )
        return _make_match(
            rec,
            MATCH_TYPE_EXACT,
            f"Exact match: GPU + CPU + {resolution} + {preset}.",
        )

    if gpu_candidates:
        rec = gpu_candidates[0]
        logger.debug(
            "Benchmark lookup: %s | GPU=%s res=%s preset=%s → GPU match | source=%s",
            game_slug, gpu, resolution, preset, rec.get("source", ""),
        )
        return _make_match(
            rec,
            MATCH_TYPE_GPU,
            f"GPU match ({_normalise_gpu(gpu)}) at {resolution} / {preset}. CPU differed.",
        )

    if approx_candidates:
        rec = approx_candidates[0]
        logger.debug(
            "Benchmark lookup: %s | GPU=%s → approximate match | source=%s",
            game_slug, gpu, rec.get("source", ""),
        )
        return _make_match(
            rec,
            MATCH_TYPE_APPROXIMATE,
            f"Approximate match: GPU ({_normalise_gpu(gpu)}) matched but resolution/preset differ.",
        )

    logger.debug(
        "Benchmark lookup: %s | GPU=%s → no match found → fallback",
        game_slug, gpu,
    )
    return BenchmarkMatch(
        found=False,
        match_type=MATCH_TYPE_NONE,
        reason=f"No benchmark record found for GPU '{_normalise_gpu(gpu)}' in game '{game_slug}'.",
    )
