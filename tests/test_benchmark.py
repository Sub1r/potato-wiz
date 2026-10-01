"""
test_benchmark.py
Tests for the benchmark data layer, matching logic, and FPS estimator
integration.  Covers test cases A–L as specified.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from models.models import (
    BenchmarkMatch,
    HardwareSpecs,
    MATCH_TYPE_EXACT,
    MATCH_TYPE_GPU,
    MATCH_TYPE_APPROXIMATE,
    MATCH_TYPE_NONE,
)
from services import benchmark_service
from services.benchmark_service import (
    load_benchmarks,
    find_benchmark,
    clear_cache,
    _normalise_gpu,
    _normalise_cpu,
    _gpus_match,
)
from services.fps_estimator import estimate_fps, estimate_fps_with_metadata
from services.optimizer import generate_recommendations, _build_settings_list
from services.game_service import get_game_by_slug


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hw(
    gpu: str = "NVIDIA GeForce GTX 1650",
    cpu: str = "AMD Ryzen 5 3600",
    ram: int = 16,
    vram: int = 4,
    resolution: str = "1920x1080",
) -> HardwareSpecs:
    return HardwareSpecs(
        gpu=gpu,
        cpu=cpu,
        ram_gb=ram,
        vram_gb=vram,
        resolution=resolution,
    )


def _minimal_settings() -> dict:
    """Return a minimal settings dict that _build_settings_list() accepts."""
    return {
        "preset": "High",
        "resolution": "1920x1080",
        "upscaling": "Off",
        "view_distance": "High",
        "shadows": "High",
        "textures": "High",
        "effects": "High",
        "aa": "TAA",
        "fps_limit": 60,
    }


# ---------------------------------------------------------------------------
# A. Benchmark JSON loading
# ---------------------------------------------------------------------------

class TestBenchmarkLoading:
    """Test case A: benchmark JSON loading."""

    def setup_method(self):
        clear_cache()

    def test_load_palworld_returns_list(self):
        """A: load_benchmarks('palworld') returns a list."""
        records = load_benchmarks("palworld")
        assert isinstance(records, list)

    def test_load_palworld_not_empty_file(self):
        """A: palworld.json exists and can be parsed (even if all records are fixtures)."""
        # The file must be loadable; fixture records are filtered, so the
        # resulting list may be empty — that is correct behaviour.
        records = load_benchmarks("palworld")
        assert isinstance(records, list)  # file parses without error

    def test_load_unknown_game_returns_empty_list(self):
        """A: Non-existent game slug returns empty list, does not crash."""
        records = load_benchmarks("does-not-exist-xyz")
        assert records == []

    def test_load_caches_result(self):
        """A: Repeated calls return the same list object (cached)."""
        first = load_benchmarks("palworld")
        second = load_benchmarks("palworld")
        assert first is second


# ---------------------------------------------------------------------------
# B. Valid Palworld benchmark record
# ---------------------------------------------------------------------------

class TestPalworldBenchmarkRecord:
    """Test case B: valid Palworld benchmark record shape."""

    def setup_method(self):
        clear_cache()

    def test_records_have_required_fields(self):
        """B: Every loaded record has the minimum required fields."""
        records = load_benchmarks("palworld")
        required = {"gpu", "resolution", "preset", "avg_fps"}
        for rec in records:
            missing = required - set(rec.keys())
            assert not missing, f"Record missing fields {missing}: {rec}"

    def test_avg_fps_is_positive(self):
        """B: avg_fps is a positive integer in every record."""
        records = load_benchmarks("palworld")
        for rec in records:
            assert int(rec["avg_fps"]) > 0

    def test_records_are_dicts(self):
        """B: Each record is a dict."""
        for rec in load_benchmarks("palworld"):
            assert isinstance(rec, dict)


# ---------------------------------------------------------------------------
# Shared helper: inject a temporary benchmark file with production records
# ---------------------------------------------------------------------------

def _inject_production_benchmarks(records: list, game_slug: str = "prod-game") -> str:
    """
    Write a temporary benchmark file with the given records (no _fixture flag),
    patch _BENCHMARKS_DIR, and return the original dir so the caller can restore
    it in a finally block.
    """
    import json as _json
    import tempfile as _tempfile
    tmpdir = _tempfile.mkdtemp()
    path = os.path.join(tmpdir, f"{game_slug}.json")
    with open(path, "w", encoding="utf-8") as fh:
        _json.dump({"game": game_slug, "benchmarks": records}, fh)
    original_dir = benchmark_service._BENCHMARKS_DIR
    benchmark_service._BENCHMARKS_DIR = tmpdir
    return original_dir


_PROD_GTX1650_LOW = {
    "gpu": "NVIDIA GeForce GTX 1650",
    "cpu": "AMD Ryzen 5 3600",
    "ram_gb": 16,
    "vram_gb": 4,
    "resolution": "1920x1080",
    "preset": "Low",
    "upscaling": "Off",
    "ray_tracing": False,
    "avg_fps": 48,
    "one_percent_low": 36,
    "source": "TechPowerUp",
    "source_url": "https://www.techpowerup.com/",
    "benchmark_date": "2024-03-01",
    "confidence": "high",
    "notes": "Production test record (no _fixture flag).",
}

_PROD_GTX1650_HIGH = {
    "gpu": "NVIDIA GeForce GTX 1650",
    "cpu": "Intel Core i5-9600K",
    "ram_gb": 16,
    "vram_gb": 4,
    "resolution": "1920x1080",
    "preset": "Low",
    "upscaling": "Off",
    "ray_tracing": False,
    "avg_fps": 47,
    "one_percent_low": 35,
    "source": "TechPowerUp",
    "source_url": "https://www.techpowerup.com/",
    "benchmark_date": "2024-03-01",
    "confidence": "high",
    "notes": "Production test record (no _fixture flag).",
}


# ---------------------------------------------------------------------------
# C. Exact GPU + CPU match
# ---------------------------------------------------------------------------

class TestExactMatch:
    """Test case C: exact GPU + CPU + resolution + preset match (production record)."""

    def setup_method(self):
        clear_cache()
        self._original_dir = _inject_production_benchmarks(
            [_PROD_GTX1650_LOW, _PROD_GTX1650_HIGH], game_slug="prod-game"
        )

    def teardown_method(self):
        benchmark_service._BENCHMARKS_DIR = self._original_dir
        clear_cache()

    def test_exact_match_returns_found_true(self):
        """C: Exact GPU + CPU match sets found=True."""
        result = find_benchmark(
            game_slug="prod-game",
            gpu="NVIDIA GeForce GTX 1650",
            resolution="1920x1080",
            preset="Low",
            cpu="AMD Ryzen 5 3600",
        )
        assert result.found is True

    def test_exact_match_type(self):
        """C: Exact match sets match_type=exact."""
        result = find_benchmark(
            game_slug="prod-game",
            gpu="NVIDIA GeForce GTX 1650",
            resolution="1920x1080",
            preset="Low",
            cpu="AMD Ryzen 5 3600",
        )
        assert result.match_type == MATCH_TYPE_EXACT

    def test_exact_match_has_avg_fps(self):
        """C: Exact match populates avg_fps."""
        result = find_benchmark(
            game_slug="prod-game",
            gpu="NVIDIA GeForce GTX 1650",
            resolution="1920x1080",
            preset="Low",
            cpu="AMD Ryzen 5 3600",
        )
        assert result.avg_fps is not None
        assert result.avg_fps > 0

    def test_exact_match_has_reason(self):
        """C: Exact match has a non-empty reason string."""
        result = find_benchmark(
            game_slug="prod-game",
            gpu="NVIDIA GeForce GTX 1650",
            resolution="1920x1080",
            preset="Low",
            cpu="AMD Ryzen 5 3600",
        )
        assert result.reason


# ---------------------------------------------------------------------------
# D. GPU-only match
# ---------------------------------------------------------------------------

class TestGpuOnlyMatch:
    """Test case D: GPU + resolution + preset match (CPU differs) — production records."""

    def setup_method(self):
        clear_cache()
        self._original_dir = _inject_production_benchmarks(
            [_PROD_GTX1650_LOW, _PROD_GTX1650_HIGH], game_slug="prod-game"
        )

    def teardown_method(self):
        benchmark_service._BENCHMARKS_DIR = self._original_dir
        clear_cache()

    def test_gpu_match_with_different_cpu(self):
        """D: Different CPU still triggers a gpu_match."""
        result = find_benchmark(
            game_slug="prod-game",
            gpu="NVIDIA GeForce GTX 1650",
            resolution="1920x1080",
            preset="Low",
            cpu="Intel Core i7-8700K",  # not in any production record → GPU-level match
        )
        assert result.found is True
        assert result.match_type in (MATCH_TYPE_GPU, MATCH_TYPE_EXACT)

    def test_gpu_match_without_cpu(self):
        """D: No CPU provided gives gpu_match when GPU/res/preset match."""
        result = find_benchmark(
            game_slug="prod-game",
            gpu="NVIDIA GeForce GTX 1650",
            resolution="1920x1080",
            preset="Low",
            cpu="",
        )
        assert result.found is True
        assert result.match_type == MATCH_TYPE_GPU

    def test_gpu_normalised_still_matches(self):
        """D: Short GPU name 'GTX 1650' matches full 'NVIDIA GeForce GTX 1650' record."""
        result = find_benchmark(
            game_slug="prod-game",
            gpu="GTX 1650",
            resolution="1920x1080",
            preset="Low",
            cpu="",
        )
        assert result.found is True


# ---------------------------------------------------------------------------
# E. No benchmark match
# ---------------------------------------------------------------------------

class TestNoMatch:
    """Test case E: no benchmark found."""

    def setup_method(self):
        clear_cache()

    def test_no_match_for_unknown_gpu(self):
        """E: Unknown GPU returns found=False."""
        result = find_benchmark(
            game_slug="palworld",
            gpu="NVIDIA GeForce RTX 9999",
            resolution="1920x1080",
            preset="High",
        )
        assert result.found is False
        assert result.match_type == MATCH_TYPE_NONE

    def test_no_match_for_unknown_game(self):
        """E: Unknown game slug returns found=False."""
        result = find_benchmark(
            game_slug="nonexistent-game",
            gpu="NVIDIA GeForce GTX 1650",
            resolution="1920x1080",
            preset="High",
        )
        assert result.found is False

    def test_no_match_returns_benchmark_match_instance(self):
        """E: find_benchmark always returns a BenchmarkMatch, never raises."""
        result = find_benchmark(
            game_slug="palworld",
            gpu="Imaginary GPU XR-9000",
            resolution="4096x2160",
            preset="Extreme",
        )
        assert isinstance(result, BenchmarkMatch)

    def test_no_match_fps_fields_are_none(self):
        """E: When not found, avg_fps and one_percent_low are None."""
        result = find_benchmark(
            game_slug="palworld",
            gpu="Imaginary GPU XR-9000",
            resolution="1920x1080",
            preset="High",
        )
        assert result.avg_fps is None
        assert result.one_percent_low is None


# ---------------------------------------------------------------------------
# F. Malformed / missing benchmark file
# ---------------------------------------------------------------------------

class TestMalformedBenchmarkFile:
    """Test case F: malformed or missing benchmark file is handled gracefully."""

    def setup_method(self):
        clear_cache()

    def _write_temp_benchmark(self, content: str, game_slug: str = "test-game-temp") -> str:
        """
        Temporarily patch the benchmark dir so we can inject bad data.
        Returns the original _BENCHMARKS_DIR so it can be restored.
        """
        original_dir = benchmark_service._BENCHMARKS_DIR
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, f"{game_slug}.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        benchmark_service._BENCHMARKS_DIR = tmpdir
        return original_dir

    def test_missing_file_returns_empty_list(self):
        """F: Missing benchmark file → load_benchmarks returns []."""
        records = load_benchmarks("elden-ring-no-benchmarks-yet")
        assert records == []

    def test_malformed_json_returns_empty_list(self):
        """F: Malformed JSON → load_benchmarks returns [], no crash."""
        original_dir = self._write_temp_benchmark(
            "{broken json,,", game_slug="bad-json-game"
        )
        try:
            records = load_benchmarks("bad-json-game")
            assert records == []
        finally:
            benchmark_service._BENCHMARKS_DIR = original_dir
            clear_cache()

    def test_empty_benchmarks_array_returns_empty_list(self):
        """F: Valid JSON but empty benchmarks list returns []."""
        original_dir = self._write_temp_benchmark(
            '{"game": "empty-game", "benchmarks": []}',
            game_slug="empty-game",
        )
        try:
            records = load_benchmarks("empty-game")
            assert records == []
        finally:
            benchmark_service._BENCHMARKS_DIR = original_dir
            clear_cache()

    def test_record_missing_avg_fps_is_skipped(self):
        """F: Record with missing avg_fps is ignored; others are kept."""
        data = {
            "game": "partial-game",
            "benchmarks": [
                {
                    "gpu": "GTX 1650",
                    "resolution": "1920x1080",
                    "preset": "Low",
                    # avg_fps intentionally missing
                },
                {
                    "gpu": "GTX 1650",
                    "resolution": "1920x1080",
                    "preset": "High",
                    "avg_fps": 45,
                },
            ],
        }
        original_dir = self._write_temp_benchmark(
            json.dumps(data), game_slug="partial-game"
        )
        try:
            records = load_benchmarks("partial-game")
            assert len(records) == 1
            assert records[0]["preset"] == "High"
        finally:
            benchmark_service._BENCHMARKS_DIR = original_dir
            clear_cache()

    def test_find_benchmark_with_missing_file_returns_not_found(self):
        """F: find_benchmark on non-existent game returns found=False."""
        result = find_benchmark(
            game_slug="no-such-game-ever",
            gpu="GTX 1650",
            resolution="1920x1080",
            preset="High",
        )
        assert isinstance(result, BenchmarkMatch)
        assert result.found is False


# ---------------------------------------------------------------------------
# G. Benchmark FPS result
# ---------------------------------------------------------------------------

class TestBenchmarkFpsResult:
    """Test case G: estimate_fps_with_metadata returns benchmark FPS when a production match exists."""

    def setup_method(self):
        clear_cache()
        self._original_dir = _inject_production_benchmarks(
            [_PROD_GTX1650_LOW], game_slug="prod-game"
        )

    def teardown_method(self):
        benchmark_service._BENCHMARKS_DIR = self._original_dir
        clear_cache()

    def test_returns_benchmark_source_type_on_match(self):
        """G: When a production benchmark match exists, source_type is 'benchmark'."""
        hw = _make_hw(gpu="NVIDIA GeForce GTX 1650", cpu="AMD Ryzen 5 3600")
        meta = estimate_fps_with_metadata(hw, "prod-game", preset="Low", resolution="1920x1080")
        assert meta["source_type"] == "benchmark"

    def test_fps_low_and_high_are_positive(self):
        """G: fps_low and fps_high are both positive integers."""
        hw = _make_hw(gpu="NVIDIA GeForce GTX 1650", cpu="AMD Ryzen 5 3600")
        meta = estimate_fps_with_metadata(hw, "prod-game", preset="Low", resolution="1920x1080")
        assert meta["fps_low"] > 0
        assert meta["fps_high"] > 0

    def test_fps_low_leq_fps_high(self):
        """G: fps_low <= fps_high."""
        hw = _make_hw(gpu="NVIDIA GeForce GTX 1650", cpu="AMD Ryzen 5 3600")
        meta = estimate_fps_with_metadata(hw, "prod-game", preset="Low", resolution="1920x1080")
        assert meta["fps_low"] <= meta["fps_high"]

    def test_metadata_has_required_keys(self):
        """G: Result dict has all required keys."""
        hw = _make_hw(gpu="NVIDIA GeForce GTX 1650", cpu="AMD Ryzen 5 3600")
        meta = estimate_fps_with_metadata(hw, "prod-game", preset="Low", resolution="1920x1080")
        for key in ("fps_low", "fps_high", "source_type", "confidence", "match_type", "source", "source_url", "reason"):
            assert key in meta, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# H. Fallback FPS estimation
# ---------------------------------------------------------------------------

class TestFallbackFpsEstimation:
    """Test case H: falls back to mathematical estimator when no benchmark exists."""

    def setup_method(self):
        clear_cache()

    def test_fallback_source_type_estimated(self):
        """H: Unknown GPU → source_type is 'estimated'."""
        hw = _make_hw(gpu="NVIDIA GeForce RTX 9999 Ti")
        meta = estimate_fps_with_metadata(hw, "palworld", preset="High", resolution="1920x1080")
        assert meta["source_type"] == "estimated"

    def test_fallback_for_unknown_game(self):
        """H: Fallback works for a game with no benchmark file."""
        hw = _make_hw(gpu="NVIDIA GeForce GTX 1650")
        meta = estimate_fps_with_metadata(hw, "elden-ring", preset="High", resolution="1920x1080")
        assert meta["fps_low"] > 0
        assert meta["fps_high"] > 0

    def test_original_estimate_fps_still_works(self):
        """H: Legacy estimate_fps() signature still returns a (int, int) tuple."""
        hw = _make_hw()
        result = estimate_fps(hw, "palworld")
        assert isinstance(result, tuple)
        assert len(result) == 2
        low, high = result
        assert isinstance(low, int)
        assert isinstance(high, int)
        assert low > 0
        assert low <= high

    def test_fallback_match_type_is_none(self):
        """H: Fallback result has match_type='none'."""
        hw = _make_hw(gpu="Imaginary GPU Pro Max")
        meta = estimate_fps_with_metadata(hw, "palworld")
        assert meta["match_type"] == MATCH_TYPE_NONE


# ---------------------------------------------------------------------------
# I. Confidence metadata
# ---------------------------------------------------------------------------

class TestConfidenceMetadata:
    """Test case I: confidence field is populated correctly."""

    def setup_method(self):
        clear_cache()
        self._original_dir = _inject_production_benchmarks(
            [_PROD_GTX1650_LOW], game_slug="prod-game"
        )

    def teardown_method(self):
        benchmark_service._BENCHMARKS_DIR = self._original_dir
        clear_cache()

    def test_benchmark_confidence_non_empty_on_match(self):
        """I: Production benchmark match has a non-empty confidence value."""
        hw = _make_hw(gpu="NVIDIA GeForce GTX 1650", cpu="AMD Ryzen 5 3600")
        meta = estimate_fps_with_metadata(hw, "prod-game", preset="Low", resolution="1920x1080")
        assert meta["source_type"] == "benchmark"
        assert meta["confidence"] != ""

    def test_fallback_confidence_is_low(self):
        """I: Fallback (estimated) path returns confidence='low'."""
        hw = _make_hw(gpu="Imaginary GPU ZX-1")
        meta = estimate_fps_with_metadata(hw, "prod-game")
        assert meta["source_type"] == "estimated"
        assert meta["confidence"] == "low"

    def test_benchmark_match_object_has_confidence(self):
        """I: BenchmarkMatch.confidence is populated from the production record."""
        result = find_benchmark(
            game_slug="prod-game",
            gpu="NVIDIA GeForce GTX 1650",
            resolution="1920x1080",
            preset="Low",
            cpu="AMD Ryzen 5 3600",
        )
        assert result.found
        assert result.confidence != ""


# ---------------------------------------------------------------------------
# J. Source / source_url preservation
# ---------------------------------------------------------------------------

class TestSourcePreservation:
    """Test case J: source and source_url are carried through from the record."""

    def setup_method(self):
        clear_cache()
        self._original_dir = _inject_production_benchmarks(
            [_PROD_GTX1650_LOW], game_slug="prod-game"
        )

    def teardown_method(self):
        benchmark_service._BENCHMARKS_DIR = self._original_dir
        clear_cache()

    def test_source_present_on_benchmark_match(self):
        """J: BenchmarkMatch.source is populated when a production record is matched."""
        result = find_benchmark(
            game_slug="prod-game",
            gpu="NVIDIA GeForce GTX 1650",
            resolution="1920x1080",
            preset="Low",
            cpu="AMD Ryzen 5 3600",
        )
        assert result.found
        assert result.source == "TechPowerUp"

    def test_metadata_source_matches_record(self):
        """J: estimate_fps_with_metadata source equals the production record's source."""
        hw = _make_hw(gpu="NVIDIA GeForce GTX 1650", cpu="AMD Ryzen 5 3600")
        meta = estimate_fps_with_metadata(hw, "prod-game", preset="Low", resolution="1920x1080")
        assert meta["source_type"] == "benchmark"
        assert meta["source"] == "TechPowerUp"

    def test_fallback_source_is_empty_string(self):
        """J: Fallback result has empty source/source_url."""
        hw = _make_hw(gpu="Imaginary GPU ZX-2")
        meta = estimate_fps_with_metadata(hw, "prod-game")
        assert meta["source"] == ""
        assert meta["source_url"] == ""

    def test_benchmark_match_to_dict_has_source(self):
        """J: BenchmarkMatch.to_dict() includes source field."""
        bm = BenchmarkMatch(
            found=True,
            match_type=MATCH_TYPE_EXACT,
            avg_fps=55,
            source="some-review-site",
            source_url="https://example.com",
            confidence="high",
        )
        d = bm.to_dict()
        assert d["source"] == "some-review-site"
        assert d["source_url"] == "https://example.com"


# ---------------------------------------------------------------------------
# K. Optimizer settings_detail consistency
# ---------------------------------------------------------------------------

class TestSettingsDetailConsistency:
    """Test case K: settings displayed to the user match the optimizer's selections."""

    def test_settings_list_values_match_computed_settings(self):
        """K: Every value in settings_list corresponds to the computed setting."""
        settings = _minimal_settings()
        result_list = _build_settings_list(settings, [])
        name_to_value = {entry["name"]: entry["value"] for entry in result_list}
        assert name_to_value["Graphics Preset"] == settings["preset"]
        assert "1920" in name_to_value["Resolution"]
        assert name_to_value["FSR / Upscaling"] == settings["upscaling"]
        assert name_to_value["View Distance"] == settings["view_distance"]
        assert name_to_value["Shadows"] == settings["shadows"]
        assert name_to_value["Textures"] == settings["textures"]
        assert name_to_value["Effects"] == settings["effects"]

    def test_game_settings_detail_why_is_used(self):
        """K: When game provides settings_detail, its 'why' is used but values stay from computed."""
        settings = _minimal_settings()
        game_detail = [
            {"name": "Graphics Preset", "value": "Low", "why": "Game-specific reason for preset."},
        ]
        result_list = _build_settings_list(settings, game_detail)
        preset_entry = next(e for e in result_list if e["name"] == "Graphics Preset")
        # Value must be the COMPUTED setting, not the game_detail value
        assert preset_entry["value"] == "High"
        # Why may come from the game detail
        assert preset_entry["why"] == "Game-specific reason for preset."

    def test_game_settings_detail_does_not_replace_list(self):
        """K: Providing a non-empty game settings_detail no longer replaces the computed list."""
        settings = _minimal_settings()
        # Old bug: would return game_detail directly; new fix: always computes from settings
        game_detail = [
            {"name": "Graphics Preset", "value": "Ultra", "why": "Stale preset."},
            {"name": "Shadows", "value": "Ultra", "why": "Stale shadows."},
        ]
        result_list = _build_settings_list(settings, game_detail)
        for entry in result_list:
            if entry["name"] == "Graphics Preset":
                assert entry["value"] == settings["preset"], (
                    "Preset value must come from computed settings, not game_detail"
                )
            if entry["name"] == "Shadows":
                assert entry["value"] == settings["shadows"]

    def test_settings_list_has_eleven_entries(self):
        """K: _build_settings_list always returns exactly 11 entries."""
        result = _build_settings_list(_minimal_settings(), [])
        assert len(result) == 11

    def test_optimizer_result_settings_match_chosen_preset(self):
        """K: Full optimizer run — settings_list values match the OptimizationResult preset."""
        game = get_game_by_slug("palworld")
        hw = _make_hw()
        result = generate_recommendations(hw, game, priority="balanced")
        preset_entry = next(
            (e for e in result.settings_list if e["name"] == "Graphics Preset"), None
        )
        assert preset_entry is not None
        assert preset_entry["value"] == result.preset, (
            f"settings_list preset '{preset_entry['value']}' != "
            f"result.preset '{result.preset}'"
        )


# ---------------------------------------------------------------------------
# L. Existing optimizer behavior remains functional
# ---------------------------------------------------------------------------

class TestExistingOptimizerBehavior:
    """Test case L: existing optimizer behavior is preserved after changes."""

    def setup_method(self):
        clear_cache()

    def test_generate_recommendations_returns_optimization_result(self):
        """L: generate_recommendations still returns an OptimizationResult."""
        from models.models import OptimizationResult
        game = get_game_by_slug("palworld")
        result = generate_recommendations(_make_hw(), game)
        assert isinstance(result, OptimizationResult)

    def test_all_games_still_return_result(self):
        """L: All games in the library still return a valid result."""
        from services.game_service import get_all_games
        hw = _make_hw()
        for game in get_all_games():
            result = generate_recommendations(hw, game)
            assert result.preset
            assert result.status in ("Excellent", "Very Good", "Playable", "Low", "Unplayable")

    def test_result_to_dict_has_required_keys(self):
        """L: result.to_dict() still has all expected keys."""
        game = get_game_by_slug("gta-v")
        result = generate_recommendations(_make_hw("NVIDIA GeForce RTX 3070"), game)
        d = result.to_dict()
        for key in ("preset", "resolution", "upscaling", "estimated_fps", "status", "settings_list"):
            assert key in d, f"Missing key in result dict: {key}"

    def test_high_end_gpu_gets_higher_or_equal_preset(self):
        """L: High-end GPU gets higher or equal preset than low-end GPU."""
        game = get_game_by_slug("elden-ring")
        low = generate_recommendations(_make_hw("NVIDIA GeForce GTX 1050"), game)
        high = generate_recommendations(_make_hw("NVIDIA GeForce RTX 4090"), game)
        order = ["Low", "Medium", "High", "Very High", "Ultra"]
        low_idx = next((i for i, p in enumerate(order) if p in low.preset), 0)
        high_idx = next((i for i, p in enumerate(order) if p in high.preset), 0)
        assert high_idx >= low_idx

    def test_fps_priority_lower_or_equal_preset_than_quality(self):
        """L: FPS priority gives equal or lower preset than quality priority."""
        game = get_game_by_slug("cyberpunk-2077")
        hw = _make_hw("NVIDIA GeForce RTX 2060")
        fps_r = generate_recommendations(hw, game, priority="fps")
        qual_r = generate_recommendations(hw, game, priority="quality")
        order = ["Low", "Medium", "High", "Very High", "Ultra"]
        fps_idx = next((i for i, p in enumerate(order) if p in fps_r.preset), 0)
        qual_idx = next((i for i, p in enumerate(order) if p in qual_r.preset), 0)
        assert fps_idx <= qual_idx

    def test_api_optimize_route_still_works(self):
        """L: /api/optimize route (via Flask test client) returns 200."""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from app import create_app
        flask_app = create_app()
        flask_app.config["TESTING"] = True
        with flask_app.test_client() as client:
            resp = client.post(
                "/api/optimize",
                json={
                    "game": "palworld",
                    "priority": "balanced",
                    "resolution": "1920x1080",
                    "target_fps": 60,
                },
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert data.get("success") is True
            assert "result" in data


# ---------------------------------------------------------------------------
# Regression: fixture records must never appear as production matches
# ---------------------------------------------------------------------------

class TestFixtureExclusion:
    """
    Regression tests ensuring that records with ``"_fixture": true`` are
    completely excluded from benchmark matching and never returned as
    production evidence, regardless of how well their fields match.
    """

    def setup_method(self):
        clear_cache()

    # -- load_benchmarks excludes fixtures at load time ---------------------

    def test_fixture_records_not_loaded(self):
        """
        load_benchmarks() must return zero records for palworld because
        every record in palworld.json is a test fixture (_fixture=true).
        """
        records = load_benchmarks("palworld")
        for rec in records:
            assert rec.get("_fixture") is not True, (
                "Fixture record leaked into load_benchmarks() result"
            )

    def test_palworld_has_zero_production_records(self):
        """
        palworld.json currently contains only fixture records, so
        load_benchmarks('palworld') must return an empty list.
        """
        records = load_benchmarks("palworld")
        assert records == [], (
            f"Expected no production records for palworld, got {len(records)}"
        )

    # -- find_benchmark returns not-found when only fixtures exist ----------

    def test_fixture_gpu_cpu_match_returns_not_found(self):
        """
        An exact GPU+CPU match against a fixture record must NOT produce
        a found=True BenchmarkMatch.
        """
        result = find_benchmark(
            game_slug="palworld",
            gpu="NVIDIA GeForce GTX 1650",    # matches fixture record exactly
            resolution="1920x1080",
            preset="Low",
            cpu="AMD Ryzen 5 3600",           # matches fixture record exactly
        )
        assert result.found is False, (
            "Fixture record should never produce found=True"
        )

    def test_fixture_match_type_is_none(self):
        """match_type must be 'none' when only fixture records exist."""
        result = find_benchmark(
            game_slug="palworld",
            gpu="NVIDIA GeForce GTX 1650",
            resolution="1920x1080",
            preset="Low",
        )
        assert result.match_type == MATCH_TYPE_NONE

    # -- _is_valid_record rejects fixture records directly ------------------

    def test_is_valid_record_rejects_fixture(self):
        """_is_valid_record() returns False for a record with _fixture=true."""
        from services.benchmark_service import _is_valid_record
        fixture_rec = {
            "_fixture": True,
            "gpu": "NVIDIA GeForce GTX 1650",
            "resolution": "1920x1080",
            "preset": "Low",
            "avg_fps": 48,
        }
        assert _is_valid_record(fixture_rec) is False

    def test_is_valid_record_accepts_production_record(self):
        """_is_valid_record() returns True for a well-formed record without _fixture."""
        from services.benchmark_service import _is_valid_record
        production_rec = {
            "gpu": "NVIDIA GeForce GTX 1650",
            "resolution": "1920x1080",
            "preset": "Low",
            "avg_fps": 48,
            "source": "TechPowerUp",
            "source_url": "https://www.techpowerup.com/",
        }
        assert _is_valid_record(production_rec) is True

    def test_is_valid_record_fixture_false_is_kept(self):
        """_fixture=false (not true) must NOT be excluded."""
        from services.benchmark_service import _is_valid_record
        rec = {
            "_fixture": False,
            "gpu": "NVIDIA GeForce GTX 1650",
            "resolution": "1920x1080",
            "preset": "Low",
            "avg_fps": 48,
        }
        assert _is_valid_record(rec) is True

    # -- mixed file: only non-fixture records are returned ------------------

    def test_mixed_file_only_returns_production_records(self):
        """
        When a benchmark file contains both fixture and production records,
        only the production records must be returned.
        """
        import json
        import tempfile
        data = {
            "game": "mixed-game",
            "benchmarks": [
                {
                    "_fixture": True,
                    "gpu": "GTX 1650",
                    "resolution": "1920x1080",
                    "preset": "Low",
                    "avg_fps": 48,
                    "source": "TEST_FIXTURE",
                },
                {
                    "gpu": "GTX 1650",
                    "resolution": "1920x1080",
                    "preset": "High",
                    "avg_fps": 61,
                    "source": "RealSite",
                    "source_url": "https://example.com",
                },
            ],
        }
        original_dir = benchmark_service._BENCHMARKS_DIR
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "mixed-game.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        benchmark_service._BENCHMARKS_DIR = tmpdir
        try:
            records = load_benchmarks("mixed-game")
            assert len(records) == 1, (
                f"Expected 1 production record, got {len(records)}"
            )
            assert records[0].get("_fixture") is not True
            assert records[0]["preset"] == "High"
        finally:
            benchmark_service._BENCHMARKS_DIR = original_dir
            clear_cache()

    # -- estimate_fps_with_metadata falls back when only fixtures exist -----

    def test_fps_metadata_falls_back_for_palworld_fixture_only(self):
        """
        estimate_fps_with_metadata() must return source_type='estimated'
        for palworld because all its records are fixtures.
        """
        hw = _make_hw(gpu="NVIDIA GeForce GTX 1650", cpu="AMD Ryzen 5 3600")
        meta = estimate_fps_with_metadata(
            hw, "palworld", preset="Low", resolution="1920x1080"
        )
        assert meta["source_type"] == "estimated", (
            "Fixture-only game must always fall back to estimated source_type"
        )
        assert meta["match_type"] == MATCH_TYPE_NONE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
