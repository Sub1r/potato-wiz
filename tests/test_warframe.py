"""
tests/test_warframe.py
~~~~~~~~~~~~~~~~~~~~~~
Warframe integration tests.

Covers the four requirements that make Warframe a fully supported game:
  1. it appears in the catalog, search, and slug lookup,
  2. it renders its detail page without showing invented FPS data,
  3. the screenshot reader recognises Warframe's settings vocabulary,
  4. no fabricated benchmark or recommended-spec data is stored.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from services.game_service import (
    get_all_games,
    get_game_by_slug,
    get_popular_games,
    search_games,
)
from services.ai.vision_prompts import (
    build_game_context,
    canonical_key,
    validate_extraction,
)

WARFRAME = get_game_by_slug("warframe")


# ── Catalog presence ──────────────────────────────────────────────────────────

def test_warframe_is_in_catalog():
    assert WARFRAME is not None
    assert WARFRAME["name"] == "Warframe"
    assert WARFRAME["id"] == 11


def test_warframe_slug_is_unique():
    slugs = [game["slug"] for game in get_all_games()]
    assert slugs.count("warframe") == 1
    assert len(slugs) == len(set(slugs))


def test_warframe_id_is_unique():
    ids = [game["id"] for game in get_all_games()]
    assert ids.count(11) == 1


def test_warframe_core_metadata():
    assert WARFRAME["genre"]
    assert isinstance(WARFRAME["genre"], list)
    assert WARFRAME["release_year"] == 2013
    assert WARFRAME["developer"] == "Digital Extremes"
    assert WARFRAME["image"] == "warframe.jpg"
    assert WARFRAME["image_placeholder"]


def test_warframe_is_popular_but_not_featured():
    assert WARFRAME["popular"] is True
    assert WARFRAME["featured"] is False
    popular = get_popular_games()
    assert any(game["slug"] == "warframe" for game in popular)


def test_warframe_is_free_to_play():
    assert "free" in WARFRAME["game_type"].lower()


# ── Search / discovery ────────────────────────────────────────────────────────

def test_search_games_finds_warframe_by_name():
    results = search_games("warframe")
    assert any(game["slug"] == "warframe" for game in results)


def test_search_games_finds_warframe_case_insensitively():
    results = search_games("WARFRAME")
    assert any(game["slug"] == "warframe" for game in results)


def test_search_games_finds_warframe_by_genre():
    results = search_games("sci-fi")
    assert any(game["slug"] == "warframe" for game in results)


def test_search_by_developer_and_engine_do_not_match_warframe():
    # Guard against a loose search that would surface Warframe for unrelated
    # queries and pollute search results.
    assert not any(game["slug"] == "warframe" for game in search_games("witcher"))


# ── Detail page requirements ──────────────────────────────────────────────────

def test_warframe_minimum_specs_are_official_values():
    minimum = WARFRAME["minimum_specs"]
    assert minimum["os"] == "Windows 7 64-bit"
    assert "Intel Core i7-860" in minimum["cpu"]
    assert minimum["ram_gb"] == 4
    assert minimum["vram_gb"] == 2
    assert minimum["storage_gb"] == 75


def test_warframe_minimum_specs_have_no_zero_placeholders():
    for field in ("ram_gb", "vram_gb", "storage_gb"):
        assert WARFRAME["minimum_specs"][field] > 0


def test_warframe_recommended_specs_are_empty_and_explained():
    recommended = WARFRAME["recommended_specs"]
    assert recommended["cpu"] == ""
    assert recommended["gpu"] == ""
    assert recommended["ram_gb"] == 0
    assert recommended["vram_gb"] == 0
    assert "minimum" in WARFRAME["recommended_specs_notes"].lower()


def test_warframe_lists_official_sources():
    sources = WARFRAME["sources"]
    assert sources, "Warframe must cite its official sources"
    for source in sources:
        assert source["type"] == "OFFICIAL_INFORMATION"
        assert source["title"]
        assert source["url"].startswith("https://")


def test_warframe_lists_official_minimum_specs_source():
    urls = " ".join(source["url"] for source in WARFRAME["sources"])
    assert "support.warframe.com" in urls


def test_warframe_graphics_api_metadata():
    apis = WARFRAME["graphics_api"]
    assert "DirectX 11" in apis
    assert "DirectX 12" in apis
    assert WARFRAME["graphics_api_notes"]


def test_warframe_upscaling_is_capability_metadata_only():
    support = WARFRAME["upscaling_support"]
    assert "FSR" in support
    assert "Dynamic Resolution" in support
    notes = WARFRAME["upscaling_support_notes"].lower()
    assert "not a claim" in notes


def test_warframe_resolutions_include_1080p():
    assert "1920x1080" in WARFRAME["resolutions"]


def test_warframe_settings_detail_is_populated():
    detail = WARFRAME["settings_detail"]
    assert len(detail) >= 10
    for entry in detail:
        assert entry["name"]
        assert entry["value"]
        assert entry["why"]


# ── No fabricated performance data ────────────────────────────────────────────

def test_warframe_has_no_fps_profiles():
    assert WARFRAME["fps_profiles"] == {}
    assert WARFRAME["fps_profiles_notes"]


def test_warframe_has_no_estimated_fps():
    estimated = WARFRAME["recommended_settings"]["estimated_fps"]
    assert estimated == ""


def test_warframe_baseline_settings_have_no_fps_language():
    recommended = WARFRAME["recommended_settings"]
    assert recommended["status"] == "AI Research"
    notes = WARFRAME["recommended_settings_notes"].lower()
    assert "not measured" in notes
    assert recommended["fps_limit"] == 0
    for value in recommended.values():
        if isinstance(value, str):
            assert "fps" not in value.lower()


def test_warframe_settings_detail_has_no_measured_fps_claims():
    number_then_fps = re.compile(r"\d+\s*(fps|frames per second)")
    for entry in WARFRAME["settings_detail"]:
        blob = f"{entry['name']} {entry['value']} {entry['why']}"
        assert not number_then_fps.search(blob.lower())
        assert "benchmark" not in blob.lower()


# ── Screenshot reader vocabulary ──────────────────────────────────────────────

@pytest.mark.parametrize("label,expected", [
    ("Graphics Preset", "graphics_preset"),
    ("Graphics Engine", "graphics_engine"),
    ("Graphics API", "graphics_api"),
    ("Upscaling", "upscaling"),
    ("Dynamic Resolution", "upscaling"),
    ("SSAO Quality", "ssao_quality"),
    ("Sun Shadows", "shadows"),
    ("Shadow Quality", "shadows"),
    ("Enhanced Decals", "enhanced_decals"),
    ("GPU Particles Quality", "gpu_particles"),
    ("Texture Quality", "textures"),
    ("Anti-Aliasing", "anti_aliasing"),
    ("Motion Blur", "motion_blur"),
    ("V-Sync", "vsync"),
])
def test_warframe_labels_map_to_canonical_settings(label, expected):
    assert canonical_key(label) == expected


def test_build_game_context_lists_warframe_labels():
    context = build_game_context(WARFRAME)
    assert "Warframe" in context
    assert "Graphics Engine" in context
    assert "Enhanced Decals" in context


def test_build_game_context_without_game():
    assert build_game_context(None) == "No game is selected."


def test_vision_extraction_keeps_warframe_values_as_read():
    parsed = {
        "settings": [
            {"name": "Graphics Engine", "value": "Enhanced",
             "confidence": "high"},
            {"name": "SSAO Quality", "value": "Low", "confidence": "high"},
            {"name": "Resolution", "value": "1920 x 1080", "confidence": "high"},
            {"name": "Graphics Preset", "value": "Medium", "confidence": "high"},
        ],
        "unreadable": [],
        "notes": [],
    }
    result = validate_extraction(parsed, WARFRAME)
    by_name = {entry["name"]: entry for entry in result["settings"]}

    assert by_name["Graphics Engine"]["value"] == "Enhanced"
    assert by_name["Graphics Engine"]["canonical_name"] == "graphics_engine"
    assert by_name["Graphics Engine"]["status"] == "recognized"
    assert by_name["SSAO Quality"]["value"] == "Low"
    assert by_name["Resolution"]["value"] == "1920x1080"
    assert by_name["Graphics Preset"]["value"] == "Medium"
    assert result["recognized_count"] == 4
    assert result["requires_confirmation"] is True
    assert result["source"] == "USER_SCREENSHOT"


def test_vision_extraction_never_invents_missing_values():
    parsed = {
        "settings": [
            {"name": "Graphics Engine", "value": None, "confidence": "high"},
        ],
        "unreadable": ["Graphics Engine"],
        "notes": [],
    }
    result = validate_extraction(parsed, WARFRAME)
    entry = result["settings"][0]
    assert entry["status"] == "unreadable"
    assert entry["value"] is None
    assert entry["note"]
    assert result["current_settings_text"] == ""


def test_vision_extraction_flags_unlisted_warframe_setting():
    parsed = {
        "settings": [
            {"name": "Vegetation Density", "value": "Low", "confidence": "medium"},
        ],
        "unreadable": [],
        "notes": [],
    }
    result = validate_extraction(parsed, WARFRAME)
    entry = result["settings"][0]
    assert entry["status"] == "unrecognized"
    assert entry["value"] == "Low"
    assert entry["name"] == "Vegetation Density"


# ── Routes ────────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    from app import create_app

    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def test_detail_page_renders_without_fake_fps(client):
    response = client.get("/games/warframe")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Warframe" in body
    assert "Digital Extremes" in body
    assert "~" not in body.split("Estimated FPS")[-1][:200]


def test_detail_page_shows_fps_placeholder_for_warframe(client):
    body = client.get("/games/warframe").get_data(as_text=True)
    assert "No stored benchmark for this game" in body
    assert "Expected FPS by GPU Tier" not in body


def test_detail_page_notes_missing_recommended_specs(client):
    body = client.get("/games/warframe").get_data(as_text=True)
    assert "Not published" in body
    assert "Windows 7 64-bit" in body


def test_games_index_lists_warframe(client):
    body = client.get("/games").get_data(as_text=True)
    assert "/games/warframe" in body


def test_search_api_finds_warframe(client):
    response = client.get("/api/search?q=warframe")
    assert response.status_code == 200
    payload = response.get_json()
    slugs = [
        item.get("slug") if isinstance(item, dict) else item
        for item in payload
    ]
    assert "warframe" in slugs


def test_optimizer_page_offers_warframe(client):
    body = client.get("/optimizer").get_data(as_text=True)
    assert 'value="warframe"' in body


if __name__ == "__main__":
    pytest.main([__file__, "-v"])