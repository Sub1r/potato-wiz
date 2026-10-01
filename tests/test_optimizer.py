"""
test_optimizer.py
Tests for the recommendation engine.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models.models import HardwareSpecs, OptimizationResult
from services.optimizer import generate_recommendations
from services.game_service import get_game_by_slug


def make_hw(gpu='NVIDIA GeForce GTX 1650', ram=16, vram=4):
    return HardwareSpecs(gpu=gpu, ram_gb=ram, vram_gb=vram)


def test_generate_recommendations_returns_result():
    game = get_game_by_slug('palworld')
    assert game is not None
    hw = make_hw()
    result = generate_recommendations(hw, game)
    assert isinstance(result, OptimizationResult)


def test_result_has_required_fields():
    game = get_game_by_slug('gta-v')
    hw = make_hw('NVIDIA GeForce RTX 3070')
    result = generate_recommendations(hw, game)
    assert result.preset
    assert result.resolution
    assert result.estimated_fps
    assert result.status in ('Excellent', 'Very Good', 'Playable', 'Low', 'Unplayable')


def test_high_end_gpu_gets_higher_preset():
    game = get_game_by_slug('elden-ring')
    low_hw  = make_hw('NVIDIA GeForce GTX 1050')
    high_hw = make_hw('NVIDIA GeForce RTX 4090')

    low_result  = generate_recommendations(low_hw,  game)
    high_result = generate_recommendations(high_hw, game)

    preset_order = ['Low', 'Medium', 'High', 'Very High', 'Ultra', 'Maximum', 'Extreme']
    low_idx  = next((i for i, p in enumerate(preset_order) if p in low_result.preset), 0)
    high_idx = next((i for i, p in enumerate(preset_order) if p in high_result.preset), 0)

    assert high_idx >= low_idx, "High-end GPU should get equal or higher preset"


def test_fps_priority_vs_quality():
    game = get_game_by_slug('cyberpunk-2077')
    hw = make_hw('NVIDIA GeForce RTX 2060')

    fps_result     = generate_recommendations(hw, game, priority='fps')
    quality_result = generate_recommendations(hw, game, priority='quality')

    preset_order = ['Low', 'Medium', 'High', 'Very High', 'Ultra']
    fps_idx     = next((i for i, p in enumerate(preset_order) if p in fps_result.preset), 0)
    qual_idx    = next((i for i, p in enumerate(preset_order) if p in quality_result.preset), 0)

    assert fps_idx <= qual_idx, "FPS priority should get equal or lower preset than quality"


def test_result_to_dict():
    game = get_game_by_slug('palworld')
    hw = make_hw()
    result = generate_recommendations(hw, game)
    d = result.to_dict()
    assert isinstance(d, dict)
    assert 'preset' in d
    assert 'estimated_fps' in d
    assert 'settings_list' in d


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
