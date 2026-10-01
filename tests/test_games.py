"""
test_games.py
Tests for game_service.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from services.game_service import (
    get_all_games, get_popular_games, get_game_by_slug,
    search_games, get_all_genres
)


def test_get_all_games_returns_list():
    games = get_all_games()
    assert isinstance(games, list)
    assert len(games) > 0


def test_all_games_have_required_fields():
    for game in get_all_games():
        assert 'id' in game
        assert 'name' in game
        assert 'slug' in game
        assert 'genre' in game
        assert 'recommended_settings' in game


def test_get_popular_games():
    popular = get_popular_games()
    assert isinstance(popular, list)
    assert all(g.get('popular') for g in popular)


def test_get_game_by_slug_exists():
    game = get_game_by_slug('palworld')
    assert game is not None
    assert game['name'] == 'Palworld'


def test_get_game_by_slug_not_found():
    game = get_game_by_slug('nonexistent-game-xyz')
    assert game is None


def test_search_games_by_name():
    results = search_games('elden')
    assert any('Elden Ring' in g['name'] for g in results)


def test_search_games_empty_returns_all():
    all_games = get_all_games()
    results = search_games('')
    assert len(results) == len(all_games)


def test_get_all_genres_returns_list():
    genres = get_all_genres()
    assert isinstance(genres, list)
    assert len(genres) > 0
    assert all(isinstance(g, str) for g in genres)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
