"""
game_service.py
Loads and queries the games.json data file.
"""

import json
import os
from typing import List, Optional, Dict, Any


_GAMES_CACHE: Optional[List[Dict]] = None


def _load_games() -> List[Dict]:
    global _GAMES_CACHE
    if _GAMES_CACHE is not None:
        return _GAMES_CACHE

    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'games.json'
    )
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            _GAMES_CACHE = json.load(f)
    except Exception as e:
        _GAMES_CACHE = []
    return _GAMES_CACHE


def get_all_games() -> List[Dict]:
    return _load_games()


def get_popular_games() -> List[Dict]:
    return [g for g in _load_games() if g.get('popular')]


def get_game_by_slug(slug: str) -> Optional[Dict]:
    for game in _load_games():
        if game.get('slug') == slug:
            return game
    return None


def get_game_by_id(game_id: int) -> Optional[Dict]:
    for game in _load_games():
        if game.get('id') == game_id:
            return game
    return None


def search_games(query: str) -> List[Dict]:
    """Search games by name, genre, or description."""
    query_lower = query.lower().strip()
    if not query_lower:
        return _load_games()

    results = []
    for game in _load_games():
        name = game.get('name', '').lower()
        genres = ' '.join(game.get('genre', [])).lower()
        desc = game.get('description', '').lower()
        dev = game.get('developer', '').lower()

        if (query_lower in name or
                query_lower in genres or
                query_lower in desc or
                query_lower in dev):
            results.append(game)
    return results


def get_games_by_genre(genre: str) -> List[Dict]:
    genre_lower = genre.lower()
    return [g for g in _load_games()
            if any(genre_lower in gen.lower() for gen in g.get('genre', []))]


def get_all_genres() -> List[str]:
    genres = set()
    for game in _load_games():
        for genre in game.get('genre', []):
            genres.add(genre)
    return sorted(genres)


def get_games_for_api(query: str = '') -> List[Dict]:
    """Return lightweight game list for search API."""
    games = search_games(query) if query else _load_games()
    return [
        {
            'id': g.get('id'),
            'name': g.get('name'),
            'slug': g.get('slug'),
            'genre': g.get('genre', []),
            'image': g.get('image'),
            'image_placeholder': g.get('image_placeholder', '#1a3a5c'),
        }
        for g in games
    ]
