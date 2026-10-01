from flask import Blueprint, render_template, abort, request, jsonify
from services.game_service import (
    get_all_games, get_game_by_slug,
    search_games, get_all_genres, get_games_for_api
)

games_bp = Blueprint('games', __name__)


@games_bp.route('/games')
def games_library():
    query = request.args.get('q', '').strip()
    genre_filter = request.args.get('genre', '').strip()
    performance = request.args.get('performance', '').strip()

    games = search_games(query) if query else get_all_games()

    if genre_filter:
        games = [g for g in games if any(genre_filter.lower() in gen.lower() for gen in g.get('genre', []))]

    genres = get_all_genres()
    return render_template('games.html',
                           games=games,
                           genres=genres,
                           query=query,
                           genre_filter=genre_filter)


@games_bp.route('/games/<slug>')
def game_detail(slug):
    game = get_game_by_slug(slug)
    if not game:
        abort(404)
    all_games = get_all_games()
    related = [g for g in all_games if g['slug'] != slug and any(
        genre in game.get('genre', []) for genre in g.get('genre', [])
    )][:3]
    return render_template('game_detail.html', game=game, related_games=related)


@games_bp.route('/api/search')
def api_search():
    query = request.args.get('q', '').strip()
    results = get_games_for_api(query)
    return jsonify(results[:8])
