from flask import Blueprint, render_template
from services.game_service import get_popular_games, get_all_games

home_bp = Blueprint('home', __name__)


@home_bp.route('/')
def index():
    popular_games = get_popular_games()
    featured_game = next((g for g in popular_games if g.get('featured')), popular_games[0] if popular_games else None)
    return render_template('home.html',
                           popular_games=popular_games,
                           featured_game=featured_game)
