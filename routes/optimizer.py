from flask import Blueprint, render_template, request, jsonify, session
from services.game_service import get_all_games, get_game_by_slug
from services.hardware_detector import detect_hardware
from services.optimizer import generate_recommendations

optimizer_bp = Blueprint('optimizer', __name__)


@optimizer_bp.route('/optimizer')
def optimizer():
    games = get_all_games()
    game_slug = request.args.get('game', '')
    selected_game = get_game_by_slug(game_slug) if game_slug else None
    return render_template('optimizer.html',
                           games=games,
                           selected_game=selected_game)


@optimizer_bp.route('/api/optimize', methods=['POST'])
def api_optimize():
    data = request.get_json() or {}
    game_slug = data.get('game', '')
    priority = data.get('priority', 'balanced')
    resolution = data.get('resolution', '1920x1080')
    target_fps = int(data.get('target_fps', 60))

    game = get_game_by_slug(game_slug)
    if not game:
        return jsonify({'error': 'Game not found'}), 404

    try:
        hardware = detect_hardware()
        result = generate_recommendations(
            hardware=hardware,
            game=game,
            target_fps=target_fps,
            priority=priority,
            resolution=resolution,
        )
        return jsonify({
            'success': True,
            'result': result.to_dict(),
            'hardware': hardware.to_dict(),
        })
    except Exception as e:
        return jsonify({'error': 'Optimization failed', 'detail': str(e)}), 500


@optimizer_bp.route('/guides')
def guides():
    return render_template('guides.html')


@optimizer_bp.route('/settings')
def settings():
    return render_template('settings.html')


@optimizer_bp.route('/about')
def about():
    return render_template('about.html')
